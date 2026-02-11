#!/usr/bin/env python3
"""One-time re-clustering: merge fragmented events using retrieve-then-decide.

Uses pairwise token/anchor retrieval to find duplicate event pairs,
then asks the LLM to confirm merges. Multi-pass until convergence.

Usage:
    PYTHONPATH=. python scripts/recluster_events.py [--dry-run] [--max-merges 50] [--passes 3]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

OPENCLAW_HOME = Path(__file__).resolve().parents[1]
os.environ.setdefault("OPENCLAW_HOME", str(OPENCLAW_HOME))
sys.path.insert(0, str(OPENCLAW_HOME))

from newsroom.event_manager import (  # noqa: E402
    EventTokens,
    _tokenize_event,
    build_merge_prompt,
    _compute_drop_high_df,
    parse_merge_response,
    retrieve_candidates,
)
from newsroom.gemini_client import GeminiClient  # noqa: E402
from newsroom.news_pool_db import NewsPoolDB  # noqa: E402

logger = logging.getLogger(__name__)

_DB_PATH = OPENCLAW_HOME / "data" / "newsroom" / "news_pool.sqlite3"


def _pairwise_scores(
    events: list[dict[str, Any]],
    *,
    top_k: int,
    min_score: float = 0.25,
    token_cache: dict[int, EventTokens],
    drop_high_df: set[str],
) -> list[tuple[int, int, float]]:
    """Compute pairwise retrieval scores between all root events.

    Returns list of (eid_a, eid_b, score) where score >= min_score,
    sorted by score descending.
    """
    pairs: list[tuple[int, int, float]] = []

    for i, ev_a in enumerate(events):
        # Use retrieve_candidates to find matches for ev_a among remaining events.
        # We treat ev_a as a "link" for retrieval purposes.
        link_like = {
            "title": ev_a.get("summary_en") or ev_a.get("title") or "",
            "description": ev_a.get("title") or "",
        }
        others = events[i + 1:]
        if not others:
            continue

        candidates = retrieve_candidates(
            link_like, others,
            top_k=int(top_k), min_score=float(min_score),
            token_cache=token_cache,
            drop_high_df=drop_high_df,
        )
        for ev_b, score in candidates:
            pairs.append((ev_a["id"], ev_b["id"], score))

    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs


def recluster_pass(
    *,
    db: NewsPoolDB,
    gemini: GeminiClient,
    max_merges: int,
    dry_run: bool,
    min_score: float,
    top_k: int,
    max_consecutive_failures: int,
    delay_seconds: float = 3.0,
) -> dict[str, int]:
    """Run one reclustering pass. Returns stats."""
    stats = {
        "root_events": 0,
        "pairs_found": 0,
        "llm_calls": 0,
        "merges": 0,
        "links_moved": 0,
        "errors": 0,
        "early_exit": 0,
    }

    all_events = db.get_all_fresh_events(max_age_hours=168)
    root_events = [
        ev for ev in all_events
        if ev.get("parent_event_id") is None
    ]
    stats["root_events"] = len(root_events)

    if len(root_events) < 2:
        return stats

    token_cache: dict[int, EventTokens] = {}
    drop_high_df = _compute_drop_high_df(root_events)
    pairs = _pairwise_scores(
        root_events,
        token_cache=token_cache,
        drop_high_df=drop_high_df,
        min_score=float(min_score),
        top_k=int(top_k),
    )
    stats["pairs_found"] = len(pairs)

    if not pairs:
        return stats

    ev_lookup = {ev["id"]: ev for ev in root_events}
    merged_this_pass: set[int] = set()
    consecutive_failures = 0

    for eid_a, eid_b, score in pairs:
        if consecutive_failures >= max(1, int(max_consecutive_failures)):
            logger.warning(
                "Early exit: %d consecutive LLM failures, stopping this pass",
                consecutive_failures,
            )
            stats["early_exit"] = 1
            break

        if stats["merges"] >= max_merges:
            break
        if eid_a in merged_this_pass or eid_b in merged_this_pass:
            continue

        ev_a = ev_lookup.get(eid_a)
        ev_b = ev_lookup.get(eid_b)
        if not ev_a or not ev_b:
            continue

        # Ask LLM to confirm merge.
        prompt = build_merge_prompt(
            [ev_a, ev_b],
            f"recluster ({ev_a.get('category', '?')} + {ev_b.get('category', '?')})",
        )

        try:
            raw_text = gemini.generate(prompt)
        except Exception as e:
            logger.warning("LLM call failed for pair (%d, %d): %s", eid_a, eid_b, e)
            stats["errors"] += 1
            consecutive_failures += 1
            continue

        stats["llm_calls"] += 1

        if not raw_text:
            stats["errors"] += 1
            consecutive_failures += 1
            continue

        sanitized = re.sub(r'\bE(\d+)\b', r'\1', raw_text)
        try:
            response = json.loads(
                sanitized[sanitized.find("{"):sanitized.rfind("}") + 1]
            )
        except (json.JSONDecodeError, ValueError):
            stats["errors"] += 1
            consecutive_failures += 1
            continue
        consecutive_failures = 0

        valid_ids = {eid_a, eid_b}
        groups = parse_merge_response(response, valid_ids)
        if not groups:
            continue

        for group in groups:
            group_events = [ev_lookup[eid] for eid in group if eid in ev_lookup]
            if len(group_events) < 2:
                continue

            posted_events = [e for e in group_events if e.get("status") == "posted"]
            if len(posted_events) > 1:
                logger.info(
                    "Skip merge group with multiple posted events: %s",
                    [int(e.get("id") or 0) for e in posted_events],
                )
                continue

            if len(posted_events) == 1:
                winner = posted_events[0]
            else:
                # Pick winner: most links, tiebreak oldest.
                winner = max(
                    group_events,
                    key=lambda e: (e.get("link_count", 0), -(e.get("created_at_ts", 0))),
                )
            loser_ids = [e["id"] for e in group_events if e["id"] != winner["id"]]

            if dry_run:
                for lid in loser_ids:
                    loser = ev_lookup.get(lid, {})
                    logger.info(
                        "[DRY RUN] Would merge event %d (%s) into %d (%s) | score=%.3f",
                        lid, loser.get("summary_en", "?")[:60],
                        winner["id"], winner.get("summary_en", "?")[:60],
                        score,
                    )
                stats["merges"] += 1
                merged_this_pass.update(loser_ids)
            else:
                try:
                    links_moved = db.merge_events_into(
                        winner_id=winner["id"], loser_ids=loser_ids,
                    )
                    stats["merges"] += 1
                    stats["links_moved"] += links_moved
                    merged_this_pass.update(loser_ids)
                    logger.info(
                        "Merged event(s) %s into %d, %d links moved | score=%.3f",
                        loser_ids, winner["id"], links_moved, score,
                    )
                except Exception as e:
                    logger.warning("Merge failed for %s → %d: %s", loser_ids, winner["id"], e)
                    stats["errors"] += 1

        if delay_seconds > 0:
            time.sleep(delay_seconds)

    return stats


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Re-cluster fragmented events using retrieve-then-decide.")
    parser.add_argument("--dry-run", action="store_true", help="Show proposed merges without executing.")
    parser.add_argument("--min-score", type=float, default=0.30, help="Retrieval min score threshold (default: 0.30).")
    parser.add_argument("--top-k", type=int, default=10, help="Top-k retrieval candidates per event (default: 10).")
    parser.add_argument(
        "--max-consecutive-failures",
        type=int,
        default=10,
        help="Stop early after this many consecutive LLM failures (default: 10).",
    )
    parser.add_argument("--max-merges", type=int, default=50, help="Max merges per pass (default: 50).")
    parser.add_argument("--passes", type=int, default=3, help="Max re-clustering passes (default: 3).")
    parser.add_argument("--delay", type=float, default=3.0, help="Delay between LLM calls in seconds (default: 3.0).")
    parser.add_argument("--db-path", default=str(_DB_PATH), help="Path to news_pool.sqlite3.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging.")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    db_path = Path(args.db_path)
    if not db_path.exists():
        logger.error("Database not found: %s", db_path)
        return 1

    gemini = GeminiClient()
    all_pass_stats: list[dict[str, int]] = []

    for pass_num in range(1, args.passes + 1):
        logger.info("=== Recluster pass %d/%d ===", pass_num, args.passes)

        with NewsPoolDB(path=db_path) as db:
            pass_stats = recluster_pass(
                db=db,
                gemini=gemini,
                max_merges=args.max_merges,
                dry_run=args.dry_run,
                min_score=float(args.min_score),
                top_k=int(args.top_k),
                max_consecutive_failures=int(args.max_consecutive_failures),
                delay_seconds=args.delay,
            )

        all_pass_stats.append(pass_stats)
        logger.info(
            "Pass %d: %d root events, %d pairs, %d LLM calls, %d merges, %d links moved, %d errors",
            pass_num,
            pass_stats["root_events"], pass_stats["pairs_found"],
            pass_stats["llm_calls"], pass_stats["merges"],
            pass_stats["links_moved"], pass_stats["errors"],
        )

        # Stop if no merges happened this pass.
        if pass_stats["merges"] == 0:
            logger.info("No merges in pass %d — converged.", pass_num)
            break

    # Output JSON stats.
    out = {
        "ok": True,
        "dry_run": args.dry_run,
        "passes": all_pass_stats,
        "total_merges": sum(s["merges"] for s in all_pass_stats),
        "total_links_moved": sum(s["links_moved"] for s in all_pass_stats),
        "total_errors": sum(s["errors"] for s in all_pass_stats),
    }
    print(json.dumps(out, ensure_ascii=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
