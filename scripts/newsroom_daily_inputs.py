#!/usr/bin/env python3
"""Newsroom daily planner preflight: fetch links → LLM cluster → select events → output.

Event-centric architecture (v5): uses LLM clustering to group links into events,
then selects the best 10-15 events for the daily digest with category balance,
finance deprioritization, and HK guarantee.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

OPENCLAW_HOME = Path(__file__).resolve().parents[1]
os.environ.setdefault("OPENCLAW_HOME", str(OPENCLAW_HOME))
sys.path.insert(0, str(OPENCLAW_HOME))

from newsroom.event_manager import cluster_all_pending, merge_events  # noqa: E402
from newsroom.gemini_client import GeminiClient  # noqa: E402
from newsroom.news_pool_db import NewsPoolDB  # noqa: E402
from newsroom.subprocess_json import run_json_command  # noqa: E402

# Default queries:
# - Global: exclude explicit UK terms (daily does a dedicated UK call).
# - UK: restrict to major UK outlets for higher-quality coverage.
DEFAULT_GLOBAL_QUERY = (
    "HK OR 香港 OR politics OR election OR 政治 OR sports OR entertainment OR 娛樂 OR film OR AI OR 人工智能 OR technology "
    "OR stocks OR shares OR earnings OR futures OR Fed OR CPI OR Treasury OR yields OR Nasdaq OR NYSE"
)
DEFAULT_UK_QUERY = "UK (site:co.uk OR site:theguardian.com OR site:news.sky.com OR site:itv.com OR site:channel4.com)"

_DB_PATH = OPENCLAW_HOME / "data" / "newsroom" / "news_pool.sqlite3"


def _default_subprocess_timeout_seconds() -> float:
    """Timeout for child processes spawned by the inputs scripts.

    These scripts are called from cron-driven LLM planners. A hung subprocess
    should not block the planner turn until the cron timeout.
    """
    raw = (os.environ.get("NEWSROOM_INPUTS_SUBPROCESS_TIMEOUT_SECONDS") or "").strip()
    if not raw:
        return 120.0
    try:
        return float(raw)
    except ValueError:
        return 120.0


_SUBPROCESS_TIMEOUT_SECONDS = _default_subprocess_timeout_seconds()


def _run_json(cmd: list[str]) -> dict[str, Any]:
    """Run a JSON-emitting subprocess with a hard timeout."""
    return run_json_command(cmd, timeout_seconds=_SUBPROCESS_TIMEOUT_SECONDS, max_output_chars=2000)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Newsroom daily inputs: fetch → cluster → select → output.")
    parser.add_argument("--pool-hours", type=int, default=48, help="Retention for pool update (default: 48).")
    parser.add_argument("--min-interval-seconds", type=int, default=900, help="Min seconds between Brave calls (default: 900).")
    parser.add_argument("--state-key", default="daily", help="Base key for pool fetch state (default: daily).")
    parser.add_argument("--key-label", default="free", help="Prefer a Brave API key label (default: free).")
    parser.add_argument(
        "--query", action="append", default=[],
        help="Global Brave News query override (repeatable).",
    )
    parser.add_argument(
        "--uk-query", action="append", default=[],
        help="UK Brave News query override (repeatable).",
    )
    parser.add_argument("--count", type=int, default=100, help="Brave results per request (default: 100).")
    parser.add_argument("--pages", type=int, default=1, help="Brave pages per run (default: 1).")
    parser.add_argument("--max-offset", type=int, default=1, help="Rotate offsets (default: 1).")
    parser.add_argument("--freshness", default="pd", help="Brave freshness (default: pd).")
    parser.add_argument("--limit-candidates", type=int, default=15, help="Max events to output (default: 15).")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM clustering.")
    parser.add_argument("--no-gdelt", action="store_true", help="Skip GDELT pool update.")
    parser.add_argument("--no-rss", action="store_true", help="Skip RSS pool update.")
    parser.add_argument(
        "--write-path",
        default=str(OPENCLAW_HOME / "data" / "newsroom" / "daily_inputs_last.json"),
        help="Write output JSON to this path.",
    )
    # Legacy args (ignored but accepted for backward compat).
    parser.add_argument("--channel-id", default="1467628391082496041", help=argparse.SUPPRESS)
    parser.add_argument("--recent-hours", type=int, default=24, help=argparse.SUPPRESS)
    parser.add_argument("--limit-titles", type=int, default=120, help=argparse.SUPPRESS)
    parser.add_argument("--limit-markers", type=int, default=300, help=argparse.SUPPRESS)
    parser.add_argument("--max-age-hours", type=int, default=48, help=argparse.SUPPRESS)
    parser.add_argument("--limit-clusters", type=int, default=40, help=argparse.SUPPRESS)
    parser.add_argument("--min-cluster-size", type=int, default=2, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    now = datetime.now(tz=UTC).isoformat(timespec="seconds")

    # ---------------------------------------------------------------
    # Step 1: Fetch links from Brave API (global + UK)
    # ---------------------------------------------------------------
    base_state_key = str(args.state_key).strip() or "daily"
    key_label = str(args.key_label).strip() or None

    global_cmd = [
        sys.executable,
        str(OPENCLAW_HOME / "scripts" / "news_pool_update.py"),
        "--hours", str(int(args.pool_hours)),
        "--min-interval-seconds", str(int(args.min_interval_seconds)),
        "--state-key", f"{base_state_key}_global",
        "--key-label", str(key_label or ""),
        "--count", str(int(args.count)),
        "--pages", str(int(args.pages)),
        "--max-offset", str(int(args.max_offset)),
        "--freshness", str(args.freshness),
    ]
    global_queries = [str(q).strip() for q in (args.query or []) if isinstance(q, str) and str(q).strip()]
    if not global_queries:
        global_queries = [DEFAULT_GLOBAL_QUERY]
    for q in global_queries:
        global_cmd += ["--query", q]

    uk_cmd = [
        sys.executable,
        str(OPENCLAW_HOME / "scripts" / "news_pool_update.py"),
        "--hours", str(int(args.pool_hours)),
        "--min-interval-seconds", str(int(args.min_interval_seconds)),
        "--state-key", f"{base_state_key}_uk",
        "--key-label", str(key_label or ""),
        "--count", str(int(args.count)),
        "--pages", str(int(args.pages)),
        "--max-offset", str(int(args.max_offset)),
        "--freshness", str(args.freshness),
    ]
    uk_queries = [str(q).strip() for q in (args.uk_query or []) if isinstance(q, str) and str(q).strip()]
    if not uk_queries:
        uk_queries = [DEFAULT_UK_QUERY]
    for q in uk_queries:
        uk_cmd += ["--query", q]

    # Run Brave API calls (global + UK with 1s gap for rate limiting).
    pool_global = _run_json(global_cmd)
    time.sleep(1.15)
    pool_uk = _run_json(uk_cmd)

    # ---------------------------------------------------------------
    # Step 1b: GDELT
    # ---------------------------------------------------------------
    pool_gdelt: dict[str, Any] = {"ok": False, "skipped": True}
    if not args.no_gdelt:
        gdelt_cmd = [
            sys.executable, str(OPENCLAW_HOME / "scripts" / "gdelt_pool_update.py"),
            "--hours", str(int(args.pool_hours)),
            "--min-interval-seconds", str(int(args.min_interval_seconds)),
            "--state-key", f"{base_state_key}_gdelt",
            "--timespan", "24h",
        ]
        try:
            pool_gdelt = _run_json(gdelt_cmd)
        except Exception as e:
            pool_gdelt = {"ok": False, "error": str(e)}

    # ---------------------------------------------------------------
    # Step 1c: RSS
    # ---------------------------------------------------------------
    pool_rss: dict[str, Any] = {"ok": False, "skipped": True}
    if not args.no_rss:
        rss_cmd = [
            sys.executable, str(OPENCLAW_HOME / "scripts" / "rss_pool_update.py"),
            "--hours", str(int(args.pool_hours)),
            "--min-interval-seconds", str(int(args.min_interval_seconds)),
            "--state-key", f"{base_state_key}_rss",
        ]
        try:
            pool_rss = _run_json(rss_cmd)
        except Exception as e:
            pool_rss = {"ok": False, "error": str(e)}

    # ---------------------------------------------------------------
    # Step 2: LLM clustering
    # ---------------------------------------------------------------
    clustering_stats: dict[str, Any] = {}
    if not args.no_llm:
        try:
            gemini = GeminiClient()
            db = NewsPoolDB(path=_DB_PATH)
            try:
                clustering_stats = cluster_all_pending(db=db, gemini=gemini)
            finally:
                db.close()
        except Exception as e:
            clustering_stats = {"error": str(e)}

    # ---------------------------------------------------------------
    # Step 2b: Post-clustering merge pass
    # ---------------------------------------------------------------
    merge_stats: dict[str, Any] = {}
    if not args.no_llm:
        try:
            gemini = GeminiClient()
            db = NewsPoolDB(path=_DB_PATH)
            try:
                merge_stats = merge_events(db=db, gemini=gemini)
            finally:
                db.close()
        except Exception as e:
            merge_stats = {"error": str(e)}

    # ---------------------------------------------------------------
    # Step 3: Event selection (daily: 10-15, category balance, HK guarantee)
    # ---------------------------------------------------------------
    candidates: list[dict[str, Any]] = []
    try:
        db = NewsPoolDB(path=_DB_PATH)
        try:
            candidates = db.get_daily_candidates(limit=int(args.limit_candidates))
        finally:
            db.close()
    except Exception as e:
        clustering_stats["selection_error"] = str(e)

    # Number candidates for planner.
    for i, c in enumerate(candidates, start=1):
        c["i"] = i

    # ---------------------------------------------------------------
    # Output
    # ---------------------------------------------------------------
    pool_calls = [
        {
            "name": "global",
            "ok": bool(pool_global.get("ok", False)),
            "fetch": pool_global.get("fetch"),
            "upsert": pool_global.get("upsert"),
            "pruned": pool_global.get("pruned"),
        },
        {
            "name": "uk",
            "ok": bool(pool_uk.get("ok", False)),
            "fetch": pool_uk.get("fetch"),
            "upsert": pool_uk.get("upsert"),
            "pruned": pool_uk.get("pruned"),
        },
        {
            "name": "gdelt",
            "ok": bool(pool_gdelt.get("ok", False)),
            "fetch": pool_gdelt.get("fetch"),
            "upsert": pool_gdelt.get("upsert"),
            "pruned": pool_gdelt.get("pruned"),
        },
        {
            "name": "rss",
            "ok": bool(pool_rss.get("ok", False)),
            "fetch": pool_rss.get("fetch"),
            "upsert": pool_rss.get("upsert"),
            "pruned": pool_rss.get("pruned"),
        },
    ]

    out: dict[str, Any] = {
        "ok": True,
        "generated_at": now,
        "pool": {
            "ok": bool(pool_global.get("ok", False) and pool_uk.get("ok", False)),
            "calls": pool_calls,
        },
        "clustering": clustering_stats,
        "merge": merge_stats,
        "candidates": candidates,
        "candidate_count": len(candidates),
    }

    # Persist for the writer script.
    try:
        write_path = Path(str(args.write_path)).expanduser()
        write_path.parent.mkdir(parents=True, exist_ok=True)
        write_path.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
        out["inputs_path"] = str(write_path)
    except Exception:
        pass

    print(json.dumps(out, ensure_ascii=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
