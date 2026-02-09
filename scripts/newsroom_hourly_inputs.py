#!/usr/bin/env python3
"""Newsroom hourly planner preflight: fetch links → LLM cluster → select events → output.

Event-centric architecture (v5): uses LLM clustering to group links into events,
then selects the best 1-3 events for the hourly run with development priority
and category diversity.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

OPENCLAW_HOME = Path(__file__).resolve().parents[1]
os.environ.setdefault("OPENCLAW_HOME", str(OPENCLAW_HOME))
sys.path.insert(0, str(OPENCLAW_HOME))

from newsroom.event_manager import cluster_all_pending, merge_events  # noqa: E402
from newsroom.gemini_client import GeminiClient  # noqa: E402
from newsroom.news_pool_db import NewsPoolDB  # noqa: E402

_DB_PATH = OPENCLAW_HOME / "data" / "newsroom" / "news_pool.sqlite3"


def _run_json(cmd: list[str]) -> dict[str, Any]:
    """Run a command that prints JSON to stdout and parse it."""
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True)
    except Exception as e:
        return {"ok": False, "error": "spawn_failed", "detail": str(e), "cmd": cmd}
    out = proc.stdout or ""
    if proc.returncode != 0:
        return {
            "ok": False,
            "error": "cmd_failed",
            "returncode": int(proc.returncode),
            "stdout": out[:2000],
            "stderr": (proc.stderr or "")[:2000],
            "cmd": cmd,
        }
    try:
        obj = json.loads(out)
    except Exception:
        return {"ok": False, "error": "invalid_json", "stdout": out[:2000]}
    return obj if isinstance(obj, dict) else {"ok": False, "error": "json_not_object"}


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Newsroom hourly inputs: fetch → cluster → select → output.")
    parser.add_argument("--pool-hours", type=int, default=48, help="Retention for pool update (default: 48).")
    parser.add_argument("--min-interval-seconds", type=int, default=900, help="Min seconds between Brave calls (default: 900).")
    parser.add_argument("--state-key", default="hourly", help="Fetch rotation key (default: hourly).")
    parser.add_argument("--key-label", default="free", help="Prefer a Brave API key label (default: free).")
    parser.add_argument(
        "--query", action="append", default=[],
        help="Brave News query (repeatable).",
    )
    parser.add_argument("--count", type=int, default=100, help="Brave results per request (default: 100).")
    parser.add_argument("--pages", type=int, default=1, help="Brave pages per run (default: 1).")
    parser.add_argument("--max-offset", type=int, default=1, help="Rotate offsets (default: 1).")
    parser.add_argument("--freshness", default="pd", help="Brave freshness (default: pd).")
    parser.add_argument("--limit-candidates", type=int, default=3, help="Max events to output (default: 3).")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM clustering.")
    parser.add_argument("--no-rss", action="store_true", help="Skip RSS pool update.")
    parser.add_argument(
        "--write-path",
        default=str(OPENCLAW_HOME / "data" / "newsroom" / "hourly_inputs_last.json"),
        help="Write output JSON to this path.",
    )
    # Legacy args (ignored but accepted for backward compat).
    parser.add_argument("--channel-id", default="1467628391082496041", help=argparse.SUPPRESS)
    parser.add_argument("--recent-hours", type=int, default=24, help=argparse.SUPPRESS)
    parser.add_argument("--limit-titles", type=int, default=60, help=argparse.SUPPRESS)
    parser.add_argument("--limit-markers", type=int, default=200, help=argparse.SUPPRESS)
    parser.add_argument("--max-age-hours", type=int, default=12, help=argparse.SUPPRESS)
    parser.add_argument("--limit-clusters", type=int, default=18, help=argparse.SUPPRESS)
    parser.add_argument("--min-cluster-size", type=int, default=2, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    now = datetime.now(tz=UTC).isoformat(timespec="seconds")

    # ---------------------------------------------------------------
    # Step 1: Fetch links from Brave API (single call)
    # ---------------------------------------------------------------
    pool_cmd = [
        sys.executable,
        str(OPENCLAW_HOME / "scripts" / "news_pool_update.py"),
        "--hours", str(int(args.pool_hours)),
        "--min-interval-seconds", str(int(args.min_interval_seconds)),
        "--state-key", str(args.state_key),
        "--key-label", str(args.key_label),
        "--count", str(int(args.count)),
        "--pages", str(int(args.pages)),
        "--max-offset", str(int(args.max_offset)),
        "--freshness", str(args.freshness),
    ]
    queries = [str(q).strip() for q in (args.query or []) if isinstance(q, str) and str(q).strip()]
    for q in queries:
        pool_cmd += ["--query", q]
    pool = _run_json(pool_cmd)

    # ---------------------------------------------------------------
    # Step 1b: RSS
    # ---------------------------------------------------------------
    pool_rss: dict[str, Any] = {"ok": False, "skipped": True}
    if not args.no_rss:
        rss_cmd = [
            sys.executable, str(OPENCLAW_HOME / "scripts" / "rss_pool_update.py"),
            "--hours", str(int(args.pool_hours)),
            "--min-interval-seconds", "1800",
            "--state-key", "hourly_rss",
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
    # Step 3: Event selection (hourly: 1-3, development priority, freshness)
    # ---------------------------------------------------------------
    candidates: list[dict[str, Any]] = []
    try:
        db = NewsPoolDB(path=_DB_PATH)
        try:
            candidates = db.get_hourly_candidates(limit=int(args.limit_candidates))
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
            "name": "brave",
            "ok": bool(pool.get("ok", False)),
            "fetch": pool.get("fetch"),
            "upsert": pool.get("upsert"),
            "pruned": pool.get("pruned"),
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
            "ok": bool(pool.get("ok", False)),
            "calls": pool_calls,
            "fetch": pool.get("fetch"),
            "upsert": pool.get("upsert"),
            "pruned": pool.get("pruned"),
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
