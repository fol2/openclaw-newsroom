#!/usr/bin/env python3
from __future__ import annotations

"""One-off merge runner for the newsroom events table.

This script runs the post-clustering merge pass (including cross-category
retrieval) against a given SQLite pool database.

It is intended for maintenance tasks and should not be wired into cron.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

OPENCLAW_HOME = Path(__file__).resolve().parents[1]
os.environ.setdefault("OPENCLAW_HOME", str(OPENCLAW_HOME))
sys.path.insert(0, str(OPENCLAW_HOME))

from newsroom.event_manager import merge_events  # noqa: E402
from newsroom.gemini_client import GeminiClient  # noqa: E402
from newsroom.news_pool_db import NewsPoolDB  # noqa: E402

logger = logging.getLogger(__name__)

_DB_PATH = OPENCLAW_HOME / "data" / "newsroom" / "news_pool.sqlite3"


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Run merge_events() as a one-off maintenance job.")
    parser.add_argument("--db-path", default=str(_DB_PATH), help="Path to news_pool.sqlite3.")
    parser.add_argument("--delay", type=float, default=3.0, help="Delay between LLM calls in seconds (default: 3.0).")
    parser.add_argument("--batch-size", type=int, default=50, help="Per-category merge batch size (default: 50).")
    parser.add_argument(
        "--max-consecutive-failures",
        type=int,
        default=3,
        help="Stop early after this many consecutive LLM failures (default: 3).",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging.")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    db_path = Path(args.db_path).expanduser()
    if not db_path.exists():
        logger.error("Database not found: %s", db_path)
        return 1

    gemini = GeminiClient()
    with NewsPoolDB(path=db_path) as db:
        stats = merge_events(
            db=db,
            gemini=gemini,
            delay_seconds=float(args.delay),
            batch_size=int(args.batch_size),
            max_consecutive_failures=int(args.max_consecutive_failures),
        )

    out = {"ok": True, "stats": stats}
    print(json.dumps(out, ensure_ascii=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

