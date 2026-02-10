#!/usr/bin/env python3
from __future__ import annotations

"""Dump the SQLite news pool (`links` + `events`) to JSONL for sandbox rebuilds.

This is a lightweight safety tool for clustering experiments. It creates a
read-only snapshot export of the production DB so you can restore it elsewhere
and iterate without risk.

By default the dump includes:
- Links seen in the last 48 hours (by `last_seen_ts`)
- Events created in the last 168 hours (by `created_at_ts`)
- Any events referenced by those links, plus their parent chain

Example:

  # Dump into a timestamped folder
  uv run python scripts/news_pool_dump_jsonl.py \\
    --db data/newsroom/news_pool.sqlite3 \\
    --out-dir data/newsroom/db_dumps/20260210T120000Z

  # Restore into a separate DB for experiments
  uv run python scripts/news_pool_restore_jsonl.py \\
    --dump-dir data/newsroom/db_dumps/20260210T120000Z \\
    --db data/newsroom/news_pool.sandbox.sqlite3
"""

import argparse
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

OPENCLAW_HOME = Path(__file__).resolve().parents[1]
os.environ.setdefault("OPENCLAW_HOME", str(OPENCLAW_HOME))
sys.path.insert(0, str(OPENCLAW_HOME))

from newsroom.news_pool_backup import dump_news_pool_jsonl  # noqa: E402


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Dump newsroom news_pool.sqlite3 tables to JSONL (links + events).")
    parser.add_argument("--db", default=str(OPENCLAW_HOME / "data" / "newsroom" / "news_pool.sqlite3"), help="Source SQLite db path.")
    parser.add_argument("--out-dir", default="", help="Output directory (default: data/newsroom/db_dumps/<UTC timestamp>).")
    parser.add_argument("--links-window-hours", type=int, default=48, help="Dump links seen within this window (default: 48).")
    parser.add_argument("--events-max-age-hours", type=int, default=168, help="Dump events created within this window (default: 168).")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing dump files if they already exist.")
    args = parser.parse_args(argv)

    out_dir = str(args.out_dir or "").strip()
    if not out_dir:
        ts = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
        out_dir = str(OPENCLAW_HOME / "data" / "newsroom" / "db_dumps" / ts)

    dump_news_pool_jsonl(
        source_db=Path(args.db).expanduser(),
        out_dir=Path(out_dir).expanduser(),
        links_window_hours=int(args.links_window_hours),
        events_max_age_hours=int(args.events_max_age_hours),
        overwrite=bool(args.overwrite),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

