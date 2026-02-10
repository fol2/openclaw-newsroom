#!/usr/bin/env python3
from __future__ import annotations

"""Restore a JSONL dump into a fresh SQLite news pool database.

This is the companion to scripts/news_pool_dump_jsonl.py.

Safety:
- If the target DB already exists, the script refuses to overwrite it unless
  you pass --overwrite.
- The restore writes to a temporary DB first and then atomically renames it
  into place, so a failed restore will not leave a partially written DB.

Example:

  uv run python scripts/news_pool_restore_jsonl.py \\
    --dump-dir data/newsroom/db_dumps/20260210T120000Z \\
    --db data/newsroom/news_pool.sandbox.sqlite3
"""

import argparse
import os
import sys
from pathlib import Path

OPENCLAW_HOME = Path(__file__).resolve().parents[1]
os.environ.setdefault("OPENCLAW_HOME", str(OPENCLAW_HOME))
sys.path.insert(0, str(OPENCLAW_HOME))

from newsroom.news_pool_backup import restore_news_pool_jsonl  # noqa: E402


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Restore a news pool JSONL dump into a fresh SQLite DB.")
    parser.add_argument("--dump-dir", required=True, help="Dump directory produced by scripts/news_pool_dump_jsonl.py.")
    parser.add_argument("--db", required=True, help="Target SQLite db path to create/replace.")
    parser.add_argument("--overwrite", action="store_true", help="Replace target DB if it already exists.")
    parser.add_argument("--no-foreign-key-check", action="store_true", help="Skip PRAGMA foreign_key_check after restore.")
    args = parser.parse_args(argv)

    restore_news_pool_jsonl(
        dump_dir=Path(args.dump_dir).expanduser(),
        target_db=Path(args.db).expanduser(),
        overwrite=bool(args.overwrite),
        validate_foreign_keys=not bool(args.no_foreign_key_check),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

