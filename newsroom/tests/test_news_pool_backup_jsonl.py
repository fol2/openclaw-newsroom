from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pytest

from newsroom.news_pool_backup import dump_news_pool_jsonl, restore_news_pool_jsonl
from newsroom.news_pool_db import NewsPoolDB, PoolLink


def _table_rows(db_path: Path, table: str) -> list[dict]:
    conn = sqlite3.connect(str(db_path), timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(f"SELECT * FROM {table} ORDER BY id ASC").fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def test_dump_restore_roundtrip(tmp_path: Path) -> None:
    src_db = tmp_path / "src.sqlite3"

    now = int(time.time())

    with NewsPoolDB(path=src_db) as db:
        root_id = db.create_event(
            summary_en="Root summary",
            category="Global News",
            jurisdiction="GB",
            title="Root event",
            primary_url="https://example.com/root",
        )
        child_id = db.create_event(
            summary_en="Child summary",
            category="Global News",
            jurisdiction="GB",
            title="Child event",
            primary_url="https://example.com/child",
            parent_event_id=root_id,
        )

        link = PoolLink(
            url="https://example.com/a",
            norm_url="https://example.com/a",
            domain="example.com",
            title="A",
            description="desc",
            age="1h",
            page_age=None,
            query="q",
            offset=1,
            fetched_at_ts=now,
        )
        db.upsert_links([link], now_ts=now)
        link_id = int(db._conn.execute("SELECT id FROM links WHERE norm_url = ?", (link.norm_url,)).fetchone()["id"])
        db.assign_link_to_event(link_id=link_id, event_id=child_id)

    dump_dir = tmp_path / "dump"
    manifest = dump_news_pool_jsonl(
        source_db=src_db,
        out_dir=dump_dir,
        links_window_hours=48,
        events_max_age_hours=168,
        now_ts=now + 1,
    )

    assert manifest["ok"] is True
    assert (dump_dir / "links.jsonl").exists()
    assert (dump_dir / "events.jsonl").exists()
    assert manifest["counts"]["links"] == 1
    assert manifest["counts"]["events"] >= 2

    restored_db = tmp_path / "restored.sqlite3"
    restore_news_pool_jsonl(dump_dir=dump_dir, target_db=restored_db)

    assert _table_rows(src_db, "links") == _table_rows(restored_db, "links")
    assert _table_rows(src_db, "events") == _table_rows(restored_db, "events")


def test_restore_refuses_existing_target_db(tmp_path: Path) -> None:
    src_db = tmp_path / "src.sqlite3"
    now = int(time.time())

    with NewsPoolDB(path=src_db) as db:
        eid = db.create_event(summary_en="Summary", category="Global News", jurisdiction="GB")
        link = PoolLink(
            url="https://example.com/a",
            norm_url="https://example.com/a",
            domain="example.com",
            title="A",
            description="desc",
            age="1h",
            page_age=None,
            query="q",
            offset=1,
            fetched_at_ts=now,
        )
        db.upsert_links([link], now_ts=now)
        link_id = int(db._conn.execute("SELECT id FROM links WHERE norm_url = ?", (link.norm_url,)).fetchone()["id"])
        db.assign_link_to_event(link_id=link_id, event_id=eid)

    dump_dir = tmp_path / "dump"
    dump_news_pool_jsonl(source_db=src_db, out_dir=dump_dir, now_ts=now + 1)

    target_db = tmp_path / "target.sqlite3"
    restore_news_pool_jsonl(dump_dir=dump_dir, target_db=target_db)

    with pytest.raises(FileExistsError):
        restore_news_pool_jsonl(dump_dir=dump_dir, target_db=target_db, overwrite=False)

    # Overwrite should succeed.
    restore_news_pool_jsonl(dump_dir=dump_dir, target_db=target_db, overwrite=True)

