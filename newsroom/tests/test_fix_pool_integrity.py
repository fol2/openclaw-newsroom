"""Tests for scripts.fix_pool_integrity (offline DB maintenance)."""

import tempfile
import unittest
from pathlib import Path

from newsroom.news_pool_db import NewsPoolDB, _DEFAULT_TTL_SECONDS
from scripts.fix_pool_integrity import apply_plan, build_plan


class TestFixPoolIntegrity(unittest.TestCase):
    def test_recomputes_stats_prunes_and_normalises(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "news_pool.sqlite3"

            with NewsPoolDB(path=db_path) as db:
                conn = db._conn

                now_ts = 1_700_000_000
                ttl = int(_DEFAULT_TTL_SECONDS)

                # Event with links and deliberate drift in derived fields.
                eid1 = db.create_event(summary_en="Root event", category="UK Politics", jurisdiction="UK")
                conn.execute("UPDATE events SET created_at_ts = ?, updated_at_ts = ? WHERE id = ?", (now_ts, now_ts, eid1))

                link_rows = [
                    (
                        "https://example.com/a",
                        "https://example.com/a",
                        "example.com",
                        "A",
                        "A desc",
                        None,
                        None,
                        now_ts - 1000,
                        now_ts - 100,
                        1,
                        "test",
                        1,
                        now_ts - 100,
                        eid1,
                        now_ts - 500,
                    ),
                    (
                        "https://example.com/b",
                        "https://example.com/b",
                        "example.com",
                        "B",
                        "B desc",
                        None,
                        None,
                        now_ts - 900,
                        now_ts - 50,
                        1,
                        "test",
                        1,
                        now_ts - 50,
                        eid1,
                        now_ts - 200,
                    ),
                ]
                conn.executemany(
                    """
                    INSERT INTO links(
                      url, norm_url, domain, title, description, age, page_age,
                      first_seen_ts, last_seen_ts, seen_count, last_query, last_offset, last_fetched_at_ts,
                      event_id, published_at_ts
                    ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    link_rows,
                )

                # Drift the event's derived fields.
                conn.execute(
                    "UPDATE events SET link_count = 999, best_published_ts = 1, expires_at_ts = 1 WHERE id = ?",
                    (eid1,),
                )

                # Dead non-posted event with 0 links, old enough to prune.
                eid2 = db.create_event(summary_en="Dead event", category="Global News", jurisdiction="GLOBAL")
                conn.execute(
                    "UPDATE events SET status = 'active', created_at_ts = ?, updated_at_ts = ? WHERE id = ?",
                    (now_ts - 200 * 3600, now_ts - 200 * 3600, eid2),
                )

                # Posted event should never be pruned, but can be category-normalised.
                eid3 = db.create_event(summary_en="Posted event", category="POLITICAL_EVENT", jurisdiction="UK")
                conn.execute(
                    "UPDATE events SET status = 'posted', created_at_ts = ?, updated_at_ts = ? WHERE id = ?",
                    (now_ts - 200 * 3600, now_ts, eid3),
                )

                plan = build_plan(
                    conn=conn,
                    now_ts=now_ts,
                    prune_zero_link_hours=168,
                    ttl_seconds=ttl,
                    normalise_categories=True,
                )
                self.assertGreaterEqual(len(plan.stat_fixes), 1)
                self.assertIn(eid2, plan.prune_event_ids)

                summary = apply_plan(conn=conn, plan=plan, now_ts=now_ts)
                self.assertGreaterEqual(int(summary["events_updated"]), 1)
                self.assertGreaterEqual(int(summary["categories_updated"]), 1)
                self.assertGreaterEqual(int(summary["events_pruned"]), 1)

                ev1 = db.get_event(eid1)
                assert ev1 is not None
                self.assertEqual(ev1["link_count"], 2)
                self.assertEqual(ev1["best_published_ts"], now_ts - 200)
                self.assertEqual(ev1["expires_at_ts"], (now_ts - 50) + ttl)
                self.assertEqual(ev1["category"], "UK Parliament / Politics")

                # Pruned event is gone.
                self.assertIsNone(db.get_event(eid2))

                # Posted event remains and is normalised.
                ev3 = db.get_event(eid3)
                assert ev3 is not None
                self.assertEqual(ev3["status"], "posted")
                self.assertEqual(ev3["category"], "Politics")

                # Idempotent: running again should produce no work.
                plan2 = build_plan(
                    conn=conn,
                    now_ts=now_ts,
                    prune_zero_link_hours=168,
                    ttl_seconds=ttl,
                    normalise_categories=True,
                )
                self.assertEqual(plan2.stat_fixes, [])
                self.assertEqual(plan2.category_fixes, [])
                self.assertEqual(plan2.prune_event_ids, [])

