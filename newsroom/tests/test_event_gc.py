import tempfile
import time
import unittest
from pathlib import Path

from newsroom.event_gc import run_periodic_unposted_event_gc
from newsroom.news_pool_db import NewsPoolDB


class TestPeriodicUnpostedEventGc(unittest.TestCase):
    def test_periodic_gc_runs_and_prunes_when_due(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "news_pool.sqlite3"
            now = int(time.time())

            with NewsPoolDB(path=db_path) as db:
                eid = db.create_event(summary_en="Old stale event", category="AI", jurisdiction="US")
                db._conn.execute(
                    """
                    UPDATE events
                    SET created_at_ts = ?, updated_at_ts = ?, expires_at_ts = ?, link_count = 1
                    WHERE id = ?
                    """,
                    (now - 120 * 3600, now - 110 * 3600, now - 24 * 3600, eid),
                )

            out = run_periodic_unposted_event_gc(
                db_path=db_path,
                state_key="event_gc_test",
                min_interval_seconds=0,
                stale_after_hours=96,
                low_link_max=1,
                low_quality_summary_chars=80,
                now_ts=now,
            )
            self.assertEqual(out.get("ok"), True)
            self.assertEqual(out.get("should_gc"), True)
            self.assertEqual(out.get("pruned"), 1)
            self.assertEqual(out.get("run_count"), 1)

            with NewsPoolDB(path=db_path) as db:
                self.assertIsNone(db.get_event(eid))

    def test_periodic_gc_skips_when_min_interval_not_elapsed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "news_pool.sqlite3"
            t0 = int(time.time())

            first = run_periodic_unposted_event_gc(
                db_path=db_path,
                state_key="event_gc_test",
                min_interval_seconds=0,
                now_ts=t0,
            )
            self.assertEqual(first.get("should_gc"), True)
            self.assertEqual(first.get("run_count"), 1)

            second = run_periodic_unposted_event_gc(
                db_path=db_path,
                state_key="event_gc_test",
                min_interval_seconds=3600,
                now_ts=t0 + 300,
            )
            self.assertEqual(second.get("ok"), True)
            self.assertEqual(second.get("should_gc"), False)
            self.assertEqual(second.get("reason"), "min_interval")
            self.assertEqual(second.get("pruned"), 0)
            self.assertEqual(second.get("run_count"), 1)


if __name__ == "__main__":
    unittest.main()
