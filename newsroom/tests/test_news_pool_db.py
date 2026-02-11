import sqlite3
import tempfile
import time
import unittest
from pathlib import Path

from newsroom.news_pool_db import NewsPoolDB, PoolLink, SCHEMA_VERSION, _RESERVATION_SECONDS


class TestNewsPoolDBV5Fresh(unittest.TestCase):
    """Test fresh v5 database creation."""

    def test_fresh_db_creates_v6_schema(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "news_pool.sqlite3"
            with NewsPoolDB(path=db_path) as db:
                self.assertEqual(db.get_meta_int("schema_version"), 6)

    def test_fresh_db_has_events_table(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "news_pool.sqlite3"
            with NewsPoolDB(path=db_path) as db:
                event_id = db.create_event(summary_en="Test event", category="AI", jurisdiction="US")
                self.assertIsInstance(event_id, int)
                self.assertGreater(event_id, 0)

                ev = db.get_event(event_id)
                self.assertIsNotNone(ev)
                assert ev is not None
                self.assertEqual(ev["summary_en"], "Test event")
                self.assertEqual(ev["category"], "AI")
                self.assertEqual(ev["jurisdiction"], "US")
                self.assertEqual(ev["status"], "new")
                self.assertIsNone(ev["parent_event_id"])

    def test_links_have_event_id_and_published_at_ts(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "news_pool.sqlite3"
            with NewsPoolDB(path=db_path) as db:
                link = PoolLink(
                    url="https://example.com/a",
                    norm_url="https://example.com/a",
                    domain="example.com",
                    title="Test",
                    description="Desc",
                    age=None,
                    page_age="2026-02-07T10:00:00",
                    query="test",
                    offset=1,
                    fetched_at_ts=int(time.time()),
                )
                db.upsert_links([link])

                links = list(db.iter_links_since(cutoff_ts=0))
                self.assertEqual(len(links), 1)
                self.assertIsNone(links[0]["event_id"])
                self.assertIsNotNone(links[0]["published_at_ts"])


class TestLinkSkipCluster(unittest.TestCase):
    def test_get_unassigned_links_excludes_skip_clustered_links(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "news_pool.sqlite3"
            with NewsPoolDB(path=db_path) as db:
                link = PoolLink(
                    url="https://example.com/skip-1",
                    norm_url="https://example.com/skip-1",
                    domain="example.com",
                    title="Live updates: something happened",
                    description="Rolling coverage",
                    age=None,
                    page_age="2026-02-07T10:00:00",
                    query="test",
                    offset=1,
                    fetched_at_ts=int(time.time()),
                )
                db.upsert_links([link])

                unassigned = db.get_unassigned_links()
                self.assertEqual(len(unassigned), 1)
                link_id = int(unassigned[0]["id"])

                db.mark_link_skip_cluster(link_id=link_id, reason="live_updates")

                remaining = db.get_unassigned_links()
                self.assertEqual(len(remaining), 0)

                row = db._conn.execute(
                    "SELECT skip_cluster_reason, skip_clustered_at_ts FROM links WHERE id = ?",
                    (link_id,),
                ).fetchone()
                self.assertIsNotNone(row)
                assert row is not None
                self.assertEqual(row["skip_cluster_reason"], "live_updates")
                self.assertIsNotNone(row["skip_clustered_at_ts"])


class TestNewsPoolDBForeignKeys(unittest.TestCase):
    """Test SQLite foreign key enforcement is enabled."""

    def test_foreign_keys_pragma_is_on(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "news_pool.sqlite3"
            with NewsPoolDB(path=db_path) as db:
                row = db._conn.execute("PRAGMA foreign_keys;").fetchone()
                assert row is not None
                self.assertEqual(int(row[0]), 1)

    def test_foreign_key_constraints_are_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "news_pool.sqlite3"
            with NewsPoolDB(path=db_path) as db:
                now = int(time.time())
                with self.assertRaises(sqlite3.IntegrityError):
                    db._conn.execute(
                        """
                        INSERT INTO links(
                          url, norm_url,
                          first_seen_ts, last_seen_ts,
                          last_query, last_offset, last_fetched_at_ts,
                          event_id
                        ) VALUES(?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            "https://example.com/bad",
                            "https://example.com/bad",
                            now,
                            now,
                            "test",
                            1,
                            now,
                            9999,
                        ),
                    )


class TestNewsPoolDBV5Migration(unittest.TestCase):
    """Test migration from v4 to v5."""

    def _create_v4_db(self, path: Path) -> None:
        conn = sqlite3.connect(str(path))
        conn.execute("CREATE TABLE meta (k TEXT PRIMARY KEY, v TEXT NOT NULL);")
        conn.execute("INSERT INTO meta(k, v) VALUES('schema_version', '4');")

        conn.execute("""
            CREATE TABLE links (
              id INTEGER PRIMARY KEY, url TEXT NOT NULL, norm_url TEXT NOT NULL UNIQUE,
              domain TEXT, title TEXT, description TEXT, age TEXT, page_age TEXT,
              first_seen_ts INTEGER NOT NULL, last_seen_ts INTEGER NOT NULL,
              seen_count INTEGER NOT NULL DEFAULT 1, last_query TEXT NOT NULL,
              last_offset INTEGER NOT NULL, last_fetched_at_ts INTEGER NOT NULL
            );
        """)
        conn.execute("""
            CREATE TABLE events (
              event_key TEXT PRIMARY KEY, first_seen_ts INTEGER NOT NULL,
              last_seen_ts INTEGER NOT NULL, last_cluster_size INTEGER NOT NULL,
              last_indexed_ts INTEGER NOT NULL
            );
        """)
        conn.execute("""
            CREATE TABLE semantic_keys (
              norm_url TEXT PRIMARY KEY, semantic_event_key TEXT,
              semantic_fingerprint TEXT, model TEXT,
              created_at_ts INTEGER NOT NULL, error TEXT
            );
        """)
        conn.execute("""
            CREATE TABLE posted_events (
              id INTEGER PRIMARY KEY, event_type TEXT NOT NULL,
              jurisdiction TEXT NOT NULL, entities TEXT NOT NULL,
              key_numbers TEXT NOT NULL DEFAULT '[]', summary_en TEXT,
              title TEXT NOT NULL, primary_url TEXT, dedupe_key TEXT,
              thread_id TEXT, run_id TEXT, category TEXT,
              posted_at_ts INTEGER NOT NULL, expires_at_ts INTEGER NOT NULL
            );
        """)
        conn.execute("CREATE TABLE fetch_state (key TEXT PRIMARY KEY, last_fetch_ts INTEGER NOT NULL, last_offset INTEGER NOT NULL DEFAULT 1, run_count INTEGER NOT NULL DEFAULT 0);")
        conn.execute("CREATE TABLE article_cache (norm_url TEXT PRIMARY KEY, url TEXT NOT NULL, final_url TEXT, domain TEXT, http_status INTEGER, extracted_title TEXT, extracted_text TEXT, extractor TEXT, fetched_at_ts INTEGER NOT NULL, text_chars INTEGER NOT NULL, quality_score INTEGER NOT NULL, error TEXT);")
        conn.execute("CREATE TABLE pool_runs (id INTEGER PRIMARY KEY, run_ts INTEGER NOT NULL, state_key TEXT NOT NULL, window_hours INTEGER NOT NULL, should_fetch INTEGER NOT NULL, query TEXT, offset_start INTEGER, pages INTEGER, count INTEGER, freshness TEXT, requests_made INTEGER NOT NULL DEFAULT 0, results INTEGER NOT NULL DEFAULT 0, inserted INTEGER NOT NULL DEFAULT 0, updated INTEGER NOT NULL DEFAULT 0, pruned INTEGER NOT NULL DEFAULT 0, pruned_articles INTEGER NOT NULL DEFAULT 0, notes TEXT);")

        # Insert some test data.
        conn.execute(
            "INSERT INTO links(url, norm_url, domain, title, description, age, page_age, first_seen_ts, last_seen_ts, seen_count, last_query, last_offset, last_fetched_at_ts) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?)",
            ("https://example.com/1", "https://example.com/1", "example.com", "Test Article", "Description", "2h", "2026-02-07T08:00:00", 1000, 2000, "test", 1, 2000),
        )
        conn.execute(
            "INSERT INTO events(event_key, first_seen_ts, last_seen_ts, last_cluster_size, last_indexed_ts) VALUES(?, ?, ?, ?, ?)",
            ("event:abc123", 1000, 2000, 3, 2000),
        )
        conn.execute(
            "INSERT INTO posted_events(event_type, jurisdiction, entities, title, primary_url, thread_id, run_id, category, summary_en, posted_at_ts, expires_at_ts) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("POLITICAL_EVENT", "UK", '["starmer"]', "PM faces crisis", "https://example.com/1", "thread_123", "run_456", "UK Parliament / Politics", "UK PM faces political crisis", int(time.time()) - 3600, int(time.time()) + 48 * 3600),
        )
        conn.commit()
        conn.close()

    def test_migration_v4_to_v6(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "news_pool.sqlite3"
            self._create_v4_db(db_path)

            with NewsPoolDB(path=db_path) as db:
                self.assertEqual(db.get_meta_int("schema_version"), 6)
                row = db._conn.execute("PRAGMA foreign_keys;").fetchone()
                assert row is not None
                self.assertEqual(int(row[0]), 1)

                # Old tables renamed to _legacy.
                conn = db._conn
                self.assertTrue(NewsPoolDB._table_exists(conn.cursor(), "events_legacy"))
                self.assertTrue(NewsPoolDB._table_exists(conn.cursor(), "semantic_keys_legacy"))
                self.assertTrue(NewsPoolDB._table_exists(conn.cursor(), "posted_events_legacy"))

                # New events table exists and has migrated data.
                self.assertTrue(NewsPoolDB._table_exists(conn.cursor(), "events"))
                events = db.get_fresh_events(now_ts=0)
                self.assertGreaterEqual(len(events), 1)
                migrated = events[-1]  # Oldest first in fresh_events (DESC by created_at_ts)
                self.assertEqual(migrated["status"], "posted")
                self.assertEqual(migrated["jurisdiction"], "UK")
                self.assertIn("crisis", migrated["summary_en"].lower())

                # Links have new columns.
                links = list(db.iter_links_since(cutoff_ts=0))
                self.assertEqual(len(links), 1)
                self.assertIn("published_at_ts", links[0])
                self.assertIsNotNone(links[0]["published_at_ts"])

    def test_migration_preserves_legacy_compat(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "news_pool.sqlite3"
            self._create_v4_db(db_path)

            with NewsPoolDB(path=db_path) as db:
                # Legacy methods should still work via _legacy tables.
                state = db.get_event_state(event_key="event:abc123")
                self.assertIsNotNone(state)
                assert state is not None
                self.assertEqual(state["last_cluster_size"], 3)


class TestNewsPoolDBEvents(unittest.TestCase):
    """Test event CRUD operations."""

    def test_create_and_get_event(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "news_pool.sqlite3"
            with NewsPoolDB(path=db_path) as db:
                eid = db.create_event(
                    summary_en="Tesla Q4 earnings beat expectations",
                    category="US Stocks",
                    jurisdiction="US",
                    title="Tesla Q4 Earnings",
                    primary_url="https://example.com/tesla",
                )
                ev = db.get_event(eid)
                self.assertIsNotNone(ev)
                assert ev is not None
                self.assertEqual(ev["status"], "new")
                self.assertEqual(ev["link_count"], 0)
                self.assertIsNotNone(ev["expires_at_ts"])

    def test_create_event_persists_entity_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "news_pool.sqlite3"
            with NewsPoolDB(path=db_path) as db:
                eid = db.create_event(
                    summary_en="Jimmy Lai sentenced in Hong Kong",
                    category="Hong Kong News",
                    jurisdiction="HK",
                    entity_aliases=[
                        {"label": "Jimmy Lai", "aliases": ["黎智英", "Lai"]},
                        {"entity": "Hong Kong Court", "zh": "香港法院"},
                    ],
                )
                ev = db.get_event(eid)
                self.assertIsNotNone(ev)
                assert ev is not None
                self.assertIsInstance(ev.get("entity_aliases_json"), str)
                self.assertEqual(
                    ev.get("entity_aliases"),
                    [
                        {"label": "Jimmy Lai", "aliases": ["黎智英", "Lai"]},
                        {"label": "Hong Kong Court", "aliases": ["香港法院"]},
                    ],
                )

    def test_create_development(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "news_pool.sqlite3"
            with NewsPoolDB(path=db_path) as db:
                parent_id = db.create_event(
                    summary_en="Mandelson faces inquiry",
                    category="UK Parliament / Politics",
                    jurisdiction="UK",
                )
                parent_before = db.get_event(parent_id)
                assert parent_before is not None
                old_expires = parent_before["expires_at_ts"]

                child_id = db.create_event(
                    summary_en="Police search Mandelson homes",
                    category="UK Parliament / Politics",
                    jurisdiction="UK",
                    parent_event_id=parent_id,
                    development="Police raid homes",
                )
                child = db.get_event(child_id)
                self.assertIsNotNone(child)
                assert child is not None
                self.assertEqual(child["parent_event_id"], parent_id)
                self.assertEqual(child["development"], "Police raid homes")

                # Parent's expires_at_ts should be bumped.
                parent_after = db.get_event(parent_id)
                assert parent_after is not None
                self.assertGreaterEqual(parent_after["expires_at_ts"], old_expires)

    def test_assign_link_to_event(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "news_pool.sqlite3"
            with NewsPoolDB(path=db_path) as db:
                link = PoolLink(
                    url="https://bbc.co.uk/news/1",
                    norm_url="https://bbc.co.uk/news/1",
                    domain="bbc.co.uk",
                    title="PM faces crisis",
                    description="...",
                    age=None,
                    page_age="2026-02-07T10:00:00",
                    query="test",
                    offset=1,
                    fetched_at_ts=int(time.time()),
                )
                db.upsert_links([link])

                eid = db.create_event(summary_en="UK PM crisis", category="Politics", jurisdiction="UK")

                links = db.get_unassigned_links()
                self.assertEqual(len(links), 1)
                link_id = links[0]["id"]

                db.assign_link_to_event(link_id=link_id, event_id=eid)

                ev = db.get_event(eid)
                assert ev is not None
                self.assertEqual(ev["link_count"], 1)
                self.assertEqual(ev["status"], "active")  # Promoted from 'new'.

                # Link should no longer appear as unassigned.
                remaining = db.get_unassigned_links()
                self.assertEqual(len(remaining), 0)

    def test_mark_event_posted(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "news_pool.sqlite3"
            with NewsPoolDB(path=db_path) as db:
                eid = db.create_event(summary_en="Test event", category="AI", jurisdiction="US")
                db.mark_event_posted(eid, thread_id="t_123", run_id="r_456")

                ev = db.get_event(eid)
                assert ev is not None
                self.assertEqual(ev["status"], "posted")
                self.assertEqual(ev["thread_id"], "t_123")
                self.assertEqual(ev["run_id"], "r_456")
                self.assertIsNotNone(ev["posted_at_ts"])

    def test_get_fresh_events(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "news_pool.sqlite3"
            with NewsPoolDB(path=db_path) as db:
                db.create_event(summary_en="Fresh event", category="AI", jurisdiction="US")
                events = db.get_fresh_events()
                self.assertEqual(len(events), 1)

                # With a future now_ts, no events should be fresh.
                future = int(time.time()) + 100 * 3600
                events = db.get_fresh_events(now_ts=future)
                self.assertEqual(len(events), 0)


class TestNewsPoolDBV5ToV6Migration(unittest.TestCase):
    """Test migration from v5 to v6 (reserved_until_ts column)."""

    def _create_v5_db(self, path: Path) -> None:
        conn = sqlite3.connect(str(path))
        conn.execute("CREATE TABLE meta (k TEXT PRIMARY KEY, v TEXT NOT NULL);")
        conn.execute("INSERT INTO meta(k, v) VALUES('schema_version', '5');")
        conn.execute("""
            CREATE TABLE events (
              id INTEGER PRIMARY KEY,
              parent_event_id INTEGER REFERENCES events(id),
              category TEXT, jurisdiction TEXT, summary_en TEXT NOT NULL,
              development TEXT, title TEXT, primary_url TEXT,
              link_count INTEGER NOT NULL DEFAULT 0, best_published_ts INTEGER,
              status TEXT NOT NULL DEFAULT 'new'
                CHECK (status IN ('new', 'active', 'posted')),
              created_at_ts INTEGER NOT NULL, updated_at_ts INTEGER NOT NULL,
              expires_at_ts INTEGER, posted_at_ts INTEGER,
              thread_id TEXT, run_id TEXT, model TEXT
            );
        """)
        conn.execute("""
            CREATE TABLE links (
              id INTEGER PRIMARY KEY, url TEXT NOT NULL, norm_url TEXT NOT NULL UNIQUE,
              domain TEXT, title TEXT, description TEXT, age TEXT, page_age TEXT,
              first_seen_ts INTEGER NOT NULL, last_seen_ts INTEGER NOT NULL,
              seen_count INTEGER NOT NULL DEFAULT 1, last_query TEXT NOT NULL,
              last_offset INTEGER NOT NULL, last_fetched_at_ts INTEGER NOT NULL,
              event_id INTEGER REFERENCES events(id), published_at_ts INTEGER
            );
        """)
        conn.execute("CREATE TABLE fetch_state (key TEXT PRIMARY KEY, last_fetch_ts INTEGER NOT NULL, last_offset INTEGER NOT NULL DEFAULT 1, run_count INTEGER NOT NULL DEFAULT 0);")
        conn.execute("CREATE TABLE article_cache (norm_url TEXT PRIMARY KEY, url TEXT NOT NULL, final_url TEXT, domain TEXT, http_status INTEGER, extracted_title TEXT, extracted_text TEXT, extractor TEXT, fetched_at_ts INTEGER NOT NULL, text_chars INTEGER NOT NULL, quality_score INTEGER NOT NULL, error TEXT);")
        conn.execute("CREATE TABLE pool_runs (id INTEGER PRIMARY KEY, run_ts INTEGER NOT NULL, state_key TEXT NOT NULL, window_hours INTEGER NOT NULL, should_fetch INTEGER NOT NULL, query TEXT, offset_start INTEGER, pages INTEGER, count INTEGER, freshness TEXT, requests_made INTEGER NOT NULL DEFAULT 0, results INTEGER NOT NULL DEFAULT 0, inserted INTEGER NOT NULL DEFAULT 0, updated INTEGER NOT NULL DEFAULT 0, pruned INTEGER NOT NULL DEFAULT 0, pruned_articles INTEGER NOT NULL DEFAULT 0, notes TEXT);")
        now = int(time.time())
        conn.execute(
            "INSERT INTO events(category, jurisdiction, summary_en, status, created_at_ts, updated_at_ts, expires_at_ts) VALUES(?, ?, ?, 'new', ?, ?, ?)",
            ("AI", "US", "Test v5 event", now, now, now + 48 * 3600),
        )
        conn.commit()
        conn.close()

    def test_migration_v5_to_v6_adds_column(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "news_pool.sqlite3"
            self._create_v5_db(db_path)

            with NewsPoolDB(path=db_path) as db:
                self.assertEqual(db.get_meta_int("schema_version"), 6)
                row = db._conn.execute("PRAGMA foreign_keys;").fetchone()
                assert row is not None
                self.assertEqual(int(row[0]), 1)

                # reserved_until_ts column should exist.
                cur = db._conn.cursor()
                self.assertTrue(NewsPoolDB._column_exists(cur, "events", "reserved_until_ts"))

                # Existing events should still be queryable.
                events = db.get_fresh_events(now_ts=0)
                self.assertGreaterEqual(len(events), 1)

    def test_migration_v5_to_v6_existing_events_have_null_reservation(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "news_pool.sqlite3"
            self._create_v5_db(db_path)

            with NewsPoolDB(path=db_path) as db:
                # Migrated events should have reserved_until_ts = NULL (selectable).
                candidates = db.get_daily_candidates(limit=10)
                self.assertGreater(len(candidates), 0)


class TestEventReservation(unittest.TestCase):
    """Test planner-level event reservation to prevent concurrent selection."""

    def test_daily_candidates_are_reserved(self) -> None:
        """After get_daily_candidates(), the same events should not be returned by a second call."""
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "news_pool.sqlite3"
            with NewsPoolDB(path=db_path) as db:
                now = int(time.time())
                db.create_event(summary_en="Event A", category="AI", jurisdiction="US")
                db.create_event(summary_en="Event B", category="Politics", jurisdiction="UK")

                first_call = db.get_daily_candidates(limit=10, now_ts=now)
                self.assertEqual(len(first_call), 2)

                # Second call at the same time should return nothing (both reserved).
                second_call = db.get_daily_candidates(limit=10, now_ts=now)
                self.assertEqual(len(second_call), 0)

    def test_hourly_candidates_are_reserved(self) -> None:
        """After get_hourly_candidates(), the same events should not be returned by a second call."""
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "news_pool.sqlite3"
            with NewsPoolDB(path=db_path) as db:
                now = int(time.time())
                db.create_event(summary_en="Event A", category="AI", jurisdiction="US")
                db.create_event(summary_en="Event B", category="Sports", jurisdiction="UK")

                first_call = db.get_hourly_candidates(limit=3, now_ts=now)
                self.assertEqual(len(first_call), 2)

                second_call = db.get_hourly_candidates(limit=3, now_ts=now)
                self.assertEqual(len(second_call), 0)

    def test_reservation_expires_after_timeout(self) -> None:
        """Reservations auto-expire after _RESERVATION_SECONDS."""
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "news_pool.sqlite3"
            with NewsPoolDB(path=db_path) as db:
                now = int(time.time())
                db.create_event(summary_en="Event A", category="AI", jurisdiction="US")

                first_call = db.get_daily_candidates(limit=10, now_ts=now)
                self.assertEqual(len(first_call), 1)

                # Still reserved.
                mid_call = db.get_daily_candidates(limit=10, now_ts=now + _RESERVATION_SECONDS - 1)
                self.assertEqual(len(mid_call), 0)

                # Reservation expired.
                after_expiry = now + _RESERVATION_SECONDS + 1
                expired_call = db.get_daily_candidates(limit=10, now_ts=after_expiry)
                self.assertEqual(len(expired_call), 1)

    def test_release_reservation(self) -> None:
        """release_reservation() makes the event selectable again immediately."""
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "news_pool.sqlite3"
            with NewsPoolDB(path=db_path) as db:
                now = int(time.time())
                eid = db.create_event(summary_en="Event A", category="AI", jurisdiction="US")

                first_call = db.get_daily_candidates(limit=10, now_ts=now)
                self.assertEqual(len(first_call), 1)

                # Release reservation.
                db.release_reservation(eid)

                # Should be selectable again.
                second_call = db.get_daily_candidates(limit=10, now_ts=now)
                self.assertEqual(len(second_call), 1)

    def test_mark_posted_clears_from_candidates(self) -> None:
        """Posted events should never appear as candidates, reservation or not."""
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "news_pool.sqlite3"
            with NewsPoolDB(path=db_path) as db:
                now = int(time.time())
                eid = db.create_event(summary_en="Event A", category="AI", jurisdiction="US")
                db.mark_event_posted(eid, thread_id="t1", run_id="r1")

                candidates = db.get_daily_candidates(limit=10, now_ts=now)
                self.assertEqual(len(candidates), 0)

    def test_concurrent_daily_and_hourly_do_not_overlap(self) -> None:
        """Events reserved by daily should not be picked by hourly."""
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "news_pool.sqlite3"
            with NewsPoolDB(path=db_path) as db:
                now = int(time.time())
                db.create_event(summary_en="Event A", category="AI", jurisdiction="US")
                db.create_event(summary_en="Event B", category="Sports", jurisdiction="UK")

                daily = db.get_daily_candidates(limit=10, now_ts=now)
                self.assertEqual(len(daily), 2)

                hourly = db.get_hourly_candidates(limit=3, now_ts=now)
                self.assertEqual(len(hourly), 0)


class TestDailySelection(unittest.TestCase):
    """Test daily candidate selection logic."""

    def test_finance_cap(self) -> None:
        candidates = [
            {"id": 1, "category": "US Stocks", "jurisdiction": "US"},
            {"id": 2, "category": "Crypto", "jurisdiction": "GLOBAL"},
            {"id": 3, "category": "US Stocks", "jurisdiction": "US"},
            {"id": 4, "category": "Politics", "jurisdiction": "US"},
            {"id": 5, "category": "AI", "jurisdiction": "US"},
        ]
        selected = NewsPoolDB._apply_daily_selection(candidates, limit=5)
        finance_selected = [c for c in selected if NewsPoolDB._is_finance_category(c["category"])]
        self.assertLessEqual(len(finance_selected), 2)
        # Non-finance should be included.
        non_finance = [c for c in selected if not NewsPoolDB._is_finance_category(c["category"])]
        self.assertGreater(len(non_finance), 0)

    def test_category_cap(self) -> None:
        candidates = [
            {"id": i, "category": "Politics", "jurisdiction": "US"}
            for i in range(1, 10)
        ]
        selected = NewsPoolDB._apply_daily_selection(candidates, limit=10)
        self.assertLessEqual(len(selected), 3)

    def test_hk_guarantee(self) -> None:
        candidates = [
            {"id": 1, "category": "AI", "jurisdiction": "US"},
            {"id": 2, "category": "Politics", "jurisdiction": "UK"},
            {"id": 3, "category": "Hong Kong News", "jurisdiction": "HK"},
            {"id": 4, "category": "Sports", "jurisdiction": "US"},
        ]
        selected = NewsPoolDB._apply_daily_selection(candidates, limit=4)
        hk_selected = [c for c in selected if (c.get("jurisdiction") or "").upper() == "HK"]
        self.assertGreaterEqual(len(hk_selected), 1)


class TestHourlySelection(unittest.TestCase):
    """Test hourly candidate selection logic."""

    def test_development_priority(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "news_pool.sqlite3"
            with NewsPoolDB(path=db_path) as db:
                # Create a root event and a development.
                root_id = db.create_event(summary_en="Root event", category="Politics", jurisdiction="UK")
                dev_id = db.create_event(
                    summary_en="Development",
                    category="Politics",
                    jurisdiction="UK",
                    parent_event_id=root_id,
                    development="New development",
                )
                db.create_event(summary_en="Other event", category="AI", jurisdiction="US")

                candidates = db.get_hourly_candidates(limit=3)
                self.assertGreater(len(candidates), 0)
                # Development should be first.
                self.assertEqual(candidates[0]["id"], dev_id)

    def test_category_diversity(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "news_pool.sqlite3"
            with NewsPoolDB(path=db_path) as db:
                db.create_event(summary_en="Event 1", category="AI", jurisdiction="US")
                db.create_event(summary_en="Event 2", category="AI", jurisdiction="US")
                db.create_event(summary_en="Event 3", category="Sports", jurisdiction="UK")

                candidates = db.get_hourly_candidates(limit=2)
                self.assertEqual(len(candidates), 2)
                categories = [c["category"] for c in candidates]
                # Should have diversity (not both AI).
                self.assertIn("Sports", categories)


class TestPoolRunsLedger(unittest.TestCase):
    def test_pool_runs_ledger_records_requests(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "news_pool.sqlite3"
            with NewsPoolDB(path=db_path) as db:
                db.log_pool_run(
                    run_ts=123,
                    state_key="hourly",
                    window_hours=48,
                    should_fetch=True,
                    query="breaking news",
                    offset_start=1,
                    pages=1,
                    count=20,
                    freshness="day",
                    requests_made=1,
                    results=20,
                    inserted=10,
                    updated=5,
                    pruned=0,
                    pruned_articles=0,
                    notes=None,
                )
                self.assertEqual(db.sum_requests_made(since_ts=0), 1)
                self.assertEqual(db.sum_requests_made(since_ts=0, state_key="hourly"), 1)
                self.assertEqual(db.requests_made_by_state(since_ts=0), [("hourly", 1)])


class TestArticleCache(unittest.TestCase):
    def test_article_cache_works(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "news_pool.sqlite3"
            with NewsPoolDB(path=db_path) as db:
                norm = "https://example.com/a"
                db.upsert_article_cache(
                    norm_url=norm,
                    url=norm,
                    final_url=norm,
                    domain="example.com",
                    http_status=200,
                    extracted_title="Hello",
                    extracted_text="Body text",
                    extractor="trafilatura",
                    fetched_at_ts=123,
                    quality_score=80,
                    error=None,
                )
                row = db.get_article_cache(norm_url=norm)
                self.assertIsNotNone(row)
                assert row is not None
                self.assertEqual(row["norm_url"], norm)
                self.assertEqual(row["quality_score"], 80)


class TestCandidateEnrichment(unittest.TestCase):
    """Test that candidates include backward-compat fields for newsroom_write_run_job.py."""

    def test_daily_candidates_have_compat_fields(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "news_pool.sqlite3"
            with NewsPoolDB(path=db_path) as db:
                eid = db.create_event(
                    summary_en="Tesla Q4 earnings beat",
                    category="US Stocks",
                    jurisdiction="US",
                    title="Tesla Q4 earnings",
                    primary_url="https://reuters.com/tesla-q4",
                )
                # Add two links to the event.
                link1 = PoolLink(
                    url="https://reuters.com/tesla-q4",
                    norm_url="https://reuters.com/tesla-q4",
                    domain="reuters.com",
                    title="Tesla Q4",
                    description="Earnings beat",
                    age=None,
                    page_age="2026-02-07T10:00:00",
                    query="test",
                    offset=1,
                    fetched_at_ts=int(time.time()),
                )
                link2 = PoolLink(
                    url="https://bbc.co.uk/tesla-q4",
                    norm_url="https://bbc.co.uk/tesla-q4",
                    domain="bbc.co.uk",
                    title="Tesla earnings",
                    description="Beat expectations",
                    age=None,
                    page_age="2026-02-07T11:00:00",
                    query="test",
                    offset=1,
                    fetched_at_ts=int(time.time()),
                )
                db.upsert_links([link1, link2])
                # Get link IDs for assignment.
                unassigned = db.get_unassigned_links()
                for lnk in unassigned:
                    db.assign_link_to_event(link_id=lnk["id"], event_id=eid)

                candidates = db.get_daily_candidates(limit=5)
                self.assertGreater(len(candidates), 0)
                c = candidates[0]

                # Core compat fields.
                self.assertEqual(c["suggested_category"], "US Stocks")
                self.assertEqual(c["description"], "Tesla Q4 earnings beat")
                self.assertEqual(c["event_key"], f"event:{eid}")
                self.assertEqual(c["semantic_event_key"], f"event:{eid}")
                self.assertEqual(c["event_id"], eid)
                self.assertEqual(c["cluster_size"], 2)
                self.assertIsInstance(c["age_minutes"], int)

                # supporting_urls: should have the non-primary URL.
                self.assertIn("https://bbc.co.uk/tesla-q4", c["supporting_urls"])
                # domains.
                self.assertIn("reuters.com", c["domains"])
                self.assertIn("bbc.co.uk", c["domains"])

    def test_hourly_candidates_have_compat_fields(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "news_pool.sqlite3"
            with NewsPoolDB(path=db_path) as db:
                eid = db.create_event(
                    summary_en="UK PM crisis",
                    category="Politics",
                    jurisdiction="UK",
                )
                candidates = db.get_hourly_candidates(limit=3)
                self.assertGreater(len(candidates), 0)
                c = candidates[0]
                self.assertIn("suggested_category", c)
                self.assertIn("event_id", c)
                self.assertIn("supporting_urls", c)
                self.assertIn("domains", c)

    def test_candidate_enrichment_includes_entity_aliases_terms(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "news_pool.sqlite3"
            with NewsPoolDB(path=db_path) as db:
                eid = db.create_event(
                    summary_en="Sentencing update",
                    category="Hong Kong News",
                    jurisdiction="HK",
                    entity_aliases=[
                        {"label": "Jimmy Lai", "aliases": ["黎智英"]},
                        {"label": "Hong Kong Court", "aliases": ["香港法院"]},
                    ],
                )
                link = PoolLink(
                    url="https://example.com/hk-case",
                    norm_url="https://example.com/hk-case",
                    domain="example.com",
                    title="Case update",
                    description="Sentencing update",
                    age=None,
                    page_age="2026-02-07T10:00:00",
                    query="test",
                    offset=1,
                    fetched_at_ts=int(time.time()),
                )
                db.upsert_links([link])
                unassigned = db.get_unassigned_links()
                self.assertEqual(len(unassigned), 1)
                db.assign_link_to_event(link_id=unassigned[0]["id"], event_id=eid)

                candidates = db.get_daily_candidates(limit=5)
                self.assertGreater(len(candidates), 0)
                c = candidates[0]
                self.assertEqual(
                    c.get("entity_aliases"),
                    [
                        {"label": "Jimmy Lai", "aliases": ["黎智英"]},
                        {"label": "Hong Kong Court", "aliases": ["香港法院"]},
                    ],
                )
                self.assertIn("Jimmy Lai", c.get("cluster_terms", []))
                self.assertIn("黎智英", c.get("anchor_terms", []))


class TestMergeEventsInto(unittest.TestCase):
    """Test NewsPoolDB.merge_events_into()."""

    def _setup_db(self, td: str) -> NewsPoolDB:
        return NewsPoolDB(path=Path(td) / "news_pool.sqlite3")

    def test_merge_events_into_links_moved(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            with self._setup_db(td) as db:
                winner_id = db.create_event(summary_en="Winner", category="UK News", jurisdiction="UK")
                loser_id = db.create_event(summary_en="Loser", category="UK News", jurisdiction="UK")

                # Add links to loser.
                for i in range(2):
                    link = PoolLink(
                        url=f"https://example.com/l{i}", norm_url=f"https://example.com/l{i}",
                        domain="example.com", title=f"Link {i}", description="...",
                        age=None, page_age=None, query="test", offset=1,
                        fetched_at_ts=int(time.time()),
                    )
                    db.upsert_links([link])
                    unassigned = db.get_unassigned_links()
                    db.assign_link_to_event(link_id=unassigned[0]["id"], event_id=loser_id)

                links_moved = db.merge_events_into(winner_id=winner_id, loser_ids=[loser_id])
                self.assertEqual(links_moved, 2)

                # Winner should have 2 links.
                ev = db.get_event(winner_id)
                assert ev is not None
                self.assertEqual(ev["link_count"], 2)

    def test_merge_events_into_children_reparented(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            with self._setup_db(td) as db:
                winner_id = db.create_event(summary_en="Winner", category="Politics", jurisdiction="UK")
                loser_id = db.create_event(summary_en="Loser", category="Politics", jurisdiction="UK")
                child_id = db.create_event(
                    summary_en="Child of loser", category="Politics",
                    jurisdiction="UK", parent_event_id=loser_id,
                )

                db.merge_events_into(winner_id=winner_id, loser_ids=[loser_id])

                # Child should now point to winner.
                child = db.get_event(child_id)
                assert child is not None
                self.assertEqual(child["parent_event_id"], winner_id)

    def test_merge_events_into_losers_deleted(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            with self._setup_db(td) as db:
                winner_id = db.create_event(summary_en="Winner", category="AI", jurisdiction="US")
                loser1 = db.create_event(summary_en="Loser 1", category="AI", jurisdiction="US")
                loser2 = db.create_event(summary_en="Loser 2", category="AI", jurisdiction="US")

                db.merge_events_into(winner_id=winner_id, loser_ids=[loser1, loser2])

                self.assertIsNone(db.get_event(loser1))
                self.assertIsNone(db.get_event(loser2))
                self.assertIsNotNone(db.get_event(winner_id))

    def test_merge_events_into_winner_promoted(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            with self._setup_db(td) as db:
                winner_id = db.create_event(summary_en="Winner", category="Sports", jurisdiction="UK")
                loser_id = db.create_event(summary_en="Loser", category="Sports", jurisdiction="UK")

                # Add links to both so winner ends up with >1 link.
                for i, eid in enumerate([winner_id, loser_id]):
                    link = PoolLink(
                        url=f"https://example.com/p{i}", norm_url=f"https://example.com/p{i}",
                        domain="example.com", title=f"Article {i}", description="...",
                        age=None, page_age=None, query="test", offset=1,
                        fetched_at_ts=int(time.time()),
                    )
                    db.upsert_links([link])
                    unassigned = db.get_unassigned_links()
                    db.assign_link_to_event(link_id=unassigned[0]["id"], event_id=eid)

                db.merge_events_into(winner_id=winner_id, loser_ids=[loser_id])

                ev = db.get_event(winner_id)
                assert ev is not None
                self.assertEqual(ev["status"], "active")
                self.assertEqual(ev["link_count"], 2)

    def test_merge_events_into_empty_losers(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            with self._setup_db(td) as db:
                winner_id = db.create_event(summary_en="Winner", category="AI", jurisdiction="US")
                result = db.merge_events_into(winner_id=winner_id, loser_ids=[])
                self.assertEqual(result, 0)


class TestPruneExpiredEvents(unittest.TestCase):
    """Test NewsPoolDB.prune_expired_events()."""

    def _setup_db(self, td: str) -> NewsPoolDB:
        return NewsPoolDB(path=Path(td) / "news_pool.sqlite3")

    def test_expired_old_events_are_pruned(self) -> None:
        """Events that are both expired and older than max_age_hours get deleted."""
        with tempfile.TemporaryDirectory() as td:
            with self._setup_db(td) as db:
                now = int(time.time())
                eid = db.create_event(summary_en="Old expired event", category="AI", jurisdiction="US")
                # Backdate created_at_ts and set expires_at_ts in the past.
                db._conn.execute(
                    "UPDATE events SET created_at_ts = ?, expires_at_ts = ? WHERE id = ?",
                    (now - 80 * 3600, now - 10 * 3600, eid),
                )

                pruned = db.prune_expired_events(max_age_hours=72, now_ts=now)
                self.assertEqual(pruned, 1)
                self.assertIsNone(db.get_event(eid))

    def test_posted_events_are_not_pruned(self) -> None:
        """Posted events are kept for audit trail, even if expired and old."""
        with tempfile.TemporaryDirectory() as td:
            with self._setup_db(td) as db:
                now = int(time.time())
                eid = db.create_event(summary_en="Posted event", category="AI", jurisdiction="US")
                db.mark_event_posted(eid, thread_id="t_1", run_id="r_1")
                # Backdate and expire.
                db._conn.execute(
                    "UPDATE events SET created_at_ts = ?, expires_at_ts = ? WHERE id = ?",
                    (now - 80 * 3600, now - 10 * 3600, eid),
                )

                pruned = db.prune_expired_events(max_age_hours=72, now_ts=now)
                self.assertEqual(pruned, 0)
                self.assertIsNotNone(db.get_event(eid))

    def test_non_expired_active_events_are_not_pruned(self) -> None:
        """Events that haven't expired yet are not pruned, even if old."""
        with tempfile.TemporaryDirectory() as td:
            with self._setup_db(td) as db:
                now = int(time.time())
                eid = db.create_event(summary_en="Still fresh", category="AI", jurisdiction="US")
                # Old but not expired.
                db._conn.execute(
                    "UPDATE events SET created_at_ts = ?, expires_at_ts = ? WHERE id = ?",
                    (now - 80 * 3600, now + 3600, eid),
                )

                pruned = db.prune_expired_events(max_age_hours=72, now_ts=now)
                self.assertEqual(pruned, 0)
                self.assertIsNotNone(db.get_event(eid))

    def test_recently_created_expired_events_are_not_pruned(self) -> None:
        """Events that are expired but created recently (within max_age_hours) are not pruned."""
        with tempfile.TemporaryDirectory() as td:
            with self._setup_db(td) as db:
                now = int(time.time())
                eid = db.create_event(summary_en="Recent expired", category="AI", jurisdiction="US")
                # Expired but created recently (1 hour ago).
                db._conn.execute(
                    "UPDATE events SET created_at_ts = ?, expires_at_ts = ? WHERE id = ?",
                    (now - 3600, now - 60, eid),
                )

                pruned = db.prune_expired_events(max_age_hours=72, now_ts=now)
                self.assertEqual(pruned, 0)
                self.assertIsNotNone(db.get_event(eid))

    def test_orphaned_links_are_cleaned_up(self) -> None:
        """Links referencing pruned events get their event_id set to NULL."""
        with tempfile.TemporaryDirectory() as td:
            with self._setup_db(td) as db:
                now = int(time.time())
                eid = db.create_event(summary_en="Will be pruned", category="AI", jurisdiction="US")

                # Add and assign a link.
                link = PoolLink(
                    url="https://example.com/orphan",
                    norm_url="https://example.com/orphan",
                    domain="example.com",
                    title="Orphan link",
                    description="...",
                    age=None,
                    page_age=None,
                    query="test",
                    offset=1,
                    fetched_at_ts=now,
                )
                db.upsert_links([link], now_ts=now)
                unassigned = db.get_unassigned_links()
                self.assertEqual(len(unassigned), 1)
                link_id = unassigned[0]["id"]
                db.assign_link_to_event(link_id=link_id, event_id=eid)

                # Verify link is assigned.
                remaining = db.get_unassigned_links()
                self.assertEqual(len(remaining), 0)

                # Backdate and expire the event.
                db._conn.execute(
                    "UPDATE events SET created_at_ts = ?, expires_at_ts = ? WHERE id = ?",
                    (now - 80 * 3600, now - 10 * 3600, eid),
                )

                pruned = db.prune_expired_events(max_age_hours=72, now_ts=now)
                self.assertEqual(pruned, 1)
                self.assertIsNone(db.get_event(eid))

                # Link should now be unassigned (event_id = NULL).
                unassigned_after = db.get_unassigned_links(max_age_seconds=200 * 3600)
                self.assertEqual(len(unassigned_after), 1)
                self.assertIsNone(unassigned_after[0].get("event_id"))

    def test_prune_returns_zero_when_nothing_to_prune(self) -> None:
        """No events match the criteria — returns 0."""
        with tempfile.TemporaryDirectory() as td:
            with self._setup_db(td) as db:
                now = int(time.time())
                db.create_event(summary_en="Fresh event", category="AI", jurisdiction="US")
                pruned = db.prune_expired_events(max_age_hours=72, now_ts=now)
                self.assertEqual(pruned, 0)


class TestLinkCountDrift(unittest.TestCase):
    """Test that link_count stays accurate across reassignments and pruning."""

    def _setup_db(self, td: str) -> NewsPoolDB:
        return NewsPoolDB(path=Path(td) / "news_pool.sqlite3")

    def test_assign_link_to_event_decrements_old_event(self) -> None:
        """Moving a link from event A to event B decrements A's link_count."""
        with tempfile.TemporaryDirectory() as td:
            with self._setup_db(td) as db:
                eid_a = db.create_event(summary_en="Event A", category="AI", jurisdiction="US")
                eid_b = db.create_event(summary_en="Event B", category="AI", jurisdiction="US")

                link = PoolLink(
                    url="https://example.com/x", norm_url="https://example.com/x",
                    domain="example.com", title="Link X", description="...",
                    age=None, page_age=None, query="test", offset=1,
                    fetched_at_ts=int(time.time()),
                )
                db.upsert_links([link])
                link_id = db.get_unassigned_links()[0]["id"]

                # Assign to A.
                db.assign_link_to_event(link_id=link_id, event_id=eid_a)
                ev_a = db.get_event(eid_a)
                assert ev_a is not None
                self.assertEqual(ev_a["link_count"], 1)

                # Reassign to B.
                db.assign_link_to_event(link_id=link_id, event_id=eid_b)
                ev_a = db.get_event(eid_a)
                ev_b = db.get_event(eid_b)
                assert ev_a is not None
                assert ev_b is not None
                self.assertEqual(ev_a["link_count"], 0, "Old event should have link_count decremented")
                self.assertEqual(ev_b["link_count"], 1, "New event should have link_count incremented")

    def test_assign_link_to_same_event_idempotent(self) -> None:
        """Re-assigning a link to the same event should not change link_count."""
        with tempfile.TemporaryDirectory() as td:
            with self._setup_db(td) as db:
                eid = db.create_event(summary_en="Event A", category="AI", jurisdiction="US")

                link = PoolLink(
                    url="https://example.com/y", norm_url="https://example.com/y",
                    domain="example.com", title="Link Y", description="...",
                    age=None, page_age=None, query="test", offset=1,
                    fetched_at_ts=int(time.time()),
                )
                db.upsert_links([link])
                link_id = db.get_unassigned_links()[0]["id"]

                db.assign_link_to_event(link_id=link_id, event_id=eid)
                ev = db.get_event(eid)
                assert ev is not None
                self.assertEqual(ev["link_count"], 1)

                # Assign again to same event.
                db.assign_link_to_event(link_id=link_id, event_id=eid)
                ev = db.get_event(eid)
                assert ev is not None
                self.assertEqual(ev["link_count"], 1, "link_count should not double on idempotent assign")

    def test_prune_links_updates_event_link_count(self) -> None:
        """Pruning old links should decrement affected events' link_count."""
        with tempfile.TemporaryDirectory() as td:
            with self._setup_db(td) as db:
                now = int(time.time())
                eid = db.create_event(summary_en="Event", category="AI", jurisdiction="US")

                # Insert two links: one old (will be pruned), one recent (will survive).
                old_link = PoolLink(
                    url="https://example.com/old", norm_url="https://example.com/old",
                    domain="example.com", title="Old link", description="...",
                    age=None, page_age=None, query="test", offset=1,
                    fetched_at_ts=now - 1000,
                )
                new_link = PoolLink(
                    url="https://example.com/new", norm_url="https://example.com/new",
                    domain="example.com", title="New link", description="...",
                    age=None, page_age=None, query="test", offset=1,
                    fetched_at_ts=now,
                )
                db.upsert_links([old_link], now_ts=now - 1000)
                db.upsert_links([new_link], now_ts=now)

                # Assign both links to the event.
                all_links = list(db.iter_links_since(cutoff_ts=0))
                for lnk in all_links:
                    db.assign_link_to_event(link_id=lnk["id"], event_id=eid)

                ev = db.get_event(eid)
                assert ev is not None
                self.assertEqual(ev["link_count"], 2)

                # Prune the old link (cutoff between old and new).
                cutoff = now - 500
                pruned = db.prune_links(cutoff_ts=cutoff)
                self.assertEqual(pruned, 1)

                # Event should now have link_count = 1.
                ev = db.get_event(eid)
                assert ev is not None
                self.assertEqual(ev["link_count"], 1, "link_count should reflect pruned links")

    def test_prune_links_all_removed_sets_count_to_zero(self) -> None:
        """Pruning all links from an event should set link_count to 0."""
        with tempfile.TemporaryDirectory() as td:
            with self._setup_db(td) as db:
                now = int(time.time())
                eid = db.create_event(summary_en="Event", category="AI", jurisdiction="US")

                link = PoolLink(
                    url="https://example.com/z", norm_url="https://example.com/z",
                    domain="example.com", title="Link Z", description="...",
                    age=None, page_age=None, query="test", offset=1,
                    fetched_at_ts=now - 1000,
                )
                db.upsert_links([link], now_ts=now - 1000)
                link_id = list(db.iter_links_since(cutoff_ts=0))[0]["id"]
                db.assign_link_to_event(link_id=link_id, event_id=eid)

                ev = db.get_event(eid)
                assert ev is not None
                self.assertEqual(ev["link_count"], 1)

                # Prune all links.
                pruned = db.prune_links(cutoff_ts=now)
                self.assertEqual(pruned, 1)

                ev = db.get_event(eid)
                assert ev is not None
                self.assertEqual(ev["link_count"], 0, "link_count should be 0 after all links pruned")

    def test_prune_links_no_assigned_links_no_side_effect(self) -> None:
        """Pruning links that are not assigned to any event has no side effects."""
        with tempfile.TemporaryDirectory() as td:
            with self._setup_db(td) as db:
                now = int(time.time())
                eid = db.create_event(summary_en="Event", category="AI", jurisdiction="US")

                # Insert an unassigned link.
                link = PoolLink(
                    url="https://example.com/unassigned", norm_url="https://example.com/unassigned",
                    domain="example.com", title="Unassigned", description="...",
                    age=None, page_age=None, query="test", offset=1,
                    fetched_at_ts=now - 1000,
                )
                db.upsert_links([link], now_ts=now - 1000)

                pruned = db.prune_links(cutoff_ts=now)
                self.assertEqual(pruned, 1)

                # Event link_count should remain 0.
                ev = db.get_event(eid)
                assert ev is not None
                self.assertEqual(ev["link_count"], 0)


if __name__ == "__main__":
    unittest.main()
