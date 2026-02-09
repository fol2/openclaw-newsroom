import sqlite3
import tempfile
import time
import unittest
from pathlib import Path

from newsroom.news_pool_db import NewsPoolDB, PoolLink, SCHEMA_VERSION


class TestNewsPoolDBV5Fresh(unittest.TestCase):
    """Test fresh v5 database creation."""

    def test_fresh_db_creates_v5_schema(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "news_pool.sqlite3"
            with NewsPoolDB(path=db_path) as db:
                self.assertEqual(db.get_meta_int("schema_version"), 5)

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

    def test_migration_v4_to_v5(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "news_pool.sqlite3"
            self._create_v4_db(db_path)

            with NewsPoolDB(path=db_path) as db:
                self.assertEqual(db.get_meta_int("schema_version"), 5)

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


if __name__ == "__main__":
    unittest.main()
