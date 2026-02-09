"""Tests for newsroom.event_manager — LLM clustering logic."""
import json
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from newsroom.event_manager import (
    build_clustering_prompt,
    build_merge_prompt,
    cluster_all_pending,
    cluster_link,
    merge_events,
    parse_clustering_response,
    parse_merge_response,
)
from newsroom.news_pool_db import NewsPoolDB, PoolLink


class TestBuildClusteringPrompt(unittest.TestCase):
    def test_builds_prompt_with_no_events(self) -> None:
        link = {"title": "Tesla Q4 earnings beat", "description": "Revenue up 20%"}
        prompt = build_clustering_prompt(link, [])
        self.assertIn("Tesla Q4 earnings beat", prompt)
        self.assertIn("(no fresh events)", prompt)
        self.assertIn("action", prompt)

    def test_builds_prompt_with_events(self) -> None:
        link = {"title": "FCA probes Mandelson", "description": "Investigation opened"}
        events = [
            {
                "id": 42,
                "parent_event_id": None,
                "category": "UK Parliament / Politics",
                "jurisdiction": "UK",
                "summary_en": "Starmer faces crisis over Mandelson",
                "development": None,
                "title": "Starmer crisis",
                "status": "active",
            },
        ]
        prompt = build_clustering_prompt(link, events)
        self.assertIn("FCA probes Mandelson", prompt)
        self.assertIn("[id=42]", prompt)
        self.assertIn("Starmer faces crisis", prompt)

    def test_builds_prompt_with_parent_child(self) -> None:
        events = [
            {
                "id": 1,
                "parent_event_id": None,
                "category": "Politics",
                "jurisdiction": "UK",
                "summary_en": "Root event",
                "development": None,
                "title": "Root",
                "status": "active",
            },
            {
                "id": 2,
                "parent_event_id": 1,
                "category": "Politics",
                "jurisdiction": "UK",
                "summary_en": "Child event",
                "development": "Police raid",
                "title": "Child",
                "status": "new",
            },
        ]
        prompt = build_clustering_prompt({"title": "test"}, events)
        self.assertIn("[id=1]", prompt)
        self.assertIn("(development)", prompt)


class TestParseClusteringResponse(unittest.TestCase):
    def test_parse_assign(self) -> None:
        response = {"action": "assign", "event_id": 42}
        events = [{"id": 42}, {"id": 43}]
        result = parse_clustering_response(response, {}, events)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result["action"], "assign")
        self.assertEqual(result["event_id"], 42)

    def test_parse_assign_invalid_event_id(self) -> None:
        response = {"action": "assign", "event_id": 999}
        events = [{"id": 42}]
        result = parse_clustering_response(response, {}, events)
        self.assertIsNone(result)

    def test_parse_development(self) -> None:
        response = {
            "action": "development",
            "parent_event_id": 42,
            "summary_en": "FCA investigates Mandelson",
            "development": "FCA investigation",
            "category": "UK Parliament / Politics",
            "jurisdiction": "UK",
        }
        events = [{"id": 42}]
        result = parse_clustering_response(response, {}, events)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result["action"], "development")
        self.assertEqual(result["parent_event_id"], 42)
        self.assertEqual(result["summary_en"], "FCA investigates Mandelson")

    def test_parse_new_event(self) -> None:
        response = {
            "action": "new_event",
            "summary_en": "Tesla Q4 earnings beat expectations",
            "category": "US Stocks",
            "jurisdiction": "US",
        }
        result = parse_clustering_response(response, {}, [])
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result["action"], "new_event")
        self.assertEqual(result["summary_en"], "Tesla Q4 earnings beat expectations")

    def test_parse_new_event_missing_summary(self) -> None:
        response = {"action": "new_event", "category": "AI"}
        result = parse_clustering_response(response, {}, [])
        self.assertIsNone(result)

    def test_parse_unknown_action(self) -> None:
        response = {"action": "unknown"}
        result = parse_clustering_response(response, {}, [])
        self.assertIsNone(result)


class TestClusterLink(unittest.TestCase):
    def test_cluster_link_assign(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "news_pool.sqlite3"
            with NewsPoolDB(path=Path(db_path)) as db:
                # Create a link.
                link = PoolLink(
                    url="https://bbc.co.uk/1", norm_url="https://bbc.co.uk/1",
                    domain="bbc.co.uk", title="PM crisis", description="...",
                    age=None, page_age="2026-02-07T10:00:00", query="test", offset=1,
                    fetched_at_ts=int(time.time()),
                )
                db.upsert_links([link])
                links = db.get_unassigned_links()
                self.assertEqual(len(links), 1)

                # Create an event.
                eid = db.create_event(summary_en="UK PM crisis", category="Politics", jurisdiction="UK")

                # Mock Gemini to return assign.
                gemini = MagicMock()
                gemini.generate_json.return_value = {"action": "assign", "event_id": eid}

                fresh = db.get_fresh_events()
                result = cluster_link(link=links[0], fresh_events=fresh, gemini=gemini, db=db)
                self.assertIsNotNone(result)
                assert result is not None
                self.assertEqual(result["action"], "assign")

                ev = db.get_event(eid)
                assert ev is not None
                self.assertEqual(ev["link_count"], 1)

    def test_cluster_link_new_event(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "news_pool.sqlite3"
            with NewsPoolDB(path=Path(db_path)) as db:
                link = PoolLink(
                    url="https://bbc.co.uk/2", norm_url="https://bbc.co.uk/2",
                    domain="bbc.co.uk", title="New discovery", description="Scientists find...",
                    age=None, page_age="2026-02-07T12:00:00", query="test", offset=1,
                    fetched_at_ts=int(time.time()),
                )
                db.upsert_links([link])
                links = db.get_unassigned_links()

                gemini = MagicMock()
                gemini.generate_json.return_value = {
                    "action": "new_event",
                    "summary_en": "Scientists discover new species",
                    "category": "Global News",
                    "jurisdiction": "GLOBAL",
                }

                result = cluster_link(link=links[0], fresh_events=[], gemini=gemini, db=db)
                self.assertIsNotNone(result)
                assert result is not None
                self.assertEqual(result["action"], "new_event")
                self.assertIn("event_id", result)

                # Event should exist.
                ev = db.get_event(result["event_id"])
                self.assertIsNotNone(ev)
                assert ev is not None
                self.assertEqual(ev["summary_en"], "Scientists discover new species")

    def test_cluster_link_gemini_failure(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "news_pool.sqlite3"
            with NewsPoolDB(path=Path(db_path)) as db:
                link = PoolLink(
                    url="https://bbc.co.uk/3", norm_url="https://bbc.co.uk/3",
                    domain="bbc.co.uk", title="Some news", description="...",
                    age=None, page_age=None, query="test", offset=1,
                    fetched_at_ts=int(time.time()),
                )
                db.upsert_links([link])
                links = db.get_unassigned_links()

                gemini = MagicMock()
                gemini.generate_json.return_value = None

                result = cluster_link(link=links[0], fresh_events=[], gemini=gemini, db=db)
                self.assertIsNone(result)

                # Link should still be unassigned.
                remaining = db.get_unassigned_links()
                self.assertEqual(len(remaining), 1)


class TestClusterAllPending(unittest.TestCase):
    def test_cluster_all_pending(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "news_pool.sqlite3"
            with NewsPoolDB(path=Path(db_path)) as db:
                # Insert 3 links.
                for i in range(3):
                    link = PoolLink(
                        url=f"https://example.com/{i}",
                        norm_url=f"https://example.com/{i}",
                        domain="example.com",
                        title=f"Article {i}",
                        description=f"Description {i}",
                        age=None,
                        page_age="2026-02-07T10:00:00",
                        query="test",
                        offset=1,
                        fetched_at_ts=int(time.time()),
                    )
                    db.upsert_links([link])

                # Mock Gemini: first creates new event, others assigned to it.
                call_count = [0]

                def mock_generate_json(prompt: str) -> dict[str, Any] | None:
                    call_count[0] += 1
                    if call_count[0] == 1:
                        return {
                            "action": "new_event",
                            "summary_en": "Test event cluster",
                            "category": "AI",
                            "jurisdiction": "US",
                        }
                    # For subsequent calls, find the event id.
                    fresh = db.get_fresh_events()
                    if fresh:
                        return {"action": "assign", "event_id": fresh[0]["id"]}
                    return {"action": "new_event", "summary_en": "Another", "category": "AI", "jurisdiction": "US"}

                gemini = MagicMock()
                gemini.generate_json.side_effect = mock_generate_json

                stats = cluster_all_pending(db=db, gemini=gemini, delay_seconds=0)
                self.assertEqual(stats["processed"], 3)
                self.assertEqual(stats["errors"], 0)
                self.assertGreaterEqual(stats["new_events"], 1)

                # All links should be assigned.
                remaining = db.get_unassigned_links()
                self.assertEqual(len(remaining), 0)


class TestBuildMergePrompt(unittest.TestCase):
    def test_build_merge_prompt(self) -> None:
        events = [
            {"id": 42, "summary_en": "UK PM faces crisis over Mandelson links", "link_count": 3, "jurisdiction": "UK"},
            {"id": 55, "summary_en": "Peter Mandelson investigated by FCA", "link_count": 1, "jurisdiction": "UK"},
        ]
        prompt = build_merge_prompt(events, "UK Parliament / Politics")
        self.assertIn("42.", prompt)
        self.assertIn("55.", prompt)
        self.assertIn("UK PM faces crisis", prompt)
        self.assertIn("Peter Mandelson investigated", prompt)
        self.assertIn("UK Parliament / Politics", prompt)
        self.assertIn("deduplication", prompt)
        self.assertIn("groups", prompt)
        self.assertIn("plain integer", prompt)


class TestParseMergeResponse(unittest.TestCase):
    def test_parse_merge_response_valid(self) -> None:
        response = {"groups": [[42, 55], [78, 82]], "no_merge": [90, 91]}
        valid_ids = {42, 55, 78, 82, 90, 91}
        result = parse_merge_response(response, valid_ids)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], [42, 55])
        self.assertEqual(result[1], [78, 82])

    def test_parse_merge_response_single_item_filtered(self) -> None:
        response = {"groups": [[42], [78, 82]], "no_merge": [90]}
        valid_ids = {42, 78, 82, 90}
        result = parse_merge_response(response, valid_ids)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], [78, 82])

    def test_parse_merge_response_invalid_ids(self) -> None:
        response = {"groups": [[999, 888]], "no_merge": []}
        valid_ids = {42, 55}
        result = parse_merge_response(response, valid_ids)
        self.assertIsNone(result)

    def test_parse_merge_response_none_input(self) -> None:
        result = parse_merge_response(None, {42, 55})
        self.assertIsNone(result)

    def test_parse_merge_response_no_groups_key(self) -> None:
        result = parse_merge_response({"no_merge": [42]}, {42})
        self.assertIsNone(result)

    def test_parse_merge_response_duplicate_ids_across_groups(self) -> None:
        response = {"groups": [[42, 55], [55, 78]], "no_merge": []}
        valid_ids = {42, 55, 78}
        result = parse_merge_response(response, valid_ids)
        self.assertIsNotNone(result)
        assert result is not None
        # First group gets 42, 55. Second group: 55 already seen, only 78 left → <2 → filtered.
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], [42, 55])


class TestMergeEventsIntegration(unittest.TestCase):
    def test_merge_events_integration(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "news_pool.sqlite3"
            with NewsPoolDB(path=Path(db_path)) as db:
                # Create two events in same category that are duplicates.
                eid1 = db.create_event(
                    summary_en="UK PM faces crisis over Mandelson links",
                    category="UK Parliament / Politics", jurisdiction="UK",
                )
                eid2 = db.create_event(
                    summary_en="Peter Mandelson investigated by FCA",
                    category="UK Parliament / Politics", jurisdiction="UK",
                )
                # Create an unrelated event in a different category.
                eid3 = db.create_event(
                    summary_en="Tesla Q4 earnings beat", category="US Stocks", jurisdiction="US",
                )

                # Add links to eid1 (more links = winner).
                for i in range(3):
                    link = PoolLink(
                        url=f"https://bbc.co.uk/m{i}", norm_url=f"https://bbc.co.uk/m{i}",
                        domain="bbc.co.uk", title=f"Mandelson {i}", description="...",
                        age=None, page_age=None, query="test", offset=1,
                        fetched_at_ts=int(time.time()),
                    )
                    db.upsert_links([link])
                    links = db.get_unassigned_links()
                    db.assign_link_to_event(link_id=links[0]["id"], event_id=eid1)

                # Add 1 link to eid2.
                link = PoolLink(
                    url="https://telegraph.co.uk/m", norm_url="https://telegraph.co.uk/m",
                    domain="telegraph.co.uk", title="FCA probe", description="...",
                    age=None, page_age=None, query="test", offset=1,
                    fetched_at_ts=int(time.time()),
                )
                db.upsert_links([link])
                links = db.get_unassigned_links()
                db.assign_link_to_event(link_id=links[0]["id"], event_id=eid2)

                # Mock Gemini to group eid1 + eid2.
                gemini = MagicMock()
                gemini.generate.return_value = json.dumps({
                    "groups": [[eid1, eid2]],
                    "no_merge": [],
                })

                stats = merge_events(db=db, gemini=gemini, delay_seconds=0)

                self.assertEqual(stats["groups_merged"], 1)
                self.assertEqual(stats["events_removed"], 1)
                self.assertEqual(stats["links_moved"], 1)  # 1 link moved from eid2 to eid1.
                self.assertEqual(stats["errors"], 0)

                # Winner (eid1) should have 4 links now.
                winner = db.get_event(eid1)
                assert winner is not None
                self.assertEqual(winner["link_count"], 4)

                # Loser (eid2) should be deleted.
                loser = db.get_event(eid2)
                self.assertIsNone(loser)

                # Unrelated event should be untouched.
                other = db.get_event(eid3)
                self.assertIsNotNone(other)

    def test_merge_events_skips_small_categories(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "news_pool.sqlite3"
            with NewsPoolDB(path=Path(db_path)) as db:
                # Only 1 event in category — no merge needed.
                db.create_event(summary_en="Solo event", category="AI", jurisdiction="US")

                gemini = MagicMock()
                stats = merge_events(db=db, gemini=gemini, delay_seconds=0)

                self.assertEqual(stats["llm_calls"], 0)
                gemini.generate.assert_not_called()


class TestClusterEarlyExit(unittest.TestCase):
    """cluster_all_pending stops after N consecutive Gemini failures."""

    @staticmethod
    def _make_mock_db(num_links: int) -> MagicMock:
        db = MagicMock()
        db.get_unassigned_links.return_value = [
            {"id": i, "url": f"https://example.com/{i}", "title": f"Story {i}", "description": "desc"}
            for i in range(num_links)
        ]
        db.get_fresh_events.return_value = []
        return db

    def test_early_exit_after_consecutive_failures(self) -> None:
        db = self._make_mock_db(10)

        # Gemini always returns None (failure).
        gemini = MagicMock()
        gemini.generate_json.return_value = None

        stats = cluster_all_pending(
            db=db, gemini=gemini, delay_seconds=0, max_consecutive_failures=3,
        )

        # Should process exactly 3, skip remaining 7.
        self.assertEqual(stats["processed"], 3)
        self.assertEqual(stats["errors"], 3)
        self.assertEqual(stats["skipped"], 7)

    def test_no_early_exit_on_intermittent_failures(self) -> None:
        db = self._make_mock_db(6)

        # Alternate: fail, fail, succeed, fail, fail, succeed.
        call_count = 0
        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count % 3 == 0:
                return {"action": "new_event", "summary_en": "Test", "category": "Tech"}
            return None

        gemini = MagicMock()
        gemini.generate_json.side_effect = side_effect

        stats = cluster_all_pending(
            db=db, gemini=gemini, delay_seconds=0, max_consecutive_failures=3,
        )

        # All 6 processed, no skips (consecutive never hit 3).
        self.assertEqual(stats["processed"], 6)
        self.assertEqual(stats["skipped"], 0)


if __name__ == "__main__":
    unittest.main()
