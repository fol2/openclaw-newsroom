"""Tests for newsroom.event_manager — LLM clustering logic."""
import json
import re as _re
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from newsroom.event_manager import (
    CATEGORY_LIST,
    EventTokens,
    _find_cross_category_duplicates,
    _tokenize_event,
    _tokenize_link,
    build_clustering_prompt,
    build_focused_clustering_prompt,
    build_merge_prompt,
    cluster_all_pending,
    cluster_link,
    merge_events,
    parse_clustering_response,
    parse_merge_response,
    retrieve_candidates,
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
        response = {
            "action": "assign",
            "event_id": 42,
            "confidence": 0.9,
            "summary_en": "Test event",
            "category": "Global News",
            "jurisdiction": "GLOBAL",
            "link_flags": [],
            "match_basis": [],
        }
        events = [{"id": 42}, {"id": 43}]
        result = parse_clustering_response(response, {}, events)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result["validated"]["action"], "assign")
        self.assertEqual(result["validated"]["event_id"], 42)
        self.assertEqual(result["enforced"]["action"], "assign")
        self.assertEqual(result["enforced"]["event_id"], 42)

    def test_parse_assign_invalid_event_id(self) -> None:
        response = {"action": "assign", "event_id": 999, "confidence": 0.9, "summary_en": "Test"}
        events = [{"id": 42}]
        result = parse_clustering_response(response, {}, events)
        self.assertIsNone(result)

    def test_parse_development(self) -> None:
        response = {
            "action": "development",
            "parent_event_id": 42,
            "confidence": 0.9,
            "summary_en": "FCA investigates Mandelson",
            "development": "FCA investigation",
            "category": "UK Parliament / Politics",
            "jurisdiction": "UK",
            "link_flags": [],
            "match_basis": ["entity", "organisation"],
        }
        events = [{"id": 42}]
        result = parse_clustering_response(response, {}, events)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result["validated"]["action"], "development")
        self.assertEqual(result["validated"]["parent_event_id"], 42)
        self.assertEqual(result["validated"]["summary_en"], "FCA investigates Mandelson")
        self.assertEqual(result["enforced"]["action"], "development")

    def test_parse_new_event(self) -> None:
        response = {
            "action": "new_event",
            "confidence": 0.95,
            "summary_en": "Tesla Q4 earnings beat expectations",
            "category": "US Stocks",
            "jurisdiction": "US",
            "link_flags": [],
            "match_basis": ["entity", "number"],
        }
        result = parse_clustering_response(response, {}, [])
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result["validated"]["action"], "new_event")
        self.assertEqual(result["validated"]["summary_en"], "Tesla Q4 earnings beat expectations")
        self.assertEqual(result["enforced"]["action"], "new_event")

    def test_parse_normalises_category_aliases(self) -> None:
        response = {
            "action": "new_event",
            "confidence": 0.9,
            "summary_en": "Tech story",
            "category": "Technology",
            "jurisdiction": "US",
            "link_flags": [],
            "match_basis": [],
        }
        result = parse_clustering_response(response, {}, [])
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result["validated"]["category"], "AI")

    def test_parse_defaults_unknown_category(self) -> None:
        response = {
            "action": "new_event",
            "confidence": 0.9,
            "summary_en": "Some story",
            "category": "Completely Unknown",
            "jurisdiction": "GLOBAL",
            "link_flags": [],
            "match_basis": [],
        }
        result = parse_clustering_response(response, {}, [])
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result["validated"]["category"], "Global News")

    def test_parse_new_event_missing_summary(self) -> None:
        response = {"action": "new_event", "category": "AI"}
        result = parse_clustering_response(response, {}, [])
        self.assertIsNone(result)

    def test_parse_enforces_low_confidence_abstain(self) -> None:
        response = {
            "action": "assign",
            "event_id": 42,
            "confidence": 0.2,
            "summary_en": "Some story",
            "category": "Global News",
            "jurisdiction": "GLOBAL",
            "link_flags": [],
            "match_basis": [],
        }
        events = [{"id": 42}]
        result = parse_clustering_response(response, {"title": "Some story"}, events)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result["validated"]["action"], "assign")
        self.assertEqual(result["enforced"]["action"], "new_event")
        reasons = (result["enforced"].get("enforcement") or {}).get("reasons") if isinstance(result["enforced"].get("enforcement"), dict) else None
        self.assertIsInstance(reasons, list)
        assert isinstance(reasons, list)
        self.assertTrue(any(str(r).startswith("low_confidence:") for r in reasons))

    def test_parse_enforces_abstain_flags(self) -> None:
        response = {
            "action": "assign",
            "event_id": 42,
            "confidence": 0.95,
            "summary_en": "Some story",
            "category": "Global News",
            "jurisdiction": "GLOBAL",
            "link_flags": ["roundup"],
            "match_basis": ["other"],
        }
        events = [{"id": 42}]
        result = parse_clustering_response(response, {"title": "Some story"}, events)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result["validated"]["action"], "assign")
        self.assertEqual(result["enforced"]["action"], "new_event")
        reasons = (result["enforced"].get("enforcement") or {}).get("reasons") if isinstance(result["enforced"].get("enforcement"), dict) else None
        self.assertIsInstance(reasons, list)
        assert isinstance(reasons, list)
        self.assertIn("link_flag:roundup", reasons)

    def test_parse_unknown_action(self) -> None:
        response = {"action": "unknown"}
        result = parse_clustering_response(response, {}, [])
        self.assertIsNone(result)

    def test_normalises_category_to_canonical_list(self) -> None:
        cases: list[tuple[Any, str]] = [
            ("US News", "Global News"),
            ("World News", "Global News"),
            ("Technology", "AI"),
            ("Tech", "AI"),
            ("Technology/Tech", "AI"),
            ("UK Parliament / Politics", "UK Parliament / Politics"),
            ("UK Parliament/Politics", "UK Parliament / Politics"),
            ("global news", "Global News"),
            ("", "Global News"),
            (None, "Global News"),
            ("Not a real category", "Global News"),
        ]

        for raw, expected in cases:
            response = {
                "action": "new_event",
                "confidence": 0.9,
                "summary_en": "Test",
                "category": raw,
                "jurisdiction": "GLOBAL",
                "link_flags": [],
                "match_basis": [],
            }
            result = parse_clustering_response(response, {}, [])
            self.assertIsNotNone(result)
            assert result is not None
            self.assertEqual(result["validated"]["category"], expected)
            self.assertEqual(result["enforced"]["category"], expected)
            self.assertIn(expected, CATEGORY_LIST)


class TestClusterLink(unittest.TestCase):
    def test_cluster_link_assign(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "news_pool.sqlite3"
            with NewsPoolDB(path=Path(db_path)) as db:
                link = PoolLink(
                    url="https://bbc.co.uk/1", norm_url="https://bbc.co.uk/1",
                    domain="bbc.co.uk", title="PM crisis", description="...",
                    age=None, page_age="2026-02-07T10:00:00", query="test", offset=1,
                    fetched_at_ts=int(time.time()),
                )
                db.upsert_links([link])
                links = db.get_unassigned_links()
                self.assertEqual(len(links), 1)

                eid = db.create_event(summary_en="UK PM crisis", category="Politics", jurisdiction="UK")

                gemini = MagicMock()
                gemini.generate_json.return_value = {
                    "action": "assign",
                    "event_id": eid,
                    "confidence": 0.9,
                    "summary_en": "UK PM crisis",
                    "category": "Politics",
                    "jurisdiction": "UK",
                    "link_flags": [],
                    "match_basis": "Same incident; different source",
                }

                fresh = db.get_fresh_events()
                result = cluster_link(link=links[0], all_events=fresh, gemini=gemini, db=db)
                self.assertIsNotNone(result)
                assert result is not None
                self.assertEqual(result["action"], "assign")

                ev = db.get_event(eid)
                assert ev is not None
                self.assertEqual(ev["link_count"], 1)

    def test_cluster_link_gate_skips_roundup_links(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "news_pool.sqlite3"
            with NewsPoolDB(path=Path(db_path)) as db:
                link = PoolLink(
                    url="https://example.com/roundup-1", norm_url="https://example.com/roundup-1",
                    domain="example.com", title="Morning briefing: PM crisis", description="Latest updates...",
                    age=None, page_age="2026-02-07T10:00:00", query="test", offset=1,
                    fetched_at_ts=int(time.time()),
                )
                db.upsert_links([link])
                links = db.get_unassigned_links()
                self.assertEqual(len(links), 1)
                link_id = int(links[0]["id"])

                gemini = MagicMock()
                gemini.generate_json.return_value = {"action": "new_event", "summary_en": "Should not be called"}

                result = cluster_link(link=links[0], all_events=[], gemini=gemini, db=db)
                self.assertIsNotNone(result)
                assert result is not None
                self.assertEqual(result["action"], "skip")
                self.assertIn("skip_reason", result)

                # Gate should prevent any LLM call.
                gemini.generate_json.assert_not_called()

                row = db._conn.execute(
                    "SELECT event_id, skip_cluster_reason, skip_clustered_at_ts FROM links WHERE id = ?",
                    (link_id,),
                ).fetchone()
                self.assertIsNotNone(row)
                assert row is not None
                self.assertIsNone(row["event_id"])
                self.assertIsInstance(row["skip_cluster_reason"], str)
                self.assertIsNotNone(row["skip_clustered_at_ts"])

                # Skipped links should not be returned as unassigned links any more.
                self.assertEqual(db.get_unassigned_links(), [])

    def test_cluster_link_logs_decision_assign(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "news_pool.sqlite3"
            with NewsPoolDB(path=Path(db_path)) as db:
                link = PoolLink(
                    url="https://bbc.co.uk/log-1", norm_url="https://bbc.co.uk/log-1",
                    domain="bbc.co.uk", title="PM crisis", description="...",
                    age=None, page_age="2026-02-07T10:00:00", query="test", offset=1,
                    fetched_at_ts=int(time.time()),
                )
                db.upsert_links([link])
                links = db.get_unassigned_links()
                self.assertEqual(len(links), 1)
                link_id = int(links[0]["id"])

                eid = db.create_event(summary_en="UK PM crisis", category="Politics", jurisdiction="UK")

                gemini = MagicMock()
                gemini.generate_json.return_value = {
                    "action": "assign",
                    "event_id": eid,
                    "confidence": 0.9,
                    "summary_en": "UK PM crisis",
                    "category": "Politics",
                    "jurisdiction": "UK",
                    "link_flags": [],
                    "match_basis": "Same incident; different source",
                }
                gemini.last_model_name = "gemini-test-model"

                fresh = db.get_fresh_events()
                result = cluster_link(link=links[0], all_events=fresh, gemini=gemini, db=db)
                self.assertIsNotNone(result)

                row = db._conn.execute(
                    "SELECT * FROM clustering_decisions WHERE link_id = ? ORDER BY id DESC LIMIT 1",
                    (link_id,),
                ).fetchone()
                self.assertIsNotNone(row)
                assert row is not None
                self.assertEqual(int(row["link_id"]), link_id)
                self.assertEqual(row["validated_action"], "assign")
                self.assertEqual(row["enforced_action"], "assign")
                self.assertEqual(row["model_name"], "gemini-test-model")
                self.assertIsInstance(row["prompt_sha256"], str)
                self.assertEqual(len(str(row["prompt_sha256"])), 64)
                self.assertIsInstance(row["llm_response_json"], str)

                decision_id = int(row["id"])
                cand = db._conn.execute(
                    "SELECT rank, event_id, score FROM clustering_decision_candidates WHERE decision_id = ? ORDER BY rank ASC",
                    (decision_id,),
                ).fetchall()
                self.assertGreaterEqual(len(cand), 1)
                self.assertEqual(int(cand[0]["event_id"]), eid)
                self.assertGreater(float(cand[0]["score"]), 0.0)

    def test_cluster_link_enforces_new_event_on_low_confidence_assign(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "news_pool.sqlite3"
            with NewsPoolDB(path=Path(db_path)) as db:
                link = PoolLink(
                    url="https://bbc.co.uk/lowconf-1", norm_url="https://bbc.co.uk/lowconf-1",
                    domain="bbc.co.uk", title="PM crisis", description="...",
                    age=None, page_age="2026-02-07T10:00:00", query="test", offset=1,
                    fetched_at_ts=int(time.time()),
                )
                db.upsert_links([link])
                links = db.get_unassigned_links()
                self.assertEqual(len(links), 1)
                link_id = int(links[0]["id"])

                eid = db.create_event(summary_en="UK PM crisis", category="Politics", jurisdiction="UK")

                gemini = MagicMock()
                gemini.generate_json.return_value = {
                    "action": "assign",
                    "event_id": eid,
                    "confidence": 0.69,
                    "summary_en": "UK PM crisis",
                    "category": "Politics",
                    "jurisdiction": "UK",
                    "link_flags": [],
                    "match_basis": "Weak match",
                }

                fresh = db.get_fresh_events()
                result = cluster_link(link=links[0], all_events=fresh, gemini=gemini, db=db)
                self.assertIsNotNone(result)
                assert result is not None
                self.assertEqual(result["action"], "new_event")
                self.assertIn("event_id", result)
                self.assertIn("enforcement", result)

                enforced_event_id = int(result["event_id"])
                self.assertNotEqual(enforced_event_id, eid)

                original = db.get_event(eid)
                assert original is not None
                self.assertEqual(int(original["link_count"]), 0)

                enforced_ev = db.get_event(enforced_event_id)
                assert enforced_ev is not None
                self.assertEqual(int(enforced_ev["link_count"]), 1)

                row = db._conn.execute(
                    "SELECT * FROM clustering_decisions WHERE link_id = ? ORDER BY id DESC LIMIT 1",
                    (link_id,),
                ).fetchone()
                self.assertIsNotNone(row)
                assert row is not None
                self.assertEqual(row["validated_action"], "assign")
                self.assertEqual(row["enforced_action"], "new_event")

                enforced_obj = json.loads(str(row["enforced_action_json"]))
                self.assertIn("enforcement", enforced_obj)
                reasons = enforced_obj["enforcement"].get("reasons") if isinstance(enforced_obj["enforcement"], dict) else None
                self.assertIsInstance(reasons, list)
                assert isinstance(reasons, list)
                self.assertTrue(any(str(r).startswith("low_confidence:") for r in reasons))

    def test_cluster_link_enforces_new_event_on_roundup_flag(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "news_pool.sqlite3"
            with NewsPoolDB(path=Path(db_path)) as db:
                link = PoolLink(
                    url="https://bbc.co.uk/roundup-1", norm_url="https://bbc.co.uk/roundup-1",
                    # Avoid triggering the deterministic gate; we want to exercise LLM-provided link_flags enforcement.
                    domain="bbc.co.uk", title="PM crisis continues", description="...",
                    age=None, page_age="2026-02-07T10:00:00", query="test", offset=1,
                    fetched_at_ts=int(time.time()),
                )
                db.upsert_links([link])
                links = db.get_unassigned_links()
                self.assertEqual(len(links), 1)
                link_id = int(links[0]["id"])

                eid = db.create_event(summary_en="UK PM crisis", category="Politics", jurisdiction="UK")

                gemini = MagicMock()
                gemini.generate_json.return_value = {
                    "action": "assign",
                    "event_id": eid,
                    "confidence": 0.95,
                    "summary_en": "Morning briefing includes PM crisis and other topics",
                    "category": "Politics",
                    "jurisdiction": "UK",
                    "link_flags": ["roundup"],
                    "match_basis": "Roundup / briefing",
                }

                fresh = db.get_fresh_events()
                result = cluster_link(link=links[0], all_events=fresh, gemini=gemini, db=db)
                self.assertIsNotNone(result)
                assert result is not None
                self.assertEqual(result["action"], "new_event")
                self.assertIn("event_id", result)
                self.assertIn("enforcement", result)

                row = db._conn.execute(
                    "SELECT * FROM clustering_decisions WHERE link_id = ? ORDER BY id DESC LIMIT 1",
                    (link_id,),
                ).fetchone()
                self.assertIsNotNone(row)
                assert row is not None
                self.assertEqual(row["validated_action"], "assign")
                self.assertEqual(row["enforced_action"], "new_event")

                enforced_obj = json.loads(str(row["enforced_action_json"]))
                self.assertIn("enforcement", enforced_obj)
                reasons = enforced_obj["enforcement"].get("reasons") if isinstance(enforced_obj["enforcement"], dict) else None
                self.assertIsInstance(reasons, list)
                assert isinstance(reasons, list)
                self.assertIn("link_flag:roundup", reasons)

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
                    "confidence": 0.9,
                    "summary_en": "Scientists discover new species",
                    "category": "Global News",
                    "jurisdiction": "GLOBAL",
                    "link_flags": [],
                    "match_basis": None,
                }

                result = cluster_link(link=links[0], all_events=[], gemini=gemini, db=db)
                self.assertIsNotNone(result)
                assert result is not None
                self.assertEqual(result["action"], "new_event")
                self.assertIn("event_id", result)

                ev = db.get_event(result["event_id"])
                self.assertIsNotNone(ev)
                assert ev is not None
                self.assertEqual(ev["summary_en"], "Scientists discover new species")

    def test_cluster_link_normalises_category_before_db_write(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "news_pool.sqlite3"
            with NewsPoolDB(path=Path(db_path)) as db:
                link = PoolLink(
                    url="https://bbc.co.uk/alias-1",
                    norm_url="https://bbc.co.uk/alias-1",
                    domain="bbc.co.uk",
                    title="New AI model launched",
                    description="A new model is released.",
                    age=None,
                    page_age="2026-02-07T12:00:00",
                    query="test",
                    offset=1,
                    fetched_at_ts=int(time.time()),
                )
                db.upsert_links([link])
                links = db.get_unassigned_links()

                gemini = MagicMock()
                gemini.generate_json.return_value = {
                    "action": "new_event",
                    "confidence": 0.9,
                    "summary_en": "A new AI model is launched",
                    "category": "Tech",
                    "jurisdiction": "GLOBAL",
                    "link_flags": [],
                    "match_basis": [],
                }

                result = cluster_link(link=links[0], all_events=[], gemini=gemini, db=db)
                self.assertIsNotNone(result)
                assert result is not None
                self.assertEqual(result["action"], "new_event")

                ev = db.get_event(result["event_id"])
                self.assertIsNotNone(ev)
                assert ev is not None
                self.assertEqual(ev["category"], "AI")

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

                result = cluster_link(link=links[0], all_events=[], gemini=gemini, db=db)
                self.assertIsNone(result)

                remaining = db.get_unassigned_links()
                self.assertEqual(len(remaining), 1)

    def test_cluster_link_skip_cluster_gate_does_not_call_gemini_and_records_reason(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "news_pool.sqlite3"
            with NewsPoolDB(path=Path(db_path)) as db:
                link = PoolLink(
                    url="https://example.com/live-1",
                    norm_url="https://example.com/live-1",
                    domain="example.com",
                    title="Live updates: PM crisis",
                    description="Rolling coverage of multiple developments",
                    age=None,
                    page_age="2026-02-07T10:00:00",
                    query="test",
                    offset=1,
                    fetched_at_ts=int(time.time()),
                )
                db.upsert_links([link])
                links = db.get_unassigned_links()
                self.assertEqual(len(links), 1)
                link_id = int(links[0]["id"])

                gemini = MagicMock()

                result = cluster_link(link=links[0], all_events=[], gemini=gemini, db=db)
                self.assertIsNotNone(result)
                assert result is not None
                self.assertEqual(result["action"], "skip")
                self.assertEqual(result["skip_reason"], "live_updates")

                gemini.generate_json.assert_not_called()

                row = db._conn.execute(
                    "SELECT skip_cluster_reason, skip_clustered_at_ts FROM links WHERE id = ?",
                    (link_id,),
                ).fetchone()
                self.assertIsNotNone(row)
                assert row is not None
                self.assertEqual(row["skip_cluster_reason"], "live_updates")
                self.assertIsNotNone(row["skip_clustered_at_ts"])

    def test_cluster_link_logs_decision_empty_response(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "news_pool.sqlite3"
            with NewsPoolDB(path=Path(db_path)) as db:
                link = PoolLink(
                    url="https://bbc.co.uk/log-2", norm_url="https://bbc.co.uk/log-2",
                    domain="bbc.co.uk", title="Some news", description="...",
                    age=None, page_age=None, query="test", offset=1,
                    fetched_at_ts=int(time.time()),
                )
                db.upsert_links([link])
                links = db.get_unassigned_links()
                self.assertEqual(len(links), 1)
                link_id = int(links[0]["id"])

                gemini = MagicMock()
                gemini.generate_json.return_value = None

                result = cluster_link(link=links[0], all_events=[], gemini=gemini, db=db)
                self.assertIsNone(result)

                row = db._conn.execute(
                    "SELECT * FROM clustering_decisions WHERE link_id = ? ORDER BY id DESC LIMIT 1",
                    (link_id,),
                ).fetchone()
                self.assertIsNotNone(row)
                assert row is not None
                self.assertIsNone(row["validated_action"])
                self.assertIsNone(row["enforced_action"])
                cand = db._conn.execute(
                    "SELECT COUNT(1) AS n FROM clustering_decision_candidates WHERE decision_id = ?",
                    (int(row["id"]),),
                ).fetchone()
                assert cand is not None
                self.assertEqual(int(cand["n"]), 0)

    def test_cluster_link_backward_compat_fresh_events(self) -> None:
        """fresh_events param still works as backward compat alias."""
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "news_pool.sqlite3"
            with NewsPoolDB(path=Path(db_path)) as db:
                link = PoolLink(
                    url="https://bbc.co.uk/4", norm_url="https://bbc.co.uk/4",
                    domain="bbc.co.uk", title="Test", description="...",
                    age=None, page_age=None, query="test", offset=1,
                    fetched_at_ts=int(time.time()),
                )
                db.upsert_links([link])
                links = db.get_unassigned_links()

                gemini = MagicMock()
                gemini.generate_json.return_value = {
                    "action": "new_event",
                    "confidence": 0.9,
                    "summary_en": "Test event",
                    "category": "AI",
                    "jurisdiction": "US",
                    "link_flags": [],
                    "match_basis": None,
                }

                # Use fresh_events (old param name).
                result = cluster_link(link=links[0], fresh_events=[], gemini=gemini, db=db)
                self.assertIsNotNone(result)

    def test_cluster_link_assign_to_posted_event(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "news_pool.sqlite3"
            with NewsPoolDB(path=Path(db_path)) as db:
                link = PoolLink(
                    url="https://bbc.co.uk/5", norm_url="https://bbc.co.uk/5",
                    domain="bbc.co.uk", title="Jimmy Lai sentenced to 20 years",
                    description="Former media tycoon Jimmy Lai sentenced",
                    age=None, page_age="2026-02-07T10:00:00", query="test", offset=1,
                    fetched_at_ts=int(time.time()),
                )
                db.upsert_links([link])
                links = db.get_unassigned_links()

                eid = db.create_event(
                    summary_en="Jimmy Lai sentenced to 20 years in prison",
                    category="Hong Kong News", jurisdiction="HK",
                )
                db.mark_event_posted(eid, thread_id="t1", run_id="r1")

                gemini = MagicMock()
                gemini.generate_json.return_value = {
                    "action": "assign",
                    "event_id": eid,
                    "confidence": 0.9,
                    "summary_en": "Jimmy Lai sentenced to 20 years in prison",
                    "category": "Hong Kong News",
                    "jurisdiction": "HK",
                    "link_flags": [],
                    "match_basis": "Same sentencing story",
                }

                events = db.get_all_fresh_events()
                result = cluster_link(link=links[0], all_events=events, gemini=gemini, db=db)
                self.assertIsNotNone(result)
                assert result is not None
                self.assertEqual(result["action"], "assign")
                self.assertTrue(result.get("assigned_to_posted"))


class TestClusterAllPending(unittest.TestCase):
    def test_cluster_all_pending(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "news_pool.sqlite3"
            with NewsPoolDB(path=Path(db_path)) as db:
                for i in range(3):
                    link = PoolLink(
                        url=f"https://example.com/{i}",
                        norm_url=f"https://example.com/{i}",
                        domain="example.com",
                        title=f"Jimmy Lai sentenced article {i}",
                        description=f"Jimmy Lai sentenced to 20 years {i}",
                        age=None,
                        page_age="2026-02-07T10:00:00",
                        query="test",
                        offset=1,
                        fetched_at_ts=int(time.time()),
                    )
                    db.upsert_links([link])

                def mock_generate_json(prompt: str) -> dict[str, Any] | None:
                    # Prompt-aware: if candidates shown in prompt, assign to first.
                    ids_in_prompt = [int(m) for m in _re.findall(r'\[id=(\d+)\]', prompt)]
                    if ids_in_prompt:
                        return {
                            "action": "assign",
                            "event_id": ids_in_prompt[0],
                            "confidence": 0.92,
                            "summary_en": "Jimmy Lai sentenced to 20 years",
                            "category": "Hong Kong News",
                            "jurisdiction": "HK",
                            "link_flags": [],
                            "match_basis": "Same sentencing story",
                        }
                    return {
                        "action": "new_event",
                        "confidence": 0.9,
                        "summary_en": "Jimmy Lai sentenced to 20 years",
                        "category": "Hong Kong News",
                        "jurisdiction": "HK",
                        "link_flags": [],
                        "match_basis": None,
                    }

                gemini = MagicMock()
                gemini.generate_json.side_effect = mock_generate_json

                stats = cluster_all_pending(db=db, gemini=gemini, delay_seconds=0)
                self.assertEqual(stats["processed"], 3)
                self.assertEqual(stats["errors"], 0)
                self.assertGreaterEqual(stats["new_events"], 1)

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
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], [42, 55])


class TestMergeEventsIntegration(unittest.TestCase):
    def test_merge_events_integration(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "news_pool.sqlite3"
            with NewsPoolDB(path=Path(db_path)) as db:
                eid1 = db.create_event(
                    summary_en="UK PM faces crisis over Mandelson links",
                    category="UK Parliament / Politics", jurisdiction="UK",
                )
                eid2 = db.create_event(
                    summary_en="Peter Mandelson investigated by FCA",
                    category="UK Parliament / Politics", jurisdiction="UK",
                )
                eid3 = db.create_event(
                    summary_en="Tesla Q4 earnings beat", category="US Stocks", jurisdiction="US",
                )

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

                link = PoolLink(
                    url="https://telegraph.co.uk/m", norm_url="https://telegraph.co.uk/m",
                    domain="telegraph.co.uk", title="FCA probe", description="...",
                    age=None, page_age=None, query="test", offset=1,
                    fetched_at_ts=int(time.time()),
                )
                db.upsert_links([link])
                links = db.get_unassigned_links()
                db.assign_link_to_event(link_id=links[0]["id"], event_id=eid2)

                gemini = MagicMock()
                gemini.generate.return_value = json.dumps({
                    "groups": [[eid1, eid2]],
                    "no_merge": [],
                })

                stats = merge_events(db=db, gemini=gemini, delay_seconds=0)

                self.assertEqual(stats["groups_merged"], 1)
                self.assertEqual(stats["events_removed"], 1)
                self.assertEqual(stats["links_moved"], 1)
                self.assertEqual(stats["errors"], 0)

                winner = db.get_event(eid1)
                assert winner is not None
                self.assertEqual(winner["link_count"], 4)

                loser = db.get_event(eid2)
                self.assertIsNone(loser)

                other = db.get_event(eid3)
                self.assertIsNotNone(other)

    def test_merge_events_skips_small_categories(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "news_pool.sqlite3"
            with NewsPoolDB(path=Path(db_path)) as db:
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
        db.get_all_fresh_events.return_value = []
        db.get_fresh_events.return_value = []
        return db

    def test_early_exit_after_consecutive_failures(self) -> None:
        db = self._make_mock_db(10)

        gemini = MagicMock()
        gemini.generate_json.return_value = None

        stats = cluster_all_pending(
            db=db, gemini=gemini, delay_seconds=0, max_consecutive_failures=3,
        )

        self.assertEqual(stats["processed"], 3)
        self.assertEqual(stats["errors"], 3)
        self.assertEqual(stats["skipped"], 7)

    def test_no_early_exit_on_intermittent_failures(self) -> None:
        db = self._make_mock_db(6)

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

        self.assertEqual(stats["processed"], 6)
        self.assertEqual(stats["skipped"], 0)


# ---------------------------------------------------------------------------
# New tests for retrieve-then-decide
# ---------------------------------------------------------------------------

class TestTokenization(unittest.TestCase):
    def test_tokenize_event(self) -> None:
        ev = {"summary_en": "Jimmy Lai sentenced to 20 years in prison", "title": "Lai trial"}
        tok = _tokenize_event(ev)
        self.assertIsInstance(tok, EventTokens)
        self.assertTrue(len(tok.key_tokens) > 0)

    def test_tokenize_link(self) -> None:
        link = {"title": "Jimmy Lai sentenced to 20 years", "description": "Former media tycoon"}
        tok = _tokenize_link(link)
        self.assertIsInstance(tok, EventTokens)
        self.assertTrue(len(tok.key_tokens) > 0)

    def test_tokenize_empty(self) -> None:
        tok = _tokenize_event({"summary_en": "", "title": ""})
        self.assertEqual(len(tok.key_tokens), 0)


class TestRetrieveCandidates(unittest.TestCase):
    def _make_events(self) -> list[dict[str, Any]]:
        return [
            {"id": 1, "summary_en": "Jimmy Lai sentenced to 20 years in prison", "title": "Lai trial",
             "category": "Hong Kong News", "status": "active"},
            {"id": 2, "summary_en": "Leicester City handed six-point deduction", "title": "Leicester",
             "category": "Sports", "status": "active"},
            {"id": 3, "summary_en": "Tesla Q4 earnings beat expectations", "title": "Tesla Q4",
             "category": "US Stocks", "status": "posted"},
        ]

    def test_finds_matching_event(self) -> None:
        events = self._make_events()
        link = {"title": "Jimmy Lai sentenced to 20 years by Hong Kong court", "description": "Media tycoon"}
        candidates = retrieve_candidates(link, events, min_score=0.05)
        self.assertTrue(len(candidates) > 0)
        self.assertEqual(candidates[0][0]["id"], 1)

    def test_cross_category_matching(self) -> None:
        events = [
            {"id": 1, "summary_en": "Jimmy Lai sentenced to 20 years", "title": "Lai",
             "category": "Hong Kong News", "status": "active"},
            {"id": 2, "summary_en": "Jimmy Lai jailed for 20 years under national security law", "title": "Lai",
             "category": "Global News", "status": "active"},
        ]
        link = {"title": "Jimmy Lai gets 20-year sentence", "description": "Lai sentenced"}
        candidates = retrieve_candidates(link, events, min_score=0.05)
        # Retrieval should not depend on the category label.
        found_ids = {ev["id"] for ev, _ in candidates}
        self.assertIn(1, found_ids)
        candidates2 = retrieve_candidates(link, [events[1]], min_score=0.05)
        found_ids2 = {ev["id"] for ev, _ in candidates2}
        self.assertIn(2, found_ids2)

    def test_includes_posted_events(self) -> None:
        events = self._make_events()
        link = {"title": "Tesla Q4 earnings surprise Wall Street", "description": "Tesla revenue up"}
        candidates = retrieve_candidates(link, events, min_score=0.05)
        found_ids = {ev["id"] for ev, _ in candidates}
        self.assertIn(3, found_ids)  # Posted event should be found.

    def test_respects_top_k(self) -> None:
        events = self._make_events()
        link = {"title": "Jimmy Lai sentenced", "description": "Lai trial"}
        candidates = retrieve_candidates(link, events, top_k=1, min_score=0.01)
        self.assertLessEqual(len(candidates), 1)

    def test_respects_min_score(self) -> None:
        events = self._make_events()
        link = {"title": "Jimmy Lai sentenced to 20 years", "description": "Lai"}
        candidates = retrieve_candidates(link, events, min_score=0.99)
        # Very high min_score should filter almost everything.
        self.assertEqual(len(candidates), 0)

    def test_novel_story_no_candidates(self) -> None:
        events = self._make_events()
        link = {"title": "Volcanic eruption in Iceland", "description": "Massive eruption near Reykjavik"}
        candidates = retrieve_candidates(link, events, min_score=0.10)
        self.assertEqual(len(candidates), 0)

    def test_token_cache(self) -> None:
        events = self._make_events()
        cache: dict[int, EventTokens] = {}
        link = {"title": "Jimmy Lai sentenced", "description": ""}
        retrieve_candidates(link, events, token_cache=cache)
        self.assertTrue(len(cache) > 0)
        # Second call reuses cache.
        retrieve_candidates(link, events, token_cache=cache)

    def test_empty_events(self) -> None:
        link = {"title": "Test", "description": ""}
        candidates = retrieve_candidates(link, [])
        self.assertEqual(len(candidates), 0)


class TestBuildFocusedClusteringPrompt(unittest.TestCase):
    def test_no_candidates_prompt(self) -> None:
        link = {"title": "New volcano erupts", "description": ""}
        prompt = build_focused_clustering_prompt(link, [])
        self.assertIn("NEW EVENT", prompt)
        self.assertIn("new_event", prompt)
        self.assertNotIn("ASSIGN", prompt)

    def test_with_candidates_shows_scores(self) -> None:
        ev = {"id": 42, "summary_en": "Volcano erupted", "category": "Global News",
              "jurisdiction": "GLOBAL", "status": "active", "link_count": 3}
        candidates = [(ev, 0.75)]
        prompt = build_focused_clustering_prompt({"title": "Volcano erupts again"}, candidates)
        self.assertIn("[id=42]", prompt)
        self.assertIn("score=0.75", prompt)
        self.assertIn("ASSIGN", prompt)

    def test_assign_first_guidance(self) -> None:
        ev = {"id": 1, "summary_en": "Test event", "category": "AI",
              "jurisdiction": "US", "status": "active", "link_count": 1}
        candidates = [(ev, 0.5)]
        prompt = build_focused_clustering_prompt({"title": "Test"}, candidates)
        self.assertIn("ASSIGN is likely correct", prompt)

    def test_development_criteria(self) -> None:
        ev = {"id": 1, "summary_en": "Test event", "category": "AI",
              "jurisdiction": "US", "status": "active", "link_count": 1}
        candidates = [(ev, 0.5)]
        prompt = build_focused_clustering_prompt({"title": "Test"}, candidates)
        self.assertIn("DEVELOPMENT", prompt)
        self.assertIn("new phase", prompt)


class TestMergeIncludesPosted(unittest.TestCase):
    def test_posted_status_transfer(self) -> None:
        """If loser is posted and winner is not, posted status should transfer."""
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "news_pool.sqlite3"
            with NewsPoolDB(path=Path(db_path)) as db:
                eid1 = db.create_event(
                    summary_en="Jimmy Lai sentenced (version 1)",
                    category="Hong Kong News", jurisdiction="HK",
                )
                eid2 = db.create_event(
                    summary_en="Jimmy Lai sentenced (version 2)",
                    category="Hong Kong News", jurisdiction="HK",
                )
                # Mark eid2 as posted.
                db.mark_event_posted(eid2, thread_id="t123", run_id="r123")

                # Add more links to eid1 so it wins.
                for i in range(3):
                    link = PoolLink(
                        url=f"https://example.com/p{i}", norm_url=f"https://example.com/p{i}",
                        domain="example.com", title=f"Lai {i}", description="...",
                        age=None, page_age=None, query="test", offset=1,
                        fetched_at_ts=int(time.time()),
                    )
                    db.upsert_links([link])
                    links = db.get_unassigned_links()
                    db.assign_link_to_event(link_id=links[0]["id"], event_id=eid1)

                gemini = MagicMock()
                gemini.generate.return_value = json.dumps({
                    "groups": [[eid1, eid2]], "no_merge": [],
                })

                stats = merge_events(db=db, gemini=gemini, delay_seconds=0)
                self.assertGreaterEqual(stats["groups_merged"], 1)

                winner = db.get_event(eid1)
                assert winner is not None
                self.assertEqual(winner["status"], "posted")

    def test_cross_category_merge(self) -> None:
        """Events in different categories with shared anchors should be detected."""
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "news_pool.sqlite3"
            with NewsPoolDB(path=Path(db_path)) as db:
                eid1 = db.create_event(
                    summary_en="Jimmy Lai sentenced to 20 years in prison under national security law",
                    category="Hong Kong News", jurisdiction="HK",
                )
                eid2 = db.create_event(
                    summary_en="Jimmy Lai sentenced to 20 years for sedition under Hong Kong national security law",
                    category="Global News", jurisdiction="HK",
                )

                gemini = MagicMock()
                gemini.generate.return_value = json.dumps({
                    "groups": [[eid1, eid2]], "no_merge": [],
                })

                stats = merge_events(db=db, gemini=gemini, delay_seconds=0)
                self.assertGreaterEqual(stats["cross_category_pairs"], 1)


class TestDedupeMarkerRootAncestor(unittest.TestCase):
    def test_child_event_key_matches_parent(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "news_pool.sqlite3"
            with NewsPoolDB(path=Path(db_path)) as db:
                parent_id = db.create_event(summary_en="Parent event", category="AI", jurisdiction="US")
                child_id = db.create_event(
                    summary_en="Child development", category="AI", jurisdiction="US",
                    parent_event_id=parent_id,
                )

                root = db.get_root_event_id(child_id)
                self.assertEqual(root, parent_id)

    def test_deeply_nested_resolves_to_root(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "news_pool.sqlite3"
            with NewsPoolDB(path=Path(db_path)) as db:
                root_id = db.create_event(summary_en="Root", category="AI", jurisdiction="US")
                mid_id = db.create_event(summary_en="Mid", category="AI", jurisdiction="US", parent_event_id=root_id)
                leaf_id = db.create_event(summary_en="Leaf", category="AI", jurisdiction="US", parent_event_id=mid_id)

                self.assertEqual(db.get_root_event_id(leaf_id), root_id)
                self.assertEqual(db.get_root_event_id(mid_id), root_id)

    def test_root_returns_itself(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "news_pool.sqlite3"
            with NewsPoolDB(path=Path(db_path)) as db:
                root_id = db.create_event(summary_en="Root event", category="AI", jurisdiction="US")
                self.assertEqual(db.get_root_event_id(root_id), root_id)


class TestFindCrossCategoryDuplicates(unittest.TestCase):
    def test_finds_cross_category_pairs(self) -> None:
        events = [
            {"id": 1, "summary_en": "Jimmy Lai sentenced to 20 years in prison",
             "title": "Lai trial", "category": "Hong Kong News"},
            {"id": 2, "summary_en": "Jimmy Lai sentenced to 20 years under national security law",
             "title": "Lai jailed", "category": "Global News"},
        ]
        pairs = _find_cross_category_duplicates(events, min_anchor_overlap=2)
        self.assertTrue(len(pairs) > 0)
        self.assertIn((1, 2), pairs)

    def test_same_category_excluded(self) -> None:
        events = [
            {"id": 1, "summary_en": "Jimmy Lai sentenced", "title": "Lai", "category": "HK News"},
            {"id": 2, "summary_en": "Jimmy Lai jailed", "title": "Lai", "category": "HK News"},
        ]
        pairs = _find_cross_category_duplicates(events)
        self.assertEqual(len(pairs), 0)

    def test_unrelated_events_excluded(self) -> None:
        events = [
            {"id": 1, "summary_en": "Jimmy Lai sentenced", "title": "Lai", "category": "HK News"},
            {"id": 2, "summary_en": "Tesla Q4 earnings beat", "title": "Tesla", "category": "US Stocks"},
        ]
        pairs = _find_cross_category_duplicates(events, min_anchor_overlap=2)
        self.assertEqual(len(pairs), 0)


class TestGetAllFreshEvents(unittest.TestCase):
    def test_includes_posted(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "news_pool.sqlite3"
            with NewsPoolDB(path=Path(db_path)) as db:
                eid = db.create_event(summary_en="Posted event", category="AI", jurisdiction="US")
                db.mark_event_posted(eid, thread_id="t1", run_id="r1")

                events = db.get_all_fresh_events()
                self.assertTrue(any(e["id"] == eid for e in events))

    def test_excludes_expired_unposted(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "news_pool.sqlite3"
            with NewsPoolDB(path=Path(db_path)) as db:
                eid = db.create_event(summary_en="Test event", category="AI", jurisdiction="US")
                # Expire the event.
                db._conn.execute(
                    "UPDATE events SET expires_at_ts = 1 WHERE id = ?", (eid,)
                )

                # get_fresh_events should NOT include it.
                fresh = db.get_fresh_events()
                self.assertFalse(any(e["id"] == eid for e in fresh))

                # get_all_fresh_events should also exclude expired, unposted events.
                all_events = db.get_all_fresh_events()
                self.assertFalse(any(e["id"] == eid for e in all_events))

    def test_excludes_old_posted(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "news_pool.sqlite3"
            with NewsPoolDB(path=Path(db_path)) as db:
                now = int(time.time())
                eid = db.create_event(summary_en="Old posted event", category="AI", jurisdiction="US")
                db.mark_event_posted(eid, thread_id="t1", run_id="r1")

                # Make it older than the default posted_recent_hours window (48h).
                db._conn.execute(
                    "UPDATE events SET posted_at_ts = ? WHERE id = ?",
                    (now - (49 * 3600), eid),
                )

                events = db.get_all_fresh_events(now_ts=now)
                self.assertFalse(any(e["id"] == eid for e in events))


if __name__ == "__main__":
    unittest.main()
