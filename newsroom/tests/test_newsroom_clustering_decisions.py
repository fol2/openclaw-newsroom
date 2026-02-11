import io
import json
import sqlite3
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import scripts.newsroom_clustering_decisions as cli


def _create_inspector_schema(conn: sqlite3.Connection, *, include_candidates: bool = True) -> None:
    conn.execute(
        """
        CREATE TABLE links (
          id INTEGER PRIMARY KEY,
          norm_url TEXT,
          title TEXT,
          domain TEXT,
          event_id INTEGER
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE events (
          id INTEGER PRIMARY KEY,
          category TEXT,
          jurisdiction TEXT,
          title TEXT,
          summary_en TEXT,
          status TEXT
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE clustering_decisions (
          id INTEGER PRIMARY KEY,
          link_id INTEGER NOT NULL,
          prompt_sha256 TEXT NOT NULL,
          model_name TEXT,
          llm_started_at_ts INTEGER,
          llm_finished_at_ts INTEGER,
          llm_response_json TEXT,
          validated_action TEXT,
          validated_action_json TEXT,
          enforced_action TEXT,
          enforced_action_json TEXT,
          error_type TEXT,
          error_message TEXT,
          created_at_ts INTEGER NOT NULL
        );
        """
    )

    if include_candidates:
        conn.execute(
            """
            CREATE TABLE clustering_decision_candidates (
              decision_id INTEGER NOT NULL,
              rank INTEGER NOT NULL,
              event_id INTEGER NOT NULL,
              score REAL NOT NULL
            );
            """
        )


def _seed_sample_data(conn: sqlite3.Connection, *, now_ts: int) -> None:
    conn.executemany(
        "INSERT INTO events(id, category, jurisdiction, title, summary_en, status) VALUES(?, ?, ?, ?, ?, ?)",
        [
            (100, "Politics", "UK", "Budget vote passed", "Parliament approved the vote", "active"),
            (101, "Politics", "UK", "Opposition reaction", "Debate about the vote", "active"),
            (102, "AI", "US", "Model release", "New model release announcement", "active"),
        ],
    )

    conn.executemany(
        "INSERT INTO links(id, norm_url, title, domain, event_id) VALUES(?, ?, ?, ?, ?)",
        [
            (1, "https://bbc.co.uk/story-1", "Budget vote passed", "bbc.co.uk", 100),
            (2, "https://reuters.com/story-2", "Model release", "reuters.com", 102),
        ],
    )

    conn.execute(
        """
        INSERT INTO clustering_decisions(
          id, link_id, prompt_sha256, model_name,
          llm_started_at_ts, llm_finished_at_ts, llm_response_json,
          validated_action, validated_action_json,
          enforced_action, enforced_action_json,
          error_type, error_message, created_at_ts
        ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            1,
            1,
            "sha-1",
            "gemini-flash",
            now_ts - 80,
            now_ts - 79,
            json.dumps({"action": "assign"}),
            "assign",
            json.dumps({"action": "assign", "event_id": 100}),
            "assign",
            json.dumps({"action": "assign", "event_id": 100}),
            None,
            None,
            now_ts - 60,
        ),
    )
    conn.executemany(
        "INSERT INTO clustering_decision_candidates(decision_id, rank, event_id, score) VALUES(?, ?, ?, ?)",
        [
            (1, 1, 101, 0.91),
            (1, 2, 100, 0.72),
        ],
    )

    conn.execute(
        """
        INSERT INTO clustering_decisions(
          id, link_id, prompt_sha256, model_name,
          llm_started_at_ts, llm_finished_at_ts, llm_response_json,
          validated_action, validated_action_json,
          enforced_action, enforced_action_json,
          error_type, error_message, created_at_ts
        ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            2,
            2,
            "sha-2",
            "gemini-flash",
            now_ts - 200,
            now_ts - 198,
            json.dumps({"action": "new_event"}),
            "new_event",
            json.dumps({"action": "new_event", "category": "AI"}),
            "new_event",
            json.dumps({"action": "new_event", "category": "AI"}),
            None,
            None,
            now_ts - 180,
        ),
    )
    conn.executemany(
        "INSERT INTO clustering_decision_candidates(decision_id, rank, event_id, score) VALUES(?, ?, ?, ?)",
        [
            (2, 1, 102, 0.81),
        ],
    )


class TestNewsroomClusteringDecisionsCLI(unittest.TestCase):
    def test_group_by_link_json_includes_similarity_hints(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "news_pool.sqlite3"
            conn = sqlite3.connect(str(db_path))
            try:
                _create_inspector_schema(conn)
                _seed_sample_data(conn, now_ts=int(time.time()))
                conn.commit()
            finally:
                conn.close()

            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = cli.main(["--db", str(db_path), "--group-by", "link", "--json"])

            self.assertEqual(rc, 0)
            payload = json.loads(buf.getvalue())
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["summary"]["decision_count"], 2)
            self.assertEqual(payload["group_by"], "link")

            decision_1 = next(d for d in payload["decisions"] if d["decision_id"] == 1)
            self.assertIn("assigned_non_top_candidate", decision_1["hints"])
            self.assertIn("assigned_far_below_top_candidate", decision_1["hints"])

            decision_2 = next(d for d in payload["decisions"] if d["decision_id"] == 2)
            self.assertIn("new_event_despite_strong_candidate", decision_2["hints"])

            link_group_1 = next(g for g in payload["groups"] if g["link_id"] == 1)
            self.assertEqual(link_group_1["decision_count"], 1)
            self.assertEqual(link_group_1["actions"].get("assign"), 1)
            self.assertEqual(link_group_1["hints"].get("assigned_non_top_candidate"), 1)

    def test_filters_by_event_domain_category_and_time(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "news_pool.sqlite3"
            conn = sqlite3.connect(str(db_path))
            try:
                _create_inspector_schema(conn)
                _seed_sample_data(conn, now_ts=int(time.time()))
                conn.commit()
            finally:
                conn.close()

            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = cli.main(
                    [
                        "--db",
                        str(db_path),
                        "--json",
                        "--group-by",
                        "event",
                        "--event-id",
                        "100",
                        "--domain",
                        "BBC.CO.UK",
                        "--category",
                        "Politics",
                        "--since-hours",
                        "1",
                    ]
                )

            self.assertEqual(rc, 0)
            payload = json.loads(buf.getvalue())
            self.assertEqual(payload["summary"]["decision_count"], 1)
            self.assertEqual(payload["decisions"][0]["decision_id"], 1)
            self.assertEqual(payload["decisions"][0]["link"]["domain"], "bbc.co.uk")

            group_event_ids = {int(g["event_id"]) for g in payload["groups"]}
            self.assertIn(100, group_event_ids)

    def test_human_readable_event_view(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "news_pool.sqlite3"
            conn = sqlite3.connect(str(db_path))
            try:
                _create_inspector_schema(conn)
                _seed_sample_data(conn, now_ts=int(time.time()))
                conn.commit()
            finally:
                conn.close()

            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = cli.main(["--db", str(db_path), "--group-by", "event", "--limit", "5"])

            self.assertEqual(rc, 0)
            text = buf.getvalue()
            self.assertIn("Decision Log Inspector", text)
            self.assertIn("By event:", text)
            self.assertIn("event#100", text)

    def test_graceful_fallback_when_decision_logs_missing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "news_pool.sqlite3"
            conn = sqlite3.connect(str(db_path))
            try:
                conn.execute("CREATE TABLE links (id INTEGER PRIMARY KEY, norm_url TEXT, title TEXT, domain TEXT, event_id INTEGER)")
                conn.commit()
            finally:
                conn.close()

            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = cli.main(["--db", str(db_path), "--json", "--pretty"])

            self.assertEqual(rc, 0)
            payload = json.loads(buf.getvalue())
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["summary"]["decision_count"], 0)
            warnings = payload.get("warnings") or []
            self.assertTrue(any("clustering_decisions table is missing" in w for w in warnings))


if __name__ == "__main__":
    unittest.main()
