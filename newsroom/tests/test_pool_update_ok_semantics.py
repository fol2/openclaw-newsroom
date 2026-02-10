import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import scripts.gdelt_pool_update as gdelt_pool_update
import scripts.rss_pool_update as rss_pool_update
from newsroom.gdelt_news import GdeltFetch
from newsroom.news_pool_db import NewsPoolDB
from newsroom.rss_news import RssArticle, RssFeed, RssFetch


class TestPoolUpdateOkSemantics(unittest.TestCase):
    def test_gdelt_does_not_update_fetch_state_on_fetch_failure(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "news_pool.sqlite3"
            state_key = "gdelt_test"

            def fake_fetch_gdelt_articles(**kwargs):
                return (
                    GdeltFetch(
                        ok=False,
                        cached=False,
                        requests_made=1,
                        query=str(kwargs.get("query") or ""),
                        maxrecords=int(kwargs.get("maxrecords") or 75),
                        timespan=str(kwargs.get("timespan") or "24h"),
                        fetched_at="2026-02-10T00:00:00+00:00",
                        results=[],
                        error="boom",
                    ),
                    0.0,
                )

            buf = io.StringIO()
            with patch.object(gdelt_pool_update, "fetch_gdelt_articles", side_effect=fake_fetch_gdelt_articles):
                with redirect_stdout(buf):
                    rc = gdelt_pool_update.main(
                        [
                            "--db",
                            str(db_path),
                            "--state-key",
                            state_key,
                            "--min-interval-seconds",
                            "0",
                            "--query",
                            "test",
                        ]
                    )

            self.assertEqual(rc, 0)
            out = json.loads(buf.getvalue())
            self.assertFalse(out["ok"])
            self.assertTrue(out["fetch"]["should_fetch"])
            self.assertFalse(out["fetch"]["ok"])
            self.assertEqual(out["fetch"]["error"], "boom")

            with NewsPoolDB(path=db_path) as db:
                state = db.fetch_state(state_key)
                self.assertEqual(state["run_count"], 0)
                self.assertEqual(state["last_fetch_ts"], 0)

    def test_gdelt_updates_fetch_state_on_success(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "news_pool.sqlite3"
            state_key = "gdelt_test"

            def fake_fetch_gdelt_articles(**kwargs):
                return (
                    GdeltFetch(
                        ok=True,
                        cached=False,
                        requests_made=1,
                        query=str(kwargs.get("query") or ""),
                        maxrecords=int(kwargs.get("maxrecords") or 75),
                        timespan=str(kwargs.get("timespan") or "24h"),
                        fetched_at="2026-02-10T00:00:00+00:00",
                        results=[
                            {
                                "title": "Example",
                                "url": "https://example.com/a",
                                "domain": "example.com",
                                "page_age": "2026-02-10T00:00:00+00:00",
                            }
                        ],
                    ),
                    0.0,
                )

            buf = io.StringIO()
            with patch.object(gdelt_pool_update, "fetch_gdelt_articles", side_effect=fake_fetch_gdelt_articles):
                with redirect_stdout(buf):
                    rc = gdelt_pool_update.main(
                        [
                            "--db",
                            str(db_path),
                            "--state-key",
                            state_key,
                            "--min-interval-seconds",
                            "0",
                            "--query",
                            "test",
                        ]
                    )

            self.assertEqual(rc, 0)
            out = json.loads(buf.getvalue())
            self.assertTrue(out["ok"])
            self.assertTrue(out["fetch"]["should_fetch"])
            self.assertTrue(out["fetch"]["ok"])

            with NewsPoolDB(path=db_path) as db:
                state = db.fetch_state(state_key)
                self.assertEqual(state["run_count"], 1)
                self.assertGreater(state["last_fetch_ts"], 0)

    def test_rss_does_not_update_fetch_state_if_all_feeds_fail(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "news_pool.sqlite3"
            state_key = "rss_test"

            feeds = [
                RssFeed(key="f1", url="https://example.com/f1.xml", label="Feed 1", region="global"),
                RssFeed(key="f2", url="https://example.com/f2.xml", label="Feed 2", region="global"),
            ]

            def fake_load_feeds(*, config_path: Path | None = None):
                return list(feeds)

            def fake_fetch_rss_feed(feed: RssFeed, **kwargs):
                return (
                    RssFetch(
                        ok=False,
                        feed_key=feed.key,
                        requests_made=1,
                        fetched_at="2026-02-10T00:00:00+00:00",
                        articles=[],
                        error=f"fail:{feed.key}",
                    ),
                    0.0,
                )

            buf = io.StringIO()
            with patch.object(rss_pool_update, "load_feeds", side_effect=fake_load_feeds):
                with patch.object(rss_pool_update, "fetch_rss_feed", side_effect=fake_fetch_rss_feed):
                    with redirect_stdout(buf):
                        rc = rss_pool_update.main(
                            [
                                "--db",
                                str(db_path),
                                "--state-key",
                                state_key,
                                "--min-interval-seconds",
                                "0",
                                "--feeds-config",
                                str(Path(td) / "rss_feeds.yaml"),
                            ]
                        )

            self.assertEqual(rc, 0)
            out = json.loads(buf.getvalue())
            self.assertFalse(out["ok"])
            self.assertTrue(out["fetch"]["should_fetch"])
            self.assertFalse(out["fetch"]["ok"])
            self.assertEqual(out["fetch"]["feeds_ok"], 0)
            self.assertEqual(out["fetch"]["feeds_failed"], 2)
            self.assertIn("errors", out["fetch"])

            with NewsPoolDB(path=db_path) as db:
                state = db.fetch_state(state_key)
                self.assertEqual(state["run_count"], 0)
                self.assertEqual(state["last_fetch_ts"], 0)

    def test_rss_updates_fetch_state_if_any_feed_succeeds(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "news_pool.sqlite3"
            state_key = "rss_test"

            feeds = [
                RssFeed(key="f1", url="https://example.com/f1.xml", label="Feed 1", region="global"),
                RssFeed(key="f2", url="https://example.com/f2.xml", label="Feed 2", region="global"),
            ]

            def fake_load_feeds(*, config_path: Path | None = None):
                return list(feeds)

            def fake_fetch_rss_feed(feed: RssFeed, **kwargs):
                if feed.key == "f1":
                    return (
                        RssFetch(
                            ok=False,
                            feed_key=feed.key,
                            requests_made=1,
                            fetched_at="2026-02-10T00:00:00+00:00",
                            articles=[],
                            error="fail:f1",
                        ),
                        0.0,
                    )

                return (
                    RssFetch(
                        ok=True,
                        feed_key=feed.key,
                        requests_made=1,
                        fetched_at="2026-02-10T00:00:00+00:00",
                        articles=[
                            RssArticle(
                                url="https://example.com/a",
                                title="Example",
                                description=None,
                                published="2026-02-10T00:00:00+00:00",
                                domain="example.com",
                            )
                        ],
                    ),
                    0.0,
                )

            buf = io.StringIO()
            with patch.object(rss_pool_update, "load_feeds", side_effect=fake_load_feeds):
                with patch.object(rss_pool_update, "fetch_rss_feed", side_effect=fake_fetch_rss_feed):
                    with redirect_stdout(buf):
                        rc = rss_pool_update.main(
                            [
                                "--db",
                                str(db_path),
                                "--state-key",
                                state_key,
                                "--min-interval-seconds",
                                "0",
                                "--feeds-config",
                                str(Path(td) / "rss_feeds.yaml"),
                            ]
                        )

            self.assertEqual(rc, 0)
            out = json.loads(buf.getvalue())
            self.assertTrue(out["ok"])
            self.assertTrue(out["fetch"]["should_fetch"])
            self.assertTrue(out["fetch"]["ok"])
            self.assertEqual(out["fetch"]["feeds_ok"], 1)
            self.assertEqual(out["fetch"]["feeds_failed"], 1)

            with NewsPoolDB(path=db_path) as db:
                state = db.fetch_state(state_key)
                self.assertEqual(state["run_count"], 1)
                self.assertGreater(state["last_fetch_ts"], 0)


if __name__ == "__main__":
    unittest.main()

