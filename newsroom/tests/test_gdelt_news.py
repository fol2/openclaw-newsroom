import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from newsroom.gdelt_news import GdeltFetch, fetch_gdelt_articles, parse_gdelt_seendate


class TestParseGdeltSeendate(unittest.TestCase):
    def test_valid(self) -> None:
        self.assertEqual(parse_gdelt_seendate("20260208T143000Z"), "2026-02-08T14:30:00+00:00")

    def test_midnight(self) -> None:
        self.assertEqual(parse_gdelt_seendate("20260101T000000Z"), "2026-01-01T00:00:00+00:00")

    def test_empty(self) -> None:
        self.assertIsNone(parse_gdelt_seendate(""))
        self.assertIsNone(parse_gdelt_seendate(None))

    def test_malformed(self) -> None:
        self.assertIsNone(parse_gdelt_seendate("not-a-date"))
        self.assertIsNone(parse_gdelt_seendate("2026-02-08T14:30:00Z"))  # wrong format for GDELT


class TestFetchGdeltArticles(unittest.TestCase):
    def test_empty_query_raises(self) -> None:
        with self.assertRaises(ValueError):
            fetch_gdelt_articles(query="")

    @patch("newsroom.gdelt_news.requests.get")
    def test_success(self, mock_get: MagicMock) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "articles": [
                {
                    "url": "https://example.com/article1",
                    "title": "Test Article",
                    "seendate": "20260208T100000Z",
                    "domain": "example.com",
                    "language": "English",
                    "sourcecountry": "United States",
                },
                {
                    "url": "https://example.com/article2",
                    "title": "Another Article",
                    "seendate": "20260208T110000Z",
                    "domain": "example.com",
                },
            ]
        }
        mock_get.return_value = mock_resp

        result, ts = fetch_gdelt_articles(query="test query", last_request_ts=0.0)
        self.assertTrue(result.ok)
        self.assertFalse(result.cached)
        self.assertEqual(result.requests_made, 1)
        self.assertEqual(len(result.results), 2)
        self.assertEqual(result.results[0]["title"], "Test Article")
        self.assertEqual(result.results[0]["page_age"], "2026-02-08T10:00:00+00:00")
        self.assertIsNone(result.error)

    @patch("newsroom.gdelt_news.requests.get")
    def test_http_error(self, mock_get: MagicMock) -> None:
        import requests

        mock_get.side_effect = requests.ConnectionError("Connection refused")
        result, ts = fetch_gdelt_articles(query="test", last_request_ts=0.0)
        self.assertFalse(result.ok)
        self.assertEqual(result.requests_made, 1)
        self.assertEqual(len(result.results), 0)
        self.assertIn("Connection refused", result.error or "")

    @patch("newsroom.gdelt_news.requests.get")
    def test_cache_hit(self, mock_get: MagicMock) -> None:
        with tempfile.TemporaryDirectory() as td:
            cache_dir = Path(td)

            # First call: populate cache.
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.raise_for_status = MagicMock()
            mock_resp.json.return_value = {
                "articles": [
                    {"url": "https://example.com/a", "title": "A", "seendate": "20260208T100000Z", "domain": "example.com"},
                ]
            }
            mock_get.return_value = mock_resp
            result1, _ = fetch_gdelt_articles(query="cached test", cache_dir=cache_dir, ttl_seconds=3600, last_request_ts=0.0)
            self.assertTrue(result1.ok)
            self.assertFalse(result1.cached)
            self.assertEqual(mock_get.call_count, 1)

            # Second call: cache hit.
            result2, _ = fetch_gdelt_articles(query="cached test", cache_dir=cache_dir, ttl_seconds=3600, last_request_ts=0.0)
            self.assertTrue(result2.ok)
            self.assertTrue(result2.cached)
            self.assertEqual(result2.requests_made, 0)
            # No additional HTTP call.
            self.assertEqual(mock_get.call_count, 1)

    @patch("newsroom.gdelt_news.requests.get")
    def test_skips_articles_without_url_or_title(self, mock_get: MagicMock) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "articles": [
                {"url": "", "title": "No URL"},
                {"url": "https://example.com/b", "title": ""},
                {"title": "Missing URL field"},
                {"url": "https://example.com/c", "title": "Valid"},
            ]
        }
        mock_get.return_value = mock_resp
        result, _ = fetch_gdelt_articles(query="filter test", last_request_ts=0.0)
        self.assertTrue(result.ok)
        self.assertEqual(len(result.results), 1)
        self.assertEqual(result.results[0]["title"], "Valid")


if __name__ == "__main__":
    unittest.main()
