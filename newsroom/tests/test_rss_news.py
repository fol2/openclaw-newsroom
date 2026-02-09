import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from newsroom.rss_news import (
    RssArticle,
    RssFeed,
    RssFetch,
    _parse_rss_date,
    fetch_rss_feed,
    load_feeds,
    parse_rss_xml,
)

RSS_20_SAMPLE = b"""\
<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Test Feed</title>
    <item>
      <title>Article One</title>
      <link>https://example.com/one</link>
      <description>First article description</description>
      <pubDate>Sat, 08 Feb 2026 10:00:00 GMT</pubDate>
    </item>
    <item>
      <title>Article Two</title>
      <link>https://example.com/two?utm_source=rss</link>
      <pubDate>Sat, 08 Feb 2026 11:00:00 GMT</pubDate>
    </item>
  </channel>
</rss>
"""

ATOM_SAMPLE = b"""\
<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>Atom Feed</title>
  <entry>
    <title>Atom Entry</title>
    <link href="https://example.com/atom1"/>
    <summary>Atom summary</summary>
    <published>2026-02-08T12:00:00Z</published>
  </entry>
</feed>
"""


class TestParseRssDate(unittest.TestCase):
    def test_rfc822(self) -> None:
        result = _parse_rss_date("Sat, 08 Feb 2026 10:00:00 GMT")
        self.assertIsNotNone(result)
        self.assertIn("2026-02-08", result)

    def test_iso8601(self) -> None:
        result = _parse_rss_date("2026-02-08T12:00:00Z")
        self.assertIsNotNone(result)
        self.assertIn("2026-02-08", result)

    def test_iso8601_with_offset(self) -> None:
        result = _parse_rss_date("2026-02-08T12:00:00+08:00")
        self.assertIsNotNone(result)
        self.assertIn("2026-02-08", result)

    def test_empty(self) -> None:
        self.assertIsNone(_parse_rss_date(""))
        self.assertIsNone(_parse_rss_date(None))

    def test_invalid(self) -> None:
        self.assertIsNone(_parse_rss_date("not a date"))


class TestParseRssXml(unittest.TestCase):
    def test_rss20(self) -> None:
        articles = parse_rss_xml(RSS_20_SAMPLE)
        self.assertEqual(len(articles), 2)
        self.assertEqual(articles[0].title, "Article One")
        self.assertEqual(articles[0].url, "https://example.com/one")
        self.assertEqual(articles[0].description, "First article description")
        self.assertIsNotNone(articles[0].published)

    def test_rss20_strips_utm(self) -> None:
        articles = parse_rss_xml(RSS_20_SAMPLE)
        # Article Two URL has utm_source stripped by normalize_url.
        self.assertNotIn("utm_source", articles[1].url)

    def test_atom(self) -> None:
        articles = parse_rss_xml(ATOM_SAMPLE)
        self.assertEqual(len(articles), 1)
        self.assertEqual(articles[0].title, "Atom Entry")
        self.assertEqual(articles[0].url, "https://example.com/atom1")
        self.assertEqual(articles[0].description, "Atom summary")
        self.assertIsNotNone(articles[0].published)

    def test_empty_xml(self) -> None:
        articles = parse_rss_xml(b"<rss><channel></channel></rss>")
        self.assertEqual(articles, [])

    def test_malformed_xml(self) -> None:
        articles = parse_rss_xml(b"not xml at all <><>")
        self.assertEqual(articles, [])

    def test_items_without_url_or_title_skipped(self) -> None:
        xml = b"""\
<?xml version="1.0"?>
<rss version="2.0">
  <channel>
    <item><title>No link</title></item>
    <item><link>https://example.com/no-title</link></item>
    <item><title>Valid</title><link>https://example.com/valid</link></item>
  </channel>
</rss>"""
        articles = parse_rss_xml(xml)
        self.assertEqual(len(articles), 1)
        self.assertEqual(articles[0].title, "Valid")


class TestLoadFeeds(unittest.TestCase):
    def test_default_feeds(self) -> None:
        feeds = load_feeds(config_path=None)
        self.assertGreater(len(feeds), 0)
        self.assertIsInstance(feeds[0], RssFeed)

    def test_load_from_yaml(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            f.write("feeds:\n  - key: test\n    url: https://example.com/rss\n    label: Test\n    region: global\n")
            f.flush()
            feeds = load_feeds(config_path=Path(f.name))
        self.assertEqual(len(feeds), 1)
        self.assertEqual(feeds[0].key, "test")
        self.assertEqual(feeds[0].url, "https://example.com/rss")

    def test_fallback_on_invalid_yaml(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            f.write("invalid: yaml: ][")
            f.flush()
            feeds = load_feeds(config_path=Path(f.name))
        # Should fall back to defaults.
        self.assertGreater(len(feeds), 0)

    def test_nonexistent_file_returns_defaults(self) -> None:
        feeds = load_feeds(config_path=Path("/tmp/nonexistent_rss_feeds_test.yaml"))
        self.assertGreater(len(feeds), 0)


class TestFetchRssFeed(unittest.TestCase):
    @patch("newsroom.rss_news.requests.get")
    def test_success(self, mock_get: MagicMock) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.content = RSS_20_SAMPLE
        mock_get.return_value = mock_resp

        feed = RssFeed(key="test", url="https://example.com/rss", label="Test", region="global")
        result, ts = fetch_rss_feed(feed, last_request_ts=0.0)
        self.assertTrue(result.ok)
        self.assertEqual(result.feed_key, "test")
        self.assertEqual(result.requests_made, 1)
        self.assertEqual(len(result.articles), 2)
        self.assertIsNone(result.error)

    @patch("newsroom.rss_news.requests.get")
    def test_http_error(self, mock_get: MagicMock) -> None:
        import requests

        mock_get.side_effect = requests.ConnectionError("Timeout")
        feed = RssFeed(key="fail", url="https://example.com/rss", label="Fail", region="global")
        result, ts = fetch_rss_feed(feed, last_request_ts=0.0)
        self.assertFalse(result.ok)
        self.assertEqual(result.feed_key, "fail")
        self.assertEqual(len(result.articles), 0)
        self.assertIn("Timeout", result.error or "")


if __name__ == "__main__":
    unittest.main()
