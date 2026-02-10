import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from newsroom.brave_news import (
    BraveApiError,
    BraveApiKey,
    fetch_brave_news,
    record_brave_cooldown,
    record_brave_rate_limit,
    select_brave_api_key,
)


class TestBraveRateLimitCooldown(unittest.TestCase):
    @patch("newsroom.brave_news.time.sleep", autospec=True)
    @patch("newsroom.brave_news.requests.get")
    def test_cooldown_skips_key_after_429(self, mock_get: MagicMock, mock_sleep: MagicMock) -> None:
        mock_sleep.return_value = None

        mock_resp = MagicMock()
        mock_resp.status_code = 429
        mock_resp.headers = {
            "Retry-After": "10",
            "X-RateLimit-Limit": "1,1000",
            "X-RateLimit-Remaining": "0,500",
            "X-RateLimit-Reset": "1,3600",
        }
        mock_resp.json.return_value = {"error": {"message": "Too many requests"}}
        mock_resp.text = json.dumps(mock_resp.json.return_value)
        mock_get.return_value = mock_resp

        key1 = BraveApiKey(key="k1", label="one")
        key2 = BraveApiKey(key="k2", label="two")
        now_ts = 1700000000

        with tempfile.TemporaryDirectory() as td:
            openclaw_home = Path(td)

            with self.assertRaises(BraveApiError) as ctx:
                fetch_brave_news(api_key=key1.key, q="test", count=1, offset=0, freshness="day", cache_dir=None, last_request_ts=0.0)
            err = ctx.exception
            self.assertEqual(err.status_code, 429)
            self.assertEqual(err.requests_made, 3)
            self.assertIsInstance(err.rate_limit, dict)

            record_brave_rate_limit(openclaw_home=openclaw_home, key=key1, rate_limit=err.rate_limit, now_ts=now_ts)
            record_brave_cooldown(openclaw_home=openclaw_home, key=key1, cooldown_seconds=err.retry_after_s or 60, now_ts=now_ts, reason="http_429")

            selected = select_brave_api_key(openclaw_home=openclaw_home, keys=[key1, key2], now_ts=now_ts)
            self.assertEqual(selected.key_id, key2.key_id)

    @patch("newsroom.brave_news.time.sleep", autospec=True)
    @patch("newsroom.brave_news.requests.get")
    def test_requests_made_counts_retries_on_success(self, mock_get: MagicMock, mock_sleep: MagicMock) -> None:
        mock_sleep.return_value = None

        r503 = MagicMock()
        r503.status_code = 503
        r503.headers = {}
        r503.json.return_value = {"error": {"message": "Service unavailable"}}
        r503.text = json.dumps(r503.json.return_value)

        r200 = MagicMock()
        r200.status_code = 200
        r200.headers = {"X-RateLimit-Limit": "1,1000", "X-RateLimit-Remaining": "1,500", "X-RateLimit-Reset": "1,3600"}
        r200.json.return_value = {"results": [{"title": "A", "url": "https://example.com/a"}]}
        r200.text = json.dumps(r200.json.return_value)

        mock_get.side_effect = [r503, r503, r200]

        fetched, _ = fetch_brave_news(api_key="k", q="test", count=1, offset=0, freshness="day", cache_dir=None, last_request_ts=0.0)
        self.assertTrue(fetched.ok)
        self.assertEqual(fetched.requests_made, 3)
        self.assertEqual(len(fetched.results), 1)


if __name__ == "__main__":
    unittest.main()

