import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import scripts.newsroom_daily_inputs as newsroom_daily_inputs
import scripts.newsroom_hourly_inputs as newsroom_hourly_inputs


class _FakeDailyDB:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def get_daily_candidates(self, *, limit: int) -> list[dict]:
        _ = limit
        return [
            {
                "id": 101,
                "event_id": 101,
                "title": "Daily candidate",
                "primary_url": "https://example.com/daily",
                "supporting_urls": [],
            }
        ]

    def close(self) -> None:
        return None


class _FakeHourlyDB:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def get_hourly_candidates(self, *, limit: int) -> list[dict]:
        _ = limit
        return [
            {
                "id": 201,
                "event_id": 201,
                "title": "Hourly candidate",
                "primary_url": "https://example.com/hourly",
                "supporting_urls": [],
            }
        ]

    def close(self) -> None:
        return None


class TestInputsEventGcHooks(unittest.TestCase):
    def test_daily_invokes_gc_hook_and_records_summary(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out_path = Path(td) / "daily_inputs_last.json"
            gc_summary = {
                "ok": True,
                "state_key": "gc_key_daily",
                "should_gc": True,
                "pruned": 3,
                "run_count": 2,
                "last_gc_ts": 123,
                "criteria": {
                    "stale_after_hours": 96,
                    "low_link_max": 1,
                    "low_quality_summary_chars": 80,
                },
            }

            with patch.object(newsroom_daily_inputs, "_run_json", return_value={"ok": True}):
                with patch.object(newsroom_daily_inputs.time, "sleep", return_value=None):
                    with patch.object(newsroom_daily_inputs, "NewsPoolDB", _FakeDailyDB):
                        with patch.object(
                            newsroom_daily_inputs,
                            "run_periodic_unposted_event_gc",
                            return_value=gc_summary,
                        ) as gc_mock:
                            rc = newsroom_daily_inputs.main(
                                [
                                    "--no-llm",
                                    "--no-gdelt",
                                    "--no-rss",
                                    "--event-gc-state-key",
                                    "gc_key_daily",
                                    "--write-path",
                                    str(out_path),
                                ]
                            )

            self.assertEqual(rc, 0)
            gc_mock.assert_called_once()
            self.assertEqual(gc_mock.call_args.kwargs.get("state_key"), "gc_key_daily")

            payload = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(payload.get("event_gc", {}).get("pruned"), 3)
            self.assertEqual(payload.get("candidate_count"), 1)

    def test_hourly_gc_failure_is_non_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out_path = Path(td) / "hourly_inputs_last.json"

            with patch.object(newsroom_hourly_inputs, "_run_json", return_value={"ok": True}):
                with patch.object(newsroom_hourly_inputs, "NewsPoolDB", _FakeHourlyDB):
                    with patch.object(
                        newsroom_hourly_inputs,
                        "run_periodic_unposted_event_gc",
                        side_effect=RuntimeError("gc failure"),
                    ) as gc_mock:
                        rc = newsroom_hourly_inputs.main(
                            [
                                "--no-llm",
                                "--no-rss",
                                "--event-gc-state-key",
                                "gc_key_hourly",
                                "--write-path",
                                str(out_path),
                            ]
                        )

            self.assertEqual(rc, 0)
            gc_mock.assert_called_once()

            payload = json.loads(out_path.read_text(encoding="utf-8"))
            event_gc = payload.get("event_gc", {})
            self.assertEqual(event_gc.get("ok"), False)
            self.assertEqual(event_gc.get("state_key"), "gc_key_hourly")
            self.assertIn("gc failure", str(event_gc.get("error")))
            self.assertEqual(payload.get("candidate_count"), 1)


if __name__ == "__main__":
    unittest.main()
