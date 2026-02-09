import json
import tempfile
import time
import unittest
from pathlib import Path

from newsroom.gateway_client import GatewayClient
from newsroom.job_store import atomic_write_json
from newsroom.runner import NewsroomRunner, PromptRegistry


class TestRunnerResume(unittest.TestCase):
    def test_start_job_monitors_existing_worker_uses_started_at_deadline(self) -> None:
        root = Path(__file__).resolve().parents[2]
        example = json.loads((root / "newsroom" / "examples" / "story_job_example.json").read_text(encoding="utf-8"))

        with tempfile.TemporaryDirectory() as td:
            tmp_dir = Path(td)
            job_path = tmp_dir / "story.json"

            job = json.loads(json.dumps(example))
            job["state"]["status"] = "DISPATCHED"
            job["state"]["worker"]["child_session_key"] = "child:worker"
            # Old enough that the computed deadline should already be in the past.
            job["state"]["worker"]["started_at"] = "2000-01-01T00:00:00Z"
            job["monitor"]["timeout_seconds"] = 10

            atomic_write_json(job_path, job)

            gateway = GatewayClient(http_url="http://127.0.0.1:1", token="dummy", default_session_key="hook:test")
            runner = NewsroomRunner(
                openclaw_home=root,
                gateway=gateway,
                prompt_registry=PromptRegistry(openclaw_home=root),
                dry_run=True,
                lock_ttl_seconds=3600,
                log_root=tmp_dir / "logs",
            )

            rt = runner._start_job(job_path=job_path, run_defaults={"default_timeout_seconds": 900, "trigger": "manual"})  # type: ignore[attr-defined]
            self.assertIsNotNone(rt)
            assert rt is not None
            self.assertEqual(rt.phase, "worker")
            self.assertLess(rt.deadline_ts, time.time())
            rt.lock.release()

    def test_start_job_prefers_existing_rescue_over_worker(self) -> None:
        root = Path(__file__).resolve().parents[2]
        example = json.loads((root / "newsroom" / "examples" / "story_job_example.json").read_text(encoding="utf-8"))

        with tempfile.TemporaryDirectory() as td:
            tmp_dir = Path(td)
            job_path = tmp_dir / "story.json"

            job = json.loads(json.dumps(example))
            job["state"]["status"] = "DISPATCHED"
            job["state"]["worker"]["child_session_key"] = "child:worker"
            job["state"]["worker"]["started_at"] = "2000-01-01T00:00:00Z"
            job["state"]["rescue"]["child_session_key"] = "child:rescue"
            job["state"]["rescue"]["started_at"] = "2000-01-01T00:00:00Z"
            job["validation"]["validator_id"] = "news_rescue_v1"
            job["recover"]["rescue_timeout_seconds"] = 10

            atomic_write_json(job_path, job)

            gateway = GatewayClient(http_url="http://127.0.0.1:1", token="dummy", default_session_key="hook:test")
            runner = NewsroomRunner(
                openclaw_home=root,
                gateway=gateway,
                prompt_registry=PromptRegistry(openclaw_home=root),
                dry_run=True,
                lock_ttl_seconds=3600,
                log_root=tmp_dir / "logs",
            )

            rt = runner._start_job(job_path=job_path, run_defaults={"default_timeout_seconds": 900, "trigger": "manual"})  # type: ignore[attr-defined]
            self.assertIsNotNone(rt)
            assert rt is not None
            self.assertEqual(rt.phase, "rescue")
            self.assertLess(rt.deadline_ts, time.time())
            rt.lock.release()

