import json
import tempfile
import unittest
from pathlib import Path

import jsonschema

from newsroom.gateway_client import GatewayClient
from newsroom.runner import NewsroomRunner, PromptRegistry, _URL_RE  # type: ignore[attr-defined]
from newsroom.job_store import atomic_write_json


class TestSchemas(unittest.TestCase):
    def test_story_job_example_validates(self) -> None:
        root = Path(__file__).resolve().parents[2]
        schema = json.loads((root / "newsroom" / "schemas" / "story_job_v1.schema.json").read_text(encoding="utf-8"))
        example = json.loads((root / "newsroom" / "examples" / "story_job_example.json").read_text(encoding="utf-8"))
        jsonschema.validate(instance=example, schema=schema)

    def test_run_job_example_validates(self) -> None:
        root = Path(__file__).resolve().parents[2]
        schema = json.loads((root / "newsroom" / "schemas" / "run_job_v1.schema.json").read_text(encoding="utf-8"))
        example = json.loads((root / "newsroom" / "examples" / "run_job_example.json").read_text(encoding="utf-8"))
        jsonschema.validate(instance=example, schema=schema)


class TestPromptRegistry(unittest.TestCase):
    def test_registry_resolves_prompt_and_validator(self) -> None:
        root = Path(__file__).resolve().parents[2]
        reg = PromptRegistry(openclaw_home=root)
        prompt = reg.resolve_prompt("news_reporter_v2_2_inline")
        self.assertTrue(prompt.template_path.exists())
        validator = reg.resolve_validator(prompt.validator_id)
        self.assertTrue(validator.path.exists())


class TestRunnerDryRun(unittest.TestCase):
    def test_dry_run_does_not_mutate_job_file(self) -> None:
        root = Path(__file__).resolve().parents[2]
        example_path = root / "newsroom" / "examples" / "story_job_example.json"
        original = example_path.read_text(encoding="utf-8")

        with tempfile.TemporaryDirectory() as td:
            tmp_dir = Path(td)
            job_path = tmp_dir / "story.json"
            job_path.write_text(original, encoding="utf-8")

            gateway = GatewayClient(http_url="http://127.0.0.1:1", token="dummy", default_session_key="hook:test")
            runner = NewsroomRunner(
                openclaw_home=root,
                gateway=gateway,
                prompt_registry=PromptRegistry(openclaw_home=root),
                dry_run=True,
                lock_ttl_seconds=1,
                log_root=tmp_dir / "logs",
            )

            runner.run_group(job_paths=[job_path], run_json_path=None)

            self.assertEqual(job_path.read_text(encoding="utf-8"), original)
            self.assertFalse((job_path.with_suffix(job_path.suffix + ".lock")).exists())

    def test_dry_run_multi_story_run_does_not_mutate_files(self) -> None:
        root = Path(__file__).resolve().parents[2]
        story_example = json.loads((root / "newsroom" / "examples" / "story_job_example.json").read_text(encoding="utf-8"))
        run_example = json.loads((root / "newsroom" / "examples" / "run_job_example.json").read_text(encoding="utf-8"))

        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "discord-multi-2026-02-04-07-00"
            run_dir.mkdir(parents=True, exist_ok=True)

            # Write run.json.
            run_job = dict(run_example)
            run_job["run"] = {"run_id": run_dir.name, "trigger": "cron_daily", "run_time_uk": "2026-02-04 07:00"}
            run_job_path = run_dir / "run.json"
            atomic_write_json(run_job_path, run_job)

            # Write 2 story jobs.
            paths = []
            originals = []
            for i in range(1, 3):
                job = json.loads(json.dumps(story_example))
                job["run"] = {"run_id": run_dir.name, "trigger": "cron_daily", "run_time_uk": "2026-02-04 07:00"}
                job["story"]["story_id"] = f"story_{i:02d}"
                job["story"]["primary_url"] = f"https://example.com/{i}"
                job["story"]["supporting_urls"] = [f"https://example.org/{i}a", f"https://example.net/{i}b"]
                job["story"]["dedupe_key"] = f"event:dryrun{i:02d}"
                job["destination"]["title_channel_id"] = "1467628391082496041"
                p = run_dir / f"story_{i:02d}.json"
                atomic_write_json(p, job)
                paths.append(p)
                originals.append(p.read_text(encoding="utf-8"))

            gateway = GatewayClient(http_url="http://127.0.0.1:1", token="dummy", default_session_key="hook:test")
            runner = NewsroomRunner(
                openclaw_home=root,
                gateway=gateway,
                prompt_registry=PromptRegistry(openclaw_home=root),
                dry_run=True,
                lock_ttl_seconds=1,
                log_root=Path(td) / "logs",
            )

            runner.run_group(job_paths=paths, run_json_path=run_job_path)

            for p, orig in zip(paths, originals, strict=True):
                self.assertEqual(p.read_text(encoding="utf-8"), orig)
                self.assertFalse((p.with_suffix(p.suffix + ".lock")).exists())

    def test_runner_can_load_validators(self) -> None:
        root = Path(__file__).resolve().parents[2]
        reg = PromptRegistry(openclaw_home=root)
        gateway = GatewayClient(http_url="http://127.0.0.1:1", token="dummy", default_session_key="hook:test")
        with tempfile.TemporaryDirectory() as td:
            runner = NewsroomRunner(
                openclaw_home=root,
                gateway=gateway,
                prompt_registry=reg,
                dry_run=True,
                lock_ttl_seconds=1,
                log_root=Path(td) / "logs",
            )

            for validator_id in ("news_reporter_v2_1_strict", "news_rescue_v1"):
                vdef = reg.resolve_validator(validator_id)
                fn = runner._get_validator(vdef)  # type: ignore[attr-defined]
                self.assertTrue(callable(fn))

    def test_runner_skips_duplicate_jobs_via_dedupe_marker(self) -> None:
        root = Path(__file__).resolve().parents[2]
        example_path = root / "newsroom" / "examples" / "story_job_example.json"
        example = json.loads(example_path.read_text(encoding="utf-8"))

        with tempfile.TemporaryDirectory() as td:
            tmp_dir = Path(td)
            job_path = tmp_dir / "story.json"

            example["story"]["dedupe_key"] = "https://example.com/a?utm_source=x"
            # Ensure no Discord container exists yet.
            example["state"]["discord"]["title_message_id"] = None
            example["state"]["discord"]["thread_id"] = None
            job_path.write_text(json.dumps(example, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")

            log_root = tmp_dir / "logs"
            gateway = GatewayClient(http_url="http://127.0.0.1:1", token="dummy", default_session_key="hook:test")
            runner = NewsroomRunner(
                openclaw_home=root,
                gateway=gateway,
                prompt_registry=PromptRegistry(openclaw_home=root),
                dry_run=False,
                lock_ttl_seconds=1,
                log_root=log_root,
            )

            # Pre-create a fresh marker for this dedupe key.
            dedupe_key = runner._dedupe_key_for_job(example)  # type: ignore[attr-defined]
            marker_path = runner._dedupe_marker_path(dedupe_key)  # type: ignore[attr-defined]
            atomic_write_json(marker_path, {"dedupe_key": dedupe_key, "created_at": "now"})

            runner.run_group(job_paths=[job_path], run_json_path=None)

            updated = json.loads(job_path.read_text(encoding="utf-8"))
            self.assertEqual(updated["state"]["status"], "SKIPPED")
            self.assertEqual(updated["result"]["final_status"], "SKIPPED_DUPLICATE")

    def test_runner_skips_duplicate_jobs_via_legacy_marker_primary_url_index(self) -> None:
        root = Path(__file__).resolve().parents[2]
        example_path = root / "newsroom" / "examples" / "story_job_example.json"
        example = json.loads(example_path.read_text(encoding="utf-8"))

        with tempfile.TemporaryDirectory() as td:
            tmp_dir = Path(td)
            job_path = tmp_dir / "story.json"

            example["story"]["primary_url"] = "https://example.com/a?utm_source=job"
            example["story"]["dedupe_key"] = "https://example.com/a?utm_source=job|SOME_HASH"
            example["state"]["discord"]["title_message_id"] = None
            example["state"]["discord"]["thread_id"] = None
            job_path.write_text(json.dumps(example, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")

            log_root = tmp_dir / "logs"
            gateway = GatewayClient(http_url="http://127.0.0.1:1", token="dummy", default_session_key="hook:test")
            runner = NewsroomRunner(
                openclaw_home=root,
                gateway=gateway,
                prompt_registry=PromptRegistry(openclaw_home=root),
                dry_run=False,
                lock_ttl_seconds=1,
                log_root=log_root,
            )

            # Simulate a legacy marker written by an older runner keyed by url|hash.
            legacy_key = "https://example.com/a?utm_source=old|LEGACY_HASH"
            legacy_marker_path = runner._dedupe_marker_path(legacy_key)  # type: ignore[attr-defined]
            atomic_write_json(
                legacy_marker_path,
                {"dedupe_key": legacy_key, "primary_url": "https://example.com/a?utm_source=old", "created_at": "now"},
            )

            runner.run_group(job_paths=[job_path], run_json_path=None)

            updated = json.loads(job_path.read_text(encoding="utf-8"))
            self.assertEqual(updated["state"]["status"], "SKIPPED")
            self.assertEqual(updated["result"]["final_status"], "SKIPPED_DUPLICATE")

    def test_runner_skips_duplicate_jobs_when_primary_url_matches_but_dedupe_key_scheme_differs(self) -> None:
        root = Path(__file__).resolve().parents[2]
        example_path = root / "newsroom" / "examples" / "story_job_example.json"
        example = json.loads(example_path.read_text(encoding="utf-8"))

        with tempfile.TemporaryDirectory() as td:
            tmp_dir = Path(td)
            job_path = tmp_dir / "story.json"

            # Job uses an event:... key, but primary_url matches an existing anchor:... marker.
            example["story"]["primary_url"] = "https://example.com/a?utm_source=job"
            example["story"]["dedupe_key"] = "event:aaaaaaaaaaaaaaaaaaaa"
            example["state"]["discord"]["title_message_id"] = None
            example["state"]["discord"]["thread_id"] = None
            job_path.write_text(json.dumps(example, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")

            log_root = tmp_dir / "logs"
            gateway = GatewayClient(http_url="http://127.0.0.1:1", token="dummy", default_session_key="hook:test")
            runner = NewsroomRunner(
                openclaw_home=root,
                gateway=gateway,
                prompt_registry=PromptRegistry(openclaw_home=root),
                dry_run=False,
                lock_ttl_seconds=1,
                log_root=log_root,
            )

            marker_path = runner._dedupe_marker_path("anchor:bbbbbbbbbbbbbbbbbbbb")  # type: ignore[attr-defined]
            atomic_write_json(
                marker_path,
                {
                    "dedupe_key": "anchor:bbbbbbbbbbbbbbbbbbbb",
                    "primary_url": "https://example.com/a?utm_source=old",
                    "created_at": "now",
                },
            )

            runner.run_group(job_paths=[job_path], run_json_path=None)

            updated = json.loads(job_path.read_text(encoding="utf-8"))
            self.assertEqual(updated["state"]["status"], "SKIPPED")
            self.assertEqual(updated["result"]["final_status"], "SKIPPED_DUPLICATE")

    def test_strict_validator_allows_zero_images_on_success(self) -> None:
        root = Path(__file__).resolve().parents[2]
        validator_path = root / "newsroom" / "validators" / "news_reporter_v2_1_strict.py"
        import importlib.util
        import sys

        spec = importlib.util.spec_from_file_location("news_reporter_v2_1_strict", validator_path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        # Dataclasses with string annotations may consult sys.modules during import.
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)  # type: ignore[union-attr]
        validate = getattr(module, "validate")

        job = json.loads((root / "newsroom" / "examples" / "story_job_example.json").read_text(encoding="utf-8"))
        # Simulate a success result with no images.
        result_json = {
            "status": "SUCCESS",
            "story_id": job["story"]["story_id"],
            "category": job["story"]["category"],
            "title": job["story"]["title"],
            "primary_url": job["story"]["primary_url"],
            "thread_id": "123456789012345678",
            "content_posted": True,
            "content_message_ids": ["123456789012345679"],
            "images_attached_count": 0,
            "read_more_urls_count": 3,
            "report_char_count": 3000,
            "concrete_anchor_provided": job["story"]["concrete_anchor"],
            "concrete_anchor_used": True,
            "sources_used": [job["story"]["primary_url"], "https://example.com/2", "https://example.com/3"],
            "error_type": None,
            "error_message": None,
        }

        res = validate(result_json, job)
        self.assertTrue(res.ok, msg=str(res.errors))

    def test_url_regex_extracts_discord_angle_bracket_links(self) -> None:
        text = "延伸閱讀\n\n<https://example.com/a>\n<https://example.com/b?x=1#frag>\n"
        urls = _URL_RE.findall(text)
        # We keep the raw match (may include trailing '>'); we just need non-empty extraction.
        self.assertGreaterEqual(len(urls), 2)


if __name__ == "__main__":
    unittest.main()
