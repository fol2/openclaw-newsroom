import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from newsroom.brave_news import BraveApiKey
from newsroom.runner import (
    NewsroomRunner,
    PromptRegistry,
    PromptRegistryError,
    _render_template,
    discover_jobs_under,
    discover_story_job_files,
)


class _DummyGateway:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None, dict[str, object]]] = []

    def invoke(  # type: ignore[no-untyped-def]
        self,
        *,
        tool: str,
        action: str | None = None,
        args: dict[str, object] | None = None,
        **_kwargs,
    ) -> dict[str, object]:
        self.calls.append((tool, action, dict(args or {})))
        return {"ok": True}


class TestRunnerHardening(unittest.TestCase):
    def test_render_template_does_not_replace_inside_input_json(self) -> None:
        tpl = "header\n{{INPUT_JSON}}\nfooter\nhome={{OPENCLAW_HOME}}\n"
        injected = json.dumps({"x": "{{OPENCLAW_HOME}}", "y": "{{MISSING_VAR}}"}, ensure_ascii=False)
        out = _render_template(tpl, {"INPUT_JSON": injected, "OPENCLAW_HOME": "/tmp/openclaw"})
        # Placeholders inside INPUT_JSON must remain literal.
        self.assertIn('"x": "{{OPENCLAW_HOME}}"', out)
        self.assertIn('"y": "{{MISSING_VAR}}"', out)
        # Placeholders in the template itself must be rendered.
        self.assertIn("home=/tmp/openclaw", out)

    def test_render_template_rejects_mixed_case_placeholder_names(self) -> None:
        with self.assertRaises(PromptRegistryError):
            _render_template("{{Input_JSON}}", {"INPUT_JSON": "{}"})

    def test_render_template_rejects_missing_placeholders(self) -> None:
        with self.assertRaises(PromptRegistryError):
            _render_template("{{MISSING_VAR}}", {"INPUT_JSON": "{}"})

    def test_send_dm_summary_is_best_effort_and_continues(self) -> None:
        root = Path(__file__).resolve().parents[2]

        class FlakyGateway(_DummyGateway):
            def invoke(self, *, tool: str, action: str | None = None, args=None, **kwargs):  # type: ignore[no-untyped-def]
                # Fail only the first send.
                if len(self.calls) == 0:
                    self.calls.append((tool, action, dict(args or {})))
                    raise RuntimeError("boom")
                return super().invoke(tool=tool, action=action, args=args, **kwargs)

        gw = FlakyGateway()
        runner = NewsroomRunner(
            openclaw_home=root,
            gateway=gw,  # type: ignore[arg-type]
            prompt_registry=PromptRegistry(openclaw_home=root),
            dry_run=False,
            lock_ttl_seconds=3600,
            log_root=root / "logs" / "newsroom-test",
        )

        with self.assertLogs("newsroom.runner", level="WARNING") as cm:
            runner.send_dm_summary(dm_targets=["discord:1", "discord:2"], summary="test\n")

        self.assertGreaterEqual(len(gw.calls), 2)
        self.assertTrue(any("DM summary send failed" in msg for msg in cm.output))

    def test_discover_story_job_files_jails_unreadable_job_like_json(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            (d / "run.json").write_text("{}", encoding="utf-8")

            bad = d / "story_01.json"
            bad.write_text("{not-json", encoding="utf-8")

            with self.assertLogs("newsroom.runner", level="WARNING") as cm:
                jobs = discover_story_job_files(d)

            self.assertEqual(jobs, [])
            self.assertFalse(bad.exists())
            self.assertTrue((d / "story_01.json.jail").exists())
            self.assertTrue((d / "story_01.json.jail.reason").exists())
            self.assertTrue(any("Unreadable JSON while discovering story jobs" in msg for msg in cm.output))

    def test_discover_story_job_files_warns_but_does_not_jail_non_job_like_json(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            bad = d / "foo.json"
            bad.write_text("{not-json", encoding="utf-8")

            with self.assertLogs("newsroom.runner", level="WARNING") as cm:
                jobs = discover_story_job_files(d)

            self.assertEqual(jobs, [])
            self.assertTrue(bad.exists())
            self.assertFalse((d / "foo.json.jail").exists())
            self.assertTrue(any("Unreadable JSON while discovering story jobs" in msg for msg in cm.output))

    def test_brave_fetch_pacing_is_per_key_id(self) -> None:
        root = Path(__file__).resolve().parents[2]
        runner = NewsroomRunner(
            openclaw_home=root,
            gateway=_DummyGateway(),  # type: ignore[arg-type]
            prompt_registry=PromptRegistry(openclaw_home=root),
            dry_run=True,
            lock_ttl_seconds=3600,
            log_root=root / "logs" / "newsroom-test",
        )

        key1 = BraveApiKey(key="k1", label="a")
        key2 = BraveApiKey(key="k2", label="b")

        expected_last: dict[str, float] = {"k1": 0.0, "k2": 0.0}
        new_ts: dict[str, float] = {"k1": 111.0, "k2": 222.0}

        def _fake_fetch(  # type: ignore[no-untyped-def]
            *,
            api_key: str,
            q: str,
            last_request_ts: float = 0.0,
            **_kwargs,
        ):
            self.assertEqual(q, "test")
            self.assertAlmostEqual(float(last_request_ts), expected_last[api_key], places=6)
            expected_last[api_key] = new_ts[api_key]
            fetch = SimpleNamespace(results=[], rate_limit=None)
            return fetch, new_ts[api_key]

        with patch("newsroom.runner.fetch_brave_news", side_effect=_fake_fetch):
            runner._fetch_brave_news_paced(key=key1, q="test")  # type: ignore[attr-defined]
            runner._fetch_brave_news_paced(key=key2, q="test")  # type: ignore[attr-defined]
            runner._fetch_brave_news_paced(key=key1, q="test")  # type: ignore[attr-defined]

    def test_discover_jobs_under_jails_unreadable_job_like_json(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            jobs_root = Path(td)

            # Create a corrupt job directly under jobs_root (not under a run dir).
            bad = jobs_root / "story_01.json"
            bad.write_text("{not-json", encoding="utf-8")

            with self.assertLogs("newsroom.runner", level="WARNING") as cm:
                jobs = discover_jobs_under(jobs_root)

            self.assertEqual(jobs, [])
            self.assertFalse(bad.exists())
            self.assertTrue((jobs_root / "story_01.json.jail").exists())
            self.assertTrue(any("Unreadable JSON while discovering story jobs" in msg for msg in cm.output))

