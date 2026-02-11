import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import jsonschema


class TestWriteRunJobScript(unittest.TestCase):
    def test_write_run_job_creates_valid_files(self) -> None:
        root = Path(__file__).resolve().parents[2]
        run_schema = json.loads((root / "newsroom" / "schemas" / "run_job_v1.schema.json").read_text(encoding="utf-8"))
        story_schema = json.loads((root / "newsroom" / "schemas" / "story_job_v1.schema.json").read_text(encoding="utf-8"))

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            jobs_root = tmp / "jobs"
            jobs_root.mkdir(parents=True, exist_ok=True)

            inputs_path = tmp / "inputs.json"
            candidates = []
            for i in range(1, 7):
                candidates.append(
                    {
                        "i": i,
                        "event_key": f"event:{i:02d}abc",
                        "anchor_key": f"anchor:{i:02d}def",
                        "suggested_category": "AI" if i % 2 == 0 else "Global News",
                        "title": f"Example Title {i}",
                        "description": f"Example description {i}.",
                        "lang_hint": "en" if i % 2 == 0 else "zh",
                        "primary_url": f"https://example.com/{i}",
                        "supporting_urls": [f"https://example.org/{i}a", f"https://example.net/{i}b"],
                        "age_minutes": 30 * i,
                        "is_breaking_2h": i == 1,
                        "is_developing_6h": i <= 3,
                        "cluster_size": 3,
                        "domains": ["example.com", "example.org", "example.net"],
                        "cluster_terms": ["foo", "bar"],
                    }
                )
            inputs_obj = {"index": {"candidates": candidates}}
            inputs_path.write_text(json.dumps(inputs_obj, ensure_ascii=False) + "\n", encoding="utf-8")

            cmd = [
                sys.executable,
                str(root / "scripts" / "newsroom_write_run_job.py"),
                "--jobs-root",
                str(jobs_root),
                "--inputs-json-path",
                str(inputs_path),
                "--channel-id",
                "1467628391082496041",
                "--run-time-uk",
                "2026-02-04 07:00 AM",
                "--trigger",
                "cron_daily",
                "--timeout-seconds",
                "1234",
                "--pick",
                "1",
                "--pick",
                "2",
                "--pick",
                "3",
                "--pick",
                "4",
                "--pick",
                "5",
            ]

            out = subprocess.check_output(cmd, text=True)
            summary = json.loads(out)
            self.assertTrue(summary.get("ok"))

            run_dir = Path(summary["run_dir"])
            self.assertTrue(run_dir.exists())

            run_json = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
            jsonschema.validate(instance=run_json, schema=run_schema)
            self.assertEqual(run_json["runner"]["default_timeout_seconds"], 1234)

            story_files = sorted(run_dir.glob("story_*.json"))
            self.assertEqual(len(story_files), 5)
            for p in story_files:
                story = json.loads(p.read_text(encoding="utf-8"))
                jsonschema.validate(instance=story, schema=story_schema)
                self.assertEqual(story["run"]["run_id"], run_dir.name)
                self.assertEqual(story["run"]["trigger"], "cron_daily")
                self.assertEqual(story["destination"]["title_channel_id"], "1467628391082496041")
                self.assertEqual(story["monitor"]["timeout_seconds"], 1234)
                self.assertIn(story["story"]["lang_hint"], {"en", "zh"})
                self.assertEqual(story["spawn"]["input_mapping"]["lang_hint"], "$.story.lang_hint")


    def test_write_run_job_timeout_seconds_has_sane_minimum(self) -> None:
        """timeout_seconds should be clamped to a sensible minimum for story monitor settings."""
        root = Path(__file__).resolve().parents[2]
        story_schema = json.loads((root / "newsroom" / "schemas" / "story_job_v1.schema.json").read_text(encoding="utf-8"))

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            jobs_root = tmp / "jobs"
            jobs_root.mkdir(parents=True, exist_ok=True)

            inputs_path = tmp / "inputs.json"
            candidates = [
                {
                    "i": 1,
                    "event_key": "event:01abc",
                    "anchor_key": "anchor:01def",
                    "suggested_category": "Global News",
                    "title": "Example Title 1",
                    "description": "Example description 1.",
                    "lang_hint": "en",
                    "primary_url": "https://example.com/1",
                    "supporting_urls": ["https://example.org/1a"],
                    "age_minutes": 30,
                    "cluster_size": 1,
                    "domains": ["example.com", "example.org"],
                    "cluster_terms": ["foo"],
                }
            ]
            inputs_obj = {"index": {"candidates": candidates}}
            inputs_path.write_text(json.dumps(inputs_obj, ensure_ascii=False) + "\n", encoding="utf-8")

            cmd = [
                sys.executable,
                str(root / "scripts" / "newsroom_write_run_job.py"),
                "--jobs-root",
                str(jobs_root),
                "--inputs-json-path",
                str(inputs_path),
                "--channel-id",
                "1467628391082496041",
                "--run-time-uk",
                "2026-02-04 07:01 AM",
                "--trigger",
                "cron_daily",
                "--expected-stories",
                "1",
                "--timeout-seconds",
                "10",
                "--pick",
                "1",
            ]

            out = subprocess.check_output(cmd, text=True)
            summary = json.loads(out)
            run_dir = Path(summary["run_dir"])
            story_files = sorted(run_dir.glob("story_*.json"))
            self.assertEqual(len(story_files), 1)

            story = json.loads(story_files[0].read_text(encoding="utf-8"))
            jsonschema.validate(instance=story, schema=story_schema)
            self.assertEqual(story["monitor"]["timeout_seconds"], 60)
            self.assertEqual(story["story"]["lang_hint"], "en")


    def test_write_run_job_v5_top_level_candidates(self) -> None:
        """New v5 format: candidates at top level + event_id propagation."""
        root = Path(__file__).resolve().parents[2]
        story_schema = json.loads((root / "newsroom" / "schemas" / "story_job_v1.schema.json").read_text(encoding="utf-8"))

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            jobs_root = tmp / "jobs"
            jobs_root.mkdir(parents=True, exist_ok=True)

            # New v5 format: top-level candidates[].
            candidates = [
                {
                    "i": 1,
                    "id": 42,
                    "event_id": 42,
                    "event_key": "event:42",
                    "semantic_event_key": "event:42",
                    "anchor_key": "event:42",
                    "suggested_category": "Politics",
                    "category": "Politics",
                    "title": "PM crisis deepens",
                    "description": "UK PM faces renewed pressure",
                    "lang_hint": "en",
                    "summary_en": "UK PM faces renewed pressure",
                    "primary_url": "https://bbc.co.uk/pm-crisis",
                    "supporting_urls": ["https://guardian.com/pm-crisis"],
                    "age_minutes": 60,
                    "cluster_size": 3,
                    "link_count": 3,
                    "domains": ["bbc.co.uk", "guardian.com"],
                    "cluster_terms": [],
                    "anchor_terms": [],
                    "suggest_flags": [],
                },
                {
                    "i": 2,
                    "id": 43,
                    "event_id": 43,
                    "event_key": "event:43",
                    "semantic_event_key": "event:43",
                    "anchor_key": "event:43",
                    "suggested_category": "AI",
                    "category": "AI",
                    "title": "New AI model released",
                    "description": "Company releases AI model",
                    "lang_hint": "mixed",
                    "summary_en": "Company releases AI model",
                    "primary_url": "https://techcrunch.com/ai-model",
                    "supporting_urls": [],
                    "age_minutes": 120,
                    "cluster_size": 2,
                    "link_count": 2,
                    "domains": ["techcrunch.com"],
                    "cluster_terms": [],
                    "anchor_terms": [],
                    "suggest_flags": [],
                },
            ]
            inputs_obj = {"ok": True, "candidates": candidates, "candidate_count": 2}
            inputs_path = tmp / "inputs.json"
            inputs_path.write_text(json.dumps(inputs_obj) + "\n", encoding="utf-8")

            cmd = [
                sys.executable,
                str(root / "scripts" / "newsroom_write_run_job.py"),
                "--jobs-root", str(jobs_root),
                "--inputs-json-path", str(inputs_path),
                "--channel-id", "1467628391082496041",
                "--run-time-uk", "2026-02-08 07:00",
                "--trigger", "cron_daily",
                "--expected-stories", "2",
                "--pick", "1",
                "--pick", "2",
            ]

            out = subprocess.check_output(cmd, text=True)
            summary = json.loads(out)
            self.assertTrue(summary.get("ok"))

            run_dir = Path(summary["run_dir"])
            story_files = sorted(run_dir.glob("story_*.json"))
            self.assertEqual(len(story_files), 2)

            # Verify event_id is propagated into story jobs.
            s1 = json.loads(story_files[0].read_text(encoding="utf-8"))
            s2 = json.loads(story_files[1].read_text(encoding="utf-8"))
            self.assertEqual(s1["story"]["event_id"], 42)
            self.assertEqual(s2["story"]["event_id"], 43)

            # Verify category and title.
            self.assertEqual(s1["story"]["category"], "Politics")
            self.assertEqual(s2["story"]["category"], "AI")
            self.assertEqual(s1["story"]["lang_hint"], "en")
            self.assertEqual(s2["story"]["lang_hint"], "mixed")


if __name__ == "__main__":
    unittest.main()
