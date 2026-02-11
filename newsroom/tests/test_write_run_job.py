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
                        "primary_url": f"https://reuters.com/example-{i}",
                        "supporting_urls": [f"https://theguardian.com/example-{i}", f"https://cnn.com/example-{i}"],
                        "age_minutes": 30 * i,
                        "is_breaking_2h": i == 1,
                        "is_developing_6h": i <= 3,
                        "cluster_size": 3,
                        "domains": ["reuters.com", "theguardian.com", "cnn.com"],
                        "unique_domain_count": 3,
                        "domain_tier_counts": {"tier_1": 1, "tier_2": 2},
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
                    "primary_url": "https://reuters.com/example-1",
                    "supporting_urls": ["https://cnn.com/example-1a"],
                    "age_minutes": 30,
                    "cluster_size": 1,
                    "domains": ["reuters.com", "cnn.com"],
                    "unique_domain_count": 2,
                    "domain_tier_counts": {"tier_1": 1, "tier_2": 1},
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
                    "supporting_urls": ["https://theguardian.com/pm-crisis"],
                    "age_minutes": 60,
                    "cluster_size": 3,
                    "link_count": 3,
                    "domains": ["bbc.co.uk", "theguardian.com"],
                    "unique_domain_count": 2,
                    "domain_tier_counts": {"tier_1": 1, "tier_2": 1},
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
                    "primary_url": "https://cnn.com/ai-model",
                    "supporting_urls": ["https://npr.org/ai-model"],
                    "age_minutes": 120,
                    "cluster_size": 2,
                    "link_count": 2,
                    "domains": ["cnn.com", "npr.org"],
                    "unique_domain_count": 2,
                    "domain_tier_counts": {"tier_2": 2},
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

    def test_write_run_job_quality_gate_accepts_two_mid_tier_domains(self) -> None:
        root = Path(__file__).resolve().parents[2]

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            jobs_root = tmp / "jobs"
            jobs_root.mkdir(parents=True, exist_ok=True)

            inputs_obj = {
                "ok": True,
                "candidates": [
                    {
                        "i": 1,
                        "event_key": "event:mid-tier-pass",
                        "anchor_key": "event:mid-tier-pass",
                        "suggested_category": "Global News",
                        "title": "Mid-tier corroboration story",
                        "description": "Developing story",
                        "lang_hint": "en",
                        "primary_url": "https://cnn.com/mid-tier-pass",
                        "supporting_urls": ["https://npr.org/mid-tier-pass"],
                        "domains": ["cnn.com", "npr.org"],
                        "unique_domain_count": 2,
                        "domain_tier_counts": {"tier_2": 2},
                        "age_minutes": 45,
                        "cluster_terms": [],
                    }
                ],
                "candidate_count": 1,
            }
            inputs_path = tmp / "inputs.json"
            inputs_path.write_text(json.dumps(inputs_obj) + "\n", encoding="utf-8")

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
                "2026-02-09 07:00",
                "--trigger",
                "cron_daily",
                "--expected-stories",
                "1",
                "--pick",
                "1",
            ]

            out = subprocess.check_output(cmd, text=True)
            summary = json.loads(out)
            self.assertEqual(summary["story_count"], 1)
            self.assertEqual(summary["held_count"], 0)

            run_dir = Path(summary["run_dir"])
            story_files = sorted(run_dir.glob("story_*.json"))
            self.assertEqual(len(story_files), 1)

    def test_write_run_job_holds_daily_mail_single_source_and_logs_reason(self) -> None:
        root = Path(__file__).resolve().parents[2]

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            jobs_root = tmp / "jobs"
            jobs_root.mkdir(parents=True, exist_ok=True)

            inputs_obj = {
                "ok": True,
                "candidates": [
                    {
                        "i": 1,
                        "event_key": "event:daily-mail-only",
                        "anchor_key": "event:daily-mail-only",
                        "suggested_category": "Global News",
                        "title": "Single source tabloid recap",
                        "description": "Daily Mail only",
                        "lang_hint": "en",
                        "primary_url": "https://dailymail.co.uk/single-source",
                        "supporting_urls": [],
                        "domains": ["dailymail.co.uk"],
                        "unique_domain_count": 1,
                        "domain_tier_counts": {"tier_3": 1},
                        "age_minutes": 30,
                        "cluster_terms": [],
                    },
                    {
                        "i": 2,
                        "event_key": "event:trusted-pass",
                        "anchor_key": "event:trusted-pass",
                        "suggested_category": "Global News",
                        "title": "Trusted multi-domain report",
                        "description": "Strong corroboration",
                        "lang_hint": "en",
                        "primary_url": "https://reuters.com/trusted-pass",
                        "supporting_urls": ["https://theguardian.com/trusted-pass"],
                        "domains": ["reuters.com", "theguardian.com"],
                        "unique_domain_count": 2,
                        "domain_tier_counts": {"tier_1": 1, "tier_2": 1},
                        "age_minutes": 25,
                        "cluster_terms": [],
                    },
                ],
                "candidate_count": 2,
            }
            inputs_path = tmp / "inputs.json"
            inputs_path.write_text(json.dumps(inputs_obj) + "\n", encoding="utf-8")

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
                "2026-02-09 08:00",
                "--trigger",
                "cron_daily",
                "--expected-stories",
                "2",
                "--pick",
                "1",
                "--pick",
                "2",
            ]

            proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
            summary = json.loads(proc.stdout)
            self.assertEqual(summary["story_count"], 1)
            self.assertEqual(summary["held_count"], 1)
            self.assertIn("held_candidates", summary)
            self.assertIn("holds_log_path", summary)
            self.assertIn("HOLD pick=1", proc.stderr)

            held = summary["held_candidates"][0]
            self.assertEqual(held["pick"], 1)
            self.assertIn("source_quality_gate_failed", held["reason"])
            self.assertIn("dailymail.co.uk", held["reason"])

            holds_log_path = Path(summary["holds_log_path"])
            self.assertTrue(holds_log_path.exists())
            log_lines = [line for line in holds_log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(len(log_lines), 1)
            log_entry = json.loads(log_lines[0])
            self.assertEqual(log_entry["pick"], 1)
            self.assertIn("source_quality_gate_failed", log_entry["reason"])

            run_dir = Path(summary["run_dir"])
            story_files = sorted(run_dir.glob("story_*.json"))
            self.assertEqual(len(story_files), 1)
            story = json.loads(story_files[0].read_text(encoding="utf-8"))
            self.assertIn("reuters.com/trusted-pass", story["story"]["primary_url"])


if __name__ == "__main__":
    unittest.main()
