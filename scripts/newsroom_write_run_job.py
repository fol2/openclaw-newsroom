#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from shutil import which
from typing import Any

OPENCLAW_HOME = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OPENCLAW_HOME))

from newsroom.brave_news import normalize_url  # noqa: E402
from newsroom.job_store import atomic_write_json  # noqa: E402
from newsroom.prompt_policy import prompt_id_for_category  # noqa: E402


_TRIGGER_ALIASES = {"manual_run": "manual"}
_VALID_TRIGGERS = {"cron_hourly", "cron_daily", "manual"}


def _is_snowflake(v: str) -> bool:
    s = (v or "").strip()
    return s.isdigit() and len(s) >= 15


def _run_dir_for_multi(*, jobs_root: Path, run_time_uk: str) -> tuple[str, Path]:
    s = run_time_uk.strip()
    dt = None
    # Cron sometimes supplies seconds (HH:MM:SS); accept both.
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %I:%M %p", "%Y-%m-%d %I:%M:%S %p"):
        try:
            dt = datetime.strptime(s, fmt)
            break
        except ValueError:
            continue
    if dt is None:
        raise ValueError(
            "run-time-uk must be 'YYYY-MM-DD HH:MM', 'YYYY-MM-DD HH:MM:SS', "
            f"or 'YYYY-MM-DD HH:MM AM/PM', got: {run_time_uk!r}"
        )
    folder = f"discord-multi-{dt.strftime('%Y-%m-%d-%H-%M')}"
    return folder, jobs_root / folder


def _as_int(v: Any) -> int | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, str) and v.strip().isdigit():
        try:
            return int(v.strip())
        except ValueError:
            return None
    return None


def _clean_urls(urls: list[str], *, primary_url: str | None = None, max_urls: int = 4) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    primary = str(primary_url or "").strip()
    if primary:
        seen.add(primary)
    for u in urls:
        u = str(u or "").strip()
        if not u or not u.startswith("http"):
            continue
        if u in seen:
            continue
        seen.add(u)
        out.append(u)
        if len(out) >= int(max_urls):
            break
    return out


def _norm_dedupe_key(v: Any) -> str | None:
    if not isinstance(v, str) or not v.strip():
        return None
    s = v.strip()
    if s.startswith(("event:", "anchor:")):
        return s
    if s.startswith("http"):
        try:
            return normalize_url(s)
        except Exception:
            return s
    return s


def _concrete_anchor_from_candidate(c: dict[str, Any]) -> str:
    desc = c.get("description")
    if isinstance(desc, str) and desc.strip():
        return desc.strip()[:220]
    title = c.get("title")
    if isinstance(title, str) and title.strip():
        return title.strip()[:220]
    terms = c.get("cluster_terms")
    if isinstance(terms, list):
        words = [str(t).strip() for t in terms if isinstance(t, str) and str(t).strip()]
        if words:
            return ("Key terms: " + ", ".join(words[:8]))[:220]
    return "See linked sources for details."


def _candidate_by_index(inputs_obj: dict[str, Any]) -> dict[int, dict[str, Any]]:
    idx: dict[int, dict[str, Any]] = {}
    # New format (v5): top-level candidates[].
    cands = inputs_obj.get("candidates", [])
    # Old format fallback: index.candidates[].
    if not cands:
        cands = ((inputs_obj.get("index") or {}) if isinstance(inputs_obj.get("index"), dict) else {}).get("candidates", [])
    if not isinstance(cands, list):
        return idx
    for c in cands:
        if not isinstance(c, dict):
            continue
        i = _as_int(c.get("i"))
        if i is None:
            continue
        idx[int(i)] = c
    return idx


def _launch_runner_background(*, openclaw_home: Path, run_dir: Path) -> int:
    runner = openclaw_home / "scripts" / "newsroom_runner.py"
    if not runner.exists():
        raise SystemExit(f"Runner script not found: {runner}")

    uv_bin = which("uv")
    if not uv_bin:
        raise SystemExit("uv not found in PATH")

    # Detach so planners/crons can exit immediately.
    out_path = run_dir / "runner_background.out.log"
    err_path = run_dir / "runner_background.err.log"
    out_f = out_path.open("a", encoding="utf-8")
    err_f = err_path.open("a", encoding="utf-8")
    try:
        proc = subprocess.Popen(
            [
                str(uv_bin),
                "run",
                "python3",
                str(runner),
                "--path",
                str(run_dir),
                "--summary-only",
            ],
            cwd=str(openclaw_home),
            stdout=out_f,
            stderr=err_f,
            start_new_session=True,
        )
    finally:
        out_f.close()
        err_f.close()
    return int(proc.pid)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Create a multi-story run folder (run_job_v1 + story_job_v1 files) for the newsroom runner.")
    parser.add_argument("--jobs-root", default=str(OPENCLAW_HOME / "workspace" / "jobs"), help="Jobs root directory (default: <openclaw_home>/workspace/jobs).")
    parser.add_argument("--inputs-json-path", default=str(OPENCLAW_HOME / "data" / "newsroom" / "daily_inputs_last.json"), help="Path to daily_inputs JSON (default: data/newsroom/daily_inputs_last.json).")
    parser.add_argument("--channel-id", default="1467628391082496041", help="Discord title channel id.")
    parser.add_argument("--run-time-uk", required=True, help="UK time in 'YYYY-MM-DD HH:MM' (used for folder naming).")
    parser.add_argument("--trigger", default="cron_daily", help="Job trigger (default: cron_daily).")
    parser.add_argument("--pick", action="append", default=[], help="Candidate index to include (repeatable, 1-based).")
    parser.add_argument("--expected-stories", type=int, default=5, help="Expected number of story files to write (default: 5).")
    parser.add_argument("--concurrency", type=int, default=3, help="Runner concurrency for this run (default: 3).")
    parser.add_argument("--stagger-seconds", type=int, default=60, help="Runner stagger seconds (default: 60).")
    parser.add_argument("--timeout-seconds", type=int, default=900, help="Default timeout seconds per story (default: 900).")
    parser.add_argument("--dm-target", action="append", default=[], help="Runner dm_targets entry (repeatable).")
    parser.add_argument("--stop-run-on-failure", action="store_true", help="Stop spawning new stories on first failure.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing run folder if present.")
    parser.add_argument("--launch-runner", action="store_true", help="Launch the deterministic runner in the background after writing files.")
    args = parser.parse_args(argv)

    channel_id = str(args.channel_id).strip()
    if not _is_snowflake(channel_id):
        raise SystemExit("channel-id must be a numeric Discord snowflake")

    run_time_uk = str(args.run_time_uk).strip()
    jobs_root = Path(str(args.jobs_root)).expanduser()
    folder, run_dir = _run_dir_for_multi(jobs_root=jobs_root, run_time_uk=run_time_uk)

    if run_dir.exists() and not bool(args.overwrite):
        raise SystemExit(f"Refusing to overwrite existing run directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    inputs_path = Path(str(args.inputs_json_path)).expanduser()
    if not inputs_path.exists():
        raise SystemExit(f"inputs-json-path not found: {inputs_path}")
    inputs_obj = json.loads(inputs_path.read_text(encoding="utf-8"))
    if not isinstance(inputs_obj, dict):
        raise SystemExit("inputs-json-path must contain a JSON object")

    by_i = _candidate_by_index(inputs_obj)
    picks: list[int] = []
    for raw in list(args.pick or []):
        try:
            i = int(str(raw).strip())
        except Exception:
            raise SystemExit(f"Invalid --pick: {raw!r}") from None
        picks.append(i)

    expected = int(max(1, args.expected_stories))
    if len(picks) != expected:
        raise SystemExit(f"Expected exactly {expected} --pick values, got {len(picks)}")

    selected: list[dict[str, Any]] = []
    for i in picks:
        c = by_i.get(int(i))
        if not c:
            raise SystemExit(f"--pick {i} not found in candidates list")
        selected.append(c)

    # Enforce uniqueness within the run (avoid accidental double-picks).
    seen_keys: set[str] = set()
    for c in selected:
        dk = (
            _norm_dedupe_key(c.get("semantic_event_key"))
            or _norm_dedupe_key(c.get("event_key"))
            or _norm_dedupe_key(c.get("anchor_key"))
            or _norm_dedupe_key(c.get("primary_url"))
        )
        if dk:
            if dk in seen_keys:
                raise SystemExit(f"Duplicate selection within run (dedupe_key collision): {dk}")
            seen_keys.add(dk)

    raw_trigger = str(args.trigger).strip() or "cron_daily"
    trigger = _TRIGGER_ALIASES.get(raw_trigger, raw_trigger)
    if trigger not in _VALID_TRIGGERS:
        print(f"WARNING: unknown trigger {trigger!r}, defaulting to 'manual'", file=sys.stderr)
        trigger = "manual"

    run_job: dict[str, Any] = {
        "schema_version": "run_job_v1",
        "run": {"run_id": folder, "trigger": trigger, "run_time_uk": run_time_uk},
        "destination": {"platform": "discord"},
        "runner": {
            "concurrency": int(max(1, args.concurrency)),
            "stagger_seconds": int(max(0, args.stagger_seconds)),
            "default_timeout_seconds": int(max(60, args.timeout_seconds)),
            "dm_targets": [str(t).strip() for t in (args.dm_target or []) if isinstance(t, str) and str(t).strip()],
            "stop_run_on_failure": bool(args.stop_run_on_failure),
        },
    }
    atomic_write_json(run_dir / "run.json", run_job)

    stories_out: list[dict[str, Any]] = []
    for n, c in enumerate(selected, start=1):
        story_id = f"story_{n:02d}"

        category = str(c.get("suggested_category") or "Global News").strip() or "Global News"
        title = str(c.get("title") or "").strip() or "Untitled"
        primary_url = str(c.get("primary_url") or "").strip()
        if not primary_url.startswith("http"):
            raise SystemExit(f"Invalid primary_url for pick {picks[n-1]}: {primary_url!r}")

        supporting_raw = c.get("supporting_urls")
        supporting_list = supporting_raw if isinstance(supporting_raw, list) else []
        supporting_urls = _clean_urls([str(u) for u in supporting_list], primary_url=primary_url, max_urls=4)

        flags_raw = c.get("suggest_flags")
        flags = [str(f).strip() for f in (flags_raw or []) if isinstance(f, str) and str(f).strip()] if isinstance(flags_raw, list) else []
        # Auto-infer breaking/developing from event data when flags are empty
        if not flags:
            age = c.get("age_minutes")
            if c.get("parent_event_id"):
                flags.append("developing")
            if isinstance(age, (int, float)) and age <= 120:
                flags.append("breaking")
            elif isinstance(age, (int, float)) and age <= 360 and "developing" not in flags:
                flags.append("developing")

        dedupe_key = (
            _norm_dedupe_key(c.get("semantic_event_key"))
            or _norm_dedupe_key(c.get("event_key"))
            or _norm_dedupe_key(c.get("anchor_key"))
            or _norm_dedupe_key(primary_url)
            or primary_url
        )

        job: dict[str, Any] = {
            "schema_version": "story_job_v1",
            "run": {"run_id": folder, "trigger": trigger, "run_time_uk": run_time_uk},
            "story": {
                "story_id": story_id,
                "content_type": "news_deep_dive",
                "category": category,
                "title": title,
                "primary_url": primary_url,
                "supporting_urls": supporting_urls,
                "concrete_anchor": _concrete_anchor_from_candidate(c),
                "flags": flags,
                "dedupe_key": dedupe_key,
                # Stable cluster identifiers (best-effort; used for audit + future dedupe improvements).
                "event_key": c.get("event_key"),
                "semantic_event_key": c.get("semantic_event_key"),
                "anchor_key": c.get("anchor_key"),
                "anchor_terms": c.get("anchor_terms"),
                "cluster_terms": c.get("cluster_terms"),
                "cluster_size": c.get("cluster_size"),
                "age_minutes": c.get("age_minutes"),
                "domains": c.get("domains"),
                # Event-centric fields (v5): used by runner to mark events as posted.
                "event_id": _as_int(c.get("event_id")),
            },
            "destination": {"platform": "discord", "title_channel_id": channel_id, "thread_name_template": "{title}"},
            "spawn": {
                "prompt_id": prompt_id_for_category(category),
                "agent_id": "main",
                "publisher_mode": "script_posts",
                "input_mapping": {
                    "story_id": "$.story.story_id",
                    "category": "$.story.category",
                    "title": "$.story.title",
                    "primary_url": "$.story.primary_url",
                    "supporting_urls": "$.story.supporting_urls",
                    "concrete_anchor": "$.story.concrete_anchor",
                    "flags": "$.story.flags",
                    "thread_id": "$.state.discord.thread_id",
                    "run_time_uk": "$.run.run_time_uk",
                },
            },
            "monitor": {"poll_seconds": 5, "timeout_seconds": 900, "result_json_required": True},
            "post": {"on_success_dm_targets": [], "on_failure_dm_targets": []},
            "recover": {"enabled": True, "rescue_prompt_id": "news_rescue_script_v1", "max_rescue_attempts": 1, "rescue_timeout_seconds": 600},
            "validation": {"validator_id": "news_reporter_script_v1", "stop_run_on_failure": False},
            "state": {
                "status": "PLANNED",
                "locked_by": None,
                "locked_at": None,
                "discord": {"title_message_id": None, "thread_id": None},
                "worker": {"attempt": 0, "child_session_key": None, "run_id": None, "started_at": None, "ended_at": None},
                "rescue": {"attempt": 0, "child_session_key": None, "started_at": None, "ended_at": None},
            },
            "result": {"final_status": None, "worker_result_json": None, "rescue_result_json": None, "errors": []},
        }

        json.dumps(job, ensure_ascii=False)
        story_path = run_dir / f"{story_id}.json"
        atomic_write_json(story_path, job)

        stories_out.append({"story_id": story_id, "path": str(story_path), "title": title, "primary_url": primary_url, "dedupe_key": dedupe_key})

    runner_pid = None
    if bool(args.launch_runner):
        runner_pid = _launch_runner_background(openclaw_home=OPENCLAW_HOME, run_dir=run_dir)

    out: dict[str, Any] = {"ok": True, "run_dir": str(run_dir), "run_json": str(run_dir / "run.json"), "stories": stories_out}
    if runner_pid is not None:
        out["runner_launched"] = True
        out["runner_pid"] = int(runner_pid)

    print(json.dumps(out, ensure_ascii=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
