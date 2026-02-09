#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
import sys
from pathlib import Path

# Ensure the repo root (OpenClaw home) is on sys.path even when executed from the agent workspace.
OPENCLAW_HOME = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OPENCLAW_HOME))

from newsroom.gateway_client import GatewayClient, load_gateway_config  # noqa: E402
from newsroom.runner import (  # noqa: E402
    NewsroomRunner,
    PromptRegistry,
    discover_jobs_under,
    discover_story_job_files,
)


def _openclaw_home_from_this_file() -> Path:
    # scripts/ lives directly under the OpenClaw home dir in this repo layout.
    return OPENCLAW_HOME


def _summarize_group(job_paths: list[Path]) -> dict[str, object]:
    rows = []
    for p in sorted(job_paths):
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        story = obj.get("story", {}) or {}
        state = obj.get("state", {}) or {}
        result = obj.get("result", {}) or {}
        discord_state = state.get("discord", {}) or {}
        rows.append(
            {
                "path": str(p),
                "story_id": story.get("story_id"),
                "title": story.get("title"),
                "status": state.get("status"),
                "final_status": result.get("final_status"),
                "thread_id": discord_state.get("thread_id"),
            }
        )
    return {"stories": rows}


def _collect_dm_targets(run_dir: Path, job_paths: list[Path]) -> list[str]:
    targets: set[str] = set()

    run_json = run_dir / "run.json"
    if run_json.exists():
        try:
            run_obj = json.loads(run_json.read_text(encoding="utf-8"))
            runner_cfg = run_obj.get("runner", {}) or {}
            for t in runner_cfg.get("dm_targets", []) or []:
                if isinstance(t, str) and t.strip():
                    targets.add(t.strip())
        except Exception:
            pass

    for p in job_paths:
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        post = obj.get("post", {}) or {}
        result = obj.get("result", {}) or {}
        final_status = result.get("final_status")
        if final_status in ("SUCCESS", "RESCUED"):
            for t in post.get("on_success_dm_targets", []) or []:
                if isinstance(t, str) and t.strip():
                    targets.add(t.strip())
        elif final_status == "FAILURE":
            for t in post.get("on_failure_dm_targets", []) or []:
                if isinstance(t, str) and t.strip():
                    targets.add(t.strip())

    return sorted(targets)


def _format_dm_summary(run_dir: Path, job_paths: list[Path]) -> str:
    rows = _summarize_group(job_paths).get("stories", [])
    # Derive run_id from the first story job, best-effort.
    run_id = None
    try:
        obj0 = json.loads(job_paths[0].read_text(encoding="utf-8"))
        run_id = (obj0.get("run", {}) or {}).get("run_id")
    except Exception:
        run_id = None

    counts = {"SUCCESS": 0, "RESCUED": 0, "FAILURE": 0, "OTHER": 0}
    for r in rows:  # type: ignore[assignment]
        fs = r.get("final_status")
        if fs in counts:
            counts[fs] += 1
        else:
            counts["OTHER"] += 1

    lines = []
    lines.append(f"Newsroom run complete: {run_id or run_dir.name}")
    lines.append(f"Total: {len(rows)}  Success: {counts['SUCCESS']}  Rescued: {counts['RESCUED']}  Failed: {counts['FAILURE']}")
    lines.append("")
    for r in rows:  # type: ignore[assignment]
        status = r.get("final_status") or r.get("status") or "UNKNOWN"
        story_id = r.get("story_id") or "unknown"
        title = r.get("title") or ""
        thread_id = r.get("thread_id")
        if thread_id:
            lines.append(f"[{status}] {story_id} — {title} (thread_id={thread_id})")
        else:
            lines.append(f"[{status}] {story_id} — {title}")
    return "\n".join(lines).strip() + "\n"


def _write_run_summary(
    run_dir: Path,
    *,
    job_paths: list[Path],
    log_root: Path,
    summary: dict[str, object],
    detail: dict[str, object],
) -> None:
    # This artifact is the primary “status record” when the runner is triggered in the background
    # from cron (so there is no cron polling / status-check job).
    out: dict[str, object] = {
        "ok": True,
        "generated_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "log_root": str(log_root),
        "summary": summary,
        "detail": detail,
    }

    payload = json.dumps(out, indent=2, ensure_ascii=True) + "\n"

    # 1) Write inside the run directory (easy to discover from planner output).
    (run_dir / "run_summary.json").write_text(payload, encoding="utf-8")

    # 2) Also write under logs/newsroom/<run_id>/ for consistency with JSONL logs.
    run_id = None
    try:
        obj0 = json.loads(job_paths[0].read_text(encoding="utf-8"))
        run_id = (obj0.get("run", {}) or {}).get("run_id")
    except Exception:
        run_id = None

    if isinstance(run_id, str) and run_id.strip():
        log_dir = log_root / run_id.strip()
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / "run_summary.json").write_text(payload, encoding="utf-8")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="OpenClaw Newsroom deterministic runner (Discord).")
    parser.add_argument("--jobs-root", default=None, help="Jobs root directory (default: <openclaw_home>/workspace/jobs)")
    parser.add_argument("--path", action="append", default=[], help="A job file or run directory to process (repeatable).")
    parser.add_argument("--dry-run", action="store_true", help="Validate + render prompts, but do not post/spawn or modify job files.")
    parser.add_argument("--summary-only", action="store_true", help="Print only per-run summary (omit per-story detail).")
    parser.add_argument("--session-key", default="hook:newsroom-runner", help="Session key used for tools.invoke calls.")
    parser.add_argument("--lock-ttl-seconds", type=int, default=3600, help="Treat locks older than this as stale.")
    parser.add_argument("--log-root", default=None, help="Log output root (default: <openclaw_home>/logs/newsroom)")
    args = parser.parse_args(argv)

    openclaw_home = _openclaw_home_from_this_file()
    jobs_root = Path(args.jobs_root) if args.jobs_root else (openclaw_home / "workspace" / "jobs")
    log_root = Path(args.log_root) if args.log_root else (openclaw_home / "logs" / "newsroom")

    gateway_cfg = load_gateway_config(openclaw_home)
    gateway = GatewayClient(http_url=gateway_cfg.http_url, token=gateway_cfg.token, default_session_key=args.session_key)
    registry = PromptRegistry(openclaw_home=openclaw_home)

    runner = NewsroomRunner(
        openclaw_home=openclaw_home,
        gateway=gateway,
        prompt_registry=registry,
        dry_run=args.dry_run,
        lock_ttl_seconds=int(args.lock_ttl_seconds),
        log_root=log_root,
    )

    # Group jobs by run directory (so run.json defaults apply consistently).
    groups: dict[Path, list[Path]] = {}

    if args.path:
        for raw in args.path:
            p = Path(raw)
            if p.is_dir():
                jobs = discover_story_job_files(p)
                if jobs:
                    groups.setdefault(p, []).extend(jobs)
            else:
                groups.setdefault(p.parent, []).append(p)
    else:
        for jp in discover_jobs_under(jobs_root):
            groups.setdefault(jp.parent, []).append(jp)

    if not groups:
        print(json.dumps({"ok": True, "message": "no_jobs"}))
        return 0

    summaries = []
    for run_dir, job_paths in sorted(groups.items(), key=lambda kv: str(kv[0])):
        unique_jobs = sorted(set(job_paths))
        run_json = run_dir / "run.json"
        summary = runner.run_group(job_paths=unique_jobs, run_json_path=run_json if run_json.exists() else None)
        detail = _summarize_group(unique_jobs)
        dm_targets = _collect_dm_targets(run_dir, unique_jobs)
        if dm_targets:
            runner.send_dm_summary(dm_targets=dm_targets, summary=_format_dm_summary(run_dir, unique_jobs))

        # Always persist a status artifact so users can check progress without cron polling.
        try:
            status = {
                "run": summary.get("run"),
                "total": summary.get("total"),
                "completed": summary.get("completed"),
                "failed": summary.get("failed"),
                "dry_run": summary.get("dry_run"),
            }
            _write_run_summary(run_dir, job_paths=unique_jobs, log_root=log_root, summary=status, detail=detail)
        except Exception:
            # Status file failure must never crash a run.
            pass

        row = {"run_dir": str(run_dir), "summary": summary}
        if not args.summary_only:
            row["detail"] = detail
        summaries.append(row)

    print(json.dumps({"ok": True, "runs": summaries}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
