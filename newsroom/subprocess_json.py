"""Helpers for running JSON-emitting subprocesses safely.

These inputs scripts are invoked from cron-driven LLM planners. If a child
process hangs (network stall, deadlock, etc.), the planner turn can be
blocked until the cron timeout. To avoid that, we always run subprocesses
with a hard timeout and return structured error objects with truncated
stdout/stderr for debugging.
"""

from __future__ import annotations

import json
import subprocess
from typing import Any


def _as_text(v: str | bytes | None) -> str:
    if v is None:
        return ""
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace")
    return v


def _truncate(s: str | bytes | None, max_chars: int) -> str:
    text = _as_text(s)
    if max_chars <= 0:
        return ""
    return text[:max_chars]


def run_json_command(
    cmd: list[str],
    *,
    timeout_seconds: float,
    max_output_chars: int = 2000,
) -> dict[str, Any]:
    """Run *cmd* and parse a JSON object from stdout.

    On failure, returns a structured object with `ok: false` and truncated
    stdout/stderr to keep planner payloads bounded.
    """
    try:
        proc = subprocess.run(
            cmd,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as e:
        return {
            "ok": False,
            "error": "timeout",
            "timeout_seconds": float(timeout_seconds),
            "stdout": _truncate(getattr(e, "stdout", None), max_output_chars),
            "stderr": _truncate(getattr(e, "stderr", None), max_output_chars),
            "cmd": cmd,
        }
    except Exception as e:
        return {
            "ok": False,
            "error": "spawn_failed",
            "detail": str(e),
            "stdout": "",
            "stderr": "",
            "cmd": cmd,
        }

    out = proc.stdout or ""
    err = proc.stderr or ""

    if proc.returncode != 0:
        return {
            "ok": False,
            "error": "cmd_failed",
            "returncode": int(proc.returncode),
            "stdout": _truncate(out, max_output_chars),
            "stderr": _truncate(err, max_output_chars),
            "cmd": cmd,
        }

    try:
        obj = json.loads(out)
    except Exception as e:
        return {
            "ok": False,
            "error": "invalid_json",
            "detail": str(e),
            "stdout": _truncate(out, max_output_chars),
            "stderr": _truncate(err, max_output_chars),
            "cmd": cmd,
        }

    if isinstance(obj, dict):
        return obj

    return {
        "ok": False,
        "error": "json_not_object",
        "stdout": _truncate(out, max_output_chars),
        "stderr": _truncate(err, max_output_chars),
        "cmd": cmd,
    }

