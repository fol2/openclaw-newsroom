from __future__ import annotations

import types
from datetime import UTC, datetime, timedelta
from pathlib import Path

from newsroom.runner import JsonlLogger, NewsroomRunner


def _make_runner(tmp_path: Path, *, tool_response: dict) -> NewsroomRunner:
    runner: NewsroomRunner = NewsroomRunner.__new__(NewsroomRunner)
    runner._dry_run = False  # type: ignore[attr-defined]

    def _fake_tool(
        self: NewsroomRunner,
        *,
        tool: str,
        action: str | None,
        args: dict,
        run_log: JsonlLogger,
        story_log: JsonlLogger,
        **_: object,
    ) -> dict:
        assert tool == "message"
        assert action == "read"
        return dict(tool_response)

    runner._tool_invoke_retry = types.MethodType(_fake_tool, runner)  # type: ignore[method-assign]
    return runner


def test_recent_discord_titles_parses_gateway_invoke_shape(tmp_path: Path) -> None:
    now = datetime.now(tz=UTC)
    within = (now - timedelta(minutes=5)).isoformat()
    old = (now - timedelta(hours=48)).isoformat()

    resp = {
        "ok": True,
        "result": {
            "content": [],
            "details": {
                "ok": True,
                "messages": [
                    {"id": "1", "content": "  Hello   world  ", "timestamp": within},
                    {"id": "2", "content": "Too old", "timestamp": old},
                ],
            },
        },
    }
    runner = _make_runner(tmp_path, tool_response=resp)
    titles = runner._recent_discord_titles(
        channel_id="1467628391082496041",
        hours=24,
        limit=60,
        run_log=JsonlLogger(tmp_path / "run.jsonl"),
        story_log=JsonlLogger(tmp_path / "story.jsonl"),
    )
    assert titles == ["Hello world"]


def test_recent_discord_titles_parses_invoke_result_json_shape(tmp_path: Path) -> None:
    now = datetime.now(tz=UTC)
    within = (now - timedelta(minutes=5)).isoformat()

    resp = {
        "ok": True,
        "messages": [
            {"id": "1", "content": "A title", "timestamp": within},
        ],
    }
    runner = _make_runner(tmp_path, tool_response=resp)
    titles = runner._recent_discord_titles(
        channel_id="1467628391082496041",
        hours=24,
        limit=60,
        run_log=JsonlLogger(tmp_path / "run.jsonl"),
        story_log=JsonlLogger(tmp_path / "story.jsonl"),
    )
    assert titles == ["A title"]

