from __future__ import annotations

import types
from pathlib import Path

from newsroom.runner import JsonlLogger, NewsroomRunner


def _make_runner(tmp_path: Path, *, og_path: str) -> tuple[NewsroomRunner, list[dict[str, object]]]:
    runner: NewsroomRunner = NewsroomRunner.__new__(NewsroomRunner)
    runner._dry_run = False  # type: ignore[attr-defined]
    runner._openclaw_home = tmp_path  # type: ignore[attr-defined]
    runner._log_root = tmp_path / "logs"  # type: ignore[attr-defined]

    # No-op job file writes for unit tests.
    def _noop_update_job_file(self: NewsroomRunner, job_path: Path, job: dict, story_log: JsonlLogger, *, event: str, **fields: object) -> None:
        return None

    runner._update_job_file = types.MethodType(_noop_update_job_file, runner)  # type: ignore[method-assign]

    # Avoid network work; rely on our injected assets.
    def _noop_assets_pack(self: NewsroomRunner, *, job_path: Path, job: dict, run_log: JsonlLogger, story_log: JsonlLogger) -> None:
        return None

    runner._ensure_assets_pack = types.MethodType(_noop_assets_pack, runner)  # type: ignore[method-assign]

    def _fake_og_paths(self: NewsroomRunner, *, job_path: Path, job: dict, run_log: JsonlLogger, story_log: JsonlLogger, max_downloads: int = 1) -> list[str]:
        return [og_path]

    runner._ensure_og_image_paths = types.MethodType(_fake_og_paths, runner)  # type: ignore[method-assign]

    calls: list[dict[str, object]] = []

    def _fake_tool(self: NewsroomRunner, *, tool: str, action: str | None, args: dict, run_log: JsonlLogger, story_log: JsonlLogger, **_: object) -> dict:
        calls.append({"tool": tool, "action": action, "args": dict(args)})
        mid = str(1460000000000000000 + len(calls))
        return {"result": {"details": {"messageId": mid}}}

    runner._tool_invoke_retry = types.MethodType(_fake_tool, runner)  # type: ignore[method-assign]
    return runner, calls


def test_script_posts_finance_attaches_chart_and_og(tmp_path: Path) -> None:
    chart = tmp_path / "chart.png"
    chart.write_bytes(b"fakepng")
    og = tmp_path / "og.jpg"
    og.write_bytes(b"fakejpg")

    runner, calls = _make_runner(tmp_path, og_path=str(og))

    job = {
        "destination": {"platform": "discord"},
        "story": {"category": "US Stocks"},
        "state": {
            "discord": {"thread_id": "1460000000000000001", "published_message_ids": []},
            "assets": {"attachments": {"chart_paths": [str(chart)], "og_image_paths": [str(og)], "og_image_urls": []}},
        },
    }
    body = "中" * 2000  # force 2 body chunks
    result_json = {"status": "SUCCESS", "draft": {"body": body, "read_more_urls": ["https://a.com/1", "https://b.com/2", "https://c.com/3"]}}

    out = runner._publish_script_posts_draft(
        job_path=tmp_path / "job.json",
        job=job,
        result_json=result_json,
        run_log=JsonlLogger(tmp_path / "run.jsonl"),
        story_log=JsonlLogger(tmp_path / "story.jsonl"),
    )

    # 2 body chunks + read more => 3 sends.
    assert len(calls) == 3
    assert calls[0]["args"].get("filePath") == str(chart)
    assert calls[1]["args"].get("filePath") == str(og)
    assert "filePath" not in calls[2]["args"]
    assert out["images_attached_count"] == 2


def test_script_posts_nonfinance_attaches_og_only(tmp_path: Path) -> None:
    og = tmp_path / "og.jpg"
    og.write_bytes(b"fakejpg")

    runner, calls = _make_runner(tmp_path, og_path=str(og))

    job = {
        "destination": {"platform": "discord"},
        "story": {"category": "Politics"},
        "state": {
            "discord": {"thread_id": "1460000000000000001", "published_message_ids": []},
            "assets": {"attachments": {"chart_paths": [], "og_image_paths": [str(og)], "og_image_urls": []}},
        },
    }
    body = "中" * 1200  # 1 body chunk
    result_json = {"status": "SUCCESS", "draft": {"body": body, "read_more_urls": ["https://a.com/1", "https://b.com/2", "https://c.com/3"]}}

    out = runner._publish_script_posts_draft(
        job_path=tmp_path / "job.json",
        job=job,
        result_json=result_json,
        run_log=JsonlLogger(tmp_path / "run.jsonl"),
        story_log=JsonlLogger(tmp_path / "story.jsonl"),
    )

    # 1 body chunk + read more => 2 sends.
    assert len(calls) == 2
    assert calls[0]["args"].get("filePath") == str(og)
    assert "filePath" not in calls[1]["args"]
    assert out["images_attached_count"] == 1


def test_script_posts_prefers_infographic_over_og(tmp_path: Path) -> None:
    chart = tmp_path / "chart.png"
    chart.write_bytes(b"fakepng")
    og = tmp_path / "og.jpg"
    og.write_bytes(b"fakejpg")
    info = tmp_path / "info.png"
    info.write_bytes(b"fakepng")

    runner, calls = _make_runner(tmp_path, og_path=str(og))

    job = {
        "destination": {"platform": "discord"},
        "story": {"category": "US Stocks"},
        "state": {
            "discord": {"thread_id": "1460000000000000001", "published_message_ids": []},
            "assets": {
                "attachments": {
                    "chart_paths": [str(chart)],
                    "og_image_paths": [str(og)],
                    "infographic_paths": [str(info)],
                    "og_image_urls": [],
                }
            },
        },
    }
    body = "中" * 2000  # force 2 body chunks
    result_json = {"status": "SUCCESS", "draft": {"body": body, "read_more_urls": ["https://a.com/1", "https://b.com/2", "https://c.com/3"]}}

    out = runner._publish_script_posts_draft(
        job_path=tmp_path / "job.json",
        job=job,
        result_json=result_json,
        run_log=JsonlLogger(tmp_path / "run.jsonl"),
        story_log=JsonlLogger(tmp_path / "story.jsonl"),
    )

    assert len(calls) == 3
    assert calls[0]["args"].get("filePath") == str(chart)
    assert calls[1]["args"].get("filePath") == str(info)
    assert out["images_attached_count"] == 2


def test_script_posts_prefers_card_over_infographic(tmp_path: Path) -> None:
    chart = tmp_path / "chart.png"
    chart.write_bytes(b"fakepng")
    og = tmp_path / "og.jpg"
    og.write_bytes(b"fakejpg")
    info = tmp_path / "info.png"
    info.write_bytes(b"fakepng")
    card = tmp_path / "card.png"
    card.write_bytes(b"fakepng")

    runner, calls = _make_runner(tmp_path, og_path=str(og))

    job = {
        "destination": {"platform": "discord"},
        "story": {"category": "US Stocks"},
        "state": {
            "discord": {"thread_id": "1460000000000000001", "published_message_ids": []},
            "assets": {
                "attachments": {
                    "chart_paths": [str(chart)],
                    "og_image_paths": [str(og)],
                    "infographic_paths": [str(info)],
                    "card_paths": [str(card)],
                    "og_image_urls": [],
                }
            },
        },
    }
    body = "中" * 2000  # force 2 body chunks
    result_json = {"status": "SUCCESS", "draft": {"body": body, "read_more_urls": ["https://a.com/1", "https://b.com/2", "https://c.com/3"]}}

    out = runner._publish_script_posts_draft(
        job_path=tmp_path / "job.json",
        job=job,
        result_json=result_json,
        run_log=JsonlLogger(tmp_path / "run.jsonl"),
        story_log=JsonlLogger(tmp_path / "story.jsonl"),
    )

    assert len(calls) == 3
    assert calls[0]["args"].get("filePath") == str(chart)
    assert calls[1]["args"].get("filePath") == str(card)
    assert out["images_attached_count"] == 2


def test_card_prompt_blocks_infographic_generation_attempt(tmp_path: Path) -> None:
    og = tmp_path / "og.jpg"
    og.write_bytes(b"fakejpg")

    runner, calls = _make_runner(tmp_path, og_path=str(og))

    # Simulate: card prompt exists but card generation produces nothing; runner must NOT
    # attempt infographic generation as a fallback (one generated image max per story).
    def _fake_card_paths(self: NewsroomRunner, **_: object) -> list[str]:  # noqa: ARG001
        return []

    def _boom_infographic(self: NewsroomRunner, **_: object) -> list[str]:  # noqa: ARG001
        raise AssertionError("infographic generation should be skipped when card_prompt is set")

    runner._ensure_card_paths = types.MethodType(_fake_card_paths, runner)  # type: ignore[method-assign]
    runner._ensure_infographic_paths = types.MethodType(_boom_infographic, runner)  # type: ignore[method-assign]

    job = {
        "destination": {"platform": "discord"},
        "story": {"category": "Politics"},
        "state": {
            "discord": {"thread_id": "1460000000000000001", "published_message_ids": []},
            "assets": {"attachments": {"chart_paths": [], "og_image_paths": [str(og)], "og_image_urls": []}},
        },
    }
    body = "中" * 1200  # 1 body chunk
    result_json = {
        "status": "SUCCESS",
        "draft": {
            "body": body,
            "read_more_urls": ["https://a.com/1", "https://b.com/2", "https://c.com/3"],
            "card_prompt": "比例：2:3。大標題 + 三個重點（只用已核實事實）。",
            "infographic_prompt": "比例：2:3。時間線資訊圖（應該被忽略）。",
        },
    }

    out = runner._publish_script_posts_draft(
        job_path=tmp_path / "job.json",
        job=job,
        result_json=result_json,
        run_log=JsonlLogger(tmp_path / "run.jsonl"),
        story_log=JsonlLogger(tmp_path / "story.jsonl"),
    )

    assert len(calls) == 2  # body + read-more
    assert calls[0]["args"].get("filePath") == str(og)
    assert out["images_attached_count"] == 1
