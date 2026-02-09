from newsroom.validators.news_rescue_v1 import validate


def _base_job(thread_id: str = "123") -> dict:
    return {"state": {"discord": {"thread_id": thread_id}}}


def _base_success_result(*, thread_id: str = "123", primary_url: str = "https://example.com/a") -> dict:
    return {
        "status": "SUCCESS",
        "mode": "RESCUE",
        "story_id": "story_01",
        "category": "Global News",
        "title": "Example Title",
        "primary_url": primary_url,
        "thread_id": thread_id,
        "content_posted": True,
        "content_message_ids": ["1"],
        "images_attached_count": 0,
        "read_more_urls_count": 2,
        "report_char_count": 900,
        "sources_used": [primary_url],
        "error_type": None,
        "error_message": None,
    }


def test_rescue_success_requires_sources_nonempty() -> None:
    job = _base_job()
    result = _base_success_result()
    result["sources_used"] = []
    out = validate(result, job)
    assert not out.ok
    assert "success_requires:sources_used_nonempty" in out.errors


def test_rescue_success_requires_sources_include_primary() -> None:
    job = _base_job()
    result = _base_success_result()
    result["sources_used"] = ["https://example.org/other"]
    out = validate(result, job)
    assert not out.ok
    assert "success_requires:sources_used_includes_primary_url" in out.errors

