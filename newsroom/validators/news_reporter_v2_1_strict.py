from __future__ import annotations

from dataclasses import dataclass
from typing import Any


ALLOWED_FAILURE_TYPES = {
    "discord_429",
    "discord_403",
    "discord_error",
    "http_429",
    "http_503",
    "timeout",
    "paywall",
    "missing_data",
    "validation_failed",
    "unknown",
}


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    errors: list[str]


def _is_non_empty_str(v: Any) -> bool:
    return isinstance(v, str) and v.strip() != ""


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


def validate(result_json: dict[str, Any], job: dict[str, Any]) -> ValidationResult:
    errors: list[str] = []

    required_keys = [
        "status",
        "story_id",
        "category",
        "title",
        "primary_url",
        "thread_id",
        "content_posted",
        "content_message_ids",
        "images_attached_count",
        "read_more_urls_count",
        "report_char_count",
        "concrete_anchor_provided",
        "concrete_anchor_used",
        "sources_used",
        "error_type",
        "error_message",
    ]
    for k in required_keys:
        if k not in result_json:
            errors.append(f"missing_key:{k}")

    status = result_json.get("status")
    if status not in ("SUCCESS", "FAILURE"):
        errors.append("invalid:status")

    # Cross-check identity fields when possible.
    try:
        job_story = job.get("story", {}) or {}
        job_state = job.get("state", {}) or {}
        job_thread = (job_state.get("discord", {}) or {}).get("thread_id")

        if _is_non_empty_str(job_story.get("story_id")) and result_json.get("story_id") != job_story.get("story_id"):
            errors.append("mismatch:story_id")
        if _is_non_empty_str(job_story.get("primary_url")) and result_json.get("primary_url") != job_story.get("primary_url"):
            errors.append("mismatch:primary_url")
        if _is_non_empty_str(job_thread) and result_json.get("thread_id") != job_thread:
            errors.append("mismatch:thread_id")
    except Exception:
        # Avoid validator crash.
        errors.append("validator_error:identity_check")

    # Field types
    if "content_posted" in result_json and not isinstance(result_json.get("content_posted"), bool):
        errors.append("invalid:content_posted_type")

    if "content_message_ids" in result_json:
        ids = result_json.get("content_message_ids")
        if not isinstance(ids, list) or not all(_is_non_empty_str(x) for x in ids):
            errors.append("invalid:content_message_ids")

    images = _as_int(result_json.get("images_attached_count"))
    if images is None:
        errors.append("invalid:images_attached_count")
    read_more = _as_int(result_json.get("read_more_urls_count"))
    if read_more is None:
        errors.append("invalid:read_more_urls_count")
    chars = _as_int(result_json.get("report_char_count"))
    if chars is None:
        errors.append("invalid:report_char_count")

    if "concrete_anchor_used" in result_json and not isinstance(result_json.get("concrete_anchor_used"), bool):
        errors.append("invalid:concrete_anchor_used_type")

    sources = result_json.get("sources_used")
    if not isinstance(sources, list) or not all(_is_non_empty_str(x) for x in sources):
        errors.append("invalid:sources_used")

    if status == "SUCCESS":
        if result_json.get("content_posted") is not True:
            errors.append("success_requires:content_posted_true")
        if isinstance(result_json.get("content_message_ids"), list) and len(result_json.get("content_message_ids")) < 1:
            errors.append("success_requires:content_message_ids_nonempty")

        if images is not None and not (0 <= images <= 3):
            errors.append("success_requires:images_attached_count_0_to_3")
        if read_more is not None and not (3 <= read_more <= 5):
            errors.append("success_requires:read_more_urls_count_3_to_5")
        if chars is not None and not (2700 <= chars <= 3300):
            errors.append("success_requires:report_char_count_2700_to_3300")
        if result_json.get("concrete_anchor_used") is not True:
            errors.append("success_requires:concrete_anchor_used_true")

        # Best-effort: ensure the primary_url appears in sources_used.
        primary_url = result_json.get("primary_url")
        if _is_non_empty_str(primary_url) and isinstance(sources, list) and primary_url not in sources:
            errors.append("success_requires:sources_used_includes_primary_url")

        # error_type/error_message should be null-ish on success.
        if result_json.get("error_type") not in (None, "null", ""):
            errors.append("success_requires:error_type_null")
        if result_json.get("error_message") not in (None, "null", ""):
            errors.append("success_requires:error_message_null")

    if status == "FAILURE":
        err_type = result_json.get("error_type")
        if err_type not in ALLOWED_FAILURE_TYPES:
            errors.append("failure_requires:valid_error_type")
        if not _is_non_empty_str(result_json.get("error_message")):
            errors.append("failure_requires:error_message_nonempty")

    return ValidationResult(ok=len(errors) == 0, errors=errors)
