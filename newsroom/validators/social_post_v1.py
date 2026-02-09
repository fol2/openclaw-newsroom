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
        "mode",
        "content_type",
        "story_id",
        "title",
        "primary_url",
        "thread_id",
        "content_posted",
        "content_message_ids",
        "images_attached_count",
        "sources_used",
        "error_type",
        "error_message",
    ]
    for k in required_keys:
        if k not in result_json:
            errors.append(f"missing_key:{k}")

    if result_json.get("mode") != "WORKER":
        errors.append("invalid:mode")

    content_type = result_json.get("content_type")
    if content_type not in ("twitter_post", "facebook_post"):
        errors.append("invalid:content_type")

    status = result_json.get("status")
    if status not in ("SUCCESS", "FAILURE"):
        errors.append("invalid:status")

    if "content_posted" in result_json and not isinstance(result_json.get("content_posted"), bool):
        errors.append("invalid:content_posted_type")

    if "content_message_ids" in result_json:
        ids = result_json.get("content_message_ids")
        if not isinstance(ids, list) or not all(_is_non_empty_str(x) for x in ids):
            errors.append("invalid:content_message_ids")

    images = _as_int(result_json.get("images_attached_count"))
    if images is None:
        errors.append("invalid:images_attached_count")

    sources = result_json.get("sources_used")
    if not isinstance(sources, list) or not all(_is_non_empty_str(x) for x in sources):
        errors.append("invalid:sources_used")

    if status == "SUCCESS":
        if result_json.get("content_posted") is not True:
            errors.append("success_requires:content_posted_true")
        if isinstance(result_json.get("content_message_ids"), list) and len(result_json.get("content_message_ids")) < 1:
            errors.append("success_requires:content_message_ids_nonempty")
        if images is not None and not (0 <= images <= 1):
            errors.append("success_requires:images_attached_count_0_to_1")
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

