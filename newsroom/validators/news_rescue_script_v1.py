from __future__ import annotations

from typing import Any
from urllib.parse import urlsplit

try:
    from .._util import ALLOWED_FAILURE_TYPES, ValidationResult, as_int, count_cjk, is_non_empty_str
except (ImportError, SystemError):
    # Fallback for dynamic file-based loading (no parent package context).
    import importlib.util as _iu
    from pathlib import Path as _P
    _spec = _iu.spec_from_file_location("_util", str(_P(__file__).resolve().parent.parent / "_util.py"))
    _mod = _iu.module_from_spec(_spec)  # type: ignore[arg-type]
    _spec.loader.exec_module(_mod)  # type: ignore[union-attr]
    ALLOWED_FAILURE_TYPES = _mod.ALLOWED_FAILURE_TYPES  # type: ignore[assignment]
    ValidationResult = _mod.ValidationResult  # type: ignore[assignment,misc]
    as_int = _mod.as_int
    count_cjk = _mod.count_cjk
    is_non_empty_str = _mod.is_non_empty_str

# Keep module-private aliases so call-sites stay unchanged.
_is_non_empty_str = is_non_empty_str
_as_int = as_int
_count_cjk = count_cjk


def _domain(url: str) -> str:
    try:
        return (urlsplit(url).hostname or "").lower()
    except Exception:
        return ""


def validate(result_json: dict[str, Any], job: dict[str, Any]) -> ValidationResult:
    errors: list[str] = []

    required_keys = [
        "status",
        "mode",
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
        "draft",
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

    if result_json.get("mode") != "RESCUE":
        errors.append("invalid:mode")

    if "content_posted" in result_json and not isinstance(result_json.get("content_posted"), bool):
        errors.append("invalid:content_posted_type")

    if "content_message_ids" in result_json:
        ids = result_json.get("content_message_ids")
        if not isinstance(ids, list) or not all(_is_non_empty_str(x) for x in ids):
            if ids != []:
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

    sources = result_json.get("sources_used")
    if not isinstance(sources, list) or not all(_is_non_empty_str(x) for x in sources):
        errors.append("invalid:sources_used")

    draft = result_json.get("draft")
    if not isinstance(draft, dict):
        draft = {}
        errors.append("invalid:draft")

    body = draft.get("body")
    if not isinstance(body, str):
        errors.append("invalid:draft.body_type")
        body = ""
    body = body.strip()

    read_more_urls = draft.get("read_more_urls")
    if not isinstance(read_more_urls, list):
        errors.append("invalid:draft.read_more_urls_type")
        read_more_urls = []

    if status == "SUCCESS":
        if not body:
            errors.append("success_requires:body_nonempty")

        if not all(_is_non_empty_str(x) for x in read_more_urls):
            errors.append("success_requires:read_more_urls_all_nonempty")

        cjk = _count_cjk(body)
        # Rescue output is allowed to be a bit longer than the target range to avoid
        # failing a whole job due to small length overshoots.
        if not (400 <= cjk <= 2000):
            errors.append("success_requires:body_cjk_400_to_2000")

        if "http://" in body or "https://" in body:
            errors.append("success_requires:no_urls_in_body")
        if "延伸閱讀" in body:
            errors.append("success_requires:no_read_more_section_in_body")

        if not (2 <= len(read_more_urls) <= 4):
            errors.append("success_requires:read_more_urls_2_to_4")

        primary_url = result_json.get("primary_url")
        if _is_non_empty_str(primary_url) and primary_url not in read_more_urls:
            errors.append("success_requires:read_more_includes_primary_url")

        # Best-effort: ensure read-more isn't all the same domain if we have enough links.
        # Skip when the job itself only provided a single domain (niche stories).
        domains = {_domain(u) for u in read_more_urls if isinstance(u, str)}
        domains.discard("")
        if _is_non_empty_str(primary_url):
            primary_dom = _domain(primary_url)
            job_domains = (job.get("story", {}) or {}).get("domains") if isinstance(job, dict) else None
            single_domain_job = isinstance(job_domains, list) and len(job_domains) <= 1
            if primary_dom and len(read_more_urls) >= 3 and domains and domains == {primary_dom} and not single_domain_job:
                errors.append("success_requires:read_more_has_other_domain")

        if not isinstance(sources, list) or len(sources) < 2:
            errors.append("success_requires:sources_used_min_2")

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
