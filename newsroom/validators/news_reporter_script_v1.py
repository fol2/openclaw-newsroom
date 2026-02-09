from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit


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


def _count_cjk(text: str) -> int:
    total = 0
    for ch in (text or ""):
        o = ord(ch)
        if 0x3400 <= o <= 0x4DBF:  # CJK Unified Ideographs Extension A
            total += 1
        elif 0x4E00 <= o <= 0x9FFF:  # CJK Unified Ideographs
            total += 1
        elif 0x3000 <= o <= 0x303F:  # CJK Symbols and Punctuation
            total += 1
        elif 0xFF00 <= o <= 0xFFEF:  # Halfwidth and Fullwidth Forms
            total += 1
    return total


def _domain(url: str) -> str:
    try:
        return (urlsplit(url).hostname or "").lower()
    except Exception:
        return ""


def _primary_is_usable_on_topic(job: dict[str, Any], primary_url: str) -> bool:
    """Best-effort: only require primary_url in sources_used when we actually extracted usable on-topic text for it.

    Some primary URLs are bot-walled (e.g. consent pages). In that case the worker should
    rely on other sources and we shouldn't force a lie in sources_used.
    """
    try:
        state = job.get("state", {}) or {}
        pack = state.get("source_pack")
        if not isinstance(pack, dict):
            return False
        sources = pack.get("sources")
        if not isinstance(sources, list):
            return False
        for s in sources:
            if not isinstance(s, dict):
                continue
            if s.get("url") != primary_url:
                continue
            on_topic = s.get("on_topic") is True
            selected_chars = s.get("selected_chars") or 0
            try:
                selected_chars = int(selected_chars)
            except Exception:
                selected_chars = 0
            return bool(on_topic and selected_chars >= 400)
    except Exception:
        return False
    return False


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
        "draft",
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

    # Identity checks (best-effort).
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
        errors.append("validator_error:identity_check")

    if "content_posted" in result_json and not isinstance(result_json.get("content_posted"), bool):
        errors.append("invalid:content_posted_type")

    if "content_message_ids" in result_json:
        ids = result_json.get("content_message_ids")
        if not isinstance(ids, list) or not all(_is_non_empty_str(x) for x in ids):
            # In script-posts mode, worker should return [] initially; runner will fill later.
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

    if "concrete_anchor_used" in result_json and not isinstance(result_json.get("concrete_anchor_used"), bool):
        errors.append("invalid:concrete_anchor_used_type")

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

    title = result_json.get("title")
    if not isinstance(title, str) or not title.strip():
        errors.append("invalid:title")
        title = ""

    if status == "SUCCESS":
        if not body:
            errors.append("success_requires:body_nonempty")

        if not all(_is_non_empty_str(x) for x in read_more_urls):
            errors.append("success_requires:read_more_urls_all_nonempty")

        # Title should not be purely English in this workflow.
        if _count_cjk(title) < 4:
            errors.append("success_requires:title_traditional_chinese")

        cjk = _count_cjk(body)
        if not (600 <= cjk <= 3000):
            errors.append("success_requires:body_cjk_600_to_3000")

        if "http://" in body or "https://" in body:
            errors.append("success_requires:no_urls_in_body")
        if "延伸閱讀" in body:
            errors.append("success_requires:no_read_more_section_in_body")

        if not (3 <= len(read_more_urls) <= 5):
            errors.append("success_requires:read_more_urls_3_to_5")

        primary_url = result_json.get("primary_url")
        if _is_non_empty_str(primary_url) and primary_url not in read_more_urls:
            errors.append("success_requires:read_more_includes_primary_url")

        # Must include at least one different domain — unless the job itself
        # only provided a single domain (niche stories with limited coverage).
        domains = {_domain(u) for u in read_more_urls if isinstance(u, str)}
        domains.discard("")
        if _is_non_empty_str(primary_url):
            primary_dom = _domain(primary_url)
            job_domains = (job.get("story", {}) or {}).get("domains") if isinstance(job, dict) else None
            single_domain_job = isinstance(job_domains, list) and len(job_domains) <= 1
            if primary_dom and domains and domains == {primary_dom} and not single_domain_job:
                errors.append("success_requires:read_more_has_other_domain")

        # Best-effort: ensure sources_used includes primary_url and at least 2 sources.
        if not isinstance(sources, list) or len(sources) < 2:
            errors.append("success_requires:sources_used_min_2")
        if (
            _is_non_empty_str(primary_url)
            and isinstance(sources, list)
            and primary_url not in sources
            and _primary_is_usable_on_topic(job, str(primary_url))
        ):
            errors.append("success_requires:sources_used_includes_primary_url")

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
