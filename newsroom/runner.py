from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import logging
import os
import re
import socket
import subprocess
import sys
import time
import urllib.parse
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from shutil import which
from typing import Any, Callable
from zoneinfo import ZoneInfo

import jsonschema
import requests

from .gateway_client import GatewayClient
from .job_store import FileLock, LockHeldError, atomic_write_json, jail_job_file, load_json_file, utc_iso
from .brave_news import fetch_brave_news, load_brave_api_keys, normalize_url, record_brave_rate_limit, select_brave_api_key
from .dedupe import best_semantic_duplicate
from .news_pool_db import NewsPoolDB
from .result_repair import repair_result_json
from .charts import render_line_chart_png
from ._util import count_cjk
from .image_fetch import download_image, fetch_og_image_url
from .market_data import build_market_assets
from .source_pack import build_source_pack
from .story_index import anchor_terms, choose_key_tokens, tokenize_text

_log = logging.getLogger(__name__)


class JobSchemaError(RuntimeError):
    pass


class PromptRegistryError(RuntimeError):
    pass


class ResultParseError(RuntimeError):
    pass


_INFOGRAPHIC_COLOR_RE = re.compile(
    r"\b([A-Z][A-Za-z0-9_-]{1,20})\s+(Blue|Red|Green|Orange|Yellow|Pink|Purple|Black|White|Gray|Grey)\b"
)
_INFOGRAPHIC_COLOR_LOWER_RE = re.compile(
    r"\b([A-Z][A-Za-z0-9_-]{1,20})\s+(blue|red|green|orange|yellow|pink|purple|black|white|gray|grey)\b"
)

_DEFAULT_GENERATED_IMAGE_ASPECT = "2:3"
_ALLOWED_GENERATED_IMAGE_ASPECTS: dict[str, tuple[int, int]] = {
    "2:3": (2, 3),
    "3:2": (3, 2),
}

_ASPECT_16_9_RE = re.compile(r"(?i)\b16\s*[:x]\s*9\b")
_ASPECT_9_16_RE = re.compile(r"(?i)\b9\s*[:x]\s*16\b")
_ASPECT_3_2_RE = re.compile(r"(?i)\b3\s*[:x]\s*2\b")
_ASPECT_2_3_RE = re.compile(r"(?i)\b2\s*[:x]\s*3\b")


def _normalize_generated_image_aspect(aspect: str | None) -> str:
    a = str(aspect or "").strip()
    a = a.replace("X", ":").replace("x", ":")
    if a in _ALLOWED_GENERATED_IMAGE_ASPECTS:
        return a
    return _DEFAULT_GENERATED_IMAGE_ASPECT


def _pick_generated_image_aspect_from_prompt(prompt: str) -> str:
    s = str(prompt or "")
    if _ASPECT_3_2_RE.search(s) or _ASPECT_16_9_RE.search(s):
        return "3:2"
    if _ASPECT_2_3_RE.search(s) or _ASPECT_9_16_RE.search(s):
        return "2:3"
    return _DEFAULT_GENERATED_IMAGE_ASPECT


def _aspect_orientation(aspect: str) -> str:
    return "portrait" if _normalize_generated_image_aspect(aspect) == "2:3" else "landscape"


def _sanitize_generated_image_prompt(prompt: str) -> str:
    """Sanitize worker-provided prompt for more reliable image generation.

    This is intentionally conservative: we remove/replace common trademark-related
    wording that often causes the image model to return an empty response.
    """
    s = " ".join((prompt or "").strip().split())
    if not s:
        return ""

    # Avoid explicit logo/trademark requests (we already instruct "NO logos" in the prefix).
    s = re.sub(r"(?i)\blogos?\b", "icons", s)
    s = re.sub(r"(?i)\bbadge\b", "label", s)
    s = re.sub(r"(?i)\btrademark(ed)?\b", "", s)

    # Normalize aspect ratio mentions to our allowed set.
    s = _ASPECT_16_9_RE.sub("3:2", s)
    s = _ASPECT_9_16_RE.sub("2:3", s)
    s = _ASPECT_3_2_RE.sub("3:2", s)
    s = _ASPECT_2_3_RE.sub("2:3", s)

    # "Sky Blue" (and similar Brand+Color patterns) can trigger "brand styling" blocks.
    # Replace "<Brand> <Color>" with just the generic color.
    s = _INFOGRAPHIC_COLOR_RE.sub(lambda m: m.group(2).lower(), s)
    s = _INFOGRAPHIC_COLOR_LOWER_RE.sub(lambda m: m.group(2).lower(), s)

    # Re-normalize whitespace after removals.
    s = " ".join(s.split())
    return s


def _infographic_final_prefix(aspect: str) -> str:
    aspect = _normalize_generated_image_aspect(aspect)
    orient = _aspect_orientation(aspect)
    return (
        f"Create a clean flat vector-style {aspect} {orient} news infographic. "
        "Use generic icons and simple shapes. "
        "NO logos, NO trademarked brand assets, NO photorealistic faces or identifiable people. "
        "Keep text minimal and LARGE (avoid small text). "
        "All on-image text MUST be Traditional Chinese (Hong Kong) with Cantonese phrasing. "
        "Do NOT use English words except unavoidable proper nouns/tickers. "
        "High contrast, modern newsroom look. "
        "Now generate an infographic for: "
    )


def _infographic_fallback_prefix(aspect: str) -> str:
    aspect = _normalize_generated_image_aspect(aspect)
    orient = _aspect_orientation(aspect)
    return (
        f"Create a clean flat vector-style {aspect} {orient} news infographic. "
        "Use ONLY generic icons and simple shapes. "
        "NO logos, NO trademarked brand assets, NO photorealistic faces or identifiable people. "
        "Keep text minimal and LARGE. "
        "All on-image text MUST be Traditional Chinese (Hong Kong) with Cantonese phrasing. "
        "Layout: title bar + 3 boxes labelled: '發生咩事', '關鍵事實', '後續睇咩'. "
        "Use this information (plain text only): "
    )


def _card_final_prefix(aspect: str) -> str:
    aspect = _normalize_generated_image_aspect(aspect)
    orient = _aspect_orientation(aspect)
    return (
        f"Create a clean flat vector-style {aspect} {orient} news summary card. "
        "Use generic icons and simple shapes. "
        "NO logos, NO trademarked brand assets, NO photorealistic faces or identifiable people. "
        "Keep text minimal and LARGE (avoid small text). "
        "All on-image text MUST be Traditional Chinese (Hong Kong) with Cantonese phrasing. "
        "Do NOT use English words except unavoidable proper nouns/tickers. "
        "High contrast, modern newsroom look. "
        "Layout: big headline + 3 short key points. "
        "Now generate a card for: "
    )


def _card_fallback_prefix(aspect: str) -> str:
    aspect = _normalize_generated_image_aspect(aspect)
    orient = _aspect_orientation(aspect)
    return (
        f"Create a clean flat vector-style {aspect} {orient} news summary card. "
        "Use ONLY generic icons and simple shapes. "
        "NO logos, NO trademarked brand assets, NO photorealistic faces or identifiable people. "
        "Keep text minimal and LARGE. "
        "All on-image text MUST be Traditional Chinese (Hong Kong) with Cantonese phrasing. "
        "Layout: title bar + 3 boxes labelled: '發生咩事', '關鍵事實', '後續睇咩'. "
        "Use this information (plain text only): "
    )

def _pad_png_to_aspect(path: Path, aspect: str) -> bool:
    """Pad an existing PNG to an exact 2:3 or 3:2 canvas (centered).

    We prefer padding over cropping for generated images to avoid cutting text.
    Returns True if the file was modified.
    """
    try:
        from PIL import Image  # type: ignore[import-not-found]
    except Exception:
        return False

    if not path.exists() or path.suffix.lower() != ".png":
        return False
    try:
        img = Image.open(path)
        img.load()
    except Exception:
        return False

    w, h = img.size
    if w <= 0 or h <= 0:
        return False

    aspect = _normalize_generated_image_aspect(aspect)
    rw, rh = _ALLOWED_GENERATED_IMAGE_ASPECTS[aspect]
    # We'll pad to the nearest integer canvas size that keeps the larger dimension
    # and expands the other.
    target_ratio = rw / rh
    cur_ratio = w / h

    # If already close enough, keep it.
    if abs(cur_ratio - target_ratio) <= 0.001:
        return False

    if cur_ratio > target_ratio:
        # Too wide => increase height.
        new_h = int(round(w / target_ratio))
        new_w = w
    else:
        # Too tall => increase width.
        new_w = int(round(h * target_ratio))
        new_h = h

    if new_w <= w and new_h <= h:
        return False

    # Use a dark neutral background to match our prompt prefix.
    bg = (12, 16, 24)  # near-black blue
    if img.mode in ("RGBA", "LA"):
        canvas = Image.new("RGBA", (new_w, new_h), bg + (255,))
        if img.mode != "RGBA":
            img = img.convert("RGBA")
    else:
        canvas = Image.new("RGB", (new_w, new_h), bg)
        if img.mode != "RGB":
            img = img.convert("RGB")

    x = (new_w - w) // 2
    y = (new_h - h) // 2
    canvas.paste(img, (x, y))

    try:
        canvas.save(path, "PNG")
    except Exception:
        return False
    return True


@dataclass(frozen=True)
class PromptDef:
    prompt_id: str
    template_path: Path
    validator_id: str


@dataclass(frozen=True)
class ValidatorDef:
    validator_id: str
    path: Path


@dataclass(frozen=True)
class ValidationOutcome:
    ok: bool
    errors: list[str]


class PromptRegistry:
    def __init__(self, *, openclaw_home: Path) -> None:
        self._openclaw_home = openclaw_home
        self._registry_path = openclaw_home / "newsroom" / "prompt_registry.json"
        self._raw = json.loads(self._registry_path.read_text(encoding="utf-8"))

        if self._raw.get("schema_version") != "prompt_registry_v1":
            raise PromptRegistryError("Unsupported prompt registry schema_version")

    def resolve_prompt(self, prompt_id: str) -> PromptDef:
        prompts = self._raw.get("prompts", {}) or {}
        spec = prompts.get(prompt_id)
        if not spec:
            raise PromptRegistryError(f"Unknown prompt_id: {prompt_id}")
        template_path = self._openclaw_home / str(spec.get("template_path", ""))
        validator_id = str(spec.get("validator_id", "")).strip()
        if not template_path.exists():
            raise PromptRegistryError(f"Missing template file for prompt_id={prompt_id}: {template_path}")
        if not validator_id:
            raise PromptRegistryError(f"Missing validator_id for prompt_id={prompt_id}")
        return PromptDef(prompt_id=prompt_id, template_path=template_path, validator_id=validator_id)

    def resolve_validator(self, validator_id: str) -> ValidatorDef:
        validators = self._raw.get("validators", {}) or {}
        spec = validators.get(validator_id)
        if not spec:
            raise PromptRegistryError(f"Unknown validator_id: {validator_id}")
        if spec.get("type") != "python":
            raise PromptRegistryError(f"Unsupported validator type for {validator_id}: {spec.get('type')}")
        path = self._openclaw_home / str(spec.get("path", ""))
        if not path.exists():
            raise PromptRegistryError(f"Missing validator file for validator_id={validator_id}: {path}")
        return ValidatorDef(validator_id=validator_id, path=path)

    def default_prompt_for_content_type(self, content_type: str) -> str | None:
        ct = (self._raw.get("content_types", {}) or {}).get(content_type)
        if not ct:
            return None
        val = ct.get("default_prompt_id")
        return str(val).strip() if val else None


def _render_template(template_text: str, vars_map: dict[str, str]) -> str:
    if "OPENCLAW_HOME" not in vars_map:
        vars_map["OPENCLAW_HOME"] = os.environ.get("OPENCLAW_HOME", str(Path.home() / ".openclaw"))
    out = template_text
    for k, v in vars_map.items():
        out = out.replace(f"{{{{{k}}}}}", v)

    # Detect unreplaced placeholders early (prevents silent prompt bugs).
    leftovers = sorted(set(re.findall(r"\\{\\{([A-Z0-9_]+)\\}\\}", out)))
    if leftovers:
        raise PromptRegistryError(f"Unreplaced template placeholders: {leftovers}")
    return out


def _jsonpath_get(obj: Any, expr: str) -> Any:
    """Minimal $.a.b.c resolver used for spawn.input_mapping.

    Supports:
    - $.foo.bar (dict traversal)
    - $.foo.0.bar (list index)
    """
    expr = (expr or "").strip()
    if not expr.startswith("$."):
        return None
    cur: Any = obj
    for part in expr[2:].split("."):
        if part == "":
            continue
        if isinstance(cur, dict):
            cur = cur.get(part)
            continue
        if isinstance(cur, list):
            try:
                idx = int(part)
            except ValueError:
                return None
            if idx < 0 or idx >= len(cur):
                return None
            cur = cur[idx]
            continue
        return None
    return cur


def build_worker_input(job: dict[str, Any]) -> dict[str, Any]:
    mapping = (job.get("spawn", {}) or {}).get("input_mapping", {}) or {}
    if not isinstance(mapping, dict):
        return {}
    out: dict[str, Any] = {}
    for key, expr in mapping.items():
        if not isinstance(key, str) or not isinstance(expr, str):
            continue
        out[key] = _jsonpath_get(job, expr)
    return out


def _extract_first_json_object(text: str) -> dict[str, Any] | None:
    """Best-effort extraction of a single JSON object from a string.

    The worker contracts should return ONLY JSON, but we still protect ourselves
    from accidental prose/codefences.
    """
    if not text:
        return None

    # Fast path: whole string is JSON
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            obj = json.loads(stripped)
            return obj if isinstance(obj, dict) else None
        except Exception:
            pass

    # Strip code fences
    stripped = re.sub(r"^```(?:json)?\\s*", "", stripped)
    stripped = re.sub(r"\\s*```$", "", stripped)
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            obj = json.loads(stripped)
            return obj if isinstance(obj, dict) else None
        except Exception:
            pass

    # Brace matching extraction
    start = stripped.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(stripped)):
        ch = stripped[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = stripped[start : i + 1]
                try:
                    obj = json.loads(candidate)
                    return obj if isinstance(obj, dict) else None
                except Exception:
                    return None
    return None


def find_result_json_in_messages(messages: list[dict[str, Any]]) -> dict[str, Any] | None:
    for msg in reversed(messages):
        if not isinstance(msg, dict):
            continue
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", [])
        # Handle content as list of blocks (session file format).
        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict) or block.get("type") != "text":
                    continue
                obj = _extract_first_json_object(str(block.get("text", "")))
                if obj and isinstance(obj, dict) and "status" in obj:
                    return obj
        # Handle content as string (sessions_history API format).
        elif isinstance(content, str):
            obj = _extract_first_json_object(content)
            if obj and isinstance(obj, dict) and "status" in obj:
                return obj
    return None


def _assistant_terminal_empty_output(messages: list[dict[str, Any]]) -> bool:
    """Detect a terminal assistant message with empty text output.

    We've observed some providers occasionally return an assistant message with
    stopReason but an empty text block, which would otherwise make the runner
    poll until timeout.
    """
    for msg in reversed(messages or []):
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        stop_reason = msg.get("stopReason")
        if not stop_reason:
            return False
        content = msg.get("content") or []
        if not isinstance(content, list):
            return True
        text = ""
        for blk in content:
            if isinstance(blk, dict) and blk.get("type") == "text":
                text += str(blk.get("text") or "")
        return text.strip() == ""
    return False


def _load_validator_callable(path: Path) -> Callable[[dict[str, Any], dict[str, Any]], Any]:
    spec = importlib.util.spec_from_file_location(f"newsroom_validator_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise PromptRegistryError(f"Unable to import validator: {path}")
    module = importlib.util.module_from_spec(spec)
    # Some validators define dataclasses, which rely on sys.modules being populated
    # for the module during execution (importlib does this implicitly; our loader doesn't).
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    validate = getattr(module, "validate", None)
    if not callable(validate):
        raise PromptRegistryError(f"Validator module does not export validate(): {path}")
    return validate


class JsonlLogger:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: str, **fields: Any) -> None:
        row = {"ts": utc_iso(), "event": event}
        row.update(fields)
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def _runner_id() -> str:
    return f"{socket.gethostname()}:{os.getpid()}"


def _find_first_key(obj: Any, key_names: set[str]) -> Any:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in key_names:
                return v
        for v in obj.values():
            found = _find_first_key(v, key_names)
            if found is not None:
                return found
    if isinstance(obj, list):
        for v in obj:
            found = _find_first_key(v, key_names)
            if found is not None:
                return found
    return None


def _is_discord_snowflake(v: Any) -> bool:
    if isinstance(v, int):
        s = str(v)
    elif isinstance(v, str):
        s = v.strip()
    else:
        return False
    return s.isdigit() and len(s) >= 15


# Extract URLs from Discord message content. Avoid capturing trailing angle brackets
# from Discord's `<https://...>` autolink form.
_URL_RE = re.compile(r"https?://[^\s<>]+")


def _strip_urls(text: str) -> str:
    return _URL_RE.sub("", text or "")


def _split_discord_messages(text: str, *, max_chars: int = 1900) -> list[str]:
    """Split long text into Discord-safe message chunks.

    Prefer paragraph boundaries, but fall back to hard splits when needed.
    """
    t = (text or "").strip()
    if not t:
        return []
    if len(t) <= max_chars:
        return [t]

    parts: list[str] = []
    buf = ""
    for para in re.split(r"\n{2,}", t):
        p = para.strip()
        if not p:
            continue
        candidate = p if not buf else f"{buf}\n\n{p}"
        if len(candidate) <= max_chars:
            buf = candidate
            continue
        if buf:
            parts.append(buf)
            buf = ""
        if len(p) <= max_chars:
            buf = p
            continue
        # Hard split an oversized paragraph.
        for i in range(0, len(p), max_chars):
            parts.append(p[i : i + max_chars])
    if buf:
        parts.append(buf)
    return parts


_REPORT_SECTION_NUM_RE = re.compile(r"^\s*([1-6])[\)）]\s*(.*)$")


def _normalize_report_body(body: str) -> str:
    """Normalize worker draft formatting for Discord.

    The reporter prompt recommends section headings like `【背景脈絡】`, but some
    models still emit numbered headers (`1) ... 6) ...`). Users found that style
    noisy, so we deterministically convert the *known* 6 section headings while
    leaving other content untouched.
    """
    t = (body or "").strip()
    if not t:
        return ""

    section_labels = {
        "1": ("背景脈絡", ["背景脈絡"]),
        "2": ("今次發生咩事", ["今次發生咩事"]),
        "3": ("關鍵人物／機構", ["關鍵人物／機構", "關鍵人物/機構", "關鍵人物或機構"]),
        "4": ("數據與細節", ["數據與細節", "數據同細節", "數據與詳情", "數據及細節"]),
        "5": ("各方反應", ["各方反應", "市場反應", "各界反應"]),
        "6": ("後續可能點走", ["後續可能點走", "後續走勢", "後續發展", "後續可能點樣走"]),
    }

    out: list[str] = []
    for line in t.splitlines():
        m = _REPORT_SECTION_NUM_RE.match(line)
        if not m:
            out.append(line)
            continue
        num = m.group(1)
        rest = (m.group(2) or "").strip()
        label_entry = section_labels.get(num)
        if not label_entry:
            out.append(line)
            continue
        label, aliases = label_entry

        alias_set = set(aliases or [])
        alias_set.add(label)

        # Only rewrite when the line actually looks like our report heading.
        matched = False
        # 1) Plain: "背景脈絡 ..." or "背景脈絡：..."
        for a in alias_set:
            if rest.startswith(a):
                rest = rest[len(a) :].lstrip()
                matched = True
                break
        # 2) Bracketed: "【背景脈絡】..." or "【背景脈絡】：..."
        if not matched and rest.startswith("【"):
            end = rest.find("】")
            if end != -1:
                inside = rest[1:end].strip()
                if inside in alias_set:
                    rest = rest[end + 1 :].lstrip()
                    matched = True

        if not matched:
            out.append(line)
            continue

        if rest.startswith(("：", ":")):
            rest = rest[1:].lstrip()

        if rest:
            out.append(f"【{label}】\n{rest}")
        else:
            out.append(f"【{label}】")
    return "\n".join(out).strip()


# Keep module-private alias so existing call-sites stay unchanged.
_count_cjk = count_cjk


def _looks_like_english_title(title: str) -> bool:
    """Heuristic: title is mostly ASCII (common failure mode when we require Cantonese)."""
    t = (title or "").strip()
    if not t:
        return False
    cjk = _count_cjk(t)
    # If we have enough CJK, treat it as non-English even if it contains some tickers/proper nouns.
    if cjk >= 6:
        return False
    ascii_letters = sum(1 for ch in t if ch.isascii() and ch.isalpha())
    return cjk < 4 and ascii_letters >= 10


def _usable_sources_count(pack_obj: Any, *, min_chars: int = 400) -> int:
    """Count extracted sources with enough on-topic evidence for reporting.

    Prefer runner-computed `sources_pack.stats.on_topic_sources_count` when present
    (prevents irrelevant long pages from passing the gate).
    """
    if not isinstance(pack_obj, dict):
        return 0
    stats = pack_obj.get("stats")
    if isinstance(stats, dict):
        v = stats.get("on_topic_sources_count")
        if isinstance(v, int):
            return max(0, v)
        v = stats.get("usable_sources_count")
        if isinstance(v, int):
            # Back-compat: older packs didn't score on-topic sources.
            return max(0, v)
    sources = pack_obj.get("sources")
    if not isinstance(sources, list):
        return 0
    n = 0
    for s in sources:
        if not isinstance(s, dict):
            continue
        txt = s.get("text")
        if isinstance(txt, str) and len(txt.strip()) >= int(min_chars):
            n += 1
    return n


def _normalize_url(url: str) -> str:
    """Best-effort URL normalizer for dedupe keys.

    This is intentionally conservative: we strip fragments and common tracking
    parameters but keep the rest stable.
    """
    raw = (url or "").strip()
    if not raw:
        return ""
    try:
        parsed = urllib.parse.urlsplit(raw)
    except Exception:
        return raw

    # Drop fragment.
    fragmentless = parsed._replace(fragment="")

    # Drop common tracking query params.
    if fragmentless.query:
        q = urllib.parse.parse_qsl(fragmentless.query, keep_blank_values=True)
        filtered = []
        for k, v in q:
            lk = k.lower()
            if lk.startswith("utm_"):
                continue
            if lk in {"fbclid", "gclid", "igshid", "mc_cid", "mc_eid"}:
                continue
            filtered.append((k, v))
        query = urllib.parse.urlencode(filtered, doseq=True)
        fragmentless = fragmentless._replace(query=query)

    return urllib.parse.urlunsplit(fragmentless)


def _utc_iso_to_ts(v: Any) -> float | None:
    """Parse `utc_iso()` style strings (YYYY-MM-DDTHH:MM:SSZ) to epoch seconds."""
    if not isinstance(v, str) or not v.strip():
        return None
    s = v.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    try:
        return float(dt.timestamp())
    except Exception:
        return None


@dataclass
class JobRuntime:
    job_path: Path
    job: dict[str, Any]
    lock: FileLock
    dedupe_lock: FileLock | None
    phase: str  # worker|rescue
    deadline_ts: float
    poll_seconds: int
    next_poll_ts: float


class NewsroomRunner:
    _DEDUPE_TTL_SECONDS = 24 * 60 * 60
    _TITLE_TRANSLATE_TIMEOUT_SECONDS = 90

    def __init__(
        self,
        *,
        openclaw_home: Path,
        gateway: GatewayClient,
        prompt_registry: PromptRegistry,
        dry_run: bool,
        lock_ttl_seconds: int,
        log_root: Path,
    ) -> None:
        self._openclaw_home = openclaw_home
        self._gateway = gateway
        self._prompt_registry = prompt_registry
        self._dry_run = dry_run
        self._lock_ttl_seconds = lock_ttl_seconds
        self._log_root = log_root

        self._story_schema = json.loads((openclaw_home / "newsroom" / "schemas" / "story_job_v1.schema.json").read_text(encoding="utf-8"))
        self._run_schema = json.loads((openclaw_home / "newsroom" / "schemas" / "run_job_v1.schema.json").read_text(encoding="utf-8"))

        self._validator_cache: dict[str, Callable[[dict[str, Any], dict[str, Any]], Any]] = {}
        self._dedupe_primary_url_index: dict[str, Path] = {}
        self._dedupe_primary_url_index_loaded = False

    def _load_dedupe_primary_url_index(self) -> None:
        if self._dedupe_primary_url_index_loaded:
            return
        self._dedupe_primary_url_index_loaded = True

        dedupe_dir = self._log_root / "dedupe"
        if not dedupe_dir.exists():
            return

        # Back-compat: older runners wrote markers keyed by story.dedupe_key (url|hash). We
        # index by primary_url so newer runners can detect duplicates across marker versions.
        for p in dedupe_dir.glob("*.json"):
            if not self._dedupe_marker_is_fresh(p):
                continue
            try:
                payload = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue

            primary = payload.get("primary_url")
            if not isinstance(primary, str) or not primary.strip():
                continue
            key = _normalize_url(primary.strip())
            if not key:
                continue

            existing = self._dedupe_primary_url_index.get(key)
            if existing is None:
                self._dedupe_primary_url_index[key] = p
                continue
            try:
                if p.stat().st_mtime > existing.stat().st_mtime:
                    self._dedupe_primary_url_index[key] = p
            except Exception:
                # If we can't stat, keep the previous one.
                continue

    def _dedupe_marker_for_primary_url(self, primary_url: str) -> Path | None:
        key = _normalize_url(primary_url)
        if not key:
            return None
        self._load_dedupe_primary_url_index()
        marker = self._dedupe_primary_url_index.get(key)
        if marker is None:
            return None
        if not self._dedupe_marker_is_fresh(marker):
            self._dedupe_primary_url_index.pop(key, None)
            return None
        if not marker.exists():
            self._dedupe_primary_url_index.pop(key, None)
            return None
        return marker

    def _event_key_from_terms(self, terms: list[str]) -> str:
        terms = [t for t in (str(x).strip() for x in terms) if t]
        sig = "|".join(sorted(set(terms))[:24])
        digest = hashlib.sha1(sig.encode("utf-8")).hexdigest()[:20]
        return f"event:{digest}"

    def _derive_event_terms_for_story(self, *, title: str, primary_url: str) -> list[str]:
        """Derive a stable-ish set of terms for de-duplication keys.

        Goal: keep keys stable even if the title gets translated later.
        Strategy:
        - Prefer Latin proper nouns/tickers (often present in Cantonese titles too).
        - Fallback to light tokenization for CJK-only stories.
        """
        terms: set[str] = set()

        # 1) Latin words from the title (proper nouns/tickers tend to survive translation).
        for w in re.findall(r"[A-Za-z][A-Za-z0-9'-]{2,}", title or ""):
            w = w.strip("-'").lower()
            if w:
                terms.add(w)

        # 2) Latin-ish tokens from the primary URL path (often includes entities).
        try:
            parsed = urllib.parse.urlsplit(primary_url or "")
            path = parsed.path or ""
        except Exception:
            path = ""
        for w in re.findall(r"[A-Za-z0-9]{3,}", path):
            w = w.lower()
            # Skip pure digits.
            if w.isdigit():
                continue
            terms.add(w)

        if len(terms) >= 2:
            return sorted(terms)[:12]

        # 3) Fallback: general tokenization (CJK bigrams + longer Latin words).
        tokens = tokenize_text(title or "", None)
        key_tokens = choose_key_tokens(tokens, drop_high_df=set())
        if not key_tokens:
            return []
        return sorted(key_tokens)[:12]

    def _ensure_story_dedupe_key(self, job: dict[str, Any]) -> None:
        story = job.get("story", {}) or {}
        raw = story.get("dedupe_key")
        title = str(story.get("title") or "").strip()
        primary_url = str(story.get("primary_url") or "").strip()

        if isinstance(raw, str) and raw.strip():
            raw = raw.strip()
            if raw.startswith(("event:", "anchor:")):
                return
            if raw.startswith("http"):
                story["dedupe_key"] = _normalize_url(raw)
                job["story"] = story
                return

        # Derive an event-level key to reduce duplicates across different URLs/headlines.
        terms = self._derive_event_terms_for_story(title=title, primary_url=primary_url)
        if terms:
            story["dedupe_key"] = self._event_key_from_terms(terms)
            job["story"] = story
            return

        # Final fallback: hash the raw key if present, else use normalized primary URL.
        if isinstance(raw, str) and raw.strip():
            story["dedupe_key"] = self._event_key_from_terms([raw.strip()])
            job["story"] = story
            return
        if primary_url:
            story["dedupe_key"] = _normalize_url(primary_url)
            job["story"] = story

    def _dedupe_key_for_job(self, job: dict[str, Any]) -> str:
        story = job.get("story", {}) or {}
        raw = story.get("dedupe_key")
        if isinstance(raw, str) and raw.strip():
            raw = raw.strip()
            # If the planner provides a semantic event key (e.g. "event:..."), prefer it.
            # If it accidentally provides a URL-like key, normalize it for stability.
            return _normalize_url(raw) if raw.startswith("http") else raw
        primary = story.get("primary_url")
        if isinstance(primary, str) and primary.strip():
            return _normalize_url(primary.strip())
        return ""

    def _dedupe_marker_path(self, dedupe_key: str) -> Path:
        digest = hashlib.sha1(dedupe_key.encode("utf-8")).hexdigest()[:20]
        return self._log_root / "dedupe" / f"{digest}.json"

    def _dedupe_lock_path(self, dedupe_key: str) -> Path:
        digest = hashlib.sha1(dedupe_key.encode("utf-8")).hexdigest()[:20]
        return self._log_root / "dedupe" / f"{digest}.lock"

    def _dedupe_marker_is_fresh(self, marker_path: Path) -> bool:
        try:
            age = time.time() - marker_path.stat().st_mtime
        except FileNotFoundError:
            return False
        return age < self._DEDUPE_TTL_SECONDS

    def _write_dedupe_marker(self, *, marker_path: Path, dedupe_key: str, job_path: Path, job: dict[str, Any]) -> None:
        story = job.get("story", {}) or {}
        state = job.get("state", {}) or {}
        discord_state = state.get("discord", {}) or {}
        payload: dict[str, Any] = {
            "dedupe_key": dedupe_key,
            "run_id": (job.get("run", {}) or {}).get("run_id"),
            "job_path": str(job_path),
            "title": story.get("title"),
            "primary_url": story.get("primary_url"),
            "title_message_id": discord_state.get("title_message_id"),
            "thread_id": discord_state.get("thread_id"),
            "created_at": utc_iso(),
        }
        atomic_write_json(marker_path, payload)
        if isinstance(payload.get("primary_url"), str):
            primary = str(payload.get("primary_url") or "").strip()
            if primary:
                self._load_dedupe_primary_url_index()
                self._dedupe_primary_url_index[_normalize_url(primary)] = marker_path

    def _try_read_session_file(self, session_key: str, story_log: Any) -> dict[str, Any] | None:
        """Fallback: read session JSONL file directly when gateway returns empty messages.

        sessions.json maps session keys to sessionIds; the JSONL lives at
        agents/<agentId>/sessions/<sessionId>.jsonl.
        """
        try:
            store_path = self._openclaw_home / "agents" / "main" / "sessions" / "sessions.json"
            if not store_path.exists():
                return None
            store = json.loads(store_path.read_text(encoding="utf-8"))
            entry = store.get(session_key)
            if not isinstance(entry, dict):
                return None
            session_id = entry.get("sessionId")
            if not session_id:
                return None
            session_file = store_path.parent / f"{session_id}.jsonl"
            if not session_file.exists():
                return None
            messages: list[dict[str, Any]] = []
            for line in session_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                msg = row.get("message") if isinstance(row, dict) else None
                if isinstance(msg, dict) and msg.get("role"):
                    messages.append(msg)
            result = find_result_json_in_messages(messages)
            if result:
                story_log.log("result_found_via_session_file", session_id=session_id)
            return result
        except Exception as e:
            story_log.log("session_file_fallback_failed", error=str(e))
            return None

    def _record_posted_event(self, *, job_path: Path, job: dict[str, Any], story_log: Any) -> None:
        """Mark the event as posted in the events table (v5).

        If the story has an event_id (from LLM clustering), simply update its status.
        No LLM call needed — metadata already exists from the clustering step.
        Best-effort: failures are logged but never block the runner.
        """
        from .news_pool_db import NewsPoolDB

        story = job.get("story", {}) or {}
        state = job.get("state", {}) or {}
        discord_state = state.get("discord", {}) or {}
        run_info = job.get("run", {}) or {}

        event_id = story.get("event_id")
        if not isinstance(event_id, int):
            # No event_id means this story was not clustered (legacy flow).
            story_log.log("posted_event_skip_no_event_id")
            return

        thread_id = discord_state.get("thread_id")
        run_id = run_info.get("run_id")

        if not thread_id:
            _log.warning(
                "thread_id is None when recording posted event: event_id=%s run_id=%s job_path=%s",
                event_id, run_id, job_path,
            )
            story_log.log("posted_event_missing_thread_id", event_id=event_id, run_id=run_id)

        db_path = self._openclaw_home / "data" / "newsroom" / "news_pool.sqlite3"
        db = NewsPoolDB(path=db_path)
        try:
            db.mark_event_posted(event_id, thread_id=thread_id, run_id=run_id)
            story_log.log("posted_event_recorded", event_id=event_id, thread_id=thread_id)
        finally:
            db.close()

    def _mark_job_skipped_duplicate(self, *, job_path: Path, job: dict[str, Any], reason: str) -> None:
        state = job.get("state", {}) or {}
        state["status"] = "SKIPPED"
        state["locked_by"] = None
        state["locked_at"] = None
        job["state"] = state

        result = job.get("result", {}) or {}
        result["final_status"] = "SKIPPED_DUPLICATE"
        result.setdefault("errors", []).append({"type": "duplicate", "message": reason, "at": utc_iso()})
        job["result"] = result

        _, story_log = self._loggers_for_job(job, job_path)
        self._update_job_file(job_path, job, story_log, event="job_skipped", reason=reason)

        # Recover thread_id from the dedupe marker so _record_posted_event can
        # store it even though this job never created its own Discord thread.
        discord_state = state.get("discord", {}) or {}
        if not discord_state.get("thread_id") and "dedupe_marker_exists:" in reason:
            marker_name = reason.split("dedupe_marker_exists:")[-1].strip()
            marker_path = self._log_root / "dedupe" / marker_name
            try:
                if marker_path.is_file():
                    marker = json.loads(marker_path.read_text(encoding="utf-8"))
                    recovered_tid = marker.get("thread_id")
                    if recovered_tid:
                        discord_state["thread_id"] = str(recovered_tid)
                        state["discord"] = discord_state
                        job["state"] = state
                        story_log.log("thread_id_recovered_from_dedupe_marker", thread_id=str(recovered_tid), marker=marker_name)
            except Exception as e:
                story_log.log("thread_id_recovery_from_marker_failed", error=str(e), marker=marker_name)

        # Also mark the event as posted in the DB so the planner won't
        # pick this event_id again.  Retry up to 3 times to avoid silent double-posts.
        _event_id = job.get("story", {}).get("event_id")
        for _attempt in range(3):
            try:
                self._record_posted_event(job_path=job_path, job=job, story_log=story_log)
                break
            except Exception as e:
                if _attempt < 2:
                    time.sleep(1)
                else:
                    story_log.log(
                        "posted_event_record_failed_after_retries",
                        error=str(e),
                        event_id=_event_id,
                    )
                    _log.warning(
                        "CRITICAL: failed to mark event as posted after 3 retries: event_id=%s error=%s",
                        _event_id,
                        e,
                    )

    def _clean_discord_title(self, text: str) -> str:
        t = (text or "").strip()
        t = " ".join(t.split())
        if len(t) > 180:
            t = t[:177] + "..."
        return t

    @staticmethod
    def _messages_from_message_read(resp: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract a message list from the `message.read` tool response.

        The gateway has two common shapes:
        - invoke_result_json(): {"ok": true, "messages": [...]}
        - invoke(): {"ok": true, "result": {"details": {"ok": true, "messages": [...]}, ...}}
        """
        msgs = resp.get("messages")
        if isinstance(msgs, list):
            return [m for m in msgs if isinstance(m, dict)]
        details = (resp.get("result", {}) or {}).get("details")
        if isinstance(details, dict):
            msgs = details.get("messages")
            if isinstance(msgs, list):
                return [m for m in msgs if isinstance(m, dict)]
        return []

    def _recent_discord_titles(
        self,
        *,
        channel_id: str,
        hours: int,
        limit: int,
        run_log: JsonlLogger,
        story_log: JsonlLogger,
    ) -> list[str]:
        if not _is_discord_snowflake(channel_id):
            return []
        limit = int(max(1, min(200, limit)))
        hours = int(max(0, hours))

        resp = self._tool_invoke_retry(
            tool="message",
            action="read",
            args={"channel": "discord", "target": f"channel:{channel_id}", "limit": limit},
            run_log=run_log,
            story_log=story_log,
        )

        msgs = self._messages_from_message_read(resp)
        if not msgs:
            return []

        now = datetime.now(tz=UTC)
        cutoff = now - timedelta(hours=hours)
        titles: list[str] = []
        for m in msgs:
            if not isinstance(m, dict):
                continue
            content = m.get("content")
            if not isinstance(content, str) or not content.strip():
                continue
            ts = m.get("timestamp")
            if not isinstance(ts, str) or not ts.strip():
                continue
            try:
                dt = datetime.fromisoformat(ts.strip())
            except Exception:
                continue
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
            if dt < cutoff:
                continue
            titles.append(self._clean_discord_title(content))
        return titles

    def _semantic_dedupe_against_discord_titles(
        self,
        *,
        job: dict[str, Any],
        run_log: JsonlLogger,
        story_log: JsonlLogger,
        hours: int = 24,
        limit: int = 160,
    ) -> str | None:
        """Backstop semantic dedupe to prevent duplicate posts when planners misbehave.

        This is intentionally cheap:
        - 1 message.read call
        - deterministic title similarity (no LLM)
        """
        dest = job.get("destination", {}) or {}
        if dest.get("platform") != "discord":
            return None
        channel_id = str(dest.get("title_channel_id") or "").strip()
        if not _is_discord_snowflake(channel_id):
            return None

        story = job.get("story", {}) or {}
        title = str(story.get("title") or "").strip()
        if not title:
            return None

        recent_titles = self._recent_discord_titles(channel_id=channel_id, hours=hours, limit=limit, run_log=run_log, story_log=story_log)
        if not recent_titles:
            return None

        m, mt = best_semantic_duplicate(title, recent_titles)
        if m and m.is_duplicate:
            hit = (mt or "").strip()
            return f"semantic_duplicate:title_sim:{m.score:.2f}:of:{hit[:140]}"
        return None

    def _mark_job_skipped_policy(self, *, job_path: Path, job: dict[str, Any], reason: str) -> None:
        state = job.get("state", {}) or {}
        state["status"] = "SKIPPED"
        state["locked_by"] = None
        state["locked_at"] = None
        job["state"] = state

        result = job.get("result", {}) or {}
        result["final_status"] = "SKIPPED_POLICY"
        result.setdefault("errors", []).append({"type": "policy", "message": reason, "at": utc_iso()})
        job["result"] = result

        _, story_log = self._loggers_for_job(job, job_path)
        self._update_job_file(job_path, job, story_log, event="job_skipped", reason=reason)

    def _mark_job_skipped_missing_sources(self, *, job_path: Path, job: dict[str, Any], reason: str) -> None:
        state = job.get("state", {}) or {}
        state["status"] = "SKIPPED"
        state["locked_by"] = None
        state["locked_at"] = None
        job["state"] = state

        result = job.get("result", {}) or {}
        result["final_status"] = "SKIPPED_MISSING_SOURCES"
        result.setdefault("errors", []).append({"type": "missing_sources", "message": reason, "at": utc_iso()})
        job["result"] = result

        _, story_log = self._loggers_for_job(job, job_path)
        self._update_job_file(job_path, job, story_log, event="job_skipped", reason=reason)

    def _post_discord_failure_notice(self, *, job_path: Path, job: dict[str, Any], run_log: JsonlLogger, story_log: JsonlLogger, error_type: str, error_message: str) -> None:
        if self._dry_run:
            return
        dest = job.get("destination", {}) or {}
        if dest.get("platform") != "discord":
            return

        state = job.get("state", {}) or {}
        discord_state = state.get("discord", {}) or {}
        thread_id = discord_state.get("thread_id")
        if not _is_discord_snowflake(thread_id):
            return
        if discord_state.get("failure_notice_message_id"):
            return

        msg = f"（系統）呢單新聞出稿失敗（{error_type}）：{error_message}\n如要我再試一次，reply 呢個 thread 就得。"
        try:
            resp = self._tool_invoke_retry(
                tool="message",
                action="send",
                args={"channel": "discord", "target": f"channel:{thread_id}", "message": msg},
                run_log=run_log,
                story_log=story_log,
            )
            mid = _find_first_key(resp, {"messageId", "message_id"})
            if not mid:
                details = (resp.get("result", {}) or {}).get("details")
                mid = _find_first_key(details, {"messageId", "message_id"})
            if not mid:
                mid = _find_first_key(resp, {"id"})
            if _is_discord_snowflake(mid):
                discord_state["failure_notice_message_id"] = str(mid)
                state["discord"] = discord_state
                job["state"] = state
                self._update_job_file(job_path, job, story_log, event="state_update", field="discord.failure_notice_message_id")
        except Exception as e:
            story_log.log("failure_notice_failed", error=str(e))

    def _publish_script_posts_draft(
        self,
        *,
        job_path: Path,
        job: dict[str, Any],
        result_json: dict[str, Any],
        run_log: JsonlLogger,
        story_log: JsonlLogger,
    ) -> dict[str, Any]:
        """Publish a script_posts draft into the story thread deterministically.

        This keeps LLM tool calls at 0: worker/rescue only returns text, runner does the posting.
        """
        if self._dry_run:
            return result_json

        dest = job.get("destination", {}) or {}
        if dest.get("platform") != "discord":
            raise JobSchemaError("script_posts currently supports destination.platform=discord only")

        state = job.get("state", {}) or {}
        discord_state = state.get("discord", {}) or {}
        thread_id = discord_state.get("thread_id")
        if not _is_discord_snowflake(thread_id):
            raise JobSchemaError("state.discord.thread_id is required before script_posts publishing")
        # Ensure the persisted result JSON includes the real thread id (worker may have seen null).
        result_json["thread_id"] = str(thread_id)

        draft = result_json.get("draft") if isinstance(result_json, dict) else None
        if not isinstance(draft, dict):
            raise ResultParseError("script_posts result missing draft object")

        body = draft.get("body")
        if not isinstance(body, str):
            body = ""
        body = _normalize_report_body(body)
        draft["body"] = body

        read_more_urls = draft.get("read_more_urls")
        if not isinstance(read_more_urls, list):
            read_more_urls = []
        urls = [str(u).strip() for u in read_more_urls if isinstance(u, str) and str(u).strip()]

        body_chunks = _split_discord_messages(body, max_chars=1900)
        read_more_msg = "延伸閱讀\n" + "\n".join(urls)
        outgoing = [*body_chunks, read_more_msg]

        # Idempotency: if we crashed mid-publish, resume from published_message_ids.
        published = discord_state.get("published_message_ids")
        if not isinstance(published, list):
            published = []
        pub_ids: list[str] = []
        for mid in published:
            if isinstance(mid, int):
                mid = str(mid)
            if isinstance(mid, str) and _is_discord_snowflake(mid.strip()):
                pub_ids.append(mid.strip())

        # If we already posted everything, just reuse the ids.
        if len(pub_ids) >= len(outgoing):
            result_json["content_posted"] = True
            result_json["content_message_ids"] = pub_ids[: len(outgoing)]
            # Best-effort: infer attachment count from persisted assets state (no network).
            planned_images = 0
            try:
                state = job.get("state", {}) or {}
                assets = state.get("assets") if isinstance(state.get("assets"), dict) else None
                attachments_obj = assets.get("attachments") if isinstance(assets, dict) else None

                chart_path = None
                card_path = None
                infographic_path = None
                og_path = None
                if isinstance(attachments_obj, dict):
                    cps = attachments_obj.get("chart_paths")
                    if isinstance(cps, list):
                        for p in cps:
                            if isinstance(p, str) and p.strip() and Path(p.strip()).exists():
                                chart_path = p.strip()
                                break
                    kps = attachments_obj.get("card_paths")
                    if isinstance(kps, list):
                        for p in kps:
                            if isinstance(p, str) and p.strip() and Path(p.strip()).exists():
                                card_path = p.strip()
                                break
                    ips = attachments_obj.get("infographic_paths")
                    if isinstance(ips, list):
                        for p in ips:
                            if isinstance(p, str) and p.strip() and Path(p.strip()).exists():
                                infographic_path = p.strip()
                                break
                    ops = attachments_obj.get("og_image_paths")
                    if isinstance(ops, list):
                        for p in ops:
                            if isinstance(p, str) and p.strip() and Path(p.strip()).exists():
                                og_path = p.strip()
                                break

                story = job.get("story", {}) or {}
                category = str(story.get("category") or "").strip()
                finance_mode = category in {"US Stocks", "Crypto", "Precious Metals"}
                hero_path = card_path or infographic_path or og_path
                if finance_mode and chart_path:
                    planned_images += 1
                if hero_path:
                    planned_images += 1
            except Exception:
                planned_images = 0
            result_json["images_attached_count"] = planned_images
            result_json["read_more_urls_count"] = len(urls)
            result_json["report_char_count"] = _count_cjk(body)
            return result_json

        # Ensure deterministic assets exist before we post anything.
        # (Worker already saw the assets during generation; here we only need local paths for attachments.)
        try:
            self._ensure_assets_pack(job_path=job_path, job=job, run_log=run_log, story_log=story_log)
        except Exception as e:
            story_log.log("assets_pack_ensure_failed", error=str(e))

        state = job.get("state", {}) or {}
        assets = state.get("assets") if isinstance(state.get("assets"), dict) else None
        attachments_obj = assets.get("attachments") if isinstance(assets, dict) else None

        chart_path: str | None = None
        if isinstance(attachments_obj, dict):
            cps = attachments_obj.get("chart_paths")
            if isinstance(cps, list):
                for p in cps:
                    if isinstance(p, str) and p.strip() and Path(p.strip()).exists():
                        chart_path = p.strip()
                        break

        # Optional: generate/attach one card OR one infographic (at most one generated image).
        draft_obj = result_json.get("draft") if isinstance(result_json, dict) else None
        card_prompt_requested = bool(
            isinstance(draft_obj, dict) and isinstance(draft_obj.get("card_prompt"), str) and str(draft_obj.get("card_prompt")).strip()
        )
        card_path: str | None = None
        try:
            card_paths = self._ensure_card_paths(
                job_path=job_path,
                job=job,
                result_json=result_json,
                run_log=run_log,
                story_log=story_log,
                max_generate_seconds=120,
            )
            for p in card_paths:
                if isinstance(p, str) and p.strip() and Path(p.strip()).exists():
                    card_path = p.strip()
                    break
        except Exception as e:
            story_log.log("card_ensure_failed", error=str(e))
            card_path = None

        # Optional: generate/attach one infographic if requested by the worker (and no card exists).
        infographic_path: str | None = None
        if card_prompt_requested:
            story_log.log("infographic_skipped", reason="card_prompt_requested")
            infographic_path = None
        elif card_path is None:
            try:
                info_paths = self._ensure_infographic_paths(
                    job_path=job_path,
                    job=job,
                    result_json=result_json,
                    run_log=run_log,
                    story_log=story_log,
                    max_generate_seconds=120,
                )
                for p in info_paths:
                    if isinstance(p, str) and p.strip() and Path(p.strip()).exists():
                        infographic_path = p.strip()
                        break
            except Exception as e:
                story_log.log("infographic_ensure_failed", error=str(e))
                infographic_path = None

        og_path: str | None = None
        if card_path is None and infographic_path is None:
            # Best-effort: download one OG image for attachment (no screenshots).
            og_paths: list[str] = []
            try:
                og_paths = self._ensure_og_image_paths(
                    job_path=job_path,
                    job=job,
                    run_log=run_log,
                    story_log=story_log,
                    max_downloads=1,
                )
            except Exception as e:
                story_log.log("og_paths_ensure_failed", error=str(e))
                og_paths = []

            for p in og_paths:
                if isinstance(p, str) and p.strip() and Path(p.strip()).exists():
                    og_path = p.strip()
                    break

        # Attachment plan by outgoing message index (at most 1 file per message).
        story = job.get("story", {}) or {}
        category = str(story.get("category") or "").strip()
        finance_mode = category in {"US Stocks", "Crypto", "Precious Metals"}
        hero_path = card_path or infographic_path or og_path
        attachment_by_idx: dict[int, str] = {}
        start_idx = len(pub_ids)

        def _pick_idx(preferred: int) -> int | None:
            idx = int(preferred)
            if idx < start_idx:
                idx = start_idx
            # Avoid collisions.
            while idx < len(outgoing) and idx in attachment_by_idx:
                idx += 1
            if 0 <= idx < len(outgoing):
                return idx
            return None

        if finance_mode:
            if chart_path:
                idx = _pick_idx(0)
                if idx is not None:
                    attachment_by_idx[idx] = chart_path
            if hero_path:
                # Prefer 2nd body chunk; fallback to read-more message.
                preferred = 1 if len(body_chunks) >= 2 else (len(outgoing) - 1)
                idx = _pick_idx(preferred)
                if idx is not None:
                    attachment_by_idx[idx] = hero_path
        else:
            if hero_path:
                idx = _pick_idx(0)
                if idx is not None:
                    attachment_by_idx[idx] = hero_path

        # Publish remaining messages in order.
        planned_images = len(set(attachment_by_idx.values()))
        for idx in range(len(pub_ids), len(outgoing)):
            msg = outgoing[idx]
            args: dict[str, Any] = {"channel": "discord", "target": f"channel:{thread_id}", "message": msg}
            fp = attachment_by_idx.get(idx)
            if isinstance(fp, str) and fp.strip():
                args["filePath"] = fp.strip()
            resp = self._tool_invoke_retry(
                tool="message",
                action="send",
                args=args,
                run_log=run_log,
                story_log=story_log,
            )
            mid = _find_first_key(resp, {"messageId", "message_id"})
            if not mid:
                details = (resp.get("result", {}) or {}).get("details")
                mid = _find_first_key(details, {"messageId", "message_id"})
            if not mid:
                mid = _find_first_key(resp, {"id"})
            if not _is_discord_snowflake(mid):
                raise RuntimeError("Unable to extract message_id from message.send response (script_posts)")

            pub_ids.append(str(mid))
            discord_state["published_message_ids"] = list(pub_ids)
            state["discord"] = discord_state
            job["state"] = state
            self._update_job_file(job_path, job, story_log, event="discord_publish_progress", posted=len(pub_ids), total=len(outgoing))

        # Finalize result JSON with real Discord ids + computed counts.
        result_json["content_posted"] = True
        result_json["content_message_ids"] = pub_ids
        result_json["images_attached_count"] = planned_images
        result_json["read_more_urls_count"] = len(urls)
        result_json["report_char_count"] = _count_cjk(body)
        return result_json

    def _validate_story_job(self, job: dict[str, Any]) -> None:
        try:
            jsonschema.validate(instance=job, schema=self._story_schema)
        except jsonschema.ValidationError as e:
            raise JobSchemaError(f"story job schema validation failed: {e.message}") from e
        if job.get("schema_version") != "story_job_v1":
            raise JobSchemaError("schema_version must be story_job_v1")

    def _coerce_story_job_format(self, job_path: Path, job: dict[str, Any], run_defaults: dict[str, Any]) -> dict[str, Any]:
        """Coerce legacy/minimal planner output into `story_job_v1` when possible.

        We've observed some cron/planner agents emit a minimal JSON shape like:
          {"run": {"id": "...", "type": "story_job_v1", ...}, "story": {...}}
        which is not the Newsroom schema. For robustness, we deterministically
        upgrade that shape into a full `story_job_v1` job using our example as
        a template.
        """
        if job.get("schema_version") == "story_job_v1":
            # Planners sometimes emit `null` for optional strings (e.g. instructions),
            # which fails strict JSON schema validation. Sanitize those here so the
            # runner can proceed deterministically.
            story = job.get("story")
            if isinstance(story, dict):
                if story.get("instructions") is None:
                    story.pop("instructions", None)
                if story.get("dedupe_key") is None:
                    story.pop("dedupe_key", None)
                # Some planners emit null for required arrays; coerce to empty.
                if story.get("supporting_urls") is None:
                    story["supporting_urls"] = []
                if story.get("flags") is None:
                    story["flags"] = []
                job["story"] = story

            dest = job.get("destination")
            if isinstance(dest, dict):
                if dest.get("thread_name_template") is None:
                    dest.pop("thread_name_template", None)

                # Best-effort: infer missing title_channel_id from run_id convention.
                if dest.get("title_channel_id") is None:
                    run_id = ((job.get("run", {}) or {}).get("run_id")) or job_path.parent.name
                    m = re.match(r"discord-(\\d+)-", str(run_id))
                    if m:
                        dest["title_channel_id"] = m.group(1)

                if dest.get("title_channel_id") is not None:
                    dest["title_channel_id"] = str(dest["title_channel_id"])
                job["destination"] = dest

            spawn = job.get("spawn")
            if isinstance(spawn, dict):
                if spawn.get("agent_id") is None:
                    spawn["agent_id"] = "main"
                job["spawn"] = spawn

            return job

        # If it's some other schema version, let validation fail loudly.
        if "schema_version" in job and job.get("schema_version") != "story_job_v1":
            return job

        run = job.get("run")
        story = job.get("story")
        if not isinstance(run, dict) or not isinstance(story, dict):
            return job

        # Recognise legacy planner output.
        run_type = str(run.get("type") or "").strip()
        if run_type != "story_job_v1":
            return job

        run_id = str(run.get("id") or job_path.parent.name or job_path.stem).strip()
        if not run_id:
            return job

        # Infer Discord title channel id.
        title_channel_id: str | None = None
        params = run.get("params")
        if isinstance(params, dict):
            cid = params.get("channel_id") or params.get("title_channel_id")
            if cid is not None:
                title_channel_id = str(cid).strip()
        if not title_channel_id:
            m = re.match(r"discord-(\\d+)-", run_id)
            if m:
                title_channel_id = m.group(1)

        # Build a full job from the example template.
        tmpl_path = self._openclaw_home / "newsroom" / "examples" / "story_job_example.json"
        tmpl = load_json_file(tmpl_path)
        upgraded: dict[str, Any] = copy.deepcopy(tmpl)

        # Run block.
        upgraded_run = upgraded.get("run", {}) or {}
        upgraded_run["run_id"] = run_id
        upgraded_run["trigger"] = str(run_defaults.get("trigger") or "cron_hourly")

        # Prefer the legacy created_at when present, otherwise use now.
        run_time_uk: str | None = None
        created_at = run.get("created_at")
        if isinstance(created_at, str) and created_at.strip():
            try:
                dt = datetime.fromisoformat(created_at.strip())
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=UTC)
                run_time_uk = dt.astimezone(ZoneInfo("Europe/London")).strftime("%Y-%m-%d %H:%M")
            except Exception:
                run_time_uk = None
        if not run_time_uk:
            run_time_uk = time.strftime("%Y-%m-%d %H:%M", time.gmtime())
        upgraded_run["run_time_uk"] = run_time_uk
        upgraded["run"] = upgraded_run

        # Story block.
        upgraded_story = upgraded.get("story", {}) or {}
        upgraded_story["story_id"] = str(story.get("story_id") or upgraded_story.get("story_id") or "story_01")
        upgraded_story["content_type"] = str(story.get("content_type") or upgraded_story.get("content_type") or "news_deep_dive")
        upgraded_story["category"] = str(story.get("category") or upgraded_story.get("category") or "Global News")
        upgraded_story["title"] = str(story.get("title") or upgraded_story.get("title") or "").strip()
        upgraded_story["primary_url"] = str(story.get("primary_url") or "").strip()
        upgraded_story["supporting_urls"] = list(story.get("supporting_urls") or [])
        upgraded_story["concrete_anchor"] = str(story.get("concrete_anchor") or "").strip()
        upgraded_story["flags"] = list(story.get("flags") or [])
        if isinstance(story.get("dedupe_key"), str) and story.get("dedupe_key").strip():
            upgraded_story["dedupe_key"] = story.get("dedupe_key").strip()
        upgraded["story"] = upgraded_story

        # Destination defaults (Discord).
        dest = upgraded.get("destination", {}) or {}
        dest["platform"] = "discord"
        if title_channel_id:
            dest["title_channel_id"] = title_channel_id
        upgraded["destination"] = dest

        # Ensure we use the latest inline reporter by default.
        spawn = upgraded.get("spawn", {}) or {}
        spawn["prompt_id"] = str(spawn.get("prompt_id") or "news_reporter_v2_2_inline")
        spawn["agent_id"] = str(spawn.get("agent_id") or "main")
        spawn["publisher_mode"] = str(spawn.get("publisher_mode") or "agent_posts")
        upgraded["spawn"] = spawn

        # Reset state/result to a clean PLANNED job (runner will progress it).
        state = upgraded.get("state", {}) or {}
        state["status"] = "PLANNED"
        state["locked_by"] = None
        state["locked_at"] = None
        state.setdefault("discord", {"title_message_id": None, "thread_id": None})
        state.setdefault("worker", {"attempt": 0, "child_session_key": None, "run_id": None, "started_at": None, "ended_at": None})
        state.setdefault("rescue", {"attempt": 0, "child_session_key": None, "started_at": None, "ended_at": None})
        upgraded["state"] = state

        result = upgraded.get("result", {}) or {}
        result["final_status"] = None
        result["worker_result_json"] = None
        result["rescue_result_json"] = None
        result["errors"] = []
        upgraded["result"] = result

        return upgraded

    def _validate_run_job(self, job: dict[str, Any]) -> None:
        try:
            jsonschema.validate(instance=job, schema=self._run_schema)
        except jsonschema.ValidationError as e:
            raise JobSchemaError(f"run job schema validation failed: {e.message}") from e
        if job.get("schema_version") != "run_job_v1":
            raise JobSchemaError("schema_version must be run_job_v1")

    def _resolve_prompt_and_validator(self, job: dict[str, Any]) -> tuple[PromptDef, ValidatorDef]:
        spawn = job.get("spawn", {}) or {}
        prompt_id = spawn.get("prompt_id")
        if not prompt_id:
            content_type = (job.get("story", {}) or {}).get("content_type", "")
            prompt_id = self._prompt_registry.default_prompt_for_content_type(str(content_type))
            if not prompt_id:
                raise PromptRegistryError("Missing spawn.prompt_id and no content_type default available")
            spawn["prompt_id"] = prompt_id
            job["spawn"] = spawn

        prompt_def = self._prompt_registry.resolve_prompt(str(prompt_id))

        validation = job.get("validation", {}) or {}
        validator_id = validation.get("validator_id") or prompt_def.validator_id
        validation["validator_id"] = validator_id
        job["validation"] = validation
        validator_def = self._prompt_registry.resolve_validator(str(validator_id))

        return prompt_def, validator_def

    def _get_validator(self, validator_def: ValidatorDef) -> Callable[[dict[str, Any], dict[str, Any]], Any]:
        cached = self._validator_cache.get(validator_def.validator_id)
        if cached:
            return cached
        fn = _load_validator_callable(validator_def.path)
        self._validator_cache[validator_def.validator_id] = fn
        return fn

    def _loggers_for_job(self, job: dict[str, Any], job_path: Path) -> tuple[JsonlLogger, JsonlLogger]:
        run_id = (job.get("run", {}) or {}).get("run_id", "unknown-run")
        story_id = (job.get("story", {}) or {}).get("story_id", job_path.stem)
        run_dir = self._log_root / str(run_id)
        return JsonlLogger(run_dir / "run.jsonl"), JsonlLogger(run_dir / f"{story_id}.jsonl")

    def _update_job_file(self, job_path: Path, job: dict[str, Any], story_log: JsonlLogger, *, event: str, **fields: Any) -> None:
        if self._dry_run:
            story_log.log(event, job_path=str(job_path), dry_run=True, **fields)
            return
        atomic_write_json(job_path, job)
        story_log.log(event, job_path=str(job_path), dry_run=False, **fields)

    def _tool_invoke_retry(
        self,
        *,
        tool: str,
        action: str | None,
        args: dict[str, Any],
        run_log: JsonlLogger,
        story_log: JsonlLogger,
        retries: int = 4,
        base_sleep: float = 1.0,
    ) -> dict[str, Any]:
        # Crude retry loop; the gateway normalizes many provider errors but message sends can still 429.
        last_err: Exception | None = None
        for attempt in range(retries):
            try:
                resp = self._gateway.invoke(tool=tool, action=action, args=args)
                story_log.log("tool_ok", tool=tool, action=action, attempt=attempt + 1)
                return resp
            except Exception as e:
                last_err = e
                sleep_s = base_sleep * (2**attempt)
                run_log.log("tool_retry", tool=tool, action=action, attempt=attempt + 1, sleep_s=sleep_s, error=str(e))
                story_log.log("tool_retry", tool=tool, action=action, attempt=attempt + 1, sleep_s=sleep_s, error=str(e))
                time.sleep(sleep_s)
        raise RuntimeError(f"Tool failed after retries: {tool} action={action} err={last_err}")

    def _tool_invoke_result_json_retry(
        self,
        *,
        tool: str,
        action: str | None,
        args: dict[str, Any],
        run_log: JsonlLogger,
        story_log: JsonlLogger,
        retries: int = 4,
        base_sleep: float = 1.0,
    ) -> dict[str, Any]:
        last_err: Exception | None = None
        for attempt in range(retries):
            try:
                resp = self._gateway.invoke_result_json(tool=tool, action=action, args=args)
                story_log.log("tool_ok", tool=tool, action=action, attempt=attempt + 1)
                return resp
            except Exception as e:
                last_err = e
                sleep_s = base_sleep * (2**attempt)
                run_log.log("tool_retry", tool=tool, action=action, attempt=attempt + 1, sleep_s=sleep_s, error=str(e))
                story_log.log("tool_retry", tool=tool, action=action, attempt=attempt + 1, sleep_s=sleep_s, error=str(e))
                time.sleep(sleep_s)
        raise RuntimeError(f"Tool failed after retries: {tool} action={action} err={last_err}")

    def _augment_success_result_with_discord_metrics(
        self,
        *,
        job: dict[str, Any],
        result_json: dict[str, Any],
        run_log: JsonlLogger,
        story_log: JsonlLogger,
    ) -> dict[str, Any]:
        """Augment SUCCESS result JSON using Discord message truth when possible.

        The worker self-reports counts like report_char_count/read_more_urls_count.
        In practice these are easy to miscount. To make the runner deterministic,
        we recompute basic metrics from the posted Discord messages.
        """
        if result_json.get("status") != "SUCCESS":
            return result_json

        dest = job.get("destination", {}) or {}
        if dest.get("platform") != "discord":
            return result_json

        thread_id = result_json.get("thread_id") or (job.get("state", {}) or {}).get("discord", {}).get("thread_id")
        if not _is_discord_snowflake(thread_id):
            return result_json

        msg_ids = result_json.get("content_message_ids")
        if not isinstance(msg_ids, list) or not all(isinstance(x, str) and x.strip() for x in msg_ids):
            return result_json

        # Read enough recent thread history to cover our own posts (threads should be short).
        resp = self._tool_invoke_retry(
            tool="message",
            action="read",
            args={"channel": "discord", "target": f"channel:{thread_id}", "limit": 120},
            run_log=run_log,
            story_log=story_log,
        )
        details = (resp.get("result", {}) or {}).get("details")
        if not isinstance(details, dict):
            return result_json
        messages = details.get("messages", [])
        if not isinstance(messages, list):
            return result_json

        by_id: dict[str, dict[str, Any]] = {}
        for m in messages:
            if not isinstance(m, dict):
                continue
            mid = m.get("id")
            if _is_discord_snowflake(mid):
                by_id[str(mid)] = m

        selected: list[dict[str, Any]] = []
        for mid in msg_ids:
            m = by_id.get(str(mid))
            if m:
                selected.append(m)

        if not selected:
            return result_json

        images = 0
        report_parts: list[str] = []
        read_more_urls: set[str] = set()

        for m in selected:
            attachments = m.get("attachments", [])
            if isinstance(attachments, list):
                # Count total attachments (assumed images in our workflow).
                images += sum(1 for a in attachments if isinstance(a, dict))

            content = str(m.get("content") or "")
            if "延伸閱讀" in content:
                idx = content.find("延伸閱讀")
                # Treat everything after the marker as the "Read more" block.
                tail = content[idx:]
                for url in _URL_RE.findall(tail):
                    read_more_urls.add(url)
                head = content[:idx].strip()
                if head:
                    report_parts.append(head)
                continue
            report_parts.append(content)

        # Backstop: if the worker forgot to include the read-more message id in
        # content_message_ids, scan the whole thread batch for a "延伸閱讀" block.
        if not read_more_urls:
            for m in messages:
                if not isinstance(m, dict):
                    continue
                content = str(m.get("content") or "")
                if "延伸閱讀" not in content:
                    continue
                idx = content.find("延伸閱讀")
                tail = content[idx:]
                for url in _URL_RE.findall(tail):
                    read_more_urls.add(url)

        report_text = "\n".join(report_parts)
        report_text = _strip_urls(report_text)
        report_cjk = _count_cjk(report_text)

        # Mutate in place (the caller persists result_json into the job file).
        result_json["images_attached_count"] = images
        result_json["read_more_urls_count"] = len(read_more_urls)
        if report_cjk:
            result_json["report_char_count"] = report_cjk

        story_log.log(
            "discord_metrics",
            thread_id=str(thread_id),
            computed_images=images,
            computed_read_more_urls_count=len(read_more_urls),
            computed_report_cjk=report_cjk,
        )
        return result_json

    def _prepare_discord(self, job_path: Path, job: dict[str, Any], run_log: JsonlLogger, story_log: JsonlLogger) -> None:
        state = job.get("state", {}) or {}
        discord_state = state.get("discord", {}) or {}

        title_channel_id = (job.get("destination", {}) or {}).get("title_channel_id")
        if not title_channel_id:
            raise JobSchemaError("destination.title_channel_id is required for discord destination")

        story = job.get("story", {}) or {}
        title = story.get("title")
        if not isinstance(title, str) or not title.strip():
            raise JobSchemaError("story.title is required")
        title = title.strip()

        # Backstop: planners sometimes forget to translate the title. Translate it once here
        # so the title message + thread name are correct and workers inherit the right title.
        if _looks_like_english_title(title):
            maybe_new = self._translate_title_cantonese(
                job_path=job_path,
                job=job,
                old_title=title,
                run_log=run_log,
                story_log=story_log,
            )
            if isinstance(maybe_new, str) and maybe_new.strip() and maybe_new.strip() != title:
                title = maybe_new.strip()
                story["title"] = title
                job["story"] = story
                self._update_job_file(job_path, job, story_log, event="title_translate_applied", title=title)
                # If we already posted the title message, update it before creating the thread.
                if discord_state.get("title_message_id"):
                    try:
                        self._tool_invoke_retry(
                            tool="message",
                            action="edit",
                            args={
                                "channel": "discord",
                                "target": f"channel:{title_channel_id}",
                                "messageId": str(discord_state["title_message_id"]),
                                "message": title,
                            },
                            run_log=run_log,
                            story_log=story_log,
                        )
                        story_log.log("title_edit_ok_prethread", title_message_id=str(discord_state["title_message_id"]))
                    except Exception as e:
                        story_log.log("title_edit_failed_prethread", error=str(e), title_message_id=str(discord_state["title_message_id"]))

        # Idempotency: if we already have a thread_id, skip.
        if discord_state.get("thread_id"):
            return

        if self._dry_run:
            story_log.log("discord_prepare_dry_run", title_channel_id=title_channel_id, title=title)
            return

        # 1) Post title message if needed.
        title_message_id = discord_state.get("title_message_id")
        if not title_message_id:
            resp = self._tool_invoke_retry(
                tool="message",
                action="send",
                args={"channel": "discord", "target": f"channel:{title_channel_id}", "message": str(title)},
                run_log=run_log,
                story_log=story_log,
            )
            msg_id = _find_first_key(resp, {"messageId", "message_id"})
            if not msg_id:
                details = (resp.get("result", {}) or {}).get("details")
                msg_id = _find_first_key(details, {"messageId", "message_id"})
            if not msg_id:
                # Last resort: some backends only return `id`.
                msg_id = _find_first_key(resp, {"id"})
                if not _is_discord_snowflake(msg_id):
                    msg_id = None
            if not msg_id:
                raise RuntimeError("Unable to extract title_message_id from message.send response")
            discord_state["title_message_id"] = str(msg_id)
            state["discord"] = discord_state
            job["state"] = state
            self._update_job_file(job_path, job, story_log, event="state_update", field="discord.title_message_id")

        # 2) Create thread.
        thread_name_tmpl = (job.get("destination", {}) or {}).get("thread_name_template", "{title}")
        thread_name = str(thread_name_tmpl).replace("{title}", str(title))[:100]
        resp = self._tool_invoke_retry(
            tool="message",
            action="thread-create",
            args={
                "channel": "discord",
                "channelId": str(title_channel_id),
                "messageId": str(discord_state["title_message_id"]),
                "threadName": thread_name,
            },
            run_log=run_log,
            story_log=story_log,
        )
        thread_id = _find_first_key(resp, {"threadId", "thread_id"})
        if not thread_id:
            details = (resp.get("result", {}) or {}).get("details")
            thread_id = _find_first_key(details, {"threadId", "thread_id"})
        if not thread_id:
            # Some backends may return the created thread channel as `channelId`.
            candidate = _find_first_key(resp, {"channelId", "channel_id"})
            if _is_discord_snowflake(candidate) and str(candidate) != str(title_channel_id):
                thread_id = candidate
        if not thread_id:
            details = (resp.get("result", {}) or {}).get("details")
            candidate = _find_first_key(details, {"channelId", "channel_id"})
            if _is_discord_snowflake(candidate) and str(candidate) != str(title_channel_id):
                thread_id = candidate
        if not thread_id:
            # Last resort: some backends only return `id`.
            thread_id = _find_first_key(resp, {"id"})
            if not _is_discord_snowflake(thread_id):
                thread_id = None
        if not thread_id:
            raise RuntimeError("Unable to extract thread_id from message.thread-create response")
        if str(thread_id) == str(title_channel_id):
            raise RuntimeError("thread_id extraction resolved to the title channel id; refusing to post to the parent channel")

        discord_state["thread_id"] = str(thread_id)
        state["discord"] = discord_state
        state["status"] = "PREPARED"
        job["state"] = state
        self._update_job_file(job_path, job, story_log, event="state_update", field="discord.thread_id")

    def _translate_title_cantonese(
        self,
        *,
        job_path: Path,
        job: dict[str, Any],
        old_title: str,
        run_log: JsonlLogger,
        story_log: JsonlLogger,
    ) -> str | None:
        """Translate an English title into Traditional Chinese (Cantonese phrasing).

        This is a lightweight, best-effort repair to keep title channel/thread names in Cantonese
        even when the planner forgets. It is intentionally small compared to a full rescue run.
        """
        if self._dry_run:
            return None

        story = job.get("story", {}) or {}
        run = job.get("run", {}) or {}
        run_id = str(run.get("run_id") or job_path.parent.name or job_path.stem).strip() or "run"
        story_id = str(story.get("story_id") or job_path.stem).strip() or "story"
        category = str(story.get("category") or "").strip()
        primary_url = str(story.get("primary_url") or "").strip()

        # Keep the translation prompt short to reduce tokens.
        task = "\n".join(
            [
                "You translate a news headline into Traditional Chinese (Hong Kong Cantonese phrasing).",
                "",
                "Rules:",
                "- Keep proper nouns / tickers unchanged (e.g. SpaceX, xAI, Apple, TSLA).",
                "- Preserve numbers and units.",
                "- Keep it short for Discord (aim <= 60 chars).",
                "- Do NOT add extra facts beyond the title.",
                "",
                f"Category: {category}",
                f"Primary URL: {primary_url}",
                f"Title: {old_title}",
                "",
                'Output ONLY this JSON and nothing else: {"status":"SUCCESS","title":"..."}',
            ]
        ).strip() + "\n"

        digest = hashlib.sha1(f"{run_id}:{story_id}:title".encode("utf-8")).hexdigest()[:12]
        label = f"nr:{story_id}:title:{digest}"

        started_at = utc_iso()
        spawn = job.get("spawn", {}) or {}
        # NOTE: sessions_spawn currently only permits agentId="main" in this deployment.
        # Keep this task short so even the default agent stays cheap.
        agent_id = str(spawn.get("title_agent_id") or "main").strip() or "main"

        resp = self._gateway.invoke_result_json(
            tool="sessions_spawn",
            action="json",
            args={
                "task": task,
                "label": label,
                "agentId": agent_id,
                "runTimeoutSeconds": int(self._TITLE_TRANSLATE_TIMEOUT_SECONDS),
                # Keep session so we can reliably read the translation result via sessions_history.
                "cleanup": "keep",
            },
        )

        child_key = _find_first_key(resp, {"childSessionKey", "child_session_key"})
        run_id_resp = _find_first_key(resp, {"runId", "run_id"})

        state = job.get("state", {}) or {}
        tt = state.get("title_translate", {}) or {}
        tt["child_session_key"] = child_key
        tt["run_id"] = run_id_resp
        tt["started_at"] = started_at

        # Best-effort sessionId capture (strict).
        if isinstance(child_key, str) and child_key.strip():
            try:
                lst = self._gateway.invoke_result_json(
                    tool="sessions_list",
                    action="json",
                    args={"limit": 25, "sessionKey": child_key.strip()},
                )
                sessions = lst.get("sessions", [])
                if isinstance(sessions, list):
                    for s in sessions:
                        if isinstance(s, dict) and s.get("key") == child_key.strip():
                            sid = s.get("sessionId")
                            if isinstance(sid, str) and sid.strip():
                                tt["session_id"] = sid.strip()
                            break
            except Exception as e:
                story_log.log("title_translate_session_id_lookup_failed", error=str(e), childSessionKey=child_key)

        state["title_translate"] = tt
        job["state"] = state
        self._update_job_file(job_path, job, story_log, event="state_update", field="title_translate.child_session_key")

        if not isinstance(child_key, str) or not child_key.strip():
            story_log.log("title_translate_spawn_missing_key", response=resp)
            return None

        deadline = time.time() + float(self._TITLE_TRANSLATE_TIMEOUT_SECONDS)
        while time.time() < deadline:
            hist = self._gateway.invoke_result_json(
                tool="sessions_history",
                action="json",
                args={"sessionKey": child_key.strip(), "limit": 20, "includeTools": False},
            )
            messages = hist.get("messages", [])
            if not isinstance(messages, list):
                messages = []
            result_json = find_result_json_in_messages(messages)
            if not result_json:
                time.sleep(1)
                continue
            new_title = result_json.get("title")
            if not isinstance(new_title, str) or not new_title.strip():
                break
            new_title = new_title.strip()
            if _looks_like_english_title(new_title) or _count_cjk(new_title) < 4:
                story_log.log("title_translate_bad_result", title=new_title)
                break

            tt["ended_at"] = utc_iso()
            state["title_translate"] = tt
            job["state"] = state
            self._update_job_file(job_path, job, story_log, event="state_update", field="title_translate.ended_at")
            story_log.log("title_translate_ok", from_title=old_title, to_title=new_title)
            run_log.log("title_translate_ok", story_id=story_id)
            return new_title

        story_log.log("title_translate_failed_or_timeout", title=old_title)
        return None

    def _prepare_webhook(self, job_path: Path, job: dict[str, Any], run_log: JsonlLogger, story_log: JsonlLogger) -> None:
        # Stub publisher: kept intentionally minimal so webhook publishing can be added later
        # without changing the runner orchestration model.
        if self._dry_run:
            story_log.log("webhook_prepare_dry_run", webhook_url=(job.get("destination", {}) or {}).get("webhook_url"))
            return
        raise JobSchemaError("destination.platform=webhook is not implemented yet (stub publisher present).")

    def _render_worker_task(self, job: dict[str, Any], prompt_def: PromptDef, *, worker_error: dict[str, str] | None = None) -> str:
        template_text = prompt_def.template_path.read_text(encoding="utf-8")

        input_obj = build_worker_input(job)
        # Inject a compact, runner-generated source pack so the worker can avoid
        # expensive browsing/search tool calls. Stored under state.source_pack.
        state = job.get("state", {}) or {}
        source_pack = state.get("source_pack")
        if isinstance(source_pack, dict):
            input_obj["sources_pack"] = source_pack
        assets = state.get("assets")
        if isinstance(assets, dict):
            input_obj["assets"] = assets

        # Use real Unicode so Cantonese titles don't balloon into \\uXXXX escapes.
        input_json = json.dumps(input_obj, ensure_ascii=False, indent=2)
        vars_map = {
            "INPUT_JSON": input_json,
            "SKILL_PATH": str(self._openclaw_home / "workspace" / "skills" / "news-reporter" / "SKILL.md"),
        }
        if worker_error:
            vars_map["WORKER_ERROR_TYPE"] = worker_error.get("error_type", "") or ""
            vars_map["WORKER_ERROR_MESSAGE"] = worker_error.get("error_message", "") or ""
        return _render_template(template_text, vars_map)

    def _backfill_urls_from_pool(
        self,
        *,
        existing_norm_urls: set[str],
        focus_terms_local: set[str],
        focus_anchors: set[str],
        used_domains: set[str],
        max_add: int = 5,
        hours: int = 48,
    ) -> list[str]:
        """Find extra supporting URLs from the local 48h SQLite pool (0 API calls)."""
        max_add = int(max(0, max_add))
        if max_add <= 0:
            return []

        db_path = self._openclaw_home / "data" / "newsroom" / "news_pool.sqlite3"
        if not db_path.exists():
            return []

        cutoff_ts = int(time.time() - int(max(1, hours)) * 3600)
        try:
            with NewsPoolDB(path=db_path) as db:
                links = list(db.iter_links_since(cutoff_ts=cutoff_ts))
        except Exception:
            return []

        scored: list[tuple[int, float, str, str]] = []
        for r in links:
            if not isinstance(r, dict):
                continue
            u = r.get("url") or r.get("norm_url") or r.get("normUrl")
            if not isinstance(u, str) or not u.strip().startswith("http"):
                continue
            u = u.strip()
            try:
                norm = normalize_url(u)
            except Exception:
                norm = u
            if norm in existing_norm_urls:
                continue

            dom = str(r.get("domain") or "").strip().lower()
            if not dom:
                try:
                    dom = (urllib.parse.urlsplit(u).hostname or "").lower()
                except Exception:
                    dom = ""
            if not dom:
                continue

            # Skip common paywalls/bot-protected sites.
            if dom.endswith(("reuters.com", "bloomberg.com", "wsj.com", "ft.com")):
                continue
            if dom in {"x.com", "twitter.com", "facebook.com", "youtube.com", "youtu.be", "tiktok.com", "instagram.com"}:
                continue

            r_title = r.get("title") if isinstance(r.get("title"), str) else None
            r_desc = r.get("description") if isinstance(r.get("description"), str) else None

            hit_terms = 0
            if focus_terms_local:
                try:
                    hit_terms = len(tokenize_text(r_title, r_desc).intersection(focus_terms_local))
                except Exception:
                    hit_terms = 0

            hit_anchors = 0
            if focus_anchors:
                try:
                    hit_anchors = len(anchor_terms(r_title, r_desc).intersection(focus_anchors))
                except Exception:
                    hit_anchors = 0

            # Require a stronger match when we have anchors, otherwise locality/outlet words
            # can drag in unrelated URLs (bad "read more" links).
            if focus_anchors:
                if hit_anchors < 1 or hit_terms < 2:
                    continue
            else:
                if hit_terms < 3:
                    continue

            dom_bonus = 0.25 if dom not in used_domains else 0.0
            score = int(hit_terms + (2 * hit_anchors))
            scored.append((score, dom_bonus, dom, u))

        added: list[str] = []
        for _score, _dom_bonus, dom, u in sorted(scored, key=lambda t: (-t[0], -t[1], t[2], t[3])):
            try:
                norm = normalize_url(u)
            except Exception:
                norm = u
            if norm in existing_norm_urls:
                continue
            existing_norm_urls.add(norm)
            added.append(u)
            used_domains.add(dom)
            if len(added) >= max_add:
                break
        return added

    def _ensure_source_pack(self, *, job_path: Path, job: dict[str, Any], story_log: JsonlLogger) -> None:
        if self._dry_run:
            return
        state = job.get("state", {}) or {}
        if isinstance(state.get("source_pack"), dict):
            return

        story = job.get("story", {}) or {}
        story_id = str(story.get("story_id") or job_path.stem).strip() or "story"
        primary_url = str(story.get("primary_url") or "").strip()
        supporting_urls = story.get("supporting_urls") or []
        urls: list[str] = []
        if primary_url:
            urls.append(primary_url)
        if isinstance(supporting_urls, list):
            for u in supporting_urls:
                if isinstance(u, str) and u.strip():
                    urls.append(u.strip())

        # On-topic scoring must work even when the story title is Cantonese but sources are English.
        # Include URL path tokens (often contain English entity words) so focus_terms can match.
        focus_parts: list[str] = []
        focus_parts.append(str(story.get("title") or "").strip())
        focus_parts.append(str(story.get("concrete_anchor") or "").strip())
        for u in urls:
            try:
                p = urllib.parse.urlsplit(str(u))
                if p.path:
                    focus_parts.append(p.path.replace("-", " ").replace("_", " "))
            except Exception:
                continue
        focus_text = " ".join([p for p in focus_parts if p]).strip() or None

        try:
            pack = build_source_pack(
                story_id=story_id,
                urls=urls,
                openclaw_home=self._openclaw_home,
                cache_hours=48,
                focus_text=focus_text,
                # Give the reporter enough evidence while keeping a strict bound.
                max_selected_chars=4500,
            )
        except Exception as e:
            story_log.log("source_pack_failed", error=str(e))
            return

        # If we can't extract at least 2 usable sources OR we don't have enough distinct URLs to
        # satisfy downstream "read more" requirements (3+), try a single deterministic Brave News
        # lookup to find alternative non-paywalled sources. This avoids expensive LLM browsing/tool calls.
        if _usable_sources_count(pack) < 2 or len(urls) < 3:
            try:
                # Basic keyword query from the title (Cantonese titles often contain English proper nouns/tickers).
                title = str(story.get("title") or "").strip()
                primary_url = str(story.get("primary_url") or "").strip()
                anchor = str(story.get("concrete_anchor") or "").strip()

                focus_terms_local: set[str] = set()
                focus_anchors: set[str] = set()
                if focus_text:
                    try:
                        focus_terms_local = choose_key_tokens(tokenize_text(focus_text, None), drop_high_df=set())
                    except Exception:
                        focus_terms_local = set()
                    try:
                        focus_anchors = anchor_terms(focus_text, None)
                    except Exception:
                        focus_anchors = set()

                # Pull terms from multiple places so Chinese titles still get a usable query.
                hay = [title, anchor]
                try:
                    p = urllib.parse.urlsplit(primary_url)
                    if p.path:
                        hay.append(p.path.replace("-", " ").replace("_", " "))
                except Exception:
                    pass

                words = []
                for s in hay:
                    if not s:
                        continue
                    words.extend(re.findall(r"[A-Za-z][A-Za-z0-9'-]{2,}", s))
                # Keep deterministic: unique, sorted, longest-first.
                uniq = sorted({w.lower() for w in words if w}, key=lambda w: (-len(w), w))

                # Prefer anchors (tickers / CJK entities) so we don't drop the actual subject
                # when the title contains lots of generic finance words.
                q_terms: list[str] = []
                tickers = [t for t in sorted(focus_anchors) if isinstance(t, str) and t.isupper() and 2 <= len(t) <= 6]
                cjk_terms = [t for t in sorted(focus_anchors) if any("\u4e00" <= ch <= "\u9fff" for ch in str(t))]
                entity_terms = [t for t in sorted(focus_anchors) if isinstance(t, str) and t.isascii() and t.islower() and len(t) >= 4]
                # Prefer English entities/tickers when available; CJK anchor terms can poison
                # search queries for English-language coverage (and yield irrelevant results).
                for t in (tickers[:2] + entity_terms[:3]):
                    if t and t not in q_terms:
                        q_terms.append(t)
                for t in uniq:
                    if t and t not in q_terms:
                        q_terms.append(t)
                    if len(q_terms) >= 8:
                        break
                if not q_terms and cjk_terms:
                    q_terms.extend([t for t in cjk_terms[:2] if t])

                # First, try to backfill URLs from the local pool (0 API calls).
                existing_norm: set[str] = set()
                used_domains: set[str] = set()
                for u in urls:
                    try:
                        existing_norm.add(normalize_url(u))
                    except Exception:
                        existing_norm.add(u)
                    try:
                        dom = urllib.parse.urlsplit(u).hostname or ""
                        if dom:
                            used_domains.add(dom.lower())
                    except Exception:
                        pass

                pool_added = self._backfill_urls_from_pool(
                    existing_norm_urls=existing_norm,
                    focus_terms_local=focus_terms_local,
                    focus_anchors=focus_anchors,
                    used_domains=used_domains,
                    max_add=5,
                    hours=48,
                )
                if pool_added:
                    story_log.log("source_pack_repair_add_urls_pool", added=pool_added)
                    if isinstance(story.get("supporting_urls"), list):
                        merged = [str(x).strip() for x in (story.get("supporting_urls") or []) if isinstance(x, str) and str(x).strip()]
                    else:
                        merged = []
                    for u in pool_added:
                        if u not in merged:
                            merged.append(u)
                    story["supporting_urls"] = merged
                    job["story"] = story

                    urls.extend([u for u in pool_added if u not in urls])
                    pack = build_source_pack(
                        story_id=story_id,
                        urls=urls,
                        openclaw_home=self._openclaw_home,
                        cache_hours=48,
                        focus_text=focus_text,
                        max_selected_chars=4500,
                    )

                # If still insufficient, do a single Brave lookup as deterministic repair.
                if q_terms and (_usable_sources_count(pack) < 2 or len(urls) < 3):
                    q = " ".join(q_terms)
                    api_keys = load_brave_api_keys(openclaw_home=self._openclaw_home)
                    key = select_brave_api_key(openclaw_home=self._openclaw_home, keys=api_keys)
                    fetched, _ = fetch_brave_news(
                        api_key=key.key,
                        q=q,
                        count=20,
                        offset=1,
                        freshness="day",
                        cache_dir=self._openclaw_home / "data" / "newsroom" / "brave_news_cache",
                        ttl_seconds=900,
                        last_request_ts=0.0,
                    )
                    if isinstance(getattr(fetched, "rate_limit", None), dict):
                        record_brave_rate_limit(openclaw_home=self._openclaw_home, key=key, rate_limit=fetched.rate_limit)

                    existing = set(urls)
                    added: list[str] = []
                    scored: list[tuple[int, str, str]] = []
                    for r in fetched.results:
                        u = r.get("url")
                        dom = (r.get("domain") or "") if isinstance(r.get("domain"), str) else ""
                        dom = dom.lower().strip()
                        if not isinstance(u, str) or not u.strip():
                            continue
                        u = u.strip()
                        if u in existing:
                            continue
                        # Skip common paywalls/bot-protected sites.
                        if dom.endswith(("reuters.com", "bloomberg.com", "wsj.com", "ft.com")):
                            continue
                        if dom in {"x.com", "twitter.com", "facebook.com", "youtube.com", "youtu.be", "tiktok.com", "instagram.com"}:
                            continue

                        r_title = r.get("title") if isinstance(r.get("title"), str) else None
                        r_desc = r.get("description") if isinstance(r.get("description"), str) else None

                        hit_terms = 0
                        if focus_terms_local:
                            try:
                                hit_terms = len(tokenize_text(r_title, r_desc).intersection(focus_terms_local))
                            except Exception:
                                hit_terms = 0

                        hit_anchors = 0
                        if focus_anchors:
                            try:
                                hit_anchors = len(anchor_terms(r_title, r_desc).intersection(focus_anchors))
                            except Exception:
                                hit_anchors = 0

                        # Require a stronger relevance signal so we don't add random unrelated links.
                        if focus_anchors:
                            if hit_anchors < 1 or hit_terms < 2:
                                continue
                        else:
                            if hit_terms < 3:
                                continue

                        score = int(hit_terms + (2 * hit_anchors))
                        scored.append((score, dom, u))

                    for _score, _dom, u in sorted(scored, key=lambda t: (-t[0], t[1], t[2])):
                        added.append(u)
                        existing.add(u)
                        if len(added) >= 5:
                            break

                    if added:
                        story_log.log("source_pack_repair_add_urls", query=q, added=added)
                        # Update supporting_urls so the worker can use them for read-more without extra searching.
                        if isinstance(story.get("supporting_urls"), list):
                            merged = [str(x).strip() for x in (story.get("supporting_urls") or []) if isinstance(x, str) and str(x).strip()]
                        else:
                            merged = []
                        for u in added:
                            if u not in merged:
                                merged.append(u)
                        story["supporting_urls"] = merged
                        job["story"] = story

                        urls.extend(added)
                        pack = build_source_pack(
                            story_id=story_id,
                            urls=urls,
                            openclaw_home=self._openclaw_home,
                            cache_hours=48,
                            focus_text=focus_text,
                            max_selected_chars=4500,
                        )
                    else:
                        story_log.log("source_pack_repair_no_urls", query=q, fetched_results=len(fetched.results), scored=len(scored))
            except Exception as e:
                story_log.log("source_pack_repair_failed", error=str(e))

        state["source_pack"] = pack
        job["state"] = state
        self._update_job_file(job_path, job, story_log, event="state_update", field="state.source_pack")

    def _ensure_assets_pack(self, *, job_path: Path, job: dict[str, Any], run_log: JsonlLogger, story_log: JsonlLogger) -> None:
        """Build deterministic per-story assets (market card data, charts, OG image hints).

        Assets are optional and must never block publishing.
        Stored under state.assets and injected into worker INPUT_JSON as `assets`.
        """
        if self._dry_run:
            return
        state = job.get("state", {}) or {}
        if isinstance(state.get("assets"), dict):
            return

        story = job.get("story", {}) or {}
        category = str(story.get("category") or "").strip()
        title = str(story.get("title") or "").strip()
        concrete_anchor = str(story.get("concrete_anchor") or "").strip()
        run_time_uk = str((job.get("run", {}) or {}).get("run_time_uk") or "").strip() or None

        primary_url = str(story.get("primary_url") or "").strip()
        supporting_urls = story.get("supporting_urls") if isinstance(story.get("supporting_urls"), list) else []
        page_urls: list[str] = []
        if primary_url:
            page_urls.append(primary_url)
        for u in supporting_urls[:6]:
            if isinstance(u, str) and u.strip():
                page_urls.append(u.strip())

        source_pack = state.get("source_pack") if isinstance(state.get("source_pack"), dict) else None

        assets: dict[str, Any] = {
            "ok": True,
            "generated_at": utc_iso(),
            "category": category,
            "page_urls": page_urls[:8],
            "market": None,
            "attachments": {"chart_paths": [], "og_image_urls": [], "og_image_paths": []},
            "errors": [],
        }

        # 1) OG image hints (best-effort): reuse any URLs extracted during source_pack fetch.
        try:
            ogs: list[str] = []
            if isinstance(source_pack, dict):
                sources = source_pack.get("sources")
                if isinstance(sources, list):
                    for s in sources[:10]:
                        if not isinstance(s, dict):
                            continue
                        og = s.get("og_image_url")
                        if isinstance(og, str) and og.strip().startswith("http"):
                            ogs.append(og.strip())
            # Deterministic de-dupe preserve order.
            assets["attachments"]["og_image_urls"] = list(dict.fromkeys(ogs))[:6]
        except Exception as e:
            assets["errors"].append({"stage": "og_hint", "error": str(e)[:200]})

        # 2) Market snapshot + chart (finance categories only).
        if category in {"US Stocks", "Crypto", "Precious Metals"}:
            try:
                market = build_market_assets(
                    category=category,
                    story_title=title,
                    concrete_anchor=concrete_anchor,
                    source_pack=source_pack,
                    run_time_uk=run_time_uk,
                )
                assets["market"] = market

                series = market.get("series") if isinstance(market, dict) else None
                if isinstance(series, dict) and isinstance(series.get("points"), list) and isinstance(series.get("symbol_display"), str):
                    run_id = str((job.get("run", {}) or {}).get("run_id") or job_path.parent.name or "run").strip()
                    story_id = str(story.get("story_id") or job_path.stem).strip() or "story"
                    sym = str(series.get("symbol_display") or "").strip() or "asset"
                    digest = hashlib.sha1(f"{run_id}:{story_id}:{sym}".encode("utf-8")).hexdigest()[:20]
                    charts_dir = self._openclaw_home / "data" / "newsroom" / "assets" / "charts"
                    chart_path = charts_dir / f"chart_{digest}.png"

                    subtitle = None
                    try:
                        items = market.get("items") if isinstance(market, dict) else None
                        if isinstance(items, list) and items and isinstance(items[0], dict):
                            price = items[0].get("price")
                            ccy = items[0].get("currency")
                            chg = items[0].get("change_1d_pct")
                            if isinstance(price, (int, float)) and isinstance(ccy, str):
                                if isinstance(chg, (int, float)):
                                    subtitle = f"{price:.2f} {ccy} ({chg:+.2f}%)"
                                else:
                                    subtitle = f"{price:.2f} {ccy}"
                    except Exception:
                        subtitle = None

                    try:
                        render_line_chart_png(
                            points=list(series.get("points") or []),
                            symbol=sym,
                            out_path=chart_path,
                            title=f"{sym} 60D",
                            subtitle=subtitle,
                        )
                        assets["attachments"]["chart_paths"] = [str(chart_path)]
                    except Exception as e:
                        assets["errors"].append({"stage": "chart", "symbol": sym, "error": str(e)[:200]})
            except Exception as e:
                assets["errors"].append({"stage": "market", "error": str(e)[:200]})

        state["assets"] = assets
        job["state"] = state
        self._update_job_file(job_path, job, story_log, event="state_update", field="state.assets")
        run_log.log("assets_built", story_id=str(story.get("story_id") or job_path.stem), category=category)

    def _ensure_og_image_paths(
        self,
        *,
        job_path: Path,
        job: dict[str, Any],
        run_log: JsonlLogger,
        story_log: JsonlLogger,
        max_downloads: int = 1,
    ) -> list[str]:
        """Download at most N OG images for this story and persist their local paths.

        This is best-effort and must never fail the job.
        """
        if self._dry_run:
            return []

        state = job.get("state", {}) or {}
        assets = state.get("assets")
        if not isinstance(assets, dict):
            return []
        attachments = assets.get("attachments")
        if not isinstance(attachments, dict):
            attachments = {"chart_paths": [], "og_image_urls": [], "og_image_paths": []}
            assets["attachments"] = attachments

        # Keep only paths that still exist.
        existing: list[str] = []
        for p in attachments.get("og_image_paths") if isinstance(attachments.get("og_image_paths"), list) else []:
            if isinstance(p, str) and p.strip() and Path(p.strip()).exists():
                existing.append(p.strip())
        existing = list(dict.fromkeys(existing))
        if existing:
            if existing != attachments.get("og_image_paths"):
                attachments["og_image_paths"] = existing
                assets["attachments"] = attachments
                state["assets"] = assets
                job["state"] = state
                self._update_job_file(
                    job_path,
                    job,
                    story_log,
                    event="state_update",
                    field="state.assets.attachments.og_image_paths",
                )
            return existing

        story = job.get("story", {}) or {}
        primary_url = str(story.get("primary_url") or "").strip()
        supporting_urls = story.get("supporting_urls") if isinstance(story.get("supporting_urls"), list) else []

        # Prefer already-extracted OG URLs from source_pack; otherwise, fetch from the page(s).
        og_urls: list[str] = []
        for u in attachments.get("og_image_urls") if isinstance(attachments.get("og_image_urls"), list) else []:
            if isinstance(u, str) and u.strip().startswith("http"):
                og_urls.append(u.strip())
        og_urls = list(dict.fromkeys(og_urls))

        session = requests.Session()
        session.headers.update({"User-Agent": "Mozilla/5.0 (OpenClaw Newsroom)"})

        if not og_urls:
            # Back-compat: source_pack cache may predate og extraction; fetch once.
            try:
                if primary_url:
                    u = fetch_og_image_url(session, primary_url)
                    if isinstance(u, str) and u.strip().startswith("http"):
                        og_urls.append(u.strip())
                if not og_urls:
                    for page in supporting_urls[:2]:
                        if not isinstance(page, str) or not page.strip().startswith("http"):
                            continue
                        u = fetch_og_image_url(session, page.strip())
                        if isinstance(u, str) and u.strip().startswith("http"):
                            og_urls.append(u.strip())
                            break
            except Exception as e:
                story_log.log("og_fetch_failed", error=str(e))

        og_urls = list(dict.fromkeys(og_urls))[:6]
        if og_urls and not attachments.get("og_image_urls"):
            attachments["og_image_urls"] = og_urls

        dest_dir = self._openclaw_home / "data" / "newsroom" / "assets" / "og"
        downloaded: list[str] = []
        errors: list[dict[str, Any]] = []

        for img_url in og_urls:
            if len(downloaded) >= int(max_downloads):
                break
            try:
                res = download_image(session, img_url, dest_dir=dest_dir)
                if res.ok and isinstance(res.path, str) and res.path:
                    downloaded.append(res.path)
                else:
                    errors.append({"stage": "download", "url": img_url, "error": str(res.error or "download_failed")[:200]})
            except Exception as e:
                errors.append({"stage": "download", "url": img_url, "error": str(e)[:200]})

        downloaded = list(dict.fromkeys(downloaded))
        if downloaded or errors:
            attachments["og_image_paths"] = downloaded
            assets["attachments"] = attachments
            if isinstance(assets.get("errors"), list) and errors:
                assets["errors"].extend(errors)
            state["assets"] = assets
            job["state"] = state
            self._update_job_file(
                job_path,
                job,
                story_log,
                event="state_update",
                field="state.assets.attachments.og_image_paths",
            )
            run_log.log(
                "og_images_downloaded",
                story_id=str(story.get("story_id") or job_path.stem),
                downloaded=len(downloaded),
                attempted=len(og_urls),
            )
        return downloaded

    def _nano_banana_api_key(self) -> str | None:
        """Return GEMINI_API_KEY for nano-banana-pro from openclaw.json (best-effort)."""
        cfg_path = self._openclaw_home / "openclaw.json"
        if not cfg_path.exists():
            return None
        try:
            raw = json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        entries = ((raw.get("skills", {}) or {}).get("entries", {}) or {})
        nb = entries.get("nano-banana-pro") if isinstance(entries, dict) else None
        if not isinstance(nb, dict):
            return None
        env = nb.get("env")
        if isinstance(env, dict):
            v = env.get("GEMINI_API_KEY")
            if isinstance(v, str) and v.strip():
                return v.strip()
        v = nb.get("apiKey")
        if isinstance(v, str) and v.strip():
            return v.strip()
        return None

    def _ensure_generated_image_paths(
        self,
        *,
        kind: str,
        paths_field: str,
        prompt_field: str,
        out_dirname: str,
        final_prefix_fn: Callable[[str], str],
        fallback_prefix_fn: Callable[[str], str],
        job_path: Path,
        job: dict[str, Any],
        result_json: dict[str, Any],
        run_log: JsonlLogger,
        story_log: JsonlLogger,
        max_generate_seconds: int = 120,
    ) -> list[str]:
        """Ensure we have 0..1 generated image paths for this story (card/infographic).

        Sources of truth (in order):
        1) state.assets.attachments.<paths_field> (persisted)
        2) worker draft.<paths_field> (pre-generated by worker)
        3) worker draft.<prompt_field> (runner generates via nano-banana-pro)
        """
        if self._dry_run:
            return []

        state = job.get("state", {}) or {}
        assets = state.get("assets")
        if not isinstance(assets, dict):
            return []
        attachments = assets.get("attachments")
        if not isinstance(attachments, dict):
            attachments = {
                "chart_paths": [],
                "og_image_urls": [],
                "og_image_paths": [],
                "infographic_paths": [],
                "card_paths": [],
            }
            assets["attachments"] = attachments
        else:
            # Ensure all expected keys exist for deterministic downstream logic.
            attachments.setdefault("chart_paths", [])
            attachments.setdefault("og_image_urls", [])
            attachments.setdefault("og_image_paths", [])
            attachments.setdefault("infographic_paths", [])
            attachments.setdefault("card_paths", [])

        def _existing_paths(obj: Any) -> list[str]:
            out: list[str] = []
            if not isinstance(obj, list):
                return out
            for p in obj:
                if isinstance(p, str) and p.strip():
                    pp = p.strip()
                    if Path(pp).exists():
                        out.append(pp)
            return list(dict.fromkeys(out))[:1]

        # 1) Persisted paths
        existing = _existing_paths(attachments.get(paths_field))
        if existing:
            return existing

        draft = result_json.get("draft") if isinstance(result_json, dict) else None
        if not isinstance(draft, dict):
            return []

        # 2) Worker-provided paths
        worker_paths = _existing_paths(draft.get(paths_field))
        if worker_paths:
            attachments[paths_field] = worker_paths
            assets["attachments"] = attachments
            state["assets"] = assets
            job["state"] = state
            self._update_job_file(
                job_path,
                job,
                story_log,
                event="state_update",
                field=f"state.assets.attachments.{paths_field}",
            )
            return worker_paths

        # 3) Runner generation from prompt
        raw_prompt = draft.get(prompt_field)
        if not isinstance(raw_prompt, str) or not raw_prompt.strip():
            return []

        aspect = _pick_generated_image_aspect_from_prompt(raw_prompt)
        prompt = _sanitize_generated_image_prompt(raw_prompt)
        if not prompt:
            return []
        if len(prompt) > 1200:
            prompt = prompt[:1200].rstrip() + "..."

        final_prompt = final_prefix_fn(aspect) + prompt
        if len(final_prompt) > 1600:
            final_prompt = final_prompt[:1600].rstrip() + "..."

        story = job.get("story", {}) or {}
        run = job.get("run", {}) or {}
        run_id = str(run.get("run_id") or job_path.parent.name or "run").strip() or "run"
        story_id = str(story.get("story_id") or job_path.stem).strip() or "story"

        digest = hashlib.sha1(f"{kind}:{aspect}:{run_id}:{story_id}:{prompt}".encode("utf-8")).hexdigest()[:20]
        out_dir = self._openclaw_home / "data" / "newsroom" / "assets" / out_dirname
        out_path = out_dir / f"{kind}_{digest}.png"
        if out_path.exists():
            try:
                if out_path.stat().st_size > 4_000:
                    _pad_png_to_aspect(out_path, aspect)
                    attachments[paths_field] = [str(out_path)]
                    assets["attachments"] = attachments
                    state["assets"] = assets
                    job["state"] = state
                    self._update_job_file(
                        job_path,
                        job,
                        story_log,
                        event="state_update",
                        field=f"state.assets.attachments.{paths_field}",
                    )
                    return [str(out_path)]
            except Exception:
                pass

        _nano_env = os.environ.get("NANO_BANANA_SCRIPT", "")
        script_path = Path(_nano_env) if _nano_env else None
        if not script_path or not script_path.exists():
            assets.setdefault("errors", []).append({"stage": kind, "error": "missing_nano_banana_script"})
            return []

        uv_bin = which("uv")
        if not uv_bin:
            assets.setdefault("errors", []).append({"stage": kind, "error": "uv_not_in_PATH"})
            return []

        api_key = self._nano_banana_api_key()
        if not api_key:
            assets.setdefault("errors", []).append({"stage": kind, "error": "missing_GEMINI_API_KEY"})
            return []

        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            str(uv_bin),
            "run",
            str(script_path),
            "--prompt",
            final_prompt,
            "--filename",
            str(out_path),
            "--resolution",
            "1K",
        ]
        env = os.environ.copy()
        env["GEMINI_API_KEY"] = api_key

        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=float(max_generate_seconds))
        except subprocess.TimeoutExpired:
            assets.setdefault("errors", []).append({"stage": kind, "error": "timeout"})
            story_log.log(f"{kind}_timeout", story_id=story_id, out_path=str(out_path))
            return []
        except Exception as e:
            assets.setdefault("errors", []).append({"stage": kind, "error": str(e)[:200]})
            story_log.log(f"{kind}_exception", story_id=story_id, error=str(e))
            return []

        ok = False
        try:
            ok = proc.returncode == 0 and out_path.exists() and out_path.stat().st_size > 4_000
        except Exception:
            ok = False

        if not ok:
            stderr = (proc.stderr or "").strip()
            stdout = (proc.stdout or "").strip()

            # One cheap retry: use a simplified fallback prompt (avoids deterministic policy blocks).
            story_title = str((job.get("story", {}) or {}).get("title") or "").strip()
            anchor = str((job.get("story", {}) or {}).get("concrete_anchor") or "").strip()
            fallback_bits = [b for b in (story_title, anchor) if b]
            fallback_facts = " | ".join(fallback_bits)
            if len(fallback_facts) > 600:
                fallback_facts = fallback_facts[:600].rstrip() + "..."

            fallback_prompt = fallback_prefix_fn(aspect) + fallback_facts
            if len(fallback_prompt) > 1600:
                fallback_prompt = fallback_prompt[:1600].rstrip() + "..."

            cmd2 = list(cmd)
            try:
                i = cmd2.index("--prompt")
            except ValueError:
                i = -1
            if i != -1 and i + 1 < len(cmd2):
                cmd2[i + 1] = fallback_prompt

            try:
                proc2 = subprocess.run(cmd2, capture_output=True, text=True, env=env, timeout=float(max_generate_seconds))
            except Exception:
                proc2 = None

            if proc2 is not None:
                try:
                    ok2 = proc2.returncode == 0 and out_path.exists() and out_path.stat().st_size > 4_000
                except Exception:
                    ok2 = False
                if ok2:
                    _pad_png_to_aspect(out_path, aspect)
                    paths = [str(out_path)]
                    attachments[paths_field] = paths
                    assets["attachments"] = attachments
                    state["assets"] = assets
                    job["state"] = state
                    self._update_job_file(
                        job_path,
                        job,
                        story_log,
                        event="state_update",
                        field=f"state.assets.attachments.{paths_field}",
                    )
                    run_log.log(f"{kind}_generated_retry", story_id=story_id, path=str(out_path))
                    return paths
                stderr = (proc2.stderr or "").strip() or stderr
                stdout = (proc2.stdout or "").strip() or stdout

            assets.setdefault("errors", []).append(
                {
                    "stage": kind,
                    "error": f"generate_failed:rc={proc.returncode}",
                    "stderr": (stderr[:400] if stderr else None),
                    "stdout": (stdout[:200] if stdout else None),
                }
            )
            story_log.log(
                f"{kind}_failed",
                story_id=story_id,
                returncode=int(proc.returncode),
                stderr_head=stderr[:200],
            )
            return []

        paths = [str(out_path)]
        _pad_png_to_aspect(out_path, aspect)
        attachments[paths_field] = paths
        assets["attachments"] = attachments
        state["assets"] = assets
        job["state"] = state
        self._update_job_file(
            job_path,
            job,
            story_log,
            event="state_update",
            field=f"state.assets.attachments.{paths_field}",
        )
        run_log.log(f"{kind}_generated", story_id=story_id, path=str(out_path))
        return paths

    def _ensure_card_paths(
        self,
        *,
        job_path: Path,
        job: dict[str, Any],
        result_json: dict[str, Any],
        run_log: JsonlLogger,
        story_log: JsonlLogger,
        max_generate_seconds: int = 120,
    ) -> list[str]:
        return self._ensure_generated_image_paths(
            kind="card",
            paths_field="card_paths",
            prompt_field="card_prompt",
            out_dirname="cards",
            final_prefix_fn=_card_final_prefix,
            fallback_prefix_fn=_card_fallback_prefix,
            job_path=job_path,
            job=job,
            result_json=result_json,
            run_log=run_log,
            story_log=story_log,
            max_generate_seconds=max_generate_seconds,
        )

    def _ensure_infographic_paths(
        self,
        *,
        job_path: Path,
        job: dict[str, Any],
        result_json: dict[str, Any],
        run_log: JsonlLogger,
        story_log: JsonlLogger,
        max_generate_seconds: int = 120,
    ) -> list[str]:
        return self._ensure_generated_image_paths(
            kind="infographic",
            paths_field="infographic_paths",
            prompt_field="infographic_prompt",
            out_dirname="infographics",
            final_prefix_fn=_infographic_final_prefix,
            fallback_prefix_fn=_infographic_fallback_prefix,
            job_path=job_path,
            job=job,
            result_json=result_json,
            run_log=run_log,
            story_log=story_log,
            max_generate_seconds=max_generate_seconds,
        )

    def _spawn_subagent(
        self,
        *,
        job: dict[str, Any],
        job_path: Path,
        prompt_def: PromptDef,
        run_log: JsonlLogger,
        story_log: JsonlLogger,
        timeout_seconds: int,
        label_suffix: str,
        worker_error: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        if self._dry_run:
            _ = self._render_worker_task(job, prompt_def, worker_error=worker_error)
            story_log.log("spawn_dry_run", prompt_id=prompt_def.prompt_id, timeout_seconds=timeout_seconds)
            return {"status": "accepted", "runId": "dry-run", "childSessionKey": None}

        # Pre-pack sources deterministically (no LLM tokens) so the worker can avoid
        # browser/web_search tool calls unless absolutely required.
        self._ensure_source_pack(job_path=job_path, job=job, story_log=story_log)
        # Build deterministic assets (market snapshot, chart paths, OG image hints)
        # and inject them into the worker input JSON.
        self._ensure_assets_pack(job_path=job_path, job=job, run_log=run_log, story_log=story_log)

        task = self._render_worker_task(job, prompt_def, worker_error=worker_error)
        spawn = job.get("spawn", {}) or {}
        agent_id = str(spawn.get("agent_id") or "main")

        # The Gateway enforces a hard label length cap; keep this deterministic and short.
        run_id = str((job.get("run", {}) or {}).get("run_id", "run"))
        story_id = str((job.get("story", {}) or {}).get("story_id", job_path.stem))
        digest = hashlib.sha1(f"{run_id}:{story_id}:{label_suffix}".encode("utf-8")).hexdigest()[:12]
        label = f"nr:{story_id}:{label_suffix}:{digest}"

        resp_obj: dict[str, Any] | None = None
        last_err: Exception | None = None
        for attempt in range(1, 4):
            try:
                resp_obj = self._gateway.invoke_result_json(
                    tool="sessions_spawn",
                    action="json",
                    args={
                        "task": task,
                        "label": label,
                        "agentId": agent_id,
                        "runTimeoutSeconds": int(timeout_seconds),
                        # Keep sub-agent sessions so the runner can reliably read the final RESULT JSON
                        # via sessions_history (delete can race and lose the transcript).
                        "cleanup": "keep",
                    },
                )
            except Exception as e:
                last_err = e
                story_log.log("spawn_retry_exception", prompt_id=prompt_def.prompt_id, agent_id=agent_id, attempt=attempt, error=str(e))
                time.sleep(min(2.0 * attempt, 5.0))
                continue

            # The gateway returns {"status":"accepted","childSessionKey":"..."} on success.
            child_key = resp_obj.get("childSessionKey") if isinstance(resp_obj, dict) else None
            if resp_obj.get("status") == "accepted" and isinstance(child_key, str) and child_key.strip():
                break

            story_log.log("spawn_retry_missing_key", prompt_id=prompt_def.prompt_id, agent_id=agent_id, attempt=attempt, response=resp_obj)
            time.sleep(min(2.0 * attempt, 5.0))

        # If spawn is forbidden/misconfigured we can still get ok=true but no childSessionKey.
        if not isinstance(resp_obj, dict) or resp_obj.get("status") != "accepted" or not isinstance(resp_obj.get("childSessionKey"), str) or not str(resp_obj.get("childSessionKey") or "").strip():
            story_log.log("spawn_failed", prompt_id=prompt_def.prompt_id, agent_id=agent_id, response=resp_obj, error=str(last_err) if last_err else None)
            raise RuntimeError(f"sessions_spawn failed: {resp_obj or last_err}")

        # Best-effort: resolve sessionId now so job state can be audited/charged later even if
        # sessions.json no longer indexes completed sub-agent sessions.
        child_key = resp_obj.get("childSessionKey")
        if isinstance(child_key, str) and child_key.strip():
            try:
                lst = self._gateway.invoke_result_json(
                    tool="sessions_list",
                    action="json",
                    args={"limit": 25, "sessionKey": child_key.strip()},
                )
                sessions = lst.get("sessions", [])
                if isinstance(sessions, list) and sessions:
                    # Be defensive: some list calls may ignore filters. Only accept exact key matches.
                    for s in sessions:
                        if isinstance(s, dict) and s.get("key") == child_key.strip():
                            sid = s.get("sessionId")
                            if isinstance(sid, str) and sid.strip():
                                resp_obj["sessionId"] = sid.strip()
                            break
            except Exception as e:
                story_log.log("session_id_lookup_failed", error=str(e), childSessionKey=child_key)

        story_log.log("spawn_ok", prompt_id=prompt_def.prompt_id, childSessionKey=resp_obj.get("childSessionKey"))
        run_log.log("spawn_ok", story_id=(job.get("story", {}) or {}).get("story_id"), childSessionKey=resp_obj.get("childSessionKey"))
        return resp_obj

    def _maybe_update_discord_title_from_result(
        self,
        *,
        job: dict[str, Any],
        result_json: dict[str, Any],
        run_log: JsonlLogger,
        story_log: JsonlLogger,
    ) -> None:
        """Backstop: ensure title channel message ends up as Cantonese when worker translated it.

        Planners are instructed to write Cantonese titles, but they sometimes forget. Workers can
        translate in their RESULT JSON; if so, we edit the title message after the fact.
        """
        if self._dry_run:
            return

        dest = job.get("destination", {}) or {}
        if dest.get("platform") != "discord":
            return

        title_channel_id = dest.get("title_channel_id")
        state = job.get("state", {}) or {}
        discord_state = state.get("discord", {}) or {}
        title_message_id = discord_state.get("title_message_id")

        if not (_is_discord_snowflake(title_channel_id) and _is_discord_snowflake(title_message_id)):
            return

        story = job.get("story", {}) or {}
        old_title = str(story.get("title") or "").strip()
        new_title = result_json.get("title")
        if not isinstance(new_title, str):
            return
        new_title = new_title.strip()
        if not new_title or not old_title or new_title == old_title:
            return

        # Only rewrite when we're clearly fixing an English title into Cantonese.
        if not _looks_like_english_title(old_title):
            return
        if _looks_like_english_title(new_title) or _count_cjk(new_title) < 4:
            return

        # Update job state in memory so the final write captures it.
        story["title"] = new_title
        job["story"] = story
        story_log.log("title_rewritten", from_title=old_title, to_title=new_title)

        # Edit the title-channel message.
        try:
            self._tool_invoke_retry(
                tool="message",
                action="edit",
                args={
                    "channel": "discord",
                    "target": f"channel:{title_channel_id}",
                    "messageId": str(title_message_id),
                    "message": new_title,
                },
                run_log=run_log,
                story_log=story_log,
            )
            story_log.log("title_edit_ok", title_message_id=str(title_message_id))
        except Exception as e:
            # Do not fail the whole story for a title edit. The thread content is the main value.
            story_log.log("title_edit_failed", error=str(e), title_message_id=str(title_message_id))

    def _validate_result(self, *, job: dict[str, Any], validator_def: ValidatorDef, result_json: dict[str, Any]) -> ValidationOutcome:
        fn = self._get_validator(validator_def)
        try:
            outcome = fn(result_json, job)
            ok = bool(getattr(outcome, "ok", False))
            errors = list(getattr(outcome, "errors", []))
            return ValidationOutcome(ok=ok, errors=errors)
        except Exception as e:
            return ValidationOutcome(ok=False, errors=[f"validator_exception:{e}"])

    def _try_repair_validation_failure(
        self,
        *,
        job: dict[str, Any],
        validator_def: ValidatorDef,
        result_json: dict[str, Any],
        errors: list[str],
        story_log: JsonlLogger,
    ) -> tuple[dict[str, Any], ValidationOutcome, list[str]]:
        """Deterministic repair for common, non-substantive validation failures.

        Goal: avoid expensive rescue runs when the worker draft is good but missed
        small structural requirements (e.g. read-more URL count).
        """
        repaired, repairs = repair_result_json(result_json=result_json, job=job, errors=errors)
        if not repairs:
            return result_json, ValidationOutcome(ok=False, errors=errors), []

        story_log.log("result_repaired", repairs=repairs)
        outcome = self._validate_result(job=job, validator_def=validator_def, result_json=repaired)
        return repaired, outcome, repairs

    def _start_job(self, *, job_path: Path, run_defaults: dict[str, Any]) -> JobRuntime | None:
        try:
            job = load_json_file(job_path)
            job = self._coerce_story_job_format(job_path, job, run_defaults)
            self._validate_story_job(job)
        except (json.JSONDecodeError, JobSchemaError) as e:
            # Corrupt or invalid job file — quarantine it so the runner doesn't
            # retry it on every cron tick.
            print(f"⚠️ Jailing unreadable job {job_path}: {e}", file=sys.stderr)
            jail_job_file(job_path, reason=f"{type(e).__name__}: {e}")
            return None

        run_log, story_log = self._loggers_for_job(job, job_path)
        run_log.log("job_seen", job_path=str(job_path))
        story_log.log("job_seen", job_path=str(job_path))

        # Skip completed jobs early.
        if (job.get("state", {}) or {}).get("status") in ("SUCCESS", "FAILURE", "RESCUED", "SKIPPED"):
            story_log.log("job_skip_complete", status=(job.get("state", {}) or {}).get("status"))
            return None

        lock_path = job_path.with_suffix(job_path.suffix + ".lock")
        lock = FileLock(lock_path, owner=_runner_id(), ttl_seconds=self._lock_ttl_seconds)
        try:
            lock.acquire()
        except LockHeldError as e:
            story_log.log("job_lock_held", error=str(e))
            return None

        # From here on, we own the job file (single-writer).
        dedupe_key = ""
        dedupe_lock: FileLock | None = None
        try:
            prompt_def, validator_def = self._resolve_prompt_and_validator(job)

            # Ensure story.dedupe_key is stable and event-scoped (prefers event:...).
            # This must happen before acquiring the dedupe lock so the lock/marker keys match.
            before = str((job.get("story", {}) or {}).get("dedupe_key") or "")
            self._ensure_story_dedupe_key(job)
            after = str((job.get("story", {}) or {}).get("dedupe_key") or "")
            if after and after != before:
                story_log.log("dedupe_key_coerced", before=before, after=after)

            # Policy gate for hourly runs: only allow BREAKING/DEVELOPING jobs.
            run = job.get("run", {}) or {}
            trigger = str(run.get("trigger") or "").strip()
            if trigger == "cron_hourly":
                story = job.get("story", {}) or {}
                flags = story.get("flags") or []
                if not isinstance(flags, list):
                    flags = []
                has_bd = any(isinstance(f, str) and f.strip() in {"breaking", "developing"} for f in flags)
                discord_state = (job.get("state", {}) or {}).get("discord", {}) or {}
                # Only enforce before we create Discord containers (avoid leaving empty threads).
                if not discord_state.get("thread_id") and not has_bd:
                    reason = "hourly_policy_violation: flags must include breaking/developing"
                    self._mark_job_skipped_policy(job_path=job_path, job=job, reason=reason)
                    lock.release()
                    return None

            # Cross-run de-duplication (local marker ledger, 24h TTL).
            # This is a backstop: planners should still dedupe via Discord history,
            # but this prevents accidental double-posts when planners misbehave.
            if not self._dry_run:
                dedupe_key = self._dedupe_key_for_job(job)
                if dedupe_key:
                    marker_path = self._dedupe_marker_path(dedupe_key)
                    lock_path = self._dedupe_lock_path(dedupe_key)
                    dedupe_lock = FileLock(lock_path, owner=_runner_id(), ttl_seconds=self._DEDUPE_TTL_SECONDS)
                    try:
                        dedupe_lock.acquire()
                    except LockHeldError as e:
                        # Another runner is processing the same story; skip this job for now.
                        story_log.log("dedupe_lock_held", dedupe_key=dedupe_key, error=str(e))
                        lock.release()
                        return None

                    # If this job already has a Discord container, allow it to proceed even if
                    # the marker exists. This prevents a resumed job from skipping itself.
                    discord_state = (job.get("state", {}) or {}).get("discord", {}) or {}
                    existing_marker = None
                    # Prefer the exact marker path (new scheme). If missing, fall back to the
                    # primary_url index (back-compat with older url|hash markers).
                    if self._dedupe_marker_is_fresh(marker_path):
                        existing_marker = marker_path
                    else:
                        primary_url = str((job.get("story", {}) or {}).get("primary_url") or "").strip()
                        if primary_url:
                            existing_marker = self._dedupe_marker_for_primary_url(primary_url)

                    if not discord_state.get("thread_id") and existing_marker is not None:
                        self._mark_job_skipped_duplicate(job_path=job_path, job=job, reason=f"dedupe_marker_exists:{existing_marker.name}")
                        dedupe_lock.release()
                        lock.release()
                        return None

            # Semantic Discord-history de-duplication (meaningful; not just key match).
            # Must happen BEFORE we create Discord containers (avoid leaving empty threads).
            destination_platform = (job.get("destination", {}) or {}).get("platform")
            if destination_platform == "discord" and not self._dry_run:
                discord_state = (job.get("state", {}) or {}).get("discord", {}) or {}
                if not discord_state.get("title_message_id") and not discord_state.get("thread_id"):
                    # Dedupe against Discord titles works best when the job title is already in
                    # Cantonese Traditional Chinese. If the planner supplied an English headline,
                    # translate it *before* the semantic dedupe check so we can catch near-duplicates
                    # across runs (and avoid double-posting the same event with different languages).
                    try:
                        story = job.get("story", {}) or {}
                        old_title = str(story.get("title") or "").strip()
                        if old_title and _looks_like_english_title(old_title):
                            maybe_new = self._translate_title_cantonese(
                                job_path=job_path,
                                job=job,
                                old_title=old_title,
                                run_log=run_log,
                                story_log=story_log,
                            )
                            if (
                                isinstance(maybe_new, str)
                                and maybe_new.strip()
                                and maybe_new.strip() != old_title
                                and (not _looks_like_english_title(maybe_new.strip()))
                                and _count_cjk(maybe_new.strip()) >= 4
                            ):
                                story["title"] = maybe_new.strip()
                                job["story"] = story
                                story_log.log("title_translate_applied_pre_dedupe", from_title=old_title, to_title=maybe_new.strip())
                                self._update_job_file(job_path, job, story_log, event="title_translate_applied_pre_dedupe", title=maybe_new.strip())
                    except Exception as e:
                        story_log.log("title_translate_pre_dedupe_failed", error=str(e))

                    try:
                        dup_reason = self._semantic_dedupe_against_discord_titles(job=job, run_log=run_log, story_log=story_log)
                    except Exception as e:
                        dup_reason = None
                        story_log.log("semantic_dedupe_check_failed", error=str(e))
                    if dup_reason:
                        self._mark_job_skipped_duplicate(job_path=job_path, job=job, reason=dup_reason)
                        if dedupe_lock:
                            dedupe_lock.release()
                        lock.release()
                        return None

            # Record lock ownership in job state (optional but useful).
            state = job.get("state", {}) or {}
            state["locked_by"] = _runner_id()
            state["locked_at"] = utc_iso()
            job["state"] = state
            self._update_job_file(job_path, job, story_log, event="state_update", field="locked_by")

            # Before creating any Discord containers (title post/thread), ensure we have enough
            # extracted source text. This avoids leaving empty threads when sources are paywalled
            # or extraction fails.
            destination_platform = (job.get("destination", {}) or {}).get("platform")
            if destination_platform == "discord" and not self._dry_run:
                discord_state = (job.get("state", {}) or {}).get("discord", {}) or {}
                if not discord_state.get("title_message_id") and not discord_state.get("thread_id"):
                    self._ensure_source_pack(job_path=job_path, job=job, story_log=story_log)
                    pack_obj = (job.get("state", {}) or {}).get("source_pack")
                    if _usable_sources_count(pack_obj) < 2:
                        reason = "insufficient_sources_pack_text"
                        self._mark_job_skipped_missing_sources(job_path=job_path, job=job, reason=reason)
                        if dedupe_lock:
                            dedupe_lock.release()
                        lock.release()
                        return None

            # Prepare destination.
            spawn = job.get("spawn", {}) or {}
            publisher_mode = str(spawn.get("publisher_mode") or "agent_posts")

            if destination_platform == "discord":
                # Only create Discord containers (title post + thread) *before* the worker runs
                # when the worker is expected to publish content itself (agent_posts). In
                # script_posts mode the worker returns a draft and the runner publishes later;
                # delaying container creation avoids leaving empty threads if the worker/rescue
                # fails or the runner terminates uncleanly.
                if publisher_mode != "script_posts":
                    self._prepare_discord(job_path, job, run_log, story_log)
            elif destination_platform == "webhook":
                self._prepare_webhook(job_path, job, run_log, story_log)
            else:
                raise JobSchemaError(f"Unsupported destination.platform: {destination_platform}")

            monitor = job.get("monitor", {}) or {}
            poll_seconds = int(monitor.get("poll_seconds", 5))

            # If we already have an active rescue session, monitor it first. This matters
            # when resuming after a crash: both worker+rescue session keys may exist, but
            # only the rescue is still running.
            rescue_state = (job.get("state", {}) or {}).get("rescue", {}) or {}
            if rescue_state.get("child_session_key") and not rescue_state.get("ended_at"):
                recover = job.get("recover", {}) or {}
                rescue_timeout_seconds = int(recover.get("rescue_timeout_seconds", 600))
                started_ts = _utc_iso_to_ts(rescue_state.get("started_at"))
                deadline = (
                    (started_ts + max(1, rescue_timeout_seconds))
                    if started_ts is not None
                    else (time.time() + max(1, rescue_timeout_seconds))
                )
                story_log.log("monitor_existing_rescue", child_session_key=rescue_state.get("child_session_key"))
                return JobRuntime(
                    job_path=job_path,
                    job=job,
                    lock=lock,
                    dedupe_lock=dedupe_lock,
                    phase="rescue",
                    deadline_ts=deadline,
                    poll_seconds=poll_seconds,
                    next_poll_ts=time.time(),
                )

            # If we already have an active worker session, just monitor it.
            worker_state = (job.get("state", {}) or {}).get("worker", {}) or {}
            if worker_state.get("child_session_key"):
                timeout_seconds = int(monitor.get("timeout_seconds", run_defaults.get("default_timeout_seconds", 900)))
                started_ts = _utc_iso_to_ts(worker_state.get("started_at"))
                deadline = (
                    (started_ts + max(1, timeout_seconds))
                    if started_ts is not None
                    else (time.time() + max(1, timeout_seconds))
                )
                story_log.log("monitor_existing_worker", child_session_key=worker_state.get("child_session_key"))
                return JobRuntime(
                    job_path=job_path,
                    job=job,
                    lock=lock,
                    dedupe_lock=dedupe_lock,
                    phase="worker",
                    deadline_ts=deadline,
                    poll_seconds=poll_seconds,
                    next_poll_ts=time.time(),
                )

            # Spawn worker.
            timeout_seconds = int(monitor.get("timeout_seconds", run_defaults.get("default_timeout_seconds", 900)))
            spawn_resp = self._spawn_subagent(
                job=job,
                job_path=job_path,
                prompt_def=prompt_def,
                run_log=run_log,
                story_log=story_log,
                timeout_seconds=timeout_seconds + 60,  # give the runtime a little slack
                label_suffix="worker",
            )

            worker_state = (job.get("state", {}) or {}).get("worker", {}) or {}
            worker_state["attempt"] = int(worker_state.get("attempt", 0)) + 1
            worker_state["child_session_key"] = spawn_resp.get("childSessionKey")
            worker_state["run_id"] = spawn_resp.get("runId")
            worker_state["session_id"] = spawn_resp.get("sessionId")
            worker_state["started_at"] = utc_iso()
            (job.get("state", {}) or {})["worker"] = worker_state
            state = job.get("state", {}) or {}
            state["worker"] = worker_state
            state["status"] = "DISPATCHED"
            job["state"] = state
            self._update_job_file(job_path, job, story_log, event="state_update", field="worker.child_session_key")

            return JobRuntime(
                job_path=job_path,
                job=job,
                lock=lock,
                dedupe_lock=dedupe_lock,
                phase="worker",
                deadline_ts=time.time() + timeout_seconds,
                poll_seconds=poll_seconds,
                next_poll_ts=time.time(),
            )
        except Exception as e:
            # If we fail before returning a runtime, mark job failure (best-effort) and release lock.
            try:
                state = job.get("state", {}) or {}
                state["status"] = "FAILURE"
                job["state"] = state
                result = job.get("result", {}) or {}
                result["final_status"] = "FAILURE"
                result.setdefault("errors", []).append({"type": "runner_error", "message": str(e), "at": utc_iso()})
                job["result"] = result
                self._update_job_file(job_path, job, story_log, event="job_failed", error=str(e))
                self._post_discord_failure_notice(
                    job_path=job_path,
                    job=job,
                    run_log=run_log,
                    story_log=story_log,
                    error_type="runner_error",
                    error_message=str(e),
                )
            finally:
                lock.release()
                if dedupe_lock:
                    dedupe_lock.release()
            raise

    def _poll_and_advance(self, rt: JobRuntime, run_defaults: dict[str, Any]) -> bool:
        """Polls one job; returns True if job completed (and lock released)."""
        job_path = rt.job_path
        job = rt.job

        run_log, story_log = self._loggers_for_job(job, job_path)
        prompt_def, validator_def = self._resolve_prompt_and_validator(job)
        # If a previous rescue attempt mutated validation.validator_id, ensure that
        # worker-phase validation doesn't accidentally use the rescue validator.
        if rt.phase == "worker" and validator_def.validator_id != prompt_def.validator_id:
            recover = job.get("recover", {}) or {}
            rescue_prompt_id = recover.get("rescue_prompt_id") or "news_rescue_v1"
            rescue_validator_id = self._prompt_registry.resolve_prompt(str(rescue_prompt_id)).validator_id
            if validator_def.validator_id == rescue_validator_id:
                validator_def = self._prompt_registry.resolve_validator(prompt_def.validator_id)
                validation = job.get("validation", {}) or {}
                validation["validator_id"] = prompt_def.validator_id
                job["validation"] = validation
                self._update_job_file(job_path, job, story_log, event="state_update", field="validation.validator_id")

        now = time.time()
        if now < rt.next_poll_ts:
            return False

        # If dry-run, we treat as completed without mutating.
        if self._dry_run:
            story_log.log("dry_run_complete", phase=rt.phase)
            rt.lock.release()
            if rt.dedupe_lock:
                rt.dedupe_lock.release()
            return True

        session_key = None
        if rt.phase == "worker":
            session_key = ((job.get("state", {}) or {}).get("worker", {}) or {}).get("child_session_key")
        else:
            session_key = ((job.get("state", {}) or {}).get("rescue", {}) or {}).get("child_session_key")

        if not session_key:
            return self._start_rescue_or_fail(rt, run_defaults, reason={"error_type": "missing_session", "error_message": "missing child_session_key"})

        # Poll BEFORE checking deadline: avoids a race where the sub-agent
        # completed between the last poll tick and the deadline, which would
        # otherwise trigger an unnecessary rescue/failure.
        rt.next_poll_ts = now + max(1, int(rt.poll_seconds))
        try:
            hist = self._tool_invoke_result_json_retry(
                tool="sessions_history",
                action="json",
                args={"sessionKey": session_key, "limit": 25, "includeTools": False},
                run_log=run_log,
                story_log=story_log,
                retries=3,
                base_sleep=1.0,
            )
        except Exception as e:
            # Gateway can restart transiently; don't fail the job just because one poll failed.
            story_log.log("monitor_poll_failed", phase=rt.phase, session_key=session_key, error=str(e))
            # If past deadline and the poll itself failed, trigger timeout now.
            if now > rt.deadline_ts:
                story_log.log("timeout", phase=rt.phase)
                return self._start_rescue_or_fail(rt, run_defaults, reason={"error_type": "timeout", "error_message": "worker timed out"})
            return False
        messages = hist.get("messages", [])
        if not isinstance(messages, list):
            messages = []
        result_json = find_result_json_in_messages(messages)
        # Fallback: try reading the session file directly when the API
        # returned messages but we still couldn't extract a result JSON
        # (e.g. content format mismatch between API and JSONL).
        if not result_json:
            result_json = self._try_read_session_file(session_key, story_log)
        if not result_json:
            if _assistant_terminal_empty_output(messages):
                story_log.log("terminal_empty_output", phase=rt.phase, session_key=session_key)
                return self._start_rescue_or_fail(
                    rt,
                    run_defaults,
                    reason={"error_type": "unknown", "error_message": "terminal_empty_output"},
                )
            # Stale-poll detection: if we have assistant messages but still
            # no parseable result after many consecutive polls, the worker
            # likely finished with an unparseable output.  Escalate early
            # instead of burning the full timeout budget.
            _STALE_POLL_LIMIT = 24  # ~2 minutes at 5s poll interval
            rt._stale_polls = getattr(rt, "_stale_polls", 0) + (1 if messages else 0)
            if messages and rt._stale_polls >= _STALE_POLL_LIMIT:
                story_log.log("stale_poll_escalation", phase=rt.phase, session_key=session_key,
                              stale_polls=rt._stale_polls, msg_count=len(messages))
                return self._start_rescue_or_fail(
                    rt,
                    run_defaults,
                    reason={"error_type": "stale_poll", "error_message": f"result not found after {rt._stale_polls} polls with {len(messages)} messages"},
                )
            # No result yet; only trigger timeout if past deadline.
            if now > rt.deadline_ts:
                story_log.log("timeout", phase=rt.phase)
                return self._start_rescue_or_fail(rt, run_defaults, reason={"error_type": "timeout", "error_message": "worker timed out"})
            story_log.log("monitor_poll", phase=rt.phase, session_key=session_key)
            return False

        story_log.log("result_found", phase=rt.phase, status=result_json.get("status"))

        # Best-effort: capture token/cost usage from the history payload so we can
        # report it even when sub-agent sessions are cleaned up.
        try:
            usage_tokens = 0
            usage_cost = 0.0
            for m in messages:
                if not isinstance(m, dict) or m.get("role") != "assistant":
                    continue
                u = m.get("usage") or {}
                t = u.get("totalTokens")
                if isinstance(t, int):
                    usage_tokens += int(t)
                c = u.get("cost") or {}
                try:
                    usage_cost += float(c.get("total") or 0.0)
                except Exception:
                    pass
            if usage_tokens or usage_cost:
                state = job.get("state", {}) or {}
                if rt.phase == "worker":
                    worker_state = state.get("worker", {}) or {}
                    worker_state["usage"] = {"tokens": int(usage_tokens), "cost": float(usage_cost)}
                    state["worker"] = worker_state
                else:
                    rescue_state = state.get("rescue", {}) or {}
                    rescue_state["usage"] = {"tokens": int(usage_tokens), "cost": float(usage_cost)}
                    state["rescue"] = rescue_state
                job["state"] = state
        except Exception as e:
            story_log.log("usage_capture_failed", error=str(e), phase=rt.phase)

        spawn = job.get("spawn", {}) or {}
        publisher_mode = str(spawn.get("publisher_mode") or "agent_posts")

        # In agent_posts mode, recompute metrics from Discord truth (worker posts).
        # In script_posts mode, the worker returns a draft and the runner posts it later.
        if publisher_mode == "agent_posts":
            result_json = self._augment_success_result_with_discord_metrics(
                job=job,
                result_json=result_json,
                run_log=run_log,
                story_log=story_log,
            )

        outcome = self._validate_result(job=job, validator_def=validator_def, result_json=result_json)
        if not outcome.ok:
            story_log.log("result_invalid", errors=outcome.errors)
            # Before triggering an expensive rescue, try a small deterministic repair pass.
            try:
                repaired, repaired_outcome, _repairs = self._try_repair_validation_failure(
                    job=job,
                    validator_def=validator_def,
                    result_json=result_json,
                    errors=outcome.errors,
                    story_log=story_log,
                )
                if repaired_outcome.ok:
                    story_log.log("result_repair_ok")
                    result_json = repaired
                    outcome = repaired_outcome
                else:
                    story_log.log("result_repair_not_ok", errors=repaired_outcome.errors)
            except Exception as e:
                story_log.log("result_repair_failed", error=str(e))

            if not outcome.ok:
                # Treat validation failure as a worker failure and attempt rescue.
                return self._start_rescue_or_fail(
                    rt,
                    run_defaults,
                    reason={
                        "error_type": "validation_failed",
                        "error_message": f"validator failed: {outcome.errors}",
                    },
                    worker_result=result_json,
                )

        status = result_json.get("status")
        if status == "SUCCESS":
            if publisher_mode == "script_posts":
                # Ensure Discord containers exist *now* (title post + thread) before publishing.
                # In script_posts mode, delaying container creation avoids empty threads when the
                # worker/rescue fails, and also lets us use the worker's (validated) Cantonese
                # title to name the thread.
                destination_platform = (job.get("destination", {}) or {}).get("platform")
                if destination_platform == "discord":
                    discord_state = (job.get("state", {}) or {}).get("discord", {}) or {}
                    # If the title message hasn't been posted yet, prefer the worker's title.
                    if not discord_state.get("title_message_id"):
                        new_title = result_json.get("title")
                        if isinstance(new_title, str):
                            new_title = new_title.strip()
                        else:
                            new_title = ""
                        if new_title and (not _looks_like_english_title(new_title)) and _count_cjk(new_title) >= 4:
                            story = job.get("story", {}) or {}
                            old_title = str(story.get("title") or "").strip()
                            if new_title != old_title:
                                story["title"] = new_title
                                job["story"] = story
                                story_log.log("title_from_result_applied_prethread", from_title=old_title, to_title=new_title)
                                self._update_job_file(job_path, job, story_log, event="title_from_result_applied_prethread", title=new_title)

                    self._prepare_discord(job_path, job, run_log, story_log)

                # Publish draft now (runner is deterministic; avoids LLM tool-call loops).
                try:
                    result_json = self._publish_script_posts_draft(
                        job_path=job_path,
                        job=job,
                        result_json=result_json,
                        run_log=run_log,
                        story_log=story_log,
                    )
                except Exception as e:
                    story_log.log("script_posts_publish_failed", error=str(e))
                    return self._start_rescue_or_fail(
                        rt,
                        run_defaults,
                        reason={"error_type": "discord_error", "error_message": f"script_posts_publish_failed:{e}"},
                        worker_result=result_json,
                    )

            self._maybe_update_discord_title_from_result(job=job, result_json=result_json, run_log=run_log, story_log=story_log)
            # Mark done.
            state = job.get("state", {}) or {}
            state["status"] = "SUCCESS" if rt.phase == "worker" else "RESCUED"
            if rt.phase == "worker":
                worker_state = state.get("worker", {}) or {}
                worker_state["ended_at"] = utc_iso()
                state["worker"] = worker_state
            else:
                rescue_state = state.get("rescue", {}) or {}
                rescue_state["ended_at"] = utc_iso()
                state["rescue"] = rescue_state
            state["locked_by"] = None
            state["locked_at"] = None
            job["state"] = state

            result = job.get("result", {}) or {}
            if rt.phase == "worker":
                result["final_status"] = "SUCCESS"
                result["worker_result_json"] = result_json
            else:
                result["final_status"] = "RESCUED"
                result["rescue_result_json"] = result_json
            job["result"] = result

            self._update_job_file(job_path, job, story_log, event="job_complete", final_status=result.get("final_status"))

            # Cross-run dedupe marker: only write after SUCCESS/RESCUED so failures don't block retries.
            try:
                dedupe_key = self._dedupe_key_for_job(job)
                if dedupe_key:
                    marker_path = self._dedupe_marker_path(dedupe_key)
                    self._write_dedupe_marker(marker_path=marker_path, dedupe_key=dedupe_key, job_path=job_path, job=job)
                    story_log.log("dedupe_marker_written", dedupe_key=dedupe_key, marker=str(marker_path))
            except Exception as e:
                story_log.log("dedupe_marker_write_failed", error=str(e))

            # Record event in posted_events ledger for cross-run LLM dedupe.
            _event_id = job.get("story", {}).get("event_id")
            for _attempt in range(3):
                try:
                    self._record_posted_event(job_path=job_path, job=job, story_log=story_log)
                    break
                except Exception as e:
                    if _attempt < 2:
                        time.sleep(1)
                    else:
                        story_log.log(
                            "posted_event_record_failed_after_retries",
                            error=str(e),
                            event_id=_event_id,
                        )
                        _log.warning(
                            "CRITICAL: failed to mark event as posted after 3 retries: event_id=%s error=%s",
                            _event_id,
                            e,
                        )

            rt.lock.release()
            if rt.dedupe_lock:
                rt.dedupe_lock.release()
            return True

        # status == FAILURE
        worker_error = {
            "error_type": str(result_json.get("error_type") or "unknown"),
            "error_message": str(result_json.get("error_message") or "worker failure"),
        }
        if rt.phase == "worker":
            # Store worker result and attempt rescue.
            job_result = job.get("result", {}) or {}
            job_result["worker_result_json"] = result_json
            job["result"] = job_result
            self._update_job_file(job_path, job, story_log, event="worker_failed", error_type=worker_error["error_type"])
            return self._start_rescue_or_fail(rt, run_defaults, reason=worker_error, worker_result=result_json)

        # Rescue failed -> final failure.
        state = job.get("state", {}) or {}
        state["status"] = "FAILURE"
        state["locked_by"] = None
        state["locked_at"] = None
        rescue_state = state.get("rescue", {}) or {}
        rescue_state["ended_at"] = utc_iso()
        state["rescue"] = rescue_state
        job["state"] = state

        job_result = job.get("result", {}) or {}
        job_result["final_status"] = "FAILURE"
        job_result["rescue_result_json"] = result_json
        job["result"] = job_result

        self._update_job_file(job_path, job, story_log, event="job_failed", error_type=worker_error["error_type"])
        self._post_discord_failure_notice(
            job_path=job_path,
            job=job,
            run_log=run_log,
            story_log=story_log,
            error_type=worker_error["error_type"],
            error_message=worker_error["error_message"],
        )
        rt.lock.release()
        if rt.dedupe_lock:
            rt.dedupe_lock.release()
        return True

    def _start_rescue_or_fail(
        self,
        rt: JobRuntime,
        run_defaults: dict[str, Any],
        *,
        reason: dict[str, str],
        worker_result: dict[str, Any] | None = None,
    ) -> bool:
        job_path = rt.job_path
        job = rt.job
        run_log, story_log = self._loggers_for_job(job, job_path)

        recover = job.get("recover", {}) or {}
        if not recover.get("enabled", False):
            return self._final_fail(rt, reason=reason, worker_result=worker_result)

        max_attempts = int(recover.get("max_rescue_attempts", 0))
        rescue_state = (job.get("state", {}) or {}).get("rescue", {}) or {}
        attempt = int(rescue_state.get("attempt", 0))
        if attempt >= max_attempts:
            story_log.log("rescue_skipped_max_attempts", attempt=attempt, max_attempts=max_attempts)
            return self._final_fail(rt, reason=reason, worker_result=worker_result)

        # Best-effort: persist why rescue is being triggered (and the worker's last RESULT JSON
        # when available) *before* spawning rescue. This makes RESCUED jobs debuggable without
        # having to scrape session transcripts.
        try:
            job_result = job.get("result", {}) or {}
            if rt.phase == "worker" and worker_result is not None and job_result.get("worker_result_json") is None:
                job_result["worker_result_json"] = worker_result

            msg = str(reason.get("error_message") or "")
            if len(msg) > 400:
                msg = msg[:397] + "..."
            job_result.setdefault("errors", []).append({"type": str(reason.get("error_type") or "failure"), "message": msg, "at": utc_iso()})
            job["result"] = job_result
            self._update_job_file(job_path, job, story_log, event="rescue_triggered", reason=reason)
        except Exception as e:
            story_log.log("rescue_trigger_persist_failed", error=str(e))

        rescue_prompt_id = recover.get("rescue_prompt_id") or "news_rescue_v1"
        rescue_prompt_def = self._prompt_registry.resolve_prompt(str(rescue_prompt_id))
        rescue_validator_id = rescue_prompt_def.validator_id
        (job.get("validation", {}) or {})["validator_id"] = rescue_validator_id
        job["validation"] = job.get("validation", {}) or {"validator_id": rescue_validator_id, "stop_run_on_failure": False}
        rescue_validator_def = self._prompt_registry.resolve_validator(rescue_validator_id)

        # Optionally stop the worker session first.
        worker_session_key = ((job.get("state", {}) or {}).get("worker", {}) or {}).get("child_session_key")
        # If we already have a RESULT JSON, the worker run is effectively done; sending /stop
        # often just creates extra noisy turns (wastes tokens). Only stop when we timed out or
        # we never received a worker_result.
        if worker_session_key and worker_result is None:
            try:
                self._gateway.invoke(
                    tool="sessions_send",
                    action="json",
                    args={"sessionKey": worker_session_key, "message": "/stop", "timeoutSeconds": 0},
                )
                story_log.log("worker_stop_sent", session_key=worker_session_key)
            except Exception as e:
                story_log.log("worker_stop_failed", error=str(e))

        # Spawn rescue agent.
        rescue_timeout = int(recover.get("rescue_timeout_seconds", 600))
        spawn_resp = self._spawn_subagent(
            job=job,
            job_path=job_path,
            prompt_def=rescue_prompt_def,
            run_log=run_log,
            story_log=story_log,
            timeout_seconds=rescue_timeout + 60,
            label_suffix="rescue",
            worker_error=reason,
        )

        state = job.get("state", {}) or {}
        rescue_state = state.get("rescue", {}) or {}
        rescue_state["attempt"] = attempt + 1
        rescue_state["child_session_key"] = spawn_resp.get("childSessionKey")
        rescue_state["session_id"] = spawn_resp.get("sessionId")
        rescue_state["started_at"] = utc_iso()
        state["rescue"] = rescue_state
        state["status"] = "DISPATCHED"
        job["state"] = state
        self._update_job_file(job_path, job, story_log, event="rescue_started", attempt=rescue_state["attempt"])

        # Switch runtime to rescue mode.
        rt.phase = "rescue"
        rt.deadline_ts = time.time() + rescue_timeout
        rt.poll_seconds = int((job.get("monitor", {}) or {}).get("poll_seconds", 5))
        rt.next_poll_ts = time.time()

        # Swap validator to rescue validator for the poll loop.
        job["validation"]["validator_id"] = rescue_validator_def.validator_id
        rt.job = job
        story_log.log("validator_switched", validator_id=rescue_validator_def.validator_id)
        return False

    def _final_fail(self, rt: JobRuntime, *, reason: dict[str, str], worker_result: dict[str, Any] | None) -> bool:
        job_path = rt.job_path
        job = rt.job
        run_log, story_log = self._loggers_for_job(job, job_path)

        state = job.get("state", {}) or {}
        state["status"] = "FAILURE"
        state["locked_by"] = None
        state["locked_at"] = None
        job["state"] = state

        result = job.get("result", {}) or {}
        result["final_status"] = "FAILURE"
        if worker_result is not None:
            result["worker_result_json"] = worker_result
        result.setdefault("errors", []).append({"type": reason.get("error_type", "failure"), "message": reason.get("error_message", ""), "at": utc_iso()})
        job["result"] = result

        self._update_job_file(job_path, job, story_log, event="job_failed_final", reason=reason)
        self._post_discord_failure_notice(
            job_path=job_path,
            job=job,
            run_log=run_log,
            story_log=story_log,
            error_type=str(reason.get("error_type") or "failure"),
            error_message=str(reason.get("error_message") or ""),
        )
        rt.lock.release()
        if rt.dedupe_lock:
            rt.dedupe_lock.release()
        return True

    def run_group(self, *, job_paths: list[Path], run_json_path: Path | None = None) -> dict[str, Any]:
        # Load run defaults if present.
        run_defaults: dict[str, Any] = {
            "concurrency": 1,
            "stagger_seconds": 0,
            "default_timeout_seconds": 900,
            "dm_targets": [],
            "stop_run_on_failure": False,
        }
        run_meta: dict[str, Any] = {}
        if run_json_path and run_json_path.exists():
            run_job = load_json_file(run_json_path)
            try:
                self._validate_run_job(run_job)
            except JobSchemaError as exc:
                self._log.warning("Skipping run group %s: %s", run_json_path, exc)
                for jp in job_paths:
                    try:
                        self._mark_job_skipped_policy(
                            job_path=jp,
                            job=load_json_file(jp),
                            reason=f"run_json_validation_failed: {exc}",
                        )
                    except Exception:
                        pass
                return {"run": str(run_json_path), "total": len(job_paths), "completed": 0,
                        "failed": 0, "skipped": len(job_paths), "dry_run": self._dry_run,
                        "error": str(exc)}
            runner_cfg = run_job.get("runner", {}) or {}
            run_defaults["concurrency"] = int(runner_cfg.get("concurrency", 1))
            run_defaults["stagger_seconds"] = int(runner_cfg.get("stagger_seconds", 0))
            run_defaults["default_timeout_seconds"] = int(runner_cfg.get("default_timeout_seconds", 900))
            run_defaults["dm_targets"] = list(runner_cfg.get("dm_targets", []) or [])
            run_defaults["stop_run_on_failure"] = bool(runner_cfg.get("stop_run_on_failure", False))
            run_meta = run_job.get("run", {}) or {}

        # Sort jobs deterministically.
        job_paths = sorted(job_paths)

        active: list[JobRuntime] = []
        completed: list[Path] = []
        failures: list[Path] = []

        # A loop that (a) starts new jobs up to concurrency and (b) polls active jobs.
        i = 0
        last_spawn_ts = 0.0
        terminal = {"SUCCESS", "FAILURE", "RESCUED", "SKIPPED"}

        while True:
            # Start jobs if capacity allows.
            while i < len(job_paths) and len(active) < int(run_defaults["concurrency"]):
                # Stagger between spawns (avoid provider spikes).
                stagger = float(run_defaults["stagger_seconds"])
                now = time.time()
                if stagger > 0 and now - last_spawn_ts < stagger:
                    break

                jp = job_paths[i]
                i += 1
                try:
                    rt = self._start_job(job_path=jp, run_defaults=run_defaults)
                except Exception as e:
                    # Robustness: a single malformed job should not crash the whole run.
                    try:
                        job = load_json_file(jp)
                        # Best-effort: mark this job as failed so cron doesn't retry forever.
                        state = job.get("state", {}) or {}
                        state["status"] = "FAILURE"
                        state["locked_by"] = None
                        state["locked_at"] = None
                        job["state"] = state

                        result = job.get("result", {}) or {}
                        result["final_status"] = "FAILURE"
                        result.setdefault("errors", []).append({"type": "start_job_exception", "message": str(e), "at": utc_iso()})
                        job["result"] = result

                        if not self._dry_run:
                            atomic_write_json(jp, job)
                    except Exception as inner:
                        # If we can't read/write the job file, jail it so cron
                        # doesn't retry a broken file indefinitely.
                        if not self._dry_run:
                            jail_job_file(jp, reason=f"start_job_exception: {e} / write_back_failed: {inner}")
                    completed.append(jp)
                    failures.append(jp)
                    continue
                if rt:
                    active.append(rt)
                    last_spawn_ts = time.time()
                    continue

                # _start_job returning None can mean "already terminal", "skipped",
                # or "deferred due to locks". For run summaries, count only terminal
                # statuses so we don't misreport active/locked jobs as completed.
                try:
                    cur = load_json_file(jp)
                    status = str(((cur.get("state", {}) or {}).get("status")) or "").strip().upper()
                    if status in terminal:
                        completed.append(jp)
                        if status == "FAILURE":
                            failures.append(jp)
                except Exception:
                    pass

            # Poll active jobs.
            still_active: list[JobRuntime] = []
            any_completed = False
            for rt in active:
                try:
                    done = self._poll_and_advance(rt, run_defaults)
                except Exception as e:
                    # Fail-safe: mark job failed.
                    done = self._final_fail(rt, reason={"error_type": "runner_exception", "error_message": str(e)}, worker_result=None)
                if done:
                    any_completed = True
                    completed.append(rt.job_path)
                    # Count failures for summary.
                    status = ((rt.job.get("result", {}) or {}).get("final_status")) or ((rt.job.get("state", {}) or {}).get("status"))
                    if status in ("FAILURE",):
                        failures.append(rt.job_path)
                else:
                    still_active.append(rt)
            active = still_active

            # Exit condition: queue exhausted and no active jobs.
            if i >= len(job_paths) and not active:
                break

            # If nothing completed this tick, sleep a little.
            if not any_completed:
                time.sleep(1.0)

            # Optional: stop spawning new jobs if configured and any failure occurred.
            if run_defaults.get("stop_run_on_failure") and failures:
                i = len(job_paths)  # prevent more spawns

        return {
            "run": run_meta,
            "total": len(job_paths),
            "completed": len(completed),
            "failed": len(failures),
            "dry_run": self._dry_run,
        }

    def send_dm_summary(self, *, dm_targets: list[str], summary: str) -> None:
        if not dm_targets:
            return
        for target in dm_targets:
            t = str(target)
            if t.startswith("discord:"):
                t = t[len("discord:") :]
            if self._dry_run:
                continue
            self._gateway.invoke(tool="message", action="send", args={"channel": "discord", "target": t, "message": summary})


def discover_story_job_files(path: Path) -> list[Path]:
    candidates: list[Path] = []
    if path.is_file():
        if path.name in {"run.json", "run_summary.json"}:
            return []
        candidates = [path]
    elif path.is_dir():
        # Run directory conventions:
        # - run.json + story_*.json
        # - single-story dirs with story.json
        if (path / "run.json").exists():
            candidates = sorted(p for p in path.glob("story_*.json") if p.is_file())
        else:
            candidates = sorted(p for p in path.glob("*.json") if p.is_file() and p.name not in {"run.json", "run_summary.json"})
    else:
        return []

    # Filter to story_job_v1 only (fast check).
    out: list[Path] = []
    for p in candidates:
        try:
            obj = load_json_file(p)
        except Exception:
            continue
        if obj.get("schema_version") == "story_job_v1":
            out.append(p)
    return sorted(out)


def discover_jobs_under(jobs_root: Path) -> list[Path]:
    out: list[Path] = []
    if not jobs_root.exists():
        return []
    for child in sorted(jobs_root.iterdir()):
        if child.is_dir():
            out.extend(discover_story_job_files(child))
        elif child.is_file() and child.suffix == ".json" and child.name != "run.json":
            out.append(child)
    # Filter to story_job_v1 only (fast check).
    story_jobs: list[Path] = []
    for p in out:
        try:
            obj = load_json_file(p)
            if obj.get("schema_version") != "story_job_v1":
                continue

            # Important for cron-style runners: do not re-run already finished jobs.
            # This avoids polluting per-run logs and keeps periodic runners cheap.
            state = obj.get("state", {}) or {}
            status = str(state.get("status") or "").strip().upper()
            if status in {"SUCCESS", "FAILURE", "RESCUED", "SKIPPED"}:
                continue

            # Skip stale PLANNED/DISPATCHED jobs older than 48h to prevent
            # broken run folders from accumulating indefinitely.
            run_info = obj.get("run", {}) or {}
            run_time_str = str(run_info.get("run_time_uk") or "").strip()
            if run_time_str:
                try:
                    run_dt = datetime.strptime(run_time_str[:16], "%Y-%m-%d %H:%M")
                    run_dt = run_dt.replace(tzinfo=ZoneInfo("Europe/London"))
                    age_hours = (datetime.now(UTC) - run_dt.astimezone(UTC)).total_seconds() / 3600
                    if age_hours > 48:
                        continue
                except (ValueError, TypeError):
                    pass

            story_jobs.append(p)
        except Exception:
            continue
    return sorted(story_jobs)
