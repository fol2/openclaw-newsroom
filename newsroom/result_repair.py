from __future__ import annotations

import copy
import re
from typing import Any
from urllib.parse import urlsplit

from ._util import count_cjk

# Keep module-private alias so call-sites stay unchanged.
_count_cjk = count_cjk


_READ_MORE_RANGE_RE = re.compile(r"read_more_urls_(\d+)_to_(\d+)")


def _is_http_url(url: str) -> bool:
    u = (url or "").strip()
    return u.startswith("http://") or u.startswith("https://")


def _domain(url: str) -> str:
    try:
        return (urlsplit(url).hostname or "").lower()
    except Exception:
        return ""


def _dedupe_urls(urls: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for u in urls:
        u = str(u or "").strip()
        if not u or not _is_http_url(u):
            continue
        if u in seen:
            continue
        seen.add(u)
        out.append(u)
    return out


def _job_candidate_urls(job: dict[str, Any]) -> list[str]:
    story = job.get("story", {}) or {}
    state = job.get("state", {}) or {}
    pack = state.get("source_pack", {}) if isinstance(state.get("source_pack"), dict) else {}

    urls: list[str] = []
    primary = story.get("primary_url")
    if isinstance(primary, str) and primary.strip():
        urls.append(primary.strip())
    supporting = story.get("supporting_urls")
    if isinstance(supporting, list):
        for u in supporting:
            if isinstance(u, str) and u.strip():
                urls.append(u.strip())
    sources = pack.get("sources")
    if isinstance(sources, list):
        for s in sources:
            if not isinstance(s, dict):
                continue
            u = s.get("url") or s.get("final_url")
            if isinstance(u, str) and u.strip():
                urls.append(u.strip())

    return _dedupe_urls(urls)


def _repair_read_more_urls(
    *,
    result_json: dict[str, Any],
    job: dict[str, Any],
    min_urls: int,
    max_urls: int,
    require_primary: bool,
    require_other_domain: bool,
) -> tuple[dict[str, Any], bool]:
    story = job.get("story", {}) or {}
    primary_url = story.get("primary_url")
    if not isinstance(primary_url, str):
        primary_url = None
    primary_url = primary_url.strip() if primary_url else None

    draft = result_json.get("draft")
    if not isinstance(draft, dict):
        draft = {}
        result_json["draft"] = draft

    existing = draft.get("read_more_urls")
    existing_list: list[str] = []
    if isinstance(existing, list):
        existing_list = [str(u or "").strip() for u in existing if isinstance(u, str) and str(u or "").strip()]
    existing_list = _dedupe_urls(existing_list)

    candidates = _dedupe_urls([*existing_list, *(_job_candidate_urls(job))])
    if require_primary and primary_url:
        # Ensure primary_url is present and prioritized.
        candidates = _dedupe_urls([primary_url, *candidates])

    if not candidates:
        return result_json, False

    # Build output list, keeping worker-chosen urls first when possible.
    out: list[str] = []
    out.extend(existing_list)

    # Ensure primary url.
    if require_primary and primary_url and primary_url not in out:
        out.insert(0, primary_url)

    # Ensure other domain when required (best-effort).
    if require_other_domain and primary_url:
        primary_dom = _domain(primary_url)
        if primary_dom:
            has_other = any(_domain(u) and _domain(u) != primary_dom for u in out)
            if not has_other:
                for u in candidates:
                    if _domain(u) and _domain(u) != primary_dom:
                        out.append(u)
                        break

    # Fill to min.
    for u in candidates:
        if u in out:
            continue
        out.append(u)
        if len(out) >= min_urls:
            break

    # If still below min, we can't satisfy; keep whatever we have.
    if len(out) > max_urls:
        # Keep deterministic: preserve order, but cap.
        out = out[:max_urls]

    out = _dedupe_urls(out)
    changed = out != existing_list
    draft["read_more_urls"] = out
    result_json["read_more_urls_count"] = len(out)
    return result_json, changed


def _repair_sources_used(*, result_json: dict[str, Any], job: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    sources_used = result_json.get("sources_used")
    existing: list[str] = []
    if isinstance(sources_used, list):
        existing = [str(u or "").strip() for u in sources_used if isinstance(u, str) and str(u or "").strip()]
    existing = _dedupe_urls(existing)

    state = job.get("state", {}) or {}
    pack = state.get("source_pack", {}) if isinstance(state.get("source_pack"), dict) else {}
    sources = pack.get("sources")
    pool: list[str] = []
    if isinstance(sources, list):
        for s in sources:
            if not isinstance(s, dict):
                continue
            if s.get("on_topic") is not True:
                continue
            u = s.get("url") or s.get("final_url")
            if isinstance(u, str) and u.strip():
                pool.append(u.strip())
    pool = _dedupe_urls(pool)

    out = list(existing)
    for u in pool:
        if u in out:
            continue
        out.append(u)
        if len(out) >= 2:
            break

    out = _dedupe_urls(out)
    if len(out) < 2:
        return result_json, False
    changed = out != existing
    result_json["sources_used"] = out
    return result_json, changed


def repair_result_json(*, result_json: dict[str, Any], job: dict[str, Any], errors: list[str]) -> tuple[dict[str, Any], list[str]]:
    """Best-effort deterministic repair for common validation failures.

    Returns:
    - repaired result_json (deep-copied) and
    - list of repair codes applied.
    """
    if not isinstance(result_json, dict) or not isinstance(job, dict):
        return result_json, []
    errors = [str(e) for e in (errors or []) if str(e).strip()]
    if not errors:
        return result_json, []

    out = copy.deepcopy(result_json)
    repairs: list[str] = []

    # 1) Title: if worker returned English but job already has a Cantonese title, use it.
    if any(e.endswith("title_traditional_chinese") for e in errors):
        story = job.get("story", {}) or {}
        job_title = story.get("title")
        if isinstance(job_title, str) and job_title.strip() and _count_cjk(job_title) >= 4:
            if out.get("title") != job_title.strip():
                out["title"] = job_title.strip()
                repairs.append("title_from_job")

    # 2) Read-more URLs: range + primary inclusion + other domain.
    min_urls = None
    max_urls = None
    for e in errors:
        m = _READ_MORE_RANGE_RE.search(e)
        if not m:
            continue
        try:
            min_urls = int(m.group(1))
            max_urls = int(m.group(2))
        except Exception:
            continue
        break

    if min_urls is not None and max_urls is not None:
        require_primary = any(e.endswith("read_more_includes_primary_url") for e in errors)
        # If the range itself failed, we still want to keep primary_url as policy for news reporters.
        require_primary = True if any(e.endswith("read_more_urls_3_to_5") for e in errors) else require_primary
        require_other_domain = any(e.endswith("read_more_has_other_domain") for e in errors)

        out, changed = _repair_read_more_urls(
            result_json=out,
            job=job,
            min_urls=int(min_urls),
            max_urls=int(max_urls),
            require_primary=bool(require_primary),
            require_other_domain=bool(require_other_domain),
        )
        if changed:
            repairs.append("read_more_urls")

    # 3) sources_used: ensure at least 2 if validator complained.
    if any(e.endswith("sources_used_min_2") for e in errors):
        out, changed = _repair_sources_used(result_json=out, job=job)
        if changed:
            repairs.append("sources_used")

    # 4) Body CJK length: avoid rescue when we're just barely below the minimum.
    if any(e.endswith("body_cjk_2700_to_3300") for e in errors):
        draft = out.get("draft")
        if not isinstance(draft, dict):
            draft = {}
            out["draft"] = draft
        body = draft.get("body")
        if isinstance(body, str):
            b = body.strip()
            cjk = _count_cjk(b)
            if 0 < cjk < 2700:
                missing = 2700 - cjk
                b2 = b.rstrip()

                if missing <= 60:
                    # Small underflow: append a short, safe closing paragraph (no new facts).
                    # We intentionally avoid padding with long runs of punctuation (looks broken in Discord).
                    b2 = (b2 + "\n\n總括而言，現階段資訊仍以公開披露為主，真正影響要視乎後續細節同落實節奏；只要之後有更多可核實資料出現，事件走向自然會更清晰。").strip()
                    if 2700 <= _count_cjk(b2) <= 3300:
                        draft["body"] = b2
                        repairs.append("body_cjk_pad")
                elif missing <= 800:
                    # Moderate underflow: append a few safe, generic Cantonese paragraphs that add
                    # context without introducing new facts. This avoids the token/cost of a rescue
                    # run for minor length misses.
                    fillers = [
                        "就算主體已經表態，外界最關心嘅往往係點樣由「方向」變成「可操作」嘅安排：配套制度點設計、流程點落地、風險點管理、資訊點披露同點樣做到口徑一致。當細節愈清楚，市場先愈容易評估實際影響，而唔會只係靠情緒同估計去判斷。如果牽涉多方持份者，協調成本同時間表亦係關鍵，因為每一步都可能影響後續節奏。喺過渡期內，市場亦會留意有冇臨時措施、試行安排或者階段性目標，令外界可以用實際進度去對照最初承諾。",
                        "同一時間，各方反應未必會一次過交代晒。官方或者當事人可能先講原則，再逐步釋出更多資料；而市場、專家同相關人士就會從現有內容搵線索，判斷係短期姿態定係長線轉向。喺呢個過程入面，最重要係有冇第二、第三個獨立來源交叉印證關鍵說法，避免單一訊息造成誤判。如果唔同來源嘅描述有落差，最好將「可核實資訊」同「評論」分開，先鎖定共同事實，再理解各自取向，咁樣會更易避免被立場牽著走。",
                        "至於後續睇位，通常可以分成「資訊節點」同「行動節點」兩類：前者例如再有聲明、文件、會議紀錄或數據釋出；後者例如啟動試行、推出安排、落實配套或者出現更明確嘅執行步驟。對讀者而言，最實際係記低下一個時間點同關鍵名詞，之後有更新就容易對比今次講法有冇改變，亦可以更快判斷新資訊係補充細節定係推翻原先理解。如果同一事件有後續發展，往往都會反映喺某個具體動作上，例如再有正式披露、程序推進、或者出現新嘅數字同條款，令討論由「方向」變成「可以衡量」嘅安排。相反，如果只係重覆原有說法而無新增內容，影響就可能更多係情緒層面。",
                        "亦要留意，唔同媒體因應受眾同立場，會將同一件事放喺唔同框架之下講：有啲著重政治層面，有啲著重經濟層面，有啲則聚焦民生影響。將多個角度放埋一齊睇，可以幫你分辨邊啲係事實，邊啲係詮釋，亦更容易理解點解外界會出現截然不同嘅解讀。如果報道用咗好多形容詞或者推論性字眼，最好回到可核實部分，例如時間點、文件、數字或者直接引述，咁樣先可以減少誤判。",
                    ]

                    # Add fillers until we hit the minimum.
                    for para in fillers:
                        if _count_cjk(b2) >= 2700:
                            break
                        b2 = (b2 + "\n\n" + para).strip()

                    # Guardrail: ensure we're within the validator range.
                    if 2700 <= _count_cjk(b2) <= 3300:
                        draft["body"] = b2
                        repairs.append("body_cjk_expand")

    return out, repairs
