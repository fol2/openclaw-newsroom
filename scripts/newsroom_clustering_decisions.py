#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

OPENCLAW_HOME = Path(__file__).resolve().parents[1]
os.environ.setdefault("OPENCLAW_HOME", str(OPENCLAW_HOME))
sys.path.insert(0, str(OPENCLAW_HOME))

_LOW_TOP_SIMILARITY = 0.22
_AMBIGUOUS_TOP_GAP = 0.04
_STRONG_TOP_SIMILARITY = 0.55
_ASSIGN_GAP_OUTLIER = 0.12


def _iso(ts: int | None) -> str | None:
    if not ts:
        return None
    return datetime.fromtimestamp(int(ts), tz=UTC).isoformat(timespec="seconds")


def _normalise_list(values: list[str] | None) -> list[str]:
    out: list[str] = []
    for raw in values or []:
        for part in str(raw).split(","):
            item = part.strip()
            if item:
                out.append(item)
    return out


def _as_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        try:
            return int(s)
        except Exception:
            return None
    return None


def _as_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        try:
            return float(s)
        except Exception:
            return None
    return None


def _maybe_json_text(s: Any) -> Any:
    if s is None:
        return None
    if not isinstance(s, str):
        return s
    t = s.strip()
    if not t:
        return None
    if t[0] not in "[{":
        return t
    try:
        return json.loads(t)
    except Exception:
        return t


def _normalise_action_name(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    s = value.strip().lower()
    return s or None


def _extract_target_event_id(action_obj: Any) -> int | None:
    if not isinstance(action_obj, dict):
        return None
    for key in ("event_id", "parent_event_id"):
        eid = _as_int(action_obj.get(key))
        if eid is not None and eid > 0:
            return eid
    return None


def _table_exists(cur: sqlite3.Cursor, name: str) -> bool:
    row = cur.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ? LIMIT 1",
        (name,),
    ).fetchone()
    return row is not None


def _score_similarity(candidates: list[dict[str, Any]], *, target_event_id: int | None) -> dict[str, Any]:
    top_score: float | None = None
    top_event_id: int | None = None
    second_score: float | None = None
    target_score: float | None = None
    target_rank: int | None = None

    sorted_candidates = sorted(
        candidates,
        key=lambda c: (
            _as_int(c.get("rank")) if _as_int(c.get("rank")) is not None else 999999,
            -(_as_float(c.get("score")) if _as_float(c.get("score")) is not None else -999999.0),
        ),
    )

    for idx, cand in enumerate(sorted_candidates):
        score = _as_float(cand.get("score"))
        if score is None:
            continue
        event_id = _as_int(cand.get("event_id"))
        if top_score is None:
            top_score = score
            top_event_id = event_id
        elif second_score is None:
            second_score = score

        if target_event_id is not None and event_id == target_event_id and target_score is None:
            target_score = score
            rank = _as_int(cand.get("rank"))
            target_rank = rank if rank is not None and rank > 0 else (idx + 1)

    score_gap = None
    if top_score is not None and second_score is not None:
        score_gap = float(top_score - second_score)

    return {
        "candidate_count": len(sorted_candidates),
        "top_event_id": top_event_id,
        "top_score": top_score,
        "second_score": second_score,
        "score_gap": score_gap,
        "target_event_id": target_event_id,
        "target_rank": target_rank,
        "target_score": target_score,
    }


def _unique_preserve(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for it in items:
        if not it:
            continue
        if it in seen:
            continue
        seen.add(it)
        out.append(it)
    return out


def _build_hints(*, action_name: str | None, error_type: str | None, similarity: dict[str, Any]) -> list[str]:
    hints: list[str] = []

    top_score = _as_float(similarity.get("top_score"))
    score_gap = _as_float(similarity.get("score_gap"))
    target_score = _as_float(similarity.get("target_score"))
    target_rank = _as_int(similarity.get("target_rank"))
    candidate_count = _as_int(similarity.get("candidate_count")) or 0

    if isinstance(error_type, str) and error_type.strip():
        hints.append("llm_error")

    if candidate_count <= 0:
        hints.append("no_similarity_candidates")

    if top_score is not None and top_score < _LOW_TOP_SIMILARITY:
        hints.append("weak_top_similarity")

    if candidate_count >= 2 and score_gap is not None and score_gap < _AMBIGUOUS_TOP_GAP:
        hints.append("ambiguous_top_candidates")

    if action_name == "assign":
        if target_rank is not None and target_rank > 1:
            hints.append("assigned_non_top_candidate")
        if top_score is not None and target_score is not None and (top_score - target_score) >= _ASSIGN_GAP_OUTLIER:
            hints.append("assigned_far_below_top_candidate")

    if action_name == "new_event" and top_score is not None and top_score >= _STRONG_TOP_SIMILARITY:
        hints.append("new_event_despite_strong_candidate")

    if action_name == "development" and candidate_count <= 0:
        hints.append("development_without_candidates")

    return _unique_preserve(hints)


def _event_ids_touched(*, link_event_id: int | None, target_event_id: int | None, candidates: list[dict[str, Any]]) -> list[int]:
    out: set[int] = set()
    if link_event_id is not None and link_event_id > 0:
        out.add(link_event_id)
    if target_event_id is not None and target_event_id > 0:
        out.add(target_event_id)
    for cand in candidates:
        eid = _as_int(cand.get("event_id"))
        if eid is not None and eid > 0:
            out.add(eid)
    return sorted(out)


def _event_category(event_by_id: dict[int, dict[str, Any]], event_id: int | None) -> str | None:
    if event_id is None:
        return None
    row = event_by_id.get(event_id)
    if not isinstance(row, dict):
        return None
    category = row.get("category")
    if isinstance(category, str) and category.strip():
        return category.strip()
    return None


def _resolve_category(
    *,
    action_obj: Any,
    linked_event_category: str | None,
    target_event_category: str | None,
) -> tuple[str | None, str | None]:
    action_category: str | None = None
    if isinstance(action_obj, dict):
        raw = action_obj.get("category")
        if isinstance(raw, str) and raw.strip():
            action_category = raw.strip()

    category = action_category or target_event_category or linked_event_category
    return category, action_category


def _decision_matches_category(
    *,
    decision: dict[str, Any],
    category_filters: set[str],
    event_by_id: dict[int, dict[str, Any]],
) -> bool:
    if not category_filters:
        return True

    categories: list[str] = []
    for key in ("category", "action_category", "linked_event_category", "target_event_category"):
        value = decision.get(key)
        if isinstance(value, str) and value.strip():
            categories.append(value.strip())

    for eid in decision.get("event_ids_touched") or []:
        if not isinstance(eid, int):
            continue
        ec = _event_category(event_by_id, eid)
        if ec:
            categories.append(ec)

    for cat in categories:
        if cat.casefold() in category_filters:
            return True
    return False


def _counter_to_sorted_dict(counter: Counter[str]) -> dict[str, int]:
    return {k: int(counter[k]) for k in sorted(counter, key=lambda v: (-counter[v], v))}


def _aggregate_by_link(decisions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, dict[str, Any]] = {}

    for d in decisions:
        link = d.get("link") or {}
        link_id = _as_int(link.get("link_id"))
        if link_id is None:
            continue

        row = grouped.get(link_id)
        if row is None:
            row = {
                "link_id": link_id,
                "domain": link.get("domain"),
                "norm_url": link.get("norm_url"),
                "title": link.get("title"),
                "decision_count": 0,
                "last_decision_at": None,
                "last_decision_at_ts": 0,
                "actions": Counter(),
                "hints": Counter(),
                "target_event_ids": set(),
                "sum_top_score": 0.0,
                "top_score_count": 0,
            }
            grouped[link_id] = row

        row["decision_count"] += 1
        created_at_ts = _as_int(d.get("created_at_ts")) or 0
        if created_at_ts >= int(row.get("last_decision_at_ts") or 0):
            row["last_decision_at_ts"] = created_at_ts
            row["last_decision_at"] = d.get("created_at")

        action = ((d.get("enforced_action") or {}).get("action") if isinstance(d.get("enforced_action"), dict) else None) or "unknown"
        row["actions"][str(action)] += 1

        for hint in d.get("hints") or []:
            if isinstance(hint, str) and hint:
                row["hints"][hint] += 1

        target_event_id = _as_int(d.get("target_event_id"))
        if target_event_id is not None and target_event_id > 0:
            row["target_event_ids"].add(target_event_id)

        top_score = _as_float((d.get("similarity") or {}).get("top_score"))
        if top_score is not None:
            row["sum_top_score"] += top_score
            row["top_score_count"] += 1

    out: list[dict[str, Any]] = []
    for row in grouped.values():
        avg_top = None
        if row["top_score_count"] > 0:
            avg_top = float(row["sum_top_score"] / row["top_score_count"])
        out.append(
            {
                "link_id": row["link_id"],
                "domain": row["domain"],
                "norm_url": row["norm_url"],
                "title": row["title"],
                "decision_count": int(row["decision_count"]),
                "last_decision_at": row["last_decision_at"],
                "actions": _counter_to_sorted_dict(row["actions"]),
                "hints": _counter_to_sorted_dict(row["hints"]),
                "target_event_ids": sorted(int(v) for v in row["target_event_ids"]),
                "avg_top_similarity": avg_top,
            }
        )

    out.sort(key=lambda r: (-int(r["decision_count"]), str(r.get("last_decision_at") or ""), int(r["link_id"])))
    return out


def _aggregate_by_event(decisions: list[dict[str, Any]], event_by_id: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, dict[str, Any]] = {}

    for d in decisions:
        event_ids = {eid for eid in (d.get("event_ids_touched") or []) if isinstance(eid, int) and eid > 0}
        target_event_id = _as_int(d.get("target_event_id"))
        candidate_scores: dict[int, float] = {}
        for cand in d.get("candidates") or []:
            eid = _as_int(cand.get("event_id"))
            score = _as_float(cand.get("score"))
            if eid is not None and score is not None:
                candidate_scores[eid] = score

        for eid in event_ids:
            row = grouped.get(eid)
            if row is None:
                meta = event_by_id.get(eid) or {}
                row = {
                    "event_id": eid,
                    "category": meta.get("category"),
                    "title": meta.get("title") or meta.get("summary_en"),
                    "status": meta.get("status"),
                    "decision_count": 0,
                    "target_hits": 0,
                    "candidate_hits": 0,
                    "outlier_hits": 0,
                    "actions": Counter(),
                    "hints": Counter(),
                    "sum_candidate_score": 0.0,
                    "candidate_score_count": 0,
                }
                grouped[eid] = row

            row["decision_count"] += 1
            action = ((d.get("enforced_action") or {}).get("action") if isinstance(d.get("enforced_action"), dict) else None) or "unknown"
            row["actions"][str(action)] += 1

            if target_event_id is not None and target_event_id == eid:
                row["target_hits"] += 1

            if eid in candidate_scores:
                row["candidate_hits"] += 1
                row["sum_candidate_score"] += float(candidate_scores[eid])
                row["candidate_score_count"] += 1

            hints = [h for h in (d.get("hints") or []) if isinstance(h, str) and h]
            if hints:
                row["outlier_hits"] += 1
                for hint in hints:
                    row["hints"][hint] += 1

    out: list[dict[str, Any]] = []
    for row in grouped.values():
        avg_score = None
        if row["candidate_score_count"] > 0:
            avg_score = float(row["sum_candidate_score"] / row["candidate_score_count"])
        out.append(
            {
                "event_id": row["event_id"],
                "category": row.get("category"),
                "title": row.get("title"),
                "status": row.get("status"),
                "decision_count": int(row["decision_count"]),
                "target_hits": int(row["target_hits"]),
                "candidate_hits": int(row["candidate_hits"]),
                "outlier_hits": int(row["outlier_hits"]),
                "actions": _counter_to_sorted_dict(row["actions"]),
                "hints": _counter_to_sorted_dict(row["hints"]),
                "avg_candidate_similarity": avg_score,
            }
        )

    out.sort(
        key=lambda r: (
            -int(r["decision_count"]),
            -int(r["target_hits"]),
            int(r["event_id"]),
        )
    )
    return out


def _fmt_score(value: Any) -> str:
    score = _as_float(value)
    if score is None:
        return "-"
    return f"{score:.3f}"


def _print_human(out: dict[str, Any]) -> None:
    print("Decision Log Inspector")
    print(f"DB: {out.get('db')}")

    filters = out.get("filters") or {}
    filter_bits: list[str] = []
    for key in ("group_by", "link_id", "event_id", "since_hours", "until_hours"):
        value = filters.get(key)
        if value is not None:
            filter_bits.append(f"{key}={value}")

    domains = filters.get("domains") or []
    if domains:
        filter_bits.append(f"domains={','.join(str(x) for x in domains)}")

    categories = filters.get("categories") or []
    if categories:
        filter_bits.append(f"categories={','.join(str(x) for x in categories)}")

    if filter_bits:
        print(f"Filters: {' | '.join(filter_bits)}")

    summary = out.get("summary") or {}
    print(
        "Matched decisions: "
        f"{summary.get('decision_count', 0)} "
        f"(links={summary.get('unique_links', 0)}, events={summary.get('unique_events', 0)})"
    )

    warnings = out.get("warnings") or []
    for warning in warnings:
        print(f"Warning: {warning}")

    decisions = out.get("decisions") or []
    if not decisions:
        return

    group_by = out.get("group_by")
    if group_by == "link":
        print("\nBy link:")
        for g in out.get("groups") or []:
            print(
                f"- link#{g.get('link_id')} domain={g.get('domain') or '-'} "
                f"decisions={g.get('decision_count')} avg_top_similarity={_fmt_score(g.get('avg_top_similarity'))}"
            )
            print(
                f"  last={g.get('last_decision_at') or '-'} "
                f"actions={g.get('actions') or {}} hints={g.get('hints') or {}}"
            )
        return

    if group_by == "event":
        print("\nBy event:")
        for g in out.get("groups") or []:
            print(
                f"- event#{g.get('event_id')} category={g.get('category') or '-'} "
                f"decisions={g.get('decision_count')} targets={g.get('target_hits')} outliers={g.get('outlier_hits')}"
            )
            print(
                f"  avg_similarity={_fmt_score(g.get('avg_candidate_similarity'))} "
                f"actions={g.get('actions') or {}} hints={g.get('hints') or {}}"
            )
        return

    print("\nDecisions:")
    for d in decisions:
        link = d.get("link") or {}
        sim = d.get("similarity") or {}
        action = (d.get("enforced_action") or {}).get("action") or (d.get("validated_action") or {}).get("action") or "unknown"
        target = d.get("target_event_id")

        print(
            f"- d#{d.get('decision_id')} @ {d.get('created_at') or '-'} "
            f"link#{link.get('link_id')} domain={link.get('domain') or '-'}"
        )
        print(
            f"  action={action} target_event={target or '-'} category={d.get('category') or '-'}"
        )
        print(
            "  similarity="
            f"top:{_fmt_score(sim.get('top_score'))} "
            f"gap:{_fmt_score(sim.get('score_gap'))} "
            f"target_rank:{sim.get('target_rank') or '-'} "
            f"target_score:{_fmt_score(sim.get('target_score'))}"
        )
        hints = d.get("hints") or []
        print(f"  hints={', '.join(hints) if hints else '-'}")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Inspect clustering decision logs by link or event with similarity and outlier hints.")
    parser.add_argument("--db", default=str(OPENCLAW_HOME / "data" / "newsroom" / "news_pool.sqlite3"), help="SQLite db path.")
    parser.add_argument("--link-id", type=int, default=None, help="Filter to a single link ID.")
    parser.add_argument("--event-id", type=int, default=None, help="Filter to decisions touching a specific event ID.")
    parser.add_argument("--since-hours", type=int, default=None, help="Only decisions created within the last N hours.")
    parser.add_argument("--until-hours", type=int, default=None, help="Only decisions created at least N hours ago.")
    parser.add_argument("--domain", action="append", default=None, help="Filter by link domain. Repeatable or comma-separated.")
    parser.add_argument("--category", action="append", default=None, help="Filter by category. Repeatable or comma-separated.")
    parser.add_argument("--group-by", choices=["decision", "link", "event"], default="decision", help="Human-readable grouping mode.")
    parser.add_argument("--limit", type=int, default=20, help="Max matched decisions to inspect (default: 20).")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of human-readable text.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    args = parser.parse_args(argv)

    db_path = Path(args.db).expanduser()
    now_ts = int(time.time())

    since_hours = int(max(0, args.since_hours)) if args.since_hours is not None else None
    until_hours = int(max(0, args.until_hours)) if args.until_hours is not None else None
    if since_hours is not None and until_hours is not None and since_hours < until_hours:
        parser.error("--since-hours must be >= --until-hours when both are set")

    since_cutoff_ts = now_ts - (since_hours * 3600) if since_hours is not None else None
    until_cutoff_ts = now_ts - (until_hours * 3600) if until_hours is not None else None

    domains = _normalise_list(args.domain)
    domain_filters = {d.casefold() for d in domains}
    categories = _normalise_list(args.category)
    category_filters = {c.casefold() for c in categories}

    limit = int(max(1, args.limit))
    raw_limit = min(5000, max(200, limit * 8))

    warnings: list[str] = []
    decisions: list[dict[str, Any]] = []
    event_by_id: dict[int, dict[str, Any]] = {}

    try:
        with sqlite3.connect(str(db_path)) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()

            if not _table_exists(cur, "clustering_decisions"):
                warnings.append("clustering_decisions table is missing; no decision logs are available.")
            elif not _table_exists(cur, "links"):
                warnings.append("links table is missing; cannot resolve decision links.")
            else:
                where: list[str] = []
                params: list[Any] = []

                if args.link_id is not None:
                    where.append("d.link_id = ?")
                    params.append(int(args.link_id))
                if since_cutoff_ts is not None:
                    where.append("d.created_at_ts >= ?")
                    params.append(int(since_cutoff_ts))
                if until_cutoff_ts is not None:
                    where.append("d.created_at_ts <= ?")
                    params.append(int(until_cutoff_ts))
                if domain_filters:
                    placeholders = ", ".join(["?"] * len(domain_filters))
                    where.append(f"LOWER(COALESCE(l.domain, '')) IN ({placeholders})")
                    params.extend(sorted(domain_filters))

                where_sql = ("WHERE " + " AND ".join(where)) if where else ""

                rows = cur.execute(
                    f"""
                    SELECT
                      d.id AS decision_id,
                      d.link_id,
                      l.norm_url AS link_norm_url,
                      l.title AS link_title,
                      l.domain AS link_domain,
                      l.event_id AS link_event_id,
                      d.prompt_sha256,
                      d.model_name,
                      d.llm_started_at_ts,
                      d.llm_finished_at_ts,
                      d.llm_response_json,
                      d.validated_action,
                      d.validated_action_json,
                      d.enforced_action,
                      d.enforced_action_json,
                      d.error_type,
                      d.error_message,
                      d.created_at_ts
                    FROM clustering_decisions d
                    JOIN links l ON l.id = d.link_id
                    {where_sql}
                    ORDER BY d.created_at_ts DESC, d.id DESC
                    LIMIT ?
                    """,
                    (*params, raw_limit),
                ).fetchall()

                decision_ids = [int(r["decision_id"]) for r in rows]

                candidate_map: dict[int, list[dict[str, Any]]] = {}
                has_candidate_table = _table_exists(cur, "clustering_decision_candidates")
                if has_candidate_table and decision_ids:
                    chunk_size = 500
                    for i in range(0, len(decision_ids), chunk_size):
                        chunk = decision_ids[i : i + chunk_size]
                        placeholders = ",".join(["?"] * len(chunk))
                        c_rows = cur.execute(
                            f"""
                            SELECT decision_id, rank, event_id, score
                            FROM clustering_decision_candidates
                            WHERE decision_id IN ({placeholders})
                            ORDER BY decision_id ASC, rank ASC
                            """,
                            chunk,
                        ).fetchall()
                        for c in c_rows:
                            did = int(c["decision_id"])
                            candidate_map.setdefault(did, []).append(
                                {
                                    "rank": int(c["rank"]),
                                    "event_id": int(c["event_id"]),
                                    "score": float(c["score"]),
                                }
                            )
                elif not has_candidate_table:
                    warnings.append("clustering_decision_candidates table is missing; similarity details are limited.")

                raw_decisions: list[dict[str, Any]] = []
                all_event_ids: set[int] = set()

                for row in rows:
                    validated_obj = _maybe_json_text(row["validated_action_json"])
                    enforced_obj = _maybe_json_text(row["enforced_action_json"])

                    validated_action_name = _normalise_action_name(row["validated_action"])
                    if validated_action_name is None and isinstance(validated_obj, dict):
                        validated_action_name = _normalise_action_name(validated_obj.get("action"))

                    enforced_action_name = _normalise_action_name(row["enforced_action"])
                    if enforced_action_name is None and isinstance(enforced_obj, dict):
                        enforced_action_name = _normalise_action_name(enforced_obj.get("action"))

                    target_event_id = _extract_target_event_id(enforced_obj)
                    if target_event_id is None:
                        target_event_id = _extract_target_event_id(validated_obj)

                    link_event_id = _as_int(row["link_event_id"])
                    if link_event_id is not None and link_event_id > 0:
                        all_event_ids.add(link_event_id)
                    if target_event_id is not None and target_event_id > 0:
                        all_event_ids.add(target_event_id)

                    did = int(row["decision_id"])
                    for cand in candidate_map.get(did, []):
                        eid = _as_int(cand.get("event_id"))
                        if eid is not None and eid > 0:
                            all_event_ids.add(eid)

                    raw_decisions.append(
                        {
                            "decision_id": did,
                            "created_at_ts": _as_int(row["created_at_ts"]),
                            "link_id": int(row["link_id"]),
                            "link_norm_url": row["link_norm_url"],
                            "link_title": row["link_title"],
                            "link_domain": row["link_domain"],
                            "link_event_id": link_event_id,
                            "prompt_sha256": row["prompt_sha256"],
                            "model_name": row["model_name"],
                            "llm_started_at_ts": _as_int(row["llm_started_at_ts"]),
                            "llm_finished_at_ts": _as_int(row["llm_finished_at_ts"]),
                            "raw_llm_json": _maybe_json_text(row["llm_response_json"]),
                            "validated_action_name": validated_action_name,
                            "validated_action_obj": validated_obj,
                            "enforced_action_name": enforced_action_name,
                            "enforced_action_obj": enforced_obj,
                            "target_event_id": target_event_id,
                            "error_type": row["error_type"],
                            "error_message": row["error_message"],
                        }
                    )

                if all_event_ids and _table_exists(cur, "events"):
                    ids = sorted(all_event_ids)
                    chunk_size = 500
                    for i in range(0, len(ids), chunk_size):
                        chunk = ids[i : i + chunk_size]
                        placeholders = ",".join(["?"] * len(chunk))
                        e_rows = cur.execute(
                            f"""
                            SELECT id, category, jurisdiction, title, summary_en, status
                            FROM events
                            WHERE id IN ({placeholders})
                            """,
                            chunk,
                        ).fetchall()
                        for er in e_rows:
                            event_by_id[int(er["id"])] = dict(er)
                elif all_event_ids:
                    warnings.append("events table is missing; event category metadata is unavailable.")

                for raw in raw_decisions:
                    decision_id = int(raw["decision_id"])
                    candidates = list(candidate_map.get(decision_id, []))
                    for cand in candidates:
                        eid = _as_int(cand.get("event_id"))
                        if eid is None:
                            continue
                        ev = event_by_id.get(eid)
                        if ev and isinstance(ev.get("category"), str) and ev["category"].strip():
                            cand["event_category"] = ev["category"].strip()

                    target_event_id = _as_int(raw.get("target_event_id"))
                    similarity = _score_similarity(candidates, target_event_id=target_event_id)

                    action_obj = raw.get("enforced_action_obj")
                    if not isinstance(action_obj, dict):
                        action_obj = raw.get("validated_action_obj")

                    action_name = raw.get("enforced_action_name") or raw.get("validated_action_name")
                    hints = _build_hints(
                        action_name=action_name if isinstance(action_name, str) else None,
                        error_type=raw.get("error_type") if isinstance(raw.get("error_type"), str) else None,
                        similarity=similarity,
                    )

                    linked_event_category = _event_category(event_by_id, _as_int(raw.get("link_event_id")))
                    target_event_category = _event_category(event_by_id, target_event_id)
                    category, action_category = _resolve_category(
                        action_obj=action_obj,
                        linked_event_category=linked_event_category,
                        target_event_category=target_event_category,
                    )

                    event_ids_touched = _event_ids_touched(
                        link_event_id=_as_int(raw.get("link_event_id")),
                        target_event_id=target_event_id,
                        candidates=candidates,
                    )

                    decision = {
                        "decision_id": decision_id,
                        "created_at_ts": raw.get("created_at_ts"),
                        "created_at": _iso(_as_int(raw.get("created_at_ts"))),
                        "link": {
                            "link_id": _as_int(raw.get("link_id")),
                            "norm_url": raw.get("link_norm_url"),
                            "title": raw.get("link_title"),
                            "domain": raw.get("link_domain"),
                            "event_id": _as_int(raw.get("link_event_id")),
                        },
                        "prompt_sha256": raw.get("prompt_sha256"),
                        "model_name": raw.get("model_name"),
                        "llm_started_at": _iso(_as_int(raw.get("llm_started_at_ts"))),
                        "llm_finished_at": _iso(_as_int(raw.get("llm_finished_at_ts"))),
                        "raw_llm_json": raw.get("raw_llm_json"),
                        "validated_action": {
                            "action": raw.get("validated_action_name"),
                            "obj": raw.get("validated_action_obj"),
                        },
                        "enforced_action": {
                            "action": raw.get("enforced_action_name"),
                            "obj": raw.get("enforced_action_obj"),
                        },
                        "target_event_id": target_event_id,
                        "target_event_category": target_event_category,
                        "linked_event_category": linked_event_category,
                        "action_category": action_category,
                        "category": category,
                        "event_ids_touched": event_ids_touched,
                        "candidates": candidates,
                        "similarity": similarity,
                        "hints": hints,
                        "error": {
                            "type": raw.get("error_type"),
                            "message": raw.get("error_message"),
                        },
                    }

                    if args.event_id is not None and int(args.event_id) not in event_ids_touched:
                        continue

                    if not _decision_matches_category(
                        decision=decision,
                        category_filters=category_filters,
                        event_by_id=event_by_id,
                    ):
                        continue

                    decisions.append(decision)
                    if len(decisions) >= limit:
                        break

        if not decisions:
            warnings.append("No clustering decision logs found for the selected filters.")
    except sqlite3.Error as e:
        out = {
            "ok": False,
            "generated_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
            "db": str(db_path),
            "error": f"SQLite error: {e}",
        }
        if args.json:
            print(json.dumps(out, ensure_ascii=False, indent=2 if args.pretty else None))
        else:
            print(f"Decision Log Inspector\nDB: {db_path}\nError: {out['error']}")
        return 1

    unique_links = {d.get("link", {}).get("link_id") for d in decisions if d.get("link", {}).get("link_id") is not None}
    unique_events = {
        eid
        for d in decisions
        for eid in (d.get("event_ids_touched") or [])
        if isinstance(eid, int) and eid > 0
    }

    groups: list[dict[str, Any]]
    if args.group_by == "link":
        groups = _aggregate_by_link(decisions)
    elif args.group_by == "event":
        groups = _aggregate_by_event(decisions, event_by_id=event_by_id)
    else:
        groups = []

    out = {
        "ok": True,
        "generated_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
        "db": str(db_path),
        "filters": {
            "group_by": args.group_by,
            "link_id": args.link_id,
            "event_id": args.event_id,
            "since_hours": since_hours,
            "until_hours": until_hours,
            "domains": domains,
            "categories": categories,
            "limit": limit,
            "raw_limit": raw_limit,
        },
        "summary": {
            "decision_count": len(decisions),
            "unique_links": len(unique_links),
            "unique_events": len(unique_events),
        },
        "warnings": _unique_preserve(warnings),
        "decisions": decisions,
        "group_by": args.group_by,
        "groups": groups,
    }

    if args.json:
        if args.pretty:
            print(json.dumps(out, ensure_ascii=False, indent=2))
        else:
            print(json.dumps(out, ensure_ascii=False, separators=(",", ":")))
    else:
        _print_human(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
