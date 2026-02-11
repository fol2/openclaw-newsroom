#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sqlite3
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

OPENCLAW_HOME = Path(__file__).resolve().parents[1]
os.environ.setdefault("OPENCLAW_HOME", str(OPENCLAW_HOME))
sys.path.insert(0, str(OPENCLAW_HOME))

from newsroom.eval_dataset import (  # noqa: E402
    label_correctness,
    label_flags,
    label_language,
    label_target_from_action,
    parse_replay_enforced_action,
    prediction_target_from_action,
    top_candidate_score,
)


def _iso(ts: Any) -> str | None:
    if ts is None:
        return None
    try:
        return datetime.fromtimestamp(int(ts), tz=UTC).isoformat(timespec="seconds")
    except Exception:
        return None


def _load_json(raw: Any) -> Any:
    if raw is None:
        return None
    if isinstance(raw, (dict, list)):
        return raw
    if not isinstance(raw, str):
        return None
    text = raw.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        return None


def _as_dict(raw: Any) -> dict[str, Any]:
    return raw if isinstance(raw, dict) else {}


def _parse_lang_min(spec: str) -> dict[str, int]:
    out: dict[str, int] = {"en": 0, "zh": 0, "mixed": 0}
    text = str(spec or "").strip()
    if not text:
        return out

    for token in text.split(","):
        t = token.strip()
        if not t:
            continue
        if "=" not in t:
            raise ValueError(f"Invalid --lang-min token: {t!r}")
        k, v = t.split("=", 1)
        key = k.strip().lower()
        if key not in out:
            raise ValueError(f"Unknown language in --lang-min: {key!r}")
        try:
            n = int(v.strip())
        except ValueError as exc:
            raise ValueError(f"Invalid integer in --lang-min token: {t!r}") from exc
        if n < 0:
            raise ValueError(f"--lang-min values must be >= 0: {t!r}")
        out[key] = n
    return out


def _fetch_rows(conn: sqlite3.Connection, *, cutoff_ts: int | None) -> list[sqlite3.Row]:
    where_sql = ""
    params: list[Any] = []
    if cutoff_ts is not None:
        where_sql = "WHERE d.created_at_ts >= ?"
        params.append(int(cutoff_ts))

    sql = f"""
    SELECT
      d.id AS decision_id,
      d.link_id,
      d.prompt_sha256,
      d.model_name,
      d.created_at_ts,
      d.llm_response_json,
      d.validated_action_json,
      d.enforced_action_json,
      l.norm_url,
      l.url,
      l.domain,
      l.title,
      l.description,
      l.lang_hint,
      l.published_at_ts,
      l.event_id AS final_event_id
    FROM clustering_decisions d
    JOIN links l ON l.id = d.link_id
    {where_sql}
    ORDER BY d.created_at_ts DESC, d.id DESC
    """
    return conn.execute(sql, params).fetchall()


def _fetch_candidates(conn: sqlite3.Connection, *, decision_id: int) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT
          c.rank,
          c.event_id,
          c.score,
          e.summary_en,
          e.category,
          e.jurisdiction,
          e.status,
          e.parent_event_id
        FROM clustering_decision_candidates c
        LEFT JOIN events e ON e.id = c.event_id
        WHERE c.decision_id = ?
        ORDER BY c.rank ASC
        """,
        (int(decision_id),),
    ).fetchall()

    out: list[dict[str, Any]] = []
    for r in rows:
        event_id = r["event_id"]
        if not isinstance(event_id, int):
            continue
        out.append(
            {
                "rank": int(r["rank"]) if isinstance(r["rank"], int) else None,
                "id": int(event_id),
                "score": float(r["score"]) if r["score"] is not None else None,
                "summary_en": r["summary_en"],
                "category": r["category"],
                "jurisdiction": r["jurisdiction"],
                "status": r["status"],
                "parent_event_id": int(r["parent_event_id"]) if isinstance(r["parent_event_id"], int) else None,
            }
        )
    return out


def _build_record(
    row: sqlite3.Row,
    *,
    candidates: list[dict[str, Any]],
    low_score_threshold: float,
) -> dict[str, Any] | None:
    llm_response = _load_json(row["llm_response_json"])
    if not isinstance(llm_response, dict):
        return None

    validated_action = _as_dict(_load_json(row["validated_action_json"]))
    observed_enforced_action = _as_dict(_load_json(row["enforced_action_json"]))

    link = {
        "id": int(row["link_id"]),
        "url": row["url"],
        "norm_url": row["norm_url"],
        "domain": row["domain"],
        "title": row["title"],
        "description": row["description"],
        "lang_hint": row["lang_hint"],
        "published_at": _iso(row["published_at_ts"]),
        "final_event_id": int(row["final_event_id"]) if isinstance(row["final_event_id"], int) else None,
    }

    replay_link = {
        "id": link["id"],
        "url": link["url"],
        "norm_url": link["norm_url"],
        "title": link["title"],
        "description": link["description"],
        "lang_hint": link["lang_hint"],
    }

    enforced_action = parse_replay_enforced_action(
        link=replay_link,
        candidate_events=candidates,
        raw_llm_response=llm_response,
    )
    if not isinstance(enforced_action, dict):
        return None
    action = str(enforced_action.get("action") or "").strip().lower()

    language = label_language(
        title=link.get("title"),
        description=link.get("description"),
        existing_hint=link.get("lang_hint"),
    )

    top_score = top_candidate_score(candidates)
    predicted_target = prediction_target_from_action(enforced_action)
    labelled_target = label_target_from_action(
        enforced_action,
        top_candidate_score=top_score,
        low_score_threshold=float(low_score_threshold),
    )
    correctness = label_correctness(predicted_target=predicted_target, labelled_target=labelled_target)
    flags = label_flags(
        enforced_action=enforced_action,
        top_candidate_score=top_score,
        low_score_threshold=float(low_score_threshold),
    )

    return {
        "schema_version": "clustering_eval_sample_v1",
        "sample_id": "",
        "source": {
            "decision_id": int(row["decision_id"]),
            "decision_created_at": _iso(row["created_at_ts"]),
            "prompt_sha256": row["prompt_sha256"],
            "model_name": row["model_name"],
            "observed_enforced_action": observed_enforced_action,
            "observed_action_drift": (
                str(observed_enforced_action.get("action") or "").strip().lower() != action
                if isinstance(observed_enforced_action, dict)
                else True
            ),
        },
        "link": link,
        "prediction": {
            "action": action,
            "event_id_or_new_event": predicted_target,
            "top_candidate_score": top_score,
        },
        "labels": {
            "event_id_or_new_event": labelled_target,
            "language": language,
            "correctness": correctness,
            "flags": flags,
        },
        "replay": {
            "link": {
                "id": link["id"],
                "url": link["url"],
                "norm_url": link["norm_url"],
                "title": link["title"],
                "description": link["description"],
                "lang_hint": link["lang_hint"],
            },
            "candidate_events": candidates,
            "raw_llm_response": llm_response,
            "validated_action": validated_action,
            "enforced_action": enforced_action,
            "observed_enforced_action": observed_enforced_action,
        },
    }


def _sample_records(
    records: list[dict[str, Any]],
    *,
    sample_size: int,
    seed: int,
    lang_min: dict[str, int],
) -> list[dict[str, Any]]:
    if sample_size <= 0:
        raise ValueError("sample_size must be > 0")
    if len(records) < sample_size:
        raise ValueError(f"Not enough eligible records ({len(records)}) for sample_size={sample_size}")

    by_lang: dict[str, list[int]] = {"en": [], "zh": [], "mixed": []}
    for idx, rec in enumerate(records):
        lang = str(rec.get("labels", {}).get("language") or "").strip().lower()
        if lang in by_lang:
            by_lang[lang].append(idx)

    required_total = 0
    for lang, required in lang_min.items():
        required_total += int(required)
        available = len(by_lang.get(lang, []))
        if available < int(required):
            raise ValueError(f"Not enough '{lang}' records for quota: need {required}, have {available}")
    if required_total > sample_size:
        raise ValueError(f"Sum of language minima ({required_total}) exceeds sample_size={sample_size}")

    rng = random.Random(int(seed))
    selected: list[int] = []
    for lang in ("en", "zh", "mixed"):
        required = int(lang_min.get(lang, 0))
        if required <= 0:
            continue
        selected.extend(rng.sample(by_lang[lang], required))

    selected_set = set(selected)
    remaining = sample_size - len(selected)
    if remaining > 0:
        pool = [idx for idx in range(len(records)) if idx not in selected_set]
        selected.extend(rng.sample(pool, remaining))

    sampled = [records[idx] for idx in selected]
    sampled.sort(key=lambda r: int(r["source"]["decision_id"]), reverse=True)

    for i, rec in enumerate(sampled, start=1):
        rec["sample_id"] = f"clustering_eval_v1_{i:04d}"

    return sampled


def _stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    lang_counts = Counter(str(r.get("labels", {}).get("language") or "") for r in records)
    correctness_counts = Counter(str(r.get("labels", {}).get("correctness") or "") for r in records)
    action_counts = Counter(str(r.get("prediction", {}).get("action") or "") for r in records)

    flag_counts: Counter[str] = Counter()
    for r in records:
        flags = r.get("labels", {}).get("flags")
        if isinstance(flags, list):
            for f in flags:
                flag_counts[str(f)] += 1

    return {
        "rows": len(records),
        "languages": dict(sorted(lang_counts.items())),
        "correctness": dict(sorted(correctness_counts.items())),
        "actions": dict(sorted(action_counts.items())),
        "flags": dict(sorted(flag_counts.items())),
    }


def main(argv: list[str]) -> int:
    # Keep builder output concise even when older rows miss optional confidence.
    logging.getLogger("newsroom.event_manager").setLevel(logging.ERROR)

    parser = argparse.ArgumentParser(description="Build a labelled clustering evaluation dataset from decision audit logs.")
    parser.add_argument("--db", default=str(Path.home() / ".openclaw" / "data" / "newsroom" / "news_pool.sqlite3"), help="SQLite DB path.")
    parser.add_argument(
        "--out",
        default=str(OPENCLAW_HOME / "newsroom" / "evals" / "clustering_eval_dataset_v1.jsonl"),
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--meta-out",
        default=str(OPENCLAW_HOME / "newsroom" / "evals" / "clustering_eval_dataset_v1.meta.json"),
        help="Output metadata JSON path.",
    )
    parser.add_argument("--sample-size", type=int, default=240, help="Number of rows to sample (200-500 recommended).")
    parser.add_argument("--since-hours", type=int, default=24 * 45, help="Only include decisions from the last N hours.")
    parser.add_argument("--seed", type=int, default=25, help="Deterministic random seed.")
    parser.add_argument(
        "--lang-min",
        default="en=150,zh=40,mixed=15",
        help="Minimum per-language counts (comma-separated, e.g. en=150,zh=40,mixed=15).",
    )
    parser.add_argument(
        "--low-score-threshold",
        type=float,
        default=0.22,
        help="Assignments with top candidate retrieval score below this are labelled new_event.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting existing output files.")
    args = parser.parse_args(argv)

    out_path = Path(args.out).expanduser()
    meta_path = Path(args.meta_out).expanduser()
    db_path = Path(args.db).expanduser()

    if not db_path.exists():
        raise SystemExit(f"DB not found: {db_path}")
    if out_path.exists() and not args.overwrite:
        raise SystemExit(f"Output exists: {out_path} (use --overwrite)")
    if meta_path.exists() and not args.overwrite:
        raise SystemExit(f"Metadata output exists: {meta_path} (use --overwrite)")

    lang_min = _parse_lang_min(args.lang_min)
    cutoff_ts: int | None = None
    if int(args.since_hours) > 0:
        cutoff_ts = int(datetime.now(tz=UTC).timestamp()) - int(args.since_hours) * 3600

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = _fetch_rows(conn, cutoff_ts=cutoff_ts)
        built: list[dict[str, Any]] = []
        for row in rows:
            candidates = _fetch_candidates(conn, decision_id=int(row["decision_id"]))
            rec = _build_record(row, candidates=candidates, low_score_threshold=float(args.low_score_threshold))
            if rec is not None:
                built.append(rec)
    finally:
        conn.close()

    sampled = _sample_records(
        built,
        sample_size=int(args.sample_size),
        seed=int(args.seed),
        lang_min=lang_min,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for rec in sampled:
            fh.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")

    summary = _stats(sampled)
    meta = {
        "schema_version": "clustering_eval_dataset_meta_v1",
        "dataset_schema_version": "clustering_eval_sample_v1",
        "generated_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
        "source_db": str(db_path),
        "sample_size": int(args.sample_size),
        "seed": int(args.seed),
        "since_hours": int(args.since_hours),
        "low_score_threshold": float(args.low_score_threshold),
        "lang_min": lang_min,
        "summary": summary,
    }

    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with meta_path.open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)
        fh.write("\n")

    print(json.dumps({"ok": True, "out": str(out_path), "meta": str(meta_path), "summary": summary}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
