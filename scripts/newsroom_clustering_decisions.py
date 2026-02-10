#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

OPENCLAW_HOME = Path(__file__).resolve().parents[1]
os.environ.setdefault("OPENCLAW_HOME", str(OPENCLAW_HOME))
sys.path.insert(0, str(OPENCLAW_HOME))

from newsroom.news_pool_db import NewsPoolDB  # noqa: E402


def _iso(ts: int | None) -> str | None:
    if not ts:
        return None
    return datetime.fromtimestamp(int(ts), tz=UTC).isoformat(timespec="seconds")


def _maybe_json_text(s: Any) -> Any:
    if not isinstance(s, str):
        return None
    t = s.strip()
    if not t:
        return None
    if t[0] not in "[{":
        return t
    try:
        return json.loads(t)
    except Exception:
        return t


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Query clustering decision audit logs from the newsroom SQLite pool.")
    parser.add_argument("--db", default=str(OPENCLAW_HOME / "data" / "newsroom" / "news_pool.sqlite3"), help="SQLite db path.")
    parser.add_argument("--link-id", type=int, default=None, help="Filter to a single link ID.")
    parser.add_argument("--since-hours", type=int, default=None, help="Only decisions created within the last N hours.")
    parser.add_argument("--limit", type=int, default=20, help="Max decisions to return (default: 20).")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    args = parser.parse_args(argv)

    db_path = Path(args.db).expanduser()
    now_ts = int(time.time())
    cutoff_ts = None
    if args.since_hours is not None:
        cutoff_ts = now_ts - int(max(0, args.since_hours)) * 3600

    where: list[str] = []
    params: list[Any] = []
    if args.link_id is not None:
        where.append("d.link_id = ?")
        params.append(int(args.link_id))
    if cutoff_ts is not None:
        where.append("d.created_at_ts >= ?")
        params.append(int(cutoff_ts))
    where_sql = ("WHERE " + " AND ".join(where)) if where else ""

    out_decisions: list[dict[str, Any]] = []

    with NewsPoolDB(path=db_path) as db:
        cur = db._conn.cursor()  # intentional: internal DB utility for debug queries
        rows = cur.execute(
            f"""
            SELECT
              d.id AS decision_id,
              d.link_id,
              l.norm_url AS link_norm_url,
              l.title AS link_title,
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
            (*params, int(max(1, args.limit))),
        ).fetchall()

        for r in rows:
            decision_id = int(r["decision_id"])
            cand_rows = cur.execute(
                """
                SELECT rank, event_id, score
                FROM clustering_decision_candidates
                WHERE decision_id = ?
                ORDER BY rank ASC
                """,
                (decision_id,),
            ).fetchall()
            candidates = [{"rank": int(c["rank"]), "event_id": int(c["event_id"]), "score": float(c["score"])} for c in cand_rows]

            out_decisions.append(
                {
                    "decision_id": decision_id,
                    "created_at": _iso(int(r["created_at_ts"]) if r["created_at_ts"] is not None else None),
                    "link": {
                        "link_id": int(r["link_id"]),
                        "norm_url": r["link_norm_url"],
                        "title": r["link_title"],
                    },
                    "prompt_sha256": r["prompt_sha256"],
                    "model_name": r["model_name"],
                    "llm_started_at": _iso(int(r["llm_started_at_ts"]) if r["llm_started_at_ts"] is not None else None),
                    "llm_finished_at": _iso(int(r["llm_finished_at_ts"]) if r["llm_finished_at_ts"] is not None else None),
                    "candidates": candidates,
                    "raw_llm_json": _maybe_json_text(r["llm_response_json"]),
                    "validated_action": {
                        "action": r["validated_action"],
                        "obj": _maybe_json_text(r["validated_action_json"]),
                    },
                    "enforced_action": {
                        "action": r["enforced_action"],
                        "obj": _maybe_json_text(r["enforced_action_json"]),
                    },
                    "error": {
                        "type": r["error_type"],
                        "message": r["error_message"],
                    },
                }
            )

    out = {
        "ok": True,
        "generated_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
        "db": str(db_path),
        "filters": {"link_id": args.link_id, "since_hours": args.since_hours, "limit": int(max(1, args.limit))},
        "decisions": out_decisions,
    }

    if args.pretty:
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

