#!/usr/bin/env python3
from __future__ import annotations

"""Offline integrity fixer for the SQLite news pool database.

This script is intentionally *not* part of the clustering/runtime pipeline. It is
an offline maintenance utility that recomputes derived event fields from the
source-of-truth `links` table and optionally prunes dead events.

What it fixes (derived from links):
  - events.link_count         = COUNT(links) per event_id
  - events.best_published_ts  = MAX(links.published_at_ts) per event_id
  - events.expires_at_ts      = MAX(links.last_seen_ts) + TTL per event_id

It can also:
  - Prune non-posted, 0-link events older than a configurable age
  - Normalise drifted categories back to the canonical category set

The default mode is dry-run. Use --apply to write changes.
"""

import argparse
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

OPENCLAW_HOME = Path(__file__).resolve().parents[1]
os.environ.setdefault("OPENCLAW_HOME", str(OPENCLAW_HOME))
sys.path.insert(0, str(OPENCLAW_HOME))

from newsroom.news_pool_db import NewsPoolDB, _DEFAULT_TTL_SECONDS  # noqa: E402


CANONICAL_CATEGORIES = {
    "UK Parliament / Politics",
    "UK News",
    "Politics",
    "AI",
    "Sports",
    "Hong Kong Entertainment",
    "Entertainment",
    "US Stocks",
    "Crypto",
    "Precious Metals",
    "Global News",
    "Hong Kong News",
}

_CANONICAL_BY_LOWER = {c.lower(): c for c in CANONICAL_CATEGORIES}


@dataclass(frozen=True)
class EventStatFix:
    event_id: int
    old_link_count: int
    new_link_count: int
    old_best_published_ts: int | None
    new_best_published_ts: int | None
    old_expires_at_ts: int | None
    new_expires_at_ts: int | None


@dataclass(frozen=True)
class CategoryFix:
    event_id: int
    old_category: str | None
    new_category: str


@dataclass(frozen=True)
class FixPlan:
    stat_fixes: list[EventStatFix]
    category_fixes: list[CategoryFix]
    prune_event_ids: list[int]
    prune_cutoff_ts: int
    ttl_seconds: int


def _normalise_category(raw: str | None) -> str | None:
    """Map a drifted category label to the canonical category set.

    If the input cannot be classified confidently, this falls back to
    "Global News" rather than leaving a non-canonical value behind.
    """
    if raw is None:
        return "Global News"
    s = str(raw).strip()
    if not s:
        return "Global News"

    direct = _CANONICAL_BY_LOWER.get(s.lower())
    if direct:
        return direct

    low = s.lower()
    # Normalise separators and remove noise so we can do robust contains checks.
    norm = re.sub(r"[^a-z0-9]+", " ", low).strip()

    # UK: parliamentary / Westminster.
    if any(k in norm for k in ("parliament", "westminster", "downing street", "downing")):
        return "UK Parliament / Politics"
    if "uk" in norm and "politic" in norm:
        return "UK Parliament / Politics"
    if "uk" in norm and any(k in norm for k in ("news", "britain", "england", "scotland", "wales", "london")):
        return "UK News"

    # Hong Kong.
    if ("hong" in norm and "kong" in norm) or " hk " in f" {norm} ":
        if any(k in norm for k in ("entertain", "showbiz", "celebrity")):
            return "Hong Kong Entertainment"
        return "Hong Kong News"

    # Topics.
    if any(k in norm for k in ("sport", "football", "soccer", "nba", "nfl", "fifa")):
        return "Sports"
    if any(k in norm for k in ("ai", "artificial intelligence", "machine learning", "ml")):
        return "AI"
    if any(k in norm for k in ("entertain", "showbiz", "film", "movie", "tv", "music")):
        return "Entertainment"
    if any(k in norm for k in ("crypto", "bitcoin", "ethereum", "token")):
        return "Crypto"
    if any(k in norm for k in ("precious", "metal", "gold", "silver", "platinum", "palladium")):
        return "Precious Metals"
    if any(k in norm for k in ("stock", "stocks", "share", "shares", "equity", "equities", "finance", "market")):
        return "US Stocks"
    if "politic" in norm:
        return "Politics"
    if any(k in norm for k in ("global", "world", "international")):
        return "Global News"

    return "Global News"


def _load_link_aggregates(conn) -> dict[int, dict[str, int | None]]:
    """Return per-event aggregates computed from links."""
    stats: dict[int, dict[str, int | None]] = {}
    rows = conn.execute(
        """
        SELECT
          event_id,
          COUNT(1) AS link_count,
          MAX(published_at_ts) AS best_published_ts,
          MAX(last_seen_ts) AS max_last_seen_ts
        FROM links
        WHERE event_id IS NOT NULL
        GROUP BY event_id
        """
    ).fetchall()
    for r in rows:
        eid = int(r["event_id"])
        stats[eid] = {
            "link_count": int(r["link_count"]),
            "best_published_ts": (int(r["best_published_ts"]) if r["best_published_ts"] is not None else None),
            "max_last_seen_ts": (int(r["max_last_seen_ts"]) if r["max_last_seen_ts"] is not None else None),
        }
    return stats


def _load_parent_ids_with_children(conn) -> set[int]:
    rows = conn.execute(
        "SELECT DISTINCT parent_event_id AS pid FROM events WHERE parent_event_id IS NOT NULL"
    ).fetchall()
    out: set[int] = set()
    for r in rows:
        if r["pid"] is None:
            continue
        out.add(int(r["pid"]))
    return out


def build_plan(
    *,
    conn,
    now_ts: int,
    prune_zero_link_hours: int,
    ttl_seconds: int,
    normalise_categories: bool,
) -> FixPlan:
    link_stats = _load_link_aggregates(conn)
    parent_ids_with_children = _load_parent_ids_with_children(conn)

    stat_fixes: list[EventStatFix] = []
    category_fixes: list[CategoryFix] = []
    prune_event_ids: list[int] = []

    prune_cutoff_ts = int(now_ts) - int(max(0, prune_zero_link_hours)) * 3600

    rows = conn.execute(
        """
        SELECT
          id,
          link_count,
          best_published_ts,
          expires_at_ts,
          category,
          status,
          created_at_ts
        FROM events
        ORDER BY id ASC
        """
    ).fetchall()

    for r in rows:
        eid = int(r["id"])
        old_link_count = int(r["link_count"])
        old_best_pub = int(r["best_published_ts"]) if r["best_published_ts"] is not None else None
        old_expires = int(r["expires_at_ts"]) if r["expires_at_ts"] is not None else None
        old_cat = str(r["category"]) if r["category"] is not None else None
        status = str(r["status"] or "")
        created_at_ts = int(r["created_at_ts"])

        s = link_stats.get(eid)
        if s is None:
            new_link_count = 0
            new_best_pub = None
            new_expires = None
        else:
            new_link_count = int(s["link_count"] or 0)
            new_best_pub = int(s["best_published_ts"]) if s.get("best_published_ts") is not None else None
            mx_last_seen = int(s["max_last_seen_ts"]) if s.get("max_last_seen_ts") is not None else None
            new_expires = (int(mx_last_seen) + int(ttl_seconds)) if mx_last_seen is not None else None

        if (
            old_link_count != new_link_count
            or old_best_pub != new_best_pub
            or old_expires != new_expires
        ):
            stat_fixes.append(
                EventStatFix(
                    event_id=eid,
                    old_link_count=old_link_count,
                    new_link_count=new_link_count,
                    old_best_published_ts=old_best_pub,
                    new_best_published_ts=new_best_pub,
                    old_expires_at_ts=old_expires,
                    new_expires_at_ts=new_expires,
                )
            )

        if normalise_categories:
            new_cat = _normalise_category(old_cat)
            if new_cat is not None and new_cat != old_cat:
                category_fixes.append(CategoryFix(event_id=eid, old_category=old_cat, new_category=new_cat))

        # Prune: non-posted + 0-link + old + safe (no children).
        if (
            status != "posted"
            and new_link_count == 0
            and created_at_ts < prune_cutoff_ts
            and eid not in parent_ids_with_children
        ):
            prune_event_ids.append(eid)

    return FixPlan(
        stat_fixes=stat_fixes,
        category_fixes=category_fixes,
        prune_event_ids=prune_event_ids,
        prune_cutoff_ts=prune_cutoff_ts,
        ttl_seconds=int(ttl_seconds),
    )


def apply_plan(*, conn, plan: FixPlan, now_ts: int) -> dict[str, Any]:
    """Apply a FixPlan in a single transaction.

    Returns a summary dict (counts).
    """
    cur = conn.cursor()
    cur.execute("BEGIN")
    try:
        if plan.stat_fixes:
            cur.executemany(
                """
                UPDATE events
                SET link_count = ?,
                    best_published_ts = ?,
                    expires_at_ts = ?,
                    updated_at_ts = ?
                WHERE id = ?
                """,
                [
                    (
                        fx.new_link_count,
                        fx.new_best_published_ts,
                        fx.new_expires_at_ts,
                        int(now_ts),
                        fx.event_id,
                    )
                    for fx in plan.stat_fixes
                ],
            )

        if plan.category_fixes:
            cur.executemany(
                "UPDATE events SET category = ?, updated_at_ts = ? WHERE id = ?",
                [(fx.new_category, int(now_ts), fx.event_id) for fx in plan.category_fixes],
            )

        pruned = 0
        if plan.prune_event_ids:
            placeholders = ",".join("?" for _ in plan.prune_event_ids)
            # Safety: do not delete events that are parents (FK would block anyway),
            # and do not delete posted events.
            cur.execute(
                f"""
                DELETE FROM events
                WHERE id IN ({placeholders})
                  AND status != 'posted'
                  AND NOT EXISTS (SELECT 1 FROM events c WHERE c.parent_event_id = events.id)
                  AND NOT EXISTS (SELECT 1 FROM links l WHERE l.event_id = events.id)
                """,
                plan.prune_event_ids,
            )
            pruned = int(cur.rowcount or 0)

        cur.execute("COMMIT")
    except Exception:
        cur.execute("ROLLBACK")
        raise

    return {
        "events_updated": len(plan.stat_fixes),
        "categories_updated": len(plan.category_fixes),
        "events_pruned": pruned,
    }


def _fmt_ts(ts: int | None) -> str:
    return str(ts) if ts is not None else "NULL"


def _print_plan(plan: FixPlan, *, db_path: Path, apply: bool, normalise_categories: bool, prune_zero_link_hours: int) -> None:
    mode = "APPLY" if apply else "DRY-RUN"
    print(f"[{mode}] fix_pool_integrity")
    print(f"db={db_path}")
    print(f"ttl_seconds={plan.ttl_seconds}")
    print(f"prune_zero_link_hours={int(prune_zero_link_hours)} cutoff_ts={plan.prune_cutoff_ts}")
    print(f"normalise_categories={bool(normalise_categories)}")
    print("")

    if not plan.stat_fixes and not plan.category_fixes and not plan.prune_event_ids:
        print("No changes required.")
        return

    if plan.stat_fixes:
        n_lc = sum(1 for fx in plan.stat_fixes if fx.old_link_count != fx.new_link_count)
        n_bp = sum(1 for fx in plan.stat_fixes if fx.old_best_published_ts != fx.new_best_published_ts)
        n_ex = sum(1 for fx in plan.stat_fixes if fx.old_expires_at_ts != fx.new_expires_at_ts)
        print(f"Event stat fixes: {len(plan.stat_fixes)}")
        print(f"- link_count changes: {n_lc}")
        print(f"- best_published_ts changes: {n_bp}")
        print(f"- expires_at_ts changes: {n_ex}")

        sample = plan.stat_fixes[:20]
        if sample:
            print("")
            print("Sample (up to 20):")
            for fx in sample:
                print(
                    f"- event_id={fx.event_id} "
                    f"link_count {fx.old_link_count}->{fx.new_link_count} "
                    f"best_published_ts {_fmt_ts(fx.old_best_published_ts)}->{_fmt_ts(fx.new_best_published_ts)} "
                    f"expires_at_ts {_fmt_ts(fx.old_expires_at_ts)}->{_fmt_ts(fx.new_expires_at_ts)}"
                )
        print("")

    if plan.category_fixes:
        print(f"Category normalisation fixes: {len(plan.category_fixes)}")
        sample = plan.category_fixes[:20]
        if sample:
            print("")
            print("Sample (up to 20):")
            for fx in sample:
                print(f"- event_id={fx.event_id} category {fx.old_category!r}->{fx.new_category!r}")
        print("")

    if plan.prune_event_ids:
        print(f"Prune candidates (non-posted, 0-link, old, leaf): {len(plan.prune_event_ids)}")
        sample = plan.prune_event_ids[:20]
        if sample:
            print("")
            print("Sample (up to 20):")
            for eid in sample:
                print(f"- event_id={eid}")
        print("")

    if not apply:
        print("Dry-run only. Re-run with --apply to write changes.")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fix SQLite news pool integrity by recomputing event counters/timestamps from links, "
            "optionally pruning dead events and normalising drifted categories."
        )
    )
    parser.add_argument(
        "--db",
        default=str(OPENCLAW_HOME / "data" / "newsroom" / "news_pool.sqlite3"),
        help="SQLite db path (default: %(default)s).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write changes to the database (default: dry-run).",
    )
    parser.add_argument(
        "--prune-zero-link-hours",
        type=int,
        default=168,
        help="Prune non-posted, 0-link leaf events older than this many hours (default: 168).",
    )
    parser.add_argument(
        "--normalise-categories",
        action="store_true",
        help="Normalise drifted categories back to the canonical category set.",
    )
    args = parser.parse_args(argv)

    db_path = Path(args.db).expanduser()
    if not db_path.exists():
        print(f"ERROR: db does not exist: {db_path}", file=sys.stderr)
        return 2

    now_ts = int(time.time())
    ttl_seconds = int(_DEFAULT_TTL_SECONDS)

    with NewsPoolDB(path=db_path) as db:
        plan = build_plan(
            conn=db._conn,
            now_ts=now_ts,
            prune_zero_link_hours=int(args.prune_zero_link_hours),
            ttl_seconds=ttl_seconds,
            normalise_categories=bool(args.normalise_categories),
        )
        _print_plan(
            plan,
            db_path=db_path,
            apply=bool(args.apply),
            normalise_categories=bool(args.normalise_categories),
            prune_zero_link_hours=int(args.prune_zero_link_hours),
        )

        if not args.apply:
            return 0

        summary = apply_plan(conn=db._conn, plan=plan, now_ts=now_ts)
        print("")
        print("Applied.")
        print(f"events_updated={summary['events_updated']}")
        print(f"categories_updated={summary['categories_updated']}")
        print(f"events_pruned={summary['events_pruned']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
