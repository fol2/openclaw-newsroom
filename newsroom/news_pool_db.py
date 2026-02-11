from __future__ import annotations

import json
import logging
import sqlite3
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 6

_RESERVATION_SECONDS = 600  # 10 minutes

_DEFAULT_TTL_SECONDS = 48 * 3600  # 48 hours


def _utc_now_ts() -> int:
    return int(time.time())


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _parse_page_age_ts(page_age: str | None) -> int | None:
    """Parse ISO 8601 page_age string to Unix timestamp."""
    if not isinstance(page_age, str) or not page_age.strip():
        return None
    s = page_age.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return int(dt.timestamp())


def _clean_alias_text(v: Any, *, max_len: int = 120) -> str | None:
    if not isinstance(v, (str, int, float)):
        return None
    s = " ".join(str(v).strip().split())
    if not s:
        return None
    if len(s) > max_len:
        s = s[:max_len]
    return s


def _normalise_entity_aliases(raw: Any, *, max_entities: int = 12, max_aliases_per_entity: int = 8) -> list[dict[str, Any]]:
    """Normalise entity aliases into a stable list-of-dicts structure."""
    parsed = raw
    if isinstance(parsed, str):
        s = parsed.strip()
        if not s:
            return []
        try:
            parsed = json.loads(s)
        except Exception:
            parsed = [s]

    if isinstance(parsed, dict):
        parsed = parsed.get("entity_aliases", parsed)
    if not isinstance(parsed, list):
        return []

    out: list[dict[str, Any]] = []
    by_label: dict[str, int] = {}

    def _parse_item(item: Any) -> tuple[str | None, list[str]]:
        if isinstance(item, dict):
            label = _clean_alias_text(
                item.get("label")
                or item.get("entity")
                or item.get("name")
                or item.get("canonical")
            )
            aliases_raw = item.get("aliases")
            aliases: list[Any]
            if isinstance(aliases_raw, list):
                aliases = aliases_raw
            elif aliases_raw is None:
                aliases = []
            else:
                aliases = [aliases_raw]
            for k in ("zh", "zh_hant", "zh_tw", "english", "en"):
                if k in item:
                    aliases.append(item.get(k))
            return label, [_clean_alias_text(v) for v in aliases if _clean_alias_text(v)]
        label = _clean_alias_text(item)
        return label, []

    for item in parsed:
        label, aliases_raw = _parse_item(item)
        all_terms: list[str] = []
        seen_terms: set[str] = set()

        if label:
            seen_terms.add(label.casefold())
            all_terms.append(label)

        for alias in aliases_raw:
            low = alias.casefold()
            if low in seen_terms:
                continue
            seen_terms.add(low)
            all_terms.append(alias)
            if len(all_terms) >= max_aliases_per_entity:
                break

        if not all_terms:
            continue

        canonical = all_terms[0]
        canonical_key = canonical.casefold()
        aliases = all_terms[1:]

        existing_idx = by_label.get(canonical_key)
        if existing_idx is None:
            if len(out) >= max_entities:
                break
            out.append({"label": canonical, "aliases": aliases})
            by_label[canonical_key] = len(out) - 1
            continue

        existing = out[existing_idx]
        existing_aliases = existing.get("aliases")
        if not isinstance(existing_aliases, list):
            existing_aliases = []
        existing_seen = {
            str(existing.get("label", "")).casefold(),
            *[str(v).casefold() for v in existing_aliases if isinstance(v, str)],
        }
        for alias in aliases:
            low = alias.casefold()
            if low in existing_seen:
                continue
            existing_seen.add(low)
            existing_aliases.append(alias)
            if len(existing_aliases) >= max_aliases_per_entity:
                break
        existing["aliases"] = existing_aliases

    return out


def _entity_aliases_to_json(raw: Any) -> str | None:
    aliases = _normalise_entity_aliases(raw)
    if not aliases:
        return None
    return json.dumps(aliases, ensure_ascii=False, separators=(",", ":"))


def _entity_aliases_from_json(raw: Any) -> list[dict[str, Any]]:
    return _normalise_entity_aliases(raw)


def _entity_alias_terms(raw: Any, *, max_terms: int = 12) -> list[str]:
    aliases = _normalise_entity_aliases(raw)
    out: list[str] = []
    seen: set[str] = set()
    for item in aliases:
        label = _clean_alias_text(item.get("label"))
        if label:
            low = label.casefold()
            if low not in seen:
                seen.add(low)
                out.append(label)
                if len(out) >= max_terms:
                    break
        raw_aliases = item.get("aliases")
        if not isinstance(raw_aliases, list):
            continue
        for a in raw_aliases:
            alias = _clean_alias_text(a)
            if not alias:
                continue
            low = alias.casefold()
            if low in seen:
                continue
            seen.add(low)
            out.append(alias)
            if len(out) >= max_terms:
                break
        if len(out) >= max_terms:
            break
    return out


@dataclass(frozen=True)
class PoolLink:
    url: str
    norm_url: str
    domain: str | None
    title: str | None
    description: str | None
    age: str | None
    page_age: str | None
    query: str
    offset: int
    fetched_at_ts: int


class NewsPoolDB:
    def __init__(self, *, path: Path) -> None:
        self._path = path
        _ensure_parent_dir(path)

        # Use WAL for concurrent readers (planner) while collector is writing.
        self._conn = sqlite3.connect(str(path), timeout=30, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        # Ensure SQLite enforces declared foreign key constraints.
        self._conn.execute("PRAGMA foreign_keys=ON;")
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")

        self._ensure_schema()

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass

    def __enter__(self) -> "NewsPoolDB":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Schema management
    # ------------------------------------------------------------------

    def _ensure_schema(self) -> None:
        cur = self._conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS meta (
              k TEXT PRIMARY KEY,
              v TEXT NOT NULL
            );
            """
        )
        schema = self.get_meta_int("schema_version")

        if schema is None:
            # Fresh database — create all tables directly.
            self._create_all_tables_v5(cur)
            self._create_clustering_decisions_tables(cur)
            self._ensure_links_skip_cluster_columns(cur)
            self._ensure_events_entity_aliases_column(cur)
            self.set_meta("schema_version", str(SCHEMA_VERSION))
            return

        if schema == SCHEMA_VERSION:
            # Auxiliary tables should exist regardless of schema_version.
            self._create_clustering_decisions_tables(cur)
            self._ensure_links_skip_cluster_columns(cur)
            self._ensure_events_entity_aliases_column(cur)
            return

        if schema <= 4:
            self._migrate_to_v5(cur, from_version=schema)
            schema = 5

        if schema == 5:
            self._migrate_to_v6(cur)
            self._create_clustering_decisions_tables(cur)
            self._ensure_links_skip_cluster_columns(cur)
            self._ensure_events_entity_aliases_column(cur)
            self.set_meta("schema_version", str(SCHEMA_VERSION))
            return

        raise RuntimeError(f"Unsupported news pool schema_version={schema}, expected={SCHEMA_VERSION}")

    def _create_all_tables_v5(self, cur: sqlite3.Cursor) -> None:
        """Create all tables for a fresh v5 database."""
        # links
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS links (
              id INTEGER PRIMARY KEY,
              url TEXT NOT NULL,
              norm_url TEXT NOT NULL UNIQUE,
              domain TEXT,
              title TEXT,
              description TEXT,
              age TEXT,
              page_age TEXT,
              first_seen_ts INTEGER NOT NULL,
              last_seen_ts INTEGER NOT NULL,
              seen_count INTEGER NOT NULL DEFAULT 1,
              last_query TEXT NOT NULL,
              last_offset INTEGER NOT NULL,
              last_fetched_at_ts INTEGER NOT NULL,
              event_id INTEGER REFERENCES events(id),
              published_at_ts INTEGER,
              skip_cluster_reason TEXT,
              skip_clustered_at_ts INTEGER
            );
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_links_event_id ON links(event_id) WHERE event_id IS NOT NULL;")

        # fetch_state
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS fetch_state (
              key TEXT PRIMARY KEY,
              last_fetch_ts INTEGER NOT NULL,
              last_offset INTEGER NOT NULL DEFAULT 1,
              run_count INTEGER NOT NULL DEFAULT 0
            );
            """
        )

        # events (new v5 event-centric table)
        self._create_events_table_v5(cur)

        # article_cache
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS article_cache (
              norm_url TEXT PRIMARY KEY,
              url TEXT NOT NULL,
              final_url TEXT,
              domain TEXT,
              http_status INTEGER,
              extracted_title TEXT,
              extracted_text TEXT,
              extractor TEXT,
              fetched_at_ts INTEGER NOT NULL,
              text_chars INTEGER NOT NULL,
              quality_score INTEGER NOT NULL,
              error TEXT
            );
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_article_cache_fetched_at_ts ON article_cache(fetched_at_ts);")

        # pool_runs
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS pool_runs (
              id INTEGER PRIMARY KEY,
              run_ts INTEGER NOT NULL,
              state_key TEXT NOT NULL,
              window_hours INTEGER NOT NULL,
              should_fetch INTEGER NOT NULL,
              query TEXT,
              offset_start INTEGER,
              pages INTEGER,
              count INTEGER,
              freshness TEXT,
              requests_made INTEGER NOT NULL DEFAULT 0,
              results INTEGER NOT NULL DEFAULT 0,
              inserted INTEGER NOT NULL DEFAULT 0,
              updated INTEGER NOT NULL DEFAULT 0,
              pruned INTEGER NOT NULL DEFAULT 0,
              pruned_articles INTEGER NOT NULL DEFAULT 0,
              notes TEXT
            );
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_pool_runs_run_ts ON pool_runs(run_ts);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_pool_runs_state_key_run_ts ON pool_runs(state_key, run_ts);")

    def _create_clustering_decisions_tables(self, cur: sqlite3.Cursor) -> None:
        """Create clustering decision audit tables (auxiliary, schema-version agnostic)."""
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS clustering_decisions (
              id INTEGER PRIMARY KEY,
              link_id INTEGER NOT NULL REFERENCES links(id) ON DELETE CASCADE,
              prompt_sha256 TEXT NOT NULL,
              model_name TEXT,
              llm_started_at_ts INTEGER,
              llm_finished_at_ts INTEGER,
              llm_response_json TEXT,
              validated_action TEXT,
              validated_action_json TEXT,
              enforced_action TEXT,
              enforced_action_json TEXT,
              error_type TEXT,
              error_message TEXT,
              created_at_ts INTEGER NOT NULL
            );
            """
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_clustering_decisions_link_created ON clustering_decisions(link_id, created_at_ts);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_clustering_decisions_created ON clustering_decisions(created_at_ts);"
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS clustering_decision_candidates (
              decision_id INTEGER NOT NULL REFERENCES clustering_decisions(id) ON DELETE CASCADE,
              rank INTEGER NOT NULL,
              event_id INTEGER NOT NULL,
              score REAL NOT NULL,
              PRIMARY KEY(decision_id, rank)
            );
            """
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_clustering_decision_candidates_event_id ON clustering_decision_candidates(event_id);"
        )

    def _ensure_links_skip_cluster_columns(self, cur: sqlite3.Cursor) -> None:
        """Ensure links has persistent skip-clustering fields.

        These columns allow the clustering pipeline to deterministically skip
        multi-topic/roundup links without reprocessing them forever.
        """
        if not self._column_exists(cur, "links", "skip_cluster_reason"):
            cur.execute("ALTER TABLE links ADD COLUMN skip_cluster_reason TEXT;")
        if not self._column_exists(cur, "links", "skip_clustered_at_ts"):
            cur.execute("ALTER TABLE links ADD COLUMN skip_clustered_at_ts INTEGER;")

        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_links_skip_clustered_at_ts ON links(skip_clustered_at_ts) "
            "WHERE skip_clustered_at_ts IS NOT NULL;"
        )

    def _ensure_events_entity_aliases_column(self, cur: sqlite3.Cursor) -> None:
        """Ensure events can persist entity aliases as JSON text."""
        if not self._column_exists(cur, "events", "entity_aliases_json"):
            cur.execute("ALTER TABLE events ADD COLUMN entity_aliases_json TEXT;")

    def _create_events_table_v5(self, cur: sqlite3.Cursor) -> None:
        """Create the new event-centric events table (v5+v6 columns)."""
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
              id INTEGER PRIMARY KEY,
              parent_event_id INTEGER REFERENCES events(id),
              category TEXT,
              jurisdiction TEXT,
              summary_en TEXT NOT NULL,
              development TEXT,
              title TEXT,
              primary_url TEXT,
              link_count INTEGER NOT NULL DEFAULT 0,
              best_published_ts INTEGER,
              status TEXT NOT NULL DEFAULT 'new'
                CHECK (status IN ('new', 'active', 'posted')),
              created_at_ts INTEGER NOT NULL,
              updated_at_ts INTEGER NOT NULL,
              expires_at_ts INTEGER,
              posted_at_ts INTEGER,
              thread_id TEXT,
              run_id TEXT,
              entity_aliases_json TEXT,
              model TEXT,
              reserved_until_ts INTEGER
            );
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_events_parent ON events(parent_event_id);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_events_status ON events(status, created_at_ts);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_events_expires ON events(expires_at_ts) WHERE expires_at_ts IS NOT NULL;")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_events_category ON events(category, status);")

    def insert_clustering_decision(
        self,
        *,
        link_id: int,
        candidates: list[tuple[int, float]] | None,
        prompt_sha256: str,
        model_name: str | None = None,
        llm_response: Any | None = None,
        validated_action: dict[str, Any] | None = None,
        enforced_action: dict[str, Any] | None = None,
        llm_started_at_ts: int | None = None,
        llm_finished_at_ts: int | None = None,
        error_type: str | None = None,
        error_message: str | None = None,
        created_at_ts: int | None = None,
    ) -> int:
        """Insert a clustering decision audit row and its candidate set.

        This is intended for debugging and post-mortems; it must not affect
        the clustering outcome if it fails.
        """
        if not isinstance(link_id, int):
            raise TypeError("link_id must be int")
        if not isinstance(prompt_sha256, str) or not prompt_sha256.strip():
            raise ValueError("prompt_sha256 must be a non-empty str")

        def _json_text(obj: Any | None) -> str | None:
            if obj is None:
                return None
            try:
                return json.dumps(obj, ensure_ascii=False, sort_keys=True)
            except Exception:
                return json.dumps({"_repr": repr(obj)}, ensure_ascii=False)

        now = int(created_at_ts) if created_at_ts is not None else _utc_now_ts()
        cur = self._conn.cursor()
        cur.execute(
            """
            INSERT INTO clustering_decisions(
              link_id, prompt_sha256, model_name,
              llm_started_at_ts, llm_finished_at_ts,
              llm_response_json,
              validated_action, validated_action_json,
              enforced_action, enforced_action_json,
              error_type, error_message,
              created_at_ts
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(link_id),
                prompt_sha256.strip(),
                (model_name.strip() if isinstance(model_name, str) and model_name.strip() else None),
                (int(llm_started_at_ts) if llm_started_at_ts is not None else None),
                (int(llm_finished_at_ts) if llm_finished_at_ts is not None else None),
                _json_text(llm_response),
                (str(validated_action.get("action")) if isinstance(validated_action, dict) and validated_action.get("action") else None),
                _json_text(validated_action),
                (str(enforced_action.get("action")) if isinstance(enforced_action, dict) and enforced_action.get("action") else None),
                _json_text(enforced_action),
                (str(error_type) if isinstance(error_type, str) and error_type.strip() else None),
                (str(error_message) if isinstance(error_message, str) and error_message.strip() else None),
                int(now),
            ),
        )
        decision_id = int(cur.lastrowid)

        for rank, (event_id, score) in enumerate(candidates or [], start=1):
            cur.execute(
                """
                INSERT INTO clustering_decision_candidates(decision_id, rank, event_id, score)
                VALUES (?, ?, ?, ?)
                """,
                (decision_id, int(rank), int(event_id), float(score)),
            )
        return decision_id

    def _migrate_to_v5(self, cur: sqlite3.Cursor, *, from_version: int) -> None:
        """Migrate from schema v1-v4 to v5."""
        # Ensure v1-v4 tables exist (idempotent) before renaming.
        # Create any missing tables from earlier versions first.
        for tbl_sql in [
            "CREATE TABLE IF NOT EXISTS links (id INTEGER PRIMARY KEY, url TEXT NOT NULL, norm_url TEXT NOT NULL UNIQUE, domain TEXT, title TEXT, description TEXT, age TEXT, page_age TEXT, first_seen_ts INTEGER NOT NULL, last_seen_ts INTEGER NOT NULL, seen_count INTEGER NOT NULL DEFAULT 1, last_query TEXT NOT NULL, last_offset INTEGER NOT NULL, last_fetched_at_ts INTEGER NOT NULL);",
            "CREATE TABLE IF NOT EXISTS fetch_state (key TEXT PRIMARY KEY, last_fetch_ts INTEGER NOT NULL, last_offset INTEGER NOT NULL DEFAULT 1, run_count INTEGER NOT NULL DEFAULT 0);",
            "CREATE TABLE IF NOT EXISTS article_cache (norm_url TEXT PRIMARY KEY, url TEXT NOT NULL, final_url TEXT, domain TEXT, http_status INTEGER, extracted_title TEXT, extracted_text TEXT, extractor TEXT, fetched_at_ts INTEGER NOT NULL, text_chars INTEGER NOT NULL, quality_score INTEGER NOT NULL, error TEXT);",
            "CREATE TABLE IF NOT EXISTS pool_runs (id INTEGER PRIMARY KEY, run_ts INTEGER NOT NULL, state_key TEXT NOT NULL, window_hours INTEGER NOT NULL, should_fetch INTEGER NOT NULL, query TEXT, offset_start INTEGER, pages INTEGER, count INTEGER, freshness TEXT, requests_made INTEGER NOT NULL DEFAULT 0, results INTEGER NOT NULL DEFAULT 0, inserted INTEGER NOT NULL DEFAULT 0, updated INTEGER NOT NULL DEFAULT 0, pruned INTEGER NOT NULL DEFAULT 0, pruned_articles INTEGER NOT NULL DEFAULT 0, notes TEXT);",
        ]:
            cur.execute(tbl_sql)

        # Rename old tables that are being replaced.
        # events → events_legacy (old token-hash based events)
        if self._table_exists(cur, "events"):
            cur.execute("ALTER TABLE events RENAME TO events_legacy;")

        # semantic_keys → semantic_keys_legacy
        if self._table_exists(cur, "semantic_keys"):
            cur.execute("ALTER TABLE semantic_keys RENAME TO semantic_keys_legacy;")

        # posted_events → posted_events_legacy
        if self._table_exists(cur, "posted_events"):
            cur.execute("ALTER TABLE posted_events RENAME TO posted_events_legacy;")

        # Create new events table.
        self._create_events_table_v5(cur)

        # ALTER links: add event_id and published_at_ts columns.
        if not self._column_exists(cur, "links", "event_id"):
            cur.execute("ALTER TABLE links ADD COLUMN event_id INTEGER REFERENCES events(id);")
        if not self._column_exists(cur, "links", "published_at_ts"):
            cur.execute("ALTER TABLE links ADD COLUMN published_at_ts INTEGER;")

        cur.execute("CREATE INDEX IF NOT EXISTS idx_links_event_id ON links(event_id) WHERE event_id IS NOT NULL;")

        # Backfill published_at_ts from page_age.
        rows = cur.execute("SELECT id, page_age FROM links WHERE published_at_ts IS NULL AND page_age IS NOT NULL").fetchall()
        for row in rows:
            ts = _parse_page_age_ts(row["page_age"])
            if ts is not None:
                cur.execute("UPDATE links SET published_at_ts = ? WHERE id = ?", (ts, row["id"]))

        # Migrate posted_events_legacy data into new events table.
        if self._table_exists(cur, "posted_events_legacy"):
            self._migrate_posted_events_legacy(cur)

        # Create remaining indexes.
        cur.execute("CREATE INDEX IF NOT EXISTS idx_article_cache_fetched_at_ts ON article_cache(fetched_at_ts);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_pool_runs_run_ts ON pool_runs(run_ts);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_pool_runs_state_key_run_ts ON pool_runs(state_key, run_ts);")

    def _migrate_posted_events_legacy(self, cur: sqlite3.Cursor) -> None:
        """Migrate posted_events_legacy rows into the new events table."""
        now = _utc_now_ts()
        rows = cur.execute(
            """
            SELECT id, event_type, jurisdiction, summary_en, title, primary_url,
                   thread_id, run_id, category, posted_at_ts, expires_at_ts
            FROM posted_events_legacy
            ORDER BY posted_at_ts ASC
            """
        ).fetchall()
        for row in rows:
            cur.execute(
                """
                INSERT INTO events (
                  category, jurisdiction, summary_en, title, primary_url,
                  status, created_at_ts, updated_at_ts, expires_at_ts,
                  posted_at_ts, thread_id, run_id, model
                ) VALUES (?, ?, ?, ?, ?, 'posted', ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row["category"] or row["event_type"],
                    row["jurisdiction"],
                    row["summary_en"] or row["title"],
                    row["title"],
                    row["primary_url"],
                    row["posted_at_ts"] or now,
                    now,
                    row["expires_at_ts"],
                    row["posted_at_ts"],
                    row["thread_id"],
                    row["run_id"],
                    "migrated_from_v4",
                ),
            )

    def _migrate_to_v6(self, cur: sqlite3.Cursor) -> None:
        """Migrate from schema v5 to v6: add reserved_until_ts column."""
        if not self._column_exists(cur, "events", "reserved_until_ts"):
            cur.execute("ALTER TABLE events ADD COLUMN reserved_until_ts INTEGER;")

    @staticmethod
    def _table_exists(cur: sqlite3.Cursor, name: str) -> bool:
        row = cur.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone()
        return row is not None

    @staticmethod
    def _column_exists(cur: sqlite3.Cursor, table: str, column: str) -> bool:
        cols = cur.execute(f"PRAGMA table_info({table})").fetchall()
        return any(c["name"] == column for c in cols)

    # ------------------------------------------------------------------
    # Meta helpers
    # ------------------------------------------------------------------

    def get_meta(self, key: str) -> str | None:
        cur = self._conn.cursor()
        row = cur.execute("SELECT v FROM meta WHERE k = ?", (key,)).fetchone()
        return str(row["v"]) if row else None

    def get_meta_int(self, key: str) -> int | None:
        v = self.get_meta(key)
        if v is None:
            return None
        try:
            return int(v)
        except ValueError:
            return None

    def set_meta(self, key: str, value: str) -> None:
        self._conn.execute("INSERT INTO meta(k, v) VALUES(?, ?) ON CONFLICT(k) DO UPDATE SET v=excluded.v", (key, value))

    # ------------------------------------------------------------------
    # Fetch state
    # ------------------------------------------------------------------

    def fetch_state(self, key: str) -> dict[str, int]:
        cur = self._conn.cursor()
        row = cur.execute("SELECT last_fetch_ts, last_offset, run_count FROM fetch_state WHERE key = ?", (key,)).fetchone()
        if not row:
            return {"last_fetch_ts": 0, "last_offset": 1, "run_count": 0}
        return {"last_fetch_ts": int(row["last_fetch_ts"]), "last_offset": int(row["last_offset"]), "run_count": int(row["run_count"])}

    def update_fetch_state(self, *, key: str, last_fetch_ts: int, last_offset: int, run_count: int) -> None:
        self._conn.execute(
            """
            INSERT INTO fetch_state(key, last_fetch_ts, last_offset, run_count)
            VALUES(?, ?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
              last_fetch_ts=excluded.last_fetch_ts,
              last_offset=excluded.last_offset,
              run_count=excluded.run_count
            """,
            (key, int(last_fetch_ts), int(last_offset), int(run_count)),
        )

    # ------------------------------------------------------------------
    # Links
    # ------------------------------------------------------------------

    def upsert_links(self, links: Iterable[PoolLink], *, now_ts: int | None = None) -> dict[str, int]:
        now_ts = _utc_now_ts() if now_ts is None else int(now_ts)
        inserted = 0
        updated = 0

        cur = self._conn.cursor()
        for it in links:
            published_at_ts = _parse_page_age_ts(it.page_age)
            row = cur.execute("SELECT id FROM links WHERE norm_url = ?", (it.norm_url,)).fetchone()
            if row:
                cur.execute(
                    """
                    UPDATE links
                    SET
                      url=?,
                      domain=?,
                      title=?,
                      description=?,
                      age=?,
                      page_age=?,
                      last_seen_ts=?,
                      seen_count=seen_count + 1,
                      last_query=?,
                      last_offset=?,
                      last_fetched_at_ts=?,
                      published_at_ts=COALESCE(?, published_at_ts)
                    WHERE norm_url=?
                    """,
                    (
                        it.url,
                        it.domain,
                        it.title,
                        it.description,
                        it.age,
                        it.page_age,
                        now_ts,
                        it.query,
                        int(it.offset),
                        int(it.fetched_at_ts),
                        published_at_ts,
                        it.norm_url,
                    ),
                )
                updated += 1
            else:
                cur.execute(
                    """
                    INSERT INTO links(
                      url, norm_url, domain, title, description, age, page_age,
                      first_seen_ts, last_seen_ts, seen_count, last_query, last_offset, last_fetched_at_ts,
                      published_at_ts
                    )
                    VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?)
                    """,
                    (
                        it.url,
                        it.norm_url,
                        it.domain,
                        it.title,
                        it.description,
                        it.age,
                        it.page_age,
                        now_ts,
                        now_ts,
                        it.query,
                        int(it.offset),
                        int(it.fetched_at_ts),
                        published_at_ts,
                    ),
                )
                inserted += 1

        return {"inserted": inserted, "updated": updated}

    def prune_links(self, *, cutoff_ts: int) -> int:
        cur = self._conn.cursor()
        row = cur.execute("SELECT COUNT(1) AS n FROM links WHERE last_seen_ts < ?", (int(cutoff_ts),)).fetchone()
        n = int(row["n"]) if row else 0

        if n > 0:
            # Recalculate link_count for events that will lose links.
            # Collect affected event IDs before deleting.
            affected_rows = cur.execute(
                "SELECT DISTINCT event_id FROM links WHERE last_seen_ts < ? AND event_id IS NOT NULL",
                (int(cutoff_ts),),
            ).fetchall()
            affected_event_ids = [r["event_id"] for r in affected_rows]

            cur.execute("DELETE FROM links WHERE last_seen_ts < ?", (int(cutoff_ts),))

            # Recalculate link_count from remaining links for each affected event.
            for eid in affected_event_ids:
                cnt_row = cur.execute(
                    "SELECT COUNT(1) AS c FROM links WHERE event_id = ?", (eid,)
                ).fetchone()
                actual = int(cnt_row["c"]) if cnt_row else 0
                cur.execute(
                    "UPDATE events SET link_count = ? WHERE id = ?",
                    (actual, eid),
                )
        else:
            cur.execute("DELETE FROM links WHERE last_seen_ts < ?", (int(cutoff_ts),))

        return n

    def iter_links_since(self, *, cutoff_ts: int) -> Iterable[dict[str, Any]]:
        cur = self._conn.cursor()
        rows = cur.execute(
            """
            SELECT
              id, url, norm_url, domain, title, description, age, page_age,
              first_seen_ts, last_seen_ts, seen_count, event_id, published_at_ts
            FROM links
            WHERE last_seen_ts >= ?
            ORDER BY last_seen_ts DESC
            """,
            (int(cutoff_ts),),
        )
        for r in rows:
            yield dict(r)

    def get_unassigned_links(self, *, max_age_seconds: int = _DEFAULT_TTL_SECONDS) -> list[dict[str, Any]]:
        """Return links with event_id IS NULL, oldest first, within age window."""
        now = _utc_now_ts()
        cutoff = now - max_age_seconds
        cur = self._conn.cursor()
        rows = cur.execute(
            """
            SELECT id, url, norm_url, domain, title, description, page_age,
                   first_seen_ts, last_seen_ts, published_at_ts
            FROM links
            WHERE event_id IS NULL
              AND skip_clustered_at_ts IS NULL
              AND last_seen_ts >= ?
            ORDER BY COALESCE(published_at_ts, first_seen_ts) ASC
            """,
            (cutoff,),
        ).fetchall()
        return [dict(r) for r in rows]

    def mark_link_skip_cluster(self, *, link_id: int, reason: str, now_ts: int | None = None) -> None:
        """Mark a link as skipped for clustering (persistent).

        This prevents the same multi-topic/roundup link from being reprocessed on
        every clustering run.
        """
        now = _utc_now_ts() if now_ts is None else int(now_ts)
        r = str(reason or "").strip()
        if not r:
            r = "skip_cluster"
        if len(r) > 200:
            r = r[:200]
        self._conn.execute(
            "UPDATE links SET skip_cluster_reason = ?, skip_clustered_at_ts = ? WHERE id = ?",
            (r, now, int(link_id)),
        )

    def assign_link_to_event(self, *, link_id: int, event_id: int) -> None:
        """Set event_id FK on a link and update the event's counters.

        If the link was previously assigned to a different event, that event's
        link_count is decremented to prevent drift.
        """
        now = _utc_now_ts()
        cur = self._conn.cursor()

        # Check if link was previously assigned to a different event.
        prev_row = cur.execute("SELECT event_id, published_at_ts FROM links WHERE id = ?", (link_id,)).fetchone()
        prev_event_id = prev_row["event_id"] if prev_row else None
        link_pub_ts = prev_row["published_at_ts"] if prev_row else None

        cur.execute("UPDATE links SET event_id = ? WHERE id = ?", (event_id, link_id))

        # Decrement old event's link_count if link was reassigned.
        if prev_event_id is not None and prev_event_id != event_id:
            cur.execute(
                "UPDATE events SET link_count = MAX(link_count - 1, 0), updated_at_ts = ? WHERE id = ?",
                (now, prev_event_id),
            )

        expires = now + _DEFAULT_TTL_SECONDS
        if prev_event_id == event_id:
            # Link already assigned to this event — no count change needed.
            pass
        else:
            cur.execute(
                """
                UPDATE events SET
                  link_count = link_count + 1,
                  best_published_ts = MAX(COALESCE(best_published_ts, 0), COALESCE(?, 0)),
                  updated_at_ts = ?,
                  expires_at_ts = MAX(COALESCE(expires_at_ts, 0), ?),
                  status = CASE WHEN status = 'new' THEN 'active' ELSE status END
                WHERE id = ?
                """,
                (link_pub_ts, now, expires, event_id),
            )

    # ------------------------------------------------------------------
    # Events (v5 event-centric)
    # ------------------------------------------------------------------

    def create_event(
        self,
        *,
        summary_en: str,
        category: str | None = None,
        jurisdiction: str | None = None,
        title: str | None = None,
        primary_url: str | None = None,
        parent_event_id: int | None = None,
        development: str | None = None,
        entity_aliases: Any | None = None,
        model: str | None = None,
    ) -> int:
        """Insert a new event. Returns the event id."""
        now = _utc_now_ts()
        expires = now + _DEFAULT_TTL_SECONDS
        aliases_json = _entity_aliases_to_json(entity_aliases)
        cur = self._conn.cursor()
        cur.execute(
            """
            INSERT INTO events (
              parent_event_id, category, jurisdiction, summary_en, development,
              title, primary_url, link_count, best_published_ts, entity_aliases_json,
              status, created_at_ts, updated_at_ts, expires_at_ts, model
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 0, NULL, ?, 'new', ?, ?, ?, ?)
            """,
            (
                parent_event_id,
                category,
                jurisdiction,
                summary_en,
                development,
                title,
                primary_url,
                aliases_json,
                now,
                now,
                expires,
                model,
            ),
        )
        event_id = cur.lastrowid or 0

        # If this is a development, bump parent's expires_at_ts.
        if parent_event_id is not None:
            cur.execute(
                "UPDATE events SET expires_at_ts = MAX(COALESCE(expires_at_ts, 0), ?), updated_at_ts = ? WHERE id = ?",
                (expires, now, parent_event_id),
            )

        return event_id

    def get_event(self, event_id: int) -> dict[str, Any] | None:
        """Get a single event by id."""
        cur = self._conn.cursor()
        row = cur.execute("SELECT * FROM events WHERE id = ?", (event_id,)).fetchone()
        if not row:
            return None
        out = dict(row)
        out["entity_aliases"] = _entity_aliases_from_json(out.get("entity_aliases_json"))
        return out

    def get_all_fresh_events(
        self,
        *,
        max_age_hours: int = 168,
        now_ts: int | None = None,
        posted_recent_hours: int = 48,
        max_root_events: int = 600,
    ) -> list[dict[str, Any]]:
        """Return a bounded retrieval universe for clustering/merging.

        Notes:
        - Root events only (parent_event_id IS NULL).
        - Unposted events are included only if they are unexpired (expires_at_ts > now).
        - Posted events are included only within a short recent window for dedupe.
        - Ordering prefers updated_at_ts / best_published_ts over created_at_ts.

        Args:
            max_age_hours: Hard safety window for unposted events (based on updated/best_published).
            posted_recent_hours: Window for including posted events (dedupe).
            max_root_events: Maximum number of unposted root events to return.
        """
        now = now_ts if now_ts is not None else _utc_now_ts()
        unposted_cutoff = now - max_age_hours * 3600
        posted_cutoff = now - posted_recent_hours * 3600
        cur = self._conn.cursor()

        # Active/unposted root events: unexpired + recently updated/published.
        unposted_rows = cur.execute(
            """
            SELECT id, parent_event_id, category, jurisdiction, summary_en,
                   development, title, primary_url, link_count, best_published_ts,
                   status, created_at_ts, updated_at_ts, expires_at_ts, entity_aliases_json,
                   posted_at_ts, thread_id, run_id
            FROM events
            WHERE parent_event_id IS NULL
              AND status IN ('new', 'active')
              AND expires_at_ts > ?
              AND (
                    COALESCE(updated_at_ts, 0) >= ?
                 OR COALESCE(best_published_ts, 0) >= ?
                 OR COALESCE(created_at_ts, 0) >= ?
              )
            ORDER BY COALESCE(updated_at_ts, 0) DESC,
                     COALESCE(best_published_ts, 0) DESC,
                     id DESC
            LIMIT ?
            """,
            (now, unposted_cutoff, unposted_cutoff, unposted_cutoff, max_root_events),
        ).fetchall()

        # Recent posted root events: include only for short-window dedupe/assignment.
        posted_rows = cur.execute(
            """
            SELECT id, parent_event_id, category, jurisdiction, summary_en,
                   development, title, primary_url, link_count, best_published_ts,
                   status, created_at_ts, updated_at_ts, expires_at_ts, entity_aliases_json,
                   posted_at_ts, thread_id, run_id
            FROM events
            WHERE parent_event_id IS NULL
              AND status = 'posted'
              AND COALESCE(posted_at_ts, 0) >= ?
            ORDER BY COALESCE(posted_at_ts, 0) DESC,
                     id DESC
            """,
            (posted_cutoff,),
        ).fetchall()

        rows = list(unposted_rows) + list(posted_rows)
        out: list[dict[str, Any]] = []
        for r in rows:
            item = dict(r)
            item["entity_aliases"] = _entity_aliases_from_json(item.get("entity_aliases_json"))
            out.append(item)
        return out

    def get_root_event_id(self, event_id: int) -> int:
        """Walk parent_event_id chain to root. Returns event_id itself if already root."""
        cur = self._conn.cursor()
        current = event_id
        seen: set[int] = set()
        while True:
            if current in seen:
                break  # cycle guard
            seen.add(current)
            row = cur.execute(
                "SELECT parent_event_id FROM events WHERE id = ?", (current,)
            ).fetchone()
            if not row or row["parent_event_id"] is None:
                break
            current = row["parent_event_id"]
        return current

    def get_fresh_events(self, *, now_ts: int | None = None) -> list[dict[str, Any]]:
        """Return events where expires_at_ts > now (for clustering prompt)."""
        now = now_ts if now_ts is not None else _utc_now_ts()
        cur = self._conn.cursor()
        rows = cur.execute(
            """
            SELECT id, parent_event_id, category, jurisdiction, summary_en,
                   development, title, primary_url, link_count, best_published_ts,
                   status, created_at_ts, updated_at_ts, expires_at_ts, entity_aliases_json
            FROM events
            WHERE expires_at_ts > ?
            ORDER BY created_at_ts DESC
            """,
            (now,),
        ).fetchall()
        out: list[dict[str, Any]] = []
        for r in rows:
            item = dict(r)
            item["entity_aliases"] = _entity_aliases_from_json(item.get("entity_aliases_json"))
            out.append(item)
        return out

    def prune_expired_events(self, *, max_age_hours: int = 72, now_ts: int | None = None) -> int:
        """Delete events that are expired AND older than max_age_hours.

        Returns the number of pruned events.
        Only prunes events with status != 'posted' (keep posted for audit trail).
        Also nullifies event_id on orphaned links (no CASCADE on FK).
        """
        now = now_ts if now_ts is not None else _utc_now_ts()
        age_cutoff = now - max_age_hours * 3600
        cur = self._conn.cursor()

        # Find event IDs to prune.
        rows = cur.execute(
            """
            SELECT id FROM events
            WHERE status != 'posted'
              AND expires_at_ts < ?
              AND created_at_ts < ?
            """,
            (now, age_cutoff),
        ).fetchall()

        if not rows:
            return 0

        ids = [r["id"] for r in rows]
        placeholders = ",".join("?" for _ in ids)

        # Nullify event_id on orphaned links.
        cur.execute(
            f"UPDATE links SET event_id = NULL WHERE event_id IN ({placeholders})",
            ids,
        )

        # Delete the expired events.
        cur.execute(
            f"DELETE FROM events WHERE id IN ({placeholders})",
            ids,
        )

        n = len(ids)
        logger.info("Pruned %d expired events (max_age_hours=%d)", n, max_age_hours)
        return n

    def get_daily_candidates(self, *, limit: int = 15, now_ts: int | None = None) -> list[dict[str, Any]]:
        """Select events for daily posting with category balance and HK guarantee.

        Rules:
        - Only fresh, unposted events (status IN ('new', 'active'), expires_at_ts > now)
        - Ranked by link_count DESC, best_published_ts DESC
        - Finance categories capped at 2
        - No single category > 3
        - Reserve 1-2 slots for jurisdiction='HK'
        - Selected events are reserved for 10 minutes to prevent concurrent planners
          from picking the same events.
        """
        now = now_ts if now_ts is not None else _utc_now_ts()
        self.prune_expired_events(now_ts=now)
        cur = self._conn.cursor()

        try:
            cur.execute("BEGIN IMMEDIATE")

            rows = cur.execute(
                """
                SELECT id, parent_event_id, category, jurisdiction, summary_en,
                       development, title, primary_url, link_count, best_published_ts,
                       status, created_at_ts, updated_at_ts, expires_at_ts, entity_aliases_json
                FROM events
                WHERE status IN ('new', 'active')
                  AND expires_at_ts > ?
                  AND (reserved_until_ts IS NULL OR reserved_until_ts < ?)
                ORDER BY link_count DESC, best_published_ts DESC
                """,
                (now, now),
            ).fetchall()
            candidates = [dict(r) for r in rows]
            selected = self._apply_daily_selection(candidates, limit=limit)

            if selected:
                reserve_ts = now + _RESERVATION_SECONDS
                ids = [c["id"] for c in selected]
                placeholders = ",".join("?" for _ in ids)
                cur.execute(
                    f"UPDATE events SET reserved_until_ts = ? WHERE id IN ({placeholders})",
                    [reserve_ts] + ids,
                )

            cur.execute("COMMIT")
        except Exception:
            try:
                cur.execute("ROLLBACK")
            except Exception:
                pass
            raise

        return self._enrich_candidates(selected, now_ts=now)

    @staticmethod
    def _is_finance_category(category: str | None) -> bool:
        if not category:
            return False
        low = category.lower()
        return any(k in low for k in ("stock", "crypto", "precious metal", "finance", "earnings"))

    @staticmethod
    def _apply_daily_selection(candidates: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
        """Apply category balance, finance cap, and HK guarantee."""
        finance_max = 2
        category_max = 3

        selected: list[dict[str, Any]] = []
        category_counts: dict[str, int] = {}
        finance_count = 0

        # First pass: collect HK events (reserve up to 2 slots).
        hk_events: list[dict[str, Any]] = []
        non_hk_events: list[dict[str, Any]] = []
        for c in candidates:
            jur = (c.get("jurisdiction") or "").upper()
            if jur == "HK":
                hk_events.append(c)
            else:
                non_hk_events.append(c)

        # Guarantee 1-2 HK slots.
        hk_guaranteed = 0
        for c in hk_events:
            if hk_guaranteed >= 2:
                break
            cat = c.get("category") or "Other"
            is_fin = NewsPoolDB._is_finance_category(cat)
            if is_fin and finance_count >= finance_max:
                continue
            if category_counts.get(cat, 0) >= category_max:
                continue
            selected.append(c)
            category_counts[cat] = category_counts.get(cat, 0) + 1
            if is_fin:
                finance_count += 1
            hk_guaranteed += 1

        # Fill remaining slots.
        used_ids = {c["id"] for c in selected}
        for c in candidates:
            if len(selected) >= limit:
                break
            if c["id"] in used_ids:
                continue
            cat = c.get("category") or "Other"
            is_fin = NewsPoolDB._is_finance_category(cat)
            if is_fin and finance_count >= finance_max:
                continue
            if category_counts.get(cat, 0) >= category_max:
                continue
            selected.append(c)
            used_ids.add(c["id"])
            category_counts[cat] = category_counts.get(cat, 0) + 1
            if is_fin:
                finance_count += 1

        return selected

    def get_hourly_candidates(self, *, limit: int = 3, now_ts: int | None = None) -> list[dict[str, Any]]:
        """Select events for hourly posting with development priority.

        Priority:
        1. Developments first (parent_event_id IS NOT NULL)
        2. Freshness (created_at_ts DESC)
        3. link_count DESC
        Category diversity: 2nd+ picks differ from 1st pick's category.
        Selected events are reserved for 10 minutes to prevent concurrent planners
        from picking the same events.
        """
        now = now_ts if now_ts is not None else _utc_now_ts()
        self.prune_expired_events(now_ts=now)
        cur = self._conn.cursor()

        try:
            cur.execute("BEGIN IMMEDIATE")

            rows = cur.execute(
                """
                SELECT id, parent_event_id, category, jurisdiction, summary_en,
                       development, title, primary_url, link_count, best_published_ts,
                       status, created_at_ts, updated_at_ts, expires_at_ts, entity_aliases_json
                FROM events
                WHERE status IN ('new', 'active')
                  AND expires_at_ts > ?
                  AND (reserved_until_ts IS NULL OR reserved_until_ts < ?)
                ORDER BY
                  (CASE WHEN parent_event_id IS NOT NULL THEN 0 ELSE 1 END) ASC,
                  created_at_ts DESC,
                  link_count DESC
                """,
                (now, now),
            ).fetchall()
            candidates = [dict(r) for r in rows]

            if not candidates:
                cur.execute("COMMIT")
                return []

            selected: list[dict[str, Any]] = []
            first_category: str | None = None

            for c in candidates:
                if len(selected) >= limit:
                    break
                cat = c.get("category") or "Other"
                if len(selected) == 0:
                    selected.append(c)
                    first_category = cat
                elif cat != first_category:
                    selected.append(c)
                # If same category as first and we already have 1, skip for diversity.

            # If we couldn't fill with diverse categories, fill with same category.
            if len(selected) < limit:
                used_ids = {c["id"] for c in selected}
                for c in candidates:
                    if len(selected) >= limit:
                        break
                    if c["id"] not in used_ids:
                        selected.append(c)
                        used_ids.add(c["id"])

            if selected:
                reserve_ts = now + _RESERVATION_SECONDS
                ids = [c["id"] for c in selected]
                placeholders = ",".join("?" for _ in ids)
                cur.execute(
                    f"UPDATE events SET reserved_until_ts = ? WHERE id IN ({placeholders})",
                    [reserve_ts] + ids,
                )

            cur.execute("COMMIT")
        except Exception:
            try:
                cur.execute("ROLLBACK")
            except Exception:
                pass
            raise

        return self._enrich_candidates(selected, now_ts=now)

    def _enrich_candidates(self, candidates: list[dict[str, Any]], *, now_ts: int) -> list[dict[str, Any]]:
        """Add backward-compat fields expected by newsroom_write_run_job.py."""
        cur = self._conn.cursor()
        for c in candidates:
            eid = c["id"]

            # supporting_urls + domains from linked URLs.
            rows = cur.execute(
                "SELECT url, domain FROM links WHERE event_id = ? ORDER BY published_at_ts DESC",
                (eid,),
            ).fetchall()
            primary = c.get("primary_url") or ""
            supporting: list[str] = []
            domains: set[str] = set()
            for r in rows:
                u = r["url"]
                d = r["domain"]
                if d:
                    domains.add(d)
                if u and u != primary:
                    supporting.append(u)
            c["supporting_urls"] = supporting[:4]
            c["domains"] = sorted(domains)

            # age_minutes from best_published_ts.
            bpt = c.get("best_published_ts")
            if bpt and isinstance(bpt, (int, float)):
                c["age_minutes"] = max(0, int((now_ts - int(bpt)) / 60))
            else:
                c["age_minutes"] = None

            # Field aliases for backward compat with newsroom_write_run_job.py.
            c["suggested_category"] = c.get("category") or "Global News"
            c["description"] = c.get("summary_en") or ""
            root_eid = self.get_root_event_id(eid)
            c["event_key"] = f"event:{root_eid}"
            c["semantic_event_key"] = f"event:{root_eid}"
            c["anchor_key"] = f"event:{root_eid}"
            c["cluster_size"] = c.get("link_count") or 0
            aliases = _entity_aliases_from_json(c.get("entity_aliases_json"))
            alias_terms = _entity_alias_terms(aliases, max_terms=12)
            c["entity_aliases"] = aliases
            c["cluster_terms"] = list(alias_terms)
            c["anchor_terms"] = list(alias_terms[:8])
            c["suggest_flags"] = []
            c["event_id"] = eid

        return candidates

    def mark_event_posted(self, event_id: int, *, thread_id: str | None = None, run_id: str | None = None) -> None:
        """Mark an event as posted."""
        now = _utc_now_ts()
        self._conn.execute(
            """
            UPDATE events SET
              status = 'posted',
              posted_at_ts = ?,
              thread_id = ?,
              run_id = ?,
              updated_at_ts = ?
            WHERE id = ?
            """,
            (now, thread_id, run_id, now, event_id),
        )

    def release_reservation(self, event_id: int) -> None:
        """Clear the reservation on an event so it can be selected again."""
        self._conn.execute(
            "UPDATE events SET reserved_until_ts = NULL WHERE id = ?",
            (event_id,),
        )

    def merge_events_into(self, *, winner_id: int, loser_ids: list[int]) -> int:
        """Merge loser events into winner: move links, re-parent children, delete losers.

        Returns the number of links moved. Atomic — rolls back on failure.
        """
        if not loser_ids:
            return 0

        cur = self._conn.cursor()
        now = _utc_now_ts()
        placeholders = ",".join("?" for _ in loser_ids)

        try:
            cur.execute("BEGIN")

            # 1. Reassign links from losers to winner.
            cur.execute(
                f"UPDATE links SET event_id = ? WHERE event_id IN ({placeholders})",
                [winner_id] + loser_ids,
            )
            links_moved = cur.rowcount

            # 2. Re-parent children of losers to winner.
            cur.execute(
                f"UPDATE events SET parent_event_id = ? WHERE parent_event_id IN ({placeholders})",
                [winner_id] + loser_ids,
            )

            # 3. Recalculate winner's link_count from actual links.
            row = cur.execute(
                "SELECT COUNT(1) AS n FROM links WHERE event_id = ?", (winner_id,)
            ).fetchone()
            actual_link_count = int(row["n"]) if row else 0

            # 4. Aggregate best_published_ts and expires_at_ts across winner + losers.
            row = cur.execute(
                f"SELECT MAX(best_published_ts) AS bp, MAX(expires_at_ts) AS ex "
                f"FROM events WHERE id IN (?, {placeholders})",
                [winner_id] + loser_ids,
            ).fetchone()
            best_pub = row["bp"] if row else None
            best_exp = row["ex"] if row else None
            alias_rows = cur.execute(
                f"SELECT entity_aliases_json FROM events WHERE id IN (?, {placeholders})",
                [winner_id] + loser_ids,
            ).fetchall()
            merged_aliases: list[dict[str, Any]] = []
            for arow in alias_rows:
                merged_aliases.extend(_entity_aliases_from_json(arow["entity_aliases_json"]))
            merged_aliases_json = _entity_aliases_to_json(merged_aliases)

            # 5. Update winner: link_count, timestamps, promote if multi-link.
            new_status_expr = "CASE WHEN ? > 1 AND status = 'new' THEN 'active' ELSE status END"
            cur.execute(
                f"""
                UPDATE events SET
                  link_count = ?,
                  best_published_ts = ?,
                  expires_at_ts = ?,
                  entity_aliases_json = ?,
                  updated_at_ts = ?,
                  status = {new_status_expr}
                WHERE id = ?
                """,
                (
                    actual_link_count,
                    best_pub,
                    best_exp,
                    merged_aliases_json,
                    now,
                    actual_link_count,
                    winner_id,
                ),
            )

            # 6. Delete losers.
            cur.execute(
                f"DELETE FROM events WHERE id IN ({placeholders})",
                loser_ids,
            )

            cur.execute("COMMIT")
            return links_moved

        except Exception:
            try:
                cur.execute("ROLLBACK")
            except Exception:
                pass
            raise

    # ------------------------------------------------------------------
    # Article cache
    # ------------------------------------------------------------------

    def prune_article_cache(self, *, cutoff_ts: int) -> int:
        cur = self._conn.cursor()
        row = cur.execute("SELECT COUNT(1) AS n FROM article_cache WHERE fetched_at_ts < ?", (int(cutoff_ts),)).fetchone()
        n = int(row["n"]) if row else 0
        cur.execute("DELETE FROM article_cache WHERE fetched_at_ts < ?", (int(cutoff_ts),))
        return n

    def get_article_cache(self, *, norm_url: str) -> dict[str, Any] | None:
        cur = self._conn.cursor()
        row = cur.execute(
            """
            SELECT
              norm_url, url, final_url, domain, http_status,
              extracted_title, extracted_text, extractor,
              fetched_at_ts, text_chars, quality_score, error
            FROM article_cache
            WHERE norm_url = ?
            """,
            (str(norm_url),),
        ).fetchone()
        return dict(row) if row else None

    def upsert_article_cache(
        self,
        *,
        norm_url: str,
        url: str,
        final_url: str | None,
        domain: str | None,
        http_status: int | None,
        extracted_title: str | None,
        extracted_text: str | None,
        extractor: str | None,
        fetched_at_ts: int,
        quality_score: int,
        error: str | None,
    ) -> None:
        extracted_text = extracted_text if isinstance(extracted_text, str) else None
        text_chars = len(extracted_text) if extracted_text else 0
        self._conn.execute(
            """
            INSERT INTO article_cache(
              norm_url, url, final_url, domain, http_status,
              extracted_title, extracted_text, extractor,
              fetched_at_ts, text_chars, quality_score, error
            )
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(norm_url) DO UPDATE SET
              url=excluded.url,
              final_url=excluded.final_url,
              domain=excluded.domain,
              http_status=excluded.http_status,
              extracted_title=excluded.extracted_title,
              extracted_text=excluded.extracted_text,
              extractor=excluded.extractor,
              fetched_at_ts=excluded.fetched_at_ts,
              text_chars=excluded.text_chars,
              quality_score=excluded.quality_score,
              error=excluded.error
            """,
            (
                str(norm_url),
                str(url),
                str(final_url) if final_url else None,
                str(domain) if domain else None,
                int(http_status) if http_status is not None else None,
                str(extracted_title) if extracted_title else None,
                extracted_text,
                str(extractor) if extractor else None,
                int(fetched_at_ts),
                int(text_chars),
                int(quality_score),
                str(error) if error else None,
            ),
        )

    # ------------------------------------------------------------------
    # Semantic keys (legacy — kept for backward compat during transition)
    # ------------------------------------------------------------------

    def get_semantic_key(self, *, norm_url: str) -> dict[str, Any] | None:
        """Get semantic key — returns None if table doesn't exist (v5 fresh DB)."""
        try:
            cur = self._conn.cursor()
            row = cur.execute(
                "SELECT norm_url, semantic_event_key, semantic_fingerprint, model, created_at_ts, error FROM semantic_keys WHERE norm_url = ?",
                (str(norm_url),),
            ).fetchone()
            return dict(row) if row else None
        except sqlite3.OperationalError:
            return None

    def upsert_semantic_key(
        self,
        *,
        norm_url: str,
        semantic_event_key: str | None,
        semantic_fingerprint: str | None,
        model: str | None,
        created_at_ts: int,
        error: str | None,
    ) -> None:
        """Upsert semantic key — no-op if table doesn't exist (v5 fresh DB)."""
        try:
            self._conn.execute(
                """
                INSERT INTO semantic_keys(
                  norm_url, semantic_event_key, semantic_fingerprint, model, created_at_ts, error
                )
                VALUES(?, ?, ?, ?, ?, ?)
                ON CONFLICT(norm_url) DO UPDATE SET
                  semantic_event_key=excluded.semantic_event_key,
                  semantic_fingerprint=excluded.semantic_fingerprint,
                  model=excluded.model,
                  created_at_ts=excluded.created_at_ts,
                  error=excluded.error
                """,
                (
                    str(norm_url),
                    str(semantic_event_key) if semantic_event_key else None,
                    str(semantic_fingerprint) if semantic_fingerprint else None,
                    str(model) if model else None,
                    int(created_at_ts),
                    str(error) if error else None,
                ),
            )
        except sqlite3.OperationalError:
            pass

    def prune_semantic_keys(self, *, cutoff_ts: int) -> int:
        try:
            cur = self._conn.cursor()
            row = cur.execute("SELECT COUNT(1) AS n FROM semantic_keys WHERE created_at_ts < ?", (int(cutoff_ts),)).fetchone()
            n = int(row["n"]) if row else 0
            cur.execute("DELETE FROM semantic_keys WHERE created_at_ts < ?", (int(cutoff_ts),))
            return n
        except sqlite3.OperationalError:
            return 0

    # ------------------------------------------------------------------
    # Legacy event state (old events table — compat shims)
    # ------------------------------------------------------------------

    def get_event_state(self, *, event_key: str) -> dict[str, int] | None:
        """Get legacy event state — returns None if events_legacy doesn't exist."""
        try:
            cur = self._conn.cursor()
            row = cur.execute(
                "SELECT first_seen_ts, last_seen_ts, last_cluster_size, last_indexed_ts FROM events_legacy WHERE event_key=?",
                (event_key,),
            ).fetchone()
            if not row:
                return None
            return {
                "first_seen_ts": int(row["first_seen_ts"]),
                "last_seen_ts": int(row["last_seen_ts"]),
                "last_cluster_size": int(row["last_cluster_size"]),
                "last_indexed_ts": int(row["last_indexed_ts"]),
            }
        except sqlite3.OperationalError:
            return None

    def upsert_event_state(self, *, event_key: str, first_seen_ts: int, last_seen_ts: int, cluster_size: int, indexed_ts: int) -> None:
        """Upsert legacy event state — no-op if events_legacy doesn't exist."""
        try:
            self._conn.execute(
                """
                INSERT INTO events_legacy(event_key, first_seen_ts, last_seen_ts, last_cluster_size, last_indexed_ts)
                VALUES(?, ?, ?, ?, ?)
                ON CONFLICT(event_key) DO UPDATE SET
                  last_seen_ts=excluded.last_seen_ts,
                  last_cluster_size=excluded.last_cluster_size,
                  last_indexed_ts=excluded.last_indexed_ts
                """,
                (event_key, int(first_seen_ts), int(last_seen_ts), int(cluster_size), int(indexed_ts)),
            )
        except sqlite3.OperationalError:
            pass

    # ------------------------------------------------------------------
    # Pool runs
    # ------------------------------------------------------------------

    def log_pool_run(
        self,
        *,
        run_ts: int,
        state_key: str,
        window_hours: int,
        should_fetch: bool,
        query: str | None,
        offset_start: int | None,
        pages: int | None,
        count: int | None,
        freshness: str | None,
        requests_made: int,
        results: int,
        inserted: int,
        updated: int,
        pruned: int,
        pruned_articles: int,
        notes: str | None = None,
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO pool_runs(
              run_ts, state_key, window_hours, should_fetch,
              query, offset_start, pages, count, freshness,
              requests_made, results, inserted, updated, pruned, pruned_articles, notes
            )
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(run_ts),
                str(state_key),
                int(window_hours),
                1 if should_fetch else 0,
                str(query) if query else None,
                int(offset_start) if offset_start is not None else None,
                int(pages) if pages is not None else None,
                int(count) if count is not None else None,
                str(freshness) if freshness else None,
                int(max(0, requests_made)),
                int(max(0, results)),
                int(max(0, inserted)),
                int(max(0, updated)),
                int(max(0, pruned)),
                int(max(0, pruned_articles)),
                str(notes) if notes else None,
            ),
        )

    def sum_requests_made(self, *, since_ts: int, state_key: str | None = None) -> int:
        cur = self._conn.cursor()
        if state_key:
            row = cur.execute(
                "SELECT COALESCE(SUM(requests_made), 0) AS n FROM pool_runs WHERE run_ts >= ? AND state_key = ?",
                (int(since_ts), str(state_key)),
            ).fetchone()
        else:
            row = cur.execute(
                "SELECT COALESCE(SUM(requests_made), 0) AS n FROM pool_runs WHERE run_ts >= ?",
                (int(since_ts),),
            ).fetchone()
        return int(row["n"]) if row else 0

    def requests_made_by_state(self, *, since_ts: int) -> list[tuple[str, int]]:
        cur = self._conn.cursor()
        rows = cur.execute(
            """
            SELECT state_key, COALESCE(SUM(requests_made), 0) AS n
            FROM pool_runs
            WHERE run_ts >= ?
            GROUP BY state_key
            ORDER BY n DESC, state_key ASC
            """,
            (int(since_ts),),
        ).fetchall()
        out: list[tuple[str, int]] = []
        for r in rows:
            out.append((str(r["state_key"]), int(r["n"])))
        return out

    @staticmethod
    def dumps_compact(obj: dict[str, Any]) -> str:
        return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
