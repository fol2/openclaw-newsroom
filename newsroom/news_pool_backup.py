from __future__ import annotations

"""JSONL dump/restore utilities for the newsroom SQLite news pool.

This module exists to support reproducible clustering experiments without touching
the production database.  It provides two operations:

1) Dump a time-windowed snapshot of `links` and `events` to JSONL.
2) Restore that JSONL into a fresh SQLite database (with the current schema).

The dump uses a read-only SQLite connection and runs inside a single transaction
to ensure the `links` and `events` exports are mutually consistent.
"""

import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any

from newsroom.news_pool_db import NewsPoolDB


def _jsonl_write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            f.write("\n")


def _jsonl_read(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            out.append(json.loads(s))
    return out


def _sqlite_ro_connect(path: Path) -> sqlite3.Connection:
    # Using mode=ro ensures we never create a new DB by accident.
    uri = path.resolve().as_uri() + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def dump_news_pool_jsonl(
    *,
    source_db: Path,
    out_dir: Path,
    links_window_hours: int = 48,
    events_max_age_hours: int = 168,
    now_ts: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Dump `links` and `events` to JSONL under out_dir.

    Args:
        source_db: Existing SQLite database path.
        out_dir: Output directory. Files written: links.jsonl, events.jsonl, manifest.json.
        links_window_hours: Export links where last_seen_ts is within this window.
        events_max_age_hours: Export events where created_at_ts is within this window.
            Events referenced by the exported links (and their parent chain) are always included.
        now_ts: Override 'now' for deterministic dumps/tests.
        overwrite: If True, allow overwriting existing output files.

    Returns:
        Manifest dict written to manifest.json.
    """
    source_db = Path(source_db).expanduser()
    out_dir = Path(out_dir).expanduser()

    if not source_db.exists():
        raise FileNotFoundError(f"Source DB does not exist: {source_db}")

    if links_window_hours <= 0:
        raise ValueError("links_window_hours must be > 0")
    if events_max_age_hours <= 0:
        raise ValueError("events_max_age_hours must be > 0")

    links_path = out_dir / "links.jsonl"
    events_path = out_dir / "events.jsonl"
    manifest_path = out_dir / "manifest.json"

    if not overwrite:
        for p in (links_path, events_path, manifest_path):
            if p.exists():
                raise FileExistsError(f"Refusing to overwrite existing dump file: {p}")

    now = int(time.time()) if now_ts is None else int(now_ts)
    link_cutoff_ts = now - int(links_window_hours) * 3600
    event_cutoff_ts = now - int(events_max_age_hours) * 3600

    with _sqlite_ro_connect(source_db) as conn:
        # Single snapshot for consistency across tables.
        conn.execute("BEGIN;")

        schema_version = None
        try:
            row = conn.execute("SELECT v FROM meta WHERE k='schema_version'").fetchone()
            schema_version = int(row["v"]) if row and row["v"] is not None else None
        except Exception:
            schema_version = None

        # Links (time-windowed).
        links_rows: list[dict[str, Any]] = []
        referenced_event_ids: set[int] = set()
        for r in conn.execute(
            "SELECT * FROM links WHERE last_seen_ts >= ? ORDER BY id ASC",
            (int(link_cutoff_ts),),
        ).fetchall():
            obj = dict(r)
            links_rows.append(obj)
            eid = obj.get("event_id")
            if isinstance(eid, int):
                referenced_event_ids.add(int(eid))

        # Events within window.
        events_by_id: dict[int, dict[str, Any]] = {}
        for r in conn.execute(
            "SELECT * FROM events WHERE created_at_ts >= ? ORDER BY id ASC",
            (int(event_cutoff_ts),),
        ).fetchall():
            obj = dict(r)
            eid = obj.get("id")
            if isinstance(eid, int):
                events_by_id[int(eid)] = obj

        # Ensure referenced events (and parent chain) exist in the dump even if older.
        missing: set[int] = {eid for eid in referenced_event_ids if eid not in events_by_id}
        while missing:
            eid = missing.pop()
            row = conn.execute("SELECT * FROM events WHERE id = ?", (int(eid),)).fetchone()
            if not row:
                raise RuntimeError(f"links references missing event_id={eid} in {source_db}")
            obj = dict(row)
            events_by_id[int(eid)] = obj

            pid = obj.get("parent_event_id")
            if isinstance(pid, int) and int(pid) not in events_by_id:
                missing.add(int(pid))

        # Expand any remaining parents from window events too.
        expanded = True
        while expanded:
            expanded = False
            for ev in list(events_by_id.values()):
                pid = ev.get("parent_event_id")
                if not isinstance(pid, int):
                    continue
                pid_i = int(pid)
                if pid_i in events_by_id:
                    continue
                row = conn.execute("SELECT * FROM events WHERE id = ?", (pid_i,)).fetchone()
                if not row:
                    raise RuntimeError(f"events references missing parent_event_id={pid_i} in {source_db}")
                events_by_id[pid_i] = dict(row)
                expanded = True

        conn.execute("COMMIT;")

    # Deterministic file order.
    events_rows = [events_by_id[eid] for eid in sorted(events_by_id)]

    out_dir.mkdir(parents=True, exist_ok=True)
    _jsonl_write(links_path, links_rows)
    _jsonl_write(events_path, events_rows)

    manifest = {
        "ok": True,
        "dumped_at_ts": now,
        "source_db": str(source_db),
        "schema_version": schema_version,
        "links_window_hours": int(links_window_hours),
        "events_max_age_hours": int(events_max_age_hours),
        "counts": {"links": len(links_rows), "events": len(events_rows)},
        "files": {
            "links_jsonl": str(links_path),
            "events_jsonl": str(events_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def restore_news_pool_jsonl(
    *,
    dump_dir: Path,
    target_db: Path,
    overwrite: bool = False,
    validate_foreign_keys: bool = True,
) -> dict[str, Any]:
    """Restore a dump directory produced by dump_news_pool_jsonl() into target_db.

    The restore is implemented as a write to a temporary DB followed by an atomic
    rename into place. This avoids leaving a partially-written SQLite file on disk
    if the restore fails partway through.
    """
    dump_dir = Path(dump_dir).expanduser()
    target_db = Path(target_db).expanduser()

    links_path = dump_dir / "links.jsonl"
    events_path = dump_dir / "events.jsonl"
    manifest_path = dump_dir / "manifest.json"

    if not links_path.exists():
        raise FileNotFoundError(f"Missing dump file: {links_path}")
    if not events_path.exists():
        raise FileNotFoundError(f"Missing dump file: {events_path}")

    if target_db.exists() and not overwrite:
        raise FileExistsError(f"Target DB exists (pass --overwrite to replace): {target_db}")

    target_db.parent.mkdir(parents=True, exist_ok=True)

    # Use a temp path in the same directory so os.replace() stays atomic.
    tmp_db = target_db.with_name(target_db.name + f".tmp.{os.getpid()}.{int(time.time())}")
    if tmp_db.exists():
        tmp_db.unlink()

    # Best-effort cleanup of sidecar WAL/SHM files when overwriting.
    if target_db.exists() and overwrite:
        for sidecar in (Path(str(target_db) + "-wal"), Path(str(target_db) + "-shm")):
            try:
                sidecar.unlink()
            except FileNotFoundError:
                pass

    try:
        with NewsPoolDB(path=tmp_db) as db:
            conn = db._conn
            cur = conn.cursor()
            cur.execute("BEGIN;")

            # Restore events first (self-FK parent_event_id).
            events = _jsonl_read(events_path)
            events_by_id: dict[int, dict[str, Any]] = {}
            for row in events:
                eid = row.get("id")
                if not isinstance(eid, int):
                    raise ValueError("events.jsonl row missing integer 'id'")
                events_by_id[int(eid)] = row

            ev_cols = [r["name"] for r in cur.execute("PRAGMA table_info(events)").fetchall()]
            ev_cols_set = set(ev_cols)
            ev_insert_cols = [c for c in ev_cols if c in ev_cols_set]  # keep order
            ev_sql = f"INSERT INTO events({','.join(ev_insert_cols)}) VALUES ({','.join(['?' for _ in ev_insert_cols])});"

            remaining = set(events_by_id.keys())
            inserted: set[int] = set()
            while remaining:
                progressed = False
                for eid in sorted(remaining):
                    row = events_by_id[eid]
                    pid = row.get("parent_event_id")
                    if pid is None:
                        pass
                    elif not isinstance(pid, int):
                        raise ValueError(f"events.id={eid} has non-integer parent_event_id")
                    elif int(pid) not in events_by_id:
                        raise ValueError(f"events.id={eid} references missing parent_event_id={pid}")
                    elif int(pid) not in inserted:
                        continue  # parent not inserted yet

                    values = [row.get(c) for c in ev_insert_cols]
                    cur.execute(ev_sql, values)
                    inserted.add(eid)
                    remaining.remove(eid)
                    progressed = True
                    break
                if not progressed:
                    raise RuntimeError(f"Could not resolve parent_event_id order for events: {sorted(remaining)[:10]}")

            # Restore links after events.
            link_cols = [r["name"] for r in cur.execute("PRAGMA table_info(links)").fetchall()]
            link_sql = f"INSERT INTO links({','.join(link_cols)}) VALUES ({','.join(['?' for _ in link_cols])});"

            with links_path.open("r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if not s:
                        continue
                    row = json.loads(s)
                    if not isinstance(row, dict):
                        raise ValueError("links.jsonl must contain JSON objects per line")
                    values = [row.get(c) for c in link_cols]
                    cur.execute(link_sql, values)

            if validate_foreign_keys:
                # This returns one row per violation. We only need to know if any exist.
                bad = cur.execute("PRAGMA foreign_key_check;").fetchone()
                if bad:
                    raise RuntimeError(f"Foreign key check failed: {bad}")

            cur.execute("COMMIT;")

        # Replace into place.
        os.replace(str(tmp_db), str(target_db))

        # Best-effort remove sidecar files for the replaced DB path.
        for sidecar in (Path(str(target_db) + "-wal"), Path(str(target_db) + "-shm")):
            try:
                sidecar.unlink()
            except FileNotFoundError:
                pass

        manifest = None
        try:
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            manifest = None

        return {
            "ok": True,
            "dump_dir": str(dump_dir),
            "target_db": str(target_db),
            "manifest": manifest,
        }
    finally:
        try:
            if tmp_db.exists():
                tmp_db.unlink()
        except Exception:
            pass

