"""Provider-free Mini RAM/CPU packet for issue #898.

Stdlib-only at import time so baseline children are not contaminated.
Does not claim queue events, write canonical stores, or change product code.
Research-only Rust lives under docs/research/issue-898-ram-cpu-rust/.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import resource
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
import tracemalloc
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Callable

UNOBSERVED = "UNOBSERVED"
ISSUE = 898
CLOCK = datetime(2026, 8, 20, tzinfo=UTC)
FETCHED_AT = "2026-08-16T21:41:34.000000Z"
RIGHTS_ASSESSED_AT = "2026-08-20T00:00:00.000000Z"
RIGHTS_EXPIRES_AT = "2099-01-01T00:00:00.000000Z"
WARMUPS = 1
MEASURED_RUNS = 3
MIB = 1024 * 1024
GO_PEAK_RATIO = 0.20
GO_PEAK_BYTES = 64 * MIB
RAW_HTTP_RETENTION = timedelta(days=7)
SCAN_SCHEMA = "newsroom.issue-898.observation-scan.v1"
R2_SCHEMA = "newsroom.issue-898.bounded-units.v1"
R2_SPEC_ALLOWED = frozenset(
    {
        "configuration_digest",
        "item_key",
        "keys",
        "published_at",
        "revision_digest",
        "schema",
        "source_id",
        "temporal_policy_version",
        "updated_at",
    }
)
R2_SPEC_FORBIDDEN = frozenset(
    {
        "authority_record_ids",
        "chunk_digest",
        "chunk_ordinal",
        "ingest_id",
        "observation_digest",
        "predecessor_ingest_id",
        "proving_run_id",
        "representation_digest",
        "revision_id",
        "unit_refs",
    }
)
RUST_CRATE = (
    Path(__file__).resolve().parents[2] / "docs" / "research" / "issue-898-ram-cpu-rust"
)
RUST_TARGET = Path("/tmp/newsroom-898-rust-target")

ATOM = (
    b'<?xml version="1.0" encoding="UTF-8"?>'
    b'<feed xmlns="http://www.w3.org/2005/Atom">'
    b"<entry><id>urn:example:1</id><title>Home Office update</title>"
    b'<link href="https://www.gov.uk/example-1"/>'
    b"<summary>A retained proving item.</summary></entry></feed>"
)
JSON_DOC = (
    b'{"title":"BNO visa","base_path":"/british-national-overseas-bno-visa",'
    b'"content_id":"abc","description":"Apply for a visa."}'
)


def rss_body(*, guid: str, title: str, description: str) -> bytes:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<rss version=\"2.0\"><channel><item>"
        f"<guid>{guid}</guid><title>{title}</title>"
        "<link>https://www.news.gov.hk/a</link>"
        f"<description>{description}</description>"
        "</item></channel></rss>"
    ).encode("utf-8")


def utc_text(value: datetime) -> str:
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def raw_http_cutoff(evaluated_at: datetime) -> str:
    return utc_text(evaluated_at - RAW_HTTP_RETENTION)


def canonical_json(value: object) -> str:
    if isinstance(value, dict):
        parts = []
        for key in sorted(value):
            parts.append(json.dumps(key, ensure_ascii=False) + ":" + canonical_json(value[key]))
        return "{" + ",".join(parts) + "}"
    if isinstance(value, list):
        return "[" + ",".join(canonical_json(item) for item in value) + "]"
    return json.dumps(value, ensure_ascii=False)


def digest_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def current_rss_bytes() -> int | str:
    try:
        out = subprocess.check_output(
            ["ps", "-o", "rss=", "-p", str(os.getpid())],
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return UNOBSERVED
    text = out.strip().split()
    if not text:
        return UNOBSERVED
    return int(text[0]) * 1024


def rss_bytes_for_pid(pid: int) -> int | str:
    try:
        out = subprocess.check_output(
            ["ps", "-o", "rss=", "-p", str(pid)],
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return UNOBSERVED
    text = out.strip().split()
    if not text:
        return UNOBSERVED
    return int(text[0]) * 1024


def maxrss_bytes(raw: int) -> int:
    if sys.platform == "darwin":
        return int(raw)
    return int(raw) * 1024


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def canonical_store_paths() -> tuple[Path, Path]:
    from newsroom.control_plane.paths import (
        CANONICAL_PROVING_STORE,
        CANONICAL_UNPUBLISHED_STORE,
    )

    return CANONICAL_PROVING_STORE.resolve(), CANONICAL_UNPUBLISHED_STORE.resolve()


def refuse_canonical_store(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    proving, unpublished = canonical_store_paths()
    if resolved in {proving, unpublished}:
        raise RuntimeError(f"refusing canonical store path: {resolved}")
    return resolved


def sqlite_backup_snapshot(source: Path, destination: Path) -> dict[str, object]:
    """Copy one SQLite database through the backup API. Never reuse by size."""

    refuse_canonical_store(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    for suffix in ("-wal", "-shm", "-journal"):
        Path(str(destination) + suffix).unlink(missing_ok=True)
    if not source.is_file():
        return {"status": UNOBSERVED, "reason": "source sqlite missing"}
    wal = Path(str(source) + "-wal")
    shm = Path(str(source) + "-shm")
    journal = Path(str(source) + "-journal")
    wal_state = {
        "journal_exists": journal.exists(),
        "shm_exists": shm.exists(),
        "wal_bytes": wal.stat().st_size if wal.exists() else 0,
        "wal_exists": wal.exists(),
    }
    source_connection = sqlite3.connect(
        f"{source.resolve().as_uri()}?mode=ro",
        uri=True,
    )
    destination_connection = sqlite3.connect(destination)
    try:
        page_size = int(source_connection.execute("PRAGMA page_size").fetchone()[0])
        page_count = int(source_connection.execute("PRAGMA page_count").fetchone()[0])
        journal_mode = str(source_connection.execute("PRAGMA journal_mode").fetchone()[0])
        source_connection.backup(destination_connection)
        destination_connection.commit()
        check = destination_connection.execute("PRAGMA quick_check").fetchone()
        if check is None or str(check[0]) != "ok":
            raise RuntimeError("sqlite backup quick_check failed")
        observation_count = int(
            destination_connection.execute(
                "SELECT COUNT(*) FROM proving_observations"
            ).fetchone()[0]
        )
        body_bytes = int(
            destination_connection.execute(
                "SELECT COALESCE(SUM(LENGTH(body)),0) FROM proving_observations"
            ).fetchone()[0]
        )
        schema_sql = [
            str(row[0])
            for row in destination_connection.execute(
                "SELECT sql FROM sqlite_master WHERE sql IS NOT NULL ORDER BY name"
            )
        ]
    finally:
        destination_connection.close()
        source_connection.close()
    return {
        "body_bytes": body_bytes,
        "copy_digest": _sha256_file(destination),
        "journal_mode": journal_mode,
        "method": "sqlite3.Connection.backup",
        "observation_count": observation_count,
        "page_count": page_count,
        "page_size": page_size,
        "reused_existing_copy": False,
        "schema_digest": digest_text("\n".join(schema_sql)),
        "snapshot_bytes": destination.stat().st_size,
        "source_path_name": source.name,
        "source_size_bytes": source.stat().st_size,
        "status": "COPIED",
        "wal_state": wal_state,
        "writable_canonical": False,
    }


def backup_canonical_proving(destination: Path) -> dict[str, object]:
    proving, _unpublished = canonical_store_paths()
    if not proving.is_file():
        return {"status": UNOBSERVED, "reason": "canonical proving store missing"}
    return sqlite_backup_snapshot(proving, destination)


def _rights_packet(source_id: str) -> dict[str, object]:
    from newsroom.graphiti_adapter.evaluation_packet import (
        GRAPHITI_EVALUATION_DESTINATION_TOKENS,
    )
    from newsroom.increment9.rights import FIXTURE_DESTINATIONS, fixture_inventory

    return fixture_inventory(
        gate=f"RIGHTS_{source_id}",
        destinations=tuple(sorted({*FIXTURE_DESTINATIONS, *GRAPHITI_EVALUATION_DESTINATION_TOKENS})),
        now=RIGHTS_ASSESSED_AT,
        issued_at="2026-01-01T00:00:00.000000Z",
        expires_at=RIGHTS_EXPIRES_AT,
    )


def write_proving_store(path: Path, rows: tuple[tuple[str, bytes], ...]) -> None:
    from newsroom.authority.canonical import canonical_json_bytes, digest_bytes
    from newsroom.effective_revision import (
        create_effective_revision_schema,
        retain_observation_revision_first_seen,
    )
    from newsroom.increment9.proving import PROVING_GATES, SOURCE_URLS

    refuse_canonical_store(path)
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE proving_runs(
            run_id TEXT PRIMARY KEY,
            started_at TEXT NOT NULL,
            publication INTEGER NOT NULL DEFAULT 0,
            public_dispatch INTEGER NOT NULL DEFAULT 0,
            openrouter_invoked INTEGER NOT NULL DEFAULT 0,
            spend_gbp_minor INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE proving_observations(
            source_id TEXT NOT NULL,
            run_id TEXT NOT NULL,
            fetched_at TEXT NOT NULL,
            url TEXT NOT NULL,
            status_code INTEGER NOT NULL,
            body_digest TEXT NOT NULL,
            body BLOB NOT NULL,
            item_count INTEGER NOT NULL,
            error TEXT
        );
        CREATE TABLE proving_gates(
            run_id TEXT NOT NULL,
            gate_id TEXT NOT NULL,
            status TEXT NOT NULL,
            reason TEXT NOT NULL,
            PRIMARY KEY(run_id, gate_id)
        );
        CREATE TABLE proving_rights_packets(
            run_id TEXT NOT NULL,
            gate_id TEXT NOT NULL,
            packet_digest TEXT NOT NULL,
            packet_json TEXT NOT NULL,
            assessed_at TEXT NOT NULL,
            PRIMARY KEY(run_id, gate_id)
        );
        """
    )
    create_effective_revision_schema(connection)
    connection.execute(
        """
        INSERT INTO proving_runs(
            run_id, started_at, publication, public_dispatch,
            openrouter_invoked, spend_gbp_minor
        ) VALUES('run-1', ?, 0, 0, 0, 0)
        """,
        (FETCHED_AT,),
    )
    sources: list[str] = []
    for source_id, body in rows:
        url = SOURCE_URLS[source_id]
        connection.execute(
            "INSERT INTO proving_observations VALUES(?,?,?,?,?,?,?,?,?)",
            (
                source_id,
                "run-1",
                FETCHED_AT,
                url,
                200,
                digest_bytes(body),
                body,
                1,
                None,
            ),
        )
        retain_observation_revision_first_seen(
            connection,
            source_id=source_id,
            url=url,
            body=body,
            observed_at=FETCHED_AT,
        )
        if source_id not in sources:
            sources.append(source_id)
    for gate_id in PROVING_GATES:
        if gate_id.startswith("RIGHTS_"):
            continue
        connection.execute(
            "INSERT INTO proving_gates VALUES(?,?,?,?)",
            ("run-1", gate_id, "PASS", "fixture"),
        )
    for source_id in sources:
        gate_id = f"RIGHTS_{source_id}"
        connection.execute(
            "INSERT INTO proving_gates VALUES(?,?,?,?)",
            ("run-1", gate_id, "PASS", "fixture"),
        )
        packet = _rights_packet(source_id)
        packet_bytes = canonical_json_bytes(packet)
        connection.execute(
            "INSERT INTO proving_rights_packets VALUES(?,?,?,?,?)",
            (
                "run-1",
                gate_id,
                digest_bytes(packet_bytes),
                packet_bytes.decode("utf-8"),
                RIGHTS_ASSESSED_AT,
            ),
        )
    connection.commit()
    connection.close()


def write_unpublished_store(path: Path) -> None:
    from newsroom.control_plane.store import connect

    refuse_canonical_store(path)
    connection = connect(str(path))
    connection.close()


def measure(
    work: Callable[[], dict[str, object]],
    *,
    use_tracemalloc: bool = False,
) -> dict[str, object]:
    gc.collect()
    rss_before = current_rss_bytes()
    usage_before = resource.getrusage(resource.RUSAGE_SELF)
    if use_tracemalloc:
        tracemalloc.start()
    started = time.perf_counter()
    try:
        outcome = work()
        status = "OK"
    except Exception as exc:  # noqa: BLE001 — case outcome must stay bounded
        outcome = {"error": f"{type(exc).__name__}: {exc}"}
        status = "ERROR"
    wall = time.perf_counter() - started
    if use_tracemalloc:
        traced_current, traced_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    else:
        traced_current = traced_peak = UNOBSERVED
    gc.collect()
    usage_after = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "pid": os.getpid(),
        "status": status,
        "rss_before_bytes": rss_before,
        "rss_after_bytes": current_rss_bytes(),
        "ru_maxrss_bytes": maxrss_bytes(usage_after.ru_maxrss),
        "user_cpu_seconds": usage_after.ru_utime - usage_before.ru_utime,
        "system_cpu_seconds": usage_after.ru_stime - usage_before.ru_stime,
        "wall_seconds": wall,
        "tracemalloc_enabled": use_tracemalloc,
        "tracemalloc_peak_bytes": traced_peak,
        "tracemalloc_current_bytes": traced_current,
        "outcome": outcome,
    }


def _open_proving(path: str) -> sqlite3.Connection:
    from newsroom.control_plane.sqlite_profile import apply_control_plane_sqlite_profile

    refuse_canonical_store(path)
    connection = sqlite3.connect(path)
    apply_control_plane_sqlite_profile(connection, query_only=True)
    return connection


def scan_observations(path: str, cutoff: str) -> dict[str, object]:
    refuse_canonical_store(path)
    connection = sqlite3.connect(f"{Path(path).resolve().as_uri()}?mode=ro", uri=True)
    try:
        connection.execute("PRAGMA query_only=ON")
        run_ids = [
            str(row[0])
            for row in connection.execute(
                "SELECT run_id FROM proving_runs ORDER BY rowid ASC"
            )
        ]
        rows: list[dict[str, object]] = []
        bodies: list[bytes] = []
        for run_id in run_ids:
            values = connection.execute(
                """
                SELECT source_id, url, fetched_at, status_code, body_digest, body, error
                FROM proving_observations
                WHERE run_id=? AND fetched_at>=?
                ORDER BY source_id, fetched_at, body_digest
                """,
                (run_id, cutoff),
            )
            for source_id, url, fetched_at, status_code, body_digest, body, error in values:
                if int(status_code) != 200 or not body or error is not None:
                    continue
                body_bytes = bytes(body)
                bodies.append(body_bytes)
                rows.append(
                    {
                        "body_digest": str(body_digest),
                        "body_len": len(body_bytes),
                        "body_sha256": "sha256:" + hashlib.sha256(body_bytes).hexdigest(),
                        "fetched_at": str(fetched_at),
                        "run_id": str(run_id),
                        "source_id": str(source_id),
                        "status_code": int(status_code),
                        "url": str(url),
                    }
                )
        manifest = {
            "row_count": len(rows),
            "rows": rows,
            "schema": SCAN_SCHEMA,
        }
        digest = digest_text(canonical_json(manifest))
        body_bytes_total = sum(len(item) for item in bodies)
        del bodies
        return {
            "body_bytes": body_bytes_total,
            "cutoff": cutoff,
            "manifest_digest": digest,
            "queue_claimed": False,
            "row_count": len(rows),
            "schema": SCAN_SCHEMA,
            "writable": False,
        }
    finally:
        connection.close()


def prepare_event_identity(proving: str, clock: datetime) -> dict[str, object]:
    from newsroom.control_plane.cycle import load_graphiti_units
    from newsroom.control_plane.graphiti_events import graphiti_unit_refs

    units = load_graphiti_units(proving_store=proving, evaluated_at=clock)
    if not units:
        spec = {
            "clock": utc_text(clock),
            "event_id": "measure-event",
            "expected_selected_count": 0,
            "expected_unit_count": 0,
            "item_key": "missing",
            "landed_ingest_ids": [],
            "landed_payload_digest": "sha256:" + ("00" * 32),
            "ledger_seq": 1,
            "prep_unit_count": 0,
            "published_at": "",
            "revision_digest": "sha256:" + ("00" * 32),
            "source_id": "UK-01",
            "status": "EMPTY",
            "unit_refs": [],
            "updated_at": "",
        }
        spec["event_manifest_digest"] = digest_text(canonical_json(spec))
        return spec
    unit = units[0]
    identity = (
        unit.source_id,
        unit.item_key,
        unit.revision_digest,
        unit.published_at or "",
        unit.updated_at or "",
    )
    selected = tuple(
        item
        for item in units
        if (
            item.source_id,
            item.item_key,
            item.revision_digest,
            item.published_at or "",
            item.updated_at or "",
        )
        == identity
    )
    spec = {
        "clock": utc_text(clock),
        "event_id": "measure-event",
        "expected_selected_count": len(selected),
        "expected_unit_count": len(selected),
        "item_key": unit.item_key,
        "landed_ingest_ids": [item.ingest_id for item in selected],
        "landed_payload_digest": "sha256:" + ("00" * 32),
        "ledger_seq": 1,
        "prep_unit_count": len(units),
        "published_at": unit.published_at or "",
        "revision_digest": unit.revision_digest,
        "source_id": unit.source_id,
        "status": "OK",
        "unit_refs": graphiti_unit_refs(selected),
        "updated_at": unit.updated_at or "",
    }
    spec["event_manifest_digest"] = digest_text(canonical_json(spec))
    return spec


def _event_from_spec(spec: dict[str, object]) -> Any:
    from newsroom.control_plane.graphiti_events import GraphitiRevisionEvent

    return GraphitiRevisionEvent(
        event_id=str(spec.get("event_id") or "measure-event"),
        ledger_seq=int(spec.get("ledger_seq") or 1),
        source_id=str(spec["source_id"]),
        item_key=str(spec["item_key"]),
        revision_digest=str(spec["revision_digest"]),
        published_at=str(spec.get("published_at") or ""),
        updated_at=str(spec.get("updated_at") or ""),
        expected_unit_count=int(
            spec.get("expected_unit_count") or spec.get("expected_selected_count") or 0
        ),
        landed_ingest_ids=tuple(spec.get("landed_ingest_ids") or ()),
        landed_payload_digest=str(
            spec.get("landed_payload_digest") or ("sha256:" + ("00" * 32))
        ),
        unit_refs=tuple(spec.get("unit_refs") or ()),
        state="QUEUED",
        attempt_count=0,
        units=(),
    )


def _unit_manifest(units: tuple[Any, ...]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for unit in units:
        rows.append(
            {
                "chunk_count": unit.chunk_count,
                "chunk_digest": unit.digest,
                "chunk_ordinal": unit.chunk_ordinal,
                "ingest_id": unit.ingest_id,
                "item_key": unit.item_key,
                "predecessor_ingest_id": unit.predecessor_ingest_id,
                "published_at": unit.published_at or "",
                "representation_digest": unit.representation_digest,
                "revision_digest": unit.revision_digest,
                "source_id": unit.source_id,
                "updated_at": unit.updated_at or "",
            }
        )
    return rows


def bounded_observation_keys(spec: dict[str, object]) -> list[tuple[str, str, str]]:
    """Deduplicate exact proving-row coordinates from retained unit_refs."""

    source_id = str(spec.get("source_id") or "")
    keys: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for ref in spec.get("unit_refs") or ():
        if not isinstance(ref, dict):
            continue
        run_id = str(ref.get("proving_run_id") or "")
        digest = str(ref.get("observation_digest") or "")
        if not run_id or not digest or not source_id:
            continue
        key = (run_id, source_id, digest)
        if key in seen:
            continue
        seen.add(key)
        keys.append(key)
    return keys


def r2_spec_from_event(event: dict[str, object]) -> dict[str, object]:
    """Build the measured R2 input. Coordinates only; unit_ref identities stay oracle."""

    from newsroom.graphiti_adapter.identity import configuration_digest
    from newsroom.graphiti_adapter.temporal_vocabulary import TEMPORAL_POLICY_VERSION

    spec = {
        "configuration_digest": configuration_digest(),
        "item_key": str(event.get("item_key") or ""),
        "keys": [
            {"observation_digest": digest, "run_id": run_id}
            for run_id, _source_id, digest in bounded_observation_keys(event)
        ],
        "published_at": str(event.get("published_at") or ""),
        "revision_digest": str(event.get("revision_digest") or ""),
        "schema": R2_SCHEMA,
        "source_id": str(event.get("source_id") or ""),
        "temporal_policy_version": TEMPORAL_POLICY_VERSION,
        "updated_at": str(event.get("updated_at") or ""),
    }
    extra = set(spec) - R2_SPEC_ALLOWED
    leaked = set(spec) & R2_SPEC_FORBIDDEN
    if extra or leaked:
        raise ValueError(f"R2 spec must not carry oracle fields: {sorted(extra | leaked)}")
    return spec


def write_r2_spec(path: Path, event: dict[str, object]) -> dict[str, object]:
    spec = r2_spec_from_event(event)
    path.write_text(
        json.dumps(spec, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return spec


def _r2_unit_row(
    *,
    source_id: str,
    item_key: str,
    published_at: str | None,
    updated_at: str | None,
    observation_digest: str,
    proving_run_id: str,
    observed_at: str,
    headline: str,
    body: str,
    canonical_url: str,
) -> list[dict[str, object]]:
    from newsroom.control_plane.corpus import chunk_text
    from newsroom.graphiti_adapter.identity import (
        content_digest,
        ingest_key,
        representation_digest_for,
        source_revision_id,
    )

    revision_digest = content_digest(
        headline=headline, body=body, canonical_url=canonical_url
    )
    representation_digest = representation_digest_for(
        source_id=source_id,
        item_key=item_key,
        revision_digest=revision_digest,
        published_at=published_at,
        updated_at=updated_at,
    )
    revision_id = str(
        source_revision_id(
            source_id=source_id,
            item_key=item_key,
            revision_digest=revision_digest,
            published_at=published_at,
            updated_at=updated_at,
        )
    )
    full_text = "\n".join(
        part for part in (headline.strip(), body.strip(), canonical_url.strip()) if part
    )
    chunks = chunk_text(full_text)
    predecessor: str | None = None
    rows: list[dict[str, object]] = []
    for ordinal, chunk in enumerate(chunks, start=1):
        chunk_digest = content_digest(headline="", body=chunk, canonical_url="")
        ingest_id = ingest_key(
            source_id=source_id,
            item_key=item_key,
            content_digest_value=revision_digest,
            revision_id=revision_id,
            representation_digest=representation_digest,
            published_at=published_at,
            updated_at=updated_at,
            chunk_ordinal=ordinal,
        )
        rows.append(
            {
                "chunk_count": len(chunks),
                "chunk_digest": chunk_digest,
                "chunk_ordinal": ordinal,
                "ingest_id": ingest_id,
                "item_key": item_key,
                "observation_digest": observation_digest,
                "observed_at": observed_at,
                "predecessor_ingest_id": predecessor,
                "proving_run_id": proving_run_id,
                "published_at": published_at,
                "representation_digest": representation_digest,
                "revision_digest": revision_digest,
                "revision_id": revision_id,
                "source_id": source_id,
                "updated_at": updated_at,
            }
        )
        predecessor = ingest_id
    return rows


def _unique_r2_units(units: list[dict[str, object]]) -> list[dict[str, object]]:
    selected: dict[tuple[object, ...], dict[str, object]] = {}
    for unit in units:
        key = (
            unit["source_id"],
            unit["item_key"],
            unit["revision_digest"],
            unit.get("published_at") or "",
            unit.get("updated_at") or "",
            unit["chunk_ordinal"],
        )
        previous = selected.get(key)
        if previous is None or str(unit.get("observed_at") or "") < str(
            previous.get("observed_at") or ""
        ):
            selected[key] = unit
    return sorted(
        selected.values(),
        key=lambda item: (
            str(item.get("observed_at") or ""),
            str(item.get("revision_id") or ""),
            int(item["chunk_ordinal"]),
        ),
    )


def r2_oracle_match(
    units: list[dict[str, object]], unit_refs: object
) -> dict[str, object]:
    refs = [item for item in (unit_refs or ()) if isinstance(item, dict)]
    compared = (
        "chunk_digest",
        "chunk_ordinal",
        "ingest_id",
        "observation_digest",
        "predecessor_ingest_id",
        "proving_run_id",
        "representation_digest",
        "revision_id",
    )
    mismatches: list[dict[str, object]] = []
    if len(units) != len(refs):
        return {
            "match": False,
            "compared_fields": list(compared),
            "reason": f"unit count {len(units)} != unit_refs {len(refs)}",
            "unit_count": len(units),
            "unit_ref_count": len(refs),
        }
    for index, (unit, ref) in enumerate(zip(units, refs, strict=True)):
        for field in compared:
            if unit.get(field) != ref.get(field):
                mismatches.append(
                    {"index": index, "field": field, "computed": unit.get(field)}
                )
    return {
        "match": not mismatches,
        "compared_fields": list(compared),
        "mismatch_count": len(mismatches),
        "unit_count": len(units),
        "unit_ref_count": len(refs),
    }


def bounded_useful_units(proving: str, event: dict[str, object]) -> dict[str, object]:
    from newsroom.control_plane.items import parse_observation

    keys = bounded_observation_keys(event)
    if not keys:
        return {
            "body_bytes": 0,
            "manifest_digest": UNOBSERVED,
            "oracle": {
                "match": False,
                "reason": "prepared event has no unit_refs coordinates",
            },
            "queue_claimed": False,
            "reason": (
                "R2 has a retained unit_refs seam, but this event spec has no "
                "row coordinates. The missing-column HOLD is withdrawn; this "
                "run did not use the existing exact-row lookup."
            ),
            "row_count": 0,
            "schema": R2_SCHEMA,
            "status": "HOLD",
            "unit_count": 0,
            "writable": False,
        }
    refuse_canonical_store(proving)
    connection = sqlite3.connect(
        f"{Path(proving).resolve().as_uri()}?mode=ro", uri=True
    )
    try:
        connection.execute("PRAGMA query_only=ON")
        collected: list[dict[str, object]] = []
        body_bytes = 0
        rows_found = 0
        identity = (
            str(event.get("item_key") or ""),
            str(event.get("revision_digest") or ""),
            str(event.get("published_at") or ""),
            str(event.get("updated_at") or ""),
        )
        for run_id, source_id, digest in keys:
            row = connection.execute(
                """
                SELECT source_id, url, fetched_at, status_code, body_digest, body, error
                FROM proving_observations
                WHERE run_id=? AND source_id=? AND body_digest=?
                """,
                (run_id, source_id, digest),
            ).fetchone()
            if row is None:
                continue
            source_id_row, url, fetched_at, status_code, body_digest, body, error = row
            if int(status_code) != 200 or not body or error is not None:
                continue
            payload = bytes(body)
            body_bytes += len(payload)
            rows_found += 1
            for item in parse_observation(
                source_id=str(source_id_row), url=str(url), body=payload
            ):
                units = _r2_unit_row(
                    source_id=str(source_id_row),
                    item_key=item.item_key,
                    published_at=item.published_at,
                    updated_at=item.updated_at,
                    observation_digest=str(body_digest),
                    proving_run_id=run_id,
                    observed_at=str(fetched_at),
                    headline=item.headline,
                    body=item.retained_corpus_body,
                    canonical_url=item.canonical_url,
                )
                for unit in units:
                    if (
                        unit["item_key"],
                        unit["revision_digest"],
                        unit.get("published_at") or "",
                        unit.get("updated_at") or "",
                    ) == identity:
                        collected.append(unit)
        units = _unique_r2_units(collected)
        comparable = [
            {key: value for key, value in unit.items() if key != "observed_at"}
            for unit in units
        ]
        manifest = {
            "row_count": rows_found,
            "schema": R2_SCHEMA,
            "unit_count": len(comparable),
            "units": comparable,
        }
        oracle = r2_oracle_match(units, event.get("unit_refs"))
        status = "OK" if oracle["match"] and comparable else "HOLD"
        reason = None
        if not keys:
            reason = "no unit_refs coordinates"
        elif rows_found == 0:
            reason = "unit_refs coordinates did not match proving_observations rows"
            status = "HOLD"
        elif not comparable:
            reason = (
                "bounded rows were read, but parser/identity produced no units "
                "matching the prepared event identity"
            )
            status = "HOLD"
        elif not oracle["match"]:
            reason = (
                "bounded rows were read and units were computed from bodies, "
                "but the computed useful output does not match retained "
                "unit_refs (parser, identity, chunking, or ordering)"
            )
            status = "HOLD"
        return {
            "body_bytes": body_bytes,
            "manifest_digest": digest_text(canonical_json(manifest)),
            "oracle": oracle,
            "queue_claimed": False,
            "reason": reason,
            "row_count": rows_found,
            "schema": R2_SCHEMA,
            "status": status,
            "unit_count": len(comparable),
            "writable": False,
        }
    finally:
        connection.close()


def case_bare_interpreter() -> dict[str, object]:
    return measure(lambda: {"imported": False})


def case_import_cycle(*, use_tracemalloc: bool = False) -> dict[str, object]:
    def work() -> dict[str, object]:
        from newsroom.control_plane import cycle as cycle_module

        return {"module": cycle_module.__name__}

    return measure(work, use_tracemalloc=use_tracemalloc)


def case_instantiate_runner() -> dict[str, object]:
    def work() -> dict[str, object]:
        from newsroom.control_plane.graphiti import EvaluationGraphitiRunner

        runner = EvaluationGraphitiRunner()
        return {
            "runner": type(runner).__name__,
            "graphiti_core_imported": "graphiti_core" in sys.modules,
        }

    return measure(work)


def case_idle_worker() -> dict[str, object]:
    def work() -> dict[str, object]:
        import scripts.hermes_graphiti_worker as worker
        from newsroom.control_plane.graphiti import EvaluationGraphitiRunner

        runner = EvaluationGraphitiRunner()
        return {
            "worker_module": worker.__name__,
            "runner": type(runner).__name__,
            "graphiti_core_imported": "graphiti_core" in sys.modules,
        }

    return measure(work)


def case_hermes_cycle_import() -> dict[str, object]:
    def work() -> dict[str, object]:
        import scripts.hermes_control_plane as hermes
        from newsroom.control_plane.cycle import run_cycle

        return {
            "hermes_module": hermes.__name__,
            "run_cycle": run_cycle.__name__,
        }

    return measure(work)


def case_process_tree() -> dict[str, object]:
    needles = (
        "hermes_control_plane",
        "hermes_graphiti_worker",
        "newsroom-hub",
        "neo4j",
        "com.jamesto.newsroom",
    )
    try:
        out = subprocess.check_output(
            ["ps", "-axo", "pid=,rss=,command="],
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        return {
            "status": UNOBSERVED,
            "reason": f"{type(exc).__name__}: {exc}",
            "signalled": False,
        }
    rows: list[dict[str, object]] = []
    for line in out.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if not any(needle in stripped.lower() or needle in stripped for needle in needles):
            continue
        pid_text, rss_text, *command = stripped.split()
        try:
            pid = int(pid_text)
            rss_kib = int(rss_text)
        except ValueError:
            continue
        rows.append(
            {
                "pid": pid,
                "rss_bytes": rss_kib * 1024,
                "command": " ".join(command)[:200],
            }
        )
    return {
        "status": "OK",
        "signalled": False,
        "process_count": len(rows),
        "processes": rows,
        "largest_rss_bytes": max((int(item["rss_bytes"]) for item in rows), default=0),
    }


def case_load_units(proving: str, *, clock: datetime = CLOCK) -> dict[str, object]:
    def work() -> dict[str, object]:
        from newsroom.control_plane.cycle import load_graphiti_units

        units = load_graphiti_units(proving_store=proving, evaluated_at=clock)
        return {
            "unit_count": len(units),
            "source_count": len({unit.source_id for unit in units}),
            "chunk_count": len(units),
            "queue_claimed": False,
        }

    return measure(work)


def case_resolve_event(
    proving: str,
    event: dict[str, object],
    *,
    use_tracemalloc: bool = False,
) -> dict[str, object]:
    def work() -> dict[str, object]:
        from newsroom.control_plane.cycle import _resolve_graphiti_event_units

        clock = parse_utc(str(event["clock"]))
        selected = _resolve_graphiti_event_units(
            proving_store=proving,
            event=_event_from_spec(event),
            evaluated_at=clock,
        )
        manifest = _unit_manifest(selected)
        return {
            "event_manifest_digest": event.get("event_manifest_digest"),
            "queue_claimed": False,
            "selected_unit_count": len(selected),
            "source_id": event.get("source_id"),
            "unit_count": event.get("prep_unit_count"),
            "unit_manifest_digest": digest_text(canonical_json(manifest)),
            "unit_manifest": manifest,
        }

    return measure(work, use_tracemalloc=use_tracemalloc)


def case_permitted_rows(proving: str, *, clock: datetime = CLOCK) -> dict[str, object]:
    def work() -> dict[str, object]:
        from newsroom.control_plane.cycle import _permitted_rows, _utc_text

        connection = _open_proving(proving)
        try:
            run_id, latest, corpus = _permitted_rows(
                connection,
                evaluated_at=_utc_text(clock),
            )
        finally:
            connection.close()
        body_bytes = sum(len(row.body) for row in corpus)
        return {
            "run_id_present": bool(run_id),
            "latest_row_count": len(latest),
            "corpus_row_count": len(corpus),
            "body_bytes": body_bytes,
        }

    return measure(work)


def case_parsed_observations(proving: str, *, clock: datetime = CLOCK) -> dict[str, object]:
    def work() -> dict[str, object]:
        from newsroom.control_plane.cycle import (
            _parsed_observations,
            _permitted_rows,
            _utc_text,
        )

        connection = _open_proving(proving)
        try:
            _run_id, _latest, corpus = _permitted_rows(
                connection,
                evaluated_at=_utc_text(clock),
            )
            parsed = _parsed_observations(corpus)
        finally:
            connection.close()
        return {
            "corpus_row_count": len(corpus),
            "parsed_item_count": len(parsed),
            "body_bytes": sum(len(row.body) for row in corpus),
        }

    return measure(work)


def case_units_from(proving: str, *, clock: datetime = CLOCK) -> dict[str, object]:
    def work() -> dict[str, object]:
        from newsroom.control_plane.corpus import units_from
        from newsroom.control_plane.cycle import (
            _parsed_observations,
            _permitted_rows,
            _utc_text,
        )
        from newsroom.effective_revision import EffectiveRevisionIdentityResolver

        connection = _open_proving(proving)
        try:
            _run_id, _latest, corpus = _permitted_rows(
                connection,
                evaluated_at=_utc_text(clock),
            )
            resolver = EffectiveRevisionIdentityResolver(connection)
            collected: list[Any] = []
            for row in corpus:
                collected.extend(
                    units_from(
                        _parsed_observations((row,)),
                        proving_run_id=row.run_id,
                        rights_authority_run_id=row.rights_authority_run_id,
                        rights_gate_id=row.rights_gate_id,
                        rights_gate_reason=row.rights_gate_reason,
                        source_definition_url=row.url,
                        effective_revision_resolver=resolver,
                    )
                )
        finally:
            connection.close()
        return {"unit_count": len(collected), "corpus_row_count": len(corpus)}

    return measure(work)


def case_unique_and_revisions(proving: str, *, clock: datetime = CLOCK) -> dict[str, object]:
    def work() -> dict[str, object]:
        from newsroom.control_plane.corpus import revisions_from, unique_chunk_units
        from newsroom.control_plane.cycle import load_graphiti_units

        units = load_graphiti_units(proving_store=proving, evaluated_at=clock)
        unique = unique_chunk_units(units)
        revisions = revisions_from(unique)
        return {
            "unit_count": len(units),
            "unique_unit_count": len(unique),
            "revision_count": len(revisions),
        }

    return measure(work)


def case_admission_generation() -> dict[str, object]:
    def work() -> dict[str, object]:
        from newsroom.control_plane.graphiti_admission import (
            graphiti_admission_generation_identity,
        )

        digest = "sha256:" + ("ab" * 32)
        ingest_ids = ("ingest-a", "ingest-b")
        receipts = (
            {"ingest_id": "ingest-a", "receipt_digest": digest, "proposal_count": 1},
            {"ingest_id": "ingest-b", "receipt_digest": digest, "proposal_count": 1},
        )
        members = (
            {
                "ingest_id": "ingest-a",
                "proposal_key": "proposal-a",
                "proposal_envelope_id": "00000000-0000-4000-8000-000000000001",
                "decision_digest": digest,
                "decision": {"action": "HOLD"},
            },
            {
                "ingest_id": "ingest-b",
                "proposal_key": "proposal-b",
                "proposal_envelope_id": "00000000-0000-4000-8000-000000000002",
                "decision_digest": digest,
                "decision": {"action": "HOLD"},
            },
        )
        cohort_digest, generation_id = graphiti_admission_generation_identity(
            ingest_ids=ingest_ids,
            source_receipts=receipts,
            members=members,
        )
        return {
            "cohort_digest": cohort_digest,
            "generation_id_present": bool(generation_id),
            "member_count": len(members),
        }

    return measure(work)


def case_cycle_max_writes_0(
    proving: str,
    unpublished: str | None = None,
    *,
    clock: datetime = CLOCK,
) -> dict[str, object]:
    owned_unpublished: Path | None = None
    if unpublished is None:
        handle = tempfile.NamedTemporaryFile(
            prefix="newsroom-898-unpub-",
            suffix=".sqlite3",
            delete=False,
        )
        handle.close()
        owned_unpublished = Path(handle.name)
        write_unpublished_store(owned_unpublished)
        unpublished = str(owned_unpublished)

    def work() -> dict[str, object]:
        from newsroom.control_plane.cycle import run_cycle
        from newsroom.control_plane.evidence import package_for
        from newsroom.control_plane.writer import FixtureWriter

        report = run_cycle(
            proving_store=proving,
            unpublished_store=unpublished,
            writer=FixtureWriter(),
            max_writes=0,
            graphiti=None,
            max_graphiti=0,
            max_writer_provider_dispatches=0,
            max_writer_fallback_dispatches=0,
            clock=lambda: clock,
            evidence_package_builder=package_for,
        )
        return {
            "poll_observation_count": report.poll_observation_count,
            "candidates": report.candidates,
            "minted": report.minted,
            "graphiti": report.graphiti,
            "write_ready": report.write_ready,
            "queue_claimed": False,
        }

    try:
        return measure(work)
    finally:
        if owned_unpublished is not None:
            owned_unpublished.unlink(missing_ok=True)
            for suffix in ("-wal", "-shm"):
                Path(str(owned_unpublished) + suffix).unlink(missing_ok=True)


def case_observation_scan(proving: str, cutoff: str) -> dict[str, object]:
    return measure(lambda: scan_observations(proving, cutoff))


def case_python_r2(proving: str, event: dict[str, object]) -> dict[str, object]:
    return measure(lambda: bounded_useful_units(proving, event))


def case_rust_r2(binary: str, proving: str, spec_path: str) -> dict[str, object]:
    return _run_rust_binary(binary, ["r2", "--db", proving, "--spec", spec_path])


def case_rust_r2_e2e(binary: str, proving: str, spec_path: str) -> dict[str, object]:
    def work() -> dict[str, object]:
        proc = subprocess.Popen(
            [binary, "r2", "--db", proving, "--spec", spec_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        combined_peak: int | str = 0
        while proc.poll() is None:
            parent = current_rss_bytes()
            child = rss_bytes_for_pid(proc.pid)
            if isinstance(parent, int) and isinstance(child, int):
                if combined_peak == 0 or (
                    isinstance(combined_peak, int) and parent + child > combined_peak
                ):
                    combined_peak = parent + child
            time.sleep(0.02)
        stdout, stderr = proc.communicate()
        try:
            payload = json.loads(stdout.strip().splitlines()[-1])
        except (json.JSONDecodeError, IndexError, ValueError):
            payload = {
                "status": "ERROR",
                "outcome": {
                    "error": "rust stdout was not JSON",
                    "stderr_tail": stderr[-1000:],
                },
            }
        outcome = payload.get("outcome", {})
        if not isinstance(outcome, dict):
            outcome = {"value": outcome}
        parent = current_rss_bytes()
        if isinstance(parent, int) and isinstance(combined_peak, int):
            combined_peak = max(combined_peak, parent)
        outcome = {
            **outcome,
            "child_status": payload.get("status"),
            "combined_peak_rss_bytes": combined_peak,
            "parent_rss_after_bytes": parent,
            "queue_claimed": False,
            "returncode": proc.returncode,
        }
        return outcome

    return measure(work)


def fixture_rows(kind: str) -> tuple[tuple[str, bytes], ...]:
    from newsroom.graphiti_adapter.identity import MAX_EPISODE_BYTES

    if kind == "solo":
        return (("UK-01", ATOM),)
    if kind == "representative":
        return (
            ("UK-01", ATOM),
            ("HK-01", rss_body(guid="hk-1", title="香港政府新聞", description="保留來源正文。")),
            ("UK-02", JSON_DOC),
        )
    if kind == "times10":
        extras = tuple(
            (
                "HK-01",
                rss_body(
                    guid=f"hk-extra-{index}",
                    title=f"Unrelated item {index}",
                    description=f"Source-safe unrelated observation {index}.",
                ),
            )
            for index in range(10)
        )
        return (("UK-01", ATOM),) + extras
    if kind == "scaled":
        extras = tuple(
            (
                "HK-01",
                rss_body(
                    guid=f"hk-scaled-{index}",
                    title=f"Scaled retained item {index}",
                    description=("D" * 8192),
                ),
            )
            for index in range(40)
        )
        return (("UK-01", ATOM),) + extras
    if kind == "json":
        return (("UK-02", JSON_DOC),)
    if kind == "rss":
        return (("HK-01", rss_body(guid="hk-1", title="香港政府新聞", description="保留來源正文。")),)
    if kind == "atom":
        return (("UK-01", ATOM),)
    if kind == "large":
        return (
            (
                "HK-01",
                rss_body(
                    guid="large-1",
                    title="Large retained body",
                    description="L" * MAX_EPISODE_BYTES,
                ),
            ),
        )
    if kind == "malformed":
        return (("UK-02", b"{not-json"),)
    if kind == "empty":
        return (("UK-01", b""),)
    raise ValueError(f"unknown fixture kind: {kind}")


def _write_event(path: Path, proving: str, clock: datetime) -> dict[str, object]:
    spec = prepare_event_identity(proving, clock)
    path.write_text(json.dumps(spec, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    write_r2_spec(path.with_name("r2-spec-" + path.name.removeprefix("event-")), spec)
    return spec


def build_workspace(root: Path) -> dict[str, str]:
    root.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    kinds = (
        "solo",
        "representative",
        "times10",
        "scaled",
        "json",
        "rss",
        "atom",
        "large",
        "malformed",
        "empty",
    )
    for kind in kinds:
        proving = root / f"proving-{kind}.sqlite3"
        unpublished = root / f"unpublished-{kind}.sqlite3"
        proving.unlink(missing_ok=True)
        unpublished.unlink(missing_ok=True)
        for suffix in ("-wal", "-shm"):
            Path(str(proving) + suffix).unlink(missing_ok=True)
            Path(str(unpublished) + suffix).unlink(missing_ok=True)
        write_proving_store(proving, fixture_rows(kind))
        write_unpublished_store(unpublished)
        paths[f"proving_{kind}"] = str(proving)
        paths[f"unpublished_{kind}"] = str(unpublished)
        _write_event(root / f"event-{kind}.json", str(proving), CLOCK)
        paths[f"event_{kind}"] = str(root / f"event-{kind}.json")
        paths[f"cutoff_{kind}"] = raw_http_cutoff(CLOCK)
    copied = root / "proving-copied.sqlite3"
    print("workspace: sqlite backup begin", flush=True)
    copy_meta = backup_canonical_proving(copied)
    print(f"workspace: sqlite backup {copy_meta.get('status')}", flush=True)
    paths["copy_meta"] = json.dumps(copy_meta, sort_keys=True)
    if copy_meta.get("status") == "COPIED":
        paths["proving_copied"] = str(copied)
        unpublished_copied = root / "unpublished-copied.sqlite3"
        write_unpublished_store(unpublished_copied)
        paths["unpublished_copied"] = str(unpublished_copied)
        copied_clock = datetime.now(tz=UTC)
        print("workspace: copied event identity begin", flush=True)
        _write_event(root / "event-copied.json", str(copied), copied_clock)
        print("workspace: copied event identity done", flush=True)
        paths["event_copied"] = str(root / "event-copied.json")
        paths["cutoff_copied"] = raw_http_cutoff(copied_clock)
        paths["clock_copied"] = utc_text(copied_clock)
    print("workspace: rust build begin", flush=True)
    rust_meta = build_research_rust(root / "issue-898-ram-cpu")
    paths["rust_meta"] = json.dumps(rust_meta, sort_keys=True)
    if rust_meta.get("status") == "OK":
        paths["rust_binary"] = str(root / "issue-898-ram-cpu")
    return paths


def build_research_rust(binary: Path) -> dict[str, object]:
    if shutil.which("cargo") is None or shutil.which("rustc") is None:
        return {"status": UNOBSERVED, "reason": "cargo or rustc missing"}
    if not (RUST_CRATE / "Cargo.toml").is_file():
        return {"status": UNOBSERVED, "reason": "research crate missing"}
    env = dict(os.environ)
    env["CARGO_TARGET_DIR"] = str(RUST_TARGET)
    env["CARGO_TERM_COLOR"] = "never"
    completed = subprocess.run(
        [
            "cargo",
            "build",
            "--release",
            "--manifest-path",
            str(RUST_CRATE / "Cargo.toml"),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    built = RUST_TARGET / "release" / "issue-898-ram-cpu"
    if completed.returncode != 0 or not built.is_file():
        return {
            "status": "ERROR",
            "reason": "cargo build failed",
            "stderr_tail": completed.stderr[-2000:],
        }
    if binary.exists():
        binary.unlink()
    shutil.copy2(built, binary)
    os.chmod(binary, 0o755)
    rustc = subprocess.check_output(["rustc", "-vV"], text=True)
    cargo = subprocess.check_output(["cargo", "--version"], text=True).strip()
    lock = RUST_CRATE / "Cargo.lock"
    sources = sorted((RUST_CRATE / "src").glob("*.rs"))
    source_hasher = hashlib.sha256()
    for path in sources:
        source_hasher.update(path.name.encode("utf-8"))
        source_hasher.update(path.read_bytes())
    return {
        "status": "OK",
        "binary_digest": _sha256_file(binary),
        "build_flags": ["--release"],
        "cargo": cargo,
        "cargo_lock_digest": _sha256_file(lock) if lock.is_file() else UNOBSERVED,
        "crate_path": str(RUST_CRATE.relative_to(RUST_CRATE.parents[2])),
        "profile": "release",
        "source_digest": "sha256:" + source_hasher.hexdigest(),
        "source_files": [path.name for path in sources],
        "toolchain": rustc.strip().splitlines(),
    }


CASE_SPECS: tuple[tuple[str, str, str], ...] = (
    ("A1_bare_interpreter", "bare", ""),
    ("A2_import_cycle", "import_cycle", ""),
    ("A3_instantiate_runner", "instantiate_runner", ""),
    ("A4_idle_worker", "idle_worker", ""),
    ("A5_hermes_cycle_import", "hermes_cycle_import", ""),
    ("A6_process_tree", "process_tree", ""),
    ("B1_one_event_solo", "resolve", "solo"),
    ("B2_one_event_representative", "resolve", "representative"),
    ("B3_one_event_times10", "resolve", "times10"),
    ("B4_shape_json", "resolve", "json"),
    ("B5_shape_rss", "resolve", "rss"),
    ("B6_shape_atom", "resolve", "atom"),
    ("B7_large_body", "resolve", "large"),
    ("B8_malformed", "resolve", "malformed"),
    ("B9_empty", "resolve", "empty"),
    ("B10_copied_one_event", "resolve", "copied"),
    ("B11_scaled_one_event", "resolve", "scaled"),
    ("C1_permitted_rows", "permitted_rows", "scaled"),
    ("C2_parsed_observations", "parsed_observations", "scaled"),
    ("C3_units_from", "units_from", "scaled"),
    ("C4_unique_and_revisions", "unique_and_revisions", "scaled"),
    ("C5_load_graphiti_units", "load", "scaled"),
    ("C6_resolve_event_units", "resolve", "scaled"),
    ("C7_admission_generation", "admission_generation", ""),
    ("D1_cycle_max_writes_0", "cycle", "copied"),
    ("E1_copied_full_corpus_load", "load", "copied"),
    ("R0_rust_process_baseline", "rust_r0", ""),
    ("R1_python_observation_scan", "python_scan", "copied"),
    ("R1_rust_observation_scan", "rust_r1", "copied"),
    ("R1_rust_e2e_parent", "rust_e2e", "copied"),
    ("R2_python_bounded_units", "python_r2", "copied"),
    ("R2_bounded_candidate", "rust_r2", "copied"),
    ("R2_rust_e2e_parent", "rust_r2_e2e", "copied"),
    ("S1_import_cycle_tracemalloc", "import_cycle_tracemalloc", ""),
    ("S2_resolve_solo_tracemalloc", "resolve_tracemalloc", "solo"),
)


def _event_spec(workspace: dict[str, str], store: str) -> dict[str, object]:
    path = Path(workspace[f"event_{store}"])
    return json.loads(path.read_text(encoding="utf-8"))


def _store_path(workspace: dict[str, str], store: str) -> str | None:
    if store == "copied":
        return workspace.get("proving_copied")
    return workspace.get(f"proving_{store}")


def _cutoff(workspace: dict[str, str], store: str) -> str:
    if store == "copied" and "cutoff_copied" in workspace:
        return workspace["cutoff_copied"]
    return workspace.get(f"cutoff_{store}", raw_http_cutoff(CLOCK))


def _clock(workspace: dict[str, str], store: str) -> datetime:
    if store == "copied" and "clock_copied" in workspace:
        return parse_utc(workspace["clock_copied"])
    return CLOCK


def run_named_case(
    name: str,
    workspace: dict[str, str],
    *,
    use_tracemalloc: bool = False,
) -> dict[str, object]:
    spec = {item[0]: item for item in CASE_SPECS}[name]
    _case_name, kind, store = spec
    if kind == "bare":
        return case_bare_interpreter()
    if kind == "import_cycle":
        return case_import_cycle(use_tracemalloc=use_tracemalloc)
    if kind == "import_cycle_tracemalloc":
        return case_import_cycle(use_tracemalloc=True)
    if kind == "instantiate_runner":
        return case_instantiate_runner()
    if kind == "idle_worker":
        return case_idle_worker()
    if kind == "hermes_cycle_import":
        return case_hermes_cycle_import()
    if kind == "process_tree":
        return case_process_tree()
    if kind == "admission_generation":
        return case_admission_generation()
    proving = _store_path(workspace, store) if store else None
    if kind == "resolve":
        if not proving:
            return {"status": UNOBSERVED, "reason": f"{store} proving store missing"}
        return case_resolve_event(
            proving,
            _event_spec(workspace, store),
            use_tracemalloc=use_tracemalloc,
        )
    if kind == "resolve_tracemalloc":
        if not proving:
            return {"status": UNOBSERVED, "reason": f"{store} proving store missing"}
        return case_resolve_event(
            proving,
            _event_spec(workspace, store),
            use_tracemalloc=True,
        )
    if kind == "permitted_rows":
        return case_permitted_rows(proving or workspace["proving_scaled"], clock=_clock(workspace, store or "scaled"))
    if kind == "parsed_observations":
        return case_parsed_observations(proving or workspace["proving_scaled"], clock=_clock(workspace, store or "scaled"))
    if kind == "units_from":
        return case_units_from(proving or workspace["proving_scaled"], clock=_clock(workspace, store or "scaled"))
    if kind == "unique_and_revisions":
        return case_unique_and_revisions(proving or workspace["proving_scaled"], clock=_clock(workspace, store or "scaled"))
    if kind == "load":
        if not proving:
            return {"status": UNOBSERVED, "reason": f"{store} proving store missing"}
        return case_load_units(proving, clock=_clock(workspace, store))
    if kind == "cycle":
        if not proving:
            return case_cycle_max_writes_0(workspace["proving_representative"])
        unpublished = workspace.get("unpublished_copied") or workspace.get(
            "unpublished_representative"
        )
        return case_cycle_max_writes_0(
            proving,
            unpublished,
            clock=_clock(workspace, store),
        )
    if kind == "python_scan":
        target = proving or workspace.get("proving_scaled")
        if not target:
            return {"status": UNOBSERVED, "reason": "scan store missing"}
        return case_observation_scan(target, _cutoff(workspace, store or "scaled"))
    if kind == "rust_r0":
        binary = workspace.get("rust_binary")
        if not binary:
            return {"status": UNOBSERVED, "reason": "research rust binary missing"}
        return _run_rust_binary(binary, ["r0"])
    if kind == "rust_r1":
        binary = workspace.get("rust_binary")
        target = proving or workspace.get("proving_scaled")
        if not binary or not target:
            return {"status": UNOBSERVED, "reason": "rust binary or scan store missing"}
        return _run_rust_binary(
            binary,
            ["r1", "--db", target, "--cutoff", _cutoff(workspace, store or "scaled")],
        )
    if kind == "rust_e2e":
        binary = workspace.get("rust_binary")
        target = proving or workspace.get("proving_scaled")
        if not binary or not target:
            return {"status": UNOBSERVED, "reason": "rust binary or scan store missing"}
        return case_rust_e2e(
            binary,
            target,
            _cutoff(workspace, store or "scaled"),
        )
    if kind == "python_r2":
        target = proving or workspace.get("proving_scaled")
        event_name = store or "scaled"
        if not target:
            return {"status": UNOBSERVED, "reason": "r2 proving store missing"}
        return case_python_r2(target, _event_spec(workspace, event_name))
    if kind in {"rust_r2", "rust_r2_e2e"}:
        binary = workspace.get("rust_binary")
        target = proving or workspace.get("proving_scaled")
        spec_path = workspace.get(f"r2_spec_{store or 'scaled'}")
        if not binary or not target or not spec_path:
            return {
                "status": UNOBSERVED,
                "reason": "rust binary, r2 store or r2 spec missing",
            }
        if kind == "rust_r2_e2e":
            return case_rust_r2_e2e(binary, target, spec_path)
        return case_rust_r2(binary, target, spec_path)
    raise ValueError(f"unknown case kind: {kind}")


def _run_rust_binary(binary: str, args: list[str]) -> dict[str, object]:
    gc.collect()
    rss_before = current_rss_bytes()
    usage_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    started = time.perf_counter()
    completed = subprocess.run(
        [binary, *args],
        check=False,
        capture_output=True,
        text=True,
    )
    wall = time.perf_counter() - started
    usage_after = resource.getrusage(resource.RUSAGE_CHILDREN)
    try:
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError, ValueError):
        payload = {
            "status": "ERROR",
            "outcome": {
                "error": "rust stdout was not JSON",
                "stderr_tail": completed.stderr[-1000:],
            },
        }
    outcome = payload.get("outcome", payload)
    return {
        "pid": os.getpid(),
        "status": payload.get("status", "ERROR"),
        "rss_before_bytes": rss_before,
        "rss_after_bytes": payload.get("rss_after_bytes", current_rss_bytes()),
        "rss_held_bytes": payload.get("rss_held_bytes", UNOBSERVED),
        "ru_maxrss_bytes": UNOBSERVED,
        "child_user_cpu_seconds": usage_after.ru_utime - usage_before.ru_utime,
        "child_system_cpu_seconds": usage_after.ru_stime - usage_before.ru_stime,
        "user_cpu_seconds": usage_after.ru_utime - usage_before.ru_utime,
        "system_cpu_seconds": usage_after.ru_stime - usage_before.ru_stime,
        "wall_seconds": wall,
        "tracemalloc_enabled": False,
        "tracemalloc_peak_bytes": UNOBSERVED,
        "tracemalloc_current_bytes": UNOBSERVED,
        "returncode": completed.returncode,
        "outcome": outcome if isinstance(outcome, dict) else {"value": outcome},
    }


def case_rust_e2e(binary: str, proving: str, cutoff: str) -> dict[str, object]:
    def work() -> dict[str, object]:
        proc = subprocess.Popen(
            [binary, "r1", "--db", proving, "--cutoff", cutoff],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        combined_peak: int | str = 0
        while proc.poll() is None:
            parent = current_rss_bytes()
            child = rss_bytes_for_pid(proc.pid)
            if isinstance(parent, int) and isinstance(child, int):
                if combined_peak == 0 or (
                    isinstance(combined_peak, int) and parent + child > combined_peak
                ):
                    combined_peak = parent + child
            time.sleep(0.02)
        stdout, stderr = proc.communicate()
        try:
            payload = json.loads(stdout.strip().splitlines()[-1])
        except (json.JSONDecodeError, IndexError, ValueError):
            payload = {
                "status": "ERROR",
                "outcome": {"error": "rust stdout was not JSON", "stderr_tail": stderr[-1000:]},
            }
        outcome = payload.get("outcome", {})
        if not isinstance(outcome, dict):
            outcome = {"value": outcome}
        parent = current_rss_bytes()
        if isinstance(parent, int) and isinstance(combined_peak, int):
            combined_peak = max(combined_peak, parent)
        outcome = {
            **outcome,
            "child_status": payload.get("status"),
            "combined_peak_rss_bytes": combined_peak,
            "parent_rss_after_bytes": parent,
            "queue_claimed": False,
            "returncode": proc.returncode,
        }
        return outcome

    return measure(work)


def _parse_time_l(stderr: str) -> int | str:
    for line in stderr.splitlines():
        if "maximum resident set size" in line:
            number = line.strip().split()[0]
            try:
                return int(number)
            except ValueError:
                return UNOBSERVED
    return UNOBSERVED


def _parse_time_l_cpu(stderr: str) -> tuple[object, object]:
    for line in stderr.splitlines():
        parts = line.split()
        if "real" in parts and "user" in parts and "sys" in parts:
            try:
                user = float(parts[parts.index("user") - 1])
                system = float(parts[parts.index("sys") - 1])
            except (ValueError, IndexError):
                return UNOBSERVED, UNOBSERVED
            return user, system
    return UNOBSERVED, UNOBSERVED


def run_child_case(
    *,
    executable: str,
    module: str,
    case: str,
    workspace: Path,
    env: dict[str, str],
    extra_args: tuple[str, ...] = (),
) -> dict[str, object]:
    command = [
        "/usr/bin/time",
        "-l",
        executable,
        "-m",
        module,
        "--case",
        case,
        "--workspace",
        str(workspace),
        *extra_args,
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    try:
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError, ValueError):
        payload = {
            "status": "ERROR",
            "outcome": {
                "error": "child stdout was not JSON",
                "stderr_tail": completed.stderr[-1000:],
                "returncode": completed.returncode,
            },
        }
    payload["time_l_maxrss_bytes"] = _parse_time_l(completed.stderr)
    if payload.get("ru_maxrss_bytes") in {None, UNOBSERVED} and isinstance(
        payload.get("time_l_maxrss_bytes"), int
    ):
        payload["ru_maxrss_bytes"] = payload["time_l_maxrss_bytes"]
    payload["returncode"] = completed.returncode
    payload["command"] = command
    return payload


def run_timed_command(command: list[str], env: dict[str, str]) -> dict[str, object]:
    completed = subprocess.run(
        ["/usr/bin/time", "-l", *command],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    try:
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError, ValueError):
        payload = {
            "status": "ERROR",
            "outcome": {
                "error": "timed stdout was not JSON",
                "stderr_tail": completed.stderr[-1000:],
            },
        }
    time_l = _parse_time_l(completed.stderr)
    user, system = _parse_time_l_cpu(completed.stderr)
    if isinstance(time_l, int):
        payload["ru_maxrss_bytes"] = time_l
    payload["time_l_maxrss_bytes"] = time_l
    payload["peak_from_time_l_bytes"] = time_l
    payload["returncode"] = completed.returncode
    payload["command"] = command
    if payload.get("user_cpu_seconds") in {None, UNOBSERVED}:
        payload["user_cpu_seconds"] = user
        payload["system_cpu_seconds"] = system
    return payload


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if not ordered:
        raise ValueError("median of empty sample")
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def _rss_samples(item: dict[str, object]) -> list[int]:
    values: list[int] = []
    for key in ("time_l_maxrss_bytes", "ru_maxrss_bytes", "rss_held_bytes"):
        sample = item.get(key)
        if isinstance(sample, int):
            values.append(sample)
    outcome = item.get("outcome") if isinstance(item.get("outcome"), dict) else {}
    if isinstance(outcome, dict):
        combined = outcome.get("combined_peak_rss_bytes")
        if isinstance(combined, int):
            values.append(combined)
    return values


def summarise_case(name: str, runs: list[dict[str, object]]) -> dict[str, object]:
    measured = runs[WARMUPS:] if len(runs) > WARMUPS else runs
    peaks: list[int] = []
    cpus: list[float] = []
    retained: list[int] = []
    for item in measured:
        samples = _rss_samples(item)
        if samples:
            peaks.append(max(samples))
        after = item.get("rss_after_bytes")
        user = item.get("user_cpu_seconds")
        system = item.get("system_cpu_seconds")
        if isinstance(after, int):
            retained.append(after)
        if isinstance(user, (int, float)) and isinstance(system, (int, float)):
            cpus.append(float(user) + float(system))
    return {
        "case": name,
        "warmup_count": 0 if name in {"A6_process_tree", "S1_import_cycle_tracemalloc", "S2_resolve_solo_tracemalloc"} else WARMUPS,
        "measured_count": len(measured),
        "max_peak_rss_bytes": max(peaks) if peaks else UNOBSERVED,
        "median_cpu_seconds": _median(cpus) if cpus else UNOBSERVED,
        "max_retained_rss_bytes": max(retained) if retained else UNOBSERVED,
        "runs": runs,
    }


def _last_outcome(summary: dict[str, object]) -> dict[str, object]:
    runs = summary.get("runs") or []
    for item in reversed(runs):
        if not isinstance(item, dict):
            continue
        outcome = item.get("outcome") if isinstance(item.get("outcome"), dict) else item
        if isinstance(outcome, dict):
            return outcome
    return {}


def _last_digest(summary: dict[str, object]) -> object:
    outcome = _last_outcome(summary)
    digest = outcome.get("manifest_digest")
    return digest if isinstance(digest, str) and digest.startswith("sha256:") else UNOBSERVED


def _parity_from_summaries(summaries: dict[str, dict[str, object]]) -> dict[str, object]:
    python_digest = _last_digest(summaries.get("R1_python_observation_scan", {}))
    rust_digest = _last_digest(summaries.get("R1_rust_observation_scan", {}))
    e2e_digest = _last_digest(summaries.get("R1_rust_e2e_parent", {}))
    match = (
        isinstance(python_digest, str)
        and python_digest.startswith("sha256:")
        and python_digest == rust_digest
        and (e2e_digest in {rust_digest, UNOBSERVED} or e2e_digest == python_digest)
    )
    return {
        "boundary": "retained_observation_body_scan",
        "e2e_manifest_digest": e2e_digest,
        "match": match,
        "python_manifest_digest": python_digest,
        "rust_manifest_digest": rust_digest,
        "schema": SCAN_SCHEMA,
        "unit_parity_claimed": False,
    }


def _r2_parity_from_summaries(summaries: dict[str, dict[str, object]]) -> dict[str, object]:
    python = summaries.get("R2_python_bounded_units", {})
    rust = summaries.get("R2_bounded_candidate", {})
    e2e = summaries.get("R2_rust_e2e_parent", {})
    python_digest = _last_digest(python)
    rust_digest = _last_digest(rust)
    e2e_digest = _last_digest(e2e)
    python_outcome = _last_outcome(python)
    oracle = python_outcome.get("oracle") if isinstance(python_outcome.get("oracle"), dict) else {}
    match = (
        isinstance(python_digest, str)
        and python_digest.startswith("sha256:")
        and python_digest == rust_digest
        and (e2e_digest in {rust_digest, UNOBSERVED} or e2e_digest == python_digest)
    )
    return {
        "boundary": "bounded_useful_units",
        "e2e_manifest_digest": e2e_digest,
        "match": match,
        "oracle_match": oracle.get("match") is True,
        "python_manifest_digest": python_digest,
        "rust_manifest_digest": rust_digest,
        "schema": R2_SCHEMA,
        "unit_parity_claimed": match and oracle.get("match") is True,
    }


def _migration_atom(
    summaries: dict[str, dict[str, object]],
    r2_parity: dict[str, object],
) -> dict[str, object]:
    python = summaries.get("R2_python_bounded_units", {})
    rust = summaries.get("R2_bounded_candidate", {})
    e2e = summaries.get("R2_rust_e2e_parent", {})
    python_outcome = _last_outcome(python)
    rust_outcome = _last_outcome(rust)
    python_peak = python.get("max_peak_rss_bytes")
    e2e_peak = e2e.get("max_peak_rss_bytes")
    rust_child_peak = rust.get("max_peak_rss_bytes")
    python_cpu = python.get("median_cpu_seconds")
    rust_child_cpu = rust.get("median_cpu_seconds")
    rust_parent_cpu = e2e.get("median_cpu_seconds")
    rust_total_cpu = (
        float(rust_child_cpu) + float(rust_parent_cpu)
        if isinstance(rust_child_cpu, (int, float))
        and isinstance(rust_parent_cpu, (int, float))
        else UNOBSERVED
    )
    keys_present = bool(
        (python_outcome.get("row_count") or rust_outcome.get("row_count")) not in {0, None, UNOBSERVED}
    )
    if not python and not rust:
        return {
            "first_migration_atom": "HOLD",
            "r2_hold": True,
            "r2_reason": (
                "R2 bounded useful-output path was not measured; first migration "
                "atom remains HOLD"
            ),
            "r2_parity": r2_parity,
            "unit_parity_claimed": False,
        }
    if python_outcome.get("status") == "HOLD" and not r2_parity.get("oracle_match"):
        return {
            "first_migration_atom": "HOLD",
            "r2_hold": True,
            "r2_reason": str(
                python_outcome.get("reason")
                or "bounded rows exist but useful-output parity with unit_refs is unproved"
            ),
            "r2_parity": r2_parity,
            "unit_parity_claimed": False,
        }
    if not r2_parity.get("oracle_match"):
        return {
            "first_migration_atom": "HOLD",
            "r2_hold": True,
            "r2_reason": (
                "bounded proving rows are selectable from unit_refs, but computed "
                "useful output does not match the retained unit_refs oracle"
            ),
            "r2_parity": r2_parity,
            "unit_parity_claimed": False,
        }
    if r2_parity.get("match") is not True:
        return {
            "first_migration_atom": "HOLD",
            "r2_hold": True,
            "r2_reason": (
                "Python bounded useful output matches unit_refs, but the research "
                "Rust comparator digest does not match (parser/identity/chunking)"
            ),
            "r2_parity": r2_parity,
            "unit_parity_claimed": False,
        }
    if not isinstance(python_peak, int) or not isinstance(e2e_peak, int):
        return {
            "first_migration_atom": "HOLD",
            "r2_hold": True,
            "r2_reason": "R2 Python or Rust end-to-end peak RSS is UNOBSERVED",
            "r2_parity": r2_parity,
            "unit_parity_claimed": True,
        }
    removable = max(0, python_peak - e2e_peak)
    threshold_ok = removable >= GO_PEAK_BYTES or (
        python_peak > 0 and removable / python_peak >= GO_PEAK_RATIO
    )
    cpu_regress = (
        isinstance(python_cpu, (int, float))
        and isinstance(rust_total_cpu, (int, float))
        and python_cpu > 0
        and rust_total_cpu > python_cpu * 1.2
    )
    if rust_total_cpu is UNOBSERVED:
        return {
            "first_migration_atom": "HOLD",
            "r2_hold": True,
            "r2_reason": "R2 useful-output parity holds, but Rust child-plus-parent CPU is UNOBSERVED",
            "r2_parity": r2_parity,
            "r2_python_peak_rss_bytes": python_peak,
            "r2_rust_e2e_peak_rss_bytes": e2e_peak,
            "r2_removable_peak_rss_bytes": removable,
            "unit_parity_claimed": True,
        }
    if cpu_regress:
        return {
            "first_migration_atom": "HOLD",
            "r2_hold": True,
            "r2_reason": (
                "R2 useful-output parity holds, but Rust child-plus-parent CPU "
                "regresses more than 20%"
            ),
            "r2_parity": r2_parity,
            "r2_python_cpu_seconds": python_cpu,
            "r2_rust_total_cpu_seconds": rust_total_cpu,
            "unit_parity_claimed": True,
        }
    if not threshold_ok:
        return {
            "first_migration_atom": "HOLD",
            "r2_hold": True,
            "r2_reason": (
                "R2 reconstructed exact useful output from bounded rows, but "
                "peak-RSS reduction versus the Python bounded path is below "
                "both the 20% and 64 MiB gates"
            ),
            "r2_parity": r2_parity,
            "r2_python_peak_rss_bytes": python_peak,
            "r2_rust_child_peak_rss_bytes": rust_child_peak,
            "r2_rust_e2e_peak_rss_bytes": e2e_peak,
            "r2_removable_peak_rss_bytes": removable,
            "unit_parity_claimed": True,
        }
    return {
        "first_migration_atom": "GO",
        "r2_hold": False,
        "r2_reason": (
            "R2 GO: research Rust reconstructed exact useful output from bounded "
            "unit_refs rows and reduced end-to-end peak RSS by at least 20% or "
            "64 MiB after launch/IPC. This still authorises no product migration."
        ),
        "r2_parity": r2_parity,
        "r2_python_peak_rss_bytes": python_peak,
        "r2_rust_child_peak_rss_bytes": rust_child_peak,
        "r2_rust_e2e_peak_rss_bytes": e2e_peak,
        "r2_python_cpu_seconds": python_cpu,
        "r2_rust_child_cpu_seconds": rust_child_cpu,
        "r2_rust_e2e_parent_cpu_seconds": rust_parent_cpu,
        "r2_rust_total_cpu_seconds": rust_total_cpu,
        "r2_removable_peak_rss_bytes": removable,
        "unit_parity_claimed": True,
        "keys_present": keys_present,
    }


def decide(
    summaries: dict[str, dict[str, object]],
    *,
    parity: dict[str, object] | None = None,
) -> dict[str, object]:
    parity = parity or _parity_from_summaries(summaries)
    r2_parity = _r2_parity_from_summaries(summaries)
    atom = _migration_atom(summaries, r2_parity)
    python_scan = summaries.get("R1_python_observation_scan", {})
    rust_scan = summaries.get("R1_rust_observation_scan", {})
    rust_e2e = summaries.get("R1_rust_e2e_parent", {})
    rust_r0 = summaries.get("R0_rust_process_baseline", {})
    python_peak = python_scan.get("max_peak_rss_bytes")
    rust_child_peak = rust_scan.get("max_peak_rss_bytes")
    e2e_peak = rust_e2e.get("max_peak_rss_bytes")
    python_cpu = python_scan.get("median_cpu_seconds")
    rust_child_cpu = rust_scan.get("median_cpu_seconds")
    rust_parent_cpu = rust_e2e.get("median_cpu_seconds")
    rust_total_cpu = (
        float(rust_child_cpu) + float(rust_parent_cpu)
        if isinstance(rust_child_cpu, (int, float))
        and isinstance(rust_parent_cpu, (int, float))
        else UNOBSERVED
    )
    rust_present = rust_r0.get("max_peak_rss_bytes") is not UNOBSERVED or any(
        run.get("status") == "OK" for run in (rust_r0.get("runs") or [])
    )
    if not rust_present or rust_scan.get("max_peak_rss_bytes") is UNOBSERVED:
        decision = "HOLD"
        reason = "research Rust comparator was not measured on this host"
    elif python_peak is UNOBSERVED or e2e_peak is UNOBSERVED:
        decision = "HOLD"
        reason = "Python or Rust end-to-end observation-scan RSS is UNOBSERVED"
    elif parity.get("match") is not True:
        decision = "HOLD"
        reason = "observation-scan manifest parity is incomplete or mismatched"
    elif not isinstance(python_peak, int) or not isinstance(e2e_peak, int):
        decision = "HOLD"
        reason = "observation-scan RSS is not an integer measurement"
    else:
        removable = max(0, python_peak - e2e_peak)
        threshold_ok = removable >= GO_PEAK_BYTES or (
            python_peak > 0 and removable / python_peak >= GO_PEAK_RATIO
        )
        cpu_regress = (
            isinstance(python_cpu, (int, float))
            and isinstance(rust_total_cpu, (int, float))
            and python_cpu > 0
            and rust_total_cpu > python_cpu * 1.2
        )
        if threshold_ok and rust_total_cpu is UNOBSERVED:
            decision = "HOLD"
            reason = (
                "Rust reduces observation-scan RSS enough to clear the gate, but "
                "Rust child-plus-parent CPU is UNOBSERVED"
            )
        elif threshold_ok and cpu_regress:
            decision = "HOLD"
            reason = (
                "Rust reduces observation-scan RSS enough to clear the gate, but "
                "local CPU (Rust child plus e2e parent) regresses more than 20% "
                "and the owner has not accepted that RAM trade-off"
            )
        elif threshold_ok:
            decision = "FEASIBILITY_GO"
            reason = (
                "R1 FEASIBILITY_GO: Rust observation-scan comparator matches "
                "Python output and reduces end-to-end peak RSS by at least 20% "
                "or 64 MiB after launch/IPC. First migration atom is "
                f"{atom['first_migration_atom']}: {atom['r2_reason']} "
                "R1 does not authorise implementation."
            )
        else:
            decision = "NO_GO"
            reason = (
                "methodologically valid Rust observation-scan comparison shows RAM "
                "improvement below both the 20% and 64 MiB thresholds after launch/IPC"
            )
        return {
            "go_or_no_go": decision,
            "first_migration_atom": atom["first_migration_atom"],
            "reason": reason,
            "threshold_ok": threshold_ok,
            "parity": parity,
            "python_scan_peak_rss_bytes": python_peak,
            "python_scan_cpu_seconds": python_cpu if isinstance(python_cpu, (int, float)) else UNOBSERVED,
            "rust_child_peak_rss_bytes": rust_child_peak if isinstance(rust_child_peak, int) else UNOBSERVED,
            "rust_child_cpu_seconds": rust_child_cpu if isinstance(rust_child_cpu, (int, float)) else UNOBSERVED,
            "rust_e2e_peak_rss_bytes": e2e_peak,
            "rust_e2e_parent_cpu_seconds": rust_parent_cpu if isinstance(rust_parent_cpu, (int, float)) else UNOBSERVED,
            "rust_total_cpu_seconds": rust_total_cpu,
            "removable_peak_rss_bytes": removable,
            "r2_hold": atom["r2_hold"],
            "r2_reason": atom["r2_reason"],
            "r2_parity": r2_parity,
            "selected_boundary": "retained_observation_body_scan",
            "unit_parity_claimed": atom["unit_parity_claimed"],
            "go_basis": "peak_rss",
            "retained_rss_note": (
                "Rust e2e retained RSS is dominated by child-process termination; "
                "the R1 feasibility GO is based on peak RSS, not post-exit parent RSS"
            ),
        }
    return {
        "go_or_no_go": decision,
        "first_migration_atom": atom["first_migration_atom"],
        "reason": reason,
        "threshold_ok": False,
        "parity": parity,
        "python_scan_peak_rss_bytes": python_peak if isinstance(python_peak, int) else UNOBSERVED,
        "python_scan_cpu_seconds": python_cpu if isinstance(python_cpu, (int, float)) else UNOBSERVED,
        "rust_child_peak_rss_bytes": rust_child_peak if isinstance(rust_child_peak, int) else UNOBSERVED,
        "rust_child_cpu_seconds": rust_child_cpu if isinstance(rust_child_cpu, (int, float)) else UNOBSERVED,
        "rust_e2e_peak_rss_bytes": e2e_peak if isinstance(e2e_peak, int) else UNOBSERVED,
        "rust_e2e_parent_cpu_seconds": rust_parent_cpu if isinstance(rust_parent_cpu, (int, float)) else UNOBSERVED,
        "rust_total_cpu_seconds": rust_total_cpu,
        "removable_peak_rss_bytes": UNOBSERVED,
        "r2_hold": atom["r2_hold"],
        "r2_reason": atom["r2_reason"],
        "r2_parity": r2_parity,
        "selected_boundary": "retained_observation_body_scan",
        "unit_parity_claimed": atom["unit_parity_claimed"],
        "go_basis": "peak_rss",
        "retained_rss_note": (
            "Rust e2e retained RSS is dominated by child-process termination; "
            "the R1 feasibility GO is based on peak RSS, not post-exit parent RSS"
        ),
    }


def host_identity() -> dict[str, object]:
    values: dict[str, object] = {
        "platform": sys.platform,
        "executable": sys.executable,
    }
    try:
        completed = subprocess.run(
            [
                "/usr/sbin/sysctl",
                "-n",
                "hw.model",
                "machdep.cpu.brand_string",
                "hw.physicalcpu",
                "hw.logicalcpu",
                "hw.memsize",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        lines = completed.stdout.splitlines()
        values.update(
            {
                "machine_model": lines[0],
                "chip": lines[1],
                "physical_cores": int(lines[2]),
                "logical_cores": int(lines[3]),
                "memory_bytes": int(lines[4]),
            }
        )
    except (OSError, subprocess.CalledProcessError, IndexError, ValueError):
        values["machine_model"] = UNOBSERVED
    return values


def git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return UNOBSERVED


def assemble_packet(
    *,
    summaries: dict[str, dict[str, object]],
    copy_meta: dict[str, object],
    rust_meta: dict[str, object],
    workspace_digest: str,
) -> dict[str, object]:
    parity = _parity_from_summaries(summaries)
    r2_parity = _r2_parity_from_summaries(summaries)
    decision = decide(summaries, parity=parity)
    questions = {
        "1_largest_idle_rss_process": summaries.get("A6_process_tree", {})
        .get("runs", [{}])[-1]
        .get("largest_rss_bytes", UNOBSERVED)
        if summaries.get("A6_process_tree")
        else UNOBSERVED,
        "2_rss_retained_before_useful_work": summaries.get(
            "A4_idle_worker", {}
        ).get("max_retained_rss_bytes", UNOBSERVED),
        "3_active_peak_resolving_one_graphiti_event": summaries.get(
            "B10_copied_one_event", summaries.get("B2_one_event_representative", {})
        ).get("max_peak_rss_bytes", UNOBSERVED),
        "4_rss_after_one_event_or_cycle": summaries.get(
            "D1_cycle_max_writes_0", {}
        ).get("max_retained_rss_bytes", UNOBSERVED),
        "5_stage_largest_removable_peak_rss": UNOBSERVED,
        "6_stage_most_local_cpu": max(
            (
                (
                    name,
                    summary.get("median_cpu_seconds"),
                )
                for name, summary in summaries.items()
                if isinstance(summary.get("median_cpu_seconds"), (int, float))
            ),
            key=lambda item: float(item[1]),
            default=(UNOBSERVED, UNOBSERVED),
        )[0],
        "7_one_event_scales_with_unrelated_corpus": {
            "solo_peak": summaries.get("B1_one_event_solo", {}).get(
                "max_peak_rss_bytes", UNOBSERVED
            ),
            "times10_peak": summaries.get("B3_one_event_times10", {}).get(
                "max_peak_rss_bytes", UNOBSERVED
            ),
            "scaled_peak": summaries.get("B11_scaled_one_event", {}).get(
                "max_peak_rss_bytes", UNOBSERVED
            ),
            "copied_peak": summaries.get("B10_copied_one_event", {}).get(
                "max_peak_rss_bytes", UNOBSERVED
            ),
        },
        "8_latest_bodies_parsed_or_materialised_more_than_once": "YES_STATIC",
        "9_dominant_memory_source": "retained_corpus_reconstruction",
        "10_bounded_rust_process_would_remove_dominant_allocation": (
            True
            if decision["go_or_no_go"] == "FEASIBILITY_GO"
            else False
            if decision["go_or_no_go"] == "NO_GO"
            else UNOBSERVED
        ),
    }
    return {
        "issue": ISSUE,
        "status": "MEASURED",
        "decision": decision["go_or_no_go"],
        "first_migration_atom": decision.get("first_migration_atom", "HOLD"),
        "decision_reason": decision["reason"],
        "previous_no_go_withdrawn": True,
        "inspection_head": git_head(),
        "inspection_date": datetime.now(tz=UTC).strftime("%Y-%m-%d"),
        "intended_hardware": host_identity(),
        "copy_meta": copy_meta,
        "rust_meta": rust_meta,
        "parity": parity,
        "r2_parity": r2_parity,
        "r2_reason": decision.get("r2_reason"),
        "workspace_digest": workspace_digest,
        "method": {
            "peak_rss_and_cpu": ["/usr/bin/time -l", "resource.getrusage"],
            "current_or_retained_rss": ["ps -o rss"],
            "python_allocation_supplement": "tracemalloc off for primary; S1/S2 only",
            "sqlite_snapshot": "sqlite3.Connection.backup from read-only source",
            "one_event": "event identity prepared outside the measured child; one _resolve_graphiti_event_units call",
            "r2": (
                "unit_refs retained on the prepared event; R2 selects "
                "(proving_run_id, source_id, observation_digest) rows and "
                "computes useful output from bodies. unit_refs are oracle only."
            ),
            "fresh_process_per_case": True,
            "warmup_plus_measured_runs": f"{WARMUPS} warmup and {MEASURED_RUNS} fresh-process executions",
            "rust": "research-only crate docs/research/issue-898-ram-cpu-rust",
            "cpu_gate": "Python scan median CPU versus Rust child plus e2e parent CPU",
            "go_basis": "peak RSS after launch/IPC; retained RSS is not the GO basis",
        },
        "go_gate": decision,
        "questions": questions,
        "cases": summaries,
        "non_effects": [
            "no product code change",
            "no Python exact-row-selection or query fix",
            "no production Rust crate, Cargo workspace or runtime integration",
            "no provider or model-catalogue call",
            "no queue claim, consume, retry or release",
            "no writable canonical-store access",
            "no Neo4j or Graphiti mutation",
            "no publication, deployment, activation or canary",
            "no live daemon restart, signal or reconfiguration",
            "no implementation issue created",
        ],
    }


def orchestrate(
    *,
    output: Path,
    workspace: Path | None = None,
    only: tuple[str, ...] = (),
    merge_into: Path | None = None,
    refresh_copied_event: bool = False,
) -> dict[str, object]:
    close_workspace = False
    if workspace is None:
        workspace = Path(tempfile.mkdtemp(prefix="newsroom-898-"))
        close_workspace = True
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        if merge_into is not None and workspace.exists() and any(workspace.glob("proving-*.sqlite3")):
            paths = load_workspace(workspace)
            rust_meta = build_research_rust(workspace / "issue-898-ram-cpu")
            paths["rust_meta"] = json.dumps(rust_meta, sort_keys=True)
            if rust_meta.get("status") == "OK":
                paths["rust_binary"] = str(workspace / "issue-898-ram-cpu")
            if refresh_copied_event and "proving_copied" in paths:
                print("workspace: refresh copied event identity begin", flush=True)
                clock = (
                    parse_utc(paths["clock_copied"])
                    if "clock_copied" in paths
                    else datetime.now(tz=UTC)
                )
                _write_event(workspace / "event-copied.json", paths["proving_copied"], clock)
                paths["event_copied"] = str(workspace / "event-copied.json")
                paths["r2_spec_copied"] = str(workspace / "r2-spec-copied.json")
                print("workspace: refresh copied event identity done", flush=True)
            copy_meta = json.loads(merge_into.read_text(encoding="utf-8")).get("copy_meta") or {}
        else:
            paths = build_workspace(workspace)
            copy_meta = json.loads(paths.get("copy_meta", "{}"))
            rust_meta = json.loads(paths.get("rust_meta", "{}"))
        selected_cases = tuple(
            item for item in CASE_SPECS if not only or item[0] in only
        )
        summaries: dict[str, dict[str, object]] = {}
        once = {
            "A6_process_tree",
            "S1_import_cycle_tracemalloc",
            "S2_resolve_solo_tracemalloc",
        }
        for name, _kind, _store in selected_cases:
            print(f"case {name} begin", flush=True)
            runs: list[dict[str, object]] = []
            repeats = 1 if name in once else WARMUPS + MEASURED_RUNS
            if name == "A1_bare_interpreter":
                script = (
                    "import gc,json,os,resource,subprocess,sys,time\n"
                    "gc.collect()\n"
                    "rss=lambda:int(subprocess.check_output(['ps','-o','rss=','-p',str(os.getpid())],text=True).split()[0])*1024\n"
                    "before=rss(); u0=resource.getrusage(resource.RUSAGE_SELF); t0=time.perf_counter()\n"
                    "ok=True\n"
                    "gc.collect(); u1=resource.getrusage(resource.RUSAGE_SELF)\n"
                    "raw=u1.ru_maxrss; maxrss=raw if sys.platform=='darwin' else raw*1024\n"
                    "print(json.dumps({'pid':os.getpid(),'status':'OK','rss_before_bytes':before,'rss_after_bytes':rss(),'ru_maxrss_bytes':maxrss,'user_cpu_seconds':u1.ru_utime-u0.ru_utime,'system_cpu_seconds':u1.ru_stime-u0.ru_stime,'wall_seconds':time.perf_counter()-t0,'tracemalloc_enabled':False,'tracemalloc_peak_bytes':'UNOBSERVED','outcome':{'imported':False}}))\n"
                )
                for _ in range(repeats):
                    runs.append(
                        run_timed_command([sys.executable, "-c", script], env)
                    )
            elif name.startswith("R0") or name.startswith("R1_rust_observation") or name == "R2_bounded_candidate":
                binary = paths.get("rust_binary")
                if not binary:
                    runs.append({"status": UNOBSERVED, "reason": "research rust binary missing", "outcome": {}})
                elif name.startswith("R0"):
                    for _ in range(repeats):
                        runs.append(run_timed_command([binary, "r0"], env))
                elif name == "R2_bounded_candidate":
                    store = paths.get("proving_copied") or paths.get("proving_scaled")
                    spec_path = paths.get("r2_spec_copied") or paths.get("r2_spec_scaled")
                    if not store or not spec_path:
                        runs.append({"status": UNOBSERVED, "reason": "r2 store or spec missing", "outcome": {}})
                    else:
                        for _ in range(repeats):
                            runs.append(
                                run_timed_command(
                                    [binary, "r2", "--db", store, "--spec", spec_path],
                                    env,
                                )
                            )
                else:
                    store = paths.get("proving_copied") or paths.get("proving_scaled")
                    cutoff = paths.get("cutoff_copied") or paths.get("cutoff_scaled")
                    if not store or not cutoff:
                        runs.append({"status": UNOBSERVED, "reason": "scan store missing", "outcome": {}})
                    else:
                        for _ in range(repeats):
                            runs.append(
                                run_timed_command(
                                    [binary, "r1", "--db", store, "--cutoff", cutoff],
                                    env,
                                )
                            )
            else:
                extra = ("--tracemalloc",) if name.startswith("S") else ()
                for _ in range(repeats):
                    runs.append(
                        run_child_case(
                            executable=sys.executable,
                            module="newsroom.research.issue_898_ram_cpu",
                            case=name,
                            workspace=workspace,
                            env=env,
                            extra_args=extra,
                        )
                    )
            summaries[name] = summarise_case(name, runs)
            print(
                f"case {name} peak={summaries[name]['max_peak_rss_bytes']} "
                f"cpu={summaries[name]['median_cpu_seconds']}",
                flush=True,
            )
        workspace_digest = hashlib.sha256(
            json.dumps(sorted(paths.items()), sort_keys=True).encode()
        ).hexdigest()
        if merge_into is not None:
            base = json.loads(merge_into.read_text(encoding="utf-8"))
            merged = dict(base.get("cases") or {})
            merged.update(summaries)
            summaries = merged
            copy_meta = base.get("copy_meta") or copy_meta
            if not rust_meta:
                rust_meta = base.get("rust_meta") or {}
        packet = assemble_packet(
            summaries=summaries,
            copy_meta=copy_meta,
            rust_meta=rust_meta,
            workspace_digest="sha256:" + workspace_digest,
        )
        if merge_into is not None:
            base_head = json.loads(merge_into.read_text(encoding="utf-8")).get(
                "inspection_head"
            )
            packet["r1_inspection_head"] = base_head
            packet["r2_measurement_head"] = packet.get("inspection_head")
            packet["inspection_head"] = base_head
            packet["r2_reused_snapshot"] = True
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(packet, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return packet
    finally:
        if close_workspace:
            shutil.rmtree(workspace, ignore_errors=True)


def load_workspace(path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for item in path.glob("proving-*.sqlite3"):
        mapping[f"proving_{item.stem.removeprefix('proving-')}"] = str(item)
    for item in path.glob("unpublished-*.sqlite3"):
        mapping[f"unpublished_{item.stem.removeprefix('unpublished-')}"] = str(item)
    for item in path.glob("event-*.json"):
        mapping[f"event_{item.stem.removeprefix('event-')}"] = str(item)
    for item in path.glob("r2-spec-*.json"):
        mapping[f"r2_spec_{item.stem.removeprefix('r2-spec-')}"] = str(item)
    rust_binary = path / "issue-898-ram-cpu"
    if rust_binary.is_file():
        mapping["rust_binary"] = str(rust_binary)
    copied_event = path / "event-copied.json"
    if copied_event.is_file():
        spec = json.loads(copied_event.read_text(encoding="utf-8"))
        mapping["clock_copied"] = str(spec.get("clock") or utc_text(datetime.now(tz=UTC)))
        mapping["cutoff_copied"] = raw_http_cutoff(parse_utc(mapping["clock_copied"]))
    for kind in (
        "solo",
        "representative",
        "times10",
        "scaled",
        "json",
        "rss",
        "atom",
        "large",
        "malformed",
        "empty",
    ):
        mapping[f"cutoff_{kind}"] = raw_http_cutoff(CLOCK)
    return mapping


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Issue #898 Mini RAM/CPU packet")
    parser.add_argument("--case")
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--tracemalloc", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/research/2026-09-02-issue-898-ram-cpu-measurements.json"),
    )
    parser.add_argument(
        "--only",
        help="comma-separated case names; omit to run the full packet",
    )
    parser.add_argument(
        "--merge-into",
        type=Path,
        help="existing packet JSON whose other cases are retained",
    )
    parser.add_argument(
        "--refresh-copied-event",
        action="store_true",
        help="rebuild copied event identity and R2 spec from an existing snapshot",
    )
    args = parser.parse_args(argv)
    if args.case:
        if args.workspace is None:
            raise SystemExit("--workspace is required with --case")
        payload = run_named_case(
            args.case,
            load_workspace(args.workspace),
            use_tracemalloc=args.tracemalloc,
        )
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return 0
    only = tuple(item for item in (args.only or "").split(",") if item)
    orchestrate(
        output=args.output,
        workspace=args.workspace,
        only=only,
        merge_into=args.merge_into,
        refresh_copied_event=args.refresh_copied_event,
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
