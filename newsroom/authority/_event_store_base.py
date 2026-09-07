from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import sqlite3
import stat
import threading
from typing import Any, Iterator

try:
    import fcntl
except ImportError:  # pragma: no cover - supported runtime/CI is POSIX
    fcntl = None  # type: ignore[assignment]

from ._capability import _CapabilityIssuer
from .migrations import (
    EXPECTED_MIGRATION_HISTORY,
    EXPECTED_SCHEMA_FINGERPRINT,
    SCHEMA_VERSION,
    apply_pending_migrations,
    prepare_pending_migration_backup,
    schema_fingerprint,
)
from .models import CommittedCommandIdentity
from .persistence import (
    AuthorityPersistenceError,
    AuthoritySchemaError,
    AuthorityWriterBusy,
)
from .policy import CommandRegistry, PayloadSchemaRegistry
from .types import ObjectAdmissionId, PayloadMode, UtcTimestamp, require_token


class _EventStoreBase:
    """SQLite lifecycle, migration, validation and writer ownership."""

    def __init__(
        self,
        path: Path,
        *,
        issuer: _CapabilityIssuer,
        command_registry: CommandRegistry,
        payload_schemas: PayloadSchemaRegistry,
        command_service_version: str,
        busy_timeout_ms: int = 5_000,
        clock: Any = UtcTimestamp.now,
    ) -> None:
        if fcntl is None:
            raise AuthoritySchemaError("POSIX advisory locking is required")
        self.path = Path(path)
        if self.path.is_symlink():
            raise AuthoritySchemaError("authority database path cannot be a symlink")
        if busy_timeout_ms <= 0:
            raise ValueError("busy_timeout_ms must be positive")
        require_token(command_service_version, field="command_service_version")
        self._issuer = issuer
        self._command_registry = command_registry
        self._payload_schemas = payload_schemas
        self._command_service_version = command_service_version
        self._busy_timeout_ms = busy_timeout_ms
        self._clock = clock
        self._lock = threading.RLock()
        self._closed = False
        self._lock_fd: int | None = None
        self._conn: sqlite3.Connection | None = None

        self._secure_directory(self.path.parent)
        existed = self.path.exists()
        if existed:
            self._validate_owned_file(self.path)
        try:
            self._acquire_writer_lock()
            self._conn = sqlite3.connect(
                self.path,
                isolation_level=None,
                timeout=busy_timeout_ms / 1000,
                check_same_thread=False,
            )
            self._conn.row_factory = sqlite3.Row
            self._configure_connection()
            if not existed:
                os.chmod(self.path, 0o600)
            self._validate_owned_file(self.path)
            self._migrate_or_validate()
        except Exception:
            self.close()
            raise

    @staticmethod
    def _secure_directory(path: Path) -> None:
        if path.exists():
            if path.is_symlink() or not path.is_dir():
                raise AuthoritySchemaError(
                    "authority database parent must be a real directory"
                )
        else:
            path.mkdir(parents=True, mode=0o700)
            os.chmod(path, 0o700)
        info = path.stat()
        if hasattr(os, "getuid") and info.st_uid != os.getuid():
            raise AuthoritySchemaError(
                "authority directory must be owned by the writer"
            )
        if stat.S_IMODE(info.st_mode) & 0o077:
            raise AuthoritySchemaError(
                "authority directory cannot grant group or other permissions"
            )

    @staticmethod
    def _validate_owned_file(path: Path) -> None:
        if path.is_symlink() or not path.is_file():
            raise AuthoritySchemaError(
                "authority database must be a regular file"
            )
        info = path.stat()
        if hasattr(os, "getuid") and info.st_uid != os.getuid():
            raise AuthoritySchemaError(
                "authority database must be owned by the writer"
            )
        if stat.S_IMODE(info.st_mode) & 0o077:
            raise AuthoritySchemaError(
                "authority database cannot grant group or other permissions"
            )

    def _acquire_writer_lock(self) -> None:
        lock_path = self.path.with_name(self.path.name + ".writer.lock")
        if lock_path.is_symlink():
            raise AuthoritySchemaError("writer lock path cannot be a symlink")
        descriptor = os.open(
            lock_path,
            os.O_CREAT | os.O_RDWR | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
        os.fchmod(descriptor, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            os.close(descriptor)
            raise AuthorityWriterBusy(
                "another authority writer is active"
            ) from exc
        self._lock_fd = descriptor

    @property
    def _connection(self) -> sqlite3.Connection:
        if self._closed or self._conn is None:
            raise AuthorityPersistenceError("authority store is closed")
        return self._conn

    def _configure_connection(self) -> None:
        conn = self._connection
        conn.execute("PRAGMA foreign_keys=ON")
        mode = str(
            conn.execute("PRAGMA journal_mode=WAL").fetchone()[0]
        ).lower()
        if mode != "wal":
            raise AuthoritySchemaError(
                f"journal_mode WAL unavailable: {mode}"
            )
        conn.execute("PRAGMA synchronous=FULL")
        conn.execute(f"PRAGMA busy_timeout={int(self._busy_timeout_ms)}")

    def _table_names(self) -> set[str]:
        return {
            str(row[0])
            for row in self._connection.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        }

    def _migrate_or_validate(self) -> None:
        conn = self._connection
        version = int(conn.execute("PRAGMA user_version").fetchone()[0])
        tables = self._table_names()
        if version > SCHEMA_VERSION:
            raise AuthoritySchemaError(
                f"database schema {version} is newer than supported "
                f"{SCHEMA_VERSION}"
            )
        if version == 0 and tables:
            raise AuthoritySchemaError(
                "refusing to adopt a non-empty unversioned authority database"
            )
        if version < SCHEMA_VERSION:
            prepare_pending_migration_backup(conn)
            apply_pending_migrations(
                conn, applied_at=self._clock().to_text()
            )
        self._validate_schema_and_integrity()

    def _validate_schema_and_integrity(self) -> None:
        conn = self._connection
        version = int(conn.execute("PRAGMA user_version").fetchone()[0])
        if version != SCHEMA_VERSION:
            raise AuthoritySchemaError(
                f"database schema {version} does not match {SCHEMA_VERSION}"
            )
        rows = conn.execute(
            "SELECT version,name,checksum FROM authority_migrations "
            "ORDER BY version"
        ).fetchall()
        history = tuple(
            (int(row["version"]), str(row["name"]), str(row["checksum"]))
            for row in rows
        )
        if history != EXPECTED_MIGRATION_HISTORY:
            raise AuthoritySchemaError(
                f"authority migration history mismatch: {history!r}"
            )
        if schema_fingerprint(conn) != EXPECTED_SCHEMA_FINGERPRINT:
            raise AuthoritySchemaError(
                "authority schema fingerprint mismatch"
            )
        # Skip full-file SQLite page walks. On the live Increment 4 store
        # those take ~25 minutes and miss the one-minute seal bound.
        if not bool(conn.execute("PRAGMA foreign_keys").fetchone()[0]):
            raise AuthoritySchemaError(
                "SQLite foreign keys are not enabled"
            )
        if (
            str(conn.execute("PRAGMA journal_mode").fetchone()[0]).lower()
            != "wal"
        ):
            raise AuthoritySchemaError("SQLite WAL mode is not active")
        if int(conn.execute("PRAGMA synchronous").fetchone()[0]) != 2:
            raise AuthoritySchemaError(
                "SQLite synchronous=FULL is not active"
            )
        self._validate_relational_invariants(conn)
        self._validate_immutable_records(conn)
        self._validate_registry_coverage(conn)

    @staticmethod
    def _validate_relational_invariants(conn: sqlite3.Connection) -> None:
        missing_head = conn.execute(
            "SELECT a.aggregate_type,a.aggregate_id,a.current_version "
            "FROM authority_aggregates a "
            "LEFT JOIN authority_aggregate_versions v "
            "ON v.aggregate_type=a.aggregate_type "
            "AND v.aggregate_id=a.aggregate_id "
            "AND v.aggregate_version=a.current_version "
            "WHERE v.aggregate_version IS NULL LIMIT 1"
        ).fetchone()
        if missing_head is not None:
            raise AuthoritySchemaError(
                "aggregate head does not reference an exact version"
            )

        incomplete = conn.execute(
            "SELECT c.command_id,COUNT(DISTINCT v.command_id) AS versions,"
            "COUNT(DISTINCT e.command_id) AS events,"
            "COUNT(DISTINCT a.command_id) AS audits "
            "FROM authority_commands c "
            "LEFT JOIN authority_aggregate_versions v "
            "ON v.command_id=c.command_id "
            "LEFT JOIN ledger_events e ON e.command_id=c.command_id "
            "LEFT JOIN authority_audit_events a ON a.command_id=c.command_id "
            "GROUP BY c.command_id "
            "HAVING versions != 1 OR events != 1 OR audits != 1 LIMIT 1"
        ).fetchone()
        if incomplete is not None:
            raise AuthoritySchemaError(
                "each command must own one version, audit and event"
            )

        mismatch = conn.execute(
            "SELECT e.event_id FROM ledger_events e "
            "JOIN authority_commands c ON c.command_id=e.command_id "
            "JOIN authority_payloads p ON p.payload_id=e.payload_id "
            "WHERE e.producer_version != c.producer_version "
            "OR e.command_definition_version != "
            "c.command_definition_version "
            "OR e.command_definition_digest != "
            "c.command_definition_digest "
            "OR e.payload_id != c.payload_id "
            "OR e.payload_mode != p.mode "
            "OR e.payload_schema_version != p.schema_version "
            "OR e.payload_schema_contract_version != "
            "p.schema_contract_version "
            "OR e.payload_schema_contract_digest != "
            "p.schema_contract_digest "
            "OR e.payload_canonicalizer_version != "
            "p.canonicalizer_implementation_version "
            "OR e.payload_digest != p.payload_digest "
            "OR NOT (e.object_admission_id IS p.object_admission_id) "
            "LIMIT 1"
        ).fetchone()
        if mismatch is not None:
            raise AuthoritySchemaError(
                "event routing envelope does not match immutable authority"
            )

    def _validate_immutable_records(
        self, conn: sqlite3.Connection
    ) -> None:
        for row in conn.execute(
            "SELECT * FROM payload_schema_contracts"
        ).fetchall():
            self._schema_record_from_row(row)
        for row in conn.execute(
            "SELECT * FROM command_definitions"
        ).fetchall():
            self._definition_record_from_row(row)
        for row in conn.execute(
            "SELECT * FROM authentication_contexts"
        ).fetchall():
            self._authentication_record_from_row(row)
        for row in conn.execute(
            "SELECT * FROM authorization_requests"
        ).fetchall():
            self._request_record_from_row(row)
        for row in conn.execute(
            "SELECT * FROM authorization_decisions"
        ).fetchall():
            self._decision_record_from_row(row)
        for row in conn.execute(
            "SELECT command_id,result_digest,result_bytes "
            "FROM authority_commands"
        ).fetchall():
            self._decode_result(
                bytes(row["result_bytes"]),
                str(row["result_digest"]),
                replayed=False,
            )

    def _validate_registry_coverage(
        self, conn: sqlite3.Connection
    ) -> None:
        for row in conn.execute(
            "SELECT command_type,definition_version,definition_digest "
            "FROM command_definitions"
        ).fetchall():
            self._command_registry.resolve_exact(
                str(row["command_type"]),
                str(row["definition_version"]),
                str(row["definition_digest"]),
            )
        for row in conn.execute(
            "SELECT schema_version,payload_mode,contract_version,"
            "contract_digest,canonicalizer_implementation_version "
            "FROM payload_schema_contracts"
        ).fetchall():
            self._payload_schemas.resolve_exact(
                str(row["schema_version"]),
                PayloadMode(str(row["payload_mode"])),
                str(row["contract_version"]),
                str(row["contract_digest"]),
                str(row["canonicalizer_implementation_version"]),
            )

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        conn = self._connection
        try:
            conn.execute("BEGIN IMMEDIATE")
            yield conn
            conn.execute("COMMIT")
        except sqlite3.OperationalError as exc:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            lowered = str(exc).lower()
            if "locked" in lowered or "busy" in lowered:
                raise AuthorityWriterBusy(
                    "authority writer remained busy"
                ) from exc
            raise AuthorityPersistenceError(str(exc)) from exc
        except sqlite3.DatabaseError as exc:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise AuthorityPersistenceError(str(exc)) from exc
        except Exception:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._conn is not None:
                self._conn.close()
                self._conn = None
            if self._lock_fd is not None:
                try:
                    fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
                finally:
                    os.close(self._lock_fd)
                    self._lock_fd = None

    def __enter__(self) -> _EventStoreBase:
        return self

    def __exit__(
        self, exc_type: object, exc: object, tb: object
    ) -> None:
        self.close()

    def find(
        self, *, idempotency_namespace: str, idempotency_key: str
    ) -> CommittedCommandIdentity | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT c.command_id,c.command_type,"
                "c.command_definition_version,"
                "c.command_definition_digest,"
                "c.idempotency_namespace,c.idempotency_key,"
                "c.stable_semantic_request_digest,"
                "p.mode AS payload_mode,p.payload_digest,"
                "p.object_admission_id,"
                "a.principal_id,a.authority_domain "
                "FROM authority_commands c "
                "JOIN authority_payloads p ON p.payload_id=c.payload_id "
                "JOIN authentication_contexts a "
                "ON a.authentication_context_id="
                "c.authentication_context_id "
                "WHERE c.idempotency_namespace=? "
                "AND c.idempotency_key=?",
                (idempotency_namespace, idempotency_key),
            ).fetchone()
            if row is None:
                return None
            return CommittedCommandIdentity(
                command_id=str(row["command_id"]),
                authority_domain=str(row["authority_domain"]),
                principal_id=str(row["principal_id"]),
                command_type=str(row["command_type"]),
                idempotency_namespace=str(
                    row["idempotency_namespace"]
                ),
                idempotency_key=str(row["idempotency_key"]),
                command_definition_version=str(
                    row["command_definition_version"]
                ),
                command_definition_digest=str(
                    row["command_definition_digest"]
                ),
                stable_semantic_request_digest=str(
                    row["stable_semantic_request_digest"]
                ),
                payload_mode=str(row["payload_mode"]),
                payload_digest=str(row["payload_digest"]),
                object_admission_id=(
                    None
                    if row["object_admission_id"] is None
                    else ObjectAdmissionId.parse(
                        str(row["object_admission_id"])
                    )
                ),
            )
