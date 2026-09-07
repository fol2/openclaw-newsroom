"""Immutable SQLite snapshots for operational evidence readers."""

from __future__ import annotations

import hashlib
import sqlite3
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from newsroom.control_plane.sqlite_profile import apply_control_plane_sqlite_profile


class ReadOnlySnapshotError(RuntimeError):
    """A stable read-only snapshot could not be established."""


def _digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _source_file_observations(database: Path) -> tuple[dict[str, object], ...]:
    stat = database.stat()
    return (
        {
            "name": database.name,
            "device": stat.st_dev,
            "inode": stat.st_ino,
        },
    )


@dataclass(frozen=True, slots=True)
class ReadOnlySnapshot:
    connection: sqlite3.Connection
    source_path: str
    source_files: tuple[dict[str, object], ...]
    snapshot_files: tuple[dict[str, object], ...]
    logical_content_digest: str


@contextmanager
def read_only_snapshot(path: str | Path) -> Iterator[ReadOnlySnapshot]:
    """Expose a transaction-consistent logical SQLite snapshot for reading."""

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise ReadOnlySnapshotError(f"store does not exist: {resolved}")
    with tempfile.TemporaryDirectory(prefix="newsroom-readonly-") as scratch:
        copied = Path(scratch) / resolved.name
        source = sqlite3.connect(f"{resolved.as_uri()}?mode=ro", uri=True)
        try:
            apply_control_plane_sqlite_profile(source, query_only=True, wal=None)
            source.execute("BEGIN")
            source.execute("SELECT name FROM sqlite_schema LIMIT 1").fetchone()
            destination = sqlite3.connect(copied)
            try:
                source.backup(destination)
            finally:
                destination.close()
                source.rollback()
            file_digest = _digest_file(copied)
            copied_identity = {
                "name": copied.name,
                "size": copied.stat().st_size,
                "sha256": file_digest,
            }
            connection = sqlite3.connect(
                f"{copied.as_uri()}?mode=ro&immutable=1", uri=True
            )
            try:
                apply_control_plane_sqlite_profile(
                    connection, query_only=True, wal=False
                )
                yield ReadOnlySnapshot(
                    connection=connection,
                    source_path=str(resolved),
                    source_files=_source_file_observations(resolved),
                    snapshot_files=(copied_identity,),
                    logical_content_digest=f"sha256:{file_digest}",
                )
            finally:
                connection.close()
        finally:
            source.close()
