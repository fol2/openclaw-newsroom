"""Immutable SQLite snapshots for operational evidence readers."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from newsroom.control_plane.sqlite_profile import apply_control_plane_sqlite_profile


class ReadOnlySnapshotError(RuntimeError):
    """A stable read-only snapshot could not be established."""


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


@contextmanager
def read_only_snapshot(path: str | Path) -> Iterator[ReadOnlySnapshot]:
    """Expose a transaction-consistent logical SQLite snapshot for reading.

    Isolation is a read transaction on the source, not a tempfile copy.
    Copying the live Increment 4 store is a multi-GiB backup and misses
    the one-minute operational seal bound.
    """

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise ReadOnlySnapshotError(f"store does not exist: {resolved}")
    connection = sqlite3.connect(f"{resolved.as_uri()}?mode=ro", uri=True)
    try:
        apply_control_plane_sqlite_profile(connection, query_only=True, wal=None)
        connection.execute("BEGIN")
        connection.execute("SELECT name FROM sqlite_schema LIMIT 1").fetchone()
        yield ReadOnlySnapshot(
            connection=connection,
            source_path=str(resolved),
            source_files=_source_file_observations(resolved),
            snapshot_files=(
                {
                    "name": resolved.name,
                    "size": resolved.stat().st_size,
                    "copy_omitted": True,
                },
            ),
        )
    finally:
        if connection.in_transaction:
            connection.rollback()
        connection.close()
