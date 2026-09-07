from __future__ import annotations

import inspect
import sqlite3

import pytest

from newsroom.authority.migrations import SCHEMA_VERSION
from newsroom.authority.persistence import AuthorityPersistenceError
from newsroom.authority.source_registry_migrations import (
    SOURCE_REGISTRY_MIGRATION_CHECKSUM,
    SOURCE_REGISTRY_MIGRATION_NAME,
    SOURCE_REGISTRY_SCHEMA_VERSION,
)

from .source_3a_helpers import (
    DEFINITION_ID,
    definition_request,
    open_source_system,
    proof,
    version_request,
)


def test_source_system_open_skips_page_walk_pragmas() -> None:
    from newsroom.authority._event_store_base import _EventStoreBase

    source = inspect.getsource(_EventStoreBase._validate_schema_and_integrity)
    assert 'execute("PRAGMA quick_check")' not in source
    assert 'execute("PRAGMA foreign_key_check")' not in source
    assert 'execute("PRAGMA integrity_check")' not in source


def test_increment4_open_skips_full_table_row_decode() -> None:
    from newsroom.authority._entity_store_integrity import _EntityIntegrityMixin
    from newsroom.authority._event_store_base import _EventStoreBase
    from newsroom.authority._graphiti_increment4_system import (
        _GraphitiIncrement4AuthorityStore,
    )

    assert "return False" in inspect.getsource(
        _GraphitiIncrement4AuthorityStore._should_validate_row_integrity
    )
    assert "return True" in inspect.getsource(
        _EventStoreBase._should_validate_row_integrity
    )
    entity = inspect.getsource(_EntityIntegrityMixin._validate_schema_and_integrity)
    assert "if not self._should_validate_row_integrity():" in entity
    base = inspect.getsource(_EventStoreBase._validate_schema_and_integrity)
    assert "if self._should_validate_row_integrity():" in base
    assert "_validate_immutable_records" in base


def test_checked_source_registry_migration_is_retained_in_v11(
    tmp_path,
) -> None:
    database = tmp_path / "authority.sqlite3"
    system = open_source_system(database)
    system.close()

    conn = sqlite3.connect(database)
    try:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == (
            SCHEMA_VERSION
        )
        row = conn.execute(
            "SELECT name,checksum FROM authority_migrations WHERE version=?",
            (SOURCE_REGISTRY_SCHEMA_VERSION,),
        ).fetchone()
        assert row == (
            SOURCE_REGISTRY_MIGRATION_NAME,
            SOURCE_REGISTRY_MIGRATION_CHECKSUM,
        )
    finally:
        conn.close()

    open_source_system(database).close()


def test_immutable_source_rows_reject_update_and_delete(tmp_path) -> None:
    database = tmp_path / "authority.sqlite3"
    system = open_source_system(database)
    system.sources.register_definition(definition_request(), proof=proof())
    system.close()

    conn = sqlite3.connect(database)
    try:
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            conn.execute(
                "UPDATE source_definitions SET name='changed' WHERE definition_id=?",
                (str(DEFINITION_ID),),
            )
        with pytest.raises(sqlite3.IntegrityError, match="retained"):
            conn.execute(
                "DELETE FROM source_definitions WHERE definition_id=?",
                (str(DEFINITION_ID),),
            )
    finally:
        conn.close()


def test_startup_detects_source_canonical_tampering(tmp_path) -> None:
    database = tmp_path / "authority.sqlite3"
    system = open_source_system(database)
    system.sources.register_definition(definition_request(), proof=proof())
    system.sources.record_definition_version(
        version_request(), proof=proof()
    )
    system.close()

    conn = sqlite3.connect(database)
    try:
        trigger_sql = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='trigger' AND name=?",
            ("immutable_source_definition_update",),
        ).fetchone()[0]
        conn.execute("DROP TRIGGER immutable_source_definition_update")
        conn.execute(
            "UPDATE source_definitions SET canonical_digest=? "
            "WHERE definition_id=?",
            ("sha256:" + "0" * 64, str(DEFINITION_ID)),
        )
        conn.execute(trigger_sql)
        conn.commit()
    finally:
        conn.close()

    with pytest.raises(AuthorityPersistenceError, match="canonical digest"):
        open_source_system(database)
