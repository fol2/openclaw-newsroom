from __future__ import annotations

import json
import sqlite3
from typing import Any

from newsroom.authority.canonical import canonical_json_bytes, digest_bytes
from newsroom.authority.persistence import AuthorityPersistenceError, AuthoritySchemaError
from newsroom.sources.definition_models import SourceDefinitionVersionRequest
from newsroom.sources.types import EXECUTION_AUTHORITY_DISABLED

from ._source_registry_decoding import canonical_object


_SOURCE_TABLES = frozenset(
    {
        "source_definitions",
        "source_definition_versions",
        "source_definition_version_heads",
        "source_version_roles",
        "source_version_portfolio_functions",
        "source_version_gaps",
        "source_version_coverage_mappings",
        "source_version_dependencies",
        "source_items",
        "source_locator_continuity_decisions",
        "source_revisions",
        "discovery_representations",
        "discovery_occurrences",
    }
)


class _SourceRegistryIntegrityMixin:
    def _validate_schema_and_integrity(self) -> None:
        super()._validate_schema_and_integrity()
        if not self._should_validate_row_integrity():
            return
        self._validate_source_registry_integrity(self._connection)

    def _validate_source_registry_integrity(
        self, conn: sqlite3.Connection
    ) -> None:
        missing = _SOURCE_TABLES - self._table_names()
        if missing:
            raise AuthoritySchemaError(
                f"source registry schema tables are missing: {sorted(missing)!r}"
            )
        if conn.execute(
            "SELECT 1 FROM source_definition_versions "
            "WHERE execution_authority!=? LIMIT 1",
            (EXECUTION_AUTHORITY_DISABLED,),
        ).fetchone() is not None:
            raise AuthoritySchemaError(
                "source registry contains executable source authority"
            )
        self._validate_all_source_rows(conn)
        self._validate_source_version_heads(conn)
        self._validate_source_event_coverage(conn)

    def _validate_all_source_rows(self, conn: sqlite3.Connection) -> None:
        loaders = (
            ("source_definitions", self._source_definition_from_row),
            (
                "source_definition_versions",
                self._source_version_from_row,
            ),
            ("source_items", self._source_item_from_row),
            (
                "source_locator_continuity_decisions",
                self._locator_decision_from_row,
            ),
            ("source_revisions", self._source_revision_from_row),
            (
                "discovery_representations",
                self._representation_from_row,
            ),
            ("discovery_occurrences", self._occurrence_from_row),
        )
        for table, loader in loaders:
            for row in conn.execute(f"SELECT * FROM {table}").fetchall():
                loader(conn, row, replayed=False)

    def _validate_source_version_heads(
        self, conn: sqlite3.Connection
    ) -> None:
        missing = conn.execute(
            "SELECT v.definition_id FROM source_definition_versions v "
            "LEFT JOIN source_definition_version_heads h "
            "ON h.definition_id=v.definition_id "
            "WHERE h.definition_id IS NULL LIMIT 1"
        ).fetchone()
        if missing is not None:
            raise AuthoritySchemaError(
                "source definition version lacks a retained version head"
            )
        for head in conn.execute(
            "SELECT * FROM source_definition_version_heads"
        ).fetchall():
            rows = conn.execute(
                "SELECT version_id,version_number,previous_version_id "
                "FROM source_definition_versions WHERE definition_id=? "
                "ORDER BY version_number",
                (str(head["definition_id"]),),
            ).fetchall()
            if not rows:
                raise AuthoritySchemaError(
                    "source version head has no retained versions"
                )
            expected = 1
            previous: str | None = None
            for row in rows:
                if (
                    int(row["version_number"]) != expected
                    or (
                        None
                        if row["previous_version_id"] is None
                        else str(row["previous_version_id"])
                    )
                    != previous
                ):
                    raise AuthoritySchemaError(
                        "source version chain is not contiguous"
                    )
                previous = str(row["version_id"])
                expected += 1
            if (
                int(head["current_version_number"]) != len(rows)
                or str(head["current_version_id"]) != previous
            ):
                raise AuthoritySchemaError(
                    "source version head differs from the immutable chain"
                )

    @staticmethod
    def _validate_source_event_coverage(
        conn: sqlite3.Connection,
    ) -> None:
        specs = (
            ("source.definition.registered", "source_definitions"),
            (
                "source.definition.version.recorded",
                "source_definition_versions",
            ),
            ("source.item.registered", "source_items"),
            (
                "source.locator.continuity.decided",
                "source_locator_continuity_decisions",
            ),
            ("source.revision.recorded", "source_revisions"),
            (
                "discovery.representation.recorded",
                "discovery_representations",
            ),
            ("discovery.occurrence.recorded", "discovery_occurrences"),
        )
        for event_type, table in specs:
            missing = conn.execute(
                f"SELECT e.event_id FROM ledger_events e "
                f"LEFT JOIN {table} r ON r.authority_event_id=e.event_id "
                "WHERE e.event_type=? AND r.authority_event_id IS NULL LIMIT 1",
                (event_type,),
            ).fetchone()
            if missing is not None:
                raise AuthoritySchemaError(
                    f"{event_type} has no exact source registry record"
                )

    def _validate_source_version_children(
        self,
        conn: sqlite3.Connection,
        *,
        row: sqlite3.Row,
        request: SourceDefinitionVersionRequest,
    ) -> None:
        version_id = str(request.version_id)
        roles = conn.execute(
            "SELECT * FROM source_version_roles WHERE version_id=? "
            "ORDER BY role",
            (version_id,),
        ).fetchall()
        if len(roles) != len(request.roles):
            raise AuthorityPersistenceError(
                "source version role count differs from canonical bytes"
            )
        for stored, expected in zip(roles, request.roles, strict=True):
            value = self._canonical_child(stored, identity="source role")
            if (
                value != expected.canonical_value()
                or str(stored["role"]) != expected.role.value
                or str(stored["purpose"]) != expected.purpose
                or self._canonical_list(
                    stored["limitations_bytes"], identity="role limitations"
                )
                != list(expected.limitations)
            ):
                raise AuthorityPersistenceError(
                    "source role differs from canonical source version"
                )

        functions = tuple(
            str(item["portfolio_function"])
            for item in conn.execute(
                "SELECT portfolio_function "
                "FROM source_version_portfolio_functions WHERE version_id=? "
                "ORDER BY portfolio_function",
                (version_id,),
            ).fetchall()
        )
        if functions != tuple(item.value for item in request.portfolio_functions):
            raise AuthorityPersistenceError(
                "source portfolio functions differ from canonical source version"
            )

        gaps = conn.execute(
            "SELECT * FROM source_version_gaps WHERE version_id=? ORDER BY gap_id",
            (version_id,),
        ).fetchall()
        if len(gaps) != len(request.explicit_gaps):
            raise AuthorityPersistenceError(
                "source gaps differ from canonical source version"
            )
        for stored, expected in zip(gaps, request.explicit_gaps, strict=True):
            if self._canonical_child(stored, identity="source gap") != expected.canonical_value():
                raise AuthorityPersistenceError(
                    "source gaps differ from canonical source version"
                )
            self._require_normalized_columns(
                stored,
                {
                    "gap_id": expected.gap_id,
                    "gap_class": expected.gap_class,
                    "description": expected.description,
                    "launch_blocking": int(expected.launch_blocking),
                },
                identity="source gap",
            )

        mappings = conn.execute(
            "SELECT * FROM source_version_coverage_mappings "
            "WHERE version_id=? ORDER BY obligation_id,responsibility,contribution",
            (version_id,),
        ).fetchall()
        if len(mappings) != len(request.coverage_mappings):
            raise AuthorityPersistenceError(
                "coverage mappings differ from canonical source version"
            )
        for stored, expected in zip(mappings, request.coverage_mappings, strict=True):
            if self._canonical_child(stored, identity="coverage mapping") != expected.canonical_value():
                raise AuthorityPersistenceError(
                    "coverage mappings differ from canonical source version"
                )
            self._require_normalized_columns(
                stored,
                {
                    "obligation_id": expected.obligation_id,
                    "responsibility": expected.responsibility.value,
                    "contribution": expected.contribution.value,
                    "explicit_gap_id": expected.explicit_gap_id,
                },
                identity="coverage mapping",
            )
            self._require_canonical_blob(
                stored, "geographies_bytes", list(expected.geographies), identity="coverage mapping"
            )
            self._require_canonical_blob(
                stored, "languages_bytes", list(expected.languages), identity="coverage mapping"
            )
            self._require_canonical_blob(
                stored, "limitations_bytes", list(expected.limitations), identity="coverage mapping"
            )

        dependencies = conn.execute(
            "SELECT * FROM source_version_dependencies "
            "WHERE version_id=? ORDER BY dependency_id",
            (version_id,),
        ).fetchall()
        if len(dependencies) != len(request.dependencies):
            raise AuthorityPersistenceError(
                "source dependencies differ from canonical source version"
            )
        for stored, expected in zip(dependencies, request.dependencies, strict=True):
            if self._canonical_child(stored, identity="source dependency") != expected.canonical_value():
                raise AuthorityPersistenceError(
                    "source dependencies differ from canonical source version"
                )
            self._require_normalized_columns(
                stored,
                {
                    "dependency_id": expected.dependency_id,
                    "dependency_kind": expected.kind.value,
                    "description": expected.description,
                    "upstream_definition_id": (
                        None
                        if expected.upstream_source_definition_id is None
                        else str(expected.upstream_source_definition_id)
                    ),
                },
                identity="source dependency",
            )

        expected_columns: dict[str, object] = {
            "adapter_policy_id": request.adapter_contract.policy_id,
            "adapter_policy_version": request.adapter_contract.policy_version,
            "rights_decision_id": request.rights.rights_decision_id,
            "rights_policy_version": request.rights.rights_policy_version,
            "allowed_use": request.rights.allowed_use,
            "source_retention_scope": request.rights.retention_scope,
            "observation_model": request.observation_model.value,
            "baseline_policy_id": request.baseline_policy.reference.policy_id,
            "baseline_policy_version": (
                request.baseline_policy.reference.policy_version
            ),
            "baseline_kind": request.baseline_policy.kind.value,
            "baseline_freshness_seconds": (
                request.baseline_policy.freshness_window_seconds
            ),
            "baseline_reset_requires_decision": int(
                request.baseline_policy.reset_requires_decision
            ),
            "baseline_notes": request.baseline_policy.notes,
            "item_identity_policy_id": request.item_identity_policy.policy_id,
            "item_identity_policy_version": (
                request.item_identity_policy.policy_version
            ),
            "revision_policy_id": request.revision_policy.policy_id,
            "revision_policy_version": request.revision_policy.policy_version,
            "canonicalization_policy_id": (
                request.canonicalization_policy.policy_id
            ),
            "canonicalization_policy_version": (
                request.canonicalization_policy.policy_version
            ),
            "lifecycle_stage": request.lifecycle_stage.value,
            "change_reason": request.change_reason,
            "execution_authority": request.execution_authority,
        }
        for column, expected in expected_columns.items():
            actual = row[column]
            if actual != expected:
                raise AuthorityPersistenceError(
                    f"source version column {column} differs from canonical bytes"
                )
        if self._canonical_list(
            row["extraction_scope_bytes"], identity="extraction scope"
        ) != list(request.extraction_scope):
            raise AuthorityPersistenceError(
                "source extraction scope differs from canonical source version"
            )

    @staticmethod
    def _canonical_child(row: sqlite3.Row, *, identity: str) -> dict[str, Any]:
        return canonical_object(
            bytes(row["canonical_bytes"]),
            str(row["canonical_digest"]),
            identity=identity,
        )

    @staticmethod
    def _canonical_list(value: object, *, identity: str) -> list[Any]:
        if not isinstance(value, bytes):
            raise AuthorityPersistenceError(f"{identity} is not retained bytes")
        try:
            decoded = json.loads(value.decode("utf-8", errors="strict"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise AuthorityPersistenceError(
                f"{identity} is invalid canonical JSON"
            ) from exc
        if not isinstance(decoded, list) or canonical_json_bytes(decoded) != value:
            raise AuthorityPersistenceError(
                f"{identity} is not an exact canonical list"
            )
        return decoded


__all__ = ["_SourceRegistryIntegrityMixin"]
