from __future__ import annotations

import sqlite3

from newsroom.authority.canonical import canonical_json_bytes, digest_bytes
from newsroom.authority.persistence import AuthoritySchemaError
from newsroom.graphiti_adapter.types import GraphitiWorkspaceState


class _GraphitiAdapterIntegrityMixin:
    def _validate_schema_and_integrity(self) -> None:
        super()._validate_schema_and_integrity()
        if not self._should_validate_row_integrity():
            return
        self._validate_graphiti_adapter_integrity(self._connection)

    def _validate_graphiti_adapter_integrity(
        self, conn: sqlite3.Connection
    ) -> None:
        try:
            policies = tuple(
                self._graphiti_workspace_policy_from_row(row)
                for row in conn.execute(
                    "SELECT * FROM graphiti_workspace_policies ORDER BY policy_id"
                ).fetchall()
            )
            if len(policies) != 3:
                raise AuthoritySchemaError(
                    "Graphiti workspace policy seed is incomplete"
                )
            for row in conn.execute(
                "SELECT * FROM graphiti_adapter_configurations "
                "ORDER BY configuration_id"
            ).fetchall():
                self._graphiti_configuration_from_row(conn, row, replayed=False)
            for row in conn.execute(
                "SELECT * FROM graphiti_workspaces ORDER BY workspace_id"
            ).fetchall():
                workspace = self._graphiti_workspace_from_row(row)
                self._validate_graphiti_workspace_lifecycle(conn, workspace)
                self._require_graphiti_workspace_absent(workspace)
            for row in conn.execute(
                "SELECT * FROM graphiti_input_manifests ORDER BY manifest_id"
            ).fetchall():
                self._graphiti_manifest_from_row(conn, row)
            for row in conn.execute(
                "SELECT * FROM graphiti_cleanup_receipts ORDER BY receipt_id"
            ).fetchall():
                self._graphiti_cleanup_from_row(row)
            for row in conn.execute(
                "SELECT * FROM graphiti_adapter_attempts "
                "ORDER BY run_id,attempt_number"
            ).fetchall():
                attempt = self._graphiti_attempt_from_row(
                    conn, row, replayed=False
                )
                self._validate_graphiti_attempt_lineage(conn, attempt)
            for row in conn.execute(
                "SELECT * FROM graphiti_replay_sources ORDER BY replay_source_id"
            ).fetchall():
                source = self._graphiti_replay_source_from_row(
                    conn, row, replayed=False
                )
                attempt = self._graphiti_attempt_from_row(
                    conn,
                    self._graphiti_attempt_row(
                        conn, source.source.source_attempt_id
                    ),
                    replayed=False,
                )
                self._validate_graphiti_attempt_lineage(conn, attempt)
            self._validate_graphiti_attempt_heads(conn)
            self._validate_graphiti_replay_bindings(conn)
            self._validate_graphiti_event_coverage(conn)
        except AuthoritySchemaError:
            raise
        except Exception as exc:
            raise AuthoritySchemaError(
                "Graphiti adapter authority integrity validation failed"
            ) from exc

    def _validate_graphiti_workspace_lifecycle(
        self, conn: sqlite3.Connection, workspace
    ) -> None:
        rows = conn.execute(
            "SELECT * FROM graphiti_workspace_lifecycle_events "
            "WHERE workspace_id=? ORDER BY lifecycle_ordinal",
            (str(workspace.workspace_id),),
        ).fetchall()
        if len(rows) != 3:
            raise AuthoritySchemaError(
                "Graphiti workspace lifecycle is incomplete"
            )
        expected_states = (
            GraphitiWorkspaceState.CREATED,
            GraphitiWorkspaceState.ACTIVE,
            GraphitiWorkspaceState.CLEANED,
        )
        for ordinal, row in enumerate(rows, start=1):
            state = GraphitiWorkspaceState(str(row["state"]))
            if ordinal < 3:
                expected = expected_states[ordinal - 1]
                if state is not expected or row["reason"] is not None:
                    raise AuthoritySchemaError(
                        "Graphiti workspace lifecycle prefix differs"
                    )
            elif state not in {
                GraphitiWorkspaceState.CLEANED,
                GraphitiWorkspaceState.LOST,
            }:
                raise AuthoritySchemaError(
                    "Graphiti workspace lacks terminal cleanup or loss"
                )
            value = self._workspace_lifecycle_value(
                workspace_id=str(workspace.workspace_id),
                ordinal=ordinal,
                state=state,
                reason=None if row["reason"] is None else str(row["reason"]),
                recorded_at=str(row["recorded_at"]),
            )
            data = canonical_json_bytes(value)
            if (
                int(row["lifecycle_ordinal"]) != ordinal
                or bytes(row["canonical_bytes"]) != data
                or str(row["canonical_digest"]) != digest_bytes(data)
            ):
                raise AuthoritySchemaError(
                    "Graphiti workspace lifecycle canonical authority differs"
                )
        cleanup = conn.execute(
            "SELECT * FROM graphiti_cleanup_receipts WHERE workspace_id=?",
            (str(workspace.workspace_id),),
        ).fetchone()
        if cleanup is None:
            raise AuthoritySchemaError(
                "Graphiti workspace lacks cleanup receipt"
            )
        if (
            str(cleanup["final_state"]) != str(rows[2]["state"])
            or str(cleanup["reason"]) != str(rows[2]["reason"])
            or str(cleanup["recorded_at"]) != str(rows[2]["recorded_at"])
        ):
            raise AuthoritySchemaError(
                "Graphiti cleanup receipt differs from lifecycle"
            )

    @staticmethod
    def _validate_graphiti_attempt_heads(conn: sqlite3.Connection) -> None:
        missing = conn.execute(
            "SELECT a.run_id FROM graphiti_adapter_attempts a "
            "LEFT JOIN graphiti_adapter_attempt_heads h ON h.run_id=a.run_id "
            "WHERE h.run_id IS NULL LIMIT 1"
        ).fetchone()
        if missing is not None:
            raise AuthoritySchemaError("Graphiti attempt head is missing")
        bad = conn.execute(
            "SELECT h.run_id FROM graphiti_adapter_attempt_heads h "
            "LEFT JOIN graphiti_adapter_attempts a "
            "ON a.run_id=h.run_id "
            "AND a.attempt_number=h.current_attempt_number "
            "AND a.attempt_id=h.current_attempt_id "
            "WHERE a.attempt_id IS NULL "
            "OR h.current_attempt_number!=(SELECT MAX(a2.attempt_number) "
            "FROM graphiti_adapter_attempts a2 WHERE a2.run_id=h.run_id) "
            "OR h.terminal!=CASE WHEN a.outcome IN("
            "'COMPLETE','MALFORMED_OUTPUT','PROVIDER_REJECTED',"
            "'POLICY_BLOCKED','AMBIGUOUS_EFFECT') THEN 1 ELSE 0 END "
            "LIMIT 1"
        ).fetchone()
        if bad is not None:
            raise AuthoritySchemaError("Graphiti attempt head is inconsistent")

    @staticmethod
    def _validate_graphiti_replay_bindings(conn: sqlite3.Connection) -> None:
        for row in conn.execute(
            "SELECT * FROM graphiti_adapter_attempt_replays ORDER BY attempt_id"
        ).fetchall():
            value = {
                "attempt_id": str(row["attempt_id"]),
                "replay_source_id": str(row["replay_source_id"]),
            }
            data = canonical_json_bytes(value)
            if (
                bytes(row["canonical_bytes"]) != data
                or str(row["canonical_digest"]) != digest_bytes(data)
            ):
                raise AuthoritySchemaError(
                    "Graphiti replay binding canonical authority differs"
                )
        missing = conn.execute(
            "SELECT a.attempt_id FROM graphiti_adapter_attempts a "
            "JOIN graphiti_adapter_configurations c "
            "ON c.configuration_id=a.configuration_id "
            "LEFT JOIN graphiti_adapter_attempt_replays r "
            "ON r.attempt_id=a.attempt_id "
            "WHERE (c.runtime_mode='APPROVED_REPLAY' AND r.attempt_id IS NULL) "
            "OR (c.runtime_mode!='APPROVED_REPLAY' AND r.attempt_id IS NOT NULL) "
            "LIMIT 1"
        ).fetchone()
        if missing is not None:
            raise AuthoritySchemaError(
                "Graphiti replay binding differs from adapter mode"
            )

    @staticmethod
    def _validate_graphiti_event_coverage(conn: sqlite3.Connection) -> None:
        missing = conn.execute(
            "SELECT e.event_id FROM ledger_events e "
            "LEFT JOIN graphiti_adapter_configurations c "
            "ON c.authority_event_id=e.event_id "
            "LEFT JOIN graphiti_adapter_attempts a "
            "ON a.authority_event_id=e.event_id "
            "LEFT JOIN graphiti_replay_sources r "
            "ON r.approval_event_id=e.event_id "
            "WHERE e.aggregate_type IN("
            "'graphiti_adapter_configuration','graphiti_adapter_attempt',"
            "'graphiti_replay_source') "
            "AND c.configuration_id IS NULL AND a.attempt_id IS NULL "
            "AND r.replay_source_id IS NULL LIMIT 1"
        ).fetchone()
        if missing is not None:
            raise AuthoritySchemaError(
                "Graphiti adapter event lacks a typed authority record"
            )


__all__ = ["_GraphitiAdapterIntegrityMixin"]
