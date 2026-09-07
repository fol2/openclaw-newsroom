from __future__ import annotations

import sqlite3
from collections.abc import Iterable

from newsroom.authority.canonical import canonical_json_bytes, digest_bytes
from newsroom.authority.persistence import AuthoritySchemaError
from newsroom.relations.editorial_models import (
    EDITORIAL_RELATION_ADMISSION_POLICY_VERSION,
)
from newsroom.relations.editorial_types import (
    EditorialRelationAssertionLifecycle,
    EditorialRelationCurrentState,
    EditorialRelationDecisionAction,
    EditorialRelationProjectionAction,
)


_STATE_BY_ACTION = {
    EditorialRelationDecisionAction.ACCEPT.value: EditorialRelationCurrentState.ADMITTED.value,
    EditorialRelationDecisionAction.REJECT.value: EditorialRelationCurrentState.REJECTED.value,
    EditorialRelationDecisionAction.HOLD.value: EditorialRelationCurrentState.HELD.value,
    EditorialRelationDecisionAction.UNRESOLVED.value: EditorialRelationCurrentState.UNRESOLVED.value,
    EditorialRelationDecisionAction.INVALIDATE.value: EditorialRelationCurrentState.INVALIDATED.value,
    EditorialRelationDecisionAction.REVOKE.value: EditorialRelationCurrentState.REVOKED.value,
    EditorialRelationDecisionAction.SUPERSEDE.value: EditorialRelationCurrentState.SUPERSEDED.value,
}


class _EditorialRelationIntegrityMixin:
    def _validate_schema_and_integrity(self) -> None:
        super()._validate_schema_and_integrity()
        if not self._should_validate_row_integrity():
            return
        self._validate_editorial_relation_integrity(self._connection)

    def _validate_editorial_relation_integrity(
        self, conn: sqlite3.Connection
    ) -> None:
        try:
            for row in conn.execute(
                "SELECT * FROM editorial_relation_endpoints ORDER BY endpoint_digest"
            ).fetchall():
                self._editorial_endpoint_from_row(row)
            for row in conn.execute(
                "SELECT * FROM editorial_relation_proposals ORDER BY proposal_id"
            ).fetchall():
                self._editorial_proposal_from_row(conn, row)
            for row in conn.execute(
                "SELECT * FROM editorial_relation_proposal_versions "
                "ORDER BY proposal_id,version_number"
            ).fetchall():
                self._editorial_proposal_version_from_row(
                    conn, row, replayed=False
                )
            for row in conn.execute(
                "SELECT * FROM editorial_relation_decisions "
                "ORDER BY proposal_id,decision_version"
            ).fetchall():
                self._editorial_decision_from_row(conn, row, replayed=False)
            for row in conn.execute(
                "SELECT * FROM editorial_relation_assertions ORDER BY assertion_id"
            ).fetchall():
                self._editorial_assertion_from_row(conn, row)
            for row in conn.execute(
                "SELECT * FROM editorial_relation_projection_events "
                "ORDER BY source_ledger_seq,projection_event_id"
            ).fetchall():
                self._editorial_projection_event_from_row(conn, row)
            self._validate_editorial_relation_heads(conn)
            if not self._allow_editorial_relation_projection_rebuild:
                missing_head = conn.execute(
                    "SELECT a.assertion_id FROM editorial_relation_assertions a "
                    "LEFT JOIN editorial_relation_assertion_heads h "
                    "ON h.assertion_id=a.assertion_id "
                    "WHERE h.assertion_id IS NULL LIMIT 1"
                ).fetchone()
                if missing_head is not None:
                    raise AuthoritySchemaError(
                        "editorial relation current projection is incomplete"
                    )
            self._validate_editorial_supersessions(conn)
            self._validate_editorial_projection_coverage(conn)
            self._validate_editorial_assertion_endpoint_graph(conn)
        except AuthoritySchemaError:
            raise
        except Exception as exc:
            raise AuthoritySchemaError(
                "editorial relation authority integrity validation failed"
            ) from exc

    @staticmethod
    def _validate_editorial_relation_heads(conn: sqlite3.Connection) -> None:
        bad = conn.execute(
            "SELECT h.proposal_id FROM editorial_relation_proposal_heads h "
            "LEFT JOIN editorial_relation_proposal_versions v "
            "ON v.proposal_id=h.proposal_id "
            "AND v.proposal_version_id=h.current_proposal_version_id "
            "AND v.version_number=h.current_version_number "
            "WHERE v.proposal_version_id IS NULL "
            "OR h.current_version_number!=("
            "SELECT MAX(v2.version_number) FROM editorial_relation_proposal_versions v2 "
            "WHERE v2.proposal_id=h.proposal_id) LIMIT 1"
        ).fetchone()
        if bad is not None:
            raise AuthoritySchemaError(
                "editorial relation proposal head is inconsistent"
            )

        bad = conn.execute(
            "SELECT h.proposal_id FROM editorial_relation_decision_heads h "
            "LEFT JOIN editorial_relation_decisions d "
            "ON d.proposal_id=h.proposal_id "
            "AND d.decision_id=h.current_decision_id "
            "AND d.decision_version=h.current_decision_version "
            "WHERE d.decision_id IS NULL "
            "OR h.current_decision_version!=("
            "SELECT MAX(d2.decision_version) FROM editorial_relation_decisions d2 "
            "WHERE d2.proposal_id=h.proposal_id) "
            "OR h.current_state!=CASE d.action "
            "WHEN 'ACCEPT' THEN 'ADMITTED' "
            "WHEN 'REJECT' THEN 'REJECTED' "
            "WHEN 'HOLD' THEN 'HELD' "
            "WHEN 'UNRESOLVED' THEN 'UNRESOLVED' "
            "WHEN 'INVALIDATE' THEN 'INVALIDATED' "
            "WHEN 'REVOKE' THEN 'REVOKED' "
            "ELSE 'SUPERSEDED' END LIMIT 1"
        ).fetchone()
        if bad is not None:
            raise AuthoritySchemaError(
                "editorial relation decision head is inconsistent"
            )

        bad = conn.execute(
            "SELECT h.assertion_id FROM editorial_relation_assertion_heads h "
            "LEFT JOIN editorial_relation_assertions a "
            "ON a.assertion_id=h.assertion_id "
            "LEFT JOIN editorial_relation_decisions d "
            "ON d.decision_id=h.current_decision_id "
            "WHERE a.assertion_id IS NULL OR d.decision_id IS NULL "
            "OR d.proposal_id!=a.proposal_id "
            "OR d.decision_version!=h.current_decision_version "
            "OR h.lifecycle!=CASE d.action "
            "WHEN 'ACCEPT' THEN 'ACTIVE' "
            "WHEN 'INVALIDATE' THEN 'INVALIDATED' "
            "WHEN 'REVOKE' THEN 'REVOKED' "
            "WHEN 'SUPERSEDE' THEN 'SUPERSEDED' ELSE h.lifecycle END "
            "OR (d.action='ACCEPT' AND d.assertion_id!=h.assertion_id) "
            "OR (d.action IN('INVALIDATE','REVOKE','SUPERSEDE') "
            "AND d.target_assertion_id!=h.assertion_id) LIMIT 1"
        ).fetchone()
        if bad is not None:
            raise AuthoritySchemaError(
                "editorial relation assertion head is inconsistent"
            )

        wrong_policy = conn.execute(
            "SELECT decision_id FROM editorial_relation_decisions "
            "WHERE decision_policy_version!=? LIMIT 1",
            (EDITORIAL_RELATION_ADMISSION_POLICY_VERSION,),
        ).fetchone()
        if wrong_policy is not None:
            raise AuthoritySchemaError(
                "editorial relation decision uses an unapproved policy version"
            )

    @staticmethod
    def _validate_editorial_supersessions(conn: sqlite3.Connection) -> None:
        for row in conn.execute(
            "SELECT * FROM editorial_relation_supersessions "
            "ORDER BY supersession_id"
        ).fetchall():
            value = {
                "supersession_id": str(row["supersession_id"]),
                "decision_id": str(row["decision_id"]),
                "predecessor_assertion_id": str(row["predecessor_assertion_id"]),
                "successor_assertion_id": str(row["successor_assertion_id"]),
                "recorded_at": str(row["recorded_at"]),
            }
            data = canonical_json_bytes(value)
            if (
                bytes(row["canonical_bytes"]) != data
                or str(row["canonical_digest"]) != digest_bytes(data)
            ):
                raise AuthoritySchemaError(
                    "editorial relation supersession canonical bytes differ"
                )
        mismatch = conn.execute(
            "SELECT s.supersession_id FROM editorial_relation_supersessions s "
            "LEFT JOIN editorial_relation_decisions d ON d.decision_id=s.decision_id "
            "WHERE d.decision_id IS NULL OR d.action!='SUPERSEDE' "
            "OR d.supersession_id!=s.supersession_id "
            "OR d.target_assertion_id!=s.predecessor_assertion_id "
            "OR d.successor_assertion_id!=s.successor_assertion_id LIMIT 1"
        ).fetchone()
        if mismatch is not None:
            raise AuthoritySchemaError(
                "editorial relation supersession differs from its decision"
            )

    @staticmethod
    def _validate_editorial_projection_coverage(
        conn: sqlite3.Connection,
    ) -> None:
        missing = conn.execute(
            "SELECT h.assertion_id FROM editorial_relation_assertion_heads h "
            "LEFT JOIN editorial_relation_projection_events e "
            "ON e.projection_event_id=("
            "SELECT e2.projection_event_id FROM editorial_relation_projection_events e2 "
            "WHERE e2.assertion_id=h.assertion_id "
            "ORDER BY e2.source_ledger_seq DESC,e2.projection_event_id DESC LIMIT 1) "
            "LEFT JOIN editorial_relation_decisions d "
            "ON d.decision_id=h.current_decision_id "
            "WHERE e.projection_event_id IS NULL "
            "OR e.source_event_id!=d.authority_event_id "
            "OR e.source_ledger_seq!=d.authority_ledger_seq "
            "OR e.lifecycle!=h.lifecycle "
            "OR e.action!=CASE WHEN h.lifecycle='ACTIVE' THEN 'UPSERT' ELSE 'REMOVE' END "
            "LIMIT 1"
        ).fetchone()
        if missing is not None:
            raise AuthoritySchemaError(
                "editorial relation assertion lacks exact projection coverage"
            )

        orphan = conn.execute(
            "SELECT e.projection_event_id FROM editorial_relation_projection_events e "
            "LEFT JOIN editorial_relation_assertions a "
            "ON a.assertion_id=e.assertion_id "
            "LEFT JOIN ledger_events l ON l.event_id=e.source_event_id "
            "WHERE a.assertion_id IS NULL OR l.event_id IS NULL "
            "OR l.ledger_seq!=e.source_ledger_seq LIMIT 1"
        ).fetchone()
        if orphan is not None:
            raise AuthoritySchemaError(
                "editorial relation projection event has invalid authority lineage"
            )

    @staticmethod
    def _validate_editorial_assertion_endpoint_graph(
        conn: sqlite3.Connection,
    ) -> None:
        edges: dict[str, tuple[str, ...]] = {}
        for row in conn.execute(
            "SELECT a.assertion_id,s.assertion_id AS subject_assertion_id,"
            "o.assertion_id AS object_assertion_id "
            "FROM editorial_relation_assertions a "
            "JOIN editorial_relation_endpoints s "
            "ON s.endpoint_digest=a.subject_endpoint_digest "
            "JOIN editorial_relation_endpoints o "
            "ON o.endpoint_digest=a.object_endpoint_digest "
            "ORDER BY a.assertion_id"
        ).fetchall():
            targets = tuple(
                item
                for item in (
                    row["subject_assertion_id"],
                    row["object_assertion_id"],
                )
                if item is not None
            )
            edges[str(row["assertion_id"])] = tuple(str(item) for item in targets)

        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(node: str, path: Iterable[str]) -> None:
            if node in visiting:
                raise AuthoritySchemaError(
                    "editorial relation assertion endpoint cycle is retained"
                )
            if node in visited:
                return
            if node not in edges:
                raise AuthoritySchemaError(
                    "editorial relation assertion endpoint is not retained"
                )
            visiting.add(node)
            for target in edges[node]:
                visit(target, (*path, node))
            visiting.remove(node)
            visited.add(node)

        for assertion_id in sorted(edges):
            visit(assertion_id, ())


__all__ = ["_EditorialRelationIntegrityMixin"]
