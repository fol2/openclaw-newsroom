from __future__ import annotations

import sqlite3

from newsroom.authority._check_store_support import (
    _CHECK_RECORD_SPECS,
    _outcome_observed_item_error,
)
from newsroom.authority.persistence import AuthorityPersistenceError
from newsroom.checks.types import FindingScopeKind
from newsroom.sources import SourceDefinitionId, SourceDefinitionVersionId


_TABLE_BY_COMMAND = {
    "check.request.register": "check_requests",
    "check.attempt.start": "check_attempts",
    "check.outcome.record": "check_outcomes",
    "check.baseline.decide": "baseline_decisions",
    "source.observable_transition.record": "observable_transitions",
    "operational.finding.open": "operational_findings",
    "operational.finding.occurrence.record": (
        "operational_finding_occurrences"
    ),
}


class _CheckIntegrityMixin:
    def _validate_schema_and_integrity(self) -> None:
        super()._validate_schema_and_integrity()
        if not self._should_validate_row_integrity():
            return
        conn = self._connection
        self._validate_check_records(conn)
        self._validate_attempt_chains(conn)
        self._validate_baseline_heads(conn)
        self._validate_occurrence_links(conn)
        self._validate_check_event_coverage(conn)

    def _validate_check_records(self, conn: sqlite3.Connection) -> None:
        for row in conn.execute("SELECT * FROM check_requests").fetchall():
            record = self._check_request_from_row(conn, row, replayed=False)
            request = record.request
            version = self._version_row(conn, request.definition_version_id)
            if not self._source_version_was_current_at_event(
                conn,
                definition_id=str(request.definition_id),
                version_id=str(request.definition_version_id),
                record_event_id=str(row["authority_event_id"]),
            ):
                raise AuthorityPersistenceError(
                    "Check Request source version was not current when recorded"
                )
            if (
                str(version["definition_id"]) != str(request.definition_id)
                or str(version["rights_decision_id"])
                != request.rights_decision_id
                or str(version["rights_policy_version"])
                != request.rights_policy_version
                or str(version["baseline_policy_id"])
                != request.baseline_policy.policy_id
                or str(version["baseline_policy_version"])
                != request.baseline_policy.policy_version
                or str(version["revision_policy_id"])
                != request.revision_policy.policy_id
                or str(version["revision_policy_version"])
                != request.revision_policy.policy_version
            ):
                raise AuthorityPersistenceError(
                    "Check Request source contract is inconsistent"
                )
            coverage = conn.execute(
                "SELECT 1 FROM source_version_coverage_mappings "
                "WHERE version_id=? AND obligation_id=? "
                "AND responsibility=? AND contribution=?",
                (
                    str(request.definition_version_id),
                    request.coverage.obligation_id,
                    request.coverage.responsibility.value,
                    request.coverage.contribution.value,
                ),
            ).fetchone()
            if coverage is None:
                raise AuthorityPersistenceError(
                    "Check Request coverage contract is inconsistent"
                )

        for row in conn.execute("SELECT * FROM check_attempts").fetchall():
            record = self._check_attempt_from_row(conn, row, replayed=False)
            parent = self._check_request_row(
                conn, str(record.request.request_id)
            )
            if (
                str(parent["adapter_request_digest"])
                != record.request.adapter_request_digest
            ):
                raise AuthorityPersistenceError(
                    "Check Attempt adapter contract is inconsistent"
                )

        for row in conn.execute("SELECT * FROM check_outcomes").fetchall():
            record = self._check_outcome_from_row(conn, row, replayed=False)
            request = record.request
            attempt = self._check_attempt_row(conn, str(request.attempt_id))
            parent = self._check_request_row(conn, str(request.request_id))
            if (
                str(attempt["request_id"]) != str(request.request_id)
                or str(parent["definition_id"]) != str(request.definition_id)
                or str(parent["definition_version_id"])
                != str(request.definition_version_id)
                or (
                    request.producer_slot_digest is not None
                    and request.producer_slot_digest
                    != str(parent["producer_slot_digest"])
                )
                or request.completed_at.to_text() < str(attempt["started_at"])
            ):
                raise AuthorityPersistenceError(
                    "Check Outcome lineage or chronology is inconsistent"
                )

        for row in conn.execute("SELECT * FROM baseline_decisions").fetchall():
            record = self._baseline_decision_from_row(
                conn, row, replayed=False
            )
            request = record.request
            version = self._version_row(conn, request.definition_version_id)
            parent = self._check_request_row(
                conn, str(request.check_request_id)
            )
            outcome = self._check_outcome_row(
                conn, str(request.check_outcome_id)
            )
            if not self._source_version_was_current_at_event(
                conn,
                definition_id=str(request.definition_id),
                version_id=str(request.definition_version_id),
                record_event_id=str(row["authority_event_id"]),
            ):
                raise AuthorityPersistenceError(
                    "Baseline Decision source version was not current when recorded"
                )
            if (
                str(version["definition_id"]) != str(request.definition_id)
                or str(version["observation_model"])
                != request.observation_model.value
                or str(version["baseline_policy_id"])
                != request.baseline_policy.policy_id
                or str(version["baseline_policy_version"])
                != request.baseline_policy.policy_version
                or str(parent["definition_id"]) != str(request.definition_id)
                or str(parent["definition_version_id"])
                != str(request.definition_version_id)
                or str(outcome["request_id"])
                != str(request.check_request_id)
            ):
                raise AuthorityPersistenceError(
                    "Baseline Decision source lineage is inconsistent"
                )

        for row in conn.execute(
            "SELECT * FROM observable_transitions"
        ).fetchall():
            record = self._observable_transition_from_row(
                conn, row, replayed=False
            )
            request = record.request
            version = self._version_row(conn, request.definition_version_id)
            outcome = self._check_outcome_row(
                conn, str(request.check_outcome_id)
            )
            item = self._item_row(conn, str(request.item_id))
            parent = self._check_request_row(conn, str(outcome["request_id"]))
            if not self._source_version_was_current_at_event(
                conn,
                definition_id=str(request.definition_id),
                version_id=str(request.definition_version_id),
                record_event_id=str(row["authority_event_id"]),
            ):
                raise AuthorityPersistenceError(
                    "Observable Transition source version was not current when recorded"
                )
            if (
                str(version["definition_id"]) != str(request.definition_id)
                or str(version["observation_model"])
                != request.observation_model.value
                or str(outcome["definition_id"])
                != str(request.definition_id)
                or str(outcome["definition_version_id"])
                != str(request.definition_version_id)
                or str(item["definition_id"]) != str(request.definition_id)
                or str(parent["transition_policy_id"])
                != request.transition_policy.policy_id
                or str(parent["transition_policy_version"])
                != request.transition_policy.policy_version
            ):
                raise AuthorityPersistenceError(
                    "Observable Transition source lineage is inconsistent"
                )

        for row in conn.execute("SELECT * FROM operational_findings").fetchall():
            record = self._operational_finding_from_row(
                conn, row, replayed=False
            )
            request = record.request
            error = self._finding_lineage_error(
                conn,
                scope_kind=request.scope_kind,
                scope_id=request.scope_id,
                request_id=request.opened_by_request_id,
                attempt_id=request.opened_by_attempt_id,
                outcome_id=request.opened_by_outcome_id,
                observed_at=request.opened_at,
            )
            if error is not None:
                raise AuthorityPersistenceError(error)

        for row in conn.execute(
            "SELECT * FROM operational_finding_occurrences"
        ).fetchall():
            record = self._finding_occurrence_from_row(
                conn, row, replayed=False
            )
            request = record.request
            finding = self._operational_finding_row(
                conn, str(request.finding_id)
            )
            error = self._finding_lineage_error(
                conn,
                scope_kind=FindingScopeKind(str(finding["scope_kind"])),
                scope_id=str(finding["scope_id"]),
                request_id=request.request_id,
                attempt_id=request.attempt_id,
                outcome_id=request.outcome_id,
                observed_at=request.observed_at,
            )
            if error is not None:
                raise AuthorityPersistenceError(error)

    @staticmethod
    def _validate_attempt_chains(conn: sqlite3.Connection) -> None:
        rows = conn.execute(
            "SELECT a.request_id,a.attempt_id,a.attempt_number,"
            "a.prior_attempt_id,a.started_at,r.requested_at "
            "FROM check_attempts a "
            "JOIN check_requests r ON r.request_id=a.request_id "
            "ORDER BY a.request_id,a.attempt_number"
        ).fetchall()
        prior_by_request: dict[str, tuple[int, str]] = {}
        for row in rows:
            request_id = str(row["request_id"])
            attempt_id = str(row["attempt_id"])
            number = int(row["attempt_number"])
            previous = prior_by_request.get(request_id)
            expected_number = 1 if previous is None else previous[0] + 1
            expected_id = None if previous is None else previous[1]
            actual_id = (
                None
                if row["prior_attempt_id"] is None
                else str(row["prior_attempt_id"])
            )
            if (
                number != expected_number
                or actual_id != expected_id
                or str(row["started_at"]) < str(row["requested_at"])
            ):
                raise AuthorityPersistenceError(
                    "Check Attempt chain or request chronology is inconsistent"
                )
            if previous is not None:
                prior_outcome = conn.execute(
                    "SELECT completed_at FROM check_outcomes WHERE attempt_id=?",
                    (previous[1],),
                ).fetchone()
                if (
                    prior_outcome is None
                    or str(row["started_at"])
                    < str(prior_outcome["completed_at"])
                ):
                    raise AuthorityPersistenceError(
                        "Check Attempt predecessor Outcome is incomplete or later"
                    )
            prior_by_request[request_id] = (number, attempt_id)

    @staticmethod
    def _validate_baseline_heads(conn: sqlite3.Connection) -> None:
        definitions = conn.execute(
            "SELECT DISTINCT definition_id FROM baseline_decisions"
        ).fetchall()
        for definition in definitions:
            definition_id = str(definition["definition_id"])
            rows = conn.execute(
                "SELECT d.decision_id,d.kind,d.previous_decision_id,"
                "d.decided_at,e.ledger_seq "
                "FROM baseline_decisions d "
                "JOIN ledger_events e ON e.event_id=d.authority_event_id "
                "WHERE d.definition_id=? ORDER BY e.ledger_seq",
                (definition_id,),
            ).fetchall()
            prior = None
            for index, row in enumerate(rows):
                actual = (
                    None
                    if row["previous_decision_id"] is None
                    else str(row["previous_decision_id"])
                )
                if index == 0:
                    if str(row["kind"]) != "ESTABLISH" or actual is not None:
                        raise AuthorityPersistenceError(
                            "baseline lineage does not begin with establishment"
                        )
                elif actual != prior or str(row["kind"]) == "ESTABLISH":
                    raise AuthorityPersistenceError(
                        "baseline lineage does not extend exact predecessor"
                    )
                prior = str(row["decision_id"])
            head = conn.execute(
                "SELECT current_decision_id,updated_at "
                "FROM baseline_decision_heads WHERE definition_id=?",
                (definition_id,),
            ).fetchone()
            if (
                head is None
                or str(head["current_decision_id"]) != prior
                or str(head["updated_at"]) != str(rows[-1]["decided_at"])
            ):
                raise AuthorityPersistenceError(
                    "baseline head differs from retained decision history"
                )
        orphan = conn.execute(
            "SELECT 1 FROM baseline_decision_heads h "
            "LEFT JOIN baseline_decisions d "
            "ON d.decision_id=h.current_decision_id "
            "WHERE d.decision_id IS NULL LIMIT 1"
        ).fetchone()
        if orphan is not None:
            raise AuthorityPersistenceError("baseline head is orphaned")

    def _validate_occurrence_links(self, conn: sqlite3.Connection) -> None:
        missing = conn.execute(
            "SELECT o.occurrence_id FROM discovery_occurrences o "
            "LEFT JOIN check_outcomes c ON c.outcome_id=o.check_outcome_id "
            "LEFT JOIN discovery_occurrence_check_links l "
            "ON l.occurrence_id=o.occurrence_id "
            "WHERE c.outcome_id IS NULL OR l.occurrence_id IS NULL LIMIT 1"
        ).fetchone()
        if missing is not None:
            raise AuthorityPersistenceError(
                "post-v11 discovery occurrence lacks exact Check link"
            )
        mismatch = conn.execute(
            "SELECT 1 FROM discovery_occurrence_check_links l "
            "JOIN discovery_occurrences o ON o.occurrence_id=l.occurrence_id "
            "JOIN check_outcomes c ON c.outcome_id=l.check_outcome_id "
            "WHERE o.check_outcome_id!=l.check_outcome_id "
            "OR o.definition_id!=c.definition_id "
            "OR o.definition_version_id!=c.definition_version_id "
            "OR o.observed_at!=c.completed_at "
            "OR o.receipt_digest IS NOT c.receipt_digest "
            "OR c.kind NOT IN("
            "'SUCCESS_UNCHANGED','SUCCESS_CHANGED',"
            "'SUCCESS_PARTIAL','SUCCESS_TRUNCATED') "
            "LIMIT 1"
        ).fetchone()
        if mismatch is not None:
            raise AuthorityPersistenceError(
                "discovery occurrence Check link is inconsistent"
            )
        for occurrence in conn.execute(
            "SELECT * FROM discovery_occurrences"
        ).fetchall():
            if occurrence["representation_id"] is None:
                raise AuthorityPersistenceError(
                    "post-v11 discovery occurrence lacks exact Representation"
                )
            outcome_row = self._check_outcome_row(
                conn, str(occurrence["check_outcome_id"])
            )
            outcome = self._check_outcome_from_row(
                conn, outcome_row, replayed=False
            )
            revision = self._revision_row(
                conn, str(occurrence["revision_id"])
            )
            representation = self._representation_row(
                conn, str(occurrence["representation_id"])
            )
            error = _outcome_observed_item_error(
                outcome.request,
                revision_item_id=str(revision["item_id"]),
                representation_digest=str(
                    representation["representation_digest"]
                ),
            )
            if error is not None:
                raise AuthorityPersistenceError(error)

    @staticmethod
    def _validate_check_event_coverage(conn: sqlite3.Connection) -> None:
        for command_type, (_, event_type, _) in _CHECK_RECORD_SPECS.items():
            table = _TABLE_BY_COMMAND[command_type]
            events = {
                str(row[0])
                for row in conn.execute(
                    "SELECT event_id FROM ledger_events WHERE event_type=?",
                    (event_type,),
                ).fetchall()
            }
            rows = {
                str(row[0])
                for row in conn.execute(
                    f"SELECT authority_event_id FROM {table}"
                ).fetchall()
            }
            if events != rows:
                raise AuthorityPersistenceError(
                    f"{command_type} ledger coverage differs from typed records"
                )


__all__ = ["_CheckIntegrityMixin"]
