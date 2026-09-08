from __future__ import annotations

import sqlite3
from typing import Any, Mapping

from newsroom.authority._capability import _AuthorizedCommandGrant
from newsroom.authority.canonical import digest_bytes
from newsroom.authority.persistence import AuthorityPersistenceError
from newsroom.authority.types import PayloadMode, TrustScope
from newsroom.checks.policy import (
    CHECK_ATTEMPT_START_COMMAND,
    CHECK_BASELINE_DECIDE_COMMAND,
    CHECK_OUTCOME_RECORD_COMMAND,
    CHECK_REQUEST_REGISTER_COMMAND,
    OBSERVABLE_TRANSITION_RECORD_COMMAND,
    OPERATIONAL_FINDING_OCCURRENCE_RECORD_COMMAND,
    OPERATIONAL_FINDING_OPEN_COMMAND,
)
from newsroom.checks.admission_models import deterministic_uuid4
from newsroom.checks.baseline_models import BaselineDecisionRequest
from newsroom.checks.check_models import CheckOutcomeRequest
from newsroom.checks.transition_models import ObservableTransitionRequest
from newsroom.sources import SourceItemId

from newsroom.checks.types import (
    BaselineDisposition,
    FindingScopeKind,
    ObservableTransitionKind,
)


_COMPLETE_CONFIRMATION_OUTCOMES = frozenset(
    {"SUCCESS_EMPTY", "SUCCESS_UNCHANGED", "SUCCESS_CHANGED"}
)


_FIRST_OBSERVATION_TRANSITIONS = frozenset(
    {
        ObservableTransitionKind.FIRST_OBSERVED,
        ObservableTransitionKind.ACTIVATED,
        ObservableTransitionKind.AGENDA_CREATED,
    }
)
_ENDING_TRANSITIONS = frozenset(
    {
        ObservableTransitionKind.RESOLVED_OR_CLEARED,
        ObservableTransitionKind.EXPIRED,
        ObservableTransitionKind.CANCELLED,
        ObservableTransitionKind.WITHDRAWN,
    }
)
_AGENDA_OPEN_TRANSITIONS = frozenset(
    {
        ObservableTransitionKind.AGENDA_CREATED,
        ObservableTransitionKind.AGENDA_RESCHEDULED,
    }
)

def _outcome_observed_item_error(
    store: Any,
    conn: sqlite3.Connection,
    outcome: CheckOutcomeRequest,
    *,
    revision_item_id: str,
    representation_digest: str,
) -> str | None:
    """Bind one Occurrence to an exact observed item retained by its Outcome."""

    if not isinstance(outcome, CheckOutcomeRequest):
        return "Discovery Occurrence Check Outcome is not typed"
    matches = []
    for observed in outcome.observed_items:
        expected_item_id = _observed_item_id(
            store,
            conn,
            request_id=str(outcome.request_id),
            definition_id=str(outcome.definition_id),
            item_key=observed.item_key,
        )
        if (
            str(expected_item_id) == revision_item_id
            and observed.item_digest == representation_digest
        ):
            matches.append(observed)
    if len(matches) != 1:
        return (
            "Discovery Occurrence Revision/Representation is not one exact "
            "observed item in its Check Outcome"
        )
    return None


def _observed_item_id(
    store: Any,
    conn: sqlite3.Connection,
    *,
    request_id: str,
    definition_id: str,
    item_key: str,
) -> SourceItemId:
    """Derive the proposal-admission Source Item identity for one item key."""

    request = conn.execute(
        "SELECT trigger_kind FROM check_requests WHERE request_id=?",
        (request_id,),
    ).fetchone()
    if request is None:
        raise AuthorityPersistenceError("Check Outcome Request is not retained")
    if str(request["trigger_kind"]) == "DELIVERED_INPUT":
        rows = conn.execute(
            "SELECT * FROM source_items WHERE definition_id=? AND identity_digest=?",
            (definition_id, item_key),
        ).fetchall()
        if len(rows) != 1:
            raise AuthorityPersistenceError(
                "delivered Check item is not one exact retained Source Item"
            )
        retained = store._source_item_from_row(conn, rows[0], replayed=False)
        if (
            str(retained.request.definition_id) != definition_id
            or retained.request.identity_digest != item_key
        ):
            raise AuthorityPersistenceError(
                "delivered Check item differs from retained Source Item"
            )
        return retained.request.item_id

    return deterministic_uuid4(
        SourceItemId,
        namespace="increment-3c-source-item-v1",
        semantic_value={
            "definition_id": definition_id,
            "item_key": item_key,
        },
    )


_CHECK_RECORD_SPECS: dict[str, tuple[str, str, TrustScope]] = {
    CHECK_REQUEST_REGISTER_COMMAND: (
        "check_request",
        "check.request.registered",
        TrustScope.ADMITTED,
    ),
    CHECK_ATTEMPT_START_COMMAND: (
        "check_attempt",
        "check.attempt.started",
        TrustScope.OBSERVED,
    ),
    CHECK_OUTCOME_RECORD_COMMAND: (
        "check_outcome",
        "check.outcome.recorded",
        TrustScope.OBSERVED,
    ),
    CHECK_BASELINE_DECIDE_COMMAND: (
        "baseline_decision",
        "check.baseline.decided",
        TrustScope.ADMITTED,
    ),
    OBSERVABLE_TRANSITION_RECORD_COMMAND: (
        "observable_transition",
        "source.observable_transition.recorded",
        TrustScope.ADMITTED,
    ),
    OPERATIONAL_FINDING_OPEN_COMMAND: (
        "operational_finding",
        "operational.finding.opened",
        TrustScope.ADMITTED,
    ),
    OPERATIONAL_FINDING_OCCURRENCE_RECORD_COMMAND: (
        "operational_finding_occurrence",
        "operational.finding.occurrence.recorded",
        TrustScope.OBSERVED,
    ),
}


class _CheckStoreSupport:
    def _require_check_grant(
        self,
        grant: _AuthorizedCommandGrant,
        *,
        command_type: str,
        aggregate_id: str,
        canonical_bytes: bytes,
    ) -> None:
        self._issuer.verify(grant)
        spec = _CHECK_RECORD_SPECS.get(command_type)
        if spec is None:
            raise AuthorityPersistenceError("unknown discovery Check command")
        aggregate_type, event_type, trust_scope = spec
        definition = grant.definition
        if (
            grant.command_type != command_type
            or grant.aggregate_id != aggregate_id
            or grant.expected_aggregate_version != 0
            or definition.command_type != command_type
            or definition.aggregate_type != aggregate_type
            or definition.event_type != event_type
            or definition.trust_scope is not trust_scope
            or definition.security_scope != "authority.discovery_checks"
            or definition.retention_scope != "authority.audit"
            or definition.payload_mode is not PayloadMode.INLINE
            or grant.payload.kind != PayloadMode.INLINE.value
            or grant.payload.inline_bytes != canonical_bytes
            or grant.payload.digest != digest_bytes(canonical_bytes)
        ):
            raise AuthorityPersistenceError(
                "discovery Check grant differs from the typed record"
            )


    @classmethod
    def _finding_lineage_error(
        cls,
        conn: sqlite3.Connection,
        *,
        scope_kind: FindingScopeKind,
        scope_id: str,
        request_id,
        attempt_id,
        outcome_id,
        observed_at,
    ) -> str | None:
        request_row = None
        attempt_row = None
        outcome_row = None
        if request_id is not None:
            request_row = conn.execute(
                "SELECT * FROM check_requests WHERE request_id=?",
                (str(request_id),),
            ).fetchone()
            if request_row is None:
                return "Operational Finding references an unretained Check Request"
        if attempt_id is not None:
            attempt_row = conn.execute(
                "SELECT * FROM check_attempts WHERE attempt_id=?",
                (str(attempt_id),),
            ).fetchone()
            if attempt_row is None:
                return "Operational Finding references an unretained Check Attempt"
            if (
                request_row is not None
                and str(attempt_row["request_id"]) != str(request_id)
            ):
                return "Operational Finding Attempt differs from Request"
        if outcome_id is not None:
            outcome_row = conn.execute(
                "SELECT * FROM check_outcomes WHERE outcome_id=?",
                (str(outcome_id),),
            ).fetchone()
            if outcome_row is None:
                return "Operational Finding references an unretained Check Outcome"
            if (
                request_row is not None
                and str(outcome_row["request_id"]) != str(request_id)
            ):
                return "Operational Finding Outcome differs from Request"
            if (
                attempt_row is not None
                and str(outcome_row["attempt_id"]) != str(attempt_id)
            ):
                return "Operational Finding Outcome differs from Attempt"

        resolved_request_id = (
            None if request_row is None else str(request_row["request_id"])
        )
        if resolved_request_id is None and attempt_row is not None:
            resolved_request_id = str(attempt_row["request_id"])
        if resolved_request_id is None and outcome_row is not None:
            resolved_request_id = str(outcome_row["request_id"])
        if resolved_request_id is None:
            return "Operational Finding has no resolvable Check Request"
        if request_row is None:
            request_row = conn.execute(
                "SELECT * FROM check_requests WHERE request_id=?",
                (resolved_request_id,),
            ).fetchone()
            if request_row is None:
                return "Operational Finding Check Request is unretained"

        resolved_attempt_id = (
            None if attempt_row is None else str(attempt_row["attempt_id"])
        )
        if resolved_attempt_id is None and outcome_row is not None:
            resolved_attempt_id = str(outcome_row["attempt_id"])
            attempt_row = conn.execute(
                "SELECT * FROM check_attempts WHERE attempt_id=?",
                (resolved_attempt_id,),
            ).fetchone()
            if attempt_row is None:
                return "Operational Finding Check Attempt is unretained"

        observed_text = observed_at.to_text()
        if outcome_row is not None:
            if str(outcome_row["completed_at"]) != observed_text:
                return (
                    "Operational Finding observation time differs from exact "
                    "Check Outcome"
                )
        elif attempt_row is not None:
            if observed_text < str(attempt_row["started_at"]):
                return "Operational Finding precedes its Check Attempt"
        elif observed_text < str(request_row["requested_at"]):
            return "Operational Finding precedes its Check Request"

        definition_id = str(request_row["definition_id"])
        definition_version_id = str(request_row["definition_version_id"])
        if outcome_row is not None and (
            str(outcome_row["definition_id"]) != definition_id
            or str(outcome_row["definition_version_id"])
            != definition_version_id
        ):
            return "Operational Finding source lineage is inconsistent"

        if scope_kind is FindingScopeKind.CHECK_REQUEST:
            valid = scope_id == resolved_request_id
        elif scope_kind is FindingScopeKind.CHECK_ATTEMPT:
            valid = (
                resolved_attempt_id is not None
                and scope_id == resolved_attempt_id
            )
        elif scope_kind is FindingScopeKind.CHECK_OUTCOME:
            valid = outcome_row is not None and scope_id == str(
                outcome_row["outcome_id"]
            )
        elif scope_kind is FindingScopeKind.SOURCE_DEFINITION:
            valid = scope_id == definition_id
        elif scope_kind is FindingScopeKind.SOURCE_VERSION:
            valid = scope_id == definition_version_id
        elif scope_kind is FindingScopeKind.ADAPTER:
            valid = (
                attempt_row is not None
                and scope_id == str(attempt_row["adapter_request_id"])
            )
        else:
            item = conn.execute(
                "SELECT definition_id FROM source_items WHERE item_id=?",
                (scope_id,),
            ).fetchone()
            valid = item is not None and str(item["definition_id"]) == definition_id
            if valid and outcome_row is not None:
                valid = conn.execute(
                    "SELECT 1 FROM discovery_occurrences o "
                    "JOIN source_revisions r ON r.revision_id=o.revision_id "
                    "WHERE o.check_outcome_id=? AND r.item_id=?",
                    (str(outcome_row["outcome_id"]), scope_id),
                ).fetchone() is not None
        if not valid:
            return "Operational Finding scope differs from exact Check lineage"
        return None

    @staticmethod
    def _baseline_evidence_error(
        conn: sqlite3.Connection,
        request: BaselineDecisionRequest,
    ) -> str | None:
        row = conn.execute(
            "SELECT v.definition_id AS version_definition_id,"
            "v.observation_model,v.baseline_policy_id,"
            "v.baseline_policy_version,r.definition_id AS request_definition_id,"
            "r.definition_version_id AS request_version_id,"
            "r.baseline_policy_id AS request_baseline_policy_id,"
            "r.baseline_policy_version AS request_baseline_policy_version,"
            "o.request_id AS outcome_request_id,"
            "o.definition_id AS outcome_definition_id,"
            "o.definition_version_id AS outcome_version_id,"
            "o.incomplete,o.kind,o.source_body_digest,"
            "o.producer_slot_digest,o.representation_digest,"
            "o.validator_digest,o.completed_at "
            "FROM source_definition_versions v "
            "JOIN check_requests r ON r.request_id=? "
            "JOIN check_outcomes o ON o.outcome_id=? "
            "WHERE v.version_id=?",
            (
                str(request.check_request_id),
                str(request.check_outcome_id),
                str(request.definition_version_id),
            ),
        ).fetchone()
        if row is None:
            return "Baseline Decision references unretained source or Check authority"
        if (
            str(row["version_definition_id"]) != str(request.definition_id)
            or str(row["observation_model"])
            != request.observation_model.value
            or str(row["baseline_policy_id"])
            != request.baseline_policy.policy_id
            or str(row["baseline_policy_version"])
            != request.baseline_policy.policy_version
            or str(row["request_definition_id"])
            != str(request.definition_id)
            or str(row["request_version_id"])
            != str(request.definition_version_id)
            or str(row["request_baseline_policy_id"])
            != request.baseline_policy.policy_id
            or str(row["request_baseline_policy_version"])
            != request.baseline_policy.policy_version
            or str(row["outcome_request_id"])
            != str(request.check_request_id)
            or str(row["outcome_definition_id"])
            != str(request.definition_id)
            or str(row["outcome_version_id"])
            != str(request.definition_version_id)
            or str(row["completed_at"]) != request.decided_at.to_text()
        ):
            return "Baseline Decision differs from exact source and Check lineage"
        if request.disposition is not BaselineDisposition.MANUAL_HOLD:
            if (
                bool(row["incomplete"])
                or str(row["kind"]) not in _COMPLETE_CONFIRMATION_OUTCOMES
                or row["source_body_digest"] != request.source_body_digest
                or row["producer_slot_digest"]
                != request.producer_slot_digest
                or row["representation_digest"]
                != request.representation_digest
                or row["validator_digest"] != request.validator_digest
            ):
                return (
                    "Baseline Decision does not consume its exact complete "
                    "Check Outcome"
                )
        for entry in request.entries:
            if entry.item_id is None:
                continue
            lineage = conn.execute(
                "SELECT i.definition_id AS item_definition_id,"
                "r.definition_id AS revision_definition_id,r.item_id "
                "FROM source_items i "
                "JOIN source_revisions r ON r.revision_id=? "
                "JOIN discovery_occurrences o "
                "ON o.revision_id=r.revision_id "
                "AND o.check_outcome_id=? "
                "WHERE i.item_id=?",
                (
                    str(entry.revision_id),
                    str(request.check_outcome_id),
                    str(entry.item_id),
                ),
            ).fetchone()
            if (
                lineage is None
                or str(lineage["item_definition_id"])
                != str(request.definition_id)
                or str(lineage["revision_definition_id"])
                != str(request.definition_id)
                or str(lineage["item_id"]) != str(entry.item_id)
            ):
                return "Baseline manifest entry differs from source lineage"
        return None

    @staticmethod
    def _transition_history_error(
        conn: sqlite3.Connection,
        request: ObservableTransitionRequest,
    ) -> str | None:
        outcome = conn.execute(
            "SELECT o.completed_at,e.ledger_seq FROM check_outcomes o "
            "JOIN ledger_events e ON e.event_id=o.authority_event_id "
            "WHERE o.outcome_id=?",
            (str(request.check_outcome_id),),
        ).fetchone()
        if outcome is None:
            return "Observable Transition references an unretained Check Outcome"
        completed_at = str(outcome["completed_at"])
        boundary = int(outcome["ledger_seq"])
        prior_outcome = conn.execute(
            "SELECT i.outcome_id FROM check_outcome_observed_items i "
            "JOIN check_outcomes o ON o.outcome_id=i.outcome_id "
            "JOIN ledger_events e ON e.event_id=o.authority_event_id "
            "WHERE i.item_id=? AND i.outcome_id!=? "
            "AND (o.completed_at<? OR "
            "(o.completed_at=? AND e.ledger_seq<?)) "
            "ORDER BY o.completed_at DESC,e.ledger_seq DESC LIMIT 1",
            (
                str(request.item_id),
                str(request.check_outcome_id),
                completed_at,
                completed_at,
                boundary,
            ),
        ).fetchone()
        prior_revision_id = None
        if prior_outcome is not None:
            prior_occurrences = conn.execute(
                "SELECT d.revision_id FROM discovery_occurrences d "
                "JOIN source_revisions r ON r.revision_id=d.revision_id "
                "WHERE d.check_outcome_id=? AND r.item_id=?",
                (
                    str(prior_outcome["outcome_id"]),
                    str(request.item_id),
                ),
            ).fetchall()
            if len(prior_occurrences) != 1:
                return (
                    "prior observed Check Outcome lacks one exact source "
                    "Occurrence"
                )
            prior_revision_id = str(prior_occurrences[0]["revision_id"])

        if request.kind in _FIRST_OBSERVATION_TRANSITIONS:
            if prior_outcome is not None:
                return (
                    "first or activation transition targets a previously "
                    "observed Source Item"
                )
        elif (
            request.prior_revision_id is None
            or prior_revision_id != str(request.prior_revision_id)
        ):
            return (
                "Observable Transition prior Revision is not the latest "
                "observed source state before its Check Outcome"
            )

        latest_transition = conn.execute(
            "SELECT t.kind FROM observable_transitions t "
            "JOIN check_outcomes o ON o.outcome_id=t.check_outcome_id "
            "JOIN ledger_events oe ON oe.event_id=o.authority_event_id "
            "JOIN ledger_events te ON te.event_id=t.authority_event_id "
            "WHERE t.item_id=? "
            "AND (o.completed_at<? OR "
            "(o.completed_at=? AND oe.ledger_seq<?)) "
            "ORDER BY o.completed_at DESC,oe.ledger_seq DESC,"
            "te.ledger_seq DESC LIMIT 1",
            (
                str(request.item_id),
                completed_at,
                completed_at,
                boundary,
            ),
        ).fetchone()
        latest_kind = (
            None
            if latest_transition is None
            else ObservableTransitionKind(str(latest_transition["kind"]))
        )
        if (
            request.kind is ObservableTransitionKind.REACTIVATED
            and latest_kind not in _ENDING_TRANSITIONS
        ):
            return (
                "reactivation transition requires the latest retained "
                "transition to be an ending state"
            )
        if request.kind in {
            ObservableTransitionKind.AGENDA_RESCHEDULED,
            ObservableTransitionKind.AGENDA_CANCELLED,
            ObservableTransitionKind.AGENDA_MISSED_EXPECTATION,
        } and latest_kind not in _AGENDA_OPEN_TRANSITIONS:
            return (
                "Agenda transition requires a retained created or "
                "rescheduled expectation"
            )
        if (
            request.kind is ObservableTransitionKind.AGENDA_LATE_OCCURRENCE
            and latest_kind
            is not ObservableTransitionKind.AGENDA_MISSED_EXPECTATION
        ):
            return (
                "late Agenda occurrence requires a retained missed expectation"
            )
        return None

    @staticmethod
    def _transition_evidence_error(
        conn: sqlite3.Connection,
        request: ObservableTransitionRequest,
    ) -> str | None:
        current_outcome = conn.execute(
            "SELECT o.*,r.adapter_request_digest,r.producer_slot_digest,"
            "r.coverage_obligation_id,r.coverage_responsibility,"
            "r.coverage_contribution,r.coverage_policy_id,"
            "r.coverage_policy_version,r.validator_policy_id,"
            "r.validator_policy_version,r.trigger_kind,"
            "r.expected_window_digest "
            "FROM check_outcomes o "
            "JOIN check_requests r ON r.request_id=o.request_id "
            "WHERE o.outcome_id=?",
            (str(request.check_outcome_id),),
        ).fetchone()
        if current_outcome is None:
            return "Observable Transition references an unretained Check Outcome"
        if str(current_outcome["completed_at"]) != request.observed_at.to_text():
            return (
                "Observable Transition observation time differs from the exact "
                "Check Outcome completion time"
            )
        history_error = _CheckStoreSupport._transition_history_error(
            conn,
            request,
        )
        if history_error is not None:
            return history_error

        if request.current_revision_id is not None:
            linked = conn.execute(
                "SELECT 1 FROM discovery_occurrences "
                "WHERE check_outcome_id=? AND revision_id=? "
                "AND representation_id=? AND definition_version_id=?",
                (
                    str(request.check_outcome_id),
                    str(request.current_revision_id),
                    str(request.representation_id),
                    str(request.definition_version_id),
                ),
            ).fetchone()
            if linked is None:
                return (
                    "Observable Transition current Revision and Representation "
                    "lack an exact Occurrence for this Check Outcome"
                )
        guard = request.absence_guard or request.agenda_guard
        if guard is None:
            return None
        references = guard.confirmation_outcomes
        if not references:
            return "transition guard has no exact confirmation Outcomes"
        current_outcome_id = str(request.check_outcome_id)
        if current_outcome_id not in {
            str(reference.outcome_id) for reference in references
        }:
            return (
                "transition guard does not include its exact current Check Outcome"
            )

        request_ids: set[str] = set()
        current_complete: bool | None = None
        for reference in references:
            row = conn.execute(
                "SELECT o.*,r.adapter_request_digest,r.producer_slot_digest,"
                "r.coverage_obligation_id,r.coverage_responsibility,"
                "r.coverage_contribution,r.coverage_policy_id,"
                "r.coverage_policy_version,r.validator_policy_id,"
                "r.validator_policy_version,r.trigger_kind,"
                "r.expected_window_digest "
                "FROM check_outcomes o "
                "JOIN check_requests r ON r.request_id=o.request_id "
                "WHERE o.outcome_id=?",
                (str(reference.outcome_id),),
            ).fetchone()
            if row is None:
                return "transition guard references an unretained Check Outcome"
            if (
                str(row["request_id"]) != str(reference.request_id)
                or str(row["adapter_request_digest"])
                != reference.adapter_request_digest
            ):
                return (
                    "transition guard confirmation reference differs from "
                    "retained Check authority"
                )
            request_id = str(row["request_id"])
            if request_id in request_ids:
                return (
                    "transition guard cannot count multiple Outcomes from one "
                    "Check Request as separate confirmations"
                )
            request_ids.add(request_id)
            if str(row["completed_at"]) > request.observed_at.to_text():
                return (
                    "transition guard confirmation Outcome occurs after the "
                    "transition observation"
                )
            complete = (
                not bool(row["incomplete"])
                and str(row["kind"]) in _COMPLETE_CONFIRMATION_OUTCOMES
                and str(row["quarantine"]) == "NONE"
            )
            if str(reference.outcome_id) == current_outcome_id:
                current_complete = complete
            if (
                str(row["definition_id"]) != str(request.definition_id)
                or str(row["definition_version_id"])
                != str(request.definition_version_id)
            ):
                return (
                    "transition guard confirmation belongs to another "
                    "Source Definition Version"
                )
            if request.absence_guard is not None:
                for column in (
                    "producer_slot_digest",
                    "coverage_obligation_id",
                    "coverage_responsibility",
                    "coverage_contribution",
                    "coverage_policy_id",
                    "coverage_policy_version",
                    "validator_policy_id",
                    "validator_policy_version",
                ):
                    if row[column] != current_outcome[column]:
                        return (
                            "absence confirmation differs from the exact "
                            "snapshot Check contract"
                        )
                if request.absence_guard.authorizes_ending and not complete:
                    return (
                        "absence-based ending cites an incomplete or failed "
                        "confirmation Outcome"
                    )
            else:
                assert request.agenda_guard is not None
                if (
                    str(row["trigger_kind"]) != "PLANNED_WINDOW"
                    or str(row["expected_window_digest"])
                    != request.agenda_guard.expected_window_digest
                ):
                    return (
                        "Agenda confirmation differs from the exact planned window"
                    )
                if not complete:
                    return (
                        "Agenda miss cites an incomplete or failed confirmation Outcome"
                    )

        if request.absence_guard is not None:
            if current_complete is None or (
                current_complete
                != request.absence_guard.successful_complete_outcome
            ):
                return (
                    "absence guard current-Outcome completeness differs from "
                    "retained authority"
                )
        else:
            assert request.agenda_guard is not None
            current = conn.execute(
                "SELECT r.trigger_kind,r.expected_window_digest "
                "FROM check_outcomes o "
                "JOIN check_requests r ON r.request_id=o.request_id "
                "WHERE o.outcome_id=?",
                (current_outcome_id,),
            ).fetchone()
            if (
                current is None
                or str(current["trigger_kind"]) != "PLANNED_WINDOW"
                or str(current["expected_window_digest"])
                != request.agenda_guard.expected_window_digest
            ):
                return (
                    "Agenda miss guard differs from the exact planned-window "
                    "Check Request"
                )
        return None

    @classmethod
    def _validate_record_envelope(
        cls,
        conn: sqlite3.Connection,
        row: Mapping[str, Any],
        *,
        command_type: str,
        aggregate_id: str,
        canonical_bytes: bytes,
        canonical_digest: str,
    ) -> sqlite3.Row:
        spec = _CHECK_RECORD_SPECS.get(command_type)
        if spec is None:
            return super()._validate_record_envelope(
                conn,
                row,
                command_type=command_type,
                aggregate_id=aggregate_id,
                canonical_bytes=canonical_bytes,
                canonical_digest=canonical_digest,
            )
        event = cls._record_context(
            conn,
            event_id=str(row["authority_event_id"]),
        )
        aggregate_type, event_type, trust_scope = spec
        if (
            str(event["event_type"]) != event_type
            or int(event["event_schema_version"]) != 1
            or str(event["aggregate_type"]) != aggregate_type
            or str(event["aggregate_id"]) != aggregate_id
            or int(event["aggregate_version"])
            != int(row["authority_aggregate_version"])
            or int(row["authority_aggregate_version"]) != 1
            or str(event["recorded_at"]) != str(row["recorded_at"])
            or str(event["security_scope"])
            != "authority.discovery_checks"
            or str(event["retention_scope"]) != "authority.audit"
            or str(event["trust_scope"]) != trust_scope.value
            or str(event["payload_mode"]) != PayloadMode.INLINE.value
            or str(event["payload_digest"]) != canonical_digest
            or event["payload_bytes"] is None
            or bytes(event["payload_bytes"]) != canonical_bytes
            or digest_bytes(canonical_bytes) != canonical_digest
        ):
            raise AuthorityPersistenceError(
                "Check record authority envelope is inconsistent"
            )
        return event

    @classmethod
    def _check_request_row(
        cls, conn: sqlite3.Connection, request_id: str
    ) -> sqlite3.Row:
        return cls._required_row_by_id(
            conn,
            table="check_requests",
            column="request_id",
            identifier=request_id,
            identity="Check Request",
        )

    @classmethod
    def _check_attempt_row(
        cls, conn: sqlite3.Connection, attempt_id: str
    ) -> sqlite3.Row:
        return cls._required_row_by_id(
            conn,
            table="check_attempts",
            column="attempt_id",
            identifier=attempt_id,
            identity="Check Attempt",
        )

    @classmethod
    def _check_outcome_row(
        cls, conn: sqlite3.Connection, outcome_id: str
    ) -> sqlite3.Row:
        return cls._required_row_by_id(
            conn,
            table="check_outcomes",
            column="outcome_id",
            identifier=outcome_id,
            identity="Check Outcome",
        )

    @classmethod
    def _baseline_decision_row(
        cls, conn: sqlite3.Connection, decision_id: str
    ) -> sqlite3.Row:
        return cls._required_row_by_id(
            conn,
            table="baseline_decisions",
            column="decision_id",
            identifier=decision_id,
            identity="Baseline Decision",
        )

    @classmethod
    def _observable_transition_row(
        cls, conn: sqlite3.Connection, transition_id: str
    ) -> sqlite3.Row:
        return cls._required_row_by_id(
            conn,
            table="observable_transitions",
            column="transition_id",
            identifier=transition_id,
            identity="Observable Transition",
        )

    @classmethod
    def _operational_finding_row(
        cls, conn: sqlite3.Connection, finding_id: str
    ) -> sqlite3.Row:
        return cls._required_row_by_id(
            conn,
            table="operational_findings",
            column="finding_id",
            identifier=finding_id,
            identity="Operational Finding",
        )

    @classmethod
    def _finding_occurrence_row(
        cls, conn: sqlite3.Connection, occurrence_id: str
    ) -> sqlite3.Row:
        return cls._required_row_by_id(
            conn,
            table="operational_finding_occurrences",
            column="occurrence_id",
            identifier=occurrence_id,
            identity="Operational Finding occurrence",
        )


__all__ = [
    "_CHECK_RECORD_SPECS",
    "_CheckStoreSupport",
    "_outcome_observed_item_error",
]
