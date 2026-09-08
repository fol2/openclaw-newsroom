from __future__ import annotations

from newsroom.authority._capability import _AuthorizedCommandGrant
from newsroom.authority._check_store_support import _observed_item_id
from newsroom.checks.check_models import (
    CheckAttemptRequest,
    CheckOutcomeRequest,
    CheckRequestRequest,
)
from newsroom.checks.policy import (
    CHECK_ATTEMPT_START_COMMAND,
    CHECK_OUTCOME_RECORD_COMMAND,
    CHECK_REQUEST_REGISTER_COMMAND,
)
from newsroom.checks.record_models import (
    CheckAttempt,
    CheckOutcome,
    CheckRequest,
)
from newsroom.checks.types import (
    CheckIdentifierReuse,
    CheckSemanticCollision,
    CheckStateError,
    CheckVersionConflict,
)


class _CheckStoreCommitCoreMixin:
    @staticmethod
    def _check_identifier_absent(
        conn,
        *,
        table: str,
        column: str,
        identifier: str,
        identity: str,
    ) -> None:
        if conn.execute(
            f"SELECT 1 FROM {table} WHERE {column}=?",
            (identifier,),
        ).fetchone() is not None:
            raise CheckIdentifierReuse(
                f"{identity} is already retained under different command identity"
            )

    @staticmethod
    def _check_semantic_absent(
        conn,
        *,
        table: str,
        semantic_digest: str,
        identity: str,
    ) -> None:
        if conn.execute(
            f"SELECT 1 FROM {table} WHERE semantic_digest=?",
            (semantic_digest,),
        ).fetchone() is not None:
            raise CheckSemanticCollision(
                f"{identity} already exists under a different stable identity"
            )

    def commit_check_request(
        self,
        grant: _AuthorizedCommandGrant,
        *,
        request: CheckRequestRequest,
    ) -> CheckRequest:
        if not isinstance(request, CheckRequestRequest):
            raise TypeError("Check Request commit requires a typed request")
        self._require_check_grant(
            grant,
            command_type=CHECK_REQUEST_REGISTER_COMMAND,
            aggregate_id=str(request.request_id),
            canonical_bytes=request.canonical_bytes,
        )
        with self._lock, self._transaction() as conn:
            if grant.replay_of_command_id is not None:
                committed = self._commit_grant_in_transaction(
                    conn,
                    grant,
                    recorded_at=self._clock().to_text(),
                )
                return self._check_request_for_event(
                    conn,
                    committed.event_id,
                    replayed=True,
                )
            version = self._require_current_version(
                conn,
                definition_id=request.definition_id,
                version_id=request.definition_version_id,
            )
            if (
                str(version["rights_decision_id"])
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
                raise CheckVersionConflict(
                    "Check Request policies differ from exact source version"
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
                raise CheckVersionConflict(
                    "Check Request coverage differs from exact source version"
                )
            self._check_identifier_absent(
                conn,
                table="check_requests",
                column="request_id",
                identifier=str(request.request_id),
                identity="Check Request identity",
            )
            self._check_semantic_absent(
                conn,
                table="check_requests",
                semantic_digest=request.semantic_digest,
                identity="Check Request semantics",
            )
            recorded_at = self._clock().to_text()
            committed = self._commit_grant_in_transaction(
                conn,
                grant,
                recorded_at=recorded_at,
            )
            if committed.replayed:
                return self._check_request_for_event(
                    conn,
                    committed.event_id,
                    replayed=True,
                )
            conn.execute(
                "INSERT INTO check_requests("
                "request_id,definition_id,definition_version_id,trigger_kind,"
                "trigger_id,trigger_version,expected_window_digest,"
                "coverage_obligation_id,coverage_responsibility,"
                "coverage_contribution,coverage_policy_id,"
                "coverage_policy_version,rights_decision_id,"
                "rights_policy_version,adapter_request_digest,"
                "producer_slot_digest,baseline_policy_id,"
                "baseline_policy_version,revision_policy_id,"
                "revision_policy_version,transition_policy_id,"
                "transition_policy_version,validator_policy_id,"
                "validator_policy_version,purpose,requested_at,"
                "semantic_digest,authority_event_id,"
                "authority_aggregate_version,canonical_bytes,"
                "canonical_digest,recorded_at) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    str(request.request_id),
                    str(request.definition_id),
                    str(request.definition_version_id),
                    request.trigger.kind.value,
                    request.trigger.trigger_id,
                    request.trigger.trigger_version,
                    request.trigger.expected_window_digest,
                    request.coverage.obligation_id,
                    request.coverage.responsibility.value,
                    request.coverage.contribution.value,
                    request.coverage.coverage_policy.policy_id,
                    request.coverage.coverage_policy.policy_version,
                    request.rights_decision_id,
                    request.rights_policy_version,
                    request.adapter_request_digest,
                    request.producer_slot_digest,
                    request.baseline_policy.policy_id,
                    request.baseline_policy.policy_version,
                    request.revision_policy.policy_id,
                    request.revision_policy.policy_version,
                    request.transition_policy.policy_id,
                    request.transition_policy.policy_version,
                    request.validator_policy.policy_id,
                    request.validator_policy.policy_version,
                    request.purpose,
                    request.requested_at.to_text(),
                    request.semantic_digest,
                    committed.event_id,
                    committed.aggregate_version,
                    request.canonical_bytes,
                    request.digest,
                    recorded_at,
                ),
            )
            return self._check_request_for_event(
                conn,
                committed.event_id,
                replayed=False,
            )

    def commit_check_attempt(
        self,
        grant: _AuthorizedCommandGrant,
        *,
        request: CheckAttemptRequest,
    ) -> CheckAttempt:
        if not isinstance(request, CheckAttemptRequest):
            raise TypeError("Check Attempt commit requires a typed request")
        self._require_check_grant(
            grant,
            command_type=CHECK_ATTEMPT_START_COMMAND,
            aggregate_id=str(request.attempt_id),
            canonical_bytes=request.canonical_bytes,
        )
        with self._lock, self._transaction() as conn:
            if grant.replay_of_command_id is not None:
                committed = self._commit_grant_in_transaction(
                    conn,
                    grant,
                    recorded_at=self._clock().to_text(),
                )
                return self._check_attempt_for_event(
                    conn,
                    committed.event_id,
                    replayed=True,
                )
            parent = self._check_request_row(conn, str(request.request_id))
            if (
                str(parent["adapter_request_digest"])
                != request.adapter_request_digest
                or request.started_at.to_text() < str(parent["requested_at"])
            ):
                raise CheckVersionConflict(
                    "Check Attempt adapter or chronology differs from Request"
                )
            latest = conn.execute(
                "SELECT attempt_id,attempt_number FROM check_attempts "
                "WHERE request_id=? ORDER BY attempt_number DESC LIMIT 1",
                (str(request.request_id),),
            ).fetchone()
            expected_number = 1 if latest is None else int(latest["attempt_number"]) + 1
            expected_prior = None if latest is None else str(latest["attempt_id"])
            if latest is not None:
                prior_outcome = conn.execute(
                    "SELECT completed_at FROM check_outcomes WHERE attempt_id=?",
                    (str(latest["attempt_id"]),),
                ).fetchone()
                if (
                    prior_outcome is None
                    or request.started_at.to_text()
                    < str(prior_outcome["completed_at"])
                ):
                    raise CheckVersionConflict(
                        "later Check Attempt requires a completed predecessor Outcome"
                    )
            actual_prior = (
                None
                if request.prior_attempt_id is None
                else str(request.prior_attempt_id)
            )
            if (
                request.attempt_number != expected_number
                or actual_prior != expected_prior
            ):
                raise CheckVersionConflict(
                    "Check Attempt does not extend exact retained attempt head"
                )
            self._check_identifier_absent(
                conn,
                table="check_attempts",
                column="attempt_id",
                identifier=str(request.attempt_id),
                identity="Check Attempt identity",
            )
            self._check_semantic_absent(
                conn,
                table="check_attempts",
                semantic_digest=request.semantic_digest,
                identity="Check Attempt semantics",
            )
            recorded_at = self._clock().to_text()
            committed = self._commit_grant_in_transaction(
                conn,
                grant,
                recorded_at=recorded_at,
            )
            if committed.replayed:
                return self._check_attempt_for_event(
                    conn,
                    committed.event_id,
                    replayed=True,
                )
            conn.execute(
                "INSERT INTO check_attempts("
                "attempt_id,request_id,attempt_number,kind,prior_attempt_id,"
                "adapter_request_id,adapter_request_digest,started_at,"
                "semantic_digest,authority_event_id,authority_aggregate_version,"
                "canonical_bytes,canonical_digest,recorded_at) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    str(request.attempt_id),
                    str(request.request_id),
                    request.attempt_number,
                    request.kind.value,
                    actual_prior,
                    str(request.adapter_request_id),
                    request.adapter_request_digest,
                    request.started_at.to_text(),
                    request.semantic_digest,
                    committed.event_id,
                    committed.aggregate_version,
                    request.canonical_bytes,
                    request.digest,
                    recorded_at,
                ),
            )
            return self._check_attempt_for_event(
                conn,
                committed.event_id,
                replayed=False,
            )

    def commit_check_outcome(
        self,
        grant: _AuthorizedCommandGrant,
        *,
        request: CheckOutcomeRequest,
    ) -> CheckOutcome:
        if not isinstance(request, CheckOutcomeRequest):
            raise TypeError("Check Outcome commit requires a typed request")
        self._require_check_grant(
            grant,
            command_type=CHECK_OUTCOME_RECORD_COMMAND,
            aggregate_id=str(request.outcome_id),
            canonical_bytes=request.canonical_bytes,
        )
        with self._lock, self._transaction() as conn:
            if grant.replay_of_command_id is not None:
                committed = self._commit_grant_in_transaction(
                    conn,
                    grant,
                    recorded_at=self._clock().to_text(),
                )
                return self._check_outcome_for_event(
                    conn,
                    committed.event_id,
                    replayed=True,
                )
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
                raise CheckVersionConflict(
                    "Check Outcome lineage or chronology differs from Attempt"
                )
            self._check_identifier_absent(
                conn,
                table="check_outcomes",
                column="outcome_id",
                identifier=str(request.outcome_id),
                identity="Check Outcome identity",
            )
            self._check_semantic_absent(
                conn,
                table="check_outcomes",
                semantic_digest=request.semantic_digest,
                identity="Check Outcome semantics",
            )
            if conn.execute(
                "SELECT 1 FROM check_outcomes WHERE proposal_id=?",
                (str(request.proposal_id),),
            ).fetchone() is not None:
                raise CheckStateError(
                    "adapter proposal is already retained by another Outcome"
                )
            recorded_at = self._clock().to_text()
            committed = self._commit_grant_in_transaction(
                conn,
                grant,
                recorded_at=recorded_at,
            )
            if committed.replayed:
                return self._check_outcome_for_event(
                    conn,
                    committed.event_id,
                    replayed=True,
                )
            conn.execute(
                "INSERT INTO check_outcomes("
                "outcome_id,request_id,attempt_id,proposal_id,definition_id,"
                "definition_version_id,kind,reason_codes_bytes,quarantine,"
                "incomplete,receipt_digest,capture_digest,parser_result_digest,"
                "source_body_digest,producer_slot_digest,representation_digest,"
                "validator_digest,candidate_observations_bytes,candidate_count,"
                "observed_items_bytes,observed_item_count,completed_at,"
                "admission_semantic_digest,semantic_digest,"
                "authority_event_id,authority_aggregate_version,canonical_bytes,"
                "canonical_digest,recorded_at) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    str(request.outcome_id),
                    str(request.request_id),
                    str(request.attempt_id),
                    str(request.proposal_id),
                    str(request.definition_id),
                    str(request.definition_version_id),
                    request.kind.value,
                    self._json_blob(list(request.reason_codes)),
                    request.quarantine.value,
                    int(request.incomplete),
                    request.receipt_digest,
                    request.capture_digest,
                    request.parser_result_digest,
                    request.source_body_digest,
                    request.producer_slot_digest,
                    request.representation_digest,
                    request.validator_digest,
                    self._json_blob(
                        [
                            item.canonical_value()
                            for item in request.candidate_observations
                        ]
                    ),
                    len(request.candidate_observations),
                    self._json_blob(
                        [item.canonical_value() for item in request.observed_items]
                    ),
                    len(request.observed_items),
                    request.completed_at.to_text(),
                    request.admission_semantic_digest,
                    request.semantic_digest,
                    committed.event_id,
                    committed.aggregate_version,
                    request.canonical_bytes,
                    request.digest,
                    recorded_at,
                ),
            )
            conn.executemany(
                "INSERT INTO check_outcome_observed_items("
                "outcome_id,item_key,item_digest,item_id) VALUES(?,?,?,?)",
                tuple(
                    (
                        str(request.outcome_id),
                        item.item_key,
                        item.item_digest,
                        str(
                            _observed_item_id(
                                self,
                                conn,
                                request_id=str(request.request_id),
                                definition_id=str(request.definition_id),
                                item_key=item.item_key,
                            )
                        ),
                    )
                    for item in request.observed_items
                ),
            )
            return self._check_outcome_for_event(
                conn,
                committed.event_id,
                replayed=False,
            )


__all__ = ["_CheckStoreCommitCoreMixin"]
