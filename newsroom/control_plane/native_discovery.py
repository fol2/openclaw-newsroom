"""Native discovery of retained, governed revisions delivered to Hermes.

Delivery is a new observation of retained input, not a fictitious historical
HTTP check. Source times and rights remain those of the immutable source
records. No beta Candidate identifier is promoted into editorial authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable, Mapping
import sqlite3

from newsroom.authority import AuthenticationProof, UtcTimestamp
from newsroom.authority.canonical import digest_canonical
from newsroom.checks import (
    CandidateObservationRef, CheckAttemptId, CheckAttemptKind,
    CheckAttemptRequest, CheckOutcomeId, CheckOutcomeKind, CheckOutcomeRequest,
    CheckRequestId, CheckRequestRequest, CoverageBasis, ObservableTransitionId,
    ObservableTransitionKind, ObservableTransitionRequest, QuarantineDisposition,
    TransitionBasis, TriggerKind, TriggerRef, deterministic_uuid4,
)
from newsroom.checks.record_models import CheckOutcome, ObservableTransition
from newsroom.discovery import (
    DecisionTerminality, DiscoverySignalId, DiscoverySignalRequest, GateBasis,
    GateDecisionId, GateDecisionRequest, GateOutcome, LeadDispositionDecisionId,
    LeadDispositionDecisionRequest, LeadDispositionOutcome, NewsLeadId,
    NewsLeadRequest, NextAction, NextActionKind, ObservableNewness,
    ReasonBasisClass, ReasonReference, ScopeDisposition, SignalLeadAdmissionRequest,
    StructuredReason, TimeValidity, UrgencyBasis, UrgencyRoute, DiscoveryStateError,
)
from newsroom.discovery_adapters import AdapterRequestId, ObservationProposalId
from newsroom.sources import (
    DiscoveryOccurrenceId, DiscoveryOccurrenceKind, DiscoveryOccurrenceRequest,
    DiscoveryRepresentationId, SourceDefinitionVersionId, SourceItemId,
    SourceRevisionId, VersionedPolicyRef,
)

from .corpus import CorpusIngestUnit

POLICY_VERSION = "hermes-delivered-discovery-v1"


def policy(name: str) -> VersionedPolicyRef:
    return VersionedPolicyRef(f"hermes-delivered-{name}", "v1")


def _identity(kind, phase: str, value: object):
    return deterministic_uuid4(
        kind, namespace=f"{POLICY_VERSION}:{phase}", semantic_value=value,
    )


@dataclass(frozen=True, slots=True)
class DeliveredRevision:
    unit: CorpusIngestUnit
    outcome: CheckOutcome
    transition: ObservableTransition
    occurrence_id: DiscoveryOccurrenceId


class NativeDiscovery:
    """Compose existing checked Source, Check and Discovery facades."""

    def __init__(
        self, *, sources, checks, discovery, proving: sqlite3.Connection,
        rights_for: Callable[[str, str, UtcTimestamp], Mapping[str, object] | None] | None = None,
    ) -> None:
        self.sources = sources
        self.checks = checks
        self.discovery = discovery
        self.proving = proving
        self._rights_for = rights_for

    def deliver(
        self, unit: CorpusIngestUnit, *, now: UtcTimestamp,
        proof: AuthenticationProof,
    ) -> DeliveredRevision:
        if type(unit) is not CorpusIngestUnit or unit.authority is None:
            raise ValueError("native delivery requires governed source references")
        binding = unit.authority
        version = self.sources.version_details(
            SourceDefinitionVersionId.parse(binding.definition_version_id), proof=proof,
        ).request
        item = self.sources.item(SourceItemId.parse(binding.item_id), proof=proof).request
        revision = self.sources.revision(
            SourceRevisionId.parse(binding.revision_id), proof=proof,
        ).request
        representation = self.sources.representation(
            DiscoveryRepresentationId.parse(binding.representation_id), proof=proof,
        ).request
        fields_digest = digest_canonical({
            "headline": unit.headline, "body": unit.body,
            "canonical_url": unit.canonical_url,
            "published_at": unit.published_at, "updated_at": unit.updated_at,
        })
        if (
            str(version.definition_id) != binding.definition_id
            or item.definition_id != version.definition_id
            or item.definition_version_id != version.version_id
            or revision.item_id != item.item_id
            or revision.definition_version_id != version.version_id
            or representation.revision_id != revision.revision_id
            or representation.definition_version_id != version.version_id
            or revision.permitted_state_digest != unit.revision_digest
            or representation.permitted_fields_digest != fields_digest
            or representation.representation_digest != unit.representation_digest
            or tuple((entry.name, entry.value) for entry in item.identity_components)
            != (("item_key", unit.item_key), ("source_id", unit.source_id))
            or now.value < revision.observed_at.value
            or not version.coverage_mappings
        ):
            raise ValueError("delivered input differs from exact retained source authority")
        # These are digests of the actual retained delivery, not invented HTTP
        # receipt/capture evidence. DELIVERED_INPUT keeps that boundary explicit.
        delivery = {
            "schema_version": POLICY_VERSION,
            "definition_version_digest": version.digest,
            "item_digest": item.digest, "revision_digest": revision.digest,
            "representation_digest": representation.digest,
        }
        delivery_digest = digest_canonical(delivery)
        key = str(_identity(CheckRequestId, "request", delivery))
        request_id = CheckRequestId.parse(key)
        try:
            retained_request = self.checks.request(request_id, proof=proof)
        except LookupError:
            started = now
        else:
            started = retained_request.request.requested_at
        mapping = version.coverage_mappings[0]
        coverage = CoverageBasis(
            mapping.obligation_id, mapping.responsibility, mapping.contribution,
            policy("coverage"),
        )
        request = self.checks.register_request(CheckRequestRequest(
            request_id=request_id, definition_id=version.definition_id,
            definition_version_id=version.version_id,
            trigger=TriggerRef(TriggerKind.DELIVERED_INPUT, f"revision:{revision.revision_id}", POLICY_VERSION),
            coverage=coverage, rights_decision_id=version.rights.rights_decision_id,
            rights_policy_version=version.rights.rights_policy_version,
            adapter_request_digest=delivery_digest,
            producer_slot_digest=representation.producer_slot_digest,
            baseline_policy=version.baseline_policy.reference,
            revision_policy=version.revision_policy,
            transition_policy=policy("transition"), validator_policy=policy("retained-input"),
            purpose="Observe exact governed input delivered to native Hermes discovery.",
            requested_at=started, idempotency_key=f"native-request:{key}",
        ), proof=proof)
        attempt_id = _identity(CheckAttemptId, "attempt", key)
        self.checks.start_attempt(CheckAttemptRequest(
            attempt_id=attempt_id, request_id=request_id, attempt_number=1,
            kind=CheckAttemptKind.PRIMARY, prior_attempt_id=None,
            adapter_request_id=_identity(AdapterRequestId, "adapter", key),
            adapter_request_digest=delivery_digest, started_at=started,
            idempotency_key=f"native-attempt:{key}",
        ), proof=proof)
        observation = CandidateObservationRef(
            item.identity_digest, representation.representation_digest,
        )
        outcome_id = _identity(CheckOutcomeId, "outcome", key)
        try:
            retained_outcome = self.checks.outcome(outcome_id, proof=proof)
        except LookupError:
            reobserved = bool(self.sources.occurrences(revision.revision_id, limit=1, proof=proof))
        else:
            reobserved = retained_outcome.request.kind is CheckOutcomeKind.SUCCESS_UNCHANGED
        outcome = self.checks.record_outcome(CheckOutcomeRequest(
            outcome_id=outcome_id, request_id=request_id, attempt_id=attempt_id,
            proposal_id=_identity(ObservationProposalId, "observation", key),
            definition_id=version.definition_id, definition_version_id=version.version_id,
            kind=CheckOutcomeKind.SUCCESS_UNCHANGED if reobserved else CheckOutcomeKind.SUCCESS_CHANGED,
            reason_codes=("GOVERNED_REVISION_DELIVERED",),
            quarantine=QuarantineDisposition.NONE, incomplete=False,
            receipt_digest=delivery_digest, capture_digest=fields_digest,
            parser_result_digest=representation.digest,
            source_body_digest=revision.permitted_state_digest,
            producer_slot_digest=representation.producer_slot_digest,
            representation_digest=representation.representation_digest,
            validator_digest=None, candidate_observations=() if reobserved else (observation,),
            observed_items=(observation,), completed_at=started,
            idempotency_key=f"native-outcome:{key}",
        ), proof=proof)
        occurrence_id = _identity(DiscoveryOccurrenceId, "occurrence", key)
        self.sources.record_occurrence(DiscoveryOccurrenceRequest(
            occurrence_id=occurrence_id, check_outcome_id=outcome_id,
            revision_id=revision.revision_id, representation_id=representation.representation_id,
            definition_version_id=version.version_id,
            kind=DiscoveryOccurrenceKind.DELIVERED, observed_at=started,
            receipt_digest=delivery_digest, source_asserted_time=revision.source_updated_time,
            idempotency_key=f"native-occurrence:{key}",
        ), proof=proof)
        transition_id = _identity(ObservableTransitionId, "transition", key)
        try:
            transition = self.checks.transition(transition_id, proof=proof)
        except LookupError:
            # A retained historical revision is FIRST_OBSERVED in this native
            # stream unless its predecessor has actually been delivered here.
            prior = revision.revision_id if reobserved else revision.prior_revision_id
            if prior is not None and not self.sources.occurrences(prior, limit=1, proof=proof):
                prior = None
            transition = self.checks.record_transition(ObservableTransitionRequest(
                transition_id=transition_id, definition_id=version.definition_id,
                definition_version_id=version.version_id, check_outcome_id=outcome_id,
                item_id=item.item_id,
                kind=(ObservableTransitionKind.REOBSERVED if reobserved else ObservableTransitionKind.FIRST_OBSERVED if prior is None else ObservableTransitionKind.REVISED),
                basis=TransitionBasis.REVISION, observation_model=version.observation_model,
                prior_revision_id=prior, current_revision_id=revision.revision_id,
                representation_id=representation.representation_id, related_item_id=None,
                change_facets=() if prior is None or reobserved else ("PERMITTED_STATE",),
                transition_policy=request.request.transition_policy,
                absence_guard=None, agenda_guard=None,
                source_asserted_time=revision.source_updated_time, observed_at=started,
                transition_discriminator="retained-revision",
                idempotency_key=f"native-transition:{key}",
            ), proof=proof)
        return DeliveredRevision(unit, outcome, transition, occurrence_id)

    def admit_lead(
        self, delivered: DeliveredRevision, *, now: UtcTimestamp,
        proof: AuthenticationProof,
    ):
        """Queue genuine retained revisions; freshness and rights remain explicit.

        This deterministic gate is not a publisher and does not claim Graphiti
        retrieval completeness or evidence acquisition rights. Those are checked
        at their owning downstream boundary.
        """
        if type(delivered) is not DeliveredRevision:
            raise ValueError("native gate requires an exact delivered revision")
        transition = self.checks.transition(
            delivered.transition.request.transition_id, proof=proof,
        ).request
        unit = delivered.unit
        binding = unit.authority
        assert binding is not None
        version = self.sources.version_details(transition.definition_version_id, proof=proof).request
        outcome = self.checks.outcome(transition.check_outcome_id, proof=proof)
        if outcome.request != delivered.outcome.request or transition != delivered.transition.request:
            raise ValueError("delivered discovery records differ from retained authority")
        request = self.checks.request(outcome.request.request_id, proof=proof).request
        source_item = self.sources.item(transition.item_id, proof=proof).request
        source_revision = self.sources.revision(transition.current_revision_id, proof=proof).request
        source_id = dict((entry.name, entry.value) for entry in source_item.identity_components).get("source_id")
        if source_id != unit.source_id or str(source_revision.revision_id) != binding.revision_id:
            raise ValueError("delivered discovery source identity differs")
        key = str(transition.transition_id)
        signal_id = _identity(DiscoverySignalId, "signal", key)
        try:
            current = self.discovery.current_status(signal_id, proof=proof)
        except DiscoveryStateError as exc:
            if str(exc) != "discovery_signals record is not retained":
                raise
            current = None
        summary = self.sources.current_summary(version.definition_id, proof=proof)
        current_version = summary.version_id == version.version_id
        age = (now.value - source_revision.observed_at.value).total_seconds()
        window = version.baseline_policy.freshness_window_seconds
        fresh = age >= 0 and window is not None and age <= window
        # Maintained/current-state sources need source-specific currentness;
        # elapsed time alone never invents it. Their retained Signal is visible.
        time_validity = TimeValidity.CURRENT if fresh else (TimeValidity.STALE if window else TimeValidity.UNKNOWN)
        if self._rights_for is None:
            from .cycle import _dispatch_rights_decision
            rights = _dispatch_rights_decision(
                self.proving, source_id=unit.source_id,
                source_url=version.locator, evaluated_at=now.to_text(),
            )
        else:
            rights = self._rights_for(unit.source_id, version.locator, now)
        rights_current = current_version and rights is not None
        ready = rights_current and fresh
        repeated = transition.kind is ObservableTransitionKind.REOBSERVED
        ordinal = 1 if current is None else current.current_gate.request.decision_ordinal + 1
        state_key = {"transition": key, "current_version": str(summary.version_id), "time_validity": time_validity.value}
        if current is not None and (
            current.current_gate.request.basis.rights_current == rights_current
            and current.current_gate.request.basis.policy_current == current_version
            and current.current_gate.request.basis.time_validity == time_validity
        ):
            return current
        state_key["rights_packet"] = None if rights is None else rights["packet_digest"]
        state_key["ordinal"] = ordinal
        gate_id = _identity(GateDecisionId, "gate", state_key)
        lead_id = _identity(NewsLeadId, "lead", key)
        reason = StructuredReason(
            "CHANGE.EXACT_REPEAT" if ready and repeated else "CHANGE.GENUINE_TRANSITION" if ready else "OPS.SOURCE_CURRENTNESS_PENDING",
            ReasonBasisClass.DETERMINISTIC_OBSERVATION,
            (ReasonReference("OBSERVABLE_TRANSITION", key, delivered.transition.canonical_digest),),
            "Native retained revision transition; source currentness evaluated without changing source rights.",
        )
        action = NextAction(
            NextActionKind.CLOSE if ready and repeated else NextActionKind.QUEUE_TRIAGE if ready else NextActionKind.WAIT_DEPENDENCY,
            "CLOSE_EXACT_REPEAT" if ready and repeated else "QUEUE_FOR_TRIAGE" if ready else "WAIT_SOURCE_CURRENTNESS",
            dependency=None if ready else "current-source-version",
            instructions="Resume automatically when the exact source dependency changes.",
        )
        signal = DiscoverySignalRequest(
            signal_id=signal_id, definition_id=version.definition_id,
            definition_version_id=version.version_id, item_id=transition.item_id,
            revision_id=transition.current_revision_id, representation_id=transition.representation_id,
            check_outcome_id=transition.check_outcome_id, occurrence_id=delivered.occurrence_id,
            transition_id=transition.transition_id, purpose="SOURCE_TRANSITION", discriminator="native",
            admission_policy=policy("signal"), incomplete=False, operational_finding_ids=(),
            admitted_at=delivered.outcome.request.completed_at,
            idempotency_key=f"native-signal:{key}",
        )
        gate = GateDecisionRequest(
            decision_id=gate_id, signal_id=signal_id, decision_ordinal=ordinal,
            previous_decision_id=None if current is None else current.current_gate.request.decision_id,
            evaluated_definition_version_id=version.version_id, coverage=request.coverage,
            rights_decision_id=version.rights.rights_decision_id,
            rights_policy_version=version.rights.rights_policy_version,
            signal_admission_policy=signal.admission_policy, gate_policy=policy("gate"),
            duplicate_policy=policy("exact-revision"), newness_policy=policy("transition"),
            time_validity_policy=policy("source-baseline-window"), exclusion_policy=policy("source-coverage"),
            basis=GateBasis(
                identity_integrity=True, duplicate_signal_id=None, duplicate_rule=None,
                observable_newness=ObservableNewness.EXACT_REPEAT if repeated else ObservableNewness.GENUINE_TRANSITION,
                time_validity=time_validity, scope_disposition=ScopeDisposition.ACCEPTED,
                clear_exclusion_rule=None, rights_current=rights_current,
                policy_current=current_version, operationally_executable=ready,
            ),
            outcome=GateOutcome.SUPPRESSED_NON_CHANGE if ready and repeated else GateOutcome.PROMOTED_TO_LEAD if ready else GateOutcome.OPERATIONAL_HOLD,
            terminality=DecisionTerminality.TERMINAL_EXACT_VERSION if ready else DecisionTerminality.PENDING_CONDITION,
            primary_reason=reason, supporting_reasons=(),
            reason_taxonomy_version=POLICY_VERSION, outcome_taxonomy_version=POLICY_VERSION,
            next_action=action, decided_at=now, idempotency_key=f"native-gate:{gate_id}",
        )
        urgency = UrgencyBasis(UrgencyRoute.ROUTINE, reason)
        lead = None
        disposition = None
        try:
            existing_lead = self.discovery.lead_for_signal(signal_id, proof=proof) if current is not None else None
        except LookupError:
            existing_lead = None
        if ready and not repeated and existing_lead is None:
            lead = NewsLeadRequest(
                lead_id=lead_id, signal_id=signal_id, promoting_gate_decision_id=gate_id,
                definition_id=version.definition_id, definition_version_id=version.version_id,
                item_id=transition.item_id, revision_id=transition.current_revision_id,
                representation_id=transition.representation_id, occurrence_id=delivered.occurrence_id,
                transition_id=transition.transition_id, transition_kind=transition.kind,
                coverage=request.coverage, source_roles=version.roles,
                portfolio_functions=version.portfolio_functions, source_dependencies=version.dependencies,
                incompleteness_warnings=(), urgency=urgency, lead_policy=policy("lead"),
                reason_taxonomy_version=POLICY_VERSION, outcome_taxonomy_version=POLICY_VERSION,
                created_at=now, idempotency_key=f"native-lead:{key}",
            )
            disposition = LeadDispositionDecisionRequest(
                decision_id=_identity(LeadDispositionDecisionId, "queued", key), lead_id=lead_id,
                gate_decision_id=gate_id, decision_ordinal=1, previous_decision_id=None,
                outcome=LeadDispositionOutcome.QUEUED_FOR_TRIAGE,
                terminality=DecisionTerminality.PENDING_CONDITION, primary_reason=reason,
                supporting_reasons=(), watch_condition_id=None, next_action=action,
                urgency_route=urgency, disposition_policy=policy("disposition"),
                reason_taxonomy_version=POLICY_VERSION, outcome_taxonomy_version=POLICY_VERSION,
                decided_at=now, idempotency_key=f"native-queued:{key}",
            )
        if existing_lead is not None:
            self.discovery.decide_gate(gate, proof=proof)
        else:
            self.discovery.admit_signal_to_lead(
                SignalLeadAdmissionRequest(signal, gate, lead, disposition), proof=proof,
            )
        return self.discovery.current_status(signal_id, proof=proof)
