"""Deterministic, provider-free production triage proposal worker."""

from __future__ import annotations

import uuid

from newsroom.authority.canonical import canonical_json_bytes, digest_bytes
from newsroom.discovery import NewsLead
from newsroom.increment5.retrieval_context import RETRIEVAL_CONTEXT_CONTRACT_DIGEST
from newsroom.increment6.execution import WorkerAttempt
from newsroom.increment6.proposals import (
    PROPOSAL_SCHEMA_VERSION,
    TriageProposal,
    WorkerKind,
)
from newsroom.increment6.work_items import RetrievalBindingState, TriageWorkItemVersion


AUTONOMOUS_WORKER_VERSION = "newsroom-autonomous-triage-v1"
AUTONOMOUS_TRIAGE_POLICY_VERSION = "deterministic-source-revision-no-match-v1"


class AutonomousWorkerError(ValueError):
    """The worker inputs do not form one exact, truthful proposal request."""


def _exact_inputs(
    work_item_version: TriageWorkItemVersion,
    decision_leads: tuple[NewsLead, ...],
) -> tuple[TriageWorkItemVersion, tuple[NewsLead, ...]]:
    if type(work_item_version) is not TriageWorkItemVersion:
        raise AutonomousWorkerError("work item version must be exact typed")
    version = TriageWorkItemVersion.from_canonical_bytes(
        work_item_version.canonical_bytes
    )
    if type(decision_leads) is not tuple or any(
        type(lead) is not NewsLead for lead in decision_leads
    ):
        raise AutonomousWorkerError("decision Leads must be exact committed records")
    by_id = {str(lead.request.lead_id): lead for lead in decision_leads}
    if len(by_id) != len(decision_leads) or tuple(sorted(by_id)) != tuple(
        binding.lead_id for binding in version.decision_leads
    ):
        raise AutonomousWorkerError("decision Leads differ from the Work Item")
    for binding in version.decision_leads:
        lead = by_id[binding.lead_id]
        if (
            lead.canonical_digest != binding.lead_digest
            or str(lead.event_id) != binding.lead_event_id
            or lead.aggregate_version != binding.lead_aggregate_version
            or str(lead.request.promoting_gate_decision_id)
            != binding.gate_decision_id
            or str(lead.request.definition_id) != binding.definition_id
            or str(lead.request.definition_version_id)
            != binding.definition_version_id
        ):
            raise AutonomousWorkerError(
                "decision Lead authority differs from the Work Item"
            )
    return version, tuple(by_id[key] for key in sorted(by_id))


def autonomous_worker_input_digest(
    work_item_version: TriageWorkItemVersion,
    decision_leads: tuple[NewsLead, ...],
) -> str:
    """Digest the exact native inputs before creating the Worker Attempt."""

    version, leads = _exact_inputs(work_item_version, decision_leads)
    return digest_bytes(
        canonical_json_bytes(
            {
                "worker_version": AUTONOMOUS_WORKER_VERSION,
                "policy_version": AUTONOMOUS_TRIAGE_POLICY_VERSION,
                "work_item_version_digest": version.canonical_digest,
                "decision_leads": [
                    {
                        "lead_id": str(lead.request.lead_id),
                        "lead_digest": lead.canonical_digest,
                    }
                    for lead in leads
                ],
            }
        )
    )


def _operational_hold(retrieval: object) -> tuple[str, str]:
    state = getattr(retrieval, "state", None)
    outcome = getattr(retrieval, "outcome", None)
    no_match = getattr(retrieval, "no_match", False)
    if (
        state is RetrievalBindingState.RECEIPT
        and outcome == "COMPLETE"
        and no_match is False
    ):
        return (
            "WAIT_FOR_DEPENDENCY",
            "A governed retrieval match requires an exact prior-Hypothesis relationship.",
        )
    return (
        "RETRY_RETRIEVAL",
        "Resume after a complete governed Retrieval Context is retained.",
    )


def build_autonomous_proposal(
    *,
    work_item_version: TriageWorkItemVersion,
    attempt: WorkerAttempt,
    decision_leads: tuple[NewsLead, ...],
) -> TriageProposal:
    """Build one deterministic untrusted proposal from exact native records."""

    version, leads = _exact_inputs(work_item_version, decision_leads)
    if type(attempt) is not WorkerAttempt:
        raise AutonomousWorkerError("worker attempt must be exact typed")
    attempt = WorkerAttempt.from_canonical_bytes(attempt.canonical_bytes)
    input_digest = autonomous_worker_input_digest(version, leads)
    retrieval = version.retrieval
    expected_retrieval_digest = retrieval.context_digest or retrieval.request_digest
    if (
        attempt.worker_kind is not WorkerKind.AUTONOMOUS_DETERMINISTIC
        or attempt.worker_version != AUTONOMOUS_WORKER_VERSION
        or attempt.input_digest != input_digest
        or attempt.work_item_id != version.work_item_id
        or attempt.work_item_version_id != version.version_id
        or attempt.work_item_version_digest != version.canonical_digest
        or attempt.retrieval_context_digest != expected_retrieval_digest
    ):
        raise AutonomousWorkerError("worker attempt differs from the exact inputs")

    is_new = (
        retrieval.state is RetrievalBindingState.RECEIPT
        and retrieval.outcome == "COMPLETE"
        and retrieval.no_match
    )
    action_kind, hold_condition = _operational_hold(retrieval)
    lead_ids = [str(lead.request.lead_id) for lead in leads]
    shared_urgency = min(
        (lead.request.urgency.route for lead in leads),
        key=lambda value: ("URGENT", "TIME_SENSITIVE", "PLANNED", "ROUTINE").index(
            value.value
        ),
    ).value
    shared_governing_versions = sorted(
        {
            AUTONOMOUS_TRIAGE_POLICY_VERSION,
            *(lead.request.lead_policy.policy_version for lead in leads),
        }
    )
    recommendations: list[dict[str, object]] = []
    for lead in leads:
        lead_id = str(lead.request.lead_id)
        lead_bytes = lead.request.canonical_bytes
        if not lead_bytes or len(lead_bytes) > 262_144:
            raise AutonomousWorkerError("decision Lead exceeds the citation envelope")
        information = (
            "The governed source revision has no adequate prior retrieval match."
            if is_new
            else "The governed source revision requires further deterministic triage."
        )
        recommendation: dict[str, object] = {
            "decision_lead_id": lead_id,
            "route": "NEW_EVENT_CANDIDATE" if is_new else "OPERATIONAL_HOLD",
            "confidence": {"decimal": "0.500000", "millionths": 500_000},
            "uncertainty": {"decimal": "0.500000", "millionths": 500_000},
            "input_citations": [
                {
                    "citation_id": f"citation:{lead_id}",
                    "source_kind": "DECISION_LEAD",
                    "source_id": lead_id,
                    "source_digest": lead.canonical_digest,
                    "field_path": "$",
                    "byte_start": 0,
                    "byte_end": len(lead_bytes),
                    "quote_digest": digest_bytes(lead_bytes),
                    "target_hypothesis_id": None,
                }
            ],
            "likely_new_information": information,
            "materiality_basis": (
                "The retained deterministic Gate promoted this source revision to a Lead."
            ),
            "missing_context": (
                ["Independent evidence has not yet been acquired"] if is_new else []
            ),
            "retrieval_incompleteness": ([] if is_new else [hold_condition]),
            "hypothesis": (
                {
                    "proposal_local_id": f"hypothesis:{version.work_item_id}",
                    "summary": (
                        "The governed source revision may describe a distinct new event."
                    ),
                    "relationship_kind": "NO_ADEQUATE_PRIOR_MATCH",
                    "target_hypothesis_id": None,
                }
                if is_new
                else None
            ),
            "watch_action": None,
            "supplemental_action": None,
            "operational_action": (
                None
                if is_new
                else {
                    "action_kind": action_kind,
                    "owner_id": None,
                    "dependency": (
                        "relationship-classification-policy"
                        if action_kind == "WAIT_FOR_DEPENDENCY"
                        else None
                    ),
                    "retry_condition": (
                        hold_condition if action_kind == "RETRY_RETRIEVAL" else None
                    ),
                    "review_condition": None,
                    "expiry_condition": None,
                }
            ),
            "candidate_manifest": (
                {
                    "manifest_kind": "NEW_EVENT",
                    "contributing_lead_ids": lead_ids,
                    "proposed_geography": "SOURCE_DEFINED",
                    "proposed_category": "UNCLASSIFIED",
                    "urgency": shared_urgency,
                    "likely_new_information": information,
                    "reader_utility_basis": (
                        "Independent evidence should determine whether the governed transition warrants a story."
                    ),
                    "uncertainties": ["Independent evidence has not yet been acquired"],
                    "evidence_objectives": [
                        "Acquire independent evidence for the governed source transition"
                    ],
                    "governing_versions": shared_governing_versions,
                }
                if is_new
                else None
            ),
        }
        recommendations.append(recommendation)

    proposal = {
        "proposal_id": str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"{AUTONOMOUS_WORKER_VERSION}|{attempt.attempt_id}|{input_digest}",
            )
        ),
        "work_item_binding": {
            "work_item_id": version.work_item_id,
            "work_item_version_id": version.version_id,
            "work_item_version_digest": version.canonical_digest,
        },
        "retrieval_context_binding": {
            "context_id": retrieval.context_id or retrieval.request_id,
            "context_digest": expected_retrieval_digest,
            "contract_digest": RETRIEVAL_CONTEXT_CONTRACT_DIGEST,
        },
        "worker_attempt_binding": attempt.proposal_binding.canonical_value(),
        "decision_lead_ids": lead_ids,
        "context_lead_ids": [lead.lead_id for lead in version.context_leads],
        "recommendations": recommendations,
        "rationale": (
            "A deterministic provider-free worker applied the retained retrieval outcome; "
            "the proposal remains untrusted and requires native disposition authority."
        ),
        "authority": {
            "effect": "NONE",
            "creates_hypothesis": False,
            "creates_candidate": False,
            "mutates_editorial_state": False,
            "publication_authority": False,
            "evidence_authority": False,
            "operational_authority": False,
        },
    }
    document = {
        "schema_version": PROPOSAL_SCHEMA_VERSION,
        "content_identity": digest_bytes(
            canonical_json_bytes(
                {"schema_version": PROPOSAL_SCHEMA_VERSION, "proposal": proposal}
            )
        ),
        "proposal": proposal,
    }
    return TriageProposal.from_canonical_bytes(canonical_json_bytes(document))


__all__ = [
    "AUTONOMOUS_TRIAGE_POLICY_VERSION",
    "AUTONOMOUS_WORKER_VERSION",
    "AutonomousWorkerError",
    "autonomous_worker_input_digest",
    "build_autonomous_proposal",
]
