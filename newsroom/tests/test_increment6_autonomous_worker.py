from __future__ import annotations

from dataclasses import replace

import pytest

from newsroom.authority.canonical import canonical_json_bytes, digest_bytes
from newsroom.increment6.autonomous_worker import (
    AUTONOMOUS_WORKER_VERSION,
    AutonomousWorkerError,
    autonomous_worker_input_digest,
    build_autonomous_proposal,
)
from newsroom.increment6.execution import ExecutionBatchMember, WorkerAttempt
from newsroom.increment6.proposals import ProposalRoute, WorkerKind
from newsroom.increment6.work_items import (
    DecisionLeadBinding,
    RetrievalBindingState,
    RetrievalInputBinding,
    TriageWorkItem,
)
from newsroom.tests import test_increment6a2_work_items as work_item_helpers
from newsroom.tests.discovery_3d_authority_helpers import (
    exact_admission_request,
    open_discovery_system,
    proof,
    seed_check_lineage,
)


def _native_lead(tmp_path):
    with open_discovery_system(tmp_path / "authority.sqlite3") as system:
        seed_check_lineage(system)
        admitted = system.discovery.admit_signal_to_lead(
            exact_admission_request(), proof=proof()
        )
        assert admitted.lead is not None
        assert admitted.initial_disposition is not None
        return admitted.lead, DecisionLeadBinding.from_authority(
            admitted.lead, admitted.initial_disposition
        )


def _retrieval(*, no_match: bool) -> RetrievalInputBinding:
    request_id = "00000000-0000-4000-8000-000000009001"
    context_id = "00000000-0000-4000-8000-000000009002"
    request = canonical_json_bytes(
        {"idempotency_key": "autonomous-worker", "request_id": request_id}
    )
    request_digest = digest_bytes(request)
    receipt = canonical_json_bytes(
        {
            "context_id": context_id,
            "no_match": no_match,
            "outcome": "COMPLETE",
            "reason": "NO_MATCH" if no_match else None,
            "request_digest": request_digest,
            "request_id": request_id,
        }
    )
    return RetrievalInputBinding(
        RetrievalBindingState.RECEIPT,
        request_id,
        "autonomous-worker",
        request_digest,
        request,
        context_id,
        digest_bytes(receipt),
        "COMPLETE",
        "NO_MATCH" if no_match else None,
        no_match,
        receipt,
    )


def _version(binding, retrieval):
    item = TriageWorkItem.create((binding,))
    return replace(work_item_helpers._version(item), retrieval=retrieval)


def _attempt(
    version,
    lead,
    *,
    worker_kind: WorkerKind = WorkerKind.AUTONOMOUS_DETERMINISTIC,
) -> WorkerAttempt:
    retrieval_id = version.retrieval.context_id or version.retrieval.request_id
    retrieval_digest = (
        version.retrieval.context_digest or version.retrieval.request_digest
    )
    values = (
        version.work_item_id,
        version.version_id,
        version.canonical_digest,
        retrieval_id,
        retrieval_digest,
        version.priority.selection_digest,
        "sha256:" + "a" * 64,
    )
    member = ExecutionBatchMember(
        *values,
        ExecutionBatchMember._binding_digest(*values),
    )
    return WorkerAttempt.create(
        member=member,
        ordinal=1,
        worker_kind=worker_kind,
        worker_version=AUTONOMOUS_WORKER_VERSION,
        input_digest=autonomous_worker_input_digest(version, (lead,)),
    )


def test_no_match_builds_replay_stable_untrusted_candidate_proposal(tmp_path) -> None:
    lead, binding = _native_lead(tmp_path)
    version = _version(binding, _retrieval(no_match=True))
    attempt = _attempt(version, lead)

    first = build_autonomous_proposal(
        work_item_version=version, attempt=attempt, decision_leads=(lead,)
    )
    second = build_autonomous_proposal(
        work_item_version=version, attempt=attempt, decision_leads=(lead,)
    )

    assert first.canonical_bytes == second.canonical_bytes
    assert first.worker_attempt.worker_kind is WorkerKind.AUTONOMOUS_DETERMINISTIC
    assert first.recommendations[0].route is ProposalRoute.NEW_EVENT_CANDIDATE
    assert first.recommendations[0].confidence.millionths == 500_000
    assert first.recommendations[0].uncertainty.millionths == 500_000
    assert first.recommendations[0].input_citations[0].source_digest == (
        binding.lead_digest
    )
    assert not first.grants_authority
    assert not first.creates_hypothesis
    assert not first.creates_candidate


def test_match_holds_and_exact_native_bindings_fail_closed(tmp_path) -> None:
    lead, binding = _native_lead(tmp_path)
    version = _version(binding, _retrieval(no_match=False))
    attempt = _attempt(version, lead)
    proposal = build_autonomous_proposal(
        work_item_version=version, attempt=attempt, decision_leads=(lead,)
    )

    assert proposal.recommendations[0].route is ProposalRoute.OPERATIONAL_HOLD
    assert proposal.recommendations[0].operational_action is not None
    assert (
        proposal.recommendations[0].operational_action.action_kind
        == "WAIT_FOR_DEPENDENCY"
    )
    with pytest.raises(AutonomousWorkerError, match="decision Leads differ"):
        build_autonomous_proposal(
            work_item_version=version, attempt=attempt, decision_leads=()
        )
    with pytest.raises(AutonomousWorkerError, match="worker attempt differs"):
        build_autonomous_proposal(
            work_item_version=version,
            attempt=_attempt(version, lead, worker_kind=WorkerKind.REPLAY),
            decision_leads=(lead,),
        )

    pending = _version(binding, work_item_helpers._pending())
    pending_proposal = build_autonomous_proposal(
        work_item_version=pending,
        attempt=_attempt(pending, lead),
        decision_leads=(lead,),
    )
    pending_action = pending_proposal.recommendations[0].operational_action
    assert pending_action is not None
    assert pending_action.action_kind == "RETRY_RETRIEVAL"
