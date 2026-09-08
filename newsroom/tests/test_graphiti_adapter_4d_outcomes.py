from __future__ import annotations

from contextlib import closing
from dataclasses import replace
from datetime import UTC, datetime, timedelta
import json
import sqlite3

import pytest

from newsroom.authority.canonical import (
    canonical_json_bytes,
    digest_bytes,
    digest_canonical,
)
from newsroom.authority.types import UtcTimestamp
from newsroom.authority.persistence import AuthorityPersistenceError
from newsroom.extraction import (
    ExtractionFailureCode,
    ExtractionOutcome,
    ExtractionOutputValidation,
    ExtractionProposalKind,
    ExtractionUsage,
    EvidenceRange,
    FixtureExtractionCase,
    ProposalDraft,
    ProducedExtraction,
    VersionedExtractionComponent,
)
from newsroom.graphiti_adapter import (
    DeterministicFakeGraphitiAdapter,
    GraphitiAdapterConfiguration,
    GraphitiAdapterConfigurationId,
    GraphitiAdapterExecution,
    GraphitiAdapterOutcome,
    GraphitiCleanupReason,
    GraphitiCleanupReceipt,
    GraphitiCredentialClass,
    GraphitiEgressPolicy,
    GraphitiExecutionProfile,
    GraphitiRuntimeMode,
    GraphitiAdapterStateError,
    GraphitiWorkspacePolicy,
    GraphitiWorkspacePolicyId,
    GraphitiWorkspaceDescriptor,
    GraphitiWorkspaceState,
    RealGraphitiRuntimeAuthority,
)
from newsroom.graphiti_adapter.evaluation_attempt import evaluation_attempt_for
from newsroom.graphiti_adapter.real import (
    RealGraphitiAdapter,
    _EpisodeTelemetry,
    _raw_receipt,
)
from newsroom.graphiti_adapter.models import adapter_outcome_for
from newsroom.graphiti_adapter.result_mapping import extraction_usage
from newsroom.graphiti_adapter.contracts import (
    GRAPHITI_ADAPTER_CODE_COMPONENT,
    GRAPHITI_ADAPTER_NORMALISATION_COMPONENT,
    GRAPHITI_ADAPTER_OUTPUT_SCHEMA_COMPONENT,
    GRAPHITI_ADAPTER_POLICY_COMPONENT,
    GRAPHITI_ADAPTER_TEMPORAL_COMPONENT,
    GRAPHITI_PROMPT_COMPONENT,
)

from .extraction_4a_helpers import (
    extraction_proof,
    open_extraction_system,
    run_request,
    seed_extraction_fixture,
)
from .graphiti_adapter_4d_authority_helpers import (
    fake_attempt,
    open_graphiti_system,
    seed_graphiti_authority_fixture,
)


def _evaluation_attempt(state):
    base = evaluation_attempt_for(("Hong Kong Transport Department",))
    request = run_request(
        state,
        contract_id=base.extraction_contract.contract_id,
        key="graphiti-evaluation-run-v1",
    )
    manifest = base.manifest.from_run_request(
        manifest_id=base.manifest.manifest_id,
        configuration=base.configuration,
        contract=base.extraction_contract,
        request=request,
    )
    return replace(base, extraction_request=request, manifest=manifest)


def _evaluation_proposal(attempt) -> ProposalDraft:
    needle = b"Hong Kong Transport Department"
    passage = next(
        item
        for item in attempt.extraction_request.input_binding.passages
        if needle in item.require_text().encode("utf-8")
    )
    data = passage.require_text().encode("utf-8")
    start = data.index(needle)
    return ProposalDraft(
        local_id="entity.0001",
        kind=ExtractionProposalKind.ENTITY_MENTION,
        subject_placeholder=needle.decode("utf-8"),
        object_placeholder=None,
        predicate_hint=None,
        confidence_basis_points=None,
        uncertainty_codes=(),
        rationale_codes=("GRAPHITI_EVALUATION_SPAN",),
        evidence=(
            EvidenceRange(
                passage_id=passage.passage_id,
                start_byte=start,
                end_byte=start + len(needle),
                evidence_text_digest=digest_bytes(needle),
            ),
        ),
    )


def _terminal_receipt(
    attempt, *, proposals: tuple[ProposalDraft, ...] = ()
) -> dict[str, object]:
    unsigned = {
        "workspace_group": attempt.configuration.workspace_policy.namespace_prefix,
        "generation_id": attempt.generation_id,
        "episode_uuid": attempt.episode_uuid,
        "attempt_number": attempt.attempt_number,
        "predecessor_episode_uuid": attempt.predecessor_episode_uuid,
        "temporal_basis": attempt.temporal_basis.value,
        "reference_time": (
            None
            if attempt.reference_time is None
            else attempt.reference_time.to_text()
        ),
        "passages": [item.canonical_value() for item in attempt.manifest.passages],
        "proposals": [item.canonical_value() for item in proposals],
    }
    return {
        **unsigned,
        "raw_output_digest": digest_bytes(canonical_json_bytes(unsigned)),
    }


def _evaluation_execution(attempt, *, outcome: str) -> GraphitiAdapterExecution:
    ended = datetime(2042, 3, 12, 9, 59, 59, tzinfo=UTC)
    started = ended - timedelta(seconds=20 if outcome == "LATE" else 1)
    complete = outcome in {"COMPLETE", "LATE"}
    proposals = (
        (_evaluation_proposal(attempt),)
        if outcome in {"COMPLETE", "LATE"}
        else ()
    )
    receipt = _terminal_receipt(attempt, proposals=proposals)
    produced = ProducedExtraction(
        outcome=(
            ExtractionOutcome.SUCCESS
            if complete
            else ExtractionOutcome.RETRYABLE_FAILURE
        ),
        failure_code=(
            ExtractionFailureCode.NONE
            if complete
            else (
                ExtractionFailureCode.AMBIGUOUS_EFFECT
                if outcome == "AMBIGUOUS_EFFECT"
                else ExtractionFailureCode.PRODUCER_INTERNAL_ERROR
            )
        ),
        validation=(ExtractionOutputValidation.VALID if complete else None),
        raw_output_value=(receipt if complete else None),
        proposals=proposals,
        usage=ExtractionUsage(
            elapsed_ms=0,
            input_bytes=attempt.extraction_request.input_binding.input_bytes,
            output_bytes=(
                len(canonical_json_bytes(receipt)) if complete else 0
            ),
            proposal_count=len(proposals),
            evidence_range_count=sum(len(item.evidence) for item in proposals),
            request_tokens=0,
            response_tokens=0,
            cost_microunits=0,
        ),
        attempt_receipt_value=(None if complete else receipt),
    )
    started_at = UtcTimestamp(started)
    ended_at = UtcTimestamp(ended)
    workspace = GraphitiWorkspaceDescriptor(
        workspace_id=attempt.workspace_id,
        configuration_id=attempt.configuration.configuration_id,
        policy_id=attempt.configuration.workspace_policy.policy_id,
        policy_digest=attempt.configuration.workspace_policy.canonical_digest,
        namespace=(
            f"{attempt.configuration.workspace_policy.namespace_prefix}-"
            f"{attempt.workspace_id}"
        ),
        created_at=started_at,
    )
    adapter_outcome = adapter_outcome_for(produced)
    return GraphitiAdapterExecution(
        attempt=attempt,
        outcome=adapter_outcome,
        failure_code=produced.failure_code.value,
        produced=produced,
        workspace=workspace,
        cleanup_receipt=GraphitiCleanupReceipt(
            receipt_id=attempt.cleanup_receipt_id,
            workspace_id=attempt.workspace_id,
            final_state=GraphitiWorkspaceState.CLEANED,
            reason=(
                GraphitiCleanupReason.NORMAL
                if complete
                else (
                    GraphitiCleanupReason.AMBIGUOUS_EFFECT
                    if outcome == "AMBIGUOUS_EFFECT"
                    else GraphitiCleanupReason.FAILED
                )
            ),
            private_node_count=0,
            private_relation_count=0,
            file_count=0,
            byte_count=0,
            workspace_absent=True,
            recorded_at=ended_at,
        ),
        started_at=started_at,
        ended_at=ended_at,
    )


@pytest.mark.parametrize(
    ("outcome", "expected_outcome", "has_output"),
    (
        ("COMPLETE", GraphitiAdapterOutcome.COMPLETE, True),
        ("FAILED", GraphitiAdapterOutcome.FAILED, False),
        ("AMBIGUOUS_EFFECT", GraphitiAdapterOutcome.AMBIGUOUS_EFFECT, False),
        ("LATE", GraphitiAdapterOutcome.TIMEOUT, False),
    ),
)
def test_evaluation_authority_atomically_retains_exact_terminal_receipt(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    outcome: str,
    expected_outcome: GraphitiAdapterOutcome,
    has_output: bool,
) -> None:
    state = seed_extraction_fixture(tmp_path / "authority")
    attempt = _evaluation_attempt(state)
    expected_receipt = _terminal_receipt(
        attempt,
        proposals=(
            (_evaluation_proposal(attempt),)
            if outcome in {"COMPLETE", "LATE"}
            else ()
        ),
    )
    workspace_root = (tmp_path / "workspace").resolve()
    adapter_calls = 0

    def provider_free_execution(_self, *, attempt, workspace_root):
        del workspace_root
        nonlocal adapter_calls
        adapter_calls += 1
        return _evaluation_execution(attempt, outcome=outcome)

    monkeypatch.setattr(RealGraphitiAdapter, "execute", provider_free_execution)
    with open_extraction_system(state) as extraction:
        extraction.extraction.register_contract(
            attempt.extraction_contract, proof=extraction_proof()
        )
    with open_graphiti_system(state, workspace_root=workspace_root) as system:
        system.graphiti.register_configuration(
            attempt.configuration, proof=extraction_proof()
        )
        retained = system.graphiti.execute_attempt(
            attempt,
            proof=extraction_proof(),
            execution_deadline=datetime(2042, 3, 12, 10, 0, tzinfo=UTC),
            fallback_permitted=False,
            invocation_observer=object(),
        )

    assert adapter_calls == 1
    assert retained.outcome is expected_outcome
    assert (retained.output_id is not None) is has_output
    assert retained.attempt_receipt == (None if has_output else expected_receipt)
    with open_graphiti_system(state, workspace_root=workspace_root) as reopened:
        assert reopened.graphiti.attempt(
            attempt.attempt_id, proof=extraction_proof()
        ) == retained
    with closing(sqlite3.connect(state.database)) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM graphiti_adapter_attempts WHERE attempt_id=?",
            (str(attempt.attempt_id),),
        ).fetchone()[0] == 1
        assert connection.execute(
            "SELECT COUNT(*) FROM graphiti_attempt_receipts WHERE attempt_id=?",
            (str(attempt.attempt_id),),
        ).fetchone()[0] == (0 if has_output else 1)
        raw_bytes = (
            None
            if retained.output_id is None
            else bytes(
                connection.execute(
                    "SELECT canonical_bytes FROM extraction_outputs "
                    "WHERE output_id=?",
                    (str(retained.output_id),),
                ).fetchone()[0]
            )
        )
    assert (
        None if raw_bytes is None else json.loads(raw_bytes)
    ) == (expected_receipt if has_output else None)


def test_evaluation_receipt_binding_failure_rolls_back_both_authorities(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = seed_extraction_fixture(tmp_path / "authority")
    attempt = _evaluation_attempt(state)
    execution = _evaluation_execution(attempt, outcome="FAILED")
    tampered = dict(execution.produced.attempt_receipt_value or {})
    tampered["workspace_group"] = "different-workspace"
    unsigned = dict(tampered)
    unsigned.pop("raw_output_digest")
    tampered["raw_output_digest"] = digest_bytes(canonical_json_bytes(unsigned))
    execution = replace(
        execution,
        produced=replace(execution.produced, attempt_receipt_value=tampered),
    )

    monkeypatch.setattr(
        RealGraphitiAdapter,
        "execute",
        lambda _self, **_values: execution,
    )
    with open_extraction_system(state) as extraction:
        extraction.extraction.register_contract(
            attempt.extraction_contract, proof=extraction_proof()
        )
    with open_graphiti_system(
        state, workspace_root=(tmp_path / "workspace").resolve()
    ) as system:
        system.graphiti.register_configuration(
            attempt.configuration, proof=extraction_proof()
        )
        with pytest.raises(
            AuthorityPersistenceError,
            match="terminal receipt differs from attempt authority",
        ):
            system.graphiti.execute_attempt(
                attempt,
                proof=extraction_proof(),
                execution_deadline=datetime(2042, 3, 12, 10, 0, tzinfo=UTC),
                fallback_permitted=False,
                invocation_observer=object(),
            )

    with closing(sqlite3.connect(state.database)) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM extraction_run_versions WHERE run_version_id=?",
            (str(attempt.extraction_request.run_version_id),),
        ).fetchone()[0] == 0
        assert connection.execute(
            "SELECT COUNT(*) FROM graphiti_adapter_attempts WHERE attempt_id=?",
            (str(attempt.attempt_id),),
        ).fetchone()[0] == 0


def _production_shaped_execution(
    attempt,
    *,
    proposals: tuple[ProposalDraft, ...] | None = None,
) -> GraphitiAdapterExecution:
    """Receipt shape emitted by RealGraphitiAdapter after a completed chat.

    Live Graphiti chat often returns SUCCESS with zero exact-span proposals.
    """

    if proposals is None:
        proposals = ()
    started_at = UtcTimestamp(datetime(2042, 3, 12, 9, 59, 58, tzinfo=UTC))
    ended_at = UtcTimestamp(datetime(2042, 3, 12, 9, 59, 59, tzinfo=UTC))
    telemetry = _EpisodeTelemetry(
        chat_invocations=[
            {
                "route": "cursor-agent-cli:composer-2.5",
                "outcome": "COMPLETE",
                "input_tokens": 4161,
                "output_tokens": 2059,
                "total_tokens": 6636,
            }
        ],
        embedding_usage={
            "requests": [],
            "request_count": 0,
            "embedding_tokens": 0,
            "cost_usd_microunits": 0,
            "usage_basis": "NO_EMBEDDING_CALL",
        },
        provider_attempt_number=1,
    )
    receipt = _raw_receipt(
        attempt,
        started_at=started_at,
        telemetry=telemetry,
        result=None,
        proposals=proposals,
    )
    produced = ProducedExtraction(
        outcome=ExtractionOutcome.SUCCESS,
        failure_code=ExtractionFailureCode.NONE,
        validation=ExtractionOutputValidation.VALID,
        raw_output_value=receipt,
        proposals=proposals,
        usage=extraction_usage(
            attempt,
            receipt,
            proposals,
            embedding_usage=telemetry.embedding_usage,
        ),
    )
    workspace = GraphitiWorkspaceDescriptor(
        workspace_id=attempt.workspace_id,
        configuration_id=attempt.configuration.configuration_id,
        policy_id=attempt.configuration.workspace_policy.policy_id,
        policy_digest=attempt.configuration.workspace_policy.canonical_digest,
        namespace=(
            f"{attempt.configuration.workspace_policy.namespace_prefix}-"
            f"{attempt.workspace_id}"
        ),
        created_at=started_at,
    )
    return GraphitiAdapterExecution(
        attempt=attempt,
        outcome=GraphitiAdapterOutcome.COMPLETE,
        failure_code=produced.failure_code.value,
        produced=produced,
        workspace=workspace,
        cleanup_receipt=GraphitiCleanupReceipt(
            receipt_id=attempt.cleanup_receipt_id,
            workspace_id=attempt.workspace_id,
            final_state=GraphitiWorkspaceState.CLEANED,
            reason=GraphitiCleanupReason.NORMAL,
            private_node_count=0,
            private_relation_count=0,
            file_count=0,
            byte_count=0,
            workspace_absent=True,
            recorded_at=ended_at,
        ),
        started_at=started_at,
        ended_at=ended_at,
    )


def test_production_shaped_receipt_persists_after_provider_complete(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = seed_extraction_fixture(tmp_path / "authority")
    attempt = _evaluation_attempt(state)
    workspace_root = (tmp_path / "workspace").resolve()
    workspace_root.mkdir()
    (workspace_root / "donor_identities.sqlite3").write_bytes(b"")

    monkeypatch.setattr(
        RealGraphitiAdapter,
        "execute",
        lambda _self, **_values: _production_shaped_execution(
            attempt, proposals=(_evaluation_proposal(attempt),)
        ),
    )
    with open_extraction_system(state) as extraction:
        extraction.extraction.register_contract(
            attempt.extraction_contract, proof=extraction_proof()
        )
    with open_graphiti_system(state, workspace_root=workspace_root) as system:
        system.graphiti.register_configuration(
            attempt.configuration, proof=extraction_proof()
        )
        retained = system.graphiti.execute_attempt(
            attempt,
            proof=extraction_proof(),
            execution_deadline=datetime(2042, 3, 12, 10, 0, tzinfo=UTC),
            fallback_permitted=False,
            invocation_observer=object(),
        )

    assert retained.outcome is GraphitiAdapterOutcome.COMPLETE
    assert retained.output_id is not None
    assert retained.proposal_set_id is not None
    with closing(sqlite3.connect(state.database)) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM extraction_run_versions WHERE run_version_id=?",
            (str(attempt.extraction_request.run_version_id),),
        ).fetchone()[0] == 1
        assert connection.execute(
            "SELECT COUNT(*) FROM graphiti_adapter_attempts WHERE attempt_id=?",
            (str(attempt.attempt_id),),
        ).fetchone()[0] == 1


def test_production_shaped_zero_proposal_complete_persists(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Accounted-zero Graphiti SUCCESS must persist as COMPLETE."""

    state = seed_extraction_fixture(tmp_path / "authority")
    attempt = _evaluation_attempt(state)
    workspace_root = (tmp_path / "workspace").resolve()
    workspace_root.mkdir()
    (workspace_root / "donor_identities.sqlite3").write_bytes(b"")

    monkeypatch.setattr(
        RealGraphitiAdapter,
        "execute",
        lambda _self, **_values: _production_shaped_execution(attempt),
    )
    with open_extraction_system(state) as extraction:
        extraction.extraction.register_contract(
            attempt.extraction_contract, proof=extraction_proof()
        )
    with open_graphiti_system(state, workspace_root=workspace_root) as system:
        system.graphiti.register_configuration(
            attempt.configuration, proof=extraction_proof()
        )
        retained = system.graphiti.execute_attempt(
            attempt,
            proof=extraction_proof(),
            execution_deadline=datetime(2042, 3, 12, 10, 0, tzinfo=UTC),
            fallback_permitted=False,
            invocation_observer=object(),
        )

    assert retained.outcome is GraphitiAdapterOutcome.COMPLETE
    assert retained.output_id is not None
    assert retained.proposal_set_id is None
    with closing(sqlite3.connect(state.database)) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM extraction_run_versions WHERE run_version_id=?",
            (str(attempt.extraction_request.run_version_id),),
        ).fetchone()[0] == 1
        assert connection.execute(
            "SELECT outcome, proposal_count FROM graphiti_adapter_attempts "
            "WHERE attempt_id=?",
            (str(attempt.attempt_id),),
        ).fetchone() == ("COMPLETE", 0)


def _digest(label: str) -> str:
    return digest_canonical({"contract": label})


def _real_configuration(contract) -> GraphitiAdapterConfiguration:
    workspace = GraphitiWorkspacePolicy(
        policy_id=GraphitiWorkspacePolicyId.parse(
            "00000000-0000-4000-8000-000000004881"
        ),
        policy_version="graphiti-disposable-workspace-v1",
        namespace_prefix="graphiti-real-evaluation",
        max_workspace_bytes=1024 * 1024,
        max_private_nodes=100,
        max_private_relations=100,
        egress_policy=GraphitiEgressPolicy.APPROVED_PROVIDER_ONLY,
        credential_class=GraphitiCredentialClass.PROPOSAL_WORKSPACE_ONLY,
    )
    framework = VersionedExtractionComponent(
        "graphiti.framework", "placeholder-release", _digest("framework")
    )
    model = VersionedExtractionComponent(
        "graphiti.model", "placeholder-release", _digest("model")
    )
    embedding = VersionedExtractionComponent(
        "graphiti.embedding", "placeholder-release", _digest("embedding")
    )
    authority = RealGraphitiRuntimeAuthority(
        authority_decision_digest=_digest("owner-decision"),
        framework_release="graphiti-placeholder-release",
        model_release="model-placeholder-release",
        embedding_release="embedding-placeholder-release",
        destination_contract_digest=_digest("destination"),
        data_processing_terms_digest=_digest("terms"),
        prompt_contract_digest=_digest("prompt"),
        output_schema_contract_digest=_digest("output"),
        permitted_expression_digest=_digest("expression"),
        rights_privacy_retention_digest=_digest("rights"),
        workspace_security_digest=_digest("workspace"),
        egress_credential_digest=_digest("egress"),
        budget_digest=_digest("budget"),
        evaluation_plan_digest=_digest("evaluation"),
        rollback_digest=_digest("rollback"),
    )
    return GraphitiAdapterConfiguration(
        configuration_id=GraphitiAdapterConfigurationId.parse(
            "00000000-0000-4000-8000-000000004882"
        ),
        runtime_mode=GraphitiRuntimeMode.REAL_GRAPHITI,
        execution_profile=GraphitiExecutionProfile.EVALUATION,
        framework=framework,
        model=model,
        embedding=embedding,
        prompt=GRAPHITI_PROMPT_COMPONENT,
        output_schema=GRAPHITI_ADAPTER_OUTPUT_SCHEMA_COMPONENT,
        code=GRAPHITI_ADAPTER_CODE_COMPONENT,
        normalisation=GRAPHITI_ADAPTER_NORMALISATION_COMPONENT,
        temporal_policy=GRAPHITI_ADAPTER_TEMPORAL_COMPONENT,
        adapter_policy=GRAPHITI_ADAPTER_POLICY_COMPONENT,
        extractor_contract_id=contract.contract_id,
        extractor_contract_digest=contract.digest,
        workspace_policy=workspace,
        fixture_case=None,
        real_runtime_authority=authority,
        idempotency_key="increment-4d-real-evaluation-placeholder-v1",
    )


@pytest.mark.parametrize(
    ("fixture_case", "expected", "failure_code", "has_output"),
    (
        (
            FixtureExtractionCase.RETRYABLE_FAILURE,
            GraphitiAdapterOutcome.FAILED,
            ExtractionFailureCode.FIXTURE_RETRYABLE,
            False,
        ),
        (
            FixtureExtractionCase.BLOCKING_FAILURE,
            GraphitiAdapterOutcome.PROVIDER_REJECTED,
            ExtractionFailureCode.FIXTURE_BLOCKED,
            False,
        ),
        (
            FixtureExtractionCase.INVALID_OUTPUT,
            GraphitiAdapterOutcome.MALFORMED_OUTPUT,
            ExtractionFailureCode.OUTPUT_SCHEMA_INVALID,
            True,
        ),
    ),
)
def test_authority_retains_honest_noncomplete_outcomes_without_proposal_admission(
    tmp_path, fixture_case, expected, failure_code, has_output
) -> None:
    state = seed_graphiti_authority_fixture(
        tmp_path / "authority", fixture_case=fixture_case
    )
    request = fake_attempt(state, fixture_case=fixture_case)
    workspace_root = (tmp_path / "workspace").resolve()
    with open_graphiti_system(state, workspace_root=workspace_root) as system:
        system.graphiti.register_configuration(
            request.configuration, proof=extraction_proof()
        )
        retained = system.graphiti.execute_attempt(
            request, proof=extraction_proof()
        )
        replayed = system.graphiti.execute_attempt(
            request, proof=extraction_proof()
        )

    assert retained.outcome is expected
    assert retained.failure_code == failure_code.value
    assert (retained.output_id is not None) is has_output
    assert retained.proposal_set_id is None
    assert replayed == replace(retained, replayed=True)
    assert retained.cleanup_receipt.workspace_absent is True
    assert not workspace_root.exists() or not any(workspace_root.iterdir())
    with closing(sqlite3.connect(state.database)) as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM entity_resolution_decisions"
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM editorial_relation_decisions"
        ).fetchone()[0] == 0


def test_policy_blocked_outcome_is_retained_without_output_or_proposals(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = seed_graphiti_authority_fixture(
        tmp_path / "authority", fixture_case=FixtureExtractionCase.BLOCKING_FAILURE
    )
    request = fake_attempt(
        state, fixture_case=FixtureExtractionCase.BLOCKING_FAILURE
    )
    workspace_root = (tmp_path / "workspace").resolve()
    original = DeterministicFakeGraphitiAdapter.execute

    def policy_blocked(self, *, attempt, workspace_root):
        execution = original(self, attempt=attempt, workspace_root=workspace_root)
        produced = replace(
            execution.produced,
            failure_code=ExtractionFailureCode.POLICY_BLOCKED,
        )
        return replace(
            execution,
            outcome=GraphitiAdapterOutcome.POLICY_BLOCKED,
            failure_code=ExtractionFailureCode.POLICY_BLOCKED.value,
            produced=produced,
            cleanup_receipt=replace(
                execution.cleanup_receipt,
                reason=GraphitiCleanupReason.POLICY_BLOCKED,
            ),
        )

    monkeypatch.setattr(DeterministicFakeGraphitiAdapter, "execute", policy_blocked)
    with open_graphiti_system(state, workspace_root=workspace_root) as system:
        system.graphiti.register_configuration(
            request.configuration, proof=extraction_proof()
        )
        retained = system.graphiti.execute_attempt(
            request, proof=extraction_proof()
        )

    assert retained.outcome is GraphitiAdapterOutcome.POLICY_BLOCKED
    assert retained.failure_code == ExtractionFailureCode.POLICY_BLOCKED.value
    assert retained.output_id is None
    assert retained.proposal_set_id is None
    assert retained.cleanup_receipt.reason is GraphitiCleanupReason.POLICY_BLOCKED


def test_preexisting_extraction_without_attempt_surfaces_ambiguous_effect(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = seed_graphiti_authority_fixture(tmp_path / "authority")
    request = fake_attempt(state)
    workspace_root = (tmp_path / "workspace").resolve()
    with open_extraction_system(state) as extraction:
        extraction.extraction.execute(
            request.extraction_request, proof=extraction_proof()
        )
    with open_graphiti_system(state, workspace_root=workspace_root) as system:
        system.graphiti.register_configuration(
            request.configuration, proof=extraction_proof()
        )

    def forbidden_execute(*_args, **_kwargs):
        raise AssertionError("ambiguous effect must not rerun the private workspace")

    monkeypatch.setattr(
        DeterministicFakeGraphitiAdapter, "execute", forbidden_execute
    )
    from newsroom.graphiti_adapter import GraphitiAdapterAmbiguousEffect

    with open_graphiti_system(state, workspace_root=workspace_root) as system:
        with pytest.raises(
            GraphitiAdapterAmbiguousEffect,
            match="explicit reconciliation is required",
        ):
            system.graphiti.execute_attempt(
                request, proof=extraction_proof()
            )
    with closing(sqlite3.connect(state.database)) as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM extraction_run_versions WHERE run_version_id=?",
            (str(request.extraction_request.run_version_id),),
        ).fetchone()[0] == 1
        assert conn.execute(
            "SELECT COUNT(*) FROM graphiti_adapter_attempts WHERE attempt_id=?",
            (str(request.attempt_id),),
        ).fetchone()[0] == 0
    assert not workspace_root.exists()


def test_public_authority_rejects_unapproved_real_runtime_workspace_configuration(
    tmp_path,
) -> None:
    state = seed_graphiti_authority_fixture(tmp_path / "authority")
    request = fake_attempt(state)
    configuration = _real_configuration(request.extraction_contract)
    workspace_root = (tmp_path / "workspace").resolve()
    with open_graphiti_system(state, workspace_root=workspace_root) as system:
        with pytest.raises(
            GraphitiAdapterStateError,
            match="workspace policy is not retained",
        ):
            system.graphiti.register_configuration(
                configuration, proof=extraction_proof()
            )
    with closing(sqlite3.connect(state.database)) as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM graphiti_adapter_configurations "
            "WHERE configuration_id=?",
            (str(configuration.configuration_id),),
        ).fetchone()[0] == 0
