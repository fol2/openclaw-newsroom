import json
import sqlite3
from contextlib import contextmanager, nullcontext
from datetime import UTC, datetime

import pytest

from newsroom.authority.canonical import canonical_json_bytes
from newsroom.control_plane.evidence import evidence_package_value
from newsroom.control_plane.native_assessor import (
    AutonomousNativeEvidenceAssessor,
    CONFIG_IDENTITY,
    CONTEXT_IDENTITY,
    CONTEXT_MANIFEST_SCHEMA_VERSION,
    NativeAssessmentExecution,
    NativeAssessmentUsage,
    SCHEMA_DIGEST,
    VERSION,
)
from newsroom.control_plane.native_evidence import NativeEvidenceError
from newsroom.control_plane.model_usage import (
    InvocationEfficiencyPolicy,
    ModelUsageService,
    WorkloadClass,
)
from newsroom.increment10.evidence import _base_package
from newsroom.control_plane.writer import (
    CONT_DISABLED_CAPABILITIES,
    CONT_PRIMARY_COMMAND_FLAGS,
)
from newsroom.tests.test_increment10_editorial import _ready_package
from newsroom.tests.test_increment10_ingress import _candidate


REVISION = "1" * 40


def _usage(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "newsroom.control_plane.native_assessor.read_grok_command_semantic_version",
        lambda: "1.0.8",
    )
    monkeypatch.setattr(
        "newsroom.control_plane.native_assessor.cont_writer_implementation_identity",
        lambda: (REVISION, True),
    )
    service = ModelUsageService(str(tmp_path / "usage.sqlite3"))
    policy = InvocationEfficiencyPolicy.create(
        policy_id="native-assessor-policy",
        version="v1",
        workload_class=WorkloadClass.NATIVE_EVIDENCE_ASSESSOR,
        provider="grok-build-cli",
        route="NATIVE_EVIDENCE_ASSESSOR",
        model="grok-4.6",
        reasoning="low",
        one_turn=True,
        exact_input=True,
        skills_enabled=False,
        tools_enabled=False,
        mcp_enabled=False,
        prior_message_count=0,
        command_semantic_version="1.0.8",
        command_flags=CONT_PRIMARY_COMMAND_FLAGS,
        context_manifest_schema_version=CONTEXT_MANIFEST_SCHEMA_VERSION,
        disabled_capabilities=CONT_DISABLED_CAPABILITIES,
        implementation_revision=REVISION,
        max_prompt_bytes=1_000_000,
        max_context_tokens=100_000,
        max_output_tokens=10_000,
        max_total_tokens=100_000,
        prompt_contract_version=VERSION,
        output_schema_digest=SCHEMA_DIGEST,
        allowed_context_identities=(CONTEXT_IDENTITY,),
        allowed_config_identities=(CONFIG_IDENTITY,),
        hard_estimate_ceiling_tokens=100_000,
        evidence_digest="sha256:" + "a" * 64,
        qualified=True,
    )
    return service, NativeAssessmentUsage(
        service, policy, clock=lambda: datetime(2026, 9, 8, tzinfo=UTC)
    )


def test_native_assessor_uses_exact_candidate_and_base_without_ambient_context(
    tmp_path, monkeypatch,
) -> None:
    connection, _port, candidate = _candidate(tmp_path)
    base = _base_package(_ready_package(candidate)[1])
    calls = []
    fence_active = False
    usage_service, usage = _usage(tmp_path, monkeypatch)

    def dispatch(prompt):
        assert fence_active
        with sqlite3.connect(usage_service.path) as retained:
            assert retained.execute(
                "SELECT state FROM model_transport_observations"
            ).fetchall() == [("DISPATCH_STARTED",)]
        calls.append(prompt)
        return NativeAssessmentExecution(
            canonical_json_bytes(
                {
                    "package": evidence_package_value(base),
                    "assessment_records": [],
                }
            ).decode(),
            {
                "usage_basis": "PROVIDER_REPORTED",
                "input_tokens": 1,
                "output_tokens": 1,
                "cached_read_tokens": 0,
                "cached_write_tokens": 0,
                "reasoning_tokens": 0,
                "context_tokens": 1,
                "total_tokens": 2,
            },
        )

    @contextmanager
    def fence():
        nonlocal fence_active
        fence_active = True
        try:
            yield
        finally:
            fence_active = False

    result = AutonomousNativeEvidenceAssessor(
        dispatch, usage=usage, dispatch_fence=fence
    )(
        candidate, base, (), ()
    )
    assert fence_active is False
    request = json.loads(calls[0])
    assert (
        request["candidate_version"]["version"]["version_id"]
        == candidate.version_id
    )
    assert request["base_package"] == evidence_package_value(base)
    assert request["output_schema_digest"] == SCHEMA_DIGEST
    assert result.governed_claims == ()
    with sqlite3.connect(usage_service.path) as retained:
        assert retained.execute(
            "SELECT outcome FROM model_invocation_terminals"
        ).fetchall() == [("ASSESSOR_ACCEPTED",)]

    bad = AutonomousNativeEvidenceAssessor(
        lambda _: NativeAssessmentExecution('{"package": {}}', {})
    )
    with pytest.raises(NativeEvidenceError):
        bad(candidate, base, (), ())
    connection.close()


@pytest.mark.parametrize(
    ("dispatch", "outcome"),
    (
        (lambda _: (_ for _ in ()).throw(RuntimeError("provider broke")),
         "ASSESSOR_PROVIDER_FAILED"),
        (lambda _: NativeAssessmentExecution('{"package": {}}', {}),
         "ASSESSOR_VALIDATION_FAILED"),
    ),
)
def test_native_assessor_retains_post_dispatch_failures(
    tmp_path, monkeypatch, dispatch, outcome,
) -> None:
    connection, _port, candidate = _candidate(tmp_path)
    base = _base_package(_ready_package(candidate)[1])
    service, usage = _usage(tmp_path, monkeypatch)

    with pytest.raises((RuntimeError, NativeEvidenceError)):
        AutonomousNativeEvidenceAssessor(
            dispatch, usage=usage, dispatch_fence=nullcontext
        )(
            candidate, base, (), ()
        )

    with sqlite3.connect(service.path) as retained:
        terminal = json.loads(retained.execute(
            "SELECT record_json FROM model_invocation_terminals"
        ).fetchone()[0])
        assert terminal["outcome"] == outcome
        assert terminal["pre_dispatch_zero_proved"] is False
        assert terminal["dispatch_at"] is not None
        assert retained.execute(
            "SELECT state FROM model_transport_observations"
        ).fetchall() == [("DISPATCH_STARTED",)]
    connection.close()
