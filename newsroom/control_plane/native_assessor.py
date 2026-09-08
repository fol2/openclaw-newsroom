"""Autonomous, fail-closed assessment of independently acquired evidence."""

from __future__ import annotations

import json
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from typing import Callable

from newsroom.authority.canonical import (
    canonical_json_bytes,
    digest_bytes,
    digest_canonical,
)
from newsroom.control_plane.evidence import (
    ClaimAuthorityClass,
    EvidencePackage,
    evidence_package_value,
)
from newsroom.increment10.editorial import SourceCurrentness
from newsroom.increment10.evidence import _base_package, _package_from_value

from .model_usage import (
    InvocationAllocation,
    InvocationEfficiencyPolicy,
    ModelUsageService,
    WorkEnvelope,
    WorkloadClass,
)

from .native_evidence import (
    AcquiredEvidence,
    AcquiredSourceAssessment,
    IndependentEvidenceAssessment,
    NativeEvidenceError,
    NativeEvidenceHold,
    NativeEvidenceSource,
    SourceAuthorityAssessment,
)
from .writer import (
    CONT_DISABLED_CAPABILITIES,
    CONT_PRIMARY_COMMAND_FLAGS,
    CONT_PRIMARY_MODEL,
    CONT_PRIMARY_PROVIDER,
    CONT_PRIMARY_REASONING,
    WriterDispatchError,
    _run_grok_json,
    cont_writer_implementation_identity,
    read_grok_command_semantic_version,
)
from .cycle import _complete_writer_usage

VERSION = "newsroom.native-evidence-assessor.v1"
ROUTE = "NATIVE_EVIDENCE_ASSESSOR"
CONTEXT_IDENTITY = "native-evidence-exact-acquisition-v1"
CONFIG_IDENTITY = "native-evidence-assessor-grok-hermetic-command-v1"
CONTEXT_MANIFEST_SCHEMA_VERSION = (
    "newsroom.native-evidence-assessor.context-manifest.v1"
)
SYSTEM = (
    "You are a one-turn evidence extraction transform. Use only the supplied "
    "candidate and exact source bytes. Return JSON matching the schema. Never "
    "claim facts, translations or authority absent from an exact source excerpt."
)
SCHEMA = {
    "type": "object",
    "required": ["package", "assessment_records"],
    "additionalProperties": False,
    "properties": {
        "package": {"type": "object"},
        "assessment_records": {"type": "array", "items": {"type": "object"}},
    },
}
SCHEMA_DIGEST = digest_bytes(canonical_json_bytes(SCHEMA))
INTEGRITY = (
    "ACCESS_COMPLETE",
    "ENCODING_VALID",
    "EXTRACTION_COMPLETE",
    "NOT_PAYWALL_FRAGMENT",
    "NOT_TRUNCATED",
    "VERSION_UNAMBIGUOUS",
)


@dataclass(frozen=True, slots=True)
class NativeAssessmentExecution:
    text: str
    usage: dict[str, object]


class NativeAssessmentUsage:
    """Persist exact native-assessor intent, dispatch and terminal usage."""

    def __init__(
        self,
        service: ModelUsageService,
        policy: InvocationEfficiencyPolicy,
        *,
        clock: Callable[[], datetime] = lambda: datetime.now(tz=UTC),
    ) -> None:
        if (
            type(service) is not ModelUsageService
            or type(policy) is not InvocationEfficiencyPolicy
            or policy.workload_class is not WorkloadClass.NATIVE_EVIDENCE_ASSESSOR
            or (policy.provider, policy.route, policy.model, policy.reasoning)
            != (
                CONT_PRIMARY_PROVIDER,
                ROUTE,
                CONT_PRIMARY_MODEL,
                CONT_PRIMARY_REASONING,
            )
            or policy.prompt_contract_version != VERSION
            or policy.output_schema_digest != SCHEMA_DIGEST
            or policy.command_flags != CONT_PRIMARY_COMMAND_FLAGS
            or policy.context_manifest_schema_version
            != CONTEXT_MANIFEST_SCHEMA_VERSION
            or policy.disabled_capabilities != CONT_DISABLED_CAPABILITIES
            or CONTEXT_IDENTITY not in policy.allowed_context_identities
            or CONFIG_IDENTITY not in policy.allowed_config_identities
            or not policy.qualified
        ):
            raise NativeEvidenceError("qualified native assessment usage is required")
        self._service = service
        self._policy = policy
        self._clock = clock
        service.register_policy(policy)

    def begin(self, candidate, base, prompt: str) -> InvocationAllocation:
        now = self._clock().astimezone(UTC)
        prompt_bytes = prompt.encode()
        package_bytes = canonical_json_bytes(evidence_package_value(base))
        command_version = read_grok_command_semantic_version()
        implementation_revision, implementation_clean = (
            cont_writer_implementation_identity()
        )
        policy = self._policy
        if (
            command_version != policy.command_semantic_version
            or implementation_revision != policy.implementation_revision
            or implementation_clean is not True
        ):
            raise NativeEvidenceError("native assessment runner identity differs")
        manifest = {
            "schema_version": CONTEXT_MANIFEST_SCHEMA_VERSION,
            "provider": policy.provider,
            "route": policy.route,
            "model": policy.model,
            "reasoning": policy.reasoning,
            "command_semantic_version": command_version,
            "command_flags": list(CONT_PRIMARY_COMMAND_FLAGS),
            "disabled_capabilities": list(CONT_DISABLED_CAPABILITIES),
            "implementation_revision": implementation_revision,
            "implementation_worktree_clean": True,
            "prompt_contract_version": VERSION,
            "prompt_bytes": len(prompt_bytes),
            "prompt_digest": digest_bytes(prompt_bytes),
            "schema_digest": SCHEMA_DIGEST,
            "output_schema_digest": SCHEMA_DIGEST,
            "system_digest": digest_bytes(SYSTEM.encode()),
            "evidence_package_digest": base.digest,
            "evidence_package_bytes": len(package_bytes),
            "context_identity": CONTEXT_IDENTITY,
            "config_identity": CONFIG_IDENTITY,
            "one_turn": True,
            "exact_input": True,
            "skills_enabled": False,
            "tools_enabled": False,
            "mcp_enabled": False,
            "prior_message_count": 0,
            "skill_count": 0,
            "tool_count": 0,
            "mcp_server_count": 0,
            "mcp_tool_count": 0,
        }
        manifest["request_digest"] = digest_canonical(
            {
                key: manifest[key]
                for key in (
                    "provider",
                    "route",
                    "model",
                    "reasoning",
                    "command_semantic_version",
                    "command_flags",
                    "implementation_revision",
                    "system_digest",
                    "prompt_digest",
                    "output_schema_digest",
                )
            }
        )
        manifest["context_manifest_digest"] = digest_canonical(manifest)
        cycle_id = digest_bytes(
            canonical_json_bytes([candidate.version_id, base.digest])
        )
        envelope = WorkEnvelope.create(
            cycle_id=cycle_id,
            workload_class=WorkloadClass.NATIVE_EVIDENCE_ASSESSOR,
            admitted_at=now,
            admission_decision_id=None,
            candidate_id=candidate.candidate_id,
            hypothesis_digest=candidate.governing_manifest.canonical_digest,
            evidence_package_digest=base.digest,
            ingest_id=None,
            graphiti_attempt_id=None,
        )
        self._service.open_envelope(envelope)
        self._service.retain_context_manifest(manifest)
        allocation = InvocationAllocation.create(
            envelope_id=envelope.envelope_id,
            cycle_id=cycle_id,
            leaf_ordinal=1,
            workload_class=WorkloadClass.NATIVE_EVIDENCE_ASSESSOR,
            invocation_policy_digest=policy.canonical_digest,
            provider=policy.provider,
            route=policy.route,
            model=policy.model,
            reasoning=policy.reasoning,
            prompt_contract_version=VERSION,
            prompt_bytes=len(prompt_bytes),
            prompt_digest=digest_bytes(prompt_bytes),
            request_digest=str(manifest["request_digest"]),
            output_schema_digest=SCHEMA_DIGEST,
            max_output_tokens=policy.max_output_tokens,
            context_manifest_digest=str(manifest["context_manifest_digest"]),
            context_identity=CONTEXT_IDENTITY,
            config_identity=CONFIG_IDENTITY,
            one_turn=True,
            exact_input=True,
            skills_enabled=False,
            tools_enabled=False,
            mcp_enabled=False,
            prior_message_count=0,
            allocated_at=now,
            recovery_deadline_at=now + timedelta(minutes=5),
            parent_invocation_id=None,
        )
        self._service.allocate(allocation, owner_emergency_stop=False)
        return allocation

    def mark_dispatch(self, allocation: InvocationAllocation) -> datetime:
        dispatched_at = self._clock().astimezone(UTC)
        self._service.observe_transport(
            invocation_id=allocation.invocation_id,
            observed_at=dispatched_at,
            state="DISPATCH_STARTED",
            evidence_digest=allocation.request_digest,
        )
        return dispatched_at

    def complete(
        self,
        allocation: InvocationAllocation,
        *,
        outcome: str,
        execution: NativeAssessmentExecution | None,
        provider_dispatched: bool,
        dispatch_at: datetime | None = None,
        failure_class: str | None = None,
    ) -> None:
        now = self._clock().astimezone(UTC)
        _complete_writer_usage(
            self._service,
            allocation,
            outcome=outcome,
            failure_class=failure_class,
            usage=None if execution is None else execution.usage,
            dispatch_at=dispatch_at if provider_dispatched else None,
            completed_at=now,
            provider_dispatched=provider_dispatched,
            policy=self._policy,
        )


class AutonomousNativeEvidenceAssessor:
    """Dispatch one fixed-schema transform, then prove its output locally."""

    def __init__(
        self,
        dispatch: Callable[[str], NativeAssessmentExecution] | None = None,
        *,
        usage: NativeAssessmentUsage | None = None,
        dispatch_fence: Callable[[], AbstractContextManager] | None = None,
    ) -> None:
        default_dispatch = dispatch is None
        dispatch = dispatch or _dispatch_grok
        if not callable(dispatch):
            raise NativeEvidenceError("native assessment transport is required")
        self._dispatch = dispatch
        if default_dispatch and usage is None:
            raise NativeEvidenceError("native assessment usage authority is required")
        if usage is not None and dispatch_fence is None:
            raise NativeEvidenceError("native assessment dispatch fence is required")
        if dispatch_fence is not None and not callable(dispatch_fence):
            raise NativeEvidenceError("native assessment dispatch fence differs")
        self._usage = usage
        self._dispatch_fence = dispatch_fence or nullcontext

    def __call__(self, candidate, base, sources, acquired):
        for source, result in zip(sources, acquired, strict=True):
            if (
                result.currentness_basis
                != "AUTHORITATIVE_CURRENT_CONTENT_ENDPOINT"
                or not result.text_only
                or result.rights_eligibility_digest
                != digest_canonical(
                    {
                        "rights_receipt": source.rights.record_id,
                        "body_digest": result.body_digest,
                        "transport": result.transport_evidence_digest,
                        "exclusion_signals": result.exclusion_signals,
                        "text_only": result.text_only,
                    }
                )
                or not result.licence_attribution
                or result.exclusion_signals
                or source.rights.decision != "PERMITTED"
                or source.rights.permitted_use != "PUBLICATION_EVIDENCE"
            ):
                raise NativeEvidenceHold(
                    "SOURCE_POLICY_FACTS_HOLD", source.unit.source_id
                )
        prompt = canonical_json_bytes(
            {
                "contract": VERSION,
                "candidate_version": json.loads(candidate.canonical_bytes),
                "base_package": evidence_package_value(base),
                "sources": [
                    {
                        "source_id": source.unit.source_id,
                        "source_definition_version_digest": (
                            source.source_version.canonical_digest
                        ),
                        "rights_receipt_id": source.rights.record_id,
                        "dependency_receipt_id": source.dependency.record_id,
                        "acquisition_receipt_id": result.receipt_digest,
                        "publication_time": result.publication_time,
                        "source_updated_time": result.source_updated_time,
                        "retrieval_time": result.retrieval_time,
                        "body": result.body.decode("utf-8"),
                    }
                    for source, result in zip(sources, acquired, strict=True)
                ],
                "output_schema_digest": SCHEMA_DIGEST,
            }
        ).decode()
        request = prompt
        allocation = (
            None if self._usage is None else self._usage.begin(candidate, base, request)
        )
        execution = None
        dispatch_at = None
        try:
            with self._dispatch_fence():
                if allocation is not None:
                    dispatch_at = self._usage.mark_dispatch(allocation)
                execution = self._dispatch(request)
            result = self._validated_execution(
                execution, candidate, base, sources, acquired
            )
        except WriterDispatchError as exc:
            if allocation is not None:
                self._usage.complete(
                    allocation, outcome="ASSESSOR_PROVIDER_FAILED",
                    execution=None,
                    provider_dispatched=dispatch_at is not None,
                    dispatch_at=dispatch_at,
                    failure_class=exc.failure_class,
                )
            raise
        except BaseException:
            if allocation is not None:
                self._usage.complete(
                    allocation,
                    outcome=(
                        "ASSESSOR_PROVIDER_FAILED"
                        if execution is None
                        else "ASSESSOR_VALIDATION_FAILED"
                    ),
                    execution=execution,
                    provider_dispatched=dispatch_at is not None,
                    dispatch_at=dispatch_at,
                    failure_class=(
                        "UNKNOWN_PROVIDER_FAILURE"
                        if execution is None
                        else "ASSESSMENT_VALIDATION_FAILED"
                    ),
                )
            raise
        if allocation is not None:
            self._usage.complete(
                allocation, outcome="ASSESSOR_ACCEPTED", execution=execution,
                provider_dispatched=True, dispatch_at=dispatch_at,
            )
        return result

    @staticmethod
    def _validated_execution(execution, candidate, base, sources, acquired):
        if type(execution) is not NativeAssessmentExecution:
            raise NativeEvidenceHold("ASSESSOR_TRANSPORT_HOLD", sources[0].unit.source_id)
        value = _document(execution.text)
        package = _package_from_value(value.get("package"))
        if _base_package(package) != base:
            raise NativeEvidenceHold("ASSESSOR_BASE_BINDING_HOLD", sources[0].unit.source_id)
        source_ids = {source.unit.source_id for source in sources}
        receipt_by_source = {
            source.unit.source_id: result.receipt_digest
            for source, result in zip(sources, acquired, strict=True)
        }
        for claim in package.governed_claims:
            if (
                claim.passage_index >= len(acquired)
                or claim.supporting_excerpt
                not in acquired[claim.passage_index].body.decode("utf-8")
                or set(claim.source_ids) - source_ids
                or set(claim.source_record_ids)
                != {receipt_by_source[item] for item in claim.source_ids}
            ):
                raise NativeEvidenceHold(
                    "ASSESSOR_CLAIM_BINDING_HOLD", sources[0].unit.source_id
                )
        source_by_id = {source.unit.source_id: source for source in sources}
        authority = []
        governed_claims = []
        for claim in package.governed_claims:
            selected = tuple(source_by_id[item] for item in claim.source_ids)
            source_roles = tuple(
                tuple(
                    assignment
                    for assignment in source.source_version.request.roles
                    if assignment.role.value
                    in {"ORIGINATING_AUTHORITY", "RESPONSIBLE_OPERATOR"}
                )
                for source in selected
            )
            if any(len(roles) != 1 for roles in source_roles):
                raise NativeEvidenceHold(
                    "SOURCE_AUTHORITY_HOLD", claim.source_ids[0]
                )
            roles = tuple(items[0] for items in source_roles)
            scope = "; ".join(sorted({item.purpose for item in roles}))
            decisions = tuple(
                SourceAuthorityAssessment.create(
                    source_id=source.unit.source_id,
                    governed_claim_id=claim.claim_id,
                    decision="ADMITTED",
                    authority_class="RESPONSIBLE_PRIMARY",
                    authority_scope=role.purpose,
                    evidence_digest=digest_bytes(
                        canonical_json_bytes(
                            {
                                "claim_digest": digest_bytes(claim.claim.encode()),
                                "source_definition_version_digest": (
                                    source.source_version.canonical_digest
                                ),
                                "role_assignments": [
                                    item.canonical_value()
                                    for item in source.source_version.request.roles
                                ],
                            }
                        )
                    ),
                )
                for source, role in zip(selected, roles, strict=True)
            )
            authority.extend(decisions)
            governed_claims.append(
                replace(
                    claim,
                    source_record_ids=tuple(
                        receipt_by_source[item] for item in claim.source_ids
                    ),
                    source_authority_decision_ids=tuple(
                        item.record_id for item in decisions
                    ),
                    rights_decision_ids=tuple(
                        source_by_id[item].rights.record_id
                        for item in claim.source_ids
                    ),
                    dependency_evidence_ids=tuple(
                        source_by_id[item].dependency.record_id
                        for item in claim.source_ids
                    ),
                    evidential_origin_ids=tuple(
                        source_by_id[item].dependency.evidential_origin_id
                        for item in claim.source_ids
                    ),
                    authority_class=ClaimAuthorityClass.RESPONSIBLE_PRIMARY,
                    authority_scope=scope,
                )
            )
        assessments = tuple(
            AcquiredSourceAssessment(
                source.unit.source_id,
                SourceCurrentness(
                    source.unit.source_id,
                    source.unit.authority.definition_id,
                    source.source_version.canonical_digest,
                    "CURRENT_VERSION",
                    result.publication_time,
                    result.retrieval_time,
                    None,
                    result.source_updated_time,
                    result.transport_evidence_digest,
                    result.transport_evidence_digest,
                    "PASS",
                    "CURRENT_CONTENT_API_VERSION_CONFIRMED",
                ),
                tuple((name, "PASS") for name in INTEGRITY),
            )
            for source, result in zip(sources, acquired, strict=True)
        )
        claims_by_id = {claim.claim_id: claim for claim in governed_claims}
        assessment_records = []
        for record in _objects(value.get("assessment_records")):
            if record.get("record_type") in {
                "SOURCE_RECORD",
                "SOURCE_AUTHORITY_DECISION",
                "RIGHTS_DECISION",
                "DEPENDENCY_EVIDENCE",
            }:
                raise NativeEvidenceHold(
                    "ASSESSOR_AUTHORITY_RECORD_HOLD", sources[0].unit.source_id
                )
            if "source_record_ids" in record:
                claim = claims_by_id.get(record.get("governed_claim_id"))
                if claim is None:
                    raise NativeEvidenceHold(
                        "ASSESSOR_CLAIM_BINDING_HOLD", sources[0].unit.source_id
                    )
                record = {
                    **record,
                    "source_record_ids": list(claim.source_record_ids),
                }
            assessment_records.append(record)
        return IndependentEvidenceAssessment(
            assessments,
            tuple(authority),
            package.substantive_new_information,
            tuple(governed_claims),
            package.qualification_evidence,
            tuple(assessment_records),
            package.selection_rationale,
            package.geography,
            package.categories,
            package.explicit_exclusions,
        )



def _document(text: str) -> dict[str, object]:
    def unique(pairs):
        value = dict(pairs)
        if len(value) != len(pairs):
            raise NativeEvidenceError("native assessment output has duplicate fields")
        return value

    try:
        value = json.loads(text, object_pairs_hook=unique)
    except (TypeError, json.JSONDecodeError) as exc:
        raise NativeEvidenceError("native assessment output is malformed") from exc
    if type(value) is not dict:
        raise NativeEvidenceError("native assessment output is malformed")
    if set(value) != set(SCHEMA["required"]):
        raise NativeEvidenceError("native assessment output fields differ")
    return value


def _objects(value: object) -> tuple[dict[str, object], ...]:
    if type(value) is not list or any(type(item) is not dict for item in value):
        raise NativeEvidenceError("native assessment records differ")
    return tuple(value)


def _dispatch_grok(prompt: str) -> NativeAssessmentExecution:
    execution = _run_grok_json(
        prompt,
        schema=SCHEMA,
        system_instruction=SYSTEM,
        temporary_prefix="newsroom-grok-evidence-assessor-",
    )
    return NativeAssessmentExecution(execution.text, execution.usage)
