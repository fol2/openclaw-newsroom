"""One accounted, fixed-route embedding of an exact native extraction passage.

The caller supplies the existing qualified invocation policy and signed-stop /
source-rights fence. No provider call is made while opening a runtime. No retry,
workspace Graphiti vector or fabricated provider receipt is used here.
"""

from __future__ import annotations

import json
import math
import ssl
import struct
import urllib.request
from collections.abc import Callable
from contextlib import AbstractContextManager
from datetime import UTC, datetime, timedelta
from pathlib import Path

from newsroom.authority import AuthenticationProof, GovernedObjects, ObjectAdmissionRequest
from newsroom.authority.canonical import canonical_json_bytes, digest_bytes, digest_canonical
from newsroom.graphiti_adapter.evaluation_packet import OPENROUTER_BASE_URL, OPENROUTER_EMBEDDING_SLUG
from newsroom.graphiti_adapter.embedding_meter import _usd_microunits
from newsroom.increment5.native_retrieval import (
    NATIVE_VECTOR_DIMENSIONS, NativeEmbeddingReceipt, NativeEmbeddingReference,
    NativeRetrievalHold, _vector,
)

from .govuk_evidence import _NoRedirect, _unique_object
from .model_usage import (
    InvocationAllocation, InvocationEfficiencyPolicy, InvocationTerminal,
    ModelUsageAdmissionError, ModelUsageService, UsageComponents, UsageStatus,
    WorkEnvelope, WorkloadClass,
)

VERSION = "hermes-native-passage-embedding-v1"
ROUTE = "NATIVE_RETRIEVAL_EMBEDDING"
URL = OPENROUTER_BASE_URL + "/embeddings"
TIMEOUT = 30
MAX_RESPONSE_BYTES = 1_048_576
SCHEMA_DIGEST = digest_canonical({
    "model": OPENROUTER_EMBEDDING_SLUG, "dimensions": NATIVE_VECTOR_DIMENSIONS,
    "response": "one-finite-float-vector-with-provider-id-and-usage",
})
MODEL_DIGEST = digest_canonical({
    "provider": "openrouter", "model": OPENROUTER_EMBEDDING_SLUG,
    "dimensions": NATIVE_VECTOR_DIMENSIONS, "encoding_format": "float",
    "stored_encoding": "big-endian-float32",
})


def implementation_digest() -> str:
    return digest_bytes(Path(__file__).read_bytes())


class NativePassageEmbedder:
    def __init__(
        self, *, api_key: str, objects: GovernedObjects, usage: ModelUsageService,
        policy: InvocationEfficiencyPolicy,
        dispatch_fence: Callable[[], AbstractContextManager],
        implementation_worktree_clean: bool,
        clock: Callable[[], datetime] = lambda: datetime.now(tz=UTC),
    ) -> None:
        if not api_key or not callable(dispatch_fence):
            raise ValueError("native embedding credential and dispatch fence are required")
        if (
            policy.workload_class is not WorkloadClass.NATIVE_RETRIEVAL_EMBEDDING
            or (policy.provider, policy.route, policy.model, policy.reasoning)
            != ("openrouter", ROUTE, OPENROUTER_EMBEDDING_SLUG, "none")
            or policy.output_schema_digest != SCHEMA_DIGEST
            or policy.prompt_contract_version != VERSION
            or policy.command_semantic_version != VERSION
            or policy.implementation_revision != implementation_digest()
            or implementation_worktree_clean is not True
            or not policy.qualified
        ):
            raise NativeRetrievalHold("NATIVE_EMBEDDING_POLICY_HOLD")
        self._key, self._objects, self._usage = api_key, objects, usage
        self._policy, self._fence, self._clock = policy, dispatch_fence, clock

    def retain(
        self, *, text: str, passage_id: str, cycle_id: str, proof: AuthenticationProof,
    ) -> NativeEmbeddingReference:
        if type(text) is not str or not text.strip() or not passage_id:
            raise NativeRetrievalHold("NATIVE_EMBEDDING_INPUT_HOLD")
        request = canonical_json_bytes({
            "input": text, "model": OPENROUTER_EMBEDDING_SLUG,
            "dimensions": NATIVE_VECTOR_DIMENSIONS, "encoding_format": "float",
        })
        policy, now = self._policy, self._clock()
        if len(request) > policy.max_prompt_bytes:
            raise NativeRetrievalHold("NATIVE_EMBEDDING_INPUT_BOUND")
        envelope = WorkEnvelope.create(
            cycle_id=cycle_id, workload_class=policy.workload_class, admitted_at=now,
            admission_decision_id=None, candidate_id=None, hypothesis_digest=None,
            evidence_package_digest=digest_bytes(text.encode()), ingest_id=passage_id,
            graphiti_attempt_id=None,
        )
        self._usage.open_envelope(envelope)
        manifest = self._manifest(request, text)
        self._usage.retain_context_manifest(manifest)
        allocation = InvocationAllocation.create(
            envelope_id=envelope.envelope_id, cycle_id=cycle_id, leaf_ordinal=1,
            workload_class=policy.workload_class, invocation_policy_digest=policy.canonical_digest,
            provider=policy.provider, route=policy.route, model=policy.model, reasoning="none",
            prompt_contract_version=VERSION, prompt_bytes=len(request),
            prompt_digest=digest_bytes(request), request_digest=manifest["request_digest"],
            output_schema_digest=SCHEMA_DIGEST, max_output_tokens=1,
            context_manifest_digest=manifest["context_manifest_digest"],
            context_identity=VERSION, config_identity=VERSION,
            one_turn=True, exact_input=True, skills_enabled=False, tools_enabled=False,
            mcp_enabled=False, prior_message_count=0, allocated_at=now,
            recovery_deadline_at=now + timedelta(seconds=TIMEOUT + 5), parent_invocation_id=None,
        )
        try:
            self._usage.allocate(allocation, owner_emergency_stop=False)
        except ModelUsageAdmissionError as exc:
            raise NativeRetrievalHold("NATIVE_EMBEDDING_ALLOCATION_HOLD") from exc
        dispatch_at = None
        telemetry = None
        vector = None
        error = None
        try:
            opener = urllib.request.build_opener(
                urllib.request.ProxyHandler({}), _NoRedirect(),
                urllib.request.HTTPSHandler(context=ssl.create_default_context()),
            )
            http = urllib.request.Request(URL, data=request, method="POST", headers={
                "Authorization": "Bearer " + self._key, "Content-Type": "application/json",
                "Accept": "application/json", "Accept-Encoding": "identity",
            })
            with self._fence():
                dispatch_at = self._clock()
                self._usage.observe_transport(
                    invocation_id=allocation.invocation_id, observed_at=dispatch_at,
                    state="DISPATCH_STARTED", evidence_digest=manifest["request_digest"],
                )
                with opener.open(http, timeout=TIMEOUT) as response:
                    raw = response.read(MAX_RESPONSE_BYTES + 1)
                    if response.status != 200 or response.geturl() != URL or len(raw) > MAX_RESPONSE_BYTES:
                        raise ValueError("native embedding response envelope differs")
            result = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object)
            telemetry = _telemetry(result)
            vector = _response_vector(result)
        except Exception as exc:
            # Preserve post-dispatch accounting even if vector validation fails.
            error = exc
        completed = self._clock()
        known = telemetry is not None and telemetry["total_tokens"] is not None
        zero = dispatch_at is None
        components = (
            UsageComponents(input_tokens=telemetry["prompt_tokens"], output_tokens=0,
                            total_tokens=telemetry["total_tokens"], provenance="PROVIDER_REPORTED")
            if known else UsageComponents(total_tokens=0, provenance="CLI_DERIVED")
            if zero else UsageComponents(provenance="UNAVAILABLE")
        )
        terminal = self._usage.complete(InvocationTerminal.create(
            invocation_id=allocation.invocation_id,
            outcome="NATIVE_EMBEDDING_COMPLETE" if error is None else "NATIVE_EMBEDDING_FAILED",
            failure_class=None if error is None else type(error).__name__,
            usage_status=UsageStatus.REPORTED if known or zero else UsageStatus.UNREPORTED,
            components=components, dispatch_at=dispatch_at, completed_at=completed,
            observed_at=completed, pre_dispatch_zero_proved=zero,
            provider_telemetry_digest=None if telemetry is None else digest_canonical(telemetry),
            od_011_reference="OD-011:NATIVE_RETRIEVAL_EMBEDDING",
            subscription_cli_chat_not_cash_debited=False,
        ), provider_telemetry=telemetry)
        if (error is not None or vector is None or telemetry is None
                or terminal.usage_status is not UsageStatus.REPORTED or terminal.policy_breach):
            raise NativeRetrievalHold("NATIVE_EMBEDDING_RESULT_HOLD") from error
        receipt = NativeEmbeddingReceipt(
            digest_bytes(text.encode()), digest_bytes(vector), NATIVE_VECTOR_DIMENSIONS,
            "openrouter", OPENROUTER_EMBEDDING_SLUG, MODEL_DIGEST,
            telemetry["provider_request_id"], terminal.terminal_digest,
            completed.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        )
        vector_admission = self._objects.admit(ObjectAdmissionRequest(
            "retrieval.native-vector", f"native-vector:{receipt.vector_digest}",
        ), vector, proof=proof).admission
        receipt_admission = self._objects.admit(ObjectAdmissionRequest(
            "retrieval.native-embedding-receipt", f"native-embedding:{digest_bytes(receipt.canonical_bytes)}",
        ), receipt.canonical_bytes, proof=proof).admission
        return NativeEmbeddingReference(vector_admission.admission_id, receipt_admission.admission_id)

    def _manifest(self, request: bytes, text: str) -> dict:
        policy = self._policy
        value = dict(
            schema_version=policy.context_manifest_schema_version,
            provider=policy.provider, route=policy.route, model=policy.model, reasoning="none",
            command_semantic_version=VERSION, command_flags=list(policy.command_flags),
            disabled_capabilities=list(policy.disabled_capabilities),
            implementation_revision=implementation_digest(), implementation_worktree_clean=True,
            prompt_contract_version=VERSION, prompt_bytes=len(request), prompt_digest=digest_bytes(request),
            schema_digest=SCHEMA_DIGEST, output_schema_digest=SCHEMA_DIGEST,
            system_digest=digest_bytes(b""), evidence_package_digest=digest_bytes(text.encode()),
            evidence_package_bytes=len(text.encode()), context_identity=VERSION, config_identity=VERSION,
            one_turn=True, exact_input=True, skills_enabled=False, tools_enabled=False,
            mcp_enabled=False, prior_message_count=0, skill_count=0, tool_count=0,
            mcp_server_count=0, mcp_tool_count=0,
        )
        value["request_digest"] = digest_canonical({key: value[key] for key in (
            "provider", "route", "model", "reasoning", "command_semantic_version", "command_flags",
            "implementation_revision", "system_digest", "prompt_digest", "output_schema_digest",
        )})
        return {**value, "context_manifest_digest": digest_canonical(value)}


def _telemetry(value: object) -> dict:
    if type(value) is not dict or type(value.get("usage")) is not dict:
        raise ValueError("embedding response usage is absent")
    usage = value["usage"]
    prompt, total = usage.get("prompt_tokens"), usage.get("total_tokens")
    if any(type(item) is not int or item < 0 for item in (prompt, total)):
        prompt = total = None
    return {"provider": "openrouter",
            "model": value.get("model") if type(value.get("model")) is str else None,
            "provider_request_id": value.get("id") if type(value.get("id")) is str else None,
            "prompt_tokens": prompt, "total_tokens": total,
            "cost_usd_microunits": _usd_microunits(usage.get("cost")),
            # Preserve exact numeric telemetry as JSON text, not unsupported
            # floating-point values in authority canonical JSON.
            "usage_json": json.dumps(usage, sort_keys=True, separators=(",", ":"), allow_nan=False)}


def _response_vector(value: dict) -> bytes:
    if (value.get("model") != OPENROUTER_EMBEDDING_SLUG or value.get("object") != "list"
            or type(value.get("id")) is not str or not value["id"]):
        raise ValueError("embedding provider identity differs")
    rows = value.get("data")
    if type(rows) is not list or len(rows) != 1 or rows[0].get("index") != 0:
        raise ValueError("embedding response count differs")
    values = rows[0].get("embedding")
    if (type(values) is not list or len(values) != NATIVE_VECTOR_DIMENSIONS
            or any(type(item) not in (int, float) or not math.isfinite(item) for item in values)):
        raise ValueError("embedding dimensions or values differ")
    result = struct.pack(f">{NATIVE_VECTOR_DIMENSIONS}f", *values)
    _vector(result)
    return result
