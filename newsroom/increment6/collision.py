"""Current authoritative collision eligibility and pre-effect enforcement.

The seam consumes the exact retained Increment 5 authority evidence.  It owns
no Candidate, migration, authority read or side effect: a downstream Candidate
operation receives a permit only after the evidence has been revalidated and
bound to one exact subject, request, generation, time and watermark.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
from typing import Protocol, TypeVar

from newsroom.authority.canonical import (
    CanonicalizationError,
    canonical_json_bytes,
    digest_bytes,
)
from newsroom.increment5.named_tool_authority_execution import (
    NamedAuthorityExecutionOutcome,
    NamedAuthorityExecutionReceipt,
    NamedToolAuthorityExecutionError,
)
from newsroom.increment5.named_tool_authority_receipt_validation import (
    NamedAuthorityReceiptValidationError,
    validate_named_authority_receipt,
)
from newsroom.increment5.named_tool_contracts import (
    CollisionHydrationLookupToolRequest,
    NamedToolId,
)


COLLISION_ELIGIBILITY_SCHEMA_VERSION = (
    "newsroom.increment6.candidate-collision-eligibility.v1"
)

_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:\-]{0,255}\Z")
_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_IDEMPOTENCY_PREFIX = "candidate-use:"


class CollisionEligibilityContractError(ValueError):
    """A collision binding, evidence record or decision is malformed."""


class CandidateUseOperation(StrEnum):
    ADMIT_NEW_CANDIDATE = "ADMIT_NEW_CANDIDATE"
    USE_CURRENT_CANDIDATE = "USE_CURRENT_CANDIDATE"


class CollisionState(StrEnum):
    UNKNOWN = "UNKNOWN"
    OCCUPIED = "OCCUPIED"
    UNOCCUPIED = "UNOCCUPIED"


class CollisionEligibilityOutcome(StrEnum):
    ELIGIBLE = "ELIGIBLE"
    COLLISION_CONFLICT = "COLLISION_CONFLICT"
    STALE = "STALE"
    INCOMPLETE = "INCOMPLETE"
    UNAVAILABLE = "UNAVAILABLE"
    POLICY_BLOCKED = "POLICY_BLOCKED"
    INTEGRITY_BLOCKED = "INTEGRITY_BLOCKED"
    BINDING_MISMATCH = "BINDING_MISMATCH"


class CollisionEligibilityReason(StrEnum):
    CURRENT_CANDIDATE_MATCH = "CURRENT_CANDIDATE_MATCH"
    CURRENT_SLOT_UNOCCUPIED = "CURRENT_SLOT_UNOCCUPIED"
    SLOT_ALREADY_OCCUPIED = "SLOT_ALREADY_OCCUPIED"
    EXPECTED_CANDIDATE_ABSENT = "EXPECTED_CANDIDATE_ABSENT"
    CURRENT_CANDIDATE_DIFFERS = "CURRENT_CANDIDATE_DIFFERS"
    AUTHORITY_STALE = "AUTHORITY_STALE"
    AUTHORITY_INCOMPLETE = "AUTHORITY_INCOMPLETE"
    AUTHORITY_UNAVAILABLE = "AUTHORITY_UNAVAILABLE"
    AUTHORITY_POLICY_BLOCKED = "AUTHORITY_POLICY_BLOCKED"
    EXECUTION_RECEIPT_INVALID = "EXECUTION_RECEIPT_INVALID"
    AUTHORITY_RECEIPT_INVALID = "AUTHORITY_RECEIPT_INVALID"
    AUTHORITY_RECEIPT_MISSING = "AUTHORITY_RECEIPT_MISSING"
    NAMED_REQUEST_BINDING_DIFFERS = "NAMED_REQUEST_BINDING_DIFFERS"
    GENERATION_BINDING_DIFFERS = "GENERATION_BINDING_DIFFERS"
    TIME_BINDING_DIFFERS = "TIME_BINDING_DIFFERS"
    WATERMARK_BINDING_DIFFERS = "WATERMARK_BINDING_DIFFERS"
    CURRENT_AUTHORITY_ADVANCED = "CURRENT_AUTHORITY_ADVANCED"
    CURRENT_SERVING_BOUNDARY_DIFFERS = "CURRENT_SERVING_BOUNDARY_DIFFERS"
    EXECUTION_PROVENANCE_DIFFERS = "EXECUTION_PROVENANCE_DIFFERS"
    AUTHORITY_IDENTITY_DIFFERS = "AUTHORITY_IDENTITY_DIFFERS"
    CURRENT_AUTHORITY_RECHECK_DIFFERS = "CURRENT_AUTHORITY_RECHECK_DIFFERS"


def _require_token(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _TOKEN_RE.fullmatch(value) is None:
        raise CollisionEligibilityContractError(
            f"{field} must be a bounded canonical token"
        )
    return value


def _require_digest(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        raise CollisionEligibilityContractError(
            f"{field} must be a canonical SHA-256 digest"
        )
    return value


def _parse_utc(value: object, *, field: str) -> datetime:
    if not isinstance(value, str):
        raise CollisionEligibilityContractError(f"{field} must be a UTC timestamp")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    except ValueError as exc:
        raise CollisionEligibilityContractError(
            f"{field} must be canonical second-resolution UTC"
        ) from exc
    if parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != value:
        raise CollisionEligibilityContractError(
            f"{field} must be canonical second-resolution UTC"
        )
    return parsed


def _strict_keys(
    value: Mapping[str, object], *, required: set[str], field: str
) -> None:
    if set(value) != required:
        raise CollisionEligibilityContractError(f"{field} keys are not exact")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for name, value in pairs:
        if name in result:
            raise CollisionEligibilityContractError(f"duplicate JSON key: {name}")
        result[name] = value
    return result


def _decode_canonical_object(raw: bytes, *, field: str) -> dict[str, object]:
    if not isinstance(raw, bytes) or not raw:
        raise CollisionEligibilityContractError(f"{field} bytes are required")
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CollisionEligibilityContractError(f"{field} is not JSON") from exc
    if not isinstance(value, dict):
        raise CollisionEligibilityContractError(f"{field} must be an object")
    try:
        canonical = canonical_json_bytes(value)
    except CanonicalizationError as exc:
        raise CollisionEligibilityContractError(
            f"{field} contains a non-canonical value"
        ) from exc
    if canonical != raw:
        raise CollisionEligibilityContractError(f"{field} is not canonical")
    return value


@dataclass(frozen=True, slots=True)
class CandidateUseCollisionBinding:
    """One exact Candidate-use subject and its required authority position."""

    subject_id: str
    subject_version_id: str
    subject_version_digest: str
    operation: CandidateUseOperation
    expected_candidate_id: str | None
    collision_namespace: str
    collision_key_digest: str
    generation_id: str
    query_valid_time: str
    serving_time: str
    authority_watermark: int

    def __post_init__(self) -> None:
        _require_token(self.subject_id, field="candidate_use_subject_id")
        _require_token(
            self.subject_version_id,
            field="candidate_use_subject_version_id",
        )
        _require_digest(
            self.subject_version_digest,
            field="candidate_use_subject_version_digest",
        )
        if not isinstance(self.operation, CandidateUseOperation):
            raise CollisionEligibilityContractError(
                "candidate-use operation must be typed"
            )
        if self.operation is CandidateUseOperation.ADMIT_NEW_CANDIDATE:
            if self.expected_candidate_id is not None:
                raise CollisionEligibilityContractError(
                    "new-Candidate admission cannot name an existing Candidate"
                )
        elif self.expected_candidate_id is None:
            raise CollisionEligibilityContractError(
                "current-Candidate use must name the expected Candidate"
            )
        else:
            _require_token(
                self.expected_candidate_id,
                field="expected_candidate_id",
            )
        _require_token(self.collision_namespace, field="collision_namespace")
        _require_digest(self.collision_key_digest, field="collision_key_digest")
        _require_token(self.generation_id, field="candidate_use_generation_id")
        query_valid = _parse_utc(
            self.query_valid_time,
            field="candidate_use_query_valid_time",
        )
        serving = _parse_utc(
            self.serving_time,
            field="candidate_use_serving_time",
        )
        if query_valid > serving:
            raise CollisionEligibilityContractError(
                "candidate-use query-valid time cannot follow serving time"
            )
        if (
            isinstance(self.authority_watermark, bool)
            or not isinstance(self.authority_watermark, int)
            or self.authority_watermark < 0
        ):
            raise CollisionEligibilityContractError(
                "candidate-use authority watermark must be non-negative"
            )

    def canonical_value(self) -> dict[str, object]:
        return {
            "subject_id": self.subject_id,
            "subject_version_id": self.subject_version_id,
            "subject_version_digest": self.subject_version_digest,
            "operation": self.operation.value,
            "expected_candidate_id": self.expected_candidate_id,
            "collision_namespace": self.collision_namespace,
            "collision_key_digest": self.collision_key_digest,
            "generation_id": self.generation_id,
            "query_valid_time": self.query_valid_time,
            "serving_time": self.serving_time,
            "authority_watermark": self.authority_watermark,
        }

    @property
    def binding_digest(self) -> str:
        return digest_bytes(
            canonical_json_bytes(
                {
                    "schema_version": COLLISION_ELIGIBILITY_SCHEMA_VERSION,
                    "record_type": "candidate_use_collision_binding",
                    "binding": self.canonical_value(),
                }
            )
        )

    @property
    def idempotency_key(self) -> str:
        return _IDEMPOTENCY_PREFIX + self.binding_digest

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, object],
    ) -> "CandidateUseCollisionBinding":
        _strict_keys(
            value,
            required={
                "subject_id",
                "subject_version_id",
                "subject_version_digest",
                "operation",
                "expected_candidate_id",
                "collision_namespace",
                "collision_key_digest",
                "generation_id",
                "query_valid_time",
                "serving_time",
                "authority_watermark",
            },
            field="candidate_use_collision_binding",
        )
        try:
            return cls(
                subject_id=value["subject_id"],  # type: ignore[arg-type]
                subject_version_id=value["subject_version_id"],  # type: ignore[arg-type]
                subject_version_digest=value["subject_version_digest"],  # type: ignore[arg-type]
                operation=CandidateUseOperation(value["operation"]),
                expected_candidate_id=value["expected_candidate_id"],  # type: ignore[arg-type]
                collision_namespace=value["collision_namespace"],  # type: ignore[arg-type]
                collision_key_digest=value["collision_key_digest"],  # type: ignore[arg-type]
                generation_id=value["generation_id"],  # type: ignore[arg-type]
                query_valid_time=value["query_valid_time"],  # type: ignore[arg-type]
                serving_time=value["serving_time"],  # type: ignore[arg-type]
                authority_watermark=value["authority_watermark"],  # type: ignore[arg-type]
            )
        except (TypeError, ValueError) as exc:
            raise CollisionEligibilityContractError(
                "candidate-use collision binding is malformed"
            ) from exc


@dataclass(frozen=True, slots=True)
class CurrentCollisionEligibilityRequest:
    binding: CandidateUseCollisionBinding
    named_request_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.binding, CandidateUseCollisionBinding):
            raise CollisionEligibilityContractError(
                "eligibility request binding must be typed"
            )
        _require_digest(
            self.named_request_digest,
            field="eligibility_named_request_digest",
        )

    def canonical_value(self) -> dict[str, object]:
        return {
            "binding": self.binding.canonical_value(),
            "named_request_digest": self.named_request_digest,
        }

    @property
    def request_digest(self) -> str:
        return digest_bytes(
            canonical_json_bytes(
                {
                    "schema_version": COLLISION_ELIGIBILITY_SCHEMA_VERSION,
                    "record_type": "current_collision_eligibility_request",
                    "request": self.canonical_value(),
                }
            )
        )

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, object],
    ) -> "CurrentCollisionEligibilityRequest":
        _strict_keys(
            value,
            required={"binding", "named_request_digest"},
            field="current_collision_eligibility_request",
        )
        raw_binding = value["binding"]
        if not isinstance(raw_binding, dict):
            raise CollisionEligibilityContractError(
                "eligibility request binding must be an object"
            )
        return cls(
            binding=CandidateUseCollisionBinding.from_mapping(raw_binding),
            named_request_digest=value["named_request_digest"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class CurrentCollisionReceiptEvidence:
    named_request: CollisionHydrationLookupToolRequest
    execution_receipt_bytes: bytes
    authority_receipt_bytes: bytes | None

    def __post_init__(self) -> None:
        if not isinstance(
            self.named_request,
            CollisionHydrationLookupToolRequest,
        ):
            raise CollisionEligibilityContractError(
                "current collision evidence requires the typed collision request"
            )
        if not isinstance(self.execution_receipt_bytes, bytes) or not (
            self.execution_receipt_bytes
        ):
            raise CollisionEligibilityContractError(
                "current collision execution receipt bytes are required"
            )
        if self.authority_receipt_bytes is not None and (
            not isinstance(self.authority_receipt_bytes, bytes)
            or not self.authority_receipt_bytes
        ):
            raise CollisionEligibilityContractError(
                "current collision authority receipt must be bytes or null"
            )

    @property
    def execution_receipt_digest(self) -> str:
        return digest_bytes(self.execution_receipt_bytes)

    @property
    def authority_receipt_digest(self) -> str | None:
        if self.authority_receipt_bytes is None:
            return None
        return digest_bytes(self.authority_receipt_bytes)


@dataclass(frozen=True, slots=True)
class NativeCurrentCollisionReceiptEvidence:
    """Receipts from the native Story Candidate authority reader."""

    request_digest: str
    execution_receipt_bytes: bytes
    authority_receipt_bytes: bytes

    def __post_init__(self) -> None:
        _require_digest(self.request_digest, field="native_collision_request_digest")
        for name in ("execution_receipt_bytes", "authority_receipt_bytes"):
            value = getattr(self, name)
            if not isinstance(value, bytes) or not value:
                raise CollisionEligibilityContractError(
                    f"native collision {name} are required"
                )

    @property
    def execution_receipt_digest(self) -> str:
        return digest_bytes(self.execution_receipt_bytes)

    @property
    def authority_receipt_digest(self) -> str:
        return digest_bytes(self.authority_receipt_bytes)


@dataclass(frozen=True, slots=True)
class TrustedCurrentCollisionAuthorityContext:
    """Trusted current authority position supplied by the enclosing boundary.

    This pure seam deliberately performs no ambient authority read.  The
    enclosing authoritative boundary must therefore supply the non-caller-
    selectable current position and the exact execution provenance it trusts.
    """

    generation_id: str
    authority_watermark: int
    query_valid_time: str
    serving_time: str
    authority_scope_id: str
    authority_profile_id: str
    adapter_config_digest: str
    authorization_receipt_digest: str
    authorization_decision_id: str
    port_registry_digest: str
    port_id: str

    def __post_init__(self) -> None:
        for name in (
            "generation_id",
            "authority_scope_id",
            "authority_profile_id",
            "authorization_decision_id",
            "port_id",
        ):
            _require_token(getattr(self, name), field=f"trusted_current_{name}")
        for name in (
            "adapter_config_digest",
            "authorization_receipt_digest",
            "port_registry_digest",
        ):
            _require_digest(getattr(self, name), field=f"trusted_current_{name}")
        query_valid = _parse_utc(
            self.query_valid_time,
            field="trusted_current_query_valid_time",
        )
        serving = _parse_utc(
            self.serving_time,
            field="trusted_current_serving_time",
        )
        if query_valid > serving:
            raise CollisionEligibilityContractError(
                "trusted current query-valid time cannot follow serving time"
            )
        if (
            isinstance(self.authority_watermark, bool)
            or not isinstance(self.authority_watermark, int)
            or self.authority_watermark < 0
        ):
            raise CollisionEligibilityContractError(
                "trusted current authority watermark must be non-negative"
            )

    def canonical_value(self) -> dict[str, object]:
        return {
            "generation_id": self.generation_id,
            "authority_watermark": self.authority_watermark,
            "query_valid_time": self.query_valid_time,
            "serving_time": self.serving_time,
            "authority_scope_id": self.authority_scope_id,
            "authority_profile_id": self.authority_profile_id,
            "adapter_config_digest": self.adapter_config_digest,
            "authorization_receipt_digest": self.authorization_receipt_digest,
            "authorization_decision_id": self.authorization_decision_id,
            "port_registry_digest": self.port_registry_digest,
            "port_id": self.port_id,
        }

    @property
    def context_digest(self) -> str:
        return digest_bytes(
            canonical_json_bytes(
                {
                    "schema_version": COLLISION_ELIGIBILITY_SCHEMA_VERSION,
                    "record_type": "trusted_current_collision_authority_context",
                    "context": self.canonical_value(),
                }
            )
        )

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, object],
    ) -> "TrustedCurrentCollisionAuthorityContext":
        required = {
            "generation_id",
            "authority_watermark",
            "query_valid_time",
            "serving_time",
            "authority_scope_id",
            "authority_profile_id",
            "adapter_config_digest",
            "authorization_receipt_digest",
            "authorization_decision_id",
            "port_registry_digest",
            "port_id",
        }
        _strict_keys(
            value,
            required=required,
            field="trusted_current_authority_context",
        )
        try:
            return cls(**value)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise CollisionEligibilityContractError(
                "trusted current authority context is malformed"
            ) from exc


@dataclass(frozen=True, slots=True)
class CurrentCollisionAuthoritySnapshot:
    """One live snapshot returned by the trusted authority provider."""

    evidence: CurrentCollisionReceiptEvidence | NativeCurrentCollisionReceiptEvidence
    trusted_context: TrustedCurrentCollisionAuthorityContext

    def __post_init__(self) -> None:
        if not isinstance(
            self.evidence,
            (CurrentCollisionReceiptEvidence, NativeCurrentCollisionReceiptEvidence),
        ):
            raise CollisionEligibilityContractError(
                "current authority snapshot evidence must be typed"
            )
        if not isinstance(
            self.trusted_context,
            TrustedCurrentCollisionAuthorityContext,
        ):
            raise CollisionEligibilityContractError(
                "current authority snapshot context must be typed"
            )


@dataclass(frozen=True, slots=True)
class TrustedCurrentCollisionAuthorityBoundary:
    """Authority identities fixed by the trusted composition root."""

    authority_scope_id: str
    authority_profile_id: str
    adapter_config_digest: str
    port_registry_digest: str
    port_id: str

    def __post_init__(self) -> None:
        for name in (
            "authority_scope_id",
            "authority_profile_id",
            "port_id",
        ):
            _require_token(getattr(self, name), field=f"trusted_boundary_{name}")
        for name in ("adapter_config_digest", "port_registry_digest"):
            _require_digest(getattr(self, name), field=f"trusted_boundary_{name}")

    def accepts(self, context: TrustedCurrentCollisionAuthorityContext) -> bool:
        if not isinstance(context, TrustedCurrentCollisionAuthorityContext):
            return False
        return (
            context.authority_scope_id == self.authority_scope_id
            and context.authority_profile_id == self.authority_profile_id
            and context.adapter_config_digest == self.adapter_config_digest
            and context.port_registry_digest == self.port_registry_digest
            and context.port_id == self.port_id
        )


class CurrentCollisionAuthorityProvider(Protocol):
    """Composition-root capability for a live current-authority read."""

    def __call__(
        self,
        request: CurrentCollisionEligibilityRequest,
    ) -> CurrentCollisionAuthoritySnapshot: ...


_ELIGIBLE_REASONS = frozenset(
    {
        CollisionEligibilityReason.CURRENT_CANDIDATE_MATCH,
        CollisionEligibilityReason.CURRENT_SLOT_UNOCCUPIED,
    }
)
_REASONS_BY_OUTCOME = {
    CollisionEligibilityOutcome.ELIGIBLE: _ELIGIBLE_REASONS,
    CollisionEligibilityOutcome.COLLISION_CONFLICT: frozenset(
        {
            CollisionEligibilityReason.SLOT_ALREADY_OCCUPIED,
            CollisionEligibilityReason.EXPECTED_CANDIDATE_ABSENT,
            CollisionEligibilityReason.CURRENT_CANDIDATE_DIFFERS,
        }
    ),
    CollisionEligibilityOutcome.STALE: frozenset(
        {
            CollisionEligibilityReason.AUTHORITY_STALE,
            CollisionEligibilityReason.CURRENT_AUTHORITY_ADVANCED,
            CollisionEligibilityReason.CURRENT_SERVING_BOUNDARY_DIFFERS,
            CollisionEligibilityReason.CURRENT_AUTHORITY_RECHECK_DIFFERS,
        }
    ),
    CollisionEligibilityOutcome.INCOMPLETE: frozenset(
        {CollisionEligibilityReason.AUTHORITY_INCOMPLETE}
    ),
    CollisionEligibilityOutcome.UNAVAILABLE: frozenset(
        {
            CollisionEligibilityReason.AUTHORITY_UNAVAILABLE,
            CollisionEligibilityReason.AUTHORITY_RECEIPT_MISSING,
        }
    ),
    CollisionEligibilityOutcome.POLICY_BLOCKED: frozenset(
        {CollisionEligibilityReason.AUTHORITY_POLICY_BLOCKED}
    ),
    CollisionEligibilityOutcome.INTEGRITY_BLOCKED: frozenset(
        {
            CollisionEligibilityReason.EXECUTION_RECEIPT_INVALID,
            CollisionEligibilityReason.AUTHORITY_RECEIPT_INVALID,
            CollisionEligibilityReason.EXECUTION_PROVENANCE_DIFFERS,
            CollisionEligibilityReason.AUTHORITY_IDENTITY_DIFFERS,
        }
    ),
    CollisionEligibilityOutcome.BINDING_MISMATCH: frozenset(
        {
            CollisionEligibilityReason.NAMED_REQUEST_BINDING_DIFFERS,
            CollisionEligibilityReason.GENERATION_BINDING_DIFFERS,
            CollisionEligibilityReason.TIME_BINDING_DIFFERS,
            CollisionEligibilityReason.WATERMARK_BINDING_DIFFERS,
        }
    ),
}


@dataclass(frozen=True, slots=True)
class CurrentCollisionEligibilityDecision:
    request: CurrentCollisionEligibilityRequest
    trusted_context: TrustedCurrentCollisionAuthorityContext
    outcome: CollisionEligibilityOutcome
    reason: CollisionEligibilityReason
    execution_receipt_digest: str
    authority_receipt_digest: str | None
    authority_execution_outcome: NamedAuthorityExecutionOutcome | None
    observed_authority_watermark: int | None
    collision_state: CollisionState | None
    observed_candidate_id: str | None
    authority_effect: str = "NONE"
    candidate_effect_performed: bool = False
    production_activation_authorised: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.request, CurrentCollisionEligibilityRequest):
            raise CollisionEligibilityContractError(
                "eligibility decision request must be typed"
            )
        if not isinstance(
            self.trusted_context,
            TrustedCurrentCollisionAuthorityContext,
        ):
            raise CollisionEligibilityContractError(
                "eligibility decision trusted context must be typed"
            )
        if not isinstance(self.outcome, CollisionEligibilityOutcome):
            raise CollisionEligibilityContractError(
                "eligibility outcome must be typed"
            )
        if not isinstance(self.reason, CollisionEligibilityReason):
            raise CollisionEligibilityContractError("eligibility reason must be typed")
        _require_digest(
            self.execution_receipt_digest,
            field="eligibility_execution_receipt_digest",
        )
        if self.authority_receipt_digest is not None:
            _require_digest(
                self.authority_receipt_digest,
                field="eligibility_authority_receipt_digest",
            )
        if self.authority_execution_outcome is not None and not isinstance(
            self.authority_execution_outcome,
            NamedAuthorityExecutionOutcome,
        ):
            raise CollisionEligibilityContractError(
                "authority execution outcome must be typed"
            )
        if self.observed_authority_watermark is not None and (
            isinstance(self.observed_authority_watermark, bool)
            or not isinstance(self.observed_authority_watermark, int)
            or self.observed_authority_watermark < 0
        ):
            raise CollisionEligibilityContractError(
                "observed authority watermark must be non-negative or null"
            )
        if self.collision_state is not None and not isinstance(
            self.collision_state,
            CollisionState,
        ):
            raise CollisionEligibilityContractError("collision state must be typed")
        if self.observed_candidate_id is not None:
            _require_token(
                self.observed_candidate_id,
                field="observed_candidate_id",
            )
            if self.collision_state is not CollisionState.OCCUPIED:
                raise CollisionEligibilityContractError(
                    "only occupied state can name an observed Candidate"
                )
        if self.reason not in _REASONS_BY_OUTCOME[self.outcome]:
            raise CollisionEligibilityContractError(
                "collision eligibility outcome and reason differ"
            )
        if self.outcome is CollisionEligibilityOutcome.ELIGIBLE:
            binding = self.request.binding
            if (
                self.authority_execution_outcome
                is not NamedAuthorityExecutionOutcome.COMPLETE
                or self.authority_receipt_digest is None
                or self.observed_authority_watermark
                != binding.authority_watermark
                or self.trusted_context.generation_id != binding.generation_id
                or self.trusted_context.authority_watermark
                != binding.authority_watermark
                or self.trusted_context.query_valid_time
                != binding.query_valid_time
                or self.trusted_context.serving_time != binding.serving_time
            ):
                raise CollisionEligibilityContractError(
                    "eligible decision lacks exact complete authority evidence"
                )
            if binding.operation is CandidateUseOperation.ADMIT_NEW_CANDIDATE:
                exact_operation_match = (
                    self.reason
                    is CollisionEligibilityReason.CURRENT_SLOT_UNOCCUPIED
                    and self.collision_state is CollisionState.UNOCCUPIED
                    and self.observed_candidate_id is None
                )
            else:
                exact_operation_match = (
                    self.reason
                    is CollisionEligibilityReason.CURRENT_CANDIDATE_MATCH
                    and self.collision_state is CollisionState.OCCUPIED
                    and self.observed_candidate_id == binding.expected_candidate_id
                )
            if not exact_operation_match:
                raise CollisionEligibilityContractError(
                    "eligible decision differs from its exact Candidate operation"
                )
        if (
            self.authority_effect != "NONE"
            or self.candidate_effect_performed is not False
            or self.production_activation_authorised is not False
        ):
            raise CollisionEligibilityContractError(
                "eligibility decision cannot claim authority or a Candidate effect"
            )

    @property
    def eligible(self) -> bool:
        return self.outcome is CollisionEligibilityOutcome.ELIGIBLE

    def canonical_value(self) -> dict[str, object]:
        return {
            "schema_version": COLLISION_ELIGIBILITY_SCHEMA_VERSION,
            "eligibility_request": self.request.canonical_value(),
            "eligibility_request_digest": self.request.request_digest,
            "trusted_current_authority_context": self.trusted_context.canonical_value(),
            "trusted_current_authority_context_digest": (
                self.trusted_context.context_digest
            ),
            "outcome": self.outcome.value,
            "reason": self.reason.value,
            "eligible": self.eligible,
            "execution_receipt_digest": self.execution_receipt_digest,
            "authority_receipt_digest": self.authority_receipt_digest,
            "authority_execution_outcome": (
                None
                if self.authority_execution_outcome is None
                else self.authority_execution_outcome.value
            ),
            "observed_authority_watermark": self.observed_authority_watermark,
            "collision_state": (
                None if self.collision_state is None else self.collision_state.value
            ),
            "observed_candidate_id": self.observed_candidate_id,
            "authority_effect": self.authority_effect,
            "candidate_effect_performed": self.candidate_effect_performed,
            "production_activation_authorised": (
                self.production_activation_authorised
            ),
        }

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.canonical_value())

    @property
    def decision_digest(self) -> str:
        return digest_bytes(self.canonical_bytes)

    @classmethod
    def from_canonical_bytes(
        cls,
        raw: bytes,
    ) -> "CurrentCollisionEligibilityDecision":
        value = _decode_canonical_object(raw, field="collision_eligibility_decision")
        _strict_keys(
            value,
            required={
                "schema_version",
                "eligibility_request",
                "eligibility_request_digest",
                "trusted_current_authority_context",
                "trusted_current_authority_context_digest",
                "outcome",
                "reason",
                "eligible",
                "execution_receipt_digest",
                "authority_receipt_digest",
                "authority_execution_outcome",
                "observed_authority_watermark",
                "collision_state",
                "observed_candidate_id",
                "authority_effect",
                "candidate_effect_performed",
                "production_activation_authorised",
            },
            field="collision_eligibility_decision",
        )
        if value["schema_version"] != COLLISION_ELIGIBILITY_SCHEMA_VERSION:
            raise CollisionEligibilityContractError(
                "collision eligibility decision schema is not accepted"
            )
        raw_request = value["eligibility_request"]
        if not isinstance(raw_request, dict):
            raise CollisionEligibilityContractError(
                "collision eligibility request must be an object"
            )
        request = CurrentCollisionEligibilityRequest.from_mapping(raw_request)
        if value["eligibility_request_digest"] != request.request_digest:
            raise CollisionEligibilityContractError(
                "collision eligibility request digest differs"
            )
        raw_context = value["trusted_current_authority_context"]
        if not isinstance(raw_context, dict):
            raise CollisionEligibilityContractError(
                "trusted current authority context must be an object"
            )
        trusted_context = TrustedCurrentCollisionAuthorityContext.from_mapping(
            raw_context
        )
        if value["trusted_current_authority_context_digest"] != (
            trusted_context.context_digest
        ):
            raise CollisionEligibilityContractError(
                "trusted current authority context digest differs"
            )
        raw_authority_outcome = value["authority_execution_outcome"]
        raw_state = value["collision_state"]
        try:
            decision = cls(
                request=request,
                trusted_context=trusted_context,
                outcome=CollisionEligibilityOutcome(value["outcome"]),
                reason=CollisionEligibilityReason(value["reason"]),
                execution_receipt_digest=value["execution_receipt_digest"],  # type: ignore[arg-type]
                authority_receipt_digest=value["authority_receipt_digest"],  # type: ignore[arg-type]
                authority_execution_outcome=(
                    None
                    if raw_authority_outcome is None
                    else NamedAuthorityExecutionOutcome(raw_authority_outcome)
                ),
                observed_authority_watermark=value["observed_authority_watermark"],  # type: ignore[arg-type]
                collision_state=(
                    None if raw_state is None else CollisionState(raw_state)
                ),
                observed_candidate_id=value["observed_candidate_id"],  # type: ignore[arg-type]
                authority_effect=value["authority_effect"],  # type: ignore[arg-type]
                candidate_effect_performed=value["candidate_effect_performed"],  # type: ignore[arg-type]
                production_activation_authorised=value[
                    "production_activation_authorised"
                ],  # type: ignore[arg-type]
            )
        except (TypeError, ValueError) as exc:
            raise CollisionEligibilityContractError(
                "collision eligibility decision is malformed"
            ) from exc
        if type(value["eligible"]) is not bool or value["eligible"] != (
            decision.eligible
        ):
            raise CollisionEligibilityContractError(
                "collision eligibility boolean differs from its outcome"
            )
        if decision.canonical_bytes != raw:
            raise CollisionEligibilityContractError(
                "collision eligibility decision is not canonical"
            )
        return decision


def _decision(
    *,
    request: CurrentCollisionEligibilityRequest,
    trusted_context: TrustedCurrentCollisionAuthorityContext,
    evidence: CurrentCollisionReceiptEvidence,
    outcome: CollisionEligibilityOutcome,
    reason: CollisionEligibilityReason,
    execution: NamedAuthorityExecutionReceipt | None = None,
    authority: Mapping[str, object] | None = None,
) -> CurrentCollisionEligibilityDecision:
    raw_state = None if authority is None else authority.get("collision_state")
    try:
        state = None if raw_state is None else CollisionState(raw_state)
    except (TypeError, ValueError):
        state = None
    raw_candidate = None if authority is None else authority.get("candidate_id")
    candidate_id = raw_candidate if isinstance(raw_candidate, str) else None
    attribution = None if execution is None else execution.authority_attribution
    return CurrentCollisionEligibilityDecision(
        request=request,
        trusted_context=trusted_context,
        outcome=outcome,
        reason=reason,
        execution_receipt_digest=evidence.execution_receipt_digest,
        authority_receipt_digest=evidence.authority_receipt_digest,
        authority_execution_outcome=(
            None if execution is None else execution.outcome
        ),
        observed_authority_watermark=(
            None if attribution is None else attribution.authority_watermark
        ),
        collision_state=state,
        observed_candidate_id=candidate_id,
    )


def _request_binding_reason(
    request: CurrentCollisionEligibilityRequest,
    evidence: CurrentCollisionReceiptEvidence,
) -> CollisionEligibilityReason | None:
    binding = request.binding
    named = evidence.named_request
    envelope = named.envelope
    if envelope.generation_id != binding.generation_id:
        return CollisionEligibilityReason.GENERATION_BINDING_DIFFERS
    if (
        envelope.query_valid_time != binding.query_valid_time
        or envelope.serving_time != binding.serving_time
    ):
        return CollisionEligibilityReason.TIME_BINDING_DIFFERS
    if (
        request.named_request_digest != named.request_digest
        or envelope.idempotency_key != binding.idempotency_key
        or named.collision_namespace != binding.collision_namespace
        or named.collision_key_digest != binding.collision_key_digest
        or envelope.tool_id
        is not NamedToolId.CURRENT_COLLISION_AND_AUTHORITY_HYDRATION_LOOKUP
    ):
        return CollisionEligibilityReason.NAMED_REQUEST_BINDING_DIFFERS
    return None


def decide_current_collision_eligibility(
    *,
    request: CurrentCollisionEligibilityRequest,
    evidence: CurrentCollisionReceiptEvidence,
    trusted_context: TrustedCurrentCollisionAuthorityContext,
) -> CurrentCollisionEligibilityDecision:
    """Revalidate exact retained authority evidence and decide fail-closed."""

    if not isinstance(request, CurrentCollisionEligibilityRequest):
        raise TypeError("collision eligibility request must be typed")
    if not isinstance(evidence, CurrentCollisionReceiptEvidence):
        raise TypeError("current collision evidence must be typed")
    if not isinstance(
        trusted_context,
        TrustedCurrentCollisionAuthorityContext,
    ):
        raise TypeError("trusted current authority context must be typed")

    try:
        execution = NamedAuthorityExecutionReceipt.from_canonical_bytes(
            evidence.execution_receipt_bytes
        )
        if execution.canonical_bytes != evidence.execution_receipt_bytes:
            raise CollisionEligibilityContractError(
                "execution receipt bytes are not canonical"
            )
    except (
        CollisionEligibilityContractError,
        NamedToolAuthorityExecutionError,
        TypeError,
        ValueError,
    ):
        return _decision(
            request=request,
            trusted_context=trusted_context,
            evidence=evidence,
            outcome=CollisionEligibilityOutcome.INTEGRITY_BLOCKED,
            reason=CollisionEligibilityReason.EXECUTION_RECEIPT_INVALID,
        )

    named = evidence.named_request
    if (
        execution.tool_id
        is not NamedToolId.CURRENT_COLLISION_AND_AUTHORITY_HYDRATION_LOOKUP
        or execution.tool_request_digest != named.request_digest
        or execution.tool_envelope_digest != named.envelope.envelope_digest
    ):
        return _decision(
            request=request,
            trusted_context=trusted_context,
            evidence=evidence,
            outcome=CollisionEligibilityOutcome.INTEGRITY_BLOCKED,
            reason=CollisionEligibilityReason.EXECUTION_RECEIPT_INVALID,
            execution=execution,
        )

    if (
        execution.authorization_receipt_digest
        != trusted_context.authorization_receipt_digest
        or execution.authorization_decision_id
        != trusted_context.authorization_decision_id
        or execution.port_registry_digest != trusted_context.port_registry_digest
        or execution.port_id != trusted_context.port_id
    ):
        return _decision(
            request=request,
            trusted_context=trusted_context,
            evidence=evidence,
            outcome=CollisionEligibilityOutcome.INTEGRITY_BLOCKED,
            reason=CollisionEligibilityReason.EXECUTION_PROVENANCE_DIFFERS,
            execution=execution,
        )

    raw = evidence.authority_receipt_bytes
    if raw is None or execution.authority_attribution is None:
        return _decision(
            request=request,
            trusted_context=trusted_context,
            evidence=evidence,
            outcome=CollisionEligibilityOutcome.UNAVAILABLE,
            reason=CollisionEligibilityReason.AUTHORITY_RECEIPT_MISSING,
            execution=execution,
        )

    try:
        validate_named_authority_receipt(
            request=named,
            execution_receipt=execution,
            raw_receipt_bytes=raw,
        )
        authority = _decode_canonical_object(raw, field="authority_receipt")
    except (
        CollisionEligibilityContractError,
        NamedAuthorityReceiptValidationError,
        TypeError,
        ValueError,
    ):
        return _decision(
            request=request,
            trusted_context=trusted_context,
            evidence=evidence,
            outcome=CollisionEligibilityOutcome.INTEGRITY_BLOCKED,
            reason=CollisionEligibilityReason.AUTHORITY_RECEIPT_INVALID,
            execution=execution,
        )

    attribution = execution.authority_attribution
    assert attribution is not None
    if (
        execution.outcome.value != attribution.outcome.value
        or execution.result_count != attribution.result_count
        or execution.no_match != attribution.no_match
    ):
        return _decision(
            request=request,
            trusted_context=trusted_context,
            evidence=evidence,
            outcome=CollisionEligibilityOutcome.INTEGRITY_BLOCKED,
            reason=CollisionEligibilityReason.EXECUTION_RECEIPT_INVALID,
            execution=execution,
            authority=authority,
        )

    if (
        authority.get("authority_scope_id") != trusted_context.authority_scope_id
        or attribution.authority_profile_id
        != trusted_context.authority_profile_id
        or authority.get("adapter_config_digest")
        != trusted_context.adapter_config_digest
    ):
        return _decision(
            request=request,
            trusted_context=trusted_context,
            evidence=evidence,
            outcome=CollisionEligibilityOutcome.INTEGRITY_BLOCKED,
            reason=CollisionEligibilityReason.AUTHORITY_IDENTITY_DIFFERS,
            execution=execution,
            authority=authority,
        )

    if (
        named.envelope.generation_id != trusted_context.generation_id
        or attribution.authority_watermark != trusted_context.authority_watermark
    ):
        return _decision(
            request=request,
            trusted_context=trusted_context,
            evidence=evidence,
            outcome=CollisionEligibilityOutcome.STALE,
            reason=CollisionEligibilityReason.CURRENT_AUTHORITY_ADVANCED,
            execution=execution,
            authority=authority,
        )
    if (
        named.envelope.query_valid_time != trusted_context.query_valid_time
        or named.envelope.serving_time != trusted_context.serving_time
    ):
        return _decision(
            request=request,
            trusted_context=trusted_context,
            evidence=evidence,
            outcome=CollisionEligibilityOutcome.STALE,
            reason=CollisionEligibilityReason.CURRENT_SERVING_BOUNDARY_DIFFERS,
            execution=execution,
            authority=authority,
        )

    if attribution.authority_watermark != request.binding.authority_watermark:
        return _decision(
            request=request,
            trusted_context=trusted_context,
            evidence=evidence,
            outcome=CollisionEligibilityOutcome.BINDING_MISMATCH,
            reason=CollisionEligibilityReason.WATERMARK_BINDING_DIFFERS,
            execution=execution,
            authority=authority,
        )
    binding_reason = _request_binding_reason(request, evidence)
    if binding_reason is not None:
        return _decision(
            request=request,
            trusted_context=trusted_context,
            evidence=evidence,
            outcome=CollisionEligibilityOutcome.BINDING_MISMATCH,
            reason=binding_reason,
            execution=execution,
            authority=authority,
        )

    if execution.outcome is not NamedAuthorityExecutionOutcome.COMPLETE:
        mapped = {
            NamedAuthorityExecutionOutcome.STALE: (
                CollisionEligibilityOutcome.STALE,
                CollisionEligibilityReason.AUTHORITY_STALE,
            ),
            NamedAuthorityExecutionOutcome.INCOMPLETE: (
                CollisionEligibilityOutcome.INCOMPLETE,
                CollisionEligibilityReason.AUTHORITY_INCOMPLETE,
            ),
            NamedAuthorityExecutionOutcome.UNAVAILABLE: (
                CollisionEligibilityOutcome.UNAVAILABLE,
                CollisionEligibilityReason.AUTHORITY_UNAVAILABLE,
            ),
            NamedAuthorityExecutionOutcome.POLICY_BLOCKED: (
                CollisionEligibilityOutcome.POLICY_BLOCKED,
                CollisionEligibilityReason.AUTHORITY_POLICY_BLOCKED,
            ),
        }.get(execution.outcome)
        if mapped is None:
            mapped = (
                CollisionEligibilityOutcome.INTEGRITY_BLOCKED,
                CollisionEligibilityReason.EXECUTION_RECEIPT_INVALID,
            )
        return _decision(
            request=request,
            trusted_context=trusted_context,
            evidence=evidence,
            outcome=mapped[0],
            reason=mapped[1],
            execution=execution,
            authority=authority,
        )

    binding = request.binding
    state = CollisionState(authority["collision_state"])
    candidate_id = authority["candidate_id"]
    if (
        binding.operation is CandidateUseOperation.ADMIT_NEW_CANDIDATE
        and state is CollisionState.UNOCCUPIED
        and candidate_id is None
    ):
        return _decision(
            request=request,
            trusted_context=trusted_context,
            evidence=evidence,
            outcome=CollisionEligibilityOutcome.ELIGIBLE,
            reason=CollisionEligibilityReason.CURRENT_SLOT_UNOCCUPIED,
            execution=execution,
            authority=authority,
        )
    if (
        binding.operation is CandidateUseOperation.USE_CURRENT_CANDIDATE
        and state is CollisionState.OCCUPIED
        and candidate_id == binding.expected_candidate_id
    ):
        return _decision(
            request=request,
            trusted_context=trusted_context,
            evidence=evidence,
            outcome=CollisionEligibilityOutcome.ELIGIBLE,
            reason=CollisionEligibilityReason.CURRENT_CANDIDATE_MATCH,
            execution=execution,
            authority=authority,
        )
    if binding.operation is CandidateUseOperation.ADMIT_NEW_CANDIDATE:
        reason = CollisionEligibilityReason.SLOT_ALREADY_OCCUPIED
    elif state is CollisionState.UNOCCUPIED:
        reason = CollisionEligibilityReason.EXPECTED_CANDIDATE_ABSENT
    else:
        reason = CollisionEligibilityReason.CURRENT_CANDIDATE_DIFFERS
    return _decision(
        request=request,
        trusted_context=trusted_context,
        evidence=evidence,
        outcome=CollisionEligibilityOutcome.COLLISION_CONFLICT,
        reason=reason,
        execution=execution,
        authority=authority,
    )


_NATIVE_EXECUTION_KEYS = {
    "schema_version", "request_digest", "authority_receipt_digest",
    "authorization_receipt_digest", "authorization_decision_id",
    "port_registry_digest", "port_id", "generation_id",
    "authority_watermark", "query_valid_time", "serving_time", "outcome",
}
_NATIVE_AUTHORITY_KEYS = {
    "schema_version", "request_digest", "authority_scope_id",
    "authority_profile_id", "adapter_config_digest", "generation_id",
    "authority_watermark", "query_valid_time", "serving_time",
    "collision_namespace", "collision_key_digest", "collision_state",
    "candidate_id", "candidate_version_id", "candidate_version_digest",
    "candidate_semantic_scope_digest", "subject_id", "subject_version_id",
    "subject_version_digest", "outcome",
}


def _native_integrity_decision(
    request: CurrentCollisionEligibilityRequest,
    evidence: NativeCurrentCollisionReceiptEvidence,
    context: TrustedCurrentCollisionAuthorityContext,
) -> CurrentCollisionEligibilityDecision:
    return CurrentCollisionEligibilityDecision(
        request, context, CollisionEligibilityOutcome.INTEGRITY_BLOCKED,
        CollisionEligibilityReason.AUTHORITY_RECEIPT_INVALID,
        evidence.execution_receipt_digest, evidence.authority_receipt_digest,
        None, None, None, None,
    )


def decide_native_current_collision_eligibility(
    *,
    request: CurrentCollisionEligibilityRequest,
    evidence: NativeCurrentCollisionReceiptEvidence,
    trusted_context: TrustedCurrentCollisionAuthorityContext,
) -> CurrentCollisionEligibilityDecision:
    """Validate one native live read without claiming the Increment 5 adapter."""

    if not isinstance(request, CurrentCollisionEligibilityRequest):
        raise TypeError("collision eligibility request must be typed")
    if not isinstance(evidence, NativeCurrentCollisionReceiptEvidence):
        raise TypeError("native collision evidence must be typed")
    if not isinstance(trusted_context, TrustedCurrentCollisionAuthorityContext):
        raise TypeError("trusted current authority context must be typed")
    try:
        execution = _decode_canonical_object(
            evidence.execution_receipt_bytes, field="native_collision_execution"
        )
        authority = _decode_canonical_object(
            evidence.authority_receipt_bytes, field="native_collision_authority"
        )
        _strict_keys(execution, required=_NATIVE_EXECUTION_KEYS,
                     field="native_collision_execution")
        _strict_keys(authority, required=_NATIVE_AUTHORITY_KEYS,
                     field="native_collision_authority")
        state = CollisionState(authority["collision_state"])
        candidate = authority["candidate_id"]
        candidate_position = (
            authority["candidate_version_id"],
            authority["candidate_version_digest"],
            authority["candidate_semantic_scope_digest"],
        )
        if state is CollisionState.OCCUPIED:
            _require_token(candidate, field="native_collision_candidate")
            _require_token(candidate_position[0], field="native_collision_version")
            _require_digest(candidate_position[1], field="native_collision_version")
            _require_digest(candidate_position[2], field="native_collision_scope")
        elif state is not CollisionState.UNOCCUPIED or candidate is not None or any(
            candidate_position
        ):
            raise CollisionEligibilityContractError(
                "native collision Candidate position is invalid"
            )
    except (CollisionEligibilityContractError, KeyError, TypeError, ValueError):
        return _native_integrity_decision(request, evidence, trusted_context)

    binding = request.binding
    expected_authority = {
        "schema_version": "newsroom.increment6.native-collision-authority.v1",
        "request_digest": evidence.request_digest,
        "authority_scope_id": trusted_context.authority_scope_id,
        "authority_profile_id": trusted_context.authority_profile_id,
        "adapter_config_digest": trusted_context.adapter_config_digest,
        "generation_id": trusted_context.generation_id,
        "authority_watermark": trusted_context.authority_watermark,
        "query_valid_time": trusted_context.query_valid_time,
        "serving_time": trusted_context.serving_time,
        "collision_namespace": binding.collision_namespace,
        "collision_key_digest": binding.collision_key_digest,
        "collision_state": state.value,
        "candidate_id": candidate,
        "candidate_version_id": candidate_position[0],
        "candidate_version_digest": candidate_position[1],
        "candidate_semantic_scope_digest": candidate_position[2],
        "subject_id": binding.subject_id,
        "subject_version_id": binding.subject_version_id,
        "subject_version_digest": binding.subject_version_digest,
        "outcome": "COMPLETE",
    }
    expected_execution = {
        "schema_version": "newsroom.increment6.native-collision-execution.v1",
        "request_digest": evidence.request_digest,
        "authority_receipt_digest": evidence.authority_receipt_digest,
        "authorization_receipt_digest": trusted_context.authorization_receipt_digest,
        "authorization_decision_id": trusted_context.authorization_decision_id,
        "port_registry_digest": trusted_context.port_registry_digest,
        "port_id": trusted_context.port_id,
        "generation_id": trusted_context.generation_id,
        "authority_watermark": trusted_context.authority_watermark,
        "query_valid_time": trusted_context.query_valid_time,
        "serving_time": trusted_context.serving_time,
        "outcome": "COMPLETE",
    }
    if (
        evidence.request_digest != request.named_request_digest
        or authority != expected_authority
        or execution != expected_execution
    ):
        return _native_integrity_decision(request, evidence, trusted_context)

    if trusted_context.generation_id != binding.generation_id:
        outcome, reason = (
            CollisionEligibilityOutcome.BINDING_MISMATCH,
            CollisionEligibilityReason.GENERATION_BINDING_DIFFERS,
        )
    elif trusted_context.authority_watermark != binding.authority_watermark:
        outcome, reason = (
            CollisionEligibilityOutcome.BINDING_MISMATCH,
            CollisionEligibilityReason.WATERMARK_BINDING_DIFFERS,
        )
    elif (
        trusted_context.query_valid_time != binding.query_valid_time
        or trusted_context.serving_time != binding.serving_time
    ):
        outcome, reason = (
            CollisionEligibilityOutcome.BINDING_MISMATCH,
            CollisionEligibilityReason.TIME_BINDING_DIFFERS,
        )
    elif (
        binding.operation is CandidateUseOperation.ADMIT_NEW_CANDIDATE
        and state is CollisionState.UNOCCUPIED
    ):
        outcome, reason = (
            CollisionEligibilityOutcome.ELIGIBLE,
            CollisionEligibilityReason.CURRENT_SLOT_UNOCCUPIED,
        )
    elif (
        binding.operation is CandidateUseOperation.USE_CURRENT_CANDIDATE
        and state is CollisionState.OCCUPIED
        and candidate == binding.expected_candidate_id
    ):
        outcome, reason = (
            CollisionEligibilityOutcome.ELIGIBLE,
            CollisionEligibilityReason.CURRENT_CANDIDATE_MATCH,
        )
    else:
        outcome = CollisionEligibilityOutcome.COLLISION_CONFLICT
        reason = (
            CollisionEligibilityReason.SLOT_ALREADY_OCCUPIED
            if binding.operation is CandidateUseOperation.ADMIT_NEW_CANDIDATE
            else CollisionEligibilityReason.EXPECTED_CANDIDATE_ABSENT
            if state is CollisionState.UNOCCUPIED
            else CollisionEligibilityReason.CURRENT_CANDIDATE_DIFFERS
        )
    return CurrentCollisionEligibilityDecision(
        request, trusted_context, outcome, reason,
        evidence.execution_receipt_digest, evidence.authority_receipt_digest,
        NamedAuthorityExecutionOutcome.COMPLETE,
        trusted_context.authority_watermark, state, candidate,
    )


class CurrentCollisionEligibilityBlocked(RuntimeError):
    """The pre-effect seam rejected one exact Candidate-use operation."""

    def __init__(self, decision: CurrentCollisionEligibilityDecision) -> None:
        self.decision = decision
        super().__init__(
            f"current collision eligibility blocked: {decision.outcome.value}/"
            f"{decision.reason.value}"
        )


_T = TypeVar("_T")


class CurrentCollisionEffectEnforcer:
    """Pre-effect capability assembled only at the trusted composition root.

    Candidate-use callers receive this already-bound capability.  Its public
    operation accepts only their request and effect; retained evidence and the
    current authority context are read live from the pre-bound provider.  A
    second read immediately before the effect is the commit-time recheck.
    """

    def __init__(
        self,
        *,
        current_authority_provider: CurrentCollisionAuthorityProvider,
        trusted_boundary: TrustedCurrentCollisionAuthorityBoundary,
    ) -> None:
        if not callable(current_authority_provider):
            raise TypeError("current authority provider must be callable")
        if not isinstance(
            trusted_boundary,
            TrustedCurrentCollisionAuthorityBoundary,
        ):
            raise TypeError("trusted current authority boundary must be typed")
        self._current_authority_provider = current_authority_provider
        self._trusted_boundary = trusted_boundary

    def _current_decision(
        self,
        request: CurrentCollisionEligibilityRequest,
    ) -> CurrentCollisionEligibilityDecision:
        snapshot = self._current_authority_provider(request)
        if not isinstance(snapshot, CurrentCollisionAuthoritySnapshot):
            raise TypeError(
                "current authority provider must return a typed snapshot"
            )
        if isinstance(snapshot.evidence, NativeCurrentCollisionReceiptEvidence):
            decision = decide_native_current_collision_eligibility(
                request=request, evidence=snapshot.evidence,
                trusted_context=snapshot.trusted_context,
            )
        else:
            decision = decide_current_collision_eligibility(
                request=request, evidence=snapshot.evidence,
                trusted_context=snapshot.trusted_context,
            )
        if not self._trusted_boundary.accepts(snapshot.trusted_context):
            return replace(
                decision,
                outcome=CollisionEligibilityOutcome.INTEGRITY_BLOCKED,
                reason=CollisionEligibilityReason.AUTHORITY_IDENTITY_DIFFERS,
            )
        return decision

    def enforce(
        self,
        *,
        request: CurrentCollisionEligibilityRequest,
        effect: Callable[[CurrentCollisionEligibilityDecision], _T],
    ) -> _T:
        """Invoke ``effect`` only after two identical live decisions."""

        if not isinstance(request, CurrentCollisionEligibilityRequest):
            raise TypeError("collision eligibility request must be typed")
        if not callable(effect):
            raise TypeError("candidate-use effect must be callable")
        initial = self._current_decision(request)
        if not initial.eligible:
            raise CurrentCollisionEligibilityBlocked(initial)
        rechecked = self._current_decision(request)
        if not rechecked.eligible:
            raise CurrentCollisionEligibilityBlocked(rechecked)
        if rechecked.decision_digest != initial.decision_digest:
            raise CurrentCollisionEligibilityBlocked(
                replace(
                    rechecked,
                    outcome=CollisionEligibilityOutcome.STALE,
                    reason=(
                        CollisionEligibilityReason.CURRENT_AUTHORITY_RECHECK_DIFFERS
                    ),
                )
            )
        return effect(rechecked)


__all__ = [
    "COLLISION_ELIGIBILITY_SCHEMA_VERSION",
    "CandidateUseCollisionBinding",
    "CandidateUseOperation",
    "CollisionEligibilityContractError",
    "CollisionEligibilityOutcome",
    "CollisionEligibilityReason",
    "CollisionState",
    "CurrentCollisionAuthorityProvider",
    "CurrentCollisionAuthoritySnapshot",
    "CurrentCollisionEffectEnforcer",
    "CurrentCollisionEligibilityBlocked",
    "CurrentCollisionEligibilityDecision",
    "CurrentCollisionEligibilityRequest",
    "CurrentCollisionReceiptEvidence",
    "NativeCurrentCollisionReceiptEvidence",
    "TrustedCurrentCollisionAuthorityContext",
    "TrustedCurrentCollisionAuthorityBoundary",
    "decide_current_collision_eligibility",
    "decide_native_current_collision_eligibility",
]
