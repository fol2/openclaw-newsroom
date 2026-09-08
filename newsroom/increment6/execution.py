"""Pure Increment 6B1 execution identity and ownership contract values.

These immutable phase-one values grant no grouping, dispatch, lease, worker,
Proposal, Candidate, publication, evidence, egress, or external-effect authority.
A future v20 trusted store and composition root must atomically validate current
Work Item authority before persisting any claim or attempt.
"""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Self

from newsroom.authority.canonical import (
    MAX_SAFE_INTEGER,
    canonical_json_bytes,
    digest_bytes,
)
from newsroom.increment6.outcomes import (
    ContractAuthority,
    ContractEffect,
    PrioritySelection,
)
from newsroom.increment6.proposals import WorkerAttemptBinding, WorkerKind
from newsroom.increment6.scheduling import (
    CapacityAllocationDisposition,
    CapacityItemAllocation,
    ReservedCapacityDecision,
)
from newsroom.increment6.work_items import (
    TriageWorkItemVersion,
)

EXECUTION_BATCH = "EXACT_NO_AUTHORITY_EXECUTION_BATCH"
WORKER_ATTEMPT = "DETERMINISTIC_NO_AUTHORITY_WORKER_ATTEMPT"
WORK_ITEM_LEASE_OWNERSHIP = "CAPABILITY_OWNERSHIP_CLAIM_ONLY"

EXECUTION_BATCH_SCHEMA = "newsroom.increment6.execution-batch.v1"
WORKER_ATTEMPT_SCHEMA = "newsroom.increment6.worker-attempt.v1"
WORK_ITEM_LEASE_SCHEMA = "newsroom.increment6.work-item-lease.v1"

_MAX_BATCH_MEMBERS = 48
# Members retain only exact digests and identities.  They never duplicate the
# potentially 31 MiB Work Item Version or the scheduling decision bytes.
_MAX_COMPACT_MEMBER_BYTES = 768
_MAX_CANONICAL_OVERHEAD_BYTES = 32_768
_MAX_CANONICAL_BYTES = (
    _MAX_BATCH_MEMBERS * _MAX_COMPACT_MEMBER_BYTES + _MAX_CANONICAL_OVERHEAD_BYTES
)
_MAX_CANONICAL_DEPTH = 24
_MAX_CANONICAL_NODES = _MAX_CANONICAL_BYTES // 2 + 1
_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:\-]{0,255}\Z")
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")


class ExecutionContractError(ValueError):
    """An execution contract value is malformed or exceeds its envelope."""


def _normalise[T](field: str, operation: Callable[[], T]) -> T:
    try:
        return operation()
    except ExecutionContractError:
        raise
    except Exception as exc:
        raise ExecutionContractError(f"{field} is invalid") from exc


class LeaseLifecycle(StrEnum):
    PENDING = "PENDING"
    CLAIMED = "CLAIMED"
    RELEASED = "RELEASED"
    EXPIRED = "EXPIRED"


class LeaseProgress(StrEnum):
    NOT_STARTED = "NOT_STARTED"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    INTERRUPTED = "INTERRUPTED"


def _token(value: object, field: str) -> str:
    if type(value) is not str or _TOKEN.fullmatch(value) is None:
        raise ExecutionContractError(f"{field} must be a bounded token")
    return value


def _string(value: object, field: str) -> str:
    if type(value) is not str:
        raise ExecutionContractError(f"{field} must be a string")
    return value


def _uuid(value: object, field: str) -> str:
    if type(value) is not str:
        raise ExecutionContractError(f"{field} must be a canonical UUID")
    try:
        parsed = uuid.UUID(value)
    except (TypeError, ValueError, AttributeError) as exc:
        raise ExecutionContractError(f"{field} must be a canonical UUID") from exc
    if str(parsed) != value:
        raise ExecutionContractError(f"{field} must be a canonical UUID")
    return value


def _digest(value: object, field: str) -> str:
    if type(value) is not str or _DIGEST.fullmatch(value) is None:
        raise ExecutionContractError(f"{field} must be a canonical SHA-256 digest")
    return value


def _integer(value: object, field: str, *, minimum: int = 1) -> int:
    if type(value) is not int or not minimum <= value <= MAX_SAFE_INTEGER:
        raise ExecutionContractError(
            f"{field} must be an exact interoperable bounded integer"
        )
    return value


def _utc(value: object, field: str) -> str:
    if type(value) is not str or not value.endswith("Z"):
        raise ExecutionContractError(f"{field} must be canonical UTC")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ExecutionContractError(f"{field} must be canonical UTC") from exc
    if parsed.tzinfo != UTC or parsed.isoformat().replace("+00:00", "Z") != value:
        raise ExecutionContractError(f"{field} must be canonical UTC")
    return value


def _utc_value(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ExecutionContractError("UTC value is malformed") from exc


def _exact(value: object, fields: set[str], name: str) -> dict[str, object]:
    if type(value) is not dict or set(value) != fields:
        raise ExecutionContractError(f"{name} fields differ")
    return value


def _decode(raw: bytes) -> dict[str, object]:
    if type(raw) is not bytes or not raw or len(raw) > _MAX_CANONICAL_BYTES:
        raise ExecutionContractError("canonical input exceeds the execution envelope")

    def unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ExecutionContractError(f"duplicate object name: {key}")
            result[key] = value
        return result

    def bounded_integer(text: str) -> int:
        if len(text.lstrip("-")) > 16:
            raise ExecutionContractError("canonical integer exceeds safe range")
        value = int(text)
        if not -MAX_SAFE_INTEGER <= value <= MAX_SAFE_INTEGER:
            raise ExecutionContractError("canonical integer exceeds safe range")
        return value

    def reject_float(_: str) -> float:
        raise ExecutionContractError("floating-point values are unsupported")

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=unique,
            parse_int=bounded_integer,
            parse_float=reject_float,
            parse_constant=reject_float,
        )
    except ExecutionContractError:
        raise
    except (
        UnicodeError,
        json.JSONDecodeError,
        ValueError,
        RecursionError,
        MemoryError,
    ) as exc:
        raise ExecutionContractError("canonical input is invalid UTF-8 JSON") from exc
    pending: list[tuple[object, int]] = [(value, 1)]
    nodes = 0
    while pending:
        item, depth = pending.pop()
        nodes += 1
        if depth > _MAX_CANONICAL_DEPTH or nodes > _MAX_CANONICAL_NODES:
            raise ExecutionContractError("canonical input exceeds structural bounds")
        if isinstance(item, dict):
            pending.extend((child, depth + 1) for child in item.values())
        elif isinstance(item, list):
            pending.extend((child, depth + 1) for child in item)
        elif not isinstance(item, (str, int, float, bool, type(None))):
            raise ExecutionContractError(
                "canonical input contains an unsupported value"
            )
    if not isinstance(value, dict):
        raise ExecutionContractError("canonical input must be an object")
    try:
        if canonical_json_bytes(value) != raw:
            raise ExecutionContractError("canonical input bytes differ")
    except ExecutionContractError:
        raise
    except Exception as exc:
        raise ExecutionContractError("canonical input cannot be normalised") from exc
    return value


def _canonical(value: object, field: str) -> bytes:
    try:
        result = canonical_json_bytes(value)
    except Exception as exc:
        raise ExecutionContractError(
            f"{field} is not canonically representable"
        ) from exc
    if len(result) > _MAX_CANONICAL_BYTES:
        raise ExecutionContractError(f"{field} exceeds the execution envelope")
    return result


@dataclass(frozen=True, slots=True)
class ExecutionBatchMember:
    work_item_id: str
    work_item_version_id: str
    work_item_version_digest: str
    retrieval_context_id: str
    retrieval_context_digest: str
    priority_digest: str
    scheduling_grant_digest: str
    producer_binding_digest: str

    @staticmethod
    def _binding_digest(
        work_item_id: str,
        work_item_version_id: str,
        work_item_version_digest: str,
        retrieval_context_id: str,
        retrieval_context_digest: str,
        priority_digest: str,
        scheduling_grant_digest: str,
    ) -> str:
        return digest_bytes(
            _canonical(
                {
                    "work_item_id": work_item_id,
                    "work_item_version_id": work_item_version_id,
                    "work_item_version_digest": work_item_version_digest,
                    "retrieval_context_id": retrieval_context_id,
                    "retrieval_context_digest": retrieval_context_digest,
                    "priority_digest": priority_digest,
                    "scheduling_grant_digest": scheduling_grant_digest,
                },
                "batch member producer binding",
            )
        )

    @classmethod
    def from_producers(
        cls,
        version: TriageWorkItemVersion,
        allocation: CapacityItemAllocation,
    ) -> Self:
        if (
            type(version) is not TriageWorkItemVersion
            or type(allocation) is not CapacityItemAllocation
        ):
            raise ExecutionContractError(
                "batch member producers must be exact typed values"
            )
        try:
            version_digest = version.canonical_digest
            priority = PrioritySelection.from_canonical_bytes(
                version.priority.selection_bytes
            )
            priority_digest = version.priority.selection_digest
            grant_digest = digest_bytes(
                _canonical(allocation.canonical_value(), "capacity grant")
            )
            retrieval = version.retrieval
            retrieval_id = retrieval.context_id or retrieval.request_id
            retrieval_digest = retrieval.context_digest or retrieval.request_digest
            if (
                allocation.disposition is not CapacityAllocationDisposition.GRANTED
                or allocation.item.work_item_id != version.work_item_id
                or allocation.item.work_item_version_id != version.version_id
                or allocation.item.work_item_version_digest != version_digest
                or allocation.item.priority_selection != priority
            ):
                raise ExecutionContractError(
                    "batch member differs from its exact scheduling grant"
                )
            values = (
                version.work_item_id,
                version.version_id,
                version_digest,
                retrieval_id,
                retrieval_digest,
                priority_digest,
                grant_digest,
            )
            return cls(*values, cls._binding_digest(*values))
        except ExecutionContractError:
            raise
        except Exception as exc:
            raise ExecutionContractError(
                "batch member producer binding is invalid"
            ) from exc

    def __post_init__(self) -> None:
        _uuid(self.work_item_id, "member work_item_id")
        _uuid(self.work_item_version_id, "member work_item_version_id")
        _digest(self.work_item_version_digest, "member Work Item Version digest")
        _uuid(self.retrieval_context_id, "member retrieval_context_id")
        _digest(self.retrieval_context_digest, "member retrieval digest")
        _digest(self.priority_digest, "member priority digest")
        _digest(self.scheduling_grant_digest, "member scheduling grant digest")
        _digest(self.producer_binding_digest, "member producer binding digest")
        expected = self._binding_digest(
            self.work_item_id,
            self.work_item_version_id,
            self.work_item_version_digest,
            self.retrieval_context_id,
            self.retrieval_context_digest,
            self.priority_digest,
            self.scheduling_grant_digest,
        )
        if self.producer_binding_digest != expected:
            raise ExecutionContractError("batch member producer binding differs")

    def canonical_value(self) -> dict[str, object]:
        return {
            "work_item_id": self.work_item_id,
            "work_item_version_id": self.work_item_version_id,
            "work_item_version_digest": self.work_item_version_digest,
            "retrieval_context_id": self.retrieval_context_id,
            "retrieval_context_digest": self.retrieval_context_digest,
            "priority_digest": self.priority_digest,
            "scheduling_grant_digest": self.scheduling_grant_digest,
            "producer_binding_digest": self.producer_binding_digest,
        }

    @classmethod
    def from_value(cls, value: object) -> Self:
        item = _exact(
            value,
            {
                "work_item_id",
                "work_item_version_id",
                "work_item_version_digest",
                "retrieval_context_id",
                "retrieval_context_digest",
                "priority_digest",
                "scheduling_grant_digest",
                "producer_binding_digest",
            },
            "batch member",
        )
        return cls(
            _string(item["work_item_id"], "member work_item_id"),
            _string(item["work_item_version_id"], "member version_id"),
            _string(item["work_item_version_digest"], "member version digest"),
            _string(item["retrieval_context_id"], "member retrieval id"),
            _string(item["retrieval_context_digest"], "member retrieval digest"),
            _string(item["priority_digest"], "member priority digest"),
            _string(item["scheduling_grant_digest"], "member grant digest"),
            _string(item["producer_binding_digest"], "member producer binding digest"),
        )


@dataclass(frozen=True, slots=True)
class ExecutionBatch:
    batch_id: str
    members: tuple[ExecutionBatchMember, ...]
    scheduling_decision_digest: str
    scheduling_policy_digest: str
    schema_identity: str = EXECUTION_BATCH_SCHEMA
    authority: ContractAuthority = ContractAuthority.NONE
    effect: ContractEffect = ContractEffect.NONE

    @classmethod
    def create(
        cls,
        *,
        scheduling_decision: ReservedCapacityDecision,
        work_item_versions: tuple[TriageWorkItemVersion, ...],
    ) -> Self:
        if type(scheduling_decision) is not ReservedCapacityDecision:
            raise ExecutionContractError(
                "Execution Batch scheduling decision must be exact typed"
            )
        if (
            type(work_item_versions) is not tuple
            or not 1 <= len(work_item_versions) <= _MAX_BATCH_MEMBERS
            or any(
                type(version) is not TriageWorkItemVersion
                for version in work_item_versions
            )
        ):
            raise ExecutionContractError(
                "Execution Batch Versions exceed the bounded typed envelope"
            )
        try:
            versions = {
                (version.work_item_id, version.version_id): version
                for version in work_item_versions
            }
            granted = tuple(
                allocation
                for allocation in scheduling_decision.allocations
                if allocation.disposition is CapacityAllocationDisposition.GRANTED
            )
            members = tuple(
                ExecutionBatchMember.from_producers(
                    versions[
                        (
                            allocation.item.work_item_id,
                            allocation.item.work_item_version_id,
                        )
                    ],
                    allocation,
                )
                for allocation in granted
            )
            if len(versions) != len(work_item_versions) or len(versions) != len(
                members
            ):
                raise ExecutionContractError(
                    "Execution Batch Versions differ from exact scheduling grants"
                )
            return cls._from_bindings(
                members,
                scheduling_decision.decision_digest,
                scheduling_decision.policy.policy_digest,
            )
        except KeyError as exc:
            raise ExecutionContractError(
                "Execution Batch lacks an exact granted Work Item Version"
            ) from exc
        except ExecutionContractError:
            raise
        except Exception as exc:
            raise ExecutionContractError(
                "Execution Batch producer binding is invalid"
            ) from exc

    @classmethod
    def _from_bindings(
        cls,
        members: tuple[ExecutionBatchMember, ...],
        scheduling_decision_digest: str,
        scheduling_policy_digest: str,
    ) -> Self:
        if (
            type(members) is not tuple
            or not 1 <= len(members) <= _MAX_BATCH_MEMBERS
            or any(type(member) is not ExecutionBatchMember for member in members)
        ):
            raise ExecutionContractError(
                "Execution Batch members exceed the bounded typed envelope"
            )
        members = tuple(
            _normalise(
                "Execution Batch member",
                lambda member=member: ExecutionBatchMember.from_value(
                    member.canonical_value()
                ),
            )
            for member in members
        )
        ordered = tuple(sorted(members, key=lambda member: member.work_item_id))
        identity = digest_bytes(
            _canonical(
                {
                    "scheduling_decision_digest": scheduling_decision_digest,
                    "scheduling_policy_digest": scheduling_policy_digest,
                    "members": [member.canonical_value() for member in ordered],
                },
                "batch identity",
            )
        )
        return cls(
            str(uuid.uuid5(uuid.NAMESPACE_URL, f"{EXECUTION_BATCH_SCHEMA}|{identity}")),
            ordered,
            scheduling_decision_digest,
            scheduling_policy_digest,
        )

    def __post_init__(self) -> None:
        _normalise("Execution Batch", self._validate)

    def _validate(self) -> None:
        _uuid(self.batch_id, "batch_id")
        _digest(self.scheduling_decision_digest, "scheduling decision digest")
        _digest(self.scheduling_policy_digest, "scheduling policy digest")
        if (
            type(self.schema_identity) is not str
            or self.schema_identity != EXECUTION_BATCH_SCHEMA
            or type(self.authority) is not ContractAuthority
            or self.authority is not ContractAuthority.NONE
            or type(self.effect) is not ContractEffect
            or self.effect is not ContractEffect.NONE
        ):
            raise ExecutionContractError("Execution Batch claims authority or effect")
        if (
            type(self.members) is not tuple
            or not 1 <= len(self.members) <= _MAX_BATCH_MEMBERS
            or any(type(member) is not ExecutionBatchMember for member in self.members)
        ):
            raise ExecutionContractError(
                "Execution Batch members exceed the bounded typed envelope"
            )
        if self.members != tuple(
            sorted(self.members, key=lambda member: member.work_item_id)
        ):
            raise ExecutionContractError("Execution Batch members must be sorted")
        if len({member.work_item_id for member in self.members}) != len(self.members):
            raise ExecutionContractError(
                "Execution Batch members must be unique per Work Item"
            )
        identity = digest_bytes(
            _canonical(
                {
                    "scheduling_decision_digest": self.scheduling_decision_digest,
                    "scheduling_policy_digest": self.scheduling_policy_digest,
                    "members": [member.canonical_value() for member in self.members],
                },
                "batch identity",
            )
        )
        expected = str(
            uuid.uuid5(uuid.NAMESPACE_URL, f"{EXECUTION_BATCH_SCHEMA}|{identity}")
        )
        if self.batch_id != expected:
            raise ExecutionContractError("Execution Batch identity differs")
        _ = self.canonical_bytes

    def canonical_value(self) -> dict[str, object]:
        return {
            "schema_identity": self.schema_identity,
            "authority": self.authority.value,
            "effect": self.effect.value,
            "batch_id": self.batch_id,
            "scheduling_decision_digest": self.scheduling_decision_digest,
            "scheduling_policy_digest": self.scheduling_policy_digest,
            "members": [member.canonical_value() for member in self.members],
        }

    @property
    def canonical_bytes(self) -> bytes:
        return _normalise(
            "Execution Batch",
            lambda: _canonical(self.canonical_value(), "Execution Batch"),
        )

    @property
    def canonical_digest(self) -> str:
        return _normalise(
            "Execution Batch digest", lambda: digest_bytes(self.canonical_bytes)
        )

    @classmethod
    def from_canonical_bytes(cls, raw: bytes) -> Self:
        item = _exact(
            _decode(raw),
            {
                "schema_identity",
                "authority",
                "effect",
                "batch_id",
                "members",
                "scheduling_decision_digest",
                "scheduling_policy_digest",
            },
            "Execution Batch",
        )
        if type(item["members"]) is not list:
            raise ExecutionContractError("Execution Batch members must be an array")
        try:
            authority = ContractAuthority(item["authority"])
            effect = ContractEffect(item["effect"])
            value = cls(
                _string(item["batch_id"], "batch_id"),
                tuple(
                    ExecutionBatchMember.from_value(member)
                    for member in item["members"]
                ),
                _string(
                    item["scheduling_decision_digest"], "scheduling decision digest"
                ),
                _string(item["scheduling_policy_digest"], "scheduling policy digest"),
                _string(item["schema_identity"], "batch schema_identity"),
                authority,
                effect,
            )
        except ExecutionContractError:
            raise
        except Exception as exc:
            raise ExecutionContractError("Execution Batch fields are invalid") from exc
        if value.canonical_bytes != raw:
            raise ExecutionContractError("Execution Batch bytes differ")
        return value


@dataclass(frozen=True, slots=True)
class WorkerAttempt:
    attempt_id: str
    work_item_id: str
    work_item_version_id: str
    work_item_version_digest: str
    retrieval_context_digest: str
    semantic_request_digest: str
    semantic_request_key: str
    ordinal: int
    previous_attempt_id: str | None
    previous_attempt_digest: str | None
    worker_kind: WorkerKind
    worker_version: str
    input_digest: str
    idempotency_key: str
    priority_digest: str
    schema_identity: str = WORKER_ATTEMPT_SCHEMA
    authority: ContractAuthority = ContractAuthority.NONE
    effect: ContractEffect = ContractEffect.NONE

    @classmethod
    def create(
        cls,
        *,
        member: ExecutionBatchMember,
        ordinal: int,
        previous_attempt: WorkerAttempt | None = None,
        worker_kind: WorkerKind,
        worker_version: str,
        input_digest: str,
    ) -> Self:
        if type(member) is not ExecutionBatchMember:
            raise ExecutionContractError("Worker Attempt member must be typed")
        member = _normalise(
            "Worker Attempt member",
            lambda: ExecutionBatchMember.from_value(member.canonical_value()),
        )
        if previous_attempt is not None and type(previous_attempt) is WorkerAttempt:
            previous_attempt = _normalise(
                "Worker Attempt predecessor",
                lambda: WorkerAttempt.from_canonical_bytes(
                    previous_attempt.canonical_bytes
                ),
            )
        work_item_id = member.work_item_id
        work_item_version_id = member.work_item_version_id
        work_item_version_digest = member.work_item_version_digest
        retrieval_context_digest = member.retrieval_context_digest
        priority_digest = member.priority_digest
        _integer(ordinal, "attempt ordinal")
        if type(worker_kind) is not WorkerKind:
            raise ExecutionContractError("worker kind must be typed")
        _token(worker_version, "worker version")
        _digest(input_digest, "attempt input digest")
        if ordinal == 1 and previous_attempt is not None:
            raise ExecutionContractError("first attempt cannot have a predecessor")
        if ordinal > 1 and type(previous_attempt) is not WorkerAttempt:
            raise ExecutionContractError("later attempt requires its exact predecessor")
        request_value = {
            "work_item_id": work_item_id,
            "work_item_version_id": work_item_version_id,
            "work_item_version_digest": work_item_version_digest,
            "retrieval_context_digest": retrieval_context_digest,
            "worker_kind": worker_kind.value,
            "worker_version": worker_version,
            "input_digest": input_digest,
            "priority_digest": priority_digest,
        }
        request_digest = digest_bytes(_canonical(request_value, "semantic request"))
        previous_id = None if previous_attempt is None else previous_attempt.attempt_id
        previous_digest = (
            None if previous_attempt is None else previous_attempt.canonical_digest
        )
        if previous_attempt is not None and (
            previous_attempt.work_item_id != work_item_id
            or previous_attempt.work_item_version_id != work_item_version_id
            or previous_attempt.work_item_version_digest != work_item_version_digest
            or previous_attempt.retrieval_context_digest != retrieval_context_digest
            or previous_attempt.priority_digest != priority_digest
            or previous_attempt.ordinal + 1 != ordinal
        ):
            raise ExecutionContractError("attempt predecessor binding differs")
        identity = (
            f"{WORKER_ATTEMPT_SCHEMA}|{request_digest}|{ordinal}|"
            f"{previous_id}|{previous_digest}"
        )
        attempt_id = str(uuid.uuid5(uuid.NAMESPACE_URL, identity))
        return cls(
            attempt_id,
            work_item_id,
            work_item_version_id,
            work_item_version_digest,
            retrieval_context_digest,
            request_digest,
            f"worker-request:{request_digest}",
            ordinal,
            previous_id,
            previous_digest,
            worker_kind,
            worker_version,
            input_digest,
            f"worker-request:{request_digest}",
            priority_digest,
        )

    def __post_init__(self) -> None:
        for value, field in (
            (self.attempt_id, "attempt_id"),
            (self.work_item_id, "work_item_id"),
            (self.work_item_version_id, "work_item_version_id"),
        ):
            _uuid(value, field)
        _digest(self.work_item_version_digest, "Work Item Version digest")
        _digest(self.retrieval_context_digest, "retrieval context digest")
        _digest(self.input_digest, "attempt input digest")
        _digest(self.semantic_request_digest, "semantic request digest")
        _integer(self.ordinal, "attempt ordinal")
        if type(self.worker_kind) is not WorkerKind:
            raise ExecutionContractError("worker kind must be typed")
        _digest(self.priority_digest, "attempt priority digest")
        request_value = {
            "work_item_id": self.work_item_id,
            "work_item_version_id": self.work_item_version_id,
            "work_item_version_digest": self.work_item_version_digest,
            "retrieval_context_digest": self.retrieval_context_digest,
            "worker_kind": self.worker_kind.value,
            "worker_version": self.worker_version,
            "input_digest": self.input_digest,
            "priority_digest": self.priority_digest,
        }
        request_digest = digest_bytes(_canonical(request_value, "semantic request"))
        identity = (
            f"{WORKER_ATTEMPT_SCHEMA}|{request_digest}|{self.ordinal}|"
            f"{self.previous_attempt_id}|{self.previous_attempt_digest}"
        )
        expected = str(uuid.uuid5(uuid.NAMESPACE_URL, identity))
        if (
            (self.ordinal == 1)
            != (
                self.previous_attempt_id is None
                and self.previous_attempt_digest is None
            )
            or (
                self.previous_attempt_id is not None
                and self.previous_attempt_digest is None
            )
            or (
                self.previous_attempt_id is None
                and self.previous_attempt_digest is not None
            )
        ):
            raise ExecutionContractError("Worker Attempt predecessor binding differs")
        if self.previous_attempt_id is not None:
            _uuid(self.previous_attempt_id, "previous_attempt_id")
            _digest(self.previous_attempt_digest, "previous attempt digest")
        if (
            self.attempt_id != expected
            or self.semantic_request_digest != request_digest
            or self.semantic_request_key != f"worker-request:{request_digest}"
            or self.idempotency_key != self.semantic_request_key
        ):
            raise ExecutionContractError(
                "Worker Attempt deterministic identity differs"
            )
        _token(self.worker_version, "worker version")
        if (
            type(self.schema_identity) is not str
            or self.schema_identity != WORKER_ATTEMPT_SCHEMA
            or type(self.authority) is not ContractAuthority
            or self.authority is not ContractAuthority.NONE
            or type(self.effect) is not ContractEffect
            or self.effect is not ContractEffect.NONE
            or len(self.canonical_bytes) > _MAX_CANONICAL_BYTES
        ):
            raise ExecutionContractError(
                "Worker Attempt claims authority, effect, or excess bytes"
            )

    def canonical_value(self) -> dict[str, object]:
        return {
            "schema_identity": self.schema_identity,
            "authority": self.authority.value,
            "effect": self.effect.value,
            "attempt_id": self.attempt_id,
            "work_item_id": self.work_item_id,
            "work_item_version_id": self.work_item_version_id,
            "work_item_version_digest": self.work_item_version_digest,
            "retrieval_context_digest": self.retrieval_context_digest,
            "semantic_request_digest": self.semantic_request_digest,
            "semantic_request_key": self.semantic_request_key,
            "ordinal": self.ordinal,
            "previous_attempt_id": self.previous_attempt_id,
            "previous_attempt_digest": self.previous_attempt_digest,
            "worker_kind": self.worker_kind.value,
            "worker_version": self.worker_version,
            "input_digest": self.input_digest,
            "idempotency_key": self.idempotency_key,
            "priority_digest": self.priority_digest,
        }

    @property
    def canonical_bytes(self) -> bytes:
        return _normalise(
            "Worker Attempt",
            lambda: _canonical(self.canonical_value(), "Worker Attempt"),
        )

    @property
    def canonical_digest(self) -> str:
        return _normalise(
            "Worker Attempt digest", lambda: digest_bytes(self.canonical_bytes)
        )

    @property
    def proposal_binding(self) -> WorkerAttemptBinding:
        return _normalise(
            "Worker Attempt proposal binding",
            lambda: WorkerAttemptBinding(
                self.attempt_id,
                self.canonical_digest,
                self.worker_kind,
                self.worker_version,
                self.input_digest,
                self.work_item_version_digest,
                self.retrieval_context_digest,
            ),
        )

    @classmethod
    def from_canonical_bytes(cls, raw: bytes) -> Self:
        item = _exact(
            _decode(raw),
            {
                "schema_identity",
                "authority",
                "effect",
                "attempt_id",
                "work_item_id",
                "work_item_version_id",
                "work_item_version_digest",
                "retrieval_context_digest",
                "semantic_request_digest",
                "semantic_request_key",
                "ordinal",
                "previous_attempt_id",
                "previous_attempt_digest",
                "worker_kind",
                "worker_version",
                "input_digest",
                "idempotency_key",
                "priority_digest",
            },
            "Worker Attempt",
        )
        try:
            worker_kind = WorkerKind(item["worker_kind"])
            authority = ContractAuthority(item["authority"])
            effect = ContractEffect(item["effect"])
        except Exception as exc:
            raise ExecutionContractError("Worker Attempt typed fields differ") from exc
        value = cls(
            _string(item["attempt_id"], "attempt_id"),
            _string(item["work_item_id"], "attempt work_item_id"),
            _string(item["work_item_version_id"], "attempt version_id"),
            _string(item["work_item_version_digest"], "attempt version digest"),
            _string(item["retrieval_context_digest"], "attempt retrieval digest"),
            _string(item["semantic_request_digest"], "semantic request digest"),
            _string(item["semantic_request_key"], "semantic request key"),
            _integer(item["ordinal"], "attempt ordinal"),
            None
            if item["previous_attempt_id"] is None
            else _string(item["previous_attempt_id"], "previous_attempt_id"),
            None
            if item["previous_attempt_digest"] is None
            else _string(item["previous_attempt_digest"], "previous_attempt_digest"),
            worker_kind,
            _string(item["worker_version"], "worker_version"),
            _string(item["input_digest"], "input_digest"),
            _string(item["idempotency_key"], "idempotency_key"),
            _string(item["priority_digest"], "attempt priority digest"),
            _string(item["schema_identity"], "attempt schema_identity"),
            authority,
            effect,
        )
        if value.canonical_bytes != raw:
            raise ExecutionContractError("Worker Attempt bytes differ")
        return value


@dataclass(frozen=True, slots=True)
class LeaseProgressEvidence:
    progress: LeaseProgress
    evidence_digest: str

    def __post_init__(self) -> None:
        if type(self.progress) is not LeaseProgress:
            raise ExecutionContractError("lease progress must be exact typed")
        _digest(self.evidence_digest, "lease progress evidence digest")

    def canonical_value(self) -> dict[str, object]:
        return {
            "progress": self.progress.value,
            "evidence_digest": self.evidence_digest,
        }

    @classmethod
    def from_value(cls, value: object) -> Self:
        item = _exact(value, {"progress", "evidence_digest"}, "lease progress")
        try:
            progress = LeaseProgress(item["progress"])
            return cls(
                progress, _string(item["evidence_digest"], "lease progress digest")
            )
        except ExecutionContractError:
            raise
        except Exception as exc:
            raise ExecutionContractError("lease progress fields are invalid") from exc


_ALLOWED_LEASE_TRANSITIONS = frozenset(
    {
        (LeaseLifecycle.PENDING, LeaseLifecycle.CLAIMED),
        (LeaseLifecycle.CLAIMED, LeaseLifecycle.RELEASED),
        (LeaseLifecycle.CLAIMED, LeaseLifecycle.EXPIRED),
    }
)
_PROGRESS_SUCCESSORS = {
    LeaseProgress.NOT_STARTED: frozenset(
        {
            LeaseProgress.NOT_STARTED,
            LeaseProgress.IN_PROGRESS,
            LeaseProgress.COMPLETED,
            LeaseProgress.INTERRUPTED,
        }
    ),
    LeaseProgress.IN_PROGRESS: frozenset(
        {LeaseProgress.IN_PROGRESS, LeaseProgress.COMPLETED, LeaseProgress.INTERRUPTED}
    ),
    LeaseProgress.COMPLETED: frozenset(),
    LeaseProgress.INTERRUPTED: frozenset(),
}


def _lease_successor_parameters_digest(issued_at: str, expires_at: str) -> str:
    return digest_bytes(
        _canonical(
            {"issued_at": issued_at, "expires_at": expires_at},
            "lease successor parameters",
        )
    )


def _progress_evidence(
    value: object,
    *,
    previous: LeaseProgress | None = None,
) -> tuple[LeaseProgressEvidence, ...]:
    if (
        type(value) is not tuple
        or len(value) > 32
        or any(type(item) is not LeaseProgressEvidence for item in value)
    ):
        raise ExecutionContractError(
            "lease progress evidence must be bounded exact typed"
        )
    value = tuple(
        _normalise(
            "lease progress evidence",
            lambda item=item: LeaseProgressEvidence.from_value(
                item.canonical_value()
            ),
        )
        for item in value
    )
    current = previous
    for item in value:
        if current is not None and item.progress not in _PROGRESS_SUCCESSORS[current]:
            raise ExecutionContractError(
                "lease progress cannot regress or follow a terminal state"
            )
        current = item.progress
    return value


@dataclass(frozen=True, slots=True)
class LeaseTransitionReceipt:
    transition_id: str
    lease_id: str
    predecessor_digest: str
    successor_parameters_digest: str
    actor_identity_digest: str
    from_lifecycle: LeaseLifecycle
    to_lifecycle: LeaseLifecycle
    observed_at: str
    progress: tuple[LeaseProgressEvidence, ...]

    @classmethod
    def create(
        cls,
        *,
        lease_id: str,
        predecessor_digest: str,
        successor_parameters_digest: str,
        actor_identity_digest: str,
        from_lifecycle: LeaseLifecycle,
        to_lifecycle: LeaseLifecycle,
        observed_at: str,
        progress: tuple[LeaseProgressEvidence, ...] = (),
    ) -> Self:
        _uuid(lease_id, "lease transition lease_id")
        _digest(predecessor_digest, "lease predecessor digest")
        _digest(successor_parameters_digest, "lease successor parameters digest")
        _digest(actor_identity_digest, "lease actor identity digest")
        if (
            type(from_lifecycle) is not LeaseLifecycle
            or type(to_lifecycle) is not LeaseLifecycle
        ):
            raise ExecutionContractError(
                "lease transition lifecycle must be exact typed"
            )
        if (from_lifecycle, to_lifecycle) not in _ALLOWED_LEASE_TRANSITIONS:
            raise ExecutionContractError("lease lifecycle transition is not allowed")
        _utc(observed_at, "lease transition observed_at")
        progress = _progress_evidence(progress)
        identity = _canonical(
            {
                "lease_id": lease_id,
                "predecessor_digest": predecessor_digest,
                "successor_parameters_digest": successor_parameters_digest,
                "actor_identity_digest": actor_identity_digest,
                "from_lifecycle": from_lifecycle.value,
                "to_lifecycle": to_lifecycle.value,
                "observed_at": observed_at,
                "progress": [item.canonical_value() for item in progress],
            },
            "lease transition",
        )
        return cls(
            str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"{WORK_ITEM_LEASE_SCHEMA}|transition|{digest_bytes(identity)}",
                )
            ),
            lease_id,
            predecessor_digest,
            successor_parameters_digest,
            actor_identity_digest,
            from_lifecycle,
            to_lifecycle,
            observed_at,
            progress,
        )

    def __post_init__(self) -> None:
        _normalise("lease transition", self._validate)

    def _validate(self) -> None:
        _uuid(self.transition_id, "lease transition_id")
        _uuid(self.lease_id, "lease transition lease_id")
        _digest(self.predecessor_digest, "lease predecessor digest")
        _digest(
            self.successor_parameters_digest,
            "lease successor parameters digest",
        )
        _digest(self.actor_identity_digest, "lease actor identity digest")
        if (
            type(self.from_lifecycle) is not LeaseLifecycle
            or type(self.to_lifecycle) is not LeaseLifecycle
        ):
            raise ExecutionContractError(
                "lease transition lifecycle must be exact typed"
            )
        if (self.from_lifecycle, self.to_lifecycle) not in _ALLOWED_LEASE_TRANSITIONS:
            raise ExecutionContractError("lease lifecycle transition is not allowed")
        _utc(self.observed_at, "lease transition observed_at")
        _progress_evidence(self.progress)
        identity = _canonical(
            {
                "lease_id": self.lease_id,
                "predecessor_digest": self.predecessor_digest,
                "successor_parameters_digest": self.successor_parameters_digest,
                "actor_identity_digest": self.actor_identity_digest,
                "from_lifecycle": self.from_lifecycle.value,
                "to_lifecycle": self.to_lifecycle.value,
                "observed_at": self.observed_at,
                "progress": [item.canonical_value() for item in self.progress],
            },
            "lease transition",
        )
        expected = str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"{WORK_ITEM_LEASE_SCHEMA}|transition|{digest_bytes(identity)}",
            )
        )
        if self.transition_id != expected:
            raise ExecutionContractError("lease transition identity differs")

    def canonical_value(self) -> dict[str, object]:
        return {
            "transition_id": self.transition_id,
            "lease_id": self.lease_id,
            "predecessor_digest": self.predecessor_digest,
            "successor_parameters_digest": self.successor_parameters_digest,
            "actor_identity_digest": self.actor_identity_digest,
            "from_lifecycle": self.from_lifecycle.value,
            "to_lifecycle": self.to_lifecycle.value,
            "observed_at": self.observed_at,
            "progress": [item.canonical_value() for item in self.progress],
        }

    @classmethod
    def from_value(cls, value: object) -> Self:
        item = _exact(
            value,
            {
                "transition_id",
                "lease_id",
                "predecessor_digest",
                "successor_parameters_digest",
                "actor_identity_digest",
                "from_lifecycle",
                "to_lifecycle",
                "observed_at",
                "progress",
            },
            "lease transition",
        )
        if type(item["progress"]) is not list:
            raise ExecutionContractError("lease progress must be an array")
        try:
            return cls(
                _string(item["transition_id"], "lease transition_id"),
                _string(item["lease_id"], "lease transition lease_id"),
                _string(item["predecessor_digest"], "lease predecessor digest"),
                _string(
                    item["successor_parameters_digest"],
                    "lease successor parameters digest",
                ),
                _string(
                    item["actor_identity_digest"],
                    "lease actor identity digest",
                ),
                LeaseLifecycle(item["from_lifecycle"]),
                LeaseLifecycle(item["to_lifecycle"]),
                _string(item["observed_at"], "lease transition observed_at"),
                tuple(
                    LeaseProgressEvidence.from_value(entry)
                    for entry in item["progress"]
                ),
            )
        except ExecutionContractError:
            raise
        except Exception as exc:
            raise ExecutionContractError("lease transition fields are invalid") from exc


@dataclass(frozen=True, slots=True)
class WorkItemLease:
    lease_id: str
    attempt_id: str
    attempt_digest: str
    work_item_id: str
    work_item_version_id: str
    work_item_version_digest: str
    owner_id: str
    owner_profile_digest: str
    capability_digest: str
    fence: int
    lifecycle: LeaseLifecycle
    issued_at: str | None
    expires_at: str | None
    transitions: tuple[LeaseTransitionReceipt, ...]
    schema_identity: str = WORK_ITEM_LEASE_SCHEMA
    authority: ContractAuthority = ContractAuthority.NONE
    effect: ContractEffect = ContractEffect.NONE

    @classmethod
    def pending(
        cls,
        *,
        attempt: WorkerAttempt,
        owner_id: str,
        owner_profile_digest: str,
        capability_digest: str,
        fence: int,
    ) -> Self:
        if type(attempt) is not WorkerAttempt:
            raise ExecutionContractError("lease attempt must be exact typed")
        try:
            _integer(fence, "lease fence")
            _token(owner_id, "lease owner")
            _digest(owner_profile_digest, "lease owner profile digest")
            _digest(capability_digest, "lease capability")
            values = (
                attempt.attempt_id,
                attempt.canonical_digest,
                attempt.work_item_id,
                attempt.work_item_version_id,
                attempt.work_item_version_digest,
                owner_id,
                owner_profile_digest,
                capability_digest,
                fence,
            )
            return cls(
                cls._identity(*values), *values, LeaseLifecycle.PENDING, None, None, ()
            )
        except ExecutionContractError:
            raise
        except Exception as exc:
            raise ExecutionContractError("pending Lease fields are invalid") from exc

    @staticmethod
    def _identity(
        attempt_id: str,
        attempt_digest: str,
        work_item_id: str,
        work_item_version_id: str,
        work_item_version_digest: str,
        owner_id: str,
        owner_profile_digest: str,
        capability_digest: str,
        fence: int,
    ) -> str:
        value = {
            "attempt_id": attempt_id,
            "attempt_digest": attempt_digest,
            "work_item_id": work_item_id,
            "work_item_version_id": work_item_version_id,
            "work_item_version_digest": work_item_version_digest,
            "owner_id": owner_id,
            "owner_profile_digest": owner_profile_digest,
            "capability_digest": capability_digest,
            "fence": fence,
        }
        return str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"{WORK_ITEM_LEASE_SCHEMA}|{digest_bytes(_canonical(value, 'lease identity'))}",
            )
        )

    def __post_init__(self) -> None:
        _normalise("Work Item Lease", self._validate)

    def _validate(self) -> None:
        for value, field in (
            (self.lease_id, "lease_id"),
            (self.attempt_id, "lease attempt_id"),
            (self.work_item_id, "lease work_item_id"),
            (self.work_item_version_id, "lease work_item_version_id"),
        ):
            _uuid(value, field)
        _digest(self.attempt_digest, "lease attempt digest")
        _digest(self.work_item_version_digest, "lease Work Item Version digest")
        _token(self.owner_id, "lease owner")
        _digest(self.owner_profile_digest, "lease owner profile digest")
        _digest(self.capability_digest, "lease capability")
        _integer(self.fence, "lease fence")
        if type(self.lifecycle) is not LeaseLifecycle:
            raise ExecutionContractError("Lease lifecycle must be exact typed")
        if (
            type(self.transitions) is not tuple
            or len(self.transitions) > 2
            or any(
                type(receipt) is not LeaseTransitionReceipt
                for receipt in self.transitions
            )
        ):
            raise ExecutionContractError(
                "Lease transition chain must be bounded exact typed"
            )
        if (
            type(self.schema_identity) is not str
            or self.schema_identity != WORK_ITEM_LEASE_SCHEMA
            or type(self.authority) is not ContractAuthority
            or self.authority is not ContractAuthority.NONE
            or type(self.effect) is not ContractEffect
            or self.effect is not ContractEffect.NONE
        ):
            raise ExecutionContractError("Lease contract claims authority or effect")
        expected = self._identity(
            self.attempt_id,
            self.attempt_digest,
            self.work_item_id,
            self.work_item_version_id,
            self.work_item_version_digest,
            self.owner_id,
            self.owner_profile_digest,
            self.capability_digest,
            self.fence,
        )
        if self.lease_id != expected:
            raise ExecutionContractError("Lease deterministic identity differs")
        if self.lifecycle is LeaseLifecycle.PENDING:
            if (
                self.issued_at is not None
                or self.expires_at is not None
                or self.transitions
            ):
                raise ExecutionContractError(
                    "pending Lease cannot contain acquisition evidence"
                )
        else:
            issued = _utc(self.issued_at, "lease issued_at")
            expires = _utc(self.expires_at, "lease expires_at")
            if _utc_value(expires) <= _utc_value(issued):
                raise ExecutionContractError("lease expiry must follow issue time")
            expected_count = 1 if self.lifecycle is LeaseLifecycle.CLAIMED else 2
            if len(self.transitions) != expected_count:
                raise ExecutionContractError("Lease lifecycle transition chain differs")
            state = LeaseLifecycle.PENDING
            prior_progress: LeaseProgress | None = None
            predecessor = digest_bytes(
                _canonical(
                    self._canonical_value_for(LeaseLifecycle.PENDING, None, None, ()),
                    "Lease genesis",
                )
            )
            successor_parameters_digest = _lease_successor_parameters_digest(
                issued, expires
            )
            for index, receipt in enumerate(self.transitions):
                if (
                    receipt.lease_id != self.lease_id
                    or receipt.successor_parameters_digest
                    != successor_parameters_digest
                    or receipt.from_lifecycle is not state
                    or (receipt.from_lifecycle, receipt.to_lifecycle)
                    not in _ALLOWED_LEASE_TRANSITIONS
                    or receipt.predecessor_digest != predecessor
                ):
                    raise ExecutionContractError(
                        "Lease transition predecessor chain differs"
                    )
                _progress_evidence(receipt.progress, previous=prior_progress)
                if receipt.progress:
                    prior_progress = receipt.progress[-1].progress
                state = receipt.to_lifecycle
                prefix = self.transitions[: index + 1]
                predecessor = digest_bytes(
                    _canonical(
                        self._canonical_value_for(state, issued, expires, prefix),
                        "Lease transition predecessor",
                    )
                )
            if state is not self.lifecycle:
                raise ExecutionContractError("Lease transition chain terminal differs")
            acquisition = self.transitions[0]
            if acquisition.observed_at != issued:
                raise ExecutionContractError(
                    "Lease acquisition transition time differs"
                )
            if self.lifecycle is LeaseLifecycle.RELEASED:
                observed = _utc_value(self.transitions[-1].observed_at)
                if not _utc_value(issued) <= observed < _utc_value(expires):
                    raise ExecutionContractError(
                        "Lease release transition time differs"
                    )
            if self.lifecycle is LeaseLifecycle.EXPIRED and _utc_value(
                self.transitions[-1].observed_at
            ) < _utc_value(expires):
                raise ExecutionContractError("Lease expiry transition time differs")
        _ = self.canonical_bytes

    @property
    def transition(self) -> LeaseTransitionReceipt | None:
        return _normalise(
            "Work Item Lease transition",
            lambda: None if not self.transitions else self.transitions[-1],
        )

    def claim(
        self,
        *,
        issued_at: str,
        expires_at: str,
        actor_identity_digest: str,
        progress: tuple[LeaseProgressEvidence, ...] = (),
    ) -> Self:
        if self.lifecycle is not LeaseLifecycle.PENDING:
            raise ExecutionContractError("Lease lifecycle transition is not allowed")
        _utc(issued_at, "lease issued_at")
        _utc(expires_at, "lease expires_at")
        _digest(actor_identity_digest, "lease actor identity digest")
        if _utc_value(expires_at) <= _utc_value(issued_at):
            raise ExecutionContractError("lease expiry must follow issue time")
        receipt = LeaseTransitionReceipt.create(
            lease_id=self.lease_id,
            predecessor_digest=self.canonical_digest,
            successor_parameters_digest=_lease_successor_parameters_digest(
                issued_at, expires_at
            ),
            actor_identity_digest=actor_identity_digest,
            from_lifecycle=LeaseLifecycle.PENDING,
            to_lifecycle=LeaseLifecycle.CLAIMED,
            observed_at=issued_at,
            progress=progress,
        )
        return self._with(
            LeaseLifecycle.CLAIMED, issued_at, expires_at, self.transitions + (receipt,)
        )

    def release(
        self,
        *,
        observed_at: str,
        actor_identity_digest: str,
        progress: tuple[LeaseProgressEvidence, ...] = (),
    ) -> Self:
        if self.lifecycle is not LeaseLifecycle.CLAIMED:
            raise ExecutionContractError("Lease lifecycle transition is not allowed")
        if self.is_expired_at(observed_at):
            raise ExecutionContractError(
                "Lease at or beyond expiry must use expired transition"
            )
        return self._finish(
            LeaseLifecycle.RELEASED,
            observed_at,
            actor_identity_digest,
            progress,
        )

    def expire(
        self,
        *,
        observed_at: str,
        actor_identity_digest: str,
        progress: tuple[LeaseProgressEvidence, ...] = (),
    ) -> Self:
        if self.lifecycle is not LeaseLifecycle.CLAIMED:
            raise ExecutionContractError("Lease lifecycle transition is not allowed")
        if not self.is_expired_at(observed_at):
            raise ExecutionContractError("Lease has not reached its expiry boundary")
        return self._finish(
            LeaseLifecycle.EXPIRED,
            observed_at,
            actor_identity_digest,
            progress,
        )

    def _finish(
        self,
        lifecycle: LeaseLifecycle,
        observed_at: str,
        actor_identity_digest: str,
        progress: tuple[LeaseProgressEvidence, ...],
    ) -> Self:
        _utc(observed_at, "lease transition observed_at")
        _digest(actor_identity_digest, "lease actor identity digest")
        if (
            self.issued_at is None
            or self.expires_at is None
            or _utc_value(observed_at) < _utc_value(self.issued_at)
        ):
            raise ExecutionContractError(
                "Lease transition cannot precede acquisition time"
            )
        prior = None
        for receipt in self.transitions:
            if receipt.progress:
                prior = receipt.progress[-1].progress
        _progress_evidence(progress, previous=prior)
        receipt = LeaseTransitionReceipt.create(
            lease_id=self.lease_id,
            predecessor_digest=self.canonical_digest,
            successor_parameters_digest=_lease_successor_parameters_digest(
                self.issued_at, self.expires_at
            ),
            actor_identity_digest=actor_identity_digest,
            from_lifecycle=LeaseLifecycle.CLAIMED,
            to_lifecycle=lifecycle,
            observed_at=observed_at,
            progress=progress,
        )
        return self._with(
            lifecycle, self.issued_at, self.expires_at, self.transitions + (receipt,)
        )

    def _with(
        self,
        lifecycle: LeaseLifecycle,
        issued_at: str | None,
        expires_at: str | None,
        transitions: tuple[LeaseTransitionReceipt, ...],
    ) -> Self:
        return WorkItemLease(
            self.lease_id,
            self.attempt_id,
            self.attempt_digest,
            self.work_item_id,
            self.work_item_version_id,
            self.work_item_version_digest,
            self.owner_id,
            self.owner_profile_digest,
            self.capability_digest,
            self.fence,
            lifecycle,
            issued_at,
            expires_at,
            transitions,
            self.schema_identity,
            self.authority,
            self.effect,
        )

    def is_expired_at(self, observed_at: str) -> bool:
        _utc(observed_at, "lease observation time")
        return (
            self.lifecycle is LeaseLifecycle.CLAIMED
            and self.expires_at is not None
            and _utc_value(observed_at) >= _utc_value(self.expires_at)
        )

    def _canonical_value_for(
        self,
        lifecycle: LeaseLifecycle,
        issued_at: str | None,
        expires_at: str | None,
        transitions: tuple[LeaseTransitionReceipt, ...],
    ) -> dict[str, object]:
        return {
            "schema_identity": self.schema_identity,
            "authority": self.authority.value,
            "effect": self.effect.value,
            "lease_id": self.lease_id,
            "attempt_id": self.attempt_id,
            "attempt_digest": self.attempt_digest,
            "work_item_id": self.work_item_id,
            "work_item_version_id": self.work_item_version_id,
            "work_item_version_digest": self.work_item_version_digest,
            "owner_id": self.owner_id,
            "owner_profile_digest": self.owner_profile_digest,
            "capability_digest": self.capability_digest,
            "fence": self.fence,
            "lifecycle": lifecycle.value,
            "issued_at": issued_at,
            "expires_at": expires_at,
            "transitions": [receipt.canonical_value() for receipt in transitions],
        }

    def canonical_value(self) -> dict[str, object]:
        return self._canonical_value_for(
            self.lifecycle, self.issued_at, self.expires_at, self.transitions
        )

    @property
    def canonical_bytes(self) -> bytes:
        return _normalise(
            "Work Item Lease",
            lambda: _canonical(self.canonical_value(), "Work Item Lease"),
        )

    @property
    def canonical_digest(self) -> str:
        return _normalise(
            "Work Item Lease digest", lambda: digest_bytes(self.canonical_bytes)
        )

    @classmethod
    def from_canonical_bytes(cls, raw: bytes) -> Self:
        fields = {
            "schema_identity",
            "authority",
            "effect",
            "lease_id",
            "attempt_id",
            "attempt_digest",
            "work_item_id",
            "work_item_version_id",
            "work_item_version_digest",
            "owner_id",
            "owner_profile_digest",
            "capability_digest",
            "fence",
            "lifecycle",
            "issued_at",
            "expires_at",
            "transitions",
        }
        item = _exact(_decode(raw), fields, "Work Item Lease")
        if type(item["transitions"]) is not list:
            raise ExecutionContractError("Lease transitions must be an array")
        try:
            value = cls(
                _string(item["lease_id"], "lease_id"),
                _string(item["attempt_id"], "lease attempt_id"),
                _string(item["attempt_digest"], "lease attempt digest"),
                _string(item["work_item_id"], "lease work_item_id"),
                _string(item["work_item_version_id"], "lease version_id"),
                _string(item["work_item_version_digest"], "lease version digest"),
                _string(item["owner_id"], "owner_id"),
                _string(item["owner_profile_digest"], "owner_profile_digest"),
                _string(item["capability_digest"], "capability_digest"),
                _integer(item["fence"], "lease fence"),
                LeaseLifecycle(item["lifecycle"]),
                None
                if item["issued_at"] is None
                else _string(item["issued_at"], "issued_at"),
                None
                if item["expires_at"] is None
                else _string(item["expires_at"], "expires_at"),
                tuple(
                    LeaseTransitionReceipt.from_value(entry)
                    for entry in item["transitions"]
                ),
                _string(item["schema_identity"], "lease schema_identity"),
                ContractAuthority(item["authority"]),
                ContractEffect(item["effect"]),
            )
        except ExecutionContractError:
            raise
        except Exception as exc:
            raise ExecutionContractError("Lease fields are invalid") from exc
        if value.canonical_bytes != raw:
            raise ExecutionContractError("Lease bytes differ")
        return value


__all__ = [
    "EXECUTION_BATCH",
    "EXECUTION_BATCH_SCHEMA",
    "WORKER_ATTEMPT",
    "WORKER_ATTEMPT_SCHEMA",
    "WORK_ITEM_LEASE_OWNERSHIP",
    "WORK_ITEM_LEASE_SCHEMA",
    "ExecutionBatch",
    "ExecutionBatchMember",
    "ExecutionContractError",
    "LeaseLifecycle",
    "LeaseProgress",
    "LeaseProgressEvidence",
    "LeaseTransitionReceipt",
    "WorkItemLease",
    "WorkerAttempt",
]

# The only public v20 authority composition seam.  Importing here preserves the
# phase-one value types above while keeping the trusted SQLite implementation
# private to ``newsroom.increment6``.
from ._execution_store import (  # noqa: E402
    TriageExecutionAuthority,
    TriageExecutionAuthorityError,
    open_triage_execution_authority,
)

__all__ += [
    "TriageExecutionAuthority",
    "TriageExecutionAuthorityError",
    "open_triage_execution_authority",
]
