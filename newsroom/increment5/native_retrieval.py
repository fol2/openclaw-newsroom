"""Native governed passage full-text and vector retrieval.

This is the non-fixture document seam for Increment 5.  The Neo4j state is a
rebuildable projection: exact passage bytes, embedding bytes and provider
accounting remain governed-object authority and the typed command event binds
the immutable document manifest.
"""

from __future__ import annotations

import json
import math
import re
import struct
import uuid
from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from newsroom.authority import AuthenticationProof
from newsroom.authority.canonical import canonical_json_bytes, digest_bytes, validate_sha256_digest
from newsroom.authority.models import ObjectAdmissionPayload, SemanticCommand
from newsroom.authority.persistence import AuthorityCommands, AuthorityEvents
from newsroom.authority._object_system import GovernedObjects
from newsroom.authority.objects import HydrationRequest, ObjectAdmissionRequest
from newsroom.authority.types import AggregateId, ObjectAdmissionId, TrustScope, UtcTimestamp
from newsroom.extraction.models import ExtractionRunRequest
from newsroom.extraction.types import ExtractionOutcome, ExtractionPassageId
from newsroom.authority._extraction_facade import GovernedExtractionRecords
from .fulltext_contracts import FullTextAuthorityView, FullTextDocumentBinding, FullTextProjectionSnapshot
from .branch_contracts import BranchMode, BranchOutcome
from .branch_receipts import ExactBranchReceipt
from .fulltext_receipts import FullTextBranchReceipt
from .admitted_graph_retriever import AdmittedGraphReceipt


NATIVE_DOCUMENT_ADMISSION_TYPE = "retrieval.native-document"
NATIVE_VECTOR_ADMISSION_TYPE = "retrieval.native-vector"
NATIVE_EMBEDDING_RECEIPT_ADMISSION_TYPE = "retrieval.native-embedding-receipt"
NATIVE_CONTEXT_ADMISSION_TYPE = "retrieval.native-context"
NATIVE_DOCUMENT_CLASS = "NATIVE_RETRIEVAL_DOCUMENT"
NATIVE_DOCUMENT_USE = "RETRIEVAL_PROJECTION"
NATIVE_VECTOR_CLASS = "NATIVE_RETRIEVAL_EMBEDDING_VECTOR"
NATIVE_VECTOR_USE = "RETRIEVAL_VECTOR"
NATIVE_EMBEDDING_RECEIPT_CLASS = "NATIVE_RETRIEVAL_EMBEDDING_RECEIPT"
NATIVE_EMBEDDING_RECEIPT_USE = "RETRIEVAL_ACCOUNTING"
NATIVE_CONTEXT_CLASS = "NATIVE_RETRIEVAL_CONTEXT"
NATIVE_CONTEXT_USE = "TRIAGE_RETRIEVAL"
NATIVE_DOCUMENT_COMMAND = "retrieval.native_document.admit"
NATIVE_DOCUMENT_EVENT = "retrieval.native_document.admitted"
NATIVE_CONTEXT_COMMAND = "retrieval.native_context.admit"
NATIVE_CONTEXT_EVENT = "retrieval.native_context.admitted"
NATIVE_SECURITY_SCOPE = "authority.retrieval"
NATIVE_RETENTION_SCOPE = "authority.retrieval.retained"
NATIVE_DOCUMENT_SCHEMA = "newsroom.increment5.native-retrieval-document.v1"
NATIVE_EMBEDDING_SCHEMA = "newsroom.increment5.native-embedding-receipt.v1"
NATIVE_VECTOR_DIMENSIONS = 1_024
NATIVE_RESULT_LIMIT = 8
NATIVE_VECTOR_PROFILE = "native-governed-vector-v1"
NATIVE_VECTOR_RECEIPT_SCHEMA = "newsroom.increment5.native-vector-branch-receipt.v1"
NATIVE_CONTEXT_SCHEMA = "newsroom.increment5.native-retrieval-context.v1"
NATIVE_CONTEXT_RECEIPT_SCHEMA = "newsroom.increment5.native-retrieval-context-receipt.v1"
NATIVE_CONTEXT_ADMISSION_TYPE = "retrieval.native-context"
NATIVE_CONTEXT_CLASS = "NATIVE_RETRIEVAL_CONTEXT"
NATIVE_CONTEXT_USE = "TRIAGE_RETRIEVAL"
NATIVE_CONTEXT_COMMAND = "retrieval.native_context.admit"
NATIVE_CONTEXT_EVENT = "retrieval.native_context.admitted"
NATIVE_CONTEXT_SCHEMA = "newsroom.increment5.native-retrieval-context.v1"
NATIVE_CONTEXT_RECEIPT_SCHEMA = "newsroom.increment5.native-retrieval-context-receipt.v1"

_INDEX = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,127}\Z")


class NativeRetrievalError(RuntimeError):
    """Native retrieval input or retained authority is inconsistent."""


class NativeRetrievalHold(NativeRetrievalError):
    """A mandatory native branch has no usable governed input."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _digest(value: str, field: str) -> str:
    try:
        return validate_sha256_digest(value, field=field)
    except (TypeError, ValueError) as exc:
        raise NativeRetrievalError(f"{field} differs") from exc


def _text(value: object, field: str, maximum: int = 512) -> str:
    if type(value) is not str or not value or value != value.strip() or len(value.encode()) > maximum:
        raise NativeRetrievalError(f"{field} differs")
    return value


def _json(raw: bytes, schema: str) -> dict[str, object]:
    try:
        value = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise NativeRetrievalError("governed retrieval object is malformed") from exc
    if type(value) is not dict or canonical_json_bytes(value) != raw or value.get("schema_identity") != schema:
        raise NativeRetrievalError("governed retrieval object is non-canonical")
    return value


@dataclass(frozen=True, slots=True)
class NativeEmbeddingReceipt:
    input_text_digest: str
    vector_digest: str
    dimensions: int
    provider: str
    model: str
    model_digest: str
    provider_request_id: str
    usage_receipt_digest: str
    recorded_at: str
    outcome: str = "COMPLETE"

    def __post_init__(self) -> None:
        for name in ("input_text_digest", "vector_digest", "model_digest", "usage_receipt_digest"):
            _digest(getattr(self, name), name)
        for name in ("provider", "model", "provider_request_id"):
            _text(getattr(self, name), name)
        if self.dimensions != NATIVE_VECTOR_DIMENSIONS:
            raise NativeRetrievalError("native embedding dimensions differ")
        if self.outcome != "COMPLETE":
            raise NativeRetrievalHold("EMBEDDING_NOT_COMPLETE")
        UtcTimestamp.parse(self.recorded_at)

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes({"schema_identity": NATIVE_EMBEDDING_SCHEMA, **self.canonical_value()})

    def canonical_value(self) -> dict[str, object]:
        return {
            "input_text_digest": self.input_text_digest,
            "vector_digest": self.vector_digest,
            "dimensions": self.dimensions,
            "provider": self.provider,
            "model": self.model,
            "model_digest": self.model_digest,
            "provider_request_id": self.provider_request_id,
            "usage_receipt_digest": self.usage_receipt_digest,
            "recorded_at": self.recorded_at,
            "outcome": self.outcome,
        }

    @classmethod
    def from_bytes(cls, raw: bytes) -> "NativeEmbeddingReceipt":
        value = _json(raw, NATIVE_EMBEDDING_SCHEMA)
        value.pop("schema_identity")
        try:
            return cls(**value)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise NativeRetrievalError("embedding receipt fields differ") from exc


@dataclass(frozen=True, slots=True)
class NativeEmbeddingReference:
    vector_admission_id: ObjectAdmissionId
    receipt_admission_id: ObjectAdmissionId

    def __post_init__(self) -> None:
        if type(self.vector_admission_id) is not ObjectAdmissionId or type(self.receipt_admission_id) is not ObjectAdmissionId:
            raise NativeRetrievalError("embedding references must be governed admissions")


@dataclass(frozen=True, slots=True)
class NativePassageDocument:
    generation_id: str
    passage_id: str
    dependency_root_id: str
    source_id: str
    revision_id: str
    representation_id: str
    language: str
    text: str
    text_digest: str
    rights_digest: str
    provenance_digest: str
    vector_digest: str
    vector_admission_id: str
    embedding_receipt_digest: str
    embedding_receipt_admission_id: str
    embedding_model_digest: str

    def __post_init__(self) -> None:
        for name in ("generation_id", "passage_id", "dependency_root_id", "source_id", "revision_id", "representation_id", "language", "vector_admission_id", "embedding_receipt_admission_id"):
            _text(getattr(self, name), name)
        for name in ("text_digest", "rights_digest", "provenance_digest", "vector_digest", "embedding_receipt_digest", "embedding_model_digest"):
            _digest(getattr(self, name), name)
        if digest_bytes(self.text.encode("utf-8")) != self.text_digest:
            raise NativeRetrievalError("native passage text digest differs")

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes({"schema_identity": NATIVE_DOCUMENT_SCHEMA, **self.projection_value()})

    def projection_value(self) -> dict[str, object]:
        return {
            "generation_id": self.generation_id,
            "passage_id": self.passage_id,
            "dependency_root_id": self.dependency_root_id,
            "source_id": self.source_id,
            "revision_id": self.revision_id,
            "representation_id": self.representation_id,
            "language": self.language,
            "text": self.text,
            "text_digest": self.text_digest,
            "rights_digest": self.rights_digest,
            "provenance_digest": self.provenance_digest,
            "vector_digest": self.vector_digest,
            "vector_admission_id": self.vector_admission_id,
            "embedding_receipt_digest": self.embedding_receipt_digest,
            "embedding_receipt_admission_id": self.embedding_receipt_admission_id,
            "embedding_model_digest": self.embedding_model_digest,
        }

    @property
    def digest(self) -> str:
        return digest_bytes(self.canonical_bytes)

    @classmethod
    def from_bytes(cls, raw: bytes) -> "NativePassageDocument":
        value = _json(raw, NATIVE_DOCUMENT_SCHEMA)
        value.pop("schema_identity")
        try:
            return cls(**value)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise NativeRetrievalError("native document fields differ") from exc


@dataclass(frozen=True, slots=True)
class NativeDocumentRequest:
    extraction_request: ExtractionRunRequest
    passage_id: ExtractionPassageId
    dependency_root_id: str
    generation_id: str
    embedding: NativeEmbeddingReference
    aggregate_id: AggregateId
    expected_aggregate_version: int
    idempotency_key: str

    def __post_init__(self) -> None:
        if type(self.extraction_request) is not ExtractionRunRequest or type(self.passage_id) is not ExtractionPassageId:
            raise NativeRetrievalError("native document requires exact extraction authority input")
        if type(self.embedding) is not NativeEmbeddingReference or type(self.aggregate_id) is not AggregateId:
            raise NativeRetrievalError("native document authority identity differs")
        _text(self.dependency_root_id, "dependency_root_id")
        _text(self.generation_id, "generation_id")
        _text(self.idempotency_key, "idempotency_key", 256)
        if type(self.expected_aggregate_version) is not int or self.expected_aggregate_version < 0:
            raise NativeRetrievalError("expected aggregate version differs")


@dataclass(frozen=True, slots=True)
class NativeDocumentReceipt:
    event_id: str
    command_id: str
    aggregate_id: AggregateId
    aggregate_version: int
    admission_id: ObjectAdmissionId
    document_digest: str
    vector_admission_id: ObjectAdmissionId
    embedding_receipt_admission_id: ObjectAdmissionId

    def __post_init__(self) -> None:
        _text(self.event_id, "native_document_event_id")
        _text(self.command_id, "native_document_command_id")
        if type(self.aggregate_id) is not AggregateId or type(self.admission_id) is not ObjectAdmissionId or type(self.vector_admission_id) is not ObjectAdmissionId or type(self.embedding_receipt_admission_id) is not ObjectAdmissionId:
            raise NativeRetrievalError("native document receipt identity differs")
        if type(self.aggregate_version) is not int or self.aggregate_version <= 0:
            raise NativeRetrievalError("native document aggregate version differs")
        _digest(self.document_digest, "native_document_digest")

    def projection_value(self) -> dict[str, object]:
        return {
            "event_id": self.event_id,
            "command_id": self.command_id,
            "aggregate_id": str(self.aggregate_id),
            "aggregate_version": self.aggregate_version,
            "admission_id": str(self.admission_id),
            "document_digest": self.document_digest,
            "vector_admission_id": str(self.vector_admission_id),
            "embedding_receipt_admission_id": str(
                self.embedding_receipt_admission_id
            ),
        }

    @classmethod
    def from_projection(cls, value: Mapping[str, object]) -> "NativeDocumentReceipt":
        expected = {
            "event_id",
            "command_id",
            "aggregate_id",
            "aggregate_version",
            "admission_id",
            "document_digest",
            "vector_admission_id",
            "embedding_receipt_admission_id",
        }
        if set(value) != expected:
            raise NativeRetrievalError("native projection receipt differs")
        try:
            return cls(
                str(value["event_id"]),
                str(value["command_id"]),
                AggregateId.parse(str(value["aggregate_id"])),
                value["aggregate_version"],  # type: ignore[arg-type]
                ObjectAdmissionId.parse(str(value["admission_id"])),
                str(value["document_digest"]),
                ObjectAdmissionId.parse(str(value["vector_admission_id"])),
                ObjectAdmissionId.parse(str(value["embedding_receipt_admission_id"])),
            )
        except (TypeError, ValueError) as exc:
            raise NativeRetrievalError("native projection receipt differs") from exc


@dataclass(frozen=True, slots=True)
class NativeRetrievalHit:
    passage_id: str
    dependency_root_id: str
    score: float

    def __post_init__(self) -> None:
        _text(self.passage_id, "hit_passage_id")
        _text(self.dependency_root_id, "hit_dependency_root_id")
        if type(self.score) is not float or not math.isfinite(self.score):
            raise NativeRetrievalError("native retrieval score differs")


@dataclass(frozen=True, slots=True)
class NativeRetrievalResult:
    fulltext_hits: tuple[NativeRetrievalHit, ...]
    vector_hits: tuple[NativeRetrievalHit, ...]
    generation_id: str
    outcome: str = "COMPLETE"

    def __post_init__(self) -> None:
        if (
            type(self.fulltext_hits) is not tuple
            or type(self.vector_hits) is not tuple
            or any(type(item) is not NativeRetrievalHit for item in self.fulltext_hits + self.vector_hits)
        ):
            raise NativeRetrievalError("native retrieval hits differ")
        _text(self.generation_id, "native_retrieval_generation_id")
        if self.outcome != "COMPLETE":
            raise NativeRetrievalError("native retrieval outcome differs")

    @property
    def no_match(self) -> bool:
        return not self.fulltext_hits and not self.vector_hits


@dataclass(frozen=True, slots=True)
class NativeVectorRequest:
    request_id: str
    idempotency_key: str
    query_event_id: str
    query_document_digest: str
    generation_id: str
    query_valid_time: str
    serving_time: str

    def __post_init__(self) -> None:
        try:
            if str(uuid.UUID(self.request_id)) != self.request_id:
                raise ValueError
        except (TypeError, ValueError, AttributeError) as exc:
            raise NativeRetrievalError("native vector request identity differs") from exc
        for name in ("idempotency_key", "query_event_id", "generation_id"):
            _text(getattr(self, name), name)
        _digest(self.query_document_digest, "query_document_digest")
        if UtcTimestamp.parse(self.query_valid_time).value > UtcTimestamp.parse(self.serving_time).value:
            raise NativeRetrievalError("native vector request time differs")

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes({
            "schema_identity": "newsroom.increment5.native-vector-request.v1",
            **{name: getattr(self, name) for name in self.__dataclass_fields__},
        })

    @property
    def request_digest(self) -> str:
        return digest_bytes(self.canonical_bytes)


@dataclass(frozen=True, slots=True)
class NativeVectorBranchHit:
    rank: int
    passage_id: str
    dependency_root_id: str
    source_revision_id: str
    document_digest: str
    rights_digest: str
    provenance_digest: str
    raw_score_ppm: int

    def __post_init__(self) -> None:
        if type(self.rank) is not int or not 1 <= self.rank <= NATIVE_RESULT_LIMIT:
            raise NativeRetrievalError("native vector rank differs")
        for name in ("passage_id", "dependency_root_id", "source_revision_id"):
            _text(getattr(self, name), name)
        for name in ("document_digest", "rights_digest", "provenance_digest"):
            _digest(getattr(self, name), name)
        if type(self.raw_score_ppm) is not int or not 0 <= self.raw_score_ppm <= 1_000_000:
            raise NativeRetrievalError("native vector score differs")

    def canonical_value(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class NativeVectorBranchReceipt:
    receipt_id: str
    request_digest: str
    mode: BranchMode
    outcome: BranchOutcome
    reason: str | None
    generation_id: str
    generation_digest: str
    profile_id: str
    query_valid_time: str
    serving_time: str
    hits: tuple[NativeVectorBranchHit, ...]
    authority_read_count: int
    external_call_count: int = 0
    provider_call_count: int = 0
    model_call_count: int = 0
    embedding_call_count: int = 0
    provider_spend_micros: int = 0
    read_only: bool = True
    authority_effect: str = "NONE"

    def __post_init__(self) -> None:
        try:
            if str(uuid.UUID(self.receipt_id)) != self.receipt_id:
                raise ValueError
        except (TypeError, ValueError, AttributeError) as exc:
            raise NativeRetrievalError("native vector receipt identity differs") from exc
        _digest(self.request_digest, "native_vector_request_digest")
        if self.mode is not BranchMode.VECTOR or type(self.outcome) is not BranchOutcome:
            raise NativeRetrievalError("native vector receipt mode/outcome differs")
        _text(self.generation_id, "native_vector_generation_id")
        _digest(self.generation_digest, "native_vector_generation_digest")
        if self.profile_id != NATIVE_VECTOR_PROFILE:
            raise NativeRetrievalError("native vector profile differs")
        UtcTimestamp.parse(self.query_valid_time)
        UtcTimestamp.parse(self.serving_time)
        if UtcTimestamp.parse(self.query_valid_time).value > UtcTimestamp.parse(self.serving_time).value:
            raise NativeRetrievalError("native vector receipt time differs")
        if type(self.hits) is not tuple or tuple(hit.rank for hit in self.hits) != tuple(range(1, len(self.hits) + 1)):
            raise NativeRetrievalError("native vector hits differ")
        if self.outcome is BranchOutcome.COMPLETE:
            if (not self.hits) != (self.reason == "NO_MATCH"):
                raise NativeRetrievalError("native vector complete result differs")
        elif self.hits:
            raise NativeRetrievalError("native vector non-complete result has hits")
        if self.authority_read_count < 1 or any(getattr(self, name) != 0 for name in ("external_call_count", "provider_call_count", "model_call_count", "embedding_call_count", "provider_spend_micros")) or not self.read_only or self.authority_effect != "NONE":
            raise NativeRetrievalError("native vector receipt effects differ")

    def canonical_value(self) -> dict[str, object]:
        return {
            "schema_version": NATIVE_VECTOR_RECEIPT_SCHEMA,
            "receipt_id": self.receipt_id,
            "request_digest": self.request_digest,
            "mode": self.mode.value,
            "outcome": self.outcome.value,
            "reason": self.reason,
            "generation_id": self.generation_id,
            "generation_digest": self.generation_digest,
            "profile_id": self.profile_id,
            "query_valid_time": self.query_valid_time,
            "serving_time": self.serving_time,
            "hits": [hit.canonical_value() for hit in self.hits],
            "authority_read_count": self.authority_read_count,
            "external_call_count": self.external_call_count,
            "provider_call_count": self.provider_call_count,
            "model_call_count": self.model_call_count,
            "embedding_call_count": self.embedding_call_count,
            "provider_spend_micros": self.provider_spend_micros,
            "read_only": self.read_only,
            "authority_effect": self.authority_effect,
        }

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.canonical_value())

    @property
    def receipt_digest(self) -> str:
        return digest_bytes(self.canonical_bytes)

    @classmethod
    def from_canonical_bytes(cls, raw: bytes) -> "NativeVectorBranchReceipt":
        try:
            value = json.loads(raw)
            if type(value) is not dict or canonical_json_bytes(value) != raw or value.pop("schema_version", None) != NATIVE_VECTOR_RECEIPT_SCHEMA:
                raise ValueError
            value["mode"] = BranchMode(value["mode"])
            value["outcome"] = BranchOutcome(value["outcome"])
            value["hits"] = tuple(NativeVectorBranchHit(**item) for item in value["hits"])
            receipt = cls(**value)  # type: ignore[arg-type]
        except (KeyError, TypeError, ValueError) as exc:
            raise NativeRetrievalError("native vector receipt fields differ") from exc
        if receipt.canonical_bytes != raw:
            raise NativeRetrievalError("native vector receipt is non-canonical")
        return receipt


@dataclass(frozen=True, slots=True)
class NativeRetrievalContextRequest:
    """Stable server request joining the four mandatory native branches."""

    request_id: str
    idempotency_key: str
    aggregate_id: AggregateId
    expected_aggregate_version: int
    lead_id: str
    lead_digest: str
    exact_receipt_bytes: bytes
    fulltext_receipt_bytes: bytes
    vector_receipt_bytes: bytes
    graph_receipt_bytes: bytes
    selected_documents: tuple[NativeDocumentReceipt, ...]

    def __post_init__(self) -> None:
        try:
            if str(uuid.UUID(self.request_id)) != self.request_id:
                raise ValueError
        except (TypeError, ValueError, AttributeError) as exc:
            raise NativeRetrievalError("native context request identity differs") from exc
        _text(self.idempotency_key, "native context idempotency key", 256)
        if type(self.aggregate_id) is not AggregateId or type(self.expected_aggregate_version) is not int or self.expected_aggregate_version < 0:
            raise NativeRetrievalError("native context aggregate differs")
        _text(self.lead_id, "native context lead id")
        _digest(self.lead_digest, "native context lead digest")
        self.branch_receipts()
        if type(self.selected_documents) is not tuple or not self.selected_documents or len(self.selected_documents) > NATIVE_RESULT_LIMIT or any(type(item) is not NativeDocumentReceipt for item in self.selected_documents):
            raise NativeRetrievalError("native context selected documents differ")
        if len({item.event_id for item in self.selected_documents}) != len(self.selected_documents):
            raise NativeRetrievalError("native context selected documents repeat")

    def branch_receipts(self) -> tuple[ExactBranchReceipt, FullTextBranchReceipt, NativeVectorBranchReceipt, AdmittedGraphReceipt]:
        try:
            values = (
                ExactBranchReceipt.from_canonical_bytes(self.exact_receipt_bytes),
                FullTextBranchReceipt.from_canonical_bytes(self.fulltext_receipt_bytes),
                NativeVectorBranchReceipt.from_canonical_bytes(self.vector_receipt_bytes),
                AdmittedGraphReceipt.from_canonical_bytes(self.graph_receipt_bytes),
            )
        except Exception as exc:
            raise NativeRetrievalError("native context branch receipt differs") from exc
        if any(receipt.outcome is not BranchOutcome.COMPLETE for receipt in values):
            raise NativeRetrievalHold("MANDATORY_RETRIEVAL_BRANCH_NOT_COMPLETE")
        return values

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes({
            "schema_identity": "newsroom.increment5.native-retrieval-context-request.v1",
            "request_id": self.request_id,
            "idempotency_key": self.idempotency_key,
            "aggregate_id": str(self.aggregate_id),
            "expected_aggregate_version": self.expected_aggregate_version,
            "lead_id": self.lead_id,
            "lead_digest": self.lead_digest,
            "branch_receipts": {
                "exact": json.loads(self.exact_receipt_bytes),
                "fulltext": json.loads(self.fulltext_receipt_bytes),
                "vector": json.loads(self.vector_receipt_bytes),
                "admitted_graph": json.loads(self.graph_receipt_bytes),
            },
            "selected_documents": [item.projection_value() for item in self.selected_documents],
        })

    @property
    def request_digest(self) -> str:
        return digest_bytes(self.canonical_bytes)


@dataclass(frozen=True, slots=True)
class NativeRetrievalContext:
    context_id: str
    request_id: str
    request_digest: str
    lead_id: str
    lead_digest: str
    branch_digests: tuple[str, str, str, str]
    selected_documents: tuple[dict[str, object], ...]
    outcome: str
    no_match: bool

    def __post_init__(self) -> None:
        try:
            if str(uuid.UUID(self.context_id)) != self.context_id or str(uuid.UUID(self.request_id)) != self.request_id:
                raise ValueError
        except (TypeError, ValueError, AttributeError) as exc:
            raise NativeRetrievalError("native context identity differs") from exc
        _digest(self.request_digest, "native context request digest")
        _text(self.lead_id, "native context lead id")
        _digest(self.lead_digest, "native context lead digest")
        if type(self.branch_digests) is not tuple or len(self.branch_digests) != 4:
            raise NativeRetrievalError("native context branch inventory differs")
        for value in self.branch_digests:
            _digest(value, "native context branch digest")
        if type(self.selected_documents) is not tuple or any(type(item) is not dict for item in self.selected_documents):
            raise NativeRetrievalError("native context selected inventory differs")
        if self.outcome != "COMPLETE" or type(self.no_match) is not bool:
            raise NativeRetrievalError("native context outcome differs")

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes({"schema_identity": NATIVE_CONTEXT_SCHEMA, **self.canonical_value()})

    def canonical_value(self) -> dict[str, object]:
        return {
            "context_id": self.context_id, "request_id": self.request_id,
            "request_digest": self.request_digest, "lead_id": self.lead_id,
            "lead_digest": self.lead_digest, "branch_digests": list(self.branch_digests),
            "selected_documents": list(self.selected_documents), "outcome": self.outcome,
            "no_match": self.no_match,
        }

    @property
    def digest(self) -> str:
        return digest_bytes(self.canonical_bytes)

    @classmethod
    def from_bytes(cls, raw: bytes) -> "NativeRetrievalContext":
        value = _json(raw, NATIVE_CONTEXT_SCHEMA)
        value.pop("schema_identity")
        try:
            value["branch_digests"] = tuple(value["branch_digests"])
            value["selected_documents"] = tuple(value["selected_documents"])
            return cls(**value)  # type: ignore[arg-type]
        except (KeyError, TypeError, ValueError) as exc:
            raise NativeRetrievalError("native context fields differ") from exc


@dataclass(frozen=True, slots=True)
class NativeRetrievalContextReceipt:
    context_id: str
    request_id: str
    request_digest: str
    aggregate_id: AggregateId
    aggregate_version: int
    event_id: str
    command_id: str
    admission_id: ObjectAdmissionId
    context_object_digest: str
    controller_principal_id: str
    authority_domain: str
    outcome: str = "COMPLETE"
    reason: None = None
    no_match: bool = False

    def __post_init__(self) -> None:
        NativeRetrievalContext(
            self.context_id, self.request_id, self.request_digest, "validation",
            "sha256:" + "0" * 64, tuple("sha256:" + "0" * 64 for _ in range(4)), (),
            self.outcome, self.no_match,
        )
        if type(self.aggregate_id) is not AggregateId or type(self.aggregate_version) is not int or self.aggregate_version <= 0 or type(self.admission_id) is not ObjectAdmissionId:
            raise NativeRetrievalError("native context authority receipt differs")
        for name in ("event_id", "command_id", "controller_principal_id", "authority_domain"):
            _text(getattr(self, name), name)
        _digest(self.context_object_digest, "native context object digest")
        if self.reason is not None:
            raise NativeRetrievalError("complete native context has a reason")

    def canonical_value(self) -> dict[str, object]:
        return {
            "schema_identity": NATIVE_CONTEXT_RECEIPT_SCHEMA,
            "context_id": self.context_id, "request_id": self.request_id,
            "request_digest": self.request_digest, "aggregate_id": str(self.aggregate_id),
            "aggregate_version": self.aggregate_version, "event_id": self.event_id,
            "command_id": self.command_id, "admission_id": str(self.admission_id),
            "context_object_digest": self.context_object_digest,
            "controller_principal_id": self.controller_principal_id,
            "authority_domain": self.authority_domain, "outcome": self.outcome,
            "reason": self.reason, "no_match": self.no_match,
        }

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.canonical_value())

    @property
    def receipt_digest(self) -> str:
        return digest_bytes(self.canonical_bytes)

    @classmethod
    def from_bytes(cls, raw: bytes) -> "NativeRetrievalContextReceipt":
        value = _json(raw, NATIVE_CONTEXT_RECEIPT_SCHEMA)
        value.pop("schema_identity")
        try:
            value["aggregate_id"] = AggregateId.parse(value["aggregate_id"])
            value["admission_id"] = ObjectAdmissionId.parse(value["admission_id"])
            receipt = cls(**value)  # type: ignore[arg-type]
        except (KeyError, TypeError, ValueError) as exc:
            raise NativeRetrievalError("native context receipt fields differ") from exc
        if receipt.canonical_bytes != raw:
            raise NativeRetrievalError("native context receipt is non-canonical")
        return receipt


class NativeDocumentProjection(Protocol):
    def upsert(self, receipt: NativeDocumentReceipt, document: NativePassageDocument, vector: tuple[float, ...]) -> None: ...
    def retrieve(self, *, query_text: str, query_vector: tuple[float, ...]) -> tuple[tuple[Mapping[str, object], ...], tuple[Mapping[str, object], ...]]: ...


class NativeRetrievalDocuments:
    """Trusted facade binding native extraction, embedding and projection state."""

    def __init__(self, *, objects: GovernedObjects, extraction: GovernedExtractionRecords, commands: AuthorityCommands, events: AuthorityEvents, projector: NativeDocumentProjection, reader_principal_id: str, authority_domain: str, controller_principal_id: str, passage_hydration_policy_digest: str, vector_hydration_policy_digest: str, receipt_hydration_policy_digest: str, document_hydration_policy_digest: str, document_admission_definition_digest: str, command_definition_digest: str) -> None:
        if type(objects) is not GovernedObjects or type(extraction) is not GovernedExtractionRecords or type(commands) is not AuthorityCommands or type(events) is not AuthorityEvents:
            raise NativeRetrievalError("native retrieval requires exact authority facades")
        if not callable(getattr(projector, "upsert", None)) or not callable(getattr(projector, "retrieve", None)):
            raise NativeRetrievalError("native retrieval projector differs")
        for value in (reader_principal_id, authority_domain, controller_principal_id): _text(value, "native retrieval identity")
        for value in (passage_hydration_policy_digest, vector_hydration_policy_digest, receipt_hydration_policy_digest, document_hydration_policy_digest, document_admission_definition_digest, command_definition_digest): _digest(value, "native retrieval policy")
        self._objects, self._extraction, self._commands, self._events, self._projector = objects, extraction, commands, events, projector
        self._reader, self._domain, self._controller = reader_principal_id, authority_domain, controller_principal_id
        self._passage_policy, self._vector_policy, self._receipt_policy, self._document_policy = passage_hydration_policy_digest, vector_hydration_policy_digest, receipt_hydration_policy_digest, document_hydration_policy_digest
        self._document_definition, self._command_definition = document_admission_definition_digest, command_definition_digest

    @property
    def command_definition_digest(self) -> str:
        return self._command_definition

    def admit(self, request: NativeDocumentRequest, *, proof: AuthenticationProof) -> tuple[NativeDocumentReceipt, NativePassageDocument]:
        metadata = self._extraction.metadata(request.extraction_request.run_version_id, proof=proof)
        if metadata.input_binding_digest != request.extraction_request.input_binding.digest or metadata.outcome is not ExtractionOutcome.SUCCESS or not metadata.terminal:
            raise NativeRetrievalHold("EXTRACTION_INPUT_NOT_ADMITTED")
        passage = request.extraction_request.input_binding.passage(request.passage_id)
        hydrated_passage = self._objects.hydrate(HydrationRequest(passage.admission_id, passage.purpose), proof=proof)
        self._access(hydrated_passage.decision, self._passage_policy, passage.object_class, passage.allowed_use)
        if digest_bytes(hydrated_passage.data) != passage.blob_digest or len(hydrated_passage.data) != passage.byte_length:
            raise NativeRetrievalError("retained passage bytes differ")
        try:
            text = hydrated_passage.data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise NativeRetrievalHold("PASSAGE_NOT_UTF8") from exc
        vector_object = self._objects.hydrate(HydrationRequest(request.embedding.vector_admission_id, NATIVE_VECTOR_USE), proof=proof)
        self._access(vector_object.decision, self._vector_policy, NATIVE_VECTOR_CLASS, NATIVE_VECTOR_USE)
        vector = _vector(vector_object.data)
        receipt_object = self._objects.hydrate(HydrationRequest(request.embedding.receipt_admission_id, NATIVE_EMBEDDING_RECEIPT_USE), proof=proof)
        self._access(receipt_object.decision, self._receipt_policy, NATIVE_EMBEDDING_RECEIPT_CLASS, NATIVE_EMBEDDING_RECEIPT_USE)
        embedding = NativeEmbeddingReceipt.from_bytes(receipt_object.data)
        if embedding.input_text_digest != passage.text_digest or embedding.vector_digest != digest_bytes(vector_object.data):
            raise NativeRetrievalError("embedding receipt does not bind passage/vector bytes")
        document = NativePassageDocument(
            generation_id=request.generation_id,
            passage_id=str(passage.passage_id),
            dependency_root_id=request.dependency_root_id,
            source_id=str(request.extraction_request.input_binding.definition_id), revision_id=str(request.extraction_request.input_binding.revision_id), representation_id=str(request.extraction_request.input_binding.representation_id),
            language=passage.language, text=text, text_digest=passage.text_digest,
            rights_digest=hydrated_passage.decision.state_cutoff_digest,
            provenance_digest=digest_bytes(request.extraction_request.canonical_bytes), vector_digest=embedding.vector_digest,
            vector_admission_id=str(request.embedding.vector_admission_id),
            embedding_receipt_digest=digest_bytes(receipt_object.data),
            embedding_receipt_admission_id=str(request.embedding.receipt_admission_id),
            embedding_model_digest=embedding.model_digest,
        )
        admitted = self._objects.admit(ObjectAdmissionRequest(NATIVE_DOCUMENT_ADMISSION_TYPE, request.idempotency_key), document.canonical_bytes, proof=proof).admission
        if admitted.definition_digest != self._document_definition or admitted.object_class != NATIVE_DOCUMENT_CLASS or admitted.allowed_use != NATIVE_DOCUMENT_USE or admitted.blob.blob_digest != document.digest or not admitted.active:
            raise NativeRetrievalError("native document admission differs")
        committed = self._commands.execute(SemanticCommand(NATIVE_DOCUMENT_COMMAND, request.aggregate_id, request.expected_aggregate_version, ObjectAdmissionPayload(admitted.admission_id), request.idempotency_key), proof=proof)
        result = NativeDocumentReceipt(str(committed.event_id), str(committed.command_id), request.aggregate_id, committed.aggregate_version, admitted.admission_id, document.digest, request.embedding.vector_admission_id, request.embedding.receipt_admission_id)
        self._verify_event(result, proof)
        self._projector.upsert(result, document, vector)
        return result, document

    def retrieve(self, *, query_receipt: NativeDocumentReceipt, query_text: str, proof: AuthenticationProof) -> NativeRetrievalResult:
        if type(query_receipt) is not NativeDocumentReceipt:
            raise NativeRetrievalError("native retrieval query receipt differs")
        query_document, query_vector = self._read(query_receipt, proof)
        fulltext, vector = self._projector.retrieve(
            query_text=_text(query_text, "query_text", 16_384),
            query_vector=query_vector,
        )
        return NativeRetrievalResult(
            self._hits(fulltext, query_document.generation_id, proof),
            self._hits(vector, query_document.generation_id, proof),
            query_document.generation_id,
        )

    def retrieve_vector(
        self,
        request: NativeVectorRequest,
        *,
        proof: AuthenticationProof,
    ) -> NativeVectorBranchReceipt:
        """Execute one attributed production VECTOR branch without embedding work."""
        if type(request) is not NativeVectorRequest:
            raise NativeRetrievalError("native vector request differs")
        query_receipt = self._receipt_for_event(request.query_event_id, proof)
        query_document, query_vector = self._read(query_receipt, proof)
        if (
            query_document.digest != request.query_document_digest
            or query_document.generation_id != request.generation_id
        ):
            raise NativeRetrievalError("native vector query authority differs")
        _, rows = self._projector.retrieve(query_text=query_document.text, query_vector=query_vector)
        documents = self._documents(rows, query_document.generation_id, proof)
        hits = tuple(
            NativeVectorBranchHit(
                rank=index,
                passage_id=document.passage_id,
                dependency_root_id=document.dependency_root_id,
                source_revision_id=document.revision_id,
                document_digest=document.digest,
                rights_digest=document.rights_digest,
                provenance_digest=document.provenance_digest,
                raw_score_ppm=max(0, min(1_000_000, int(round(score * 1_000_000)))),
            )
            for index, (document, score) in enumerate(documents, 1)
        )

    def fulltext_authority_view(
        self,
        receipts: tuple[NativeDocumentReceipt, ...],
        snapshot: FullTextProjectionSnapshot,
        *,
        proof: AuthenticationProof,
    ) -> FullTextAuthorityView:
        """Build the existing full-text authority view from reverified documents."""
        if type(receipts) is not tuple or not receipts or len(receipts) > 4_096 or type(snapshot) is not FullTextProjectionSnapshot:
            raise NativeRetrievalError("native full-text authority inventory differs")
        documents = tuple(self._read(receipt, proof)[0] for receipt in receipts)
        if (
            len({document.passage_id for document in documents}) != len(documents)
            or any(document.generation_id != str(snapshot.generation_id) for document in documents)
            or snapshot.index_document_count != len(documents)
            or snapshot.document_label != getattr(self._projector, "document_label", None)
            or snapshot.index_name != getattr(self._projector, "fulltext_index", None)
        ):
            raise NativeRetrievalError("native full-text snapshot differs")
        return FullTextAuthorityView(
            snapshot=snapshot,
            authority_aliases=(),
            document_bindings=tuple(sorted((
                FullTextDocumentBinding(
                    passage_id=document.passage_id,
                    dependency_root_id=document.dependency_root_id,
                    source_id=document.source_id,
                    source_identity=document.revision_id,
                    provenance_digest=document.provenance_digest,
                    language=document.language,
                    rights_current=True,
                    lifecycle="ACTIVE",
                )
                for document in documents
            ), key=lambda item: item.passage_id)),
        )
        generation_digest = digest_bytes(canonical_json_bytes({
            "generation_id": query_document.generation_id,
            "document_command_definition": self._command_definition,
            "embedding_model_digest": query_document.embedding_model_digest,
        }))
        semantic = canonical_json_bytes({
            "request_digest": request.request_digest,
            "generation_digest": generation_digest,
            "hits": [hit.canonical_value() for hit in hits],
        })
        return NativeVectorBranchReceipt(
            receipt_id=str(uuid.uuid5(uuid.NAMESPACE_URL, digest_bytes(semantic))),
            request_digest=request.request_digest,
            mode=BranchMode.VECTOR,
            outcome=BranchOutcome.COMPLETE,
            reason=None if hits else "NO_MATCH",
            generation_id=query_document.generation_id,
            generation_digest=generation_digest,
            profile_id=NATIVE_VECTOR_PROFILE,
            query_valid_time=request.query_valid_time,
            serving_time=request.serving_time,
            hits=hits,
            authority_read_count=1 + len(documents),
        )

    def _read(self, receipt: NativeDocumentReceipt, proof: AuthenticationProof) -> tuple[NativePassageDocument, tuple[float, ...]]:
        self._verify_event(receipt, proof)
        hydrated = self._objects.hydrate(HydrationRequest(receipt.admission_id, NATIVE_DOCUMENT_USE), proof=proof)
        self._access(hydrated.decision, self._document_policy, NATIVE_DOCUMENT_CLASS, NATIVE_DOCUMENT_USE)
        document = NativePassageDocument.from_bytes(hydrated.data)
        if document.digest != receipt.document_digest or document.vector_admission_id != str(receipt.vector_admission_id) or document.embedding_receipt_admission_id != str(receipt.embedding_receipt_admission_id):
            raise NativeRetrievalError("native document receipt differs")
        vector_object = self._objects.hydrate(HydrationRequest(receipt.vector_admission_id, NATIVE_VECTOR_USE), proof=proof)
        self._access(vector_object.decision, self._vector_policy, NATIVE_VECTOR_CLASS, NATIVE_VECTOR_USE)
        vector = _vector(vector_object.data)
        if digest_bytes(vector_object.data) != document.vector_digest:
            raise NativeRetrievalError("native document vector differs")
        receipt_object = self._objects.hydrate(HydrationRequest(receipt.embedding_receipt_admission_id, NATIVE_EMBEDDING_RECEIPT_USE), proof=proof)
        self._access(receipt_object.decision, self._receipt_policy, NATIVE_EMBEDDING_RECEIPT_CLASS, NATIVE_EMBEDDING_RECEIPT_USE)
        embedding = NativeEmbeddingReceipt.from_bytes(receipt_object.data)
        if digest_bytes(receipt_object.data) != document.embedding_receipt_digest or embedding.vector_digest != document.vector_digest or embedding.input_text_digest != document.text_digest:
            raise NativeRetrievalError("native embedding provenance differs")
        return document, vector

    def _verify_event(self, receipt: NativeDocumentReceipt, proof: AuthenticationProof) -> None:
        provenance = self._events.provenance(receipt.event_id, proof=proof)
        event = provenance.event
        if provenance.command_definition.command_type != NATIVE_DOCUMENT_COMMAND or provenance.command_definition.definition_digest != self._command_definition or event.command_definition_digest != self._command_definition or event.event_type != NATIVE_DOCUMENT_EVENT or event.object_admission_id != str(receipt.admission_id) or event.payload_digest != receipt.document_digest or event.command_id != receipt.command_id or event.aggregate_id != str(receipt.aggregate_id) or event.aggregate_version != receipt.aggregate_version or event.principal_id != self._controller or provenance.authentication.principal_id != self._controller or provenance.authentication.authority_domain != self._domain or event.trust_scope != TrustScope.ADMITTED.value or event.security_scope != NATIVE_SECURITY_SCOPE or event.retention_scope != NATIVE_RETENTION_SCOPE:
            raise NativeRetrievalError("native retrieval authority event differs")

    def _receipt_for_event(self, event_id: str, proof: AuthenticationProof) -> NativeDocumentReceipt:
        provenance = self._events.provenance(event_id, proof=proof)
        event = provenance.event
        if event.object_admission_id is None:
            raise NativeRetrievalError("native vector query event lacks an object")
        admission_id = ObjectAdmissionId.parse(event.object_admission_id)
        hydrated = self._objects.hydrate(
            HydrationRequest(admission_id, NATIVE_DOCUMENT_USE), proof=proof
        )
        self._access(hydrated.decision, self._document_policy, NATIVE_DOCUMENT_CLASS, NATIVE_DOCUMENT_USE)
        document = NativePassageDocument.from_bytes(hydrated.data)
        return NativeDocumentReceipt(
            event.event_id,
            event.command_id,
            AggregateId.parse(event.aggregate_id),
            event.aggregate_version,
            admission_id,
            document.digest,
            ObjectAdmissionId.parse(document.vector_admission_id),
            ObjectAdmissionId.parse(document.embedding_receipt_admission_id),
        )

    def _access(self, decision: Any, policy: str, object_class: str, allowed_use: str) -> None:
        if decision.policy_contract_digest != policy or decision.principal_id != self._reader or decision.authority_domain != self._domain or decision.object_class != object_class or decision.allowed_use != allowed_use:
            raise NativeRetrievalError("native retrieval object access differs")

    def _hits(self, rows: tuple[Mapping[str, object], ...], generation_id: str, proof: AuthenticationProof) -> tuple[NativeRetrievalHit, ...]:
        return tuple(
            NativeRetrievalHit(document.passage_id, document.dependency_root_id, score)
            for document, score in self._documents(rows, generation_id, proof)
        )

    def _documents(self, rows: tuple[Mapping[str, object], ...], generation_id: str, proof: AuthenticationProof) -> tuple[tuple[NativePassageDocument, float], ...]:
        if len(rows) > NATIVE_RESULT_LIMIT:
            raise NativeRetrievalHold("RESULT_LIMIT_EXCEEDED")
        result: list[tuple[NativePassageDocument, float]] = []
        seen: set[str] = set()
        for row in rows:
            values = dict(row)
            score = values.pop("score", None)
            receipt = NativeDocumentReceipt.from_projection(values)
            document, _ = self._read(receipt, proof)
            if (
                document.generation_id != generation_id
                or document.passage_id in seen
                or type(score) not in (float, int)
                or isinstance(score, bool)
            ):
                raise NativeRetrievalError("projection hit lacks admitted authority")
            seen.add(document.passage_id)
            result.append((document, float(score)))
        return tuple(result)


def _vector(raw: bytes) -> tuple[float, ...]:
    if type(raw) is not bytes or len(raw) != 4 * NATIVE_VECTOR_DIMENSIONS:
        raise NativeRetrievalHold("EMBEDDING_VECTOR_MISSING")
    values = struct.unpack(f">{NATIVE_VECTOR_DIMENSIONS}f", raw)
    if not all(math.isfinite(value) for value in values) or not any(value != 0.0 for value in values):
        raise NativeRetrievalHold("EMBEDDING_VECTOR_INVALID")
    return values


__all__ = [name for name in globals() if name.startswith("Native") or name.startswith("NATIVE_")]
