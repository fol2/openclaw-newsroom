from __future__ import annotations

import struct
import uuid

import pytest

from newsroom.authority.canonical import digest_bytes
from newsroom.authority.types import AggregateId, ObjectAdmissionId
from newsroom.increment5.native_retrieval import (
    NATIVE_VECTOR_DIMENSIONS,
    NativeEmbeddingReceipt,
    NativeDocumentReceipt,
    NativePassageDocument,
    NativeRetrievalDocuments,
    NativeRetrievalError,
    NativeRetrievalHold,
    NativeVectorRequest,
    NativeVectorBranchHit,
    NativeVectorBranchReceipt,
    _vector,
)
from newsroom.increment5.branch_contracts import BranchMode, BranchOutcome
from newsroom.increment5.neo4j_native_retrieval import Neo4jNativeRetrievalProjection


def _digest(character: str) -> str:
    return "sha256:" + character * 64


def _document() -> NativePassageDocument:
    text = "Hong Kong authority retained this exact passage."
    return NativePassageDocument(
        generation_id="native-generation-1",
        passage_id="00000000-0000-4000-8000-000000000001",
        dependency_root_id="event:one",
        source_id="source.one",
        revision_id="revision.one",
        representation_id="representation.one",
        language="en-GB",
        text=text,
        text_digest=digest_bytes(text.encode()),
        rights_digest=_digest("a"),
        provenance_digest=_digest("b"),
        vector_digest=_digest("c"),
        vector_admission_id="00000000-0000-4000-8000-000000000002",
        embedding_receipt_digest=_digest("d"),
        embedding_receipt_admission_id="00000000-0000-4000-8000-000000000003",
        embedding_model_digest=_digest("e"),
    )


def _receipt() -> NativeDocumentReceipt:
    return NativeDocumentReceipt(
        "event-one",
        "command-one",
        AggregateId.new(),
        1,
        ObjectAdmissionId.new(),
        _document().digest,
        ObjectAdmissionId.new(),
        ObjectAdmissionId.new(),
    )


class _Result(tuple):
    def consume(self):
        return None


class _Transaction:
    def __init__(self, calls):
        self.calls = calls

    def run(self, query, **parameters):
        self.calls.append((query, parameters))
        if "RETURN n.text_digest" in query:
            return _Result(({key: parameters[key] for key in (
                "text_digest", "vector_digest", "event_id", "command_id",
                "aggregate_id", "aggregate_version", "admission_id",
                "document_digest", "vector_admission_id",
                "embedding_receipt_admission_id",
            )},))
        if "fulltext.queryNodes" in query:
            return _Result(({**self.receipt, "score": 2.0},))
        if "vector.queryNodes" in query:
            return _Result(({**self.receipt, "score": 0.75},))
        return _Result()


class _Session:
    def __init__(self, calls, receipt): self.calls, self.receipt = calls, receipt
    def __enter__(self): return self
    def __exit__(self, *_): return None
    def execute_write(self, work):
        transaction = _Transaction(self.calls)
        transaction.receipt = self.receipt
        return work(transaction)
    def execute_read(self, work):
        transaction = _Transaction(self.calls)
        transaction.receipt = self.receipt
        return work(transaction)


class _Driver:
    def __init__(self, receipt): self.calls = []; self.sessions = []; self.receipt = receipt
    def session(self, **config):
        self.sessions.append(config)
        return _Session(self.calls, self.receipt)


def test_native_projection_executes_real_fulltext_and_vector_queries() -> None:
    receipt = _receipt()
    driver = _Driver(receipt.projection_value())
    projection = Neo4jNativeRetrievalProjection(
        driver,
        database="neo4j",
        generation_id="native-generation-1",
        fulltext_index="native_fulltext_1",
        vector_index="native_vector_1",
    )
    vector = (1.0,) + (0.0,) * (NATIVE_VECTOR_DIMENSIONS - 1)
    document = _document()

    projection.bootstrap()
    projection.upsert(receipt, document, vector)
    fulltext, vector_hits = projection.retrieve(
        query_text="Hong Kong",
        query_vector=vector,
    )

    assert fulltext == ({**receipt.projection_value(), "score": 2.0},)
    assert vector_hits == ({**receipt.projection_value(), "score": 0.75},)
    queries = "\n".join(item[0] for item in driver.calls)
    assert "CREATE FULLTEXT INDEX" in queries
    assert "CREATE VECTOR INDEX" in queries
    assert "db.index.fulltext.queryNodes" in queries
    assert "db.index.vector.queryNodes" in queries
    assert "NewsroomNativeRetrievalDocument_" in queries
    assert {item["default_access_mode"] for item in driver.sessions} == {"READ", "WRITE"}


def test_embedding_receipt_and_vector_are_exact_non_fixture_inputs() -> None:
    raw = struct.pack(f">{NATIVE_VECTOR_DIMENSIONS}f", 1.0, *((0.0,) * (NATIVE_VECTOR_DIMENSIONS - 1)))
    receipt = NativeEmbeddingReceipt(
        input_text_digest=_digest("1"),
        vector_digest=digest_bytes(raw),
        dimensions=NATIVE_VECTOR_DIMENSIONS,
        provider="openai",
        model="text-embedding-3-large",
        model_digest=_digest("2"),
        provider_request_id="provider-request-1",
        usage_receipt_digest=_digest("3"),
        recorded_at="2026-09-08T12:00:00.000000Z",
    )

    assert NativeEmbeddingReceipt.from_bytes(receipt.canonical_bytes) == receipt
    assert _vector(raw)[0] == 1.0
    with pytest.raises(NativeRetrievalHold, match="EMBEDDING_VECTOR_MISSING"):
        _vector(b"fixture-id-is-not-a-vector")


def test_projection_hits_without_admitted_document_are_rejected() -> None:
    with pytest.raises(NativeRetrievalError, match="projection receipt differs"):
        NativeDocumentReceipt.from_projection(
            {"passage_id": "private-workspace-node"}
        )


def test_native_vector_request_and_receipt_are_distinct_from_fixture_contract() -> None:
    event_id = "native-event-1"
    request = NativeVectorRequest(
        request_id=str(uuid.uuid4()),
        idempotency_key="native-vector-one",
        query_event_id=event_id,
        query_document_digest=_digest("3"),
        generation_id="native-generation-1",
        query_valid_time="2026-09-08T12:00:00Z",
        serving_time="2026-09-08T12:00:01Z",
    )
    hit = NativeVectorBranchHit(
        1, "passage-one", "root-one", "revision-one", _digest("4"),
        _digest("5"), _digest("6"), 750_000,
    )
    receipt = NativeVectorBranchReceipt(
        str(uuid.uuid4()), request.request_digest, BranchMode.VECTOR,
        BranchOutcome.COMPLETE, None, "native-generation-1", _digest("7"),
        "native-governed-vector-v1", request.query_valid_time,
        request.serving_time, (hit,), 2,
    )

    assert request.request_digest == digest_bytes(request.canonical_bytes)
    assert NativeVectorBranchReceipt.from_canonical_bytes(receipt.canonical_bytes) == receipt
