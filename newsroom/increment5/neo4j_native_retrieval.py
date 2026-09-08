"""Fixed Neo4j projection/read adapter for native Increment 5 documents."""

from __future__ import annotations

import hashlib
import re
from typing import Any, Mapping

from .native_retrieval import (
    NATIVE_RESULT_LIMIT,
    NATIVE_VECTOR_DIMENSIONS,
    NativeDocumentReceipt,
    NativePassageDocument,
    NativeRetrievalError,
)

_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,127}\Z")
class Neo4jNativeRetrievalProjection:
    """Own all Cypher for one configured native retrieval generation."""

    def __init__(self, driver: Any, *, database: str | None, generation_id: str, fulltext_index: str, vector_index: str) -> None:
        if driver is None or not callable(getattr(driver, "session", None)):
            raise TypeError("native retrieval requires a Neo4j driver")
        if any(_NAME.fullmatch(item) is None for item in (fulltext_index, vector_index)) or fulltext_index == vector_index:
            raise NativeRetrievalError("native retrieval index names differ")
        if type(generation_id) is not str or not generation_id or generation_id != generation_id.strip():
            raise NativeRetrievalError("native retrieval generation differs")
        suffix = hashlib.sha256(generation_id.encode()).hexdigest()[:16]
        self._driver = driver
        self._database = database
        self._generation = generation_id
        self._label = f"NewsroomNativeRetrievalDocument_{suffix}"
        self._constraint = f"native_retrieval_passage_{suffix}"
        self._fulltext = fulltext_index
        self._vector = vector_index

    @property
    def document_label(self) -> str:
        return self._label

    @property
    def fulltext_index(self) -> str:
        return self._fulltext

    def bootstrap(self) -> None:
        statements = (
            f"CREATE CONSTRAINT `{self._constraint}` IF NOT EXISTS FOR (n:`{self._label}`) REQUIRE n.passage_id IS UNIQUE",
            f"CREATE FULLTEXT INDEX `{self._fulltext}` IF NOT EXISTS FOR (n:`{self._label}`) ON EACH [n.retrieval_text] OPTIONS {{indexConfig: {{`fulltext.analyzer`: 'standard-no-stop-words', `fulltext.eventually_consistent`: false}}}}",
            f"CREATE VECTOR INDEX `{self._vector}` IF NOT EXISTS FOR (n:`{self._label}`) ON n.embedding OPTIONS {{indexConfig: {{`vector.dimensions`: {NATIVE_VECTOR_DIMENSIONS}, `vector.similarity_function`: 'cosine', `vector.quantization.type`: 'none'}}}}",
        )
        with self._session("WRITE") as session:
            for statement in statements:
                session.execute_write(lambda transaction, query=statement: transaction.run(query).consume())

    def upsert(self, receipt: NativeDocumentReceipt, document: NativePassageDocument, vector: tuple[float, ...]) -> None:
        if type(receipt) is not NativeDocumentReceipt or type(document) is not NativePassageDocument or len(vector) != NATIVE_VECTOR_DIMENSIONS or document.generation_id != self._generation:
            raise NativeRetrievalError("native projection input differs")
        query = f"""
MERGE (n:`{self._label}` {{passage_id:$passage_id}})
ON CREATE SET n.dependency_root_id=$dependency_root_id, n.source_id=$source_id,
 n.revision_id=$revision_id, n.representation_id=$representation_id,
 n.language=$language, n.retrieval_text=$retrieval_text,
 n.text_digest=$text_digest, n.rights_digest=$rights_digest,
 n.provenance_digest=$provenance_digest, n.vector_digest=$vector_digest,
 n.vector_admission_id=$vector_admission_id,
 n.embedding_receipt_digest=$embedding_receipt_digest,
 n.embedding_receipt_admission_id=$embedding_receipt_admission_id,
 n.embedding_model_digest=$embedding_model_digest, n.embedding=$embedding,
 n.event_id=$event_id, n.command_id=$command_id, n.aggregate_id=$aggregate_id,
 n.aggregate_version=$aggregate_version, n.admission_id=$admission_id,
 n.document_digest=$document_digest
ON MATCH SET n.passage_id=n.passage_id
RETURN n.text_digest AS text_digest,n.vector_digest AS vector_digest,
 n.event_id AS event_id,n.command_id AS command_id,
 n.aggregate_id AS aggregate_id,n.aggregate_version AS aggregate_version,
 n.admission_id AS admission_id,n.document_digest AS document_digest,
 n.vector_admission_id AS vector_admission_id,
 n.embedding_receipt_admission_id AS embedding_receipt_admission_id
""".strip()
        parameters = {
            **{key: value for key, value in document.projection_value().items() if key != "text"},
            **receipt.projection_value(),
            "retrieval_text": document.text,
            "embedding": list(vector),
        }
        with self._session("WRITE") as session:
            rows = tuple(session.execute_write(lambda transaction: transaction.run(query, **parameters)))
        expected = {
            "text_digest": document.text_digest,
            "vector_digest": document.vector_digest,
            **receipt.projection_value(),
        }
        if len(rows) != 1 or any(rows[0][key] != value for key, value in expected.items()):
            raise NativeRetrievalError("native projection acknowledgement differs")

    def retrieve(self, *, query_text: str, query_vector: tuple[float, ...]) -> tuple[tuple[Mapping[str, object], ...], tuple[Mapping[str, object], ...]]:
        if type(query_text) is not str or not query_text or len(query_vector) != NATIVE_VECTOR_DIMENSIONS:
            raise NativeRetrievalError("native retrieval query differs")
        limit = NATIVE_RESULT_LIMIT + 1
        receipt = "node.event_id AS event_id,node.command_id AS command_id,node.aggregate_id AS aggregate_id,node.aggregate_version AS aggregate_version,node.admission_id AS admission_id,node.document_digest AS document_digest,node.vector_admission_id AS vector_admission_id,node.embedding_receipt_admission_id AS embedding_receipt_admission_id"
        fulltext = f"""
CALL db.index.fulltext.queryNodes($index_name,$query_text,{{limit:$limit}}) YIELD node,score
RETURN {receipt},score ORDER BY score DESC,node.passage_id LIMIT $limit
""".strip()
        vector = f"""
CALL db.index.vector.queryNodes($index_name,$limit,$vector) YIELD node,score
RETURN {receipt},score ORDER BY score DESC,node.passage_id LIMIT $limit
""".strip()
        common = {"limit": limit}
        with self._session("READ") as session:
            fulltext_rows = tuple(session.execute_read(lambda transaction: tuple(transaction.run(fulltext, **common, index_name=self._fulltext, query_text=query_text))))
            vector_rows = tuple(session.execute_read(lambda transaction: tuple(transaction.run(vector, **common, index_name=self._vector, vector=list(query_vector)))))
        return fulltext_rows, vector_rows

    def _session(self, mode: str):
        values: dict[str, object] = {"default_access_mode": mode}
        if self._database is not None:
            values["database"] = self._database
        return self._driver.session(**values)


__all__ = ["Neo4jNativeRetrievalProjection"]
