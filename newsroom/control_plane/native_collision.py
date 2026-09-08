"""Live Story Candidate collision requests for the native Hermes runtime."""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path

from newsroom.authority import AuthenticationProof
from newsroom.authority.canonical import canonical_json_bytes, digest_bytes, validate_sha256_digest
from newsroom.increment5.named_tool_contracts import NamedToolId
from newsroom.increment6.collision import (
    CandidateUseCollisionBinding,
    CandidateUseOperation,
    CollisionState,
    CurrentCollisionAuthoritySnapshot,
    CurrentCollisionEffectEnforcer,
    CurrentCollisionEligibilityRequest,
    NativeCurrentCollisionReceiptEvidence,
    TrustedCurrentCollisionAuthorityBoundary,
    TrustedCurrentCollisionAuthorityContext,
)
from newsroom.increment6.work_items import RetrievalInputBinding

from .native_triage import NativeTriageResult


_PROFILE_ID = "hermes-native-story-candidate-collision-v1"
_PORT_ID = "hermes.native.story-candidate-collision.v1"
_QUERY = (
    "SELECT b.candidate_id,h.current_version_id,h.current_version_digest,"
    "b.semantic_scope_digest FROM story_candidate_collision_bindings b "
    "JOIN story_candidate_heads h ON h.candidate_id=b.candidate_id "
    "WHERE b.collision_namespace=? AND b.collision_key_digest=?"
)
_QUERY_DIGEST = digest_bytes(_QUERY.encode("utf-8"))
_ADAPTER_CONFIG_DIGEST = digest_bytes(canonical_json_bytes({
    "schema_version": "newsroom.control-plane.native-collision-adapter.v1",
    "query_digest": _QUERY_DIGEST,
    "tables": ["ledger_events", "story_candidate_collision_bindings", "story_candidate_heads"],
    "read_only": True,
}))
_PORT_REGISTRY_DIGEST = digest_bytes(canonical_json_bytes({
    "schema_version": "newsroom.control-plane.native-collision-port-registry.v1",
    "ports": [_PORT_ID],
}))


class NativeCollisionHold(RuntimeError):
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


@dataclass(frozen=True, slots=True)
class NativeCollisionIdentity:
    authority_scope_id: str
    actor_id: str
    authenticated_principal_digest: str
    authorization_receipt_digest: str
    authorization_decision_id: str
    retrieval_adapter_contract_digest: str
    retrieval_adapter_config_digest: str

    def __post_init__(self) -> None:
        for value in (
            self.authenticated_principal_digest,
            self.authorization_receipt_digest,
            self.retrieval_adapter_contract_digest,
            self.retrieval_adapter_config_digest,
        ):
            validate_sha256_digest(value)
        if not self.authority_scope_id or not self.actor_id:
            raise ValueError("native collision authority identities are required")


class NativeCollisionAuthority:
    """Produce exact requests and fresh, retained read-only authority snapshots."""

    def __init__(
        self,
        *,
        authority_path: Path,
        journal_path: Path,
        identity: NativeCollisionIdentity,
    ) -> None:
        if not isinstance(authority_path, Path) or not isinstance(journal_path, Path):
            raise TypeError("native collision paths must be Paths")
        if authority_path.resolve() == journal_path.resolve():
            raise ValueError("native collision journal must be separate from authority")
        if not isinstance(identity, NativeCollisionIdentity):
            raise TypeError("native collision identity must be typed")
        self._path = authority_path
        self._journal = journal_path
        self._identity = identity
        self._lock = threading.Lock()
        journal_path.parent.mkdir(parents=True, exist_ok=True)
        with self._journal_connection() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS native_collision_receipts("
                "receipt_digest TEXT PRIMARY KEY,request_digest TEXT NOT NULL,"
                "authority_watermark INTEGER NOT NULL,collision_state TEXT NOT NULL,"
                "candidate_id TEXT,execution_receipt_bytes BLOB NOT NULL,"
                "authority_receipt_bytes BLOB NOT NULL) STRICT"
            )
        self.enforcer = CurrentCollisionEffectEnforcer(
            current_authority_provider=self,
            trusted_boundary=TrustedCurrentCollisionAuthorityBoundary(
                identity.authority_scope_id,
                _PROFILE_ID,
                _ADAPTER_CONFIG_DIGEST,
                _PORT_REGISTRY_DIGEST,
                _PORT_ID,
            ),
        )

    def request(
        self,
        triage: NativeTriageResult,
        retrieval: RetrievalInputBinding,
        *,
        proof: AuthenticationProof,
    ) -> CurrentCollisionEligibilityRequest:
        del proof
        if type(triage) is not NativeTriageResult or triage.hypothesis is None:
            raise TypeError("native collision requires a retained Hypothesis")
        if type(retrieval) is not RetrievalInputBinding or retrieval.receipt_bytes is None:
            raise TypeError("native collision requires a retained retrieval receipt")
        if digest_bytes(retrieval.receipt_bytes) != retrieval.context_digest:
            raise ValueError("native collision retrieval receipt differs")
        try:
            receipt = json.loads(retrieval.receipt_bytes)
            authority = receipt["authority_evidence"]
            manifest = next(
                item for item in receipt["projection_evidence"]
                if item["tool_id"]
                == NamedToolId.CURRENT_COLLISION_AND_AUTHORITY_HYDRATION_LOOKUP.value
            )
        except (KeyError, StopIteration, TypeError, ValueError) as exc:
            raise NativeCollisionHold(
                "RETRIEVAL_COLLISION_AUTHORITY_INCOMPLETE"
            ) from exc
        if (
            receipt.get("actor_id") != self._identity.actor_id
            or receipt.get("authenticated_principal_digest")
            != self._identity.authenticated_principal_digest
            or not isinstance(authority, dict)
            or not isinstance(manifest, dict)
            or authority.get("outcome") != "COMPLETE"
            or authority.get("authority_scope_id")
            != self._identity.authority_scope_id
            or authority.get("adapter_contract_digest")
            != self._identity.retrieval_adapter_contract_digest
            or authority.get("adapter_config_digest")
            != self._identity.retrieval_adapter_config_digest
        ):
            raise NativeCollisionHold("RETRIEVAL_COLLISION_AUTHORITY_DIFFERS")
        generation_id = manifest.get("generation_id") or (
            "native-collision-" + retrieval.context_digest.removeprefix("sha256:")[:24]
        )
        watermark, candidate_id, _, _, _ = self._read(
            authority["collision_namespace"], authority["collision_key_digest"]
        )
        binding = CandidateUseCollisionBinding(
            triage.hypothesis.hypothesis_id,
            triage.hypothesis.version_id,
            triage.hypothesis.canonical_digest,
            CandidateUseOperation.ADMIT_NEW_CANDIDATE
            if candidate_id is None else CandidateUseOperation.USE_CURRENT_CANDIDATE,
            candidate_id,
            authority["collision_namespace"],
            authority["collision_key_digest"],
            generation_id,
            receipt["query_valid_time"],
            authority["serving_time"],
            watermark,
        )
        request_digest = digest_bytes(canonical_json_bytes({
            "schema_version": "newsroom.increment6.native-collision-request.v1",
            "actor_id": self._identity.actor_id,
            "authenticated_principal_digest": self._identity.authenticated_principal_digest,
            "binding": binding.canonical_value(),
        }))
        return CurrentCollisionEligibilityRequest(binding, request_digest)

    def __call__(
        self, request: CurrentCollisionEligibilityRequest
    ) -> CurrentCollisionAuthoritySnapshot:
        if type(request) is not CurrentCollisionEligibilityRequest:
            raise TypeError("native collision request must be exact typed")
        expected = digest_bytes(canonical_json_bytes({
            "schema_version": "newsroom.increment6.native-collision-request.v1",
            "actor_id": self._identity.actor_id,
            "authenticated_principal_digest": self._identity.authenticated_principal_digest,
            "binding": request.binding.canonical_value(),
        }))
        if request.named_request_digest != expected:
            raise NativeCollisionHold("COLLISION_REQUEST_IDENTITY_DIFFERS")
        (
            watermark,
            candidate_id,
            candidate_version_id,
            candidate_version_digest,
            candidate_semantic_scope_digest,
        ) = self._read(
            request.binding.collision_namespace,
            request.binding.collision_key_digest,
        )
        state = CollisionState.UNOCCUPIED if candidate_id is None else CollisionState.OCCUPIED
        context = TrustedCurrentCollisionAuthorityContext(
            request.binding.generation_id,
            watermark,
            request.binding.query_valid_time,
            request.binding.serving_time,
            self._identity.authority_scope_id,
            _PROFILE_ID,
            _ADAPTER_CONFIG_DIGEST,
            self._identity.authorization_receipt_digest,
            self._identity.authorization_decision_id,
            _PORT_REGISTRY_DIGEST,
            _PORT_ID,
        )
        authority_bytes = canonical_json_bytes({
            "schema_version": "newsroom.increment6.native-collision-authority.v1",
            "request_digest": request.named_request_digest,
            "authority_scope_id": context.authority_scope_id,
            "authority_profile_id": context.authority_profile_id,
            "adapter_config_digest": context.adapter_config_digest,
            "generation_id": context.generation_id,
            "authority_watermark": watermark,
            "query_valid_time": context.query_valid_time,
            "serving_time": context.serving_time,
            "collision_namespace": request.binding.collision_namespace,
            "collision_key_digest": request.binding.collision_key_digest,
            "collision_state": state.value,
            "candidate_id": candidate_id,
            "candidate_version_id": candidate_version_id,
            "candidate_version_digest": candidate_version_digest,
            "candidate_semantic_scope_digest": candidate_semantic_scope_digest,
            "subject_id": request.binding.subject_id,
            "subject_version_id": request.binding.subject_version_id,
            "subject_version_digest": request.binding.subject_version_digest,
            "outcome": "COMPLETE",
        })
        execution_bytes = canonical_json_bytes({
            "schema_version": "newsroom.increment6.native-collision-execution.v1",
            "request_digest": request.named_request_digest,
            "authority_receipt_digest": digest_bytes(authority_bytes),
            "authorization_receipt_digest": context.authorization_receipt_digest,
            "authorization_decision_id": context.authorization_decision_id,
            "port_registry_digest": context.port_registry_digest,
            "port_id": context.port_id,
            "generation_id": context.generation_id,
            "authority_watermark": watermark,
            "query_valid_time": context.query_valid_time,
            "serving_time": context.serving_time,
            "outcome": "COMPLETE",
        })
        evidence = NativeCurrentCollisionReceiptEvidence(
            request.named_request_digest, execution_bytes, authority_bytes
        )
        self._retain(evidence, watermark, state, candidate_id)
        return CurrentCollisionAuthoritySnapshot(evidence, context)

    def _read(
        self, namespace: str | None = None, collision_key: str | None = None
    ) -> tuple[
        int,
        str | None,
        str | None,
        str | None,
        str | None,
    ]:
        uri = f"file:{self._path.resolve()}?mode=ro"
        try:
            with sqlite3.connect(uri, uri=True, isolation_level=None) as connection:
                connection.execute("BEGIN")
                watermark = int(connection.execute(
                    "SELECT COALESCE(MAX(ledger_seq),0) FROM ledger_events"
                ).fetchone()[0])
                if namespace is None:
                    connection.execute("COMMIT")
                    return watermark, None, None, None, None
                rows = connection.execute(_QUERY, (namespace, collision_key)).fetchall()
                connection.execute("COMMIT")
        except sqlite3.Error as exc:
            raise NativeCollisionHold("CURRENT_COLLISION_AUTHORITY_UNAVAILABLE") from exc
        if len(rows) > 1:
            raise NativeCollisionHold("CURRENT_COLLISION_AUTHORITY_AMBIGUOUS")
        if not rows:
            return watermark, None, None, None, None
        row = rows[0]
        return watermark, str(row[0]), str(row[1]), str(row[2]), str(row[3])

    def _journal_connection(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._journal, isolation_level=None)
        connection.execute("PRAGMA busy_timeout=5000")
        return connection

    def _retain(
        self,
        evidence: NativeCurrentCollisionReceiptEvidence,
        watermark: int,
        state: CollisionState,
        candidate_id: str | None,
    ) -> None:
        receipt_digest = digest_bytes(canonical_json_bytes({
            "execution": evidence.execution_receipt_digest,
            "authority": evidence.authority_receipt_digest,
        }))
        with self._lock, self._journal_connection() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO native_collision_receipts VALUES(?,?,?,?,?,?,?)",
                (
                    receipt_digest,
                    evidence.request_digest,
                    watermark,
                    state.value,
                    candidate_id,
                    evidence.execution_receipt_bytes,
                    evidence.authority_receipt_bytes,
                ),
            )


__all__ = [
    "NativeCollisionAuthority",
    "NativeCollisionHold",
    "NativeCollisionIdentity",
]
