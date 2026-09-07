"""Exercise indexed contract resolution during a real SQLite store reopen."""
from __future__ import annotations

import pytest

from newsroom.authority._projection_store import _ProjectionAuthorityStore
from newsroom.increment4 import Increment4Neo4jBuildRequest
from newsroom.projection import ProjectionGenerationId, ProjectionGenerationState
from newsroom.projection.mapping import StructuralMappingContract
from newsroom.projection.models import ProjectionFamilyDefinition
from newsroom.projection.ontology import OntologyContract

from .extraction_4a_helpers import extraction_proof
from .increment4e_helpers import admitted_increment4_fixture, open_increment4_neo4j_system
from .projection_b2_helpers import MemoryNeo4jAdapter


def test_reopen_checks_retained_deliveries_without_rehashing_registry(monkeypatch, tmp_path):
    state, snapshot = admitted_increment4_fixture(tmp_path)
    adapter = MemoryNeo4jAdapter()
    generation_id = ProjectionGenerationId.parse(
        "00000000-0000-4000-8000-000000009934"
    )
    with open_increment4_neo4j_system(state, adapter) as system:
        result = system.increment4.build_and_promote(
            Increment4Neo4jBuildRequest(
                generation_id=generation_id,
                snapshot=snapshot,
                reason_code="DIGEST_INDEX_REOPEN",
                idempotency_key="digest-index-reopen",
            ),
            proof=extraction_proof(),
        )
        assert result.generation.state is ProjectionGenerationState.ACTIVE

    original = _ProjectionAuthorityStore._validate_projection_delivery_rows
    checked_rows = []

    def measured_validation(self, conn):
        rows = conn.execute("SELECT COUNT(*) FROM projection_delivery_states").fetchone()[0]
        calls = []
        with monkeypatch.context() as scope:
            for cls, name in (
                (ProjectionFamilyDefinition, "digest"),
                (StructuralMappingContract, "contract_digest"),
                (OntologyContract, "contract_digest"),
            ):
                getter = getattr(cls, name).fget

                def counted(value, getter=getter, name=name):
                    calls.append(name)
                    return getter(value)

                scope.setattr(cls, name, property(counted))
            original(self, conn)
        checked_rows.append(rows)
        assert not calls, "per-record integrity must not rehash immutable contracts"

    monkeypatch.setattr(
        _ProjectionAuthorityStore, "_validate_projection_delivery_rows", measured_validation
    )
    with open_increment4_neo4j_system(state, adapter) as system:
        status = system.increment4.generation_status(generation_id, proof=extraction_proof())
        assert status.generation.state is ProjectionGenerationState.ACTIVE
        assert system.increment4.reconcile_active(
            proof=extraction_proof()
        ).projection_state_digest == result.projection_state_digest
    assert checked_rows and sum(checked_rows) > 0
