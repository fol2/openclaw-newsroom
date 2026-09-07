"""Bound snapshot hashing at the real Increment 4 materialisation consumer."""

from __future__ import annotations

from pathlib import Path

import pytest

import newsroom.authority._increment4_neo4j_boundary as increment4_boundary_module
import newsroom.increment4.models as snapshot_models
from newsroom.increment4 import (
    Increment4Neo4jBuildRequest,
    Increment4Neo4jCurrentBuildRequest,
)
from newsroom.projection import ProjectionGenerationId, ProjectionGenerationState

from .extraction_4a_helpers import extraction_proof
from .increment4e_helpers import (
    admitted_increment4_fixture,
    open_increment4_neo4j_system,
)
from .projection_b2_helpers import MemoryNeo4jAdapter


@pytest.mark.parametrize("current_build", (False, True))
def test_increment4_materialization_hashes_immutable_snapshot_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    current_build: bool,
) -> None:
    """Do not rehash all historical events for each new delivery (#895)."""
    state, snapshot = admitted_increment4_fixture(tmp_path)
    adapter = MemoryNeo4jAdapter()
    generation_id = ProjectionGenerationId.parse(
        "00000000-0000-4000-8000-000000004991"
    )
    boundary_type = increment4_boundary_module._Increment4Neo4jBoundary
    materialize = boundary_type._materialize_generation
    snapshot_type = snapshot_models.Increment4AdmittedProjectionSnapshot
    digest_property = snapshot_type.canonical_digest
    event_digest = snapshot_models._event_digest
    observed: list[tuple[int, int, int]] = []

    def measured_materialize(self, **kwargs):
        snapshot_calls = 0
        event_calls = 0

        def counted_snapshot(value):
            nonlocal snapshot_calls
            snapshot_calls += 1
            return digest_property.fget(value)

        def counted_event(value):
            nonlocal event_calls
            event_calls += 1
            return event_digest(value)

        # Count only inside the real consumer, not the legitimate current-source
        # and promotion checks around it. Existing fixtures use no live service.
        with monkeypatch.context() as scope:
            scope.setattr(snapshot_type, "canonical_digest", property(counted_snapshot))
            scope.setattr(snapshot_models, "_event_digest", counted_event)
            result = materialize(self, **kwargs)
        observed.append(
            (snapshot_calls, event_calls, len(kwargs["request"].snapshot.events))
        )
        return result

    monkeypatch.setattr(boundary_type, "_materialize_generation", measured_materialize)
    with open_increment4_neo4j_system(state, adapter) as system:
        if current_build:
            result = system.increment4.build_current_and_promote(
                Increment4Neo4jCurrentBuildRequest(
                    generation_id=generation_id,
                    reason_code="SNAPSHOT_DIGEST_COST_REGRESSION",
                    idempotency_key="increment4-digest-cost-current",
                ),
                proof=extraction_proof(),
            )
        else:
            result = system.increment4.build_and_promote(
                Increment4Neo4jBuildRequest(
                    generation_id=generation_id,
                    snapshot=snapshot,
                    reason_code="SNAPSHOT_DIGEST_COST_REGRESSION",
                    idempotency_key="increment4-digest-cost-explicit",
                ),
                proof=extraction_proof(),
            )
        reconciliation = system.increment4.reconcile_active(proof=extraction_proof())

    assert len(observed) == 1
    snapshot_calls, event_calls, history_size = observed[0]
    assert history_size > 1
    assert snapshot_calls == 1
    assert event_calls == history_size
    assert result.generation.state is ProjectionGenerationState.ACTIVE
    assert result.checkpoint_ledger_seq >= result.source_watermark_ledger_seq
    assert result.projected_batch_count == len(adapter.deliveries)
    assert result.ignored_optional_count == (
        result.source_watermark_ledger_seq - result.projected_batch_count
    )
    assert reconciliation.projection_state_digest == result.projection_state_digest
