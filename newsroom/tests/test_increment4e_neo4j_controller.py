from __future__ import annotations

import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest

import newsroom.authority._increment4_neo4j_boundary as increment4_boundary_module
from newsroom.authority import AggregateId
from newsroom.increment4 import (
    Increment4Neo4jActiveReadRequest,
    Increment4Neo4jBuildRequest,
    Increment4Neo4jCurrentBuildRequest,
    build_increment4_admitted_batches,
    increment4_admitted_contract_registry,
)
from newsroom.projection import (
    ProjectionGenerationId,
    ProjectionGenerationState,
    ProjectionStateError,
)
from newsroom.projection.neo4j import (
    Neo4jIdentityConflict,
    Neo4jWriteError,
    StructuralActiveReconciliationRequest,
)

from .extraction_4a_helpers import extraction_proof
from .increment4e_helpers import (
    INCREMENT4_PROJECTION_SCOPES,
    admitted_increment4_fixture,
    open_increment4_neo4j_system,
)
from .authority_helpers import command as authority_command
from .projection_b2_helpers import MemoryNeo4jAdapter
from .source_3a_helpers import SOURCE_NOW


GENERATION_1 = ProjectionGenerationId.parse(
    "00000000-0000-4000-8000-000000004991"
)
GENERATION_2 = ProjectionGenerationId.parse(
    "00000000-0000-4000-8000-000000004992"
)


def _request(generation_id: ProjectionGenerationId, snapshot, *, key: str):
    return Increment4Neo4jBuildRequest(
        generation_id=generation_id,
        snapshot=snapshot,
        reason_code="INCREMENT4_ACTUAL_NEO4J_PROOF",
        idempotency_key=key,
    )


def _current_request(generation_id: ProjectionGenerationId, *, key: str):
    return Increment4Neo4jCurrentBuildRequest(
        generation_id=generation_id,
        reason_code="INCREMENT4_CURRENT_AUTHORITY_REBUILD",
        idempotency_key=key,
    )


def _entity_canonical_ids(adapter: MemoryNeo4jAdapter, generation_id):
    return tuple(
        sorted(
            {
                node.canonical_id
                for (stored_generation, _sequence), batch in adapter.deliveries.items()
                if stored_generation == str(generation_id)
                for node in batch.nodes
                if node.identity_source == "CANONICAL_ENTITY_ID"
            }
        )
    )


class _FailRetiredCleanupOnceAdapter(MemoryNeo4jAdapter):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.fail_cleanup_generation: str | None = None
        self.failed_cleanup = False

    def cleanup_generation(self, generation_id: str) -> int:
        if (
            generation_id == self.fail_cleanup_generation
            and not self.failed_cleanup
        ):
            self.cleanup_count += 1
            self.failed_cleanup = True
            raise Neo4jWriteError("fixed retired-generation cleanup failure")
        return super().cleanup_generation(generation_id)


def test_increment4_current_build_rederives_complete_admitted_authority(
    tmp_path: Path,
) -> None:
    state, snapshot = admitted_increment4_fixture(tmp_path)
    adapter = MemoryNeo4jAdapter()
    family = increment4_admitted_contract_registry().family(
        "graph.increment4.admitted"
    )
    expected = build_increment4_admitted_batches(
        snapshot,
        generation_id=GENERATION_1,
        family=family,
    )

    with open_increment4_neo4j_system(state, adapter) as system:
        result = system.increment4.build_current_and_promote(
            _current_request(
                GENERATION_1,
                key="increment4-current-authority-build-v1",
            ),
            proof=extraction_proof(),
        )
        reconciliation = system.increment4.reconcile_active(
            proof=extraction_proof()
        )
        system.commands.execute(
            authority_command(
                key="increment4-bounded-reconcile-source-advance-v1",
                aggregate_id=AggregateId.parse(
                    "00000000-0000-4000-8000-000000005101"
                ),
            ),
            proof=extraction_proof(),
        )
        after_source_advance = system.increment4.reconcile_active(
            proof=extraction_proof()
        )
        advanced_status = system.increment4.generation_status(
            GENERATION_1,
            proof=extraction_proof(),
        )

    actual = tuple(
        batch
        for (generation, _sequence), batch in sorted(adapter.deliveries.items())
        if generation == str(GENERATION_1)
    )
    assert result.generation.state is ProjectionGenerationState.ACTIVE
    assert result.source_watermark_ledger_seq == snapshot.through_ledger_seq
    assert reconciliation.generation_id == result.generation.generation_id
    assert reconciliation.checkpoint_ledger_seq == result.checkpoint_ledger_seq
    assert (
        reconciliation.projection_state_digest
        == result.projection_state_digest
    )
    assert after_source_advance == reconciliation
    assert advanced_status.source_watermark_ledger_seq > result.checkpoint_ledger_seq
    assert tuple(item.batch_digest for item in actual) == tuple(
        item.batch_digest for item in expected
    )


def test_increment4_current_build_failure_keeps_prior_generation_active(
    tmp_path: Path,
) -> None:
    state, _snapshot = admitted_increment4_fixture(tmp_path)
    adapter = MemoryNeo4jAdapter()

    with open_increment4_neo4j_system(state, adapter) as system:
        first = system.increment4.build_current_and_promote(
            _current_request(
                GENERATION_1,
                key="increment4-current-prior-v1",
            ),
            proof=extraction_proof(),
        )
        adapter.reconciliation_mismatch = True
        with pytest.raises(Neo4jIdentityConflict):
            system.increment4.build_current_and_promote(
                _current_request(
                    GENERATION_2,
                    key="increment4-current-failed-replacement-v1",
                ),
                proof=extraction_proof(),
            )
        first_status = system.increment4.generation_status(
            GENERATION_1,
            proof=extraction_proof(),
        )
        second_status = system.increment4.generation_status(
            GENERATION_2,
            proof=extraction_proof(),
        )

    assert first.generation.state is ProjectionGenerationState.ACTIVE
    assert first_status.generation.state is ProjectionGenerationState.ACTIVE
    assert second_status.generation.state is ProjectionGenerationState.VALIDATING


def test_increment4_controller_builds_validates_promotes_and_reads_active(
    tmp_path: Path,
) -> None:
    state, snapshot = admitted_increment4_fixture(tmp_path)
    adapter = MemoryNeo4jAdapter()
    request = _request(GENERATION_1, snapshot, key="increment4-build-v1")

    with open_increment4_neo4j_system(state, adapter) as system:
        result = system.increment4.build_and_promote(
            request,
            proof=extraction_proof(),
        )
        status = system.increment4.generation_status(
            GENERATION_1, proof=extraction_proof()
        )
        canonical_ids = _entity_canonical_ids(adapter, GENERATION_1)
        response = system.increment4.read_active(
            Increment4Neo4jActiveReadRequest(
                canonical_ids=canonical_ids,
                query_valid_time=SOURCE_NOW,
                limit=100,
            ),
            proof=extraction_proof(),
        )
        apply_before_changed_retry = adapter.apply_count
        cleanup_before_changed_retry = adapter.cleanup_count
        reconcile_before_changed_retry = adapter.reconcile_count
        with pytest.raises(
            ProjectionStateError,
            match="immutable build intent",
        ):
            system.increment4.build_and_promote(
                replace(request, purge_retired_generation=False),
                proof=extraction_proof(),
            )
        assert adapter.apply_count == apply_before_changed_retry
        assert adapter.cleanup_count == cleanup_before_changed_retry
        assert adapter.reconcile_count == reconcile_before_changed_retry

    assert result.generation.state is ProjectionGenerationState.ACTIVE
    assert result.generation.generation_id == GENERATION_1
    assert result.prior_generation is None
    assert result.checkpoint_ledger_seq >= snapshot.through_ledger_seq
    assert result.source_watermark_ledger_seq == snapshot.through_ledger_seq
    assert result.projected_batch_count == len(adapter.deliveries)
    assert result.ignored_optional_count == (
        snapshot.through_ledger_seq - result.projected_batch_count
    )
    conn = sqlite3.connect(state.entity.extraction.database)
    try:
        recorded = int(
            conn.execute(
                "SELECT COUNT(*) FROM projection_delivery_states "
                "WHERE generation_id=?",
                (str(GENERATION_1),),
            ).fetchone()[0]
        )
    finally:
        conn.close()
    assert result.projected_batch_count <= recorded <= result.projected_batch_count + 1
    assert recorded < snapshot.through_ledger_seq
    assert result.validation.projection_state_digest == result.projection_state_digest
    assert result.promotion.generation.generation_id == GENERATION_1
    assert status.generation == result.generation
    assert status.contiguous_ledger_seq == result.checkpoint_ledger_seq
    assert status.source_watermark_ledger_seq == snapshot.through_ledger_seq
    assert canonical_ids
    assert response.metadata.generation_id == GENERATION_1
    assert response.metadata.generation_state is ProjectionGenerationState.ACTIVE
    assert response.nodes
    assert response.relations
    assert all(item.trust_scope.value == "ADMITTED" for item in response.relations)


def test_increment4_replacement_retires_and_purges_prior_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state, snapshot = admitted_increment4_fixture(tmp_path)
    adapter = _FailRetiredCleanupOnceAdapter()
    boundary_type = increment4_boundary_module._Increment4Neo4jBoundary
    current_create_generation = boundary_type._create_generation

    def create_legacy_generation(
        self,
        *,
        request,
        snapshot_digest,
        proof,
        legacy_identity=False,
    ):
        return current_create_generation(
            self,
            request=request,
            snapshot_digest=snapshot_digest,
            proof=proof,
            legacy_identity=True,
        )

    # Seed the exact parent-release identity before exercising post-upgrade
    # ACTIVE replay and pending predecessor cleanup.
    monkeypatch.setattr(
        boundary_type,
        "_create_generation",
        create_legacy_generation,
    )
    replacement_request = _request(
        GENERATION_2,
        snapshot,
        key="increment4-retry-retired-cleanup-v1",
    )

    with open_increment4_neo4j_system(state, adapter) as system:
        first = system.increment4.build_and_promote(
            _request(
                GENERATION_1,
                snapshot,
                key="increment4-retry-retired-cleanup-prior-v1",
            ),
            proof=extraction_proof(),
        )
        adapter.fail_cleanup_generation = str(GENERATION_1)
        with pytest.raises(
            Neo4jWriteError,
            match="retired-generation cleanup failure",
        ):
            system.increment4.build_and_promote(
                replacement_request,
                proof=extraction_proof(),
            )
        monkeypatch.setattr(
            boundary_type,
            "_create_generation",
            current_create_generation,
        )
        first_status = system.increment4.generation_status(
            GENERATION_1, proof=extraction_proof()
        )
        second_status = system.increment4.generation_status(
            GENERATION_2, proof=extraction_proof()
        )
        assert first_status.generation.state is ProjectionGenerationState.RETIRED
        assert second_status.generation.state is ProjectionGenerationState.ACTIVE
        assert any(key[0] == str(GENERATION_1) for key in adapter.deliveries)
        serving_before_retry = {
            key: value
            for key, value in adapter.deliveries.items()
            if key[0] == str(GENERATION_2)
        }
        apply_before_retry = adapter.apply_count
        cleanup_before_retry = adapter.cleanup_count
        reconcile_before_retry = adapter.reconcile_count

        # Cleanup intent is part of the immutable creation-command identity.
        # A retry cannot attach to this ACTIVE generation while changing the
        # original request from purge=True to purge=False.
        with pytest.raises(
            ProjectionStateError,
            match="immutable build intent",
        ):
            system.increment4.build_and_promote(
                replace(
                    replacement_request,
                    purge_retired_generation=False,
                ),
                proof=extraction_proof(),
            )
        assert adapter.apply_count == apply_before_retry
        assert adapter.cleanup_count == cleanup_before_retry
        assert adapter.reconcile_count == reconcile_before_retry
        assert any(key[0] == str(GENERATION_1) for key in adapter.deliveries)
        assert {
            key: value
            for key, value in adapter.deliveries.items()
            if key[0] == str(GENERATION_2)
        } == serving_before_retry

        system.commands.execute(
            authority_command(
                key="increment4-retired-cleanup-source-advance-v1",
                aggregate_id=AggregateId.parse(
                    "00000000-0000-4000-8000-000000005100"
                ),
            ),
            proof=extraction_proof(),
        )

        with pytest.raises(
            ProjectionStateError,
            match="differs from exact retained admitted authority",
        ):
            system.increment4.build_and_promote(
                replacement_request,
                proof=extraction_proof(),
            )
        first_after = system.increment4.generation_status(
            GENERATION_1, proof=extraction_proof()
        )
        second_after = system.increment4.generation_status(
            GENERATION_2, proof=extraction_proof()
        )

    assert first.generation.state is ProjectionGenerationState.ACTIVE
    assert first_after.generation.state is ProjectionGenerationState.RETIRED
    assert second_after.generation.state is ProjectionGenerationState.ACTIVE
    assert second_after.source_watermark_ledger_seq > snapshot.through_ledger_seq
    assert adapter.apply_count == apply_before_retry
    assert adapter.cleanup_count == cleanup_before_retry + 1
    assert adapter.reconcile_count == reconcile_before_retry
    assert not any(key[0] == str(GENERATION_1) for key in adapter.deliveries)
    assert {
        key: value
        for key, value in adapter.deliveries.items()
        if key[0] == str(GENERATION_2)
    } == serving_before_retry


def test_increment4_exact_active_replay_is_non_mutating_and_graph_loss_fails_closed(
    tmp_path: Path,
) -> None:
    state, snapshot = admitted_increment4_fixture(tmp_path)
    adapter = _SourceRaceAdapter()
    request = _request(GENERATION_1, snapshot, key="increment4-replay-v1")

    with open_increment4_neo4j_system(state, adapter) as system:
        first = system.increment4.build_and_promote(
            request, proof=extraction_proof()
        )
        before_apply = adapter.apply_count
        before_cleanup = adapter.cleanup_count
        before_reconcile = adapter.reconcile_count
        replay = system.increment4.build_and_promote(
            request, proof=extraction_proof()
        )
        assert replay.promotion.promotion_digest == first.promotion.promotion_digest
        assert replay.validation.validation_digest == first.validation.validation_digest
        assert replay.deleted_target_graph_record_count == 0
        assert replay.purged_retired_graph_record_count == 0
        assert adapter.apply_count == before_apply
        assert adapter.cleanup_count == before_cleanup
        assert adapter.reconcile_count == before_reconcile + 1

        serving_deliveries = dict(adapter.deliveries)
        serving_markers = dict(adapter._delivery_markers)
        adapter.deliveries.clear()
        failed_apply = adapter.apply_count
        failed_cleanup = adapter.cleanup_count
        with pytest.raises(Neo4jIdentityConflict):
            system.increment4.build_and_promote(
                request, proof=extraction_proof()
            )
        assert adapter.apply_count == failed_apply
        assert adapter.cleanup_count == failed_cleanup
        assert adapter.deliveries == {}

        adapter.deliveries.update(serving_deliveries)
        adapter._delivery_markers.update(serving_markers)
        source_apply = adapter.apply_count
        source_cleanup = adapter.cleanup_count
        source_reconcile = adapter.reconcile_count
        adapter.before_first_reconcile = lambda: system.commands.execute(
            authority_command(
                key="increment4-active-source-race-authority-v1",
                aggregate_id=AggregateId.parse(
                    "00000000-0000-4000-8000-000000005099"
                ),
            ),
            proof=extraction_proof(),
        )
        with pytest.raises(
            ProjectionStateError,
            match="differs from exact retained admitted authority",
        ):
            system.increment4.build_and_promote(
                request, proof=extraction_proof()
            )
        status = system.increment4.generation_status(
            GENERATION_1, proof=extraction_proof()
        )

    assert status.generation.state is ProjectionGenerationState.ACTIVE
    assert status.source_watermark_ledger_seq > snapshot.through_ledger_seq
    assert adapter.apply_count == source_apply
    assert adapter.cleanup_count == source_cleanup
    assert adapter.reconcile_count == source_reconcile + 1


def test_increment4_graph_loss_recovers_only_through_isolated_replacement(
    tmp_path: Path,
) -> None:
    state, snapshot = admitted_increment4_fixture(tmp_path)
    adapter = MemoryNeo4jAdapter()

    with open_increment4_neo4j_system(state, adapter) as system:
        first = system.increment4.build_and_promote(
            _request(GENERATION_1, snapshot, key="increment4-loss-first-v1"),
            proof=extraction_proof(),
        )
        adapter.deliveries.clear()
        replacement = system.increment4.build_and_promote(
            _request(GENERATION_2, snapshot, key="increment4-loss-replacement-v1"),
            proof=extraction_proof(),
        )
        first_status = system.increment4.generation_status(
            GENERATION_1, proof=extraction_proof()
        )

    assert first.generation.state is ProjectionGenerationState.ACTIVE
    assert replacement.generation.state is ProjectionGenerationState.ACTIVE
    assert replacement.prior_generation is not None
    assert replacement.prior_generation.generation_id == GENERATION_1
    assert first_status.generation.state is ProjectionGenerationState.RETIRED
    assert any(key[0] == str(GENERATION_2) for key in adapter.deliveries)
    assert not any(key[0] == str(GENERATION_1) for key in adapter.deliveries)


def test_increment4_reconciliation_mismatch_never_promotes(tmp_path: Path) -> None:
    state, snapshot = admitted_increment4_fixture(tmp_path)
    adapter = MemoryNeo4jAdapter(reconciliation_mismatch=True)

    with open_increment4_neo4j_system(state, adapter) as system:
        with pytest.raises(Neo4jIdentityConflict):
            system.increment4.build_and_promote(
                _request(GENERATION_1, snapshot, key="increment4-mismatch-v1"),
                proof=extraction_proof(),
            )
        status = system.increment4.generation_status(
            GENERATION_1, proof=extraction_proof()
        )

    assert status.generation.state is ProjectionGenerationState.VALIDATING


class _SourceRaceAdapter(MemoryNeo4jAdapter):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.before_first_reconcile = None
        self._injected = False

    def reconcile_generation(self, *, generation_id, expected_batches):
        if not self._injected and self.before_first_reconcile is not None:
            self._injected = True
            self.before_first_reconcile()
        return super().reconcile_generation(
            generation_id=generation_id,
            expected_batches=expected_batches,
        )


def test_increment4_source_watermark_change_fails_atomic_validation(
    tmp_path: Path,
) -> None:
    state, snapshot = admitted_increment4_fixture(tmp_path)
    adapter = _SourceRaceAdapter()

    with open_increment4_neo4j_system(state, adapter) as system:
        adapter.before_first_reconcile = lambda: system.commands.execute(
            authority_command(
                key="increment4-source-race-v1",
                aggregate_id=AggregateId.parse(
                    "00000000-0000-4000-8000-000000004998"
                ),
            ),
            proof=extraction_proof(),
        )
        with pytest.raises(
            ProjectionStateError,
            match="source watermark changed before authority commit",
        ):
            system.increment4.build_and_promote(
                _request(GENERATION_1, snapshot, key="increment4-race-v1"),
                proof=extraction_proof(),
            )
        status = system.increment4.generation_status(
            GENERATION_1, proof=extraction_proof()
        )

    assert status.generation.state is ProjectionGenerationState.VALIDATING
    assert status.source_watermark_ledger_seq > snapshot.through_ledger_seq


def test_increment4_generation_status_and_read_require_read_scope(
    tmp_path: Path,
) -> None:
    state, snapshot = admitted_increment4_fixture(tmp_path)
    adapter = MemoryNeo4jAdapter()
    scopes = INCREMENT4_PROJECTION_SCOPES - {"authority.projection.read"}

    with open_increment4_neo4j_system(state, adapter, scopes=scopes) as system:
        result = system.increment4.build_and_promote(
            _request(GENERATION_1, snapshot, key="increment4-no-read-v1"),
            proof=extraction_proof(),
        )
        with pytest.raises(PermissionError):
            system.increment4.generation_status(
                GENERATION_1, proof=extraction_proof()
            )
        with pytest.raises(PermissionError):
            system.increment4.read_active(
                Increment4Neo4jActiveReadRequest(
                    canonical_ids=_entity_canonical_ids(adapter, GENERATION_1),
                    query_valid_time=SOURCE_NOW,
                    limit=100,
                ),
                proof=extraction_proof(),
            )

    assert result.generation.state is ProjectionGenerationState.ACTIVE


def test_increment4_family_rejects_generic_structural_reconciliation(
    tmp_path: Path,
) -> None:
    state, snapshot = admitted_increment4_fixture(tmp_path)
    adapter = MemoryNeo4jAdapter()

    with open_increment4_neo4j_system(state, adapter) as system:
        system.increment4.build_and_promote(
            _request(GENERATION_1, snapshot, key="increment4-bounded-v1"),
            proof=extraction_proof(),
        )
        with pytest.raises(
            ProjectionStateError,
            match="requires its bounded controller",
        ):
            system.structural.reconcile_active(
                StructuralActiveReconciliationRequest(
                    family_id="graph.increment4.admitted"
                ),
                proof=extraction_proof(),
            )


def test_increment4_bounded_reconciliation_rejects_graph_drift(
    tmp_path: Path,
) -> None:
    state, snapshot = admitted_increment4_fixture(tmp_path)
    adapter = MemoryNeo4jAdapter()

    with open_increment4_neo4j_system(state, adapter) as system:
        system.increment4.build_and_promote(
            _request(GENERATION_1, snapshot, key="increment4-reconcile-v1"),
            proof=extraction_proof(),
        )
        adapter.reconciliation_mismatch = True
        with pytest.raises(
            Neo4jIdentityConflict,
            match="differs from retained authority",
        ):
            system.increment4.reconcile_active(proof=extraction_proof())


def test_increment4_expected_batches_remain_exact_after_controller_build(
    tmp_path: Path,
) -> None:
    state, snapshot = admitted_increment4_fixture(tmp_path)
    adapter = MemoryNeo4jAdapter()
    family = increment4_admitted_contract_registry().family(
        "graph.increment4.admitted"
    )
    expected = build_increment4_admitted_batches(
        snapshot,
        generation_id=GENERATION_1,
        family=family,
    )

    with open_increment4_neo4j_system(state, adapter) as system:
        system.increment4.build_and_promote(
            _request(GENERATION_1, snapshot, key="increment4-exact-batches-v1"),
            proof=extraction_proof(),
        )

    actual = tuple(
        batch
        for (generation, _sequence), batch in sorted(adapter.deliveries.items())
        if generation == str(GENERATION_1)
    )
    assert tuple(item.batch_digest for item in actual) == tuple(
        item.batch_digest for item in expected
    )
