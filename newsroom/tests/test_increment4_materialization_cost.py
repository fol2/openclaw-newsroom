"""Keep materialisation hashing linear in the full retained snapshot.

Only graph/store I/O is replaced. Requests, snapshots, ledger records, delivery
requests, key derivation and the materialisation loop use production code.
"""
from __future__ import annotations

from dataclasses import fields, replace
from threading import RLock
from types import SimpleNamespace

import pytest

from newsroom.authority._increment4_neo4j_boundary import _Increment4Neo4jBoundary
from newsroom.authority.auth import AuthenticationProof
from newsroom.authority.canonical import CanonicalizationError
from newsroom.authority.persistence import LedgerEventRecord
from newsroom.increment4 import models as snapshot_models
from newsroom.increment4.models import Increment4AdmittedProjectionSnapshot
from newsroom.increment4.neo4j import Increment4Neo4jBuildRequest
from newsroom.projection.models import (
    ProjectionDeliveryOutcome,
    ProjectionGenerationId,
    ProjectionGenerationState,
    ProjectionStateError,
)


class _MemoryIO:
    def __init__(self, count: int) -> None:
        self.count = count
        self.states: dict[int, object] = {}
        self.deliveries: list[object] = []
        self.fail_at: int | None = None
        self.metadata = SimpleNamespace(
            generation=SimpleNamespace(
                state=ProjectionGenerationState.BUILDING,
                authority_aggregate_version=1,
            ),
            contiguous_ledger_seq=count,
            open_gap_count=0,
            dead_letter_count=0,
        )

    def cleanup_generation(self, generation_id: str) -> int:
        return 0

    def projection_rebuild_delivery_state(self, generation_id, ledger_seq):
        return self.states.get(ledger_seq)

    def projection_generation_metadata(self, generation_id):
        return self.metadata

    def projection_delivery_source(self, generation_id, ledger_seq):
        return SimpleNamespace(
            event=SimpleNamespace(event_id=str(ledger_seq)),
            source_event_digest=f"retained-{ledger_seq}",
        )

    def record_delivery(self, request, proof):
        if request.ledger_seq == self.fail_at:
            raise RuntimeError("injected delivery failure")
        self.deliveries.append(request)
        self.states[request.ledger_seq] = SimpleNamespace(
            finalized=True,
            outcome=request.outcome,
            source_event_id=str(request.ledger_seq),
            source_event_digest=f"retained-{request.ledger_seq}",
        )
        self.metadata.generation.authority_aggregate_version += 1


def _request(count: int) -> Increment4Neo4jBuildRequest:
    events = []
    for seq in range(1, count + 1):
        values = {field.name: "fixture" for field in fields(LedgerEventRecord)}
        for name in values:
            if "digest" in name:
                values[name] = "sha256:" + "a" * 64
            elif name.endswith("_id"):
                values[name] = f"00000000-0000-4000-8000-{seq:012d}"
        values.update(
            ledger_seq=seq, event_schema_version=1, aggregate_version=1,
            event_type="projection.delivery.recorded",
            aggregate_type="projection_generation",
            recorded_at="2026-09-01T00:00:00.000000Z", payload_mode="INLINE",
            object_admission_id=None, correlation_id=None, causation_kind=None,
            causation_identifier=None, causation_external_system=None,
            security_scope="authority.projection", retention_scope="authority.audit",
            trust_scope="ADMITTED", principal_id="fixture",
        )
        events.append(LedgerEventRecord(**values))
    return Increment4Neo4jBuildRequest(
        generation_id=ProjectionGenerationId.parse(
            "00000000-0000-4000-8000-000000999999"
        ),
        snapshot=Increment4AdmittedProjectionSnapshot(
            entities=(), relations=(), events=tuple(events), through_ledger_seq=count,
        ),
        reason_code="MATERIALIZATION_COST",
        idempotency_key="materialization-cost",
        purge_retired_generation=False,
    )


def _boundary(io: _MemoryIO) -> _Increment4Neo4jBoundary:
    return _Increment4Neo4jBoundary(
        store=io, projection_boundary=io, structural_reader=io, adapter=io,
        clock=lambda: None, operation_lock=RLock(),
    )


def _run(boundary, request):
    return boundary._materialize_generation(
        request=request, batches=(), source_watermark=request.snapshot.through_ledger_seq,
        proof=AuthenticationProof(method="STATIC_TOKEN", credential="fixture-only"),
    )


@pytest.mark.parametrize("count", [1, 8, 32])
def test_materialization_hashes_full_snapshot_once_and_preserves_delivery_keys(
    monkeypatch: pytest.MonkeyPatch, count: int,
) -> None:
    request = _request(count)
    expected_digest = request.snapshot.canonical_digest
    original = snapshot_models._event_digest
    visits: list[int] = []

    def counted(event):
        visits.append(event.ledger_seq)
        return original(event)

    monkeypatch.setattr(snapshot_models, "_event_digest", counted)
    io = _MemoryIO(count)
    boundary = _boundary(io)
    assert _run(boundary, request) == (0, 0, count)
    assert visits == list(range(1, count + 1))
    assert len(io.deliveries) == count
    for seq, delivery in enumerate(io.deliveries, start=1):
        assert delivery.ledger_seq == seq
        assert delivery.expected_authority_version == seq
        assert delivery.outcome is ProjectionDeliveryOutcome.IGNORED_OPTIONAL
        assert delivery.idempotency_key == boundary._operation_key(
            request.idempotency_key, "delivery", {
                "generation_id": str(request.generation_id), "ledger_seq": seq,
                "snapshot_digest": expected_digest, "outcome": "IGNORED_OPTIONAL",
            },
        )


def test_materialization_digest_is_not_reused_across_invocations(monkeypatch):
    request = _request(4)
    original = snapshot_models._event_digest
    visits = []

    def counted(event):
        visits.append(event.ledger_seq)
        return original(event)

    monkeypatch.setattr(snapshot_models, "_event_digest", counted)
    io = _MemoryIO(4)
    boundary = _boundary(io)
    io.fail_at = 3
    with pytest.raises(RuntimeError, match="injected delivery failure"):
        _run(boundary, request)
    assert len(io.deliveries) == 2
    io.fail_at = None
    assert _run(boundary, request) == (0, 0, 4)
    assert visits == [1, 2, 3, 4] * 2
    assert [item.ledger_seq for item in io.deliveries] == [1, 2, 3, 4]
    # A completely finalised replay neither rehashes nor records new deliveries.
    assert _run(boundary, request) == (0, 0, 4)
    assert visits == [1, 2, 3, 4] * 2
    assert len(io.deliveries) == 4
    io.states[1].source_event_digest = "drifted"
    with pytest.raises(ProjectionStateError, match="provenance changed"):
        _run(boundary, request)


def test_changed_snapshot_gets_new_keys_on_same_controller():
    request = _request(3)
    io = _MemoryIO(3)
    boundary = _boundary(io)
    _run(boundary, request)
    prior_keys = [item.idempotency_key for item in io.deliveries]
    changed = replace(request, snapshot=replace(
        request.snapshot,
        events=(replace(request.snapshot.events[0], principal_id="changed"),
                *request.snapshot.events[1:]),
    ))
    io.states.clear()
    io.deliveries.clear()
    io.metadata.generation.authority_aggregate_version = 1
    _run(boundary, changed)
    assert all(a != b.idempotency_key for a, b in zip(prior_keys, io.deliveries, strict=True))


@pytest.mark.parametrize("field,value,message", [
    ("contiguous_ledger_seq", 0, "complete source watermark"),
    ("open_gap_count", 1, "gaps or dead letters"),
    ("dead_letter_count", 1, "gaps or dead letters"),
])
def test_materialization_final_guards_remain_closed(field, value, message):
    io = _MemoryIO(3)
    setattr(io.metadata, field, value)
    with pytest.raises(ProjectionStateError, match=message):
        _run(_boundary(io), _request(3))


def test_materialization_rejects_nonbuilding_before_hashing(monkeypatch):
    request = _request(3)
    io = _MemoryIO(3)
    io.metadata.generation.state = ProjectionGenerationState.ACTIVE

    def unexpected(_event):
        raise AssertionError("snapshot must not be hashed before the state guard")

    monkeypatch.setattr(snapshot_models, "_event_digest", unexpected)
    with pytest.raises(ProjectionStateError, match="BUILDING generation"):
        _run(_boundary(io), request)
    assert not io.deliveries


def test_materialization_does_not_bypass_canonical_validation():
    request = _request(3)
    invalid = replace(request, snapshot=replace(
        request.snapshot,
        events=(replace(request.snapshot.events[0], principal_id="\ud800"),
                *request.snapshot.events[1:]),
    ))
    io = _MemoryIO(3)
    with pytest.raises(CanonicalizationError, match="lone surrogate"):
        _run(_boundary(io), invalid)
    assert not io.deliveries
