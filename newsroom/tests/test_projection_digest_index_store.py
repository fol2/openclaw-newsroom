"""Exercise digest resolution during real SQLite builds and retained-store opens."""
from __future__ import annotations

import json
import time
import warnings

import pytest

from newsroom.authority import AggregateId
from newsroom.authority import canonical
from newsroom.authority._projection_store import _ProjectionAuthorityStore
from newsroom.increment4 import Increment4Neo4jCurrentBuildRequest
from newsroom.projection import ProjectionGenerationId, ProjectionGenerationState
from newsroom.projection.mapping import StructuralMappingContract, StructuralMappingRegistry
from newsroom.projection.models import ProjectionContractError, ProjectionFamilyDefinition
from newsroom.projection.ontology import OntologyContract, OntologyRegistry
from newsroom.projection.registry import ProjectionFamilyRegistry

from .authority_helpers import command as authority_command
from .extraction_4a_helpers import extraction_proof
from .increment4e_helpers import admitted_increment4_fixture, open_increment4_neo4j_system
from .projection_b2_helpers import MemoryNeo4jAdapter


@pytest.mark.parametrize("extra_events", (0, 4000))
def test_reopen_checks_retained_deliveries_without_rehashing_registry(
    monkeypatch, tmp_path, extra_events,
):
    state, _snapshot = admitted_increment4_fixture(tmp_path)
    adapter = MemoryNeo4jAdapter()
    generation_id = ProjectionGenerationId.parse(
        "00000000-0000-4000-8000-000000009934"
    )
    started = time.perf_counter()
    with open_increment4_neo4j_system(state, adapter) as system:
        seed_started = time.perf_counter()
        for seq in range(extra_events):
            system.commands.execute(
                authority_command(
                    key=f"digest-index-history-{seq}",
                    aggregate_id=AggregateId.parse(
                        f"00000000-0000-4000-8000-{100000 + seq:012d}"
                    ),
                ),
                proof=extraction_proof(),
            )
        seed_s = time.perf_counter() - seed_started
        build_started = time.perf_counter()
        result = system.increment4.build_current_and_promote(
            Increment4Neo4jCurrentBuildRequest(
                generation_id=generation_id,
                reason_code="DIGEST_INDEX_REOPEN",
                idempotency_key="digest-index-reopen",
            ),
            proof=extraction_proof(),
        )
        build_s = time.perf_counter() - build_started
        assert result.generation.state is ProjectionGenerationState.ACTIVE
        assert result.source_watermark_ledger_seq >= extra_events

    # Recreate only the old lookup/ASCII-validation implementations over the
    # same real store. This isolates the changed costs, not a cold-machine run
    # or a complete checkout of the base revision.
    baseline_open_s = None
    if extra_events:
        def old_string_check(value, path):
            for character in value:
                if 0xD800 <= ord(character) <= 0xDFFF:
                    raise canonical.CanonicalizationError(
                        f"lone surrogate is unsupported at {path}"
                    )

        def legacy_resolver(attribute, message):
            def resolve(self, digest):
                matches = [item for item in self._by_key.values() if getattr(item, attribute) == digest]
                if len(matches) != 1:
                    raise ProjectionContractError(message)
                return matches[0]
            return resolve

        with monkeypatch.context() as scope:
            scope.setattr(canonical, "_validate_string", old_string_check)
            for cls, attribute, message in (
                (ProjectionFamilyRegistry, "digest", "unknown or ambiguous family definition digest"),
                (StructuralMappingRegistry, "contract_digest", "unknown or ambiguous mapping digest"),
                (OntologyRegistry, "contract_digest", "unknown or ambiguous ontology digest"),
            ):
                scope.setattr(cls, "resolve_digest", legacy_resolver(attribute, message))
            baseline_started = time.perf_counter()
            with open_increment4_neo4j_system(state, adapter):
                baseline_open_s = time.perf_counter() - baseline_started

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
    reopen_started = time.perf_counter()
    with open_increment4_neo4j_system(state, adapter) as system:
        fixed_open_s = time.perf_counter() - reopen_started
        status = system.increment4.generation_status(generation_id, proof=extraction_proof())
        assert status.generation.state is ProjectionGenerationState.ACTIVE
        assert system.increment4.reconcile_active(
            proof=extraction_proof()
        ).projection_state_digest == result.projection_state_digest
    assert checked_rows and sum(checked_rows) > 0
    if extra_events:
        # One compact observable measurement in existing pytest CI logs. No
        # unstable wall-clock assertion; cost bounds above are deterministic.
        warnings.warn(
            "PROJECTION_STORE_SCALE_EVIDENCE " + json.dumps({
                "additional_source_events": extra_events,
                "materialised_watermark": result.source_watermark_ledger_seq,
                "retained_delivery_states": sum(checked_rows),
                "seed_s": round(seed_s, 6),
                "build_s": round(build_s, 6),
                "baseline_lookup_and_ascii_open_s": round(baseline_open_s, 6),
                "fixed_open_s": round(fixed_open_s, 6),
                "test_total_s": round(time.perf_counter() - started, 6),
                "graph_adapter": "memory",
                "scope": "real SQLite projection authority; not full operational sealer",
            }, sort_keys=True),
            pytest.PytestWarning,
            stacklevel=1,
        )
