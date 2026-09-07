"""Digest resolution must not repeatedly canonicalise the registry snapshot."""
from __future__ import annotations

from dataclasses import replace

import pytest

from newsroom.authority.types import AggregateId
from newsroom.projection.mapping import (
    StructuralMappingContract,
    StructuralMappingRegistry,
    native_structural_mapping_v1,
)
from newsroom.projection.models import (
    ProjectionContractError,
    ProjectionFamilyDefinition,
    ProjectionFamilyKind,
)
from newsroom.projection.ontology import (
    OntologyContract,
    OntologyRegistry,
    native_ontology_v1,
)
from newsroom.projection.registry import ProjectionFamilyRegistry


def _case(kind: str, count: int = 3):
    ontology = native_ontology_v1()
    mapping = native_structural_mapping_v1(ontology)
    family = ProjectionFamilyDefinition(
        family_id="graph.digest_cost",
        authority_aggregate_id=AggregateId.parse(
            "00000000-0000-4000-8000-000000009933"
        ),
        family_kind=ProjectionFamilyKind.GRAPH,
        definition_version="v0",
        projector_version="projector-v1",
        ontology_contract_digest=ontology.contract_digest,
        mapping_contract_digest=mapping.contract_digest,
    )
    if kind == "ontology":
        values = tuple(replace(ontology, ontology_version=f"v{i}") for i in range(count))
        return values, OntologyContract, "contract_digest", lambda supplied: OntologyRegistry(
            supplied, current_versions={ontology.ontology_id: "v0"}
        )
    if kind == "mapping":
        values = tuple(replace(mapping, mapping_version=f"v{i}") for i in range(count))
        return values, StructuralMappingContract, "contract_digest", lambda supplied: StructuralMappingRegistry(
            supplied, current_versions={mapping.mapping_id: "v0"}
        )
    ontologies = OntologyRegistry((ontology,))
    mappings = StructuralMappingRegistry((mapping,))
    values = tuple(replace(family, definition_version=f"v{i}") for i in range(count))
    return values, ProjectionFamilyDefinition, "digest", lambda supplied: ProjectionFamilyRegistry(
        supplied, ontologies=ontologies, mappings=mappings,
        current_versions={family.family_id: "v0"},
    )


@pytest.mark.parametrize("kind", ("ontology", "mapping", "family"))
@pytest.mark.parametrize("count", (1, 4, 16))
def test_digest_lookup_does_not_rehash_registered_contracts(monkeypatch, kind, count):
    values, value_type, attribute, build = _case(kind, count)
    expected_digests = tuple(getattr(value, attribute) for value in values)
    canonical_before = tuple(value.canonical_value() for value in values)
    original = getattr(value_type, attribute)
    calls = 0

    def counted(value):
        nonlocal calls
        calls += 1
        return original.fget(value)

    monkeypatch.setattr(value_type, attribute, property(counted))
    registry = build(iter(values))
    construction_calls = calls
    for _ in range(16):
        for value, digest in zip(values, expected_digests, strict=True):
            assert registry.resolve_digest(digest) is value
        with pytest.raises(ProjectionContractError, match="unknown or ambiguous"):
            registry.resolve_digest("sha256:" + "0" * 64)
    assert calls == construction_calls
    assert tuple(value.canonical_value() for value in values) == canonical_before
    assert tuple(getattr(value, attribute) for value in values) == expected_digests


@pytest.mark.parametrize("kind", ("ontology", "mapping", "family"))
@pytest.mark.parametrize("unknown", (None, 1, [], {}, "", "sha256:" + "0" * 64))
def test_unknown_digest_keeps_typed_refusal(kind, unknown):
    values, _value_type, _attribute, build = _case(kind)
    with pytest.raises(ProjectionContractError, match="unknown or ambiguous"):
        build(values).resolve_digest(unknown)


@pytest.mark.parametrize("kind", ("ontology", "mapping", "family"))
def test_digest_collision_never_selects_first_or_last_contract(monkeypatch, kind):
    values, value_type, attribute, build = _case(kind)
    collision = "sha256:" + "a" * 64
    monkeypatch.setattr(value_type, attribute, property(lambda _value: collision))
    registry = build(values)
    with pytest.raises(ProjectionContractError, match="unknown or ambiguous"):
        registry.resolve_digest(collision)


@pytest.mark.parametrize("kind", ("ontology", "mapping", "family"))
def test_registry_snapshot_is_local_and_retains_current_and_historical_versions(kind):
    values, _value_type, attribute, build = _case(kind)
    supplied = list(values[:2])
    first = build(supplied)
    supplied.append(values[2])
    second = build(supplied)
    newest_digest = getattr(values[2], attribute)
    with pytest.raises(ProjectionContractError, match="unknown or ambiguous"):
        first.resolve_digest(newest_digest)
    assert second.resolve_digest(newest_digest) is values[2]
    identifier = getattr(values[0], {"ontology": "ontology_id", "mapping": "mapping_id", "family": "family_id"}[kind])
    assert first.resolve(identifier) is values[0]
    assert first.resolve(identifier, "v1") is values[1]
    listing = first.definitions() if kind == "family" else first.contracts()
    assert listing == values[:2]
    with pytest.raises(ProjectionContractError, match="unknown"):
        first.resolve(identifier, "not-retained")


@pytest.mark.parametrize("kind", ("ontology", "mapping", "family"))
def test_duplicate_registry_keys_remain_rejected(kind):
    values, _value_type, _attribute, build = _case(kind)
    with pytest.raises(ProjectionContractError, match="duplicate"):
        build((values[0], values[0]))


def test_family_registry_does_not_admit_unknown_mapping_or_ontology():
    values, _value_type, _attribute, build = _case("family", 1)
    for field in ("mapping_contract_digest", "ontology_contract_digest"):
        with pytest.raises(ProjectionContractError, match="unknown or ambiguous"):
            build((replace(values[0], **{field: "sha256:" + "0" * 64}),))
