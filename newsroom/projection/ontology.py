from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Iterable

from newsroom.authority.canonical import canonical_json_bytes, digest_bytes
from newsroom.authority.types import require_token

from .models import ProjectionContractError


class ProjectionNodeType(StrEnum):
    AUTHORITY_AGGREGATE = "AUTHORITY_AGGREGATE"
    AUTHORITY_VERSION = "AUTHORITY_VERSION"
    SOURCE_DEFINITION = "SOURCE_DEFINITION"
    SOURCE_DEFINITION_VERSION = "SOURCE_DEFINITION_VERSION"
    SOURCE_ITEM = "SOURCE_ITEM"
    SOURCE_REVISION = "SOURCE_REVISION"
    SOURCE_REPRESENTATION = "SOURCE_REPRESENTATION"
    DISCOVERY_OCCURRENCE = "DISCOVERY_OCCURRENCE"
    CHECK_REQUEST = "CHECK_REQUEST"
    CHECK_ATTEMPT = "CHECK_ATTEMPT"
    CHECK_OUTCOME = "CHECK_OUTCOME"
    OBSERVABLE_TRANSITION = "OBSERVABLE_TRANSITION"
    SIGNAL = "SIGNAL"
    GATE_DECISION = "GATE_DECISION"
    LEAD = "LEAD"
    PAYLOAD = "PAYLOAD"
    LEDGER_EVENT = "LEDGER_EVENT"


class ProjectionRelationType(StrEnum):
    HAS_VERSION = "HAS_VERSION"
    HAS_DEFINITION_VERSION = "HAS_DEFINITION_VERSION"
    DEFINES_ITEM = "DEFINES_ITEM"
    REQUESTED_CHECK = "REQUESTED_CHECK"
    ATTEMPTED_AS = "ATTEMPTED_AS"
    PRODUCED_CHECK_OUTCOME = "PRODUCED_CHECK_OUTCOME"
    HAS_REVISION = "HAS_REVISION"
    HAS_REPRESENTATION = "HAS_REPRESENTATION"
    OBSERVED_AS = "OBSERVED_AS"
    PRODUCED_OCCURRENCE = "PRODUCED_OCCURRENCE"
    TRANSITION_OF_ITEM = "TRANSITION_OF_ITEM"
    CLASSIFIED_BY_TRANSITION = "CLASSIFIED_BY_TRANSITION"
    PRODUCED_SIGNAL = "PRODUCED_SIGNAL"
    EMITTED_SIGNAL = "EMITTED_SIGNAL"
    DECIDED_BY_GATE = "DECIDED_BY_GATE"
    PROMOTED_TO_LEAD = "PROMOTED_TO_LEAD"
    OPENED_LEAD = "OPENED_LEAD"
    DUPLICATE_OF_SIGNAL = "DUPLICATE_OF_SIGNAL"
    REPLACED_BY_ITEM = "REPLACED_BY_ITEM"
    DERIVED_FROM = "DERIVED_FROM"
    CONTAINS_PAYLOAD = "CONTAINS_PAYLOAD"
    PROJECTED_FROM_EVENT = "PROJECTED_FROM_EVENT"


@dataclass(frozen=True, slots=True)
class OntologyNodeDefinition:
    node_type: ProjectionNodeType
    required_properties: frozenset[str]

    def __post_init__(self) -> None:
        if not isinstance(self.node_type, ProjectionNodeType):
            raise ProjectionContractError("ontology node type must be typed")
        if not isinstance(self.required_properties, frozenset) or not self.required_properties:
            raise ProjectionContractError("node required properties must be a non-empty frozenset")
        for item in self.required_properties:
            require_token(item, field="node_property")

    def canonical_value(self) -> dict[str, object]:
        return {
            "node_type": self.node_type.value,
            "required_properties": sorted(self.required_properties),
        }


@dataclass(frozen=True, slots=True)
class OntologyRelationDefinition:
    relation_type: ProjectionRelationType
    source_types: frozenset[ProjectionNodeType]
    target_types: frozenset[ProjectionNodeType]
    required_properties: frozenset[str]

    def __post_init__(self) -> None:
        if not isinstance(self.relation_type, ProjectionRelationType):
            raise ProjectionContractError("ontology relation type must be typed")
        for field, value in (
            ("source_types", self.source_types),
            ("target_types", self.target_types),
        ):
            if not isinstance(value, frozenset) or not value:
                raise ProjectionContractError(f"{field} must be a non-empty frozenset")
            if any(not isinstance(item, ProjectionNodeType) for item in value):
                raise ProjectionContractError(f"{field} must contain typed node types")
        if not isinstance(self.required_properties, frozenset):
            raise ProjectionContractError("relation properties must be a frozenset")
        for item in self.required_properties:
            require_token(item, field="relation_property")

    def canonical_value(self) -> dict[str, object]:
        return {
            "relation_type": self.relation_type.value,
            "source_types": sorted(item.value for item in self.source_types),
            "target_types": sorted(item.value for item in self.target_types),
            "required_properties": sorted(self.required_properties),
        }


@dataclass(frozen=True, slots=True)
class OntologyContract:
    ontology_id: str
    ontology_version: str
    implementation_version: str
    nodes: tuple[OntologyNodeDefinition, ...]
    relations: tuple[OntologyRelationDefinition, ...]

    def __post_init__(self) -> None:
        require_token(self.ontology_id, field="ontology_id")
        require_token(self.ontology_version, field="ontology_version")
        require_token(self.implementation_version, field="ontology_implementation_version")
        if not isinstance(self.nodes, tuple) or not self.nodes:
            raise ProjectionContractError("ontology requires node definitions")
        if not isinstance(self.relations, tuple) or not self.relations:
            raise ProjectionContractError("ontology requires relation definitions")
        node_types = [item.node_type for item in self.nodes]
        relation_types = [item.relation_type for item in self.relations]
        if len(node_types) != len(set(node_types)):
            raise ProjectionContractError("ontology node types must be unique")
        if len(relation_types) != len(set(relation_types)):
            raise ProjectionContractError("ontology relation types must be unique")
        known = set(node_types)
        for relation in self.relations:
            if not relation.source_types <= known or not relation.target_types <= known:
                raise ProjectionContractError("ontology relation references unknown node type")

    @property
    def node_types(self) -> frozenset[ProjectionNodeType]:
        return frozenset(item.node_type for item in self.nodes)

    @property
    def relation_types(self) -> frozenset[ProjectionRelationType]:
        return frozenset(item.relation_type for item in self.relations)

    def canonical_value(self) -> dict[str, object]:
        return {
            "ontology_id": self.ontology_id,
            "ontology_version": self.ontology_version,
            "implementation_version": self.implementation_version,
            "nodes": [
                item.canonical_value()
                for item in sorted(self.nodes, key=lambda value: value.node_type.value)
            ],
            "relations": [
                item.canonical_value()
                for item in sorted(self.relations, key=lambda value: value.relation_type.value)
            ],
        }

    @property
    def contract_digest(self) -> str:
        return digest_bytes(canonical_json_bytes(self.canonical_value()))


class OntologyRegistry:
    def __init__(
        self,
        contracts: Iterable[OntologyContract],
        *,
        current_versions: dict[str, str] | None = None,
    ) -> None:
        by_key: dict[tuple[str, str], OntologyContract] = {}
        versions: dict[str, list[str]] = {}
        for contract in contracts:
            key = (contract.ontology_id, contract.ontology_version)
            if key in by_key:
                raise ProjectionContractError("duplicate ontology contract")
            by_key[key] = contract
            versions.setdefault(contract.ontology_id, []).append(contract.ontology_version)
        if not by_key:
            raise ProjectionContractError("ontology registry cannot be empty")
        requested = dict(current_versions or {})
        selected: dict[str, str] = {}
        for ontology_id, available in versions.items():
            if ontology_id in requested:
                version = requested.pop(ontology_id)
            elif len(available) == 1:
                version = available[0]
            else:
                raise ProjectionContractError("ontology current version must be explicit")
            if (ontology_id, version) not in by_key:
                raise ProjectionContractError("unknown current ontology version")
            selected[ontology_id] = version
        if requested:
            raise ProjectionContractError("current ontology version names unknown contract")
        self._by_key = by_key
        self._current = selected

        # These frozen contracts already form a fixed registry snapshot.
        # Resolve retained identities without rehashing every contract per row.
        by_digest: dict[str, OntologyContract | None] = {}
        for item in by_key.values():
            digest = item.contract_digest
            by_digest[digest] = None if digest in by_digest else item
        self._by_digest = by_digest

    def resolve(self, ontology_id: str, version: str | None = None) -> OntologyContract:
        selected = self._current.get(ontology_id) if version is None else version
        try:
            return self._by_key[(ontology_id, selected or "")]
        except KeyError as exc:
            raise ProjectionContractError(f"unknown ontology: {ontology_id}/{selected}") from exc

    def resolve_digest(self, digest: str) -> OntologyContract:
        match = self._by_digest.get(digest) if isinstance(digest, str) else None
        if match is None:
            raise ProjectionContractError("unknown or ambiguous ontology digest")
        return match

    def contracts(self) -> tuple[OntologyContract, ...]:
        return tuple(self._by_key[key] for key in sorted(self._by_key))


_NATIVE_ONTOLOGY_V1_NODE_TYPES = frozenset(
    {
        ProjectionNodeType.AUTHORITY_AGGREGATE,
        ProjectionNodeType.AUTHORITY_VERSION,
        ProjectionNodeType.SOURCE_ITEM,
        ProjectionNodeType.SOURCE_REVISION,
        ProjectionNodeType.SOURCE_REPRESENTATION,
        ProjectionNodeType.SIGNAL,
        ProjectionNodeType.LEAD,
        ProjectionNodeType.PAYLOAD,
        ProjectionNodeType.LEDGER_EVENT,
    }
)


def native_ontology_v1() -> OntologyContract:
    # Keep the accepted Increment 1 ontology byte-for-byte stable when later
    # structural families add enum members. New families must use a separate
    # versioned ontology contract instead of silently widening ontology-v1.
    common_identity = frozenset({"canonical_id", "entity_type"})
    nodes = tuple(
        OntologyNodeDefinition(item, common_identity)
        for item in sorted(_NATIVE_ONTOLOGY_V1_NODE_TYPES, key=lambda value: value.value)
    )
    provenance = frozenset({"authority_event_id", "ledger_seq"})
    relations = (
        OntologyRelationDefinition(ProjectionRelationType.HAS_VERSION, frozenset({ProjectionNodeType.AUTHORITY_AGGREGATE, ProjectionNodeType.SOURCE_ITEM}), frozenset({ProjectionNodeType.AUTHORITY_VERSION}), provenance),
        OntologyRelationDefinition(ProjectionRelationType.HAS_REVISION, frozenset({ProjectionNodeType.SOURCE_ITEM}), frozenset({ProjectionNodeType.SOURCE_REVISION}), provenance),
        OntologyRelationDefinition(ProjectionRelationType.HAS_REPRESENTATION, frozenset({ProjectionNodeType.SOURCE_REVISION}), frozenset({ProjectionNodeType.SOURCE_REPRESENTATION}), provenance),
        OntologyRelationDefinition(ProjectionRelationType.PRODUCED_SIGNAL, frozenset({ProjectionNodeType.SOURCE_REVISION, ProjectionNodeType.SOURCE_REPRESENTATION}), frozenset({ProjectionNodeType.SIGNAL}), provenance),
        OntologyRelationDefinition(ProjectionRelationType.PROMOTED_TO_LEAD, frozenset({ProjectionNodeType.SIGNAL}), frozenset({ProjectionNodeType.LEAD}), provenance),
        OntologyRelationDefinition(ProjectionRelationType.DERIVED_FROM, _NATIVE_ONTOLOGY_V1_NODE_TYPES, _NATIVE_ONTOLOGY_V1_NODE_TYPES, provenance),
        OntologyRelationDefinition(ProjectionRelationType.CONTAINS_PAYLOAD, frozenset({ProjectionNodeType.AUTHORITY_VERSION, ProjectionNodeType.LEDGER_EVENT}), frozenset({ProjectionNodeType.PAYLOAD}), provenance),
        OntologyRelationDefinition(ProjectionRelationType.PROJECTED_FROM_EVENT, frozenset(item for item in _NATIVE_ONTOLOGY_V1_NODE_TYPES if item is not ProjectionNodeType.LEDGER_EVENT), frozenset({ProjectionNodeType.LEDGER_EVENT}), provenance),
    )
    return OntologyContract(
        ontology_id="newsroom.structural",
        ontology_version="ontology-v1",
        implementation_version="ontology-python-v1",
        nodes=nodes,
        relations=relations,
    )
