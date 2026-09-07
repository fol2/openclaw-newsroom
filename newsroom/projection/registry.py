from __future__ import annotations

from collections.abc import Iterable

from .mapping import StructuralMappingRegistry
from .models import ProjectionContractError, ProjectionFamilyDefinition
from .ontology import OntologyRegistry


class ProjectionFamilyRegistry:
    def __init__(
        self,
        definitions: Iterable[ProjectionFamilyDefinition],
        *,
        ontologies: OntologyRegistry,
        mappings: StructuralMappingRegistry,
        current_versions: dict[str, str] | None = None,
    ) -> None:
        by_key: dict[tuple[str, str], ProjectionFamilyDefinition] = {}
        versions: dict[str, list[str]] = {}
        aggregate_owners: dict[str, str] = {}
        for definition in definitions:
            key = (definition.family_id, definition.definition_version)
            if key in by_key:
                raise ProjectionContractError("duplicate projection family definition")
            aggregate_key = str(definition.authority_aggregate_id)
            owner = aggregate_owners.get(aggregate_key)
            if owner is not None and owner != definition.family_id:
                raise ProjectionContractError(
                    "projection family aggregate ID cannot be shared across families"
                )
            ontology = ontologies.resolve_digest(definition.ontology_contract_digest)
            mapping = mappings.resolve_digest(definition.mapping_contract_digest)
            mapping.validate_against(ontology)
            by_key[key] = definition
            versions.setdefault(definition.family_id, []).append(definition.definition_version)
            aggregate_owners[aggregate_key] = definition.family_id
        if not by_key:
            raise ProjectionContractError("projection family registry cannot be empty")
        requested = dict(current_versions or {})
        selected: dict[str, str] = {}
        for family_id, available in versions.items():
            if family_id in requested:
                version = requested.pop(family_id)
            elif len(available) == 1:
                version = available[0]
            else:
                raise ProjectionContractError("family current version must be explicit")
            if (family_id, version) not in by_key:
                raise ProjectionContractError("unknown current family version")
            selected[family_id] = version
        if requested:
            raise ProjectionContractError("current family version names unknown definition")
        self._by_key = by_key
        self._current = selected

        # These frozen contracts already form a fixed registry snapshot.
        # Resolve retained identities without rehashing every contract per row.
        by_digest: dict[str, ProjectionFamilyDefinition | None] = {}
        for item in by_key.values():
            digest = item.digest
            by_digest[digest] = None if digest in by_digest else item
        self._by_digest = by_digest

    def resolve(self, family_id: str, version: str | None = None) -> ProjectionFamilyDefinition:
        selected = self._current.get(family_id) if version is None else version
        try:
            return self._by_key[(family_id, selected or "")]
        except KeyError as exc:
            raise ProjectionContractError(f"unknown projection family: {family_id}/{selected}") from exc

    def resolve_digest(self, digest: str) -> ProjectionFamilyDefinition:
        match = self._by_digest.get(digest) if isinstance(digest, str) else None
        if match is None:
            raise ProjectionContractError("unknown or ambiguous family definition digest")
        return match

    def definitions(self) -> tuple[ProjectionFamilyDefinition, ...]:
        return tuple(self._by_key[key] for key in sorted(self._by_key))
