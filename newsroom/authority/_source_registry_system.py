from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, TypeVar

from newsroom.sources.definition_models import (
    SourceDefinitionRequest,
    SourceDefinitionVersionRequest,
)
from newsroom.sources.item_models import (
    LocatorContinuityDecisionRequest,
    SourceItemRequest,
)
from newsroom.sources.observation_models import (
    DiscoveryOccurrenceRequest,
    DiscoveryRepresentationRequest,
    SourceRevisionRequest,
)
from newsroom.sources.policy import (
    DISCOVERY_OCCURRENCE_RECORD_COMMAND,
    DISCOVERY_REPRESENTATION_RECORD_COMMAND,
    SOURCE_DEFINITION_REGISTER_COMMAND,
    SOURCE_DEFINITION_VERSION_RECORD_COMMAND,
    SOURCE_ITEM_REGISTER_COMMAND,
    SOURCE_LOCATOR_CONTINUITY_DECIDE_COMMAND,
    SOURCE_REVISION_RECORD_COMMAND,
    merge_source_registry_authority_registries,
)
from newsroom.sources.record_models import (
    DiscoveryOccurrence,
    DiscoveryRepresentation,
    LocatorContinuityDecision,
    SourceDefinition,
    SourceDefinitionVersion,
    SourceDefinitionVersionSummary,
    SourceItem,
    SourceRevision,
)
from newsroom.sources.types import (
    DiscoveryRepresentationId,
    SourceDefinitionId,
    SourceDefinitionVersionId,
    SourceItemId,
    SourceRegistryReadPolicy,
    SourceRevisionId,
)

from ._capability import _CapabilityIssuer
from ._security import _AuthorizationRequest
from ._source_registry_store import _SourceRegistryAuthorityStore
from .auth import AuthenticationProof
from .canonical import digest_canonical
from .models import InlinePayload, SemanticCommand
from .policy import CommandRegistry, PayloadSchemaRegistry
from .service import CommandService
from .types import AggregateId, UtcTimestamp

_Record = TypeVar("_Record")

_SOURCE_READ_SCHEMA_DIGEST = digest_canonical(
    {
        "contract": "source-registry-read-no-payload-v1",
        "payload_mode": "NO_PAYLOAD",
        "redaction": "metadata-or-explicit-sensitive-scope",
    }
)


class GovernedSources:
    """Typed public source facade; raw SQLite and locators never escape by default."""

    __slots__ = (
        "__register_definition",
        "__record_version",
        "__register_item",
        "__decide_locator",
        "__record_revision",
        "__record_representation",
        "__record_occurrence",
        "__definition",
        "__current_summary",
        "__version_details",
        "__item",
        "__revision",
        "__representation",
        "__occurrences",
    )

    def __init__(
        self,
        *,
        register_definition: Callable[
            [SourceDefinitionRequest, AuthenticationProof], SourceDefinition
        ],
        record_version: Callable[
            [SourceDefinitionVersionRequest, AuthenticationProof],
            SourceDefinitionVersion,
        ],
        register_item: Callable[
            [SourceItemRequest, AuthenticationProof], SourceItem
        ],
        decide_locator: Callable[
            [LocatorContinuityDecisionRequest, AuthenticationProof],
            LocatorContinuityDecision,
        ],
        record_revision: Callable[
            [SourceRevisionRequest, AuthenticationProof], SourceRevision
        ],
        record_representation: Callable[
            [DiscoveryRepresentationRequest, AuthenticationProof],
            DiscoveryRepresentation,
        ],
        record_occurrence: Callable[
            [DiscoveryOccurrenceRequest, AuthenticationProof],
            DiscoveryOccurrence,
        ],
        definition: Callable[
            [SourceDefinitionId, AuthenticationProof], SourceDefinition
        ],
        current_summary: Callable[
            [SourceDefinitionId, AuthenticationProof],
            SourceDefinitionVersionSummary,
        ],
        version_details: Callable[
            [SourceDefinitionVersionId, AuthenticationProof],
            SourceDefinitionVersion,
        ],
        item: Callable[[SourceItemId, AuthenticationProof], SourceItem],
        revision: Callable[
            [SourceRevisionId, AuthenticationProof], SourceRevision
        ],
        representation: Callable[
            [DiscoveryRepresentationId, AuthenticationProof],
            DiscoveryRepresentation,
        ],
        occurrences: Callable[
            [SourceRevisionId, int, AuthenticationProof],
            tuple[DiscoveryOccurrence, ...],
        ],
    ) -> None:
        self.__register_definition = register_definition
        self.__record_version = record_version
        self.__register_item = register_item
        self.__decide_locator = decide_locator
        self.__record_revision = record_revision
        self.__record_representation = record_representation
        self.__record_occurrence = record_occurrence
        self.__definition = definition
        self.__current_summary = current_summary
        self.__version_details = version_details
        self.__item = item
        self.__revision = revision
        self.__representation = representation
        self.__occurrences = occurrences

    def register_definition(
        self,
        request: SourceDefinitionRequest,
        *,
        proof: AuthenticationProof,
    ) -> SourceDefinition:
        return self.__register_definition(request, proof)

    def record_definition_version(
        self,
        request: SourceDefinitionVersionRequest,
        *,
        proof: AuthenticationProof,
    ) -> SourceDefinitionVersion:
        return self.__record_version(request, proof)

    def register_item(
        self,
        request: SourceItemRequest,
        *,
        proof: AuthenticationProof,
    ) -> SourceItem:
        return self.__register_item(request, proof)

    def decide_locator_continuity(
        self,
        request: LocatorContinuityDecisionRequest,
        *,
        proof: AuthenticationProof,
    ) -> LocatorContinuityDecision:
        return self.__decide_locator(request, proof)

    def record_revision(
        self,
        request: SourceRevisionRequest,
        *,
        proof: AuthenticationProof,
    ) -> SourceRevision:
        return self.__record_revision(request, proof)

    def record_representation(
        self,
        request: DiscoveryRepresentationRequest,
        *,
        proof: AuthenticationProof,
    ) -> DiscoveryRepresentation:
        return self.__record_representation(request, proof)

    def record_occurrence(
        self,
        request: DiscoveryOccurrenceRequest,
        *,
        proof: AuthenticationProof,
    ) -> DiscoveryOccurrence:
        return self.__record_occurrence(request, proof)

    def definition(
        self,
        definition_id: SourceDefinitionId,
        *,
        proof: AuthenticationProof,
    ) -> SourceDefinition:
        return self.__definition(definition_id, proof)

    def current_summary(
        self,
        definition_id: SourceDefinitionId,
        *,
        proof: AuthenticationProof,
    ) -> SourceDefinitionVersionSummary:
        return self.__current_summary(definition_id, proof)

    def version_details(
        self,
        version_id: SourceDefinitionVersionId,
        *,
        proof: AuthenticationProof,
    ) -> SourceDefinitionVersion:
        return self.__version_details(version_id, proof)

    def item(
        self,
        item_id: SourceItemId,
        *,
        proof: AuthenticationProof,
    ) -> SourceItem:
        return self.__item(item_id, proof)

    def revision(
        self,
        revision_id: SourceRevisionId,
        *,
        proof: AuthenticationProof,
    ) -> SourceRevision:
        return self.__revision(revision_id, proof)

    def representation(
        self,
        representation_id: DiscoveryRepresentationId,
        *,
        proof: AuthenticationProof,
    ) -> DiscoveryRepresentation:
        return self.__representation(representation_id, proof)

    def occurrences(
        self,
        revision_id: SourceRevisionId,
        *,
        limit: int,
        proof: AuthenticationProof,
    ) -> tuple[DiscoveryOccurrence, ...]:
        return self.__occurrences(revision_id, limit, proof)


class GovernedSourceRegistryAuthoritySystem:
    __slots__ = ("sources", "__close")

    def __init__(
        self,
        *,
        sources: GovernedSources,
        close: Callable[[], None],
    ) -> None:
        self.sources = sources
        self.__close = close

    def close(self) -> None:
        self.__close()

    def __enter__(self) -> "GovernedSourceRegistryAuthoritySystem":
        return self

    def __exit__(
        self, exc_type: object, exc: object, tb: object
    ) -> None:
        self.close()


class _SourceRegistryBoundary:
    def __init__(
        self,
        *,
        store: _SourceRegistryAuthorityStore,
        command_service: CommandService,
        authenticator: Any,
        authorizer: Any,
        read_policy: SourceRegistryReadPolicy,
        clock: Callable[[], UtcTimestamp],
    ) -> None:
        self._store = store
        self._command_service = command_service
        self._authenticator = authenticator
        self._authorizer = authorizer
        self._read_policy = read_policy
        self._clock = clock

    def _commit(
        self,
        request: Any,
        proof: AuthenticationProof,
        *,
        command_type: str,
        aggregate_id: AggregateId,
        commit: Callable[..., _Record],
    ) -> _Record:
        command = SemanticCommand(
            command_type=command_type,
            aggregate_id=aggregate_id,
            expected_aggregate_version=0,
            payload=InlinePayload(request.canonical_value()),
            idempotency_key=request.idempotency_key,
        )
        grant = self._command_service._authorize_for_commit(
            command, proof=proof
        )
        return commit(grant, request=request)

    def register_definition(
        self,
        request: SourceDefinitionRequest,
        proof: AuthenticationProof,
    ) -> SourceDefinition:
        if not isinstance(request, SourceDefinitionRequest):
            raise TypeError("source definition must be a typed request")
        return self._commit(
            request,
            proof,
            command_type=SOURCE_DEFINITION_REGISTER_COMMAND,
            aggregate_id=AggregateId(request.definition_id.value),
            commit=self._store.commit_source_definition,
        )

    def record_version(
        self,
        request: SourceDefinitionVersionRequest,
        proof: AuthenticationProof,
    ) -> SourceDefinitionVersion:
        if not isinstance(request, SourceDefinitionVersionRequest):
            raise TypeError("source version must be a typed request")
        return self._commit(
            request,
            proof,
            command_type=SOURCE_DEFINITION_VERSION_RECORD_COMMAND,
            aggregate_id=AggregateId(request.version_id.value),
            commit=self._store.commit_source_definition_version,
        )

    def register_item(
        self,
        request: SourceItemRequest,
        proof: AuthenticationProof,
    ) -> SourceItem:
        if not isinstance(request, SourceItemRequest):
            raise TypeError("source item must be a typed request")
        return self._commit(
            request,
            proof,
            command_type=SOURCE_ITEM_REGISTER_COMMAND,
            aggregate_id=AggregateId(request.item_id.value),
            commit=self._store.commit_source_item,
        )

    def decide_locator(
        self,
        request: LocatorContinuityDecisionRequest,
        proof: AuthenticationProof,
    ) -> LocatorContinuityDecision:
        if not isinstance(request, LocatorContinuityDecisionRequest):
            raise TypeError("locator continuity must be a typed request")
        return self._commit(
            request,
            proof,
            command_type=SOURCE_LOCATOR_CONTINUITY_DECIDE_COMMAND,
            aggregate_id=AggregateId(request.decision_id.value),
            commit=self._store.commit_locator_continuity_decision,
        )

    def record_revision(
        self,
        request: SourceRevisionRequest,
        proof: AuthenticationProof,
    ) -> SourceRevision:
        if not isinstance(request, SourceRevisionRequest):
            raise TypeError("source revision must be a typed request")
        return self._commit(
            request,
            proof,
            command_type=SOURCE_REVISION_RECORD_COMMAND,
            aggregate_id=AggregateId(request.revision_id.value),
            commit=self._store.commit_source_revision,
        )

    def record_representation(
        self,
        request: DiscoveryRepresentationRequest,
        proof: AuthenticationProof,
    ) -> DiscoveryRepresentation:
        if not isinstance(request, DiscoveryRepresentationRequest):
            raise TypeError("representation must be a typed request")
        return self._commit(
            request,
            proof,
            command_type=DISCOVERY_REPRESENTATION_RECORD_COMMAND,
            aggregate_id=AggregateId(request.representation_id.value),
            commit=self._store.commit_discovery_representation,
        )

    def record_occurrence(
        self,
        request: DiscoveryOccurrenceRequest,
        proof: AuthenticationProof,
    ) -> DiscoveryOccurrence:
        if not isinstance(request, DiscoveryOccurrenceRequest):
            raise TypeError("occurrence must be a typed request")
        return self._commit(
            request,
            proof,
            command_type=DISCOVERY_OCCURRENCE_RECORD_COMMAND,
            aggregate_id=AggregateId(request.occurrence_id.value),
            commit=self._store.commit_discovery_occurrence,
        )

    def _authorize_read(
        self,
        proof: AuthenticationProof,
        *,
        operation: str,
        aggregate_type: str,
        aggregate_id: str,
        sensitive: bool,
        limit: int | None = None,
    ) -> None:
        now = self._clock()
        authentication = self._authenticator.authenticate(proof, now=now)
        authentication.require_current(now)
        self._read_policy.require_principal(authentication.principal_id)
        if limit is not None:
            self._read_policy.require_limit(limit)
        required_scope = (
            self._read_policy.sensitive_required_scope
            if sensitive
            else self._read_policy.metadata_required_scope
        )
        stable = digest_canonical(
            {
                "contract": "source-registry-read-v1",
                "policy_digest": self._read_policy.digest,
                "operation": operation,
                "aggregate_type": aggregate_type,
                "aggregate_id": aggregate_id,
                "sensitive": sensitive,
                "limit": limit,
            }
        )
        unsigned = {
            "authentication_context_id": str(
                authentication.authentication_context_id
            ),
            "principal_id": authentication.principal_id,
            "authority_domain": authentication.authority_domain,
            "operation_type": operation,
            "required_scope": required_scope,
            "stable_semantic_request_digest": stable,
            "command_definition_digest": _SOURCE_READ_SCHEMA_DIGEST,
            "aggregate_type": aggregate_type,
            "aggregate_id": aggregate_id,
            "event_type": "source.registry.read",
            "event_schema_version": 1,
            "payload_mode": "NO_PAYLOAD",
            "payload_schema_version": "source_registry_read_v1",
            "payload_schema_contract_version": (
                "source-registry-read-no-payload-v1"
            ),
            "payload_schema_contract_digest": _SOURCE_READ_SCHEMA_DIGEST,
            "payload_canonicalizer_version": "source-registry-none-v1",
            "trust_scope": "ADMITTED",
            "security_scope": "authority.source_registry",
            "retention_scope": "authority.audit",
            "object_class": None,
            "allowed_use": None,
        }
        request = _AuthorizationRequest(
            authentication_context_id=authentication.authentication_context_id,
            principal_id=authentication.principal_id,
            authority_domain=authentication.authority_domain,
            operation_type=operation,
            required_scope=required_scope,
            stable_semantic_request_digest=stable,
            command_definition_digest=_SOURCE_READ_SCHEMA_DIGEST,
            aggregate_type=aggregate_type,
            aggregate_id=aggregate_id,
            event_type="source.registry.read",
            event_schema_version=1,
            payload_mode="NO_PAYLOAD",
            payload_schema_version="source_registry_read_v1",
            payload_schema_contract_version=(
                "source-registry-read-no-payload-v1"
            ),
            payload_schema_contract_digest=_SOURCE_READ_SCHEMA_DIGEST,
            payload_canonicalizer_version="source-registry-none-v1",
            trust_scope="ADMITTED",
            security_scope="authority.source_registry",
            retention_scope="authority.audit",
            object_class=None,
            allowed_use=None,
            request_digest=digest_canonical(unsigned),
        )
        decision = self._authorizer.authorize(
            authentication, request, now=now
        )
        if (
            decision.authentication_context_id
            != authentication.authentication_context_id
            or decision.authorization_request_digest
            != request.request_digest
        ):
            raise PermissionError(
                "source registry read authorization provenance differs"
            )
        decision.require_allowed()

    def definition(
        self,
        definition_id: SourceDefinitionId,
        proof: AuthenticationProof,
    ) -> SourceDefinition:
        if not isinstance(definition_id, SourceDefinitionId):
            raise TypeError("source definition identity must be typed")
        self._authorize_read(
            proof,
            operation="read:source_registry:definition",
            aggregate_type="source_definition",
            aggregate_id=str(definition_id),
            sensitive=False,
        )
        value = self._store.source_definition(definition_id)
        if value is None:
            raise LookupError("source definition is not retained")
        return value

    def current_summary(
        self,
        definition_id: SourceDefinitionId,
        proof: AuthenticationProof,
    ) -> SourceDefinitionVersionSummary:
        if not isinstance(definition_id, SourceDefinitionId):
            raise TypeError("source definition identity must be typed")
        self._authorize_read(
            proof,
            operation="read:source_registry:current_summary",
            aggregate_type="source_definition",
            aggregate_id=str(definition_id),
            sensitive=False,
        )
        value = self._store.current_source_definition_summary(definition_id)
        if value is None:
            raise LookupError("current source definition version is not retained")
        return value

    def version_details(
        self,
        version_id: SourceDefinitionVersionId,
        proof: AuthenticationProof,
    ) -> SourceDefinitionVersion:
        if not isinstance(version_id, SourceDefinitionVersionId):
            raise TypeError("source version identity must be typed")
        self._authorize_read(
            proof,
            operation="read:source_registry:version_details",
            aggregate_type="source_definition_version",
            aggregate_id=str(version_id),
            sensitive=True,
        )
        value = self._store.source_definition_version(version_id)
        if value is None:
            raise LookupError("source definition version is not retained")
        return value

    def item(
        self, item_id: SourceItemId, proof: AuthenticationProof
    ) -> SourceItem:
        if not isinstance(item_id, SourceItemId):
            raise TypeError("source item identity must be typed")
        self._authorize_read(
            proof,
            operation="read:source_registry:item",
            aggregate_type="source_item",
            aggregate_id=str(item_id),
            sensitive=True,
        )
        value = self._store.source_item(item_id)
        if value is None:
            raise LookupError("source item is not retained")
        return value

    def revision(
        self,
        revision_id: SourceRevisionId,
        proof: AuthenticationProof,
    ) -> SourceRevision:
        if not isinstance(revision_id, SourceRevisionId):
            raise TypeError("source revision identity must be typed")
        self._authorize_read(
            proof,
            operation="read:source_registry:revision",
            aggregate_type="source_revision",
            aggregate_id=str(revision_id),
            sensitive=True,
        )
        value = self._store.source_revision(revision_id)
        if value is None:
            raise LookupError("source revision is not retained")
        return value

    def representation(
        self,
        representation_id: DiscoveryRepresentationId,
        proof: AuthenticationProof,
    ) -> DiscoveryRepresentation:
        if not isinstance(representation_id, DiscoveryRepresentationId):
            raise TypeError("representation identity must be typed")
        self._authorize_read(
            proof,
            operation="read:source_registry:representation",
            aggregate_type="discovery_representation",
            aggregate_id=str(representation_id),
            sensitive=True,
        )
        value = self._store.discovery_representation(representation_id)
        if value is None:
            raise LookupError("discovery representation is not retained")
        return value

    def occurrences(
        self,
        revision_id: SourceRevisionId,
        limit: int,
        proof: AuthenticationProof,
    ) -> tuple[DiscoveryOccurrence, ...]:
        if not isinstance(revision_id, SourceRevisionId):
            raise TypeError("source revision identity must be typed")
        self._authorize_read(
            proof,
            operation="read:source_registry:occurrences",
            aggregate_type="source_revision",
            aggregate_id=str(revision_id),
            sensitive=False,
            limit=limit,
        )
        return self._store.occurrences_for_revision(
            revision_id, limit=limit
        )


def open_governed_source_registry_authority_system(
    *,
    path: Path,
    registry: CommandRegistry,
    payload_schemas: PayloadSchemaRegistry,
    authenticator: Any,
    authorizer: Any,
    read_policy: SourceRegistryReadPolicy,
    command_service_version: str = "authority-command-v1",
    busy_timeout_ms: int = 5_000,
    clock: Callable[[], UtcTimestamp] = UtcTimestamp.now,
) -> GovernedSourceRegistryAuthoritySystem:
    merged_registry, merged_schemas = (
        merge_source_registry_authority_registries(
            command_registry=registry,
            payload_schemas=payload_schemas,
        )
    )
    issuer = _CapabilityIssuer(
        command_registry=merged_registry,
        payload_schemas=merged_schemas,
    )
    store: _SourceRegistryAuthorityStore | None = None
    try:
        store = _SourceRegistryAuthorityStore(
            path,
            issuer=issuer,
            command_registry=merged_registry,
            payload_schemas=merged_schemas,
            command_service_version=command_service_version,
            busy_timeout_ms=busy_timeout_ms,
            clock=clock,
        )
        command_service = CommandService(
            registry=merged_registry,
            payload_schemas=merged_schemas,
            authenticator=authenticator,
            authorizer=authorizer,
            committed_lookup=store,
            clock=clock,
            _issuer=issuer,
        )
        boundary = _SourceRegistryBoundary(
            store=store,
            command_service=command_service,
            authenticator=authenticator,
            authorizer=authorizer,
            read_policy=read_policy,
            clock=clock,
        )
        closed = False

        def close() -> None:
            nonlocal closed
            if closed:
                return
            closed = True
            assert store is not None
            store.close()

        return GovernedSourceRegistryAuthoritySystem(
            sources=GovernedSources(
                register_definition=boundary.register_definition,
                record_version=boundary.record_version,
                register_item=boundary.register_item,
                decide_locator=boundary.decide_locator,
                record_revision=boundary.record_revision,
                record_representation=boundary.record_representation,
                record_occurrence=boundary.record_occurrence,
                definition=boundary.definition,
                current_summary=boundary.current_summary,
                version_details=boundary.version_details,
                item=boundary.item,
                revision=boundary.revision,
                representation=boundary.representation,
                occurrences=boundary.occurrences,
            ),
            close=close,
        )
    except Exception:
        if store is not None:
            store.close()
        raise


__all__ = [
    "GovernedSourceRegistryAuthoritySystem",
    "GovernedSources",
    "open_governed_source_registry_authority_system",
]
