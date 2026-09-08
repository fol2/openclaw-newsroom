"""One native Hermes authority writer composed over the cumulative store."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from newsroom.checks.policy import merge_discovery_check_authority_registries
from newsroom.checks.read_policy import DiscoveryCheckReadPolicy
from newsroom.discovery.policy import merge_discovery_signal_lead_registries
from newsroom.discovery.types import DiscoveryReadPolicy
from newsroom.increment6._execution_store import _open_on_connection
from newsroom.increment6.candidates import (
    _compose_story_candidate_authority,
    merge_candidate_authority_registries,
)
from newsroom.increment6.collision import CurrentCollisionEffectEnforcer
from newsroom.increment6.dispositions import ProposalDispositionStore
from newsroom.increment6.hypotheses import _compose_event_hypothesis_authority
from newsroom.increment6.lineage import (
    _compose_event_hypothesis_lineage_authority,
    merge_lineage_authority_registries,
)
from newsroom.increment6.relationships import (
    _compose_event_hypothesis_relationship_authority,
    merge_relationship_authority_registries,
)
from newsroom.increment6.work_items import RetrievalContextAuthority, TriageWorkItemStore
from newsroom.projection.neo4j.models import Neo4jProjectorConfig

from ._check_boundary import _CheckBoundary
from ._check_facade import GovernedChecks
from ._discovery_boundary import _DiscoveryBoundary
from ._discovery_facade import GovernedDiscovery
from ._event_hypothesis_lineage_system import (
    _LineageStore,
    _create_event_hypothesis_lineage_read_port,
)
from ._event_hypothesis_relationship_system import (
    _RelationshipEventStore,
    _create_event_hypothesis_relationship_read_port,
)
from ._event_hypothesis_system import _HypothesisStore
from ._event_system import _ReadBoundary
from ._graphiti_increment4_system import (
    _AUTHORITY_COMPOSITION_TOKEN,
    GovernedGraphitiIncrement4AuthoritySystem,
    open_governed_graphiti_increment4_authority_system,
)
from ._proposal_admission import _ProposalAdmissionBoundary
from ._signal_lead_admission import _SignalLeadAdmissionBoundary
from .auth import AuthenticationProof, StaticAuthenticator
from .models import SemanticCommand
from .persistence import AuthorityCommands, AuthorityEvents, CommittedCommand, EventReadPolicy
from .policy import CommandRegistry, PayloadSchemaRegistry
from .story_candidate_system import _CandidateStore, _create_story_candidate_read_port
from .types import UtcTimestamp


class _SharedAuthority:
    """Keep child facades from closing the sole authority writer."""

    __slots__ = ("_authority", "_lock")

    def __init__(self, authority: object, lock: Any) -> None:
        self._authority = authority
        self._lock = lock

    def __getattr__(self, name: str) -> object:
        value = getattr(self._authority, name)
        if not callable(value):
            return value

        def serialised(*args: object, **kwargs: object) -> object:
            with self._lock:
                return value(*args, **kwargs)

        return serialised

    def close(self) -> None:
        return None


class _SharedRelationshipStore(_RelationshipEventStore):
    def close(self) -> None:
        return None


class _SharedLineageStore(_LineageStore):
    def close(self) -> None:
        return None


class _SharedCandidateStore(_CandidateStore):
    def close(self) -> None:
        return None


def _share_store(store_type: type, root: object) -> Any:
    store = object.__new__(store_type)
    store.__dict__.update(root.__dict__)
    return store


_SYSTEM_TOKEN = object()


class HermesNativeAuthoritySystem:
    """Server-owned facades sharing one SQLite/CAS writer and lock."""

    def __init__(self, token: object, *, base: GovernedGraphitiIncrement4AuthoritySystem, **facets: object) -> None:
        if token is not _SYSTEM_TOKEN:
            raise TypeError("Hermes native authority systems require the production opener")
        self._base = base
        self.__dict__.update(facets)

    @property
    def sources(self): return self._base.sources
    @property
    def extraction(self): return self._base.extraction
    @property
    def objects(self): return self._base.objects
    @property
    def entities(self): return self._base.entities
    @property
    def relations(self): return self._base.relations
    @property
    def graphiti(self): return self._base.graphiti
    @property
    def structural(self): return self._base.structural
    @property
    def increment4(self): return self._base.increment4
    @property
    def compatibility(self): return self._base.compatibility
    @property
    def authority_store_path(self): return self._base.authority_store_path
    @property
    def graph_destination_id(self): return self._base.graph_destination_id

    def close(self) -> None:
        self._base.close()

    def __enter__(self): return self
    def __exit__(self, *_: object) -> None: self.close()


def open_hermes_native_authority_system(
    *,
    path: Path,
    object_root: Path,
    workspace_root: Path,
    registry: CommandRegistry,
    payload_schemas: PayloadSchemaRegistry,
    admission_registry: Any,
    rights_policies: Any,
    hydration_policies: Any,
    contracts: Any,
    authenticator: StaticAuthenticator,
    authorizer: Any,
    event_read_policy: EventReadPolicy,
    source_read_policy: Any,
    check_read_policy: DiscoveryCheckReadPolicy,
    discovery_read_policy: DiscoveryReadPolicy,
    extraction_read_policy: Any,
    entity_read_policy: Any,
    relation_read_policy: Any,
    graphiti_read_policy: Any,
    projection_read_policy: Any,
    object_limits: Any,
    neo4j_config: Neo4jProjectorConfig,
    retrieval_authority: RetrievalContextAuthority,
    collision_enforcer: CurrentCollisionEffectEnforcer,
    command_service_version: str = "hermes-native-authority-v1",
    busy_timeout_ms: int = 5_000,
    lease_ttl_seconds: int = 300,
    clock: Callable[[], UtcTimestamp] = UtcTimestamp.now,
    cas_fault_hook: Callable[[str], None] | None = None,
    disk_usage: Callable[[Path], Any] | None = None,
) -> HermesNativeAuthoritySystem:
    """Open the sole production writer used by one Hermes runtime."""

    commands, schemas = merge_discovery_check_authority_registries(
        command_registry=registry, payload_schemas=payload_schemas
    )
    commands, schemas = merge_discovery_signal_lead_registries(
        command_registry=commands, payload_schemas=schemas
    )
    commands, schemas = merge_relationship_authority_registries(commands, schemas)
    commands, schemas = merge_lineage_authority_registries(commands, schemas)
    commands, schemas = merge_candidate_authority_registries(commands, schemas)
    base = open_governed_graphiti_increment4_authority_system(
        path=path,
        object_root=object_root,
        workspace_root=workspace_root,
        registry=commands,
        payload_schemas=schemas,
        admission_registry=admission_registry,
        rights_policies=rights_policies,
        hydration_policies=hydration_policies,
        contracts=contracts,
        authenticator=authenticator,
        authorizer=authorizer,
        source_read_policy=source_read_policy,
        extraction_read_policy=extraction_read_policy,
        entity_read_policy=entity_read_policy,
        relation_read_policy=relation_read_policy,
        graphiti_read_policy=graphiti_read_policy,
        projection_read_policy=projection_read_policy,
        object_limits=object_limits,
        neo4j_config=neo4j_config,
        command_service_version=command_service_version,
        busy_timeout_ms=busy_timeout_ms,
        clock=clock,
        cas_fault_hook=cas_fault_hook,
        disk_usage=disk_usage,
    )
    try:
        root, service, _, _, commands, schemas = base._authority_composition(
            _AUTHORITY_COMPOSITION_TOKEN
        )
        connection, operation_lock = root._connection, root._lock

        check_boundary = _CheckBoundary(
            store=root, command_service=service, authenticator=authenticator,
            authorizer=authorizer, read_policy=check_read_policy, clock=clock,
        )
        proposal_admission = _ProposalAdmissionBoundary(store=root, command_service=service)
        discovery_boundary = _DiscoveryBoundary(
            store=root, command_service=service, authenticator=authenticator,
            authorizer=authorizer, read_policy=discovery_read_policy, clock=clock,
        )
        signal_admission = _SignalLeadAdmissionBoundary(store=root, command_service=service)
        checks = GovernedChecks(
            register_request=check_boundary.register_request,
            start_attempt=check_boundary.start_attempt,
            record_outcome=check_boundary.record_outcome,
            decide_baseline=check_boundary.decide_baseline,
            record_transition=check_boundary.record_transition,
            open_finding=check_boundary.open_finding,
            record_finding_occurrence=check_boundary.record_finding_occurrence,
            admit_proposal=proposal_admission.admit,
            request=check_boundary.request, attempt=check_boundary.attempt,
            outcome=check_boundary.outcome, attempts=check_boundary.attempts,
            outcomes=check_boundary.outcomes, baseline=check_boundary.baseline,
            current_baseline=check_boundary.current_baseline,
            transition=check_boundary.transition, finding=check_boundary.finding,
            finding_occurrences=check_boundary.finding_occurrences,
        )
        discovery = GovernedDiscovery(
            admit_signal=discovery_boundary.admit_signal,
            decide_gate=discovery_boundary.decide_gate,
            open_lead=discovery_boundary.open_lead,
            record_watch_condition=discovery_boundary.record_watch_condition,
            record_lead_disposition=discovery_boundary.record_lead_disposition,
            admit_signal_to_lead=signal_admission.admit,
            signal=discovery_boundary.signal, gate=discovery_boundary.gate,
            current_gate=discovery_boundary.current_gate, gates=discovery_boundary.gates,
            lead=discovery_boundary.lead, lead_for_signal=discovery_boundary.lead_for_signal,
            watch_condition=discovery_boundary.watch_condition,
            disposition=discovery_boundary.disposition,
            current_disposition=discovery_boundary.current_disposition,
            dispositions=discovery_boundary.dispositions,
            signals_for_revision=discovery_boundary.signals_for_revision,
            current_status=discovery_boundary.current_status,
        )

        work_items = TriageWorkItemStore(connection, retrieval_authority)
        executions = _open_on_connection(
            connection, retrieval_authority=retrieval_authority,
            authenticator=authenticator, clock=clock,
            lease_ttl_seconds=lease_ttl_seconds,
        )
        executions._TriageExecutionAuthority__store._transaction_lock = operation_lock
        dispositions = ProposalDispositionStore(connection, retrieval_authority, authenticator)
        hypothesis_store = _HypothesisStore(connection, retrieval_authority, authenticator, clock)
        hypothesis_store._lock = operation_lock

        relationship_store = _share_store(_SharedRelationshipStore, root)
        with relationship_store._hypothesis_rows():
            relationship_store._hypotheses = _HypothesisStore(
                connection, retrieval_authority, authenticator, clock
            )
        relationship_store._hypotheses._lock = operation_lock
        relationship_store._command_service = service
        with operation_lock, relationship_store._transaction():
            relationship_store._adopt()
            try: relationship_store._verify_relationships()
            finally: relationship_store._release()

        lineage_store = _share_store(_SharedLineageStore, root)
        lineage_store._port = _create_event_hypothesis_relationship_read_port(
            connection, retrieval_authority=retrieval_authority,
            authenticator=authenticator, command_registry=commands,
            payload_schemas=schemas, clock=clock,
        )
        lineage_store._service = service
        with operation_lock, lineage_store._transaction(): lineage_store._verify()

        candidate_store = _share_store(_SharedCandidateStore, root)
        candidate_store._retrieval = retrieval_authority
        candidate_store._authenticator = authenticator
        candidate_store._collision = collision_enforcer
        candidate_store._lineage = _create_event_hypothesis_lineage_read_port(
            connection, retrieval_authority=retrieval_authority,
            authenticator=authenticator, command_registry=commands,
            payload_schemas=schemas, clock=clock,
        )
        candidate_store._dispositions = dispositions
        candidate_store._service = service
        with operation_lock, candidate_store._transaction(): candidate_store._verify()

        read_boundary = _ReadBoundary(
            store=root, policy=event_read_policy, authenticator=authenticator,
            authorizer=authorizer, clock=clock,
        )
        def execute(command: SemanticCommand, proof: AuthenticationProof) -> CommittedCommand:
            grant = service._authorize_for_commit(command, proof=proof)
            return root.commit(grant)

        transaction_candidate_port = _create_story_candidate_read_port(
            connection,
            retrieval_authority=retrieval_authority,
            authenticator=authenticator,
            command_registry=commands,
            payload_schemas=schemas,
            clock=clock,
            command_service_version=command_service_version,
        )

        def candidate_version(version_id: str):
            with operation_lock:
                connection.execute("BEGIN")
                try:
                    value = transaction_candidate_port.require_retained_version_in_transaction(
                        version_id
                    )
                    connection.execute("COMMIT")
                    return value
                except BaseException:
                    if connection.in_transaction:
                        connection.execute("ROLLBACK")
                    raise

        candidate_read_port = transaction_candidate_port._with_bounded_version(
            candidate_version
        )

        return HermesNativeAuthoritySystem(
            _SYSTEM_TOKEN, base=base, checks=checks, discovery=discovery,
            work_items=_SharedAuthority(work_items, operation_lock),
            executions=_SharedAuthority(executions, operation_lock),
            proposals=_SharedAuthority(executions, operation_lock),
            dispositions=_SharedAuthority(dispositions, operation_lock),
            hypotheses=_compose_event_hypothesis_authority(_SharedAuthority(hypothesis_store, operation_lock)),
            hypothesis_relationships=_compose_event_hypothesis_relationship_authority(_SharedAuthority(relationship_store, operation_lock)),
            relationships=_compose_event_hypothesis_relationship_authority(_SharedAuthority(relationship_store, operation_lock)),
            hypothesis_lineage=_compose_event_hypothesis_lineage_authority(_SharedAuthority(lineage_store, operation_lock)),
            lineage=_compose_event_hypothesis_lineage_authority(_SharedAuthority(lineage_store, operation_lock)),
            candidates=_compose_story_candidate_authority(_SharedAuthority(candidate_store, operation_lock)),
            build_candidate_manifest=_SharedAuthority(candidate_store, operation_lock).build_manifest,
            candidate_read_port=candidate_read_port,
            candidate_version=candidate_version,
            collision=collision_enforcer,
            commands=AuthorityCommands(execute),
            events=AuthorityEvents(
                policy_id=event_read_policy.policy_id,
                read=read_boundary.events_after,
                provenance=read_boundary.provenance,
                result=read_boundary.command_result,
            ),
        )
    except BaseException:
        base.close()
        raise


__all__ = ["HermesNativeAuthoritySystem", "open_hermes_native_authority_system"]
