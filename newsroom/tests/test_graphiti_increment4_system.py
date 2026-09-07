from __future__ import annotations

import sqlite3
from collections.abc import Callable
from pathlib import Path

import pytest

from newsroom.authority import (
    ObjectAdmissionRequest,
    ObjectLimits,
    StaticAuthenticator,
    StaticAuthorizer,
    StaticPrincipal,
)
from newsroom.authority._graphiti_increment4_system import _open_with_adapter
from newsroom.authority.persistence import (
    AuthorityPersistenceError,
    AuthorityWriterBusy,
)
from newsroom.increment4 import increment4_admitted_contract_registry

from .authority_a2b_helpers import _policy_registries
from .authority_event_helpers import payload_schemas, registry_v1
from .authority_helpers import FIXED_NOW, proof
from .editorial_relation_4c_helpers import (
    RELATION_SCOPES,
    relation_read_policy,
)
from .entity_4b_helpers import ENTITY_SCOPES, entity_read_policy
from .extraction_4a_helpers import (
    contract_request,
    extraction_read_policy,
    extraction_scopes,
)
from .graphiti_adapter_4d_authority_helpers import (
    GRAPHITI_SCOPES,
    graphiti_read_policy,
)
from .increment4e_helpers import (
    INCREMENT4_PROJECTION_SCOPES,
    increment4_projection_read_policy,
)
from .projection_b2_helpers import MemoryNeo4jAdapter
from .source_3a_helpers import (
    definition_request,
    item_request,
    read_policy as source_read_policy,
    scopes as source_scopes,
    version_request,
)


class TrackingMemoryNeo4jAdapter(MemoryNeo4jAdapter):
    close_count = 0

    def close(self) -> None:
        self.close_count += 1
        super().close()


def _open(
    root: Path,
    adapter: MemoryNeo4jAdapter,
    *,
    cas_fault_hook: Callable[[str], None] | None = None,
):
    rights, hydration, admissions = _policy_registries()
    scopes = (
        extraction_scopes()
        | source_scopes()
        | ENTITY_SCOPES
        | RELATION_SCOPES
        | GRAPHITI_SCOPES
        | INCREMENT4_PROJECTION_SCOPES
        | frozenset(
            {
                "authority.fixture.events.read",
                "authority.objects.admit",
                "authority.objects.read",
                "authority.objects.manage",
                "authority.objects.lifecycle.write",
            }
        )
    )
    return _open_with_adapter(
        path=root / "authority.sqlite3",
        object_root=root / "objects",
        workspace_root=root.resolve(),
        registry=registry_v1(),
        payload_schemas=payload_schemas(),
        admission_registry=admissions,
        rights_policies=rights,
        hydration_policies=hydration,
        contracts=increment4_admitted_contract_registry(),
        authenticator=StaticAuthenticator(
            credentials={"token-1": StaticPrincipal("principal.alpha")},
            authority_domain="newsroom.authority",
        ),
        authorizer=StaticAuthorizer(
            policy_version="combined-increment4-test-v1",
            grants_by_principal={"principal.alpha": scopes},
        ),
        source_read_policy=source_read_policy(),
        extraction_read_policy=extraction_read_policy(),
        entity_read_policy=entity_read_policy(),
        relation_read_policy=relation_read_policy(),
        graphiti_read_policy=graphiti_read_policy(),
        projection_read_policy=increment4_projection_read_policy(),
        object_limits=ObjectLimits(
            global_max_bytes=1024 * 1024,
            class_max_bytes={"source_capture": 1024 * 1024},
            max_read_bytes=1024 * 1024,
            min_free_bytes=0,
            io_chunk_bytes=64,
            max_staging_bytes=1024 * 1024,
            max_range_bytes=1024 * 1024,
        ),
        adapter=adapter,
        graph_destination_id="sha256:" + "d" * 64,
        busy_timeout_ms=1,
        clock=lambda: FIXED_NOW,
        cas_fault_hook=cas_fault_hook,
    )


def test_combined_increment4_system_owns_one_writer_and_real_facades(
    tmp_path: Path,
) -> None:
    adapter = TrackingMemoryNeo4jAdapter()
    system = _open(tmp_path, adapter)
    try:
        admission = system.objects.admit(
            ObjectAdmissionRequest("source.capture", "combined-object-1"),
            b"combined-authority",
            proof=proof(),
        )
        request = contract_request()
        contract = system.extraction.register_contract(request, proof=proof())
        definition_request_value = definition_request()
        definition = system.sources.register_definition(
            definition_request_value,
            proof=proof(),
        )
        system.sources.record_definition_version(
            version_request(),
            proof=proof(),
        )
        item_request_value = item_request()
        item = system.sources.register_item(item_request_value, proof=proof())

        assert admission.admission.admission_id is not None
        assert (
            system.sources.definition(
                definition_request_value.definition_id,
                proof=proof(),
            )
            == definition
        )
        assert system.sources.item(item_request_value.item_id, proof=proof()) == item
        assert (
            system.extraction.contract(request.contract_id, proof=proof())
            == contract
        )
        assert system.graphiti is not None
        assert system.entities is not None
        assert system.relations is not None
        assert system.increment4 is not None
        assert system.structural is not None
        assert system.compatibility == adapter.verify_compatibility()
        assert system.authority_store_path == str(
            (tmp_path / "authority.sqlite3").resolve()
        )
        assert system.graph_destination_id == "sha256:" + "d" * 64
        with pytest.raises(AttributeError):
            system.graphiti = object()  # type: ignore[misc]
        assert adapter.bootstrap_count == 1

        second_adapter = TrackingMemoryNeo4jAdapter()
        with pytest.raises(AuthorityWriterBusy):
            _open(tmp_path, second_adapter)
        assert second_adapter.closed is True
        assert second_adapter.close_count == 1
    finally:
        system.close()
        system.close()

    assert adapter.closed is True
    assert adapter.close_count == 1


def test_combined_increment4_system_closes_adapter_on_cas_construction_failure(
    tmp_path: Path,
) -> None:
    adapter = TrackingMemoryNeo4jAdapter()

    def fail_cas(_checkpoint: str) -> None:
        raise RuntimeError("CAS construction failed")

    with pytest.raises(RuntimeError, match="CAS construction failed"):
        _open(tmp_path, adapter, cas_fault_hook=fail_cas)

    assert adapter.closed is True
    assert adapter.close_count == 1


def _seed_combined_increment4_store(tmp_path: Path) -> Path:
    adapter = TrackingMemoryNeo4jAdapter()
    system = _open(tmp_path, adapter)
    try:
        system.objects.admit(
            ObjectAdmissionRequest("source.capture", "combined-object-1"),
            b"combined-authority",
            proof=proof(),
        )
        system.sources.register_definition(definition_request(), proof=proof())
    finally:
        system.close()
    return tmp_path / "authority.sqlite3"


def _tamper_restoring_trigger(
    database: Path, *, trigger: str, sql: str, parameters: tuple[object, ...]
) -> None:
    connection = sqlite3.connect(database)
    try:
        trigger_sql = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type='trigger' AND name=?",
            (trigger,),
        ).fetchone()[0]
        connection.execute(f'DROP TRIGGER "{trigger}"')
        connection.execute(sql, parameters)
        connection.execute(trigger_sql)
        connection.commit()
    finally:
        connection.close()


def test_increment4_open_revalidates_clean_store(tmp_path: Path) -> None:
    _seed_combined_increment4_store(tmp_path)
    reopened = _open(tmp_path, TrackingMemoryNeo4jAdapter())
    reopened.close()


@pytest.mark.parametrize(
    ("trigger", "sql", "parameters", "match"),
    [
        (
            "immutable_authentication_contexts_update",
            "UPDATE authentication_contexts SET canonical_bytes=?",
            (b'{"tampered":true}',),
            "authentication",
        ),
        (
            "immutable_authorization_requests_update",
            "UPDATE authorization_requests SET canonical_bytes=?",
            (b'{"tampered":true}',),
            "request",
        ),
        (
            "immutable_authorization_decisions_update",
            "UPDATE authorization_decisions SET canonical_bytes=?",
            (b'{"tampered":true}',),
            "decision",
        ),
        (
            "immutable_authority_commands_update",
            "UPDATE authority_commands SET result_bytes=?",
            (b'{"tampered":true}',),
            "result digest",
        ),
        (
            "immutable_source_definition_update",
            "UPDATE source_definitions SET canonical_digest=?",
            ("sha256:" + "0" * 64,),
            "canonical digest",
        ),
    ],
    ids=(
        "authentication",
        "request",
        "decision",
        "result",
        "source-definition",
    ),
)
def test_increment4_open_refuses_corrupt_immutable_and_domain_rows(
    tmp_path: Path,
    trigger: str,
    sql: str,
    parameters: tuple[object, ...],
    match: str,
) -> None:
    database = _seed_combined_increment4_store(tmp_path)
    _tamper_restoring_trigger(
        database, trigger=trigger, sql=sql, parameters=parameters
    )
    with pytest.raises(AuthorityPersistenceError, match=match):
        _open(tmp_path, TrackingMemoryNeo4jAdapter())
