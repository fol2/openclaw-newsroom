from __future__ import annotations

import sqlite3
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from newsroom.authority import (
    AuthenticationProof,
    ObjectLimits,
    StaticAuthenticator,
    StaticAuthorizer,
    StaticPrincipal,
    UtcTimestamp,
)
from newsroom.authority._graphiti_increment4_system import _open_with_adapter
from newsroom.authority.canonical import digest_canonical
from newsroom.authority.persistence import AuthorityPersistenceError
from newsroom.control_plane.corpus import CorpusAuthorityBinding, CorpusIngestUnit
from newsroom.control_plane.graphiti_operational_readiness import (
    GraphitiOperationalReadinessError,
    OperationalAuthorityBootstrapPlan,
    _accepted_source_contract,
    _evaluation_attempt_for_unit,
    _operational_graphiti_write_scopes,
    _operational_generation_identity,
    _operational_plan_semantic_digest,
    _preflight_source_revision_semantics,
    _revision_predecessor_bindings,
    _source_contract_shape,
    _source_requests,
    bootstrap_operational_authority,
    build_and_reconcile_operational_generation,
    build_operational_campaign_input,
    operational_policy_components,
    plan_operational_authority_bootstrap,
    reopen_operational_campaign_runtime,
)
from newsroom.control_plane.graphiti_steady_state import (
    campaign_event_limits,
    graphiti_graph_destination_readback,
    graphiti_store_snapshot_digests,
)
from newsroom.effective_revision import EffectiveRevisionIdentity
from newsroom.graphiti_adapter.evaluation_packet import (
    GRAPHITI_EXTRACTION_TIMEOUT_MS,
)
from newsroom.graphiti_adapter.identity import (
    content_digest,
    observation_authority_ids,
    source_definition_version_id,
)
from newsroom.graphiti_adapter.policy import graphiti_adapter_command_definitions
from newsroom.increment4 import increment4_admitted_contract_registry
from newsroom.sources.types import (
    BaselinePolicyKind,
    CoverageContribution,
    CoverageResponsibility,
    ObservationModel,
    PortfolioFunction,
    SourceDependencyKind,
    SourceLifecycleStage,
    SourceRevisionId,
    SourceRole,
    SourceTime,
)
from scripts.hermes_graphiti_worker import GraphitiCampaignStop

from .authority_event_helpers import payload_schemas, registry_v1
from .editorial_relation_4c_helpers import relation_read_policy
from .entity_4b_helpers import entity_read_policy
from .extraction_4a_helpers import extraction_read_policy
from .graphiti_adapter_4d_authority_helpers import graphiti_read_policy
from .increment4e_helpers import increment4_projection_read_policy
from .projection_b2_helpers import MemoryNeo4jAdapter
from .source_3a_helpers import read_policy as source_read_policy
from .source_3a_helpers import scopes as source_scopes

_NOW = "2026-09-02T12:00:00.000000Z"
_REOPENED_NOW = "2026-09-02T12:05:00.000000Z"
_PROOF = AuthenticationProof(method="STATIC_TOKEN", credential="token-1")


def test_operational_graphiti_write_scopes_follow_canonical_commands() -> None:
    expected = frozenset(
        definition.required_scope
        for definition in graphiti_adapter_command_definitions()
    )

    assert _operational_graphiti_write_scopes() == expected
    assert expected == {
        "authority.graphiti.configuration",
        "authority.graphiti.execute",
        "authority.graphiti.replay.approve",
    }


def _unit() -> CorpusIngestUnit:
    source_id = "UK-01"
    item_key = "current-item"
    headline = "Current headline"
    body = "Current body with exact retained bytes."
    canonical_url = "https://example.test/current-item"
    revision_digest = content_digest(
        headline=headline,
        body=body,
        canonical_url=canonical_url,
    )
    effective_revision = EffectiveRevisionIdentity(
        source_id=source_id,
        item_key=item_key,
        revision_digest=revision_digest,
        first_observed_at=_NOW,
    )
    base = CorpusIngestUnit(
        source_id=source_id,
        item_key=item_key,
        headline=headline,
        body=body,
        canonical_url=canonical_url,
        observation_digest=digest_canonical({"observation": "current"}),
        observed_at=_NOW,
        proving_run_id="rights-run-1",
        effective_revision=effective_revision,
        published_at="2026-09-02T10:00:00.000000Z",
        source_definition_url="https://example.test/feed",
        effective_pull_first_observed_at=_NOW,
    )
    (
        admission_id,
        access_id,
        definition_id,
        item_id,
        revision_id,
        representation_id,
    ) = observation_authority_ids(
        source_id=source_id,
        item_key=item_key,
        revision_digest=base.revision_digest,
        representation_digest=base.representation_digest,
        rights_authority_run_id="rights-run-1",
        rights_gate_id="RIGHTS_UK-01",
        rights_gate_reason="retained PASS",
        published_at=base.published_at,
        updated_at=base.updated_at,
    )
    version_id = source_definition_version_id(
        source_id=source_id,
        source_url=base.source_definition_url,
    )
    records = (
        {
            "record_type": "SOURCE_DEFINITION",
            "record_id": str(definition_id),
        },
        {
            "record_type": "SOURCE_DEFINITION_VERSION",
            "record_id": str(version_id),
        },
        {"record_type": "SOURCE_ITEM", "record_id": str(item_id)},
        {"record_type": "SOURCE_REVISION", "record_id": str(revision_id)},
        {
            "record_type": "DISCOVERY_REPRESENTATION",
            "record_id": str(representation_id),
        },
        {"record_type": "OBJECT_ADMISSION", "record_id": str(admission_id)},
        {"record_type": "OBJECT_ACCESS_DECISION", "record_id": str(access_id)},
    )
    return replace(
        base,
        authority=CorpusAuthorityBinding(
            admission_id=str(admission_id),
            access_decision_id=str(access_id),
            definition_id=str(definition_id),
            definition_version_id=str(version_id),
            item_id=str(item_id),
            revision_id=str(revision_id),
            representation_id=str(representation_id),
            records=records,
        ),
    )


def _next_revision(unit: CorpusIngestUnit) -> CorpusIngestUnit:
    body = "Later current body with a materially changed exact revision."
    observed_at = "2026-09-02T12:30:00.000000Z"
    updated_at = "2026-09-02T12:20:00.000000Z"
    revision_digest = content_digest(
        headline=unit.headline,
        body=body,
        canonical_url=unit.canonical_url,
    )
    base = replace(
        unit,
        body=body,
        updated_at=updated_at,
        effective_revision=EffectiveRevisionIdentity(
            source_id=unit.source_id,
            item_key=unit.item_key,
            revision_digest=revision_digest,
            first_observed_at=observed_at,
        ),
        effective_pull_first_observed_at=observed_at,
        authority=None,
    )
    (
        admission_id,
        access_id,
        definition_id,
        item_id,
        revision_id,
        representation_id,
    ) = observation_authority_ids(
        source_id=base.source_id,
        item_key=base.item_key,
        revision_digest=base.revision_digest,
        representation_digest=base.representation_digest,
        rights_authority_run_id="rights-run-1",
        rights_gate_id="RIGHTS_UK-01",
        rights_gate_reason="retained PASS",
        published_at=base.published_at,
        updated_at=base.updated_at,
    )
    version_id = source_definition_version_id(
        source_id=base.source_id,
        source_url=base.source_definition_url,
    )
    records = (
        {"record_type": "SOURCE_DEFINITION", "record_id": str(definition_id)},
        {
            "record_type": "SOURCE_DEFINITION_VERSION",
            "record_id": str(version_id),
        },
        {"record_type": "SOURCE_ITEM", "record_id": str(item_id)},
        {"record_type": "SOURCE_REVISION", "record_id": str(revision_id)},
        {
            "record_type": "DISCOVERY_REPRESENTATION",
            "record_id": str(representation_id),
        },
        {"record_type": "OBJECT_ADMISSION", "record_id": str(admission_id)},
        {"record_type": "OBJECT_ACCESS_DECISION", "record_id": str(access_id)},
    )
    return replace(
        base,
        authority=CorpusAuthorityBinding(
            admission_id=str(admission_id),
            access_decision_id=str(access_id),
            definition_id=str(definition_id),
            definition_version_id=str(version_id),
            item_id=str(item_id),
            revision_id=str(revision_id),
            representation_id=str(representation_id),
            records=records,
        ),
    )


def _rights() -> dict[str, object]:
    return {
        "rights_authority_run_id": "rights-run-1",
        "gate_id": "RIGHTS_UK-01",
        "status": "PASS",
        "packet_digest": digest_canonical({"rights": "current"}),
        "evaluated_at": _NOW,
    }


def _plan(unit: CorpusIngestUnit) -> OperationalAuthorityBootstrapPlan:
    event = {
        "kind": "FRESH_EVENT",
        "ledger_seq": 1,
        "event_id": digest_canonical({"event": "current"}),
        "manifest_digest": digest_canonical({"manifest": "current"}),
        "ingest_ids": [unit.ingest_id],
    }
    return OperationalAuthorityBootstrapPlan(
        observed_at=UtcTimestamp.parse(_NOW),
        partition_snapshot={"snapshot_digest": digest_canonical({"snapshot": 1})},
        candidate_events=(event,),
        units=(unit,),
        rights_by_source=((unit.source_id, _rights()),),
        revision_predecessors=((unit.revision_id, None),),
        cohort_digest=digest_canonical({"cohort": "current"}),
        plan_digest=digest_canonical({"plan": "current"}),
    )


def _open_operational_test_system(
    root: Path,
    adapter: MemoryNeo4jAdapter | None = None,
    *,
    now: str = _NOW,
):
    policies = operational_policy_components()
    scopes = source_scopes() | _operational_graphiti_write_scopes() | frozenset(
        {
            "authority.objects.admit",
            "authority.objects.read",
            "authority.objects.manage",
            "authority.objects.lifecycle.write",
            "authority.observed.write",
            "authority.admitted.write",
            "authority.extraction.execute",
            "authority.extraction.manage",
            "authority.extraction.metadata.read",
            "authority.extraction.proposal.read",
            "authority.extraction.raw.read",
            "authority.entity.propose",
            "authority.entity.admit",
            "authority.entity.proposal.read",
            "authority.entity.admitted.read",
            "authority.entity.projection.read",
            "authority.relation.propose",
            "authority.relation.admit",
            "authority.relation.proposal.read",
            "authority.relation.admitted.read",
            "authority.relation.projection.read",
            "authority.graphiti.attempt.read",
            "authority.graphiti.configuration.read",
            "authority.graphiti.replay.read",
            "authority.projection.manage",
            "authority.projection.write",
            "authority.projection.read",
        }
    )
    return _open_with_adapter(
        path=root / "authority.sqlite3",
        object_root=root / "objects",
        workspace_root=root.resolve(),
        registry=registry_v1(),
        payload_schemas=payload_schemas(),
        admission_registry=policies["admission_registry"],
        rights_policies=policies["rights_policies"],
        hydration_policies=policies["hydration_policies"],
        contracts=increment4_admitted_contract_registry(),
        authenticator=StaticAuthenticator(
            credentials={"token-1": StaticPrincipal("newsroom.control-plane")},
            authority_domain="newsroom.evaluation",
        ),
        authorizer=StaticAuthorizer(
            policy_version="operational-readiness-test-v1",
            grants_by_principal={"newsroom.control-plane": scopes},
        ),
        source_read_policy=source_read_policy(),
        extraction_read_policy=extraction_read_policy(),
        entity_read_policy=entity_read_policy(),
        relation_read_policy=relation_read_policy(),
        graphiti_read_policy=graphiti_read_policy(),
        projection_read_policy=replace(
            increment4_projection_read_policy(),
            allowed_principal_ids=frozenset({"newsroom.control-plane"}),
        ),
        object_limits=ObjectLimits(
            global_max_bytes=1024 * 1024,
            class_max_bytes={"source.expression": 1024 * 1024},
            max_read_bytes=1024 * 1024,
            min_free_bytes=0,
            max_staging_bytes=1024 * 1024,
            max_range_bytes=1024 * 1024,
        ),
        adapter=adapter or MemoryNeo4jAdapter(),
        graph_destination_id=digest_canonical({"graph": "test"}),
        clock=lambda: UtcTimestamp.parse(now),
    )


def test_campaign_input_is_dormant_exact_bounded_machine_contract() -> None:
    campaign = build_operational_campaign_input(
        head_sha="a" * 40,
        tree_sha="b" * 40,
        focus_manifest_digest=digest_canonical({"focus": "exact-main"}),
        graph_destination_id=digest_canonical({"graph": "projector"}),
        candidate_event_count=25,
        recovery_identity=digest_canonical({"recovery": "authority"}),
    )

    assert campaign["campaign_authorised"] is False
    assert campaign["focus_gate"]["head_sha"] == "a" * 40
    assert campaign["selection_policy"] == {
        "policy_id": "graphiti-operational-current-cohort",
        "policy_version": "v1",
    }
    assert campaign["provider"] == {
        "provider_id": "cursor-agent-cli",
        "transport_id": "CURSOR_SDK",
        "model_id": "composer-2.5",
        "embedding_provider_id": "openrouter",
        "embedding_model_id": "openai/text-embedding-3-large",
    }
    assert campaign["caps"] == {
        "per_event": {
            "proposals": 100,
            "entity_admits": 100,
            "relation_admits": 100,
            "effects": 200,
            "retries": 0,
            "fallbacks": 0,
        },
        "total": {
            "events": 25,
            "proposals": 2_500,
            "entity_admits": 2_500,
            "relation_admits": 2_500,
            "effects": 5_000,
            "retries": 0,
            "fallbacks": 0,
            "wall_time_seconds": max(
                600, 25 * (GRAPHITI_EXTRACTION_TIMEOUT_MS // 1000)
            ),
            "spend_gbp_microunits": 12_500_000,
        },
        "rate": {"events_per_minute": 1},
    }
    assert [phase["event_limit"] for phase in campaign["ramp"]["phases"]] == [
        1,
        10,
        25,
    ]
    assert campaign["recovery"]["backup_identity"].startswith("sha256:")
    assert "PROVIDER_FAILURE" in campaign["immediate_stop_conditions"]


@pytest.mark.parametrize(
    ("event_count", "limits"),
    ((1, (1,)), (2, (1, 2)), (10, (1, 10)), (25, (1, 10, 25))),
)
def test_campaign_event_limits_collapse_to_unique_increasing_bounds(
    event_count: int, limits: tuple[int, ...]
) -> None:
    assert campaign_event_limits(event_count) == limits


def test_source_requests_bind_exact_retained_identity_and_rights() -> None:
    unit = _unit()
    definition, version, item, revision, representation = _source_requests(
        unit, _rights()
    )
    assert str(definition.definition_id) == unit.authority.definition_id
    assert definition.name == "UK-01 Home Office + UKVI Atom"
    assert definition.editorial_purpose == (
        "Observe Home Office and UKVI immigration, status and guidance updates."
    )
    assert str(version.version_id) == unit.authority.definition_version_id
    assert str(item.item_id) == unit.authority.item_id
    assert str(revision.revision_id) == unit.authority.revision_id
    assert str(representation.representation_id) == unit.authority.representation_id
    assert version.locator == unit.source_definition_url
    assert revision.permitted_state_digest == unit.revision_digest
    assert representation.representation_digest == unit.representation_digest
    assert version.rights.allowed_use == "proposal.extraction"
    assert version.rights.rights_policy_version == "control-plane-dispatch-rights-v1"
    assert version.adapter_contract.canonical_value() == {
        "policy_id": "control-plane-retained-source-adapter",
        "policy_version": "v1",
    }
    assert version.baseline_policy.reference.canonical_value() == {
        "policy_id": "operational-current-cohort-baseline",
        "policy_version": "v1",
    }
    assert version.baseline_policy.freshness_window_seconds == 7 * 24 * 60 * 60
    assert version.item_identity_policy.policy_id == "source-id-item-key-composite"
    assert version.revision_policy.policy_id == "effective-revision-identity"
    assert (
        version.canonicalization_policy.policy_id
        == "graphiti-corpus-canonicalization"
    )
    assert representation.adapter_version == (
        "control-plane-retained-source-adapter-v1"
    )
    assert representation.parser_version == "control-plane-parse-observation-v1"
    assert representation.normalizer_version == (
        "graphiti-corpus-canonicalization-v1"
    )
    assert representation.extraction_scope_version == (
        "operational-graphiti-passage-fields-v1"
    )
    assert len(version.coverage_mappings) == 1
    coverage = version.coverage_mappings[0]
    assert coverage.obligation_id == "COV-020"
    assert coverage.responsibility is CoverageResponsibility.ACTIVE
    assert coverage.contribution is CoverageContribution.DETECTION_PATH
    assert coverage.geographies == ("UK",)
    assert coverage.languages == ("en-GB",)
    assert coverage.limitations == version.roles[0].limitations
    assert version.explicit_gaps == ()
    assert [dependency.kind for dependency in version.dependencies] == [
        SourceDependencyKind.ORIGINATING_MATERIAL
    ]
    assert version.observation_model is ObservationModel.ROLLING_LIST
    assert version.baseline_policy.kind is BaselinePolicyKind.BOUNDED_BACKFILL


@pytest.mark.parametrize(
    (
        "source_id",
        "name",
        "role",
        "function",
        "dependency_kind",
        "observation_model",
        "baseline_kind",
        "lifecycle_stage",
        "limitation",
        "coverage_obligation_id",
        "coverage_responsibility",
        "coverage_contribution",
        "coverage_geography",
        "coverage_language",
    ),
    [
        (
            "UK-01",
            "UK-01 Home Office + UKVI Atom",
            SourceRole.ORIGINATING_AUTHORITY,
            PortfolioFunction.ANCHOR,
            SourceDependencyKind.ORIGINATING_MATERIAL,
            ObservationModel.ROLLING_LIST,
            BaselinePolicyKind.BOUNDED_BACKFILL,
            SourceLifecycleStage.SHADOW_SHORTLISTED,
            "Entry metadata still needs maintained-page inspection.",
            "COV-020",
            CoverageResponsibility.ACTIVE,
            CoverageContribution.DETECTION_PATH,
            "UK",
            "en-GB",
        ),
        (
            "UK-10",
            "UK-10 Met Office warnings",
            SourceRole.ORIGINATING_AUTHORITY,
            PortfolioFunction.ANCHOR,
            None,
            ObservationModel.COMPLETE_CURRENT_STATE,
            BaselinePolicyKind.COMPLETE_STATE_FIRST_OBSERVED_ACTIVE,
            SourceLifecycleStage.SHADOW_SHORTLISTED,
            "Regional and transition semantics remain Topic 5 work.",
            "COV-023",
            CoverageResponsibility.ACTIVE,
            CoverageContribution.URGENT_FAST_PATH,
            "UK",
            "en-GB",
        ),
        (
            "HK-01",
            "HK-01 news.gov.hk top stories",
            SourceRole.SPECIALIST_OR_LOCAL_RADAR,
            PortfolioFunction.ANCHOR,
            SourceDependencyKind.EDITORIAL_SELECTION,
            ObservationModel.ROLLING_LIST,
            BaselinePolicyKind.BOUNDED_BACKFILL,
            SourceLifecycleStage.SHADOW_SHORTLISTED,
            "Curated, not the full government-release universe.",
            "COV-012",
            CoverageResponsibility.ACTIVE,
            CoverageContribution.DETECTION_PATH,
            "Hong Kong",
            "zh-HK",
        ),
        (
            "RAD-01",
            "RAD-01 RTHK local news",
            SourceRole.ESTABLISHED_MEDIA_RADAR,
            PortfolioFunction.COMPARATOR,
            SourceDependencyKind.EDITORIAL_SELECTION,
            ObservationModel.ROLLING_LIST,
            BaselinePolicyKind.BOUNDED_BACKFILL,
            SourceLifecycleStage.COMPARATOR_ONLY,
            "Lead-only discovery role.",
            "COV-012",
            CoverageResponsibility.EVALUATION,
            CoverageContribution.COMPARATOR,
            "Hong Kong",
            "zh-HK",
        ),
        (
            "HK-04",
            "HK-04 Education Bureau latest news",
            SourceRole.ORIGINATING_AUTHORITY,
            PortfolioFunction.ANCHOR,
            None,
            ObservationModel.ROLLING_LIST,
            BaselinePolicyKind.BOUNDED_BACKFILL,
            SourceLifecycleStage.SHADOW_SHORTLISTED,
            "No school-level completeness.",
            "COV-012",
            CoverageResponsibility.ACTIVE,
            CoverageContribution.DETECTION_PATH,
            "Hong Kong",
            "zh-HK",
        ),
        (
            "RAD-02",
            "RAD-02 BBC UK news",
            SourceRole.ESTABLISHED_MEDIA_RADAR,
            PortfolioFunction.COMPARATOR,
            SourceDependencyKind.EDITORIAL_SELECTION,
            ObservationModel.ROLLING_LIST,
            BaselinePolicyKind.BOUNDED_BACKFILL,
            SourceLifecycleStage.COMPARATOR_ONLY,
            "Broad, duplicate-prone and weak on local completeness.",
            "COV-010",
            CoverageResponsibility.EVALUATION,
            CoverageContribution.COMPARATOR,
            "UK",
            "en-GB",
        ),
    ],
)
def test_current_sources_retain_their_accepted_contract(
    source_id: str,
    name: str,
    role: SourceRole,
    function: PortfolioFunction,
    dependency_kind: SourceDependencyKind | None,
    observation_model: ObservationModel,
    baseline_kind: BaselinePolicyKind,
    lifecycle_stage: SourceLifecycleStage,
    limitation: str,
    coverage_obligation_id: str,
    coverage_responsibility: CoverageResponsibility,
    coverage_contribution: CoverageContribution,
    coverage_geography: str,
    coverage_language: str,
) -> None:
    contract = _accepted_source_contract(source_id)

    assert _source_contract_shape(source_id) == (role, function)
    assert contract.name == name
    assert contract.role is role
    assert contract.function is function
    assert contract.observation_model is observation_model
    assert contract.baseline_kind is baseline_kind
    assert contract.lifecycle_stage is lifecycle_stage
    assert limitation in contract.limitations
    assert contract.coverage_obligation_id == coverage_obligation_id
    assert contract.coverage_responsibility is coverage_responsibility
    assert contract.coverage_contribution is coverage_contribution
    assert contract.coverage_geographies == (coverage_geography,)
    assert contract.coverage_languages == (coverage_language,)
    assert [dependency.kind for dependency in contract.dependencies] == (
        [] if dependency_kind is None else [dependency_kind]
    )


def test_source_contract_fails_closed_outside_accepted_current_cohort() -> None:
    with pytest.raises(
        GraphitiOperationalReadinessError,
        match="outside the accepted operational cohort",
    ):
        _source_contract_shape("UK-99")


@pytest.mark.parametrize(
    ("actionable", "loaded_units", "message"),
    [
        (
            [
                {"kind": "FRESH_EVENT", "ingest_ids": ["same"]},
                {"kind": "FRESH_EVENT", "ingest_ids": ["same"]},
            ],
            (),
            "empty or duplicated",
        ),
        (
            [{"kind": "FRESH_EVENT", "ingest_ids": ["unknown"]}],
            (),
            "differ from current retained input",
        ),
        (
            [{"kind": "EVENT_GAP", "ingest_ids": ["unknown"]}],
            (),
            "event gaps",
        ),
    ],
)
def test_bootstrap_plan_fails_closed_on_event_membership_drift(
    monkeypatch: pytest.MonkeyPatch,
    actionable: list[dict[str, object]],
    loaded_units: tuple[CorpusIngestUnit, ...],
    message: str,
) -> None:
    monkeypatch.setattr(
        "newsroom.control_plane.graphiti_operational_readiness."
        "graphiti_operational_partition_snapshot",
        lambda *_args, **_kwargs: {
            "actionable": actionable,
            "snapshot_digest": digest_canonical({"partition": "test"}),
        },
    )
    monkeypatch.setattr(
        "newsroom.control_plane.graphiti_operational_readiness."
        "load_graphiti_units_from_connection",
        lambda *_args, **_kwargs: loaded_units,
    )
    proving = sqlite3.connect(":memory:")
    unpublished = sqlite3.connect(":memory:")
    authority = sqlite3.connect(":memory:")
    try:
        with pytest.raises(GraphitiOperationalReadinessError, match=message):
            plan_operational_authority_bootstrap(
                proving,
                unpublished,
                authority,
                observed_at=datetime(2026, 9, 2, 12, tzinfo=UTC),
            )
    finally:
        proving.close()
        unpublished.close()
        authority.close()


def test_bootstrap_plan_cohort_contains_frontier_not_old_bindable_queued(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from newsroom.control_plane.graphiti_steady_state import (
        PRE_FRONTIER_BACKLOG_HOLD_REASON,
        graphiti_operational_partition_snapshot,
    )
    from newsroom.tests.test_graphiti_steady_state import (
        NOW,
        _current_unit,
        _nonterminal_obligation,
        _queued_rows,
        _stores,
    )

    old_unit = _unit()
    frontier_unit = _next_revision(_unit())
    proving, _unpublished, connection = _stores(tmp_path)
    _nonterminal_obligation(
        connection,
        ledger_seq=1,
        item_key="old-backlog",
        ingest_id=old_unit.ingest_id,
    )
    _nonterminal_obligation(
        connection,
        ledger_seq=2,
        item_key="frontier",
        ingest_id=frontier_unit.ingest_id,
    )
    connection.commit()
    monkeypatch.setattr(
        "newsroom.control_plane.graphiti_steady_state."
        "load_graphiti_units_from_connection",
        lambda *_args, **_kwargs: (
            _current_unit(
                item_key="old-backlog", ingest_id=old_unit.ingest_id
            ),
            _current_unit(
                item_key="frontier", ingest_id=frontier_unit.ingest_id
            ),
        ),
    )
    monkeypatch.setattr(
        "newsroom.control_plane.graphiti_operational_readiness."
        "load_graphiti_units_from_connection",
        lambda *_args, **_kwargs: (old_unit, frontier_unit),
    )
    monkeypatch.setattr(
        "newsroom.control_plane.graphiti_operational_readiness."
        "_dispatch_rights_decision",
        lambda *_args, **_kwargs: _rights(),
    )
    proving_connection = sqlite3.connect(proving)
    authority = sqlite3.connect(":memory:")
    try:
        snapshot = graphiti_operational_partition_snapshot(
            proving_connection,
            connection,
            authority=authority,
            observed_at=NOW,
        )
        plan = plan_operational_authority_bootstrap(
            proving_connection,
            connection,
            authority,
            observed_at=NOW,
        )
    finally:
        proving_connection.close()
        authority.close()
        connection.close()

    assert [item["ledger_seq"] for item in snapshot["actionable"]] == [2]
    assert snapshot["holds"][0]["reason"] == PRE_FRONTIER_BACKLOG_HOLD_REASON
    assert [item["ledger_seq"] for item in plan.candidate_events] == [2]
    assert list(plan.candidate_events[0]["ingest_ids"]) == [frontier_unit.ingest_id]
    assert [unit.ingest_id for unit in plan.units] == [frontier_unit.ingest_id]
    unpublished = sqlite3.connect(_unpublished)
    try:
        assert _queued_rows(unpublished) == [(1, "QUEUED", 0), (2, "QUEUED", 0)]
    finally:
        unpublished.close()


def test_source_requests_reject_missing_retained_or_rights_identity() -> None:
    unit = _unit()
    with pytest.raises(GraphitiOperationalReadinessError, match="retained source"):
        _source_requests(replace(unit, authority=None), _rights())
    with pytest.raises(GraphitiOperationalReadinessError, match="rights evidence"):
        _source_requests(unit, {"packet_digest": ""})


def test_prewrite_rejects_timestamp_only_reobservation_as_a_second_revision() -> None:
    unit = _unit()
    first = _source_requests(unit, _rights())[3]
    reobserved = replace(
        first,
        revision_id=SourceRevisionId.new(),
        source_published_time=SourceTime.exact(
            UtcTimestamp.parse("2026-09-02T11:00:00.000000Z")
        ),
        observed_at=UtcTimestamp.parse("2026-09-02T13:00:00.000000Z"),
        idempotency_key="issue895-source-revision:timestamp-reobservation",
    )
    assert first.revision_id != reobserved.revision_id
    assert first.revision_identity_digest == reobserved.revision_identity_digest

    authority = sqlite3.connect(":memory:")
    try:
        with pytest.raises(
            GraphitiOperationalReadinessError,
            match="multiple SourceRevision identities",
        ):
            _preflight_source_revision_semantics(authority, (first, reobserved))
    finally:
        authority.close()


def test_bootstrap_uses_real_source_and_object_authority_and_replays(
    tmp_path: Path,
) -> None:
    unit = _unit()
    plan = _plan(unit)
    system = _open_operational_test_system(tmp_path)
    try:
        first, first_binder = bootstrap_operational_authority(
            system, proof=_PROOF, plan=plan
        )
        rebound = first_binder(unit)
    finally:
        system.close()

    with sqlite3.connect(tmp_path / "authority.sqlite3") as connection:
        before = connection.execute(
            "SELECT (SELECT COUNT(*) FROM ledger_events),"
            "(SELECT COUNT(*) FROM object_access_decisions)"
        ).fetchone()

    reopened = _open_operational_test_system(tmp_path)
    try:
        second, second_binder = bootstrap_operational_authority(
            reopened, proof=_PROOF, plan=plan
        )
        replay_rebound = second_binder(unit)
    finally:
        reopened.close()

    assert first.provider_calls == second.provider_calls == 0
    assert first.bound_units[0].ingest_id == unit.ingest_id
    assert (
        first.bound_units[0].authority.admission_id
        == second.bound_units[0].authority.admission_id
    )
    assert rebound.authority == first.bound_units[0].authority
    assert replay_rebound.authority == rebound.authority
    assert rebound.authority.admission_id != unit.authority.admission_id
    assert rebound.authority.access_decision_id != unit.authority.access_decision_id
    assert {record["record_type"] for record in rebound.authority.records} == {
        "SOURCE_DEFINITION",
        "SOURCE_DEFINITION_VERSION",
        "SOURCE_ITEM",
        "SOURCE_REVISION",
        "DISCOVERY_REPRESENTATION",
        "OBJECT_ADMISSION",
        "OBJECT_ACCESS_DECISION",
    }

    with sqlite3.connect(tmp_path / "authority.sqlite3") as connection:
        after = connection.execute(
            "SELECT (SELECT COUNT(*) FROM ledger_events),"
            "(SELECT COUNT(*) FROM object_access_decisions)"
        ).fetchone()
        assert (
            connection.execute("SELECT COUNT(*) FROM source_definitions").fetchone()[0]
            == 1
        )
        assert (
            connection.execute("SELECT COUNT(*) FROM source_revisions").fetchone()[0]
            == 1
        )
        assert (
            connection.execute("SELECT COUNT(*) FROM object_admissions").fetchone()[0]
            == 1
        )
        assert connection.execute(
            "SELECT COUNT(*) FROM extractor_contracts WHERE producer_kind='GRAPHITI_EVALUATION'"
        ).fetchone()[0] == 1
        assert connection.execute(
            "SELECT COUNT(*) FROM graphiti_adapter_configurations "
            "WHERE runtime_mode='REAL_GRAPHITI' AND execution_profile='EVALUATION'"
        ).fetchone()[0] == 1
    assert before == after
    assert after[1] == 1


def test_bootstrap_binder_preserves_current_occurrence_metadata(
    tmp_path: Path,
) -> None:
    unit = _unit()
    system = _open_operational_test_system(tmp_path)
    try:
        bootstrap, binder = bootstrap_operational_authority(
            system, proof=_PROOF, plan=_plan(unit)
        )
        current = replace(
            unit,
            observation_digest=digest_canonical({"observation": "rolling"}),
            observed_at="2026-09-02T12:15:00.000000Z",
            proving_run_id="rights-run-2",
        )

        rebound = binder(current)

        assert rebound.authority == bootstrap.bound_units[0].authority
        assert rebound.observation_digest == current.observation_digest
        assert rebound.observed_at == current.observed_at
        assert rebound.proving_run_id == current.proving_run_id
        with pytest.raises(
            GraphitiOperationalReadinessError,
            match="runtime unit differs from the bootstrapped exact cohort",
        ):
            binder(
                replace(
                    current,
                    source_definition_url="https://example.test/drifted-feed",
                )
            )
    finally:
        system.close()


def test_bootstrap_orders_multiple_revisions_and_resumes_after_first_revision(
    tmp_path: Path,
) -> None:
    first = _unit()
    second = _next_revision(first)
    bindings = _revision_predecessor_bindings((second, first))
    assert dict(bindings) == {
        first.revision_id: None,
        second.revision_id: first.revision_id,
    }

    initial = _open_operational_test_system(tmp_path)
    try:
        bootstrap_operational_authority(initial, proof=_PROOF, plan=_plan(first))
    finally:
        initial.close()

    full_plan = replace(
        _plan(first),
        candidate_events=(
            {
                "kind": "FRESH_EVENT",
                "ledger_seq": 1,
                "event_id": digest_canonical({"event": "first"}),
                "manifest_digest": digest_canonical({"manifest": "first"}),
                "ingest_ids": [first.ingest_id],
            },
            {
                "kind": "FRESH_EVENT",
                "ledger_seq": 2,
                "event_id": digest_canonical({"event": "second"}),
                "manifest_digest": digest_canonical({"manifest": "second"}),
                "ingest_ids": [second.ingest_id],
            },
        ),
        units=(second, first),
        revision_predecessors=bindings,
        cohort_digest=digest_canonical({"cohort": "two-revisions"}),
        plan_digest=digest_canonical({"plan": "two-revisions"}),
    )
    resumed = _open_operational_test_system(tmp_path)
    try:
        bootstrap_operational_authority(resumed, proof=_PROOF, plan=full_plan)
    finally:
        resumed.close()
    replayed = _open_operational_test_system(tmp_path)
    try:
        bootstrap_operational_authority(replayed, proof=_PROOF, plan=full_plan)
    finally:
        replayed.close()

    with sqlite3.connect(tmp_path / "authority.sqlite3") as connection:
        assert connection.execute(
            "SELECT r.revision_id,r.prior_revision_id FROM source_revisions r "
            "JOIN ledger_events e ON e.event_id=r.authority_event_id "
            "ORDER BY e.ledger_seq"
        ).fetchall() == [
            (first.revision_id, None),
            (second.revision_id, first.revision_id),
        ]
        assert connection.execute(
            "SELECT COUNT(*) FROM object_admissions"
        ).fetchone()[0] == 2


def test_sealed_campaign_reopens_authority_and_runtime_after_process_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proving = tmp_path / "proving.sqlite3"
    unpublished = tmp_path / "unpublished.sqlite3"
    sqlite3.connect(proving).close()
    sqlite3.connect(unpublished).close()
    unit = _unit()
    plan = _plan(unit)
    adapter = MemoryNeo4jAdapter()
    system = _open_operational_test_system(tmp_path, adapter)
    try:
        bootstrap, _binder = bootstrap_operational_authority(
            system, proof=_PROOF, plan=plan
        )
        reconciliation = build_and_reconcile_operational_generation(
            system,
            proof=_PROOF,
            plan=plan,
            bootstrap=bootstrap,
        )
        graph_readback = graphiti_graph_destination_readback(
            destination_id=system.graph_destination_id,
            reconciliation=reconciliation,
        )
    finally:
        system.close()

    authority = tmp_path / "authority.sqlite3"
    snapshot_digests = graphiti_store_snapshot_digests(
        proving_store=proving,
        unpublished_store=unpublished,
        authority_store=authority,
    )
    campaign = {
        "source_snapshot_digests": snapshot_digests,
        "graph_destination_readback": graph_readback,
    }
    packet = {
        "store_snapshots": {
            name: {
                "source_path": str(path.resolve()),
                "descriptor_digest": snapshot_digests[name],
            }
            for name, path in {
                "proving": proving,
                "unpublished": unpublished,
                "authority": authority,
            }.items()
        },
        "operational_reconciliation": {
            "schema_version": "newsroom.graphiti-operational-reconciliation.v1",
            "status": "COMPLETE",
            "completed_steps": [
                "BACKUP",
                "CANONICAL_AUTHORITY_OPEN",
                "CURRENT_COHORT_PLAN",
                "SOURCE_AND_OBJECT_BOOTSTRAP",
                "ACTIVE_GENERATION_RECONCILIATION",
                "STORE_IDENTITY_SNAPSHOT",
                "DORMANT_RUNTIME_COMPOSITION",
                "AUTHENTICATED_GRAPH_READBACK",
                "DORMANT_CAMPAIGN_INPUT",
            ],
            "bootstrap": bootstrap.canonical_value(),
            "store_snapshot_digests": snapshot_digests,
            "graph_readback": graph_readback,
            "campaign_authorised": False,
            "provider_calls": 0,
            "graphiti_dispatches": 0,
            "service_loads": 0,
            "publication_effects": 0,
            "production_admission_effects": 0,
        },
    }
    module = "newsroom.control_plane.graphiti_operational_readiness"
    monkeypatch.setattr(f"{module}.CANONICAL_PROVING_STORE", proving)
    monkeypatch.setattr(f"{module}.CANONICAL_UNPUBLISHED_STORE", unpublished)
    monkeypatch.setattr(f"{module}.CANONICAL_INCREMENT4_AUTHORITY_STORE", authority)
    monkeypatch.setattr(
        f"{module}.validate_graphiti_campaign_packet",
        lambda supplied: (
            campaign if supplied is packet else pytest.fail("wrong packet")
        ),
    )

    def replay_plan(
        _proving: sqlite3.Connection,
        _unpublished: sqlite3.Connection,
        _authority: sqlite3.Connection,
        *,
        observed_at: datetime,
    ) -> OperationalAuthorityBootstrapPlan:
        assert _utc_text(observed_at) == _NOW
        return plan

    def reopen(*, credential: str):
        assert credential == "restart-credential"
        return (
            _open_operational_test_system(
                tmp_path,
                adapter,
                now=_REOPENED_NOW,
            ),
            _PROOF,
        )

    monkeypatch.setattr(f"{module}.plan_operational_authority_bootstrap", replay_plan)
    monkeypatch.setattr(f"{module}.open_operational_graphiti_authority_system", reopen)

    with reopen_operational_campaign_runtime(
        packet=packet,
        credential="restart-credential",
    ) as runtime:
        rebound = runtime.bind_unit_authority(unit)
        assert rebound.authority == bootstrap.bound_units[0].authority
        assert (
            runtime.authority_store_descriptor_digest == snapshot_digests["authority"]
        )
        reopened_readback = runtime.graph_state_fence(
            {"graph_destination_readback": graph_readback}
        )
        assert reopened_readback["serving_time"] == _REOPENED_NOW
        assert graph_readback["serving_time"] == _NOW

        drifted_readback = {
            **graph_readback,
            "checkpoint_ledger_seq": graph_readback["checkpoint_ledger_seq"] + 1,
        }
        with pytest.raises(
            GraphitiCampaignStop,
            match="campaign graph identity drifted",
        ):
            runtime.graph_state_fence(
                {"graph_destination_readback": drifted_readback}
            )

    assert adapter.closed is True

    object_files = [
        path for path in (tmp_path / "objects").rglob("*") if path.is_file()
    ]
    assert len(object_files) == 1
    object_files[0].unlink()
    with pytest.raises(
        AuthorityPersistenceError,
        match="active authoritative blob bytes are missing or corrupt",
    ):
        with reopen_operational_campaign_runtime(
            packet=packet,
            credential="restart-credential",
        ):
            pytest.fail("runtime composed without its authoritative CAS blob")


def _utc_text(value: datetime) -> str:
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def test_operational_generation_identity_resumes_across_observation_times(
    tmp_path: Path,
) -> None:
    unit = _unit()
    plan = _plan(unit)
    adapter = MemoryNeo4jAdapter()
    system = _open_operational_test_system(tmp_path, adapter)
    try:
        bootstrap, _binder = bootstrap_operational_authority(
            system, proof=_PROOF, plan=plan
        )
        first = build_and_reconcile_operational_generation(
            system,
            proof=_PROOF,
            plan=plan,
            bootstrap=bootstrap,
        )

        later_plan = replace(
            plan,
            observed_at=UtcTimestamp.parse(_REOPENED_NOW),
            rights_by_source=(
                (
                    unit.source_id,
                    {**_rights(), "evaluated_at": _REOPENED_NOW},
                ),
            ),
            cohort_digest=digest_canonical({"cohort": "same-state-later"}),
            plan_digest=digest_canonical({"plan": "same-state-later"}),
        )
        later_bootstrap = replace(
            bootstrap,
            observed_at=later_plan.observed_at,
            cohort_digest=later_plan.cohort_digest,
            plan_digest=later_plan.plan_digest,
            bound_units=(
                replace(bootstrap.bound_units[0], observed_at=_REOPENED_NOW),
            ),
        )
        resumed = build_and_reconcile_operational_generation(
            system,
            proof=_PROOF,
            plan=later_plan,
            bootstrap=later_bootstrap,
        )

        configuration = _evaluation_attempt_for_unit(
            bootstrap.bound_units[0]
        ).configuration.canonical_value()
        stable_id, stable_semantic_digest = _operational_generation_identity(
            graph_destination_id=system.graph_destination_id,
            plan_semantic_digest=bootstrap.plan_semantic_digest,
            bootstrap=bootstrap,
            configuration=configuration,
        )
        assert resumed.generation_id == first.generation_id == stable_id
        assert _operational_generation_identity(
            graph_destination_id=system.graph_destination_id,
            plan_semantic_digest=later_bootstrap.plan_semantic_digest,
            bootstrap=later_bootstrap,
            configuration=configuration,
        ) == (stable_id, stable_semantic_digest)

        with pytest.raises(
            GraphitiOperationalReadinessError,
            match="build plan differs from its exact bound cohort",
        ):
            build_and_reconcile_operational_generation(
                system,
                proof=_PROOF,
                plan=replace(
                    later_plan,
                    candidate_events=(
                        {
                            **later_plan.candidate_events[0],
                            "ingest_ids": [
                                digest_canonical({"ingest": "other"})
                            ],
                        },
                    ),
                ),
                bootstrap=later_bootstrap,
            )

        changed_event_plan = replace(
            later_plan,
            candidate_events=(
                {
                    **later_plan.candidate_events[0],
                    "event_id": digest_canonical({"event": "changed"}),
                },
            ),
        )
        with pytest.raises(
            GraphitiOperationalReadinessError,
            match="build plan semantics differ from its bootstrap",
        ):
            build_and_reconcile_operational_generation(
                system,
                proof=_PROOF,
                plan=changed_event_plan,
                bootstrap=later_bootstrap,
            )
        assert _operational_generation_identity(
            graph_destination_id=system.graph_destination_id,
            plan_semantic_digest=_operational_plan_semantic_digest(
                candidate_events=changed_event_plan.candidate_events,
                rights_by_source=changed_event_plan.rights_by_source,
                revision_predecessors=changed_event_plan.revision_predecessors,
            ),
            bootstrap=bootstrap,
            configuration=configuration,
        )[0] != stable_id
        assert _operational_generation_identity(
            graph_destination_id=system.graph_destination_id,
            plan_semantic_digest=bootstrap.plan_semantic_digest,
            bootstrap=replace(
                bootstrap,
                bound_units=(_next_revision(bootstrap.bound_units[0]),),
            ),
            configuration=configuration,
        )[0] != stable_id
        assert _operational_generation_identity(
            graph_destination_id=system.graph_destination_id,
            plan_semantic_digest=bootstrap.plan_semantic_digest,
            bootstrap=bootstrap,
            configuration={**configuration, "runtime_mode": "APPROVED_REPLAY"},
        )[0] != stable_id
        changed_rights_plan = replace(
            later_plan,
            rights_by_source=(
                (
                    unit.source_id,
                    {
                        **_rights(),
                        "packet_digest": digest_canonical({"rights": "changed"}),
                    },
                ),
            ),
        )
        with pytest.raises(
            GraphitiOperationalReadinessError,
            match="build plan semantics differ from its bootstrap",
        ):
            build_and_reconcile_operational_generation(
                system,
                proof=_PROOF,
                plan=changed_rights_plan,
                bootstrap=later_bootstrap,
            )
        assert _operational_generation_identity(
            graph_destination_id=system.graph_destination_id,
            plan_semantic_digest=_operational_plan_semantic_digest(
                candidate_events=changed_rights_plan.candidate_events,
                rights_by_source=changed_rights_plan.rights_by_source,
                revision_predecessors=changed_rights_plan.revision_predecessors,
            ),
            bootstrap=bootstrap,
            configuration=configuration,
        )[0] != stable_id
    finally:
        system.close()
