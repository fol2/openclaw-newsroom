"""Provider-free contract tests for the existing Graphiti pipeline adapter."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest

from newsroom.authority.canonical import CanonicalizationError, canonical_json_bytes
from newsroom.graphiti_adapter import real
from newsroom.graphiti_adapter.combined_temporal_extraction import (
    CombinedTemporalOutcome,
    CombinedTemporalTransportResult,
    extract_combined_temporal,
)
from newsroom.graphiti_adapter.combined_temporal_fixtures import fixture
from newsroom.graphiti_adapter.combined_temporal_pipeline import (
    CombinedTemporalPipelineError,
    ExistingGraphitiPipeline,
)
from newsroom.graphiti_adapter.evaluation_attempt import evaluation_attempt_for
from newsroom.graphiti_adapter.evaluation_packet import (
    GRAPHITI_CORE_RELEASE,
    GRAPHITI_WORKSPACE_GROUP,
)
from newsroom.graphiti_adapter.neo4j_guard import GuardError, GuardState


class _Guard:
    def __init__(self, state: object = GuardState.CREATED) -> None:
        self.calls: list[str] = []
        self.state = state
        self.completed_receipt: dict[str, object] | None = None

    async def completed_raw(self) -> dict[str, object]:
        self.calls.append("completed")
        assert self.completed_receipt is not None
        return self.completed_receipt

    async def begin(self) -> Any:
        self.calls.append("begin")
        return SimpleNamespace(
            state=self.state,
            chat_invocations=({"model": "retained"},),
            embedding_usage={"usage_basis": "RETAINED"},
        )

    async def record_pending_telemetry(self, **_kwargs: Any) -> None:
        self.calls.append("telemetry")

    @asynccontextmanager
    async def fenced_graph_mutation(self) -> AsyncIterator[None]:
        self.calls.append("fence")
        try:
            yield
        finally:
            self.calls.append("unfence")

    async def restore_preexisting(self) -> None:
        self.calls.append("restore")

    async def complete(self, receipt: dict[str, object]) -> None:
        assert receipt["provider_attempt_number"] == 1
        self.completed_receipt = dict(receipt)
        self.calls.append("complete")

    async def rollback_pending(self, **_kwargs: Any) -> bool:
        self.calls.append("rollback")
        return True


class _StatefulGuard(_Guard):
    """Model the native guard's terminal transition after rollback."""

    def __init__(self) -> None:
        super().__init__()
        self.claim_state = GuardState.CREATED
        self.pending_telemetry: dict[str, object] | None = None

    async def begin(self) -> Any:
        marker = await super().begin()
        self.claim_state = GuardState.PENDING
        return marker

    def _require_pending(self, operation: str) -> None:
        if self.claim_state is not GuardState.PENDING:
            raise GuardError(f"Graphiti {operation} lost its pending claim")

    async def record_pending_telemetry(self, **_kwargs: Any) -> None:
        self._require_pending("telemetry")
        self.pending_telemetry = dict(_kwargs)
        await super().record_pending_telemetry(**_kwargs)

    async def complete(self, receipt: dict[str, object]) -> None:
        self._require_pending("completion")
        await super().complete(receipt)
        self.claim_state = GuardState.COMPLETE

    async def rollback_pending(self, **_kwargs: Any) -> bool:
        self._require_pending("rollback")
        self.calls.append("rollback")
        self.claim_state = GuardState.RECOVERED_AMBIGUOUS
        return True


def _node(uuid: str) -> Any:
    return SimpleNamespace(uuid=uuid, attributes={})


def _edge() -> Any:
    return SimpleNamespace(
        source_node_uuid="local-source",
        target_node_uuid="local-target",
        name="ASKED_ABOUT",
        fact="asked about",
    )


def _pipeline(
    guard: _Guard,
    *,
    fail_embedding: bool = False,
) -> ExistingGraphitiPipeline:
    async def resolve_nodes(
        _nodes: list[Any],
    ) -> tuple[list[Any], dict[str, str], list[tuple[Any, Any]]]:
        guard.calls.append("resolve")
        original_ids = [str(node.uuid) for node in _nodes]
        resolved = [SimpleNamespace(**vars(node)) for node in _nodes]
        resolved[0].uuid = "existing-source"
        resolved[1].uuid = "existing-target"
        return (
            resolved,
            {
                original_ids[0]: "existing-source",
                original_ids[1]: "existing-target",
            },
            [],
        )

    def resolve_pointers(edges: list[Any], uuid_map: dict[str, str]) -> list[Any]:
        guard.calls.append("pointers")
        for edge in edges:
            edge.source_node_uuid = uuid_map[edge.source_node_uuid]
            edge.target_node_uuid = uuid_map[edge.target_node_uuid]
        return edges

    async def create_embeddings(_embedder: Any, edges: list[Any]) -> None:
        guard.calls.append("embed")
        if fail_embedding:
            raise RuntimeError("embedding failed")
        for edge in edges:
            edge.fact_embedding = [0.25]

    async def persist_graph(_nodes: list[Any], _edges: list[Any]) -> None:
        guard.calls.append("persist")

    return ExistingGraphitiPipeline(
        guard=guard,  # type: ignore[arg-type]
        resolve_nodes=resolve_nodes,
        resolve_pointers=resolve_pointers,
        create_embeddings=create_embeddings,
        persist_graph=persist_graph,
        embedder=object(),
        run_async=asyncio.run,
        chat_receipt=lambda: [{"model": "composer-2.5"}],
        embedding_receipt=lambda: {"usage_basis": "PROVIDER_REPORTED"},
    )


class _Transport:
    def __init__(self, raw: object) -> None:
        self.raw = raw
        self.calls = 0

    def generate_response(self, **_kwargs: object) -> CombinedTemporalTransportResult:
        self.calls += 1
        return CombinedTemporalTransportResult(
            raw=self.raw,
            framework_version=GRAPHITI_CORE_RELEASE,
            model_version=None,
            token_usage={"basis": "UNMEASURED"},
            provider_cost=None,
        )


def test_existing_pipeline_resolves_embeds_and_completes_durable_journal() -> None:
    guard = _Guard()
    edge = _edge()
    result = _pipeline(guard).execute(
        nodes=(_node("local-source"), _node("local-target")),
        edges=(edge,),
        receipt={"provider_attempt_number": 1},
    )

    assert guard.calls == [
        "begin",
        "resolve",
        "pointers",
        "embed",
        "fence",
        "persist",
        "restore",
        "unfence",
        "telemetry",
        "complete",
    ]
    assert result.node_resolutions == (
        "RESOLVED_EXISTING",
        "RESOLVED_EXISTING",
    )
    assert edge.source_node_uuid == "existing-source"
    assert edge.target_node_uuid == "existing-target"
    assert edge.fact_embedding == [0.25]
    assert result.graph_effect_attempted is True
    assert result.rollback_skipped is True


def test_stale_owner_is_fenced_before_graph_persistence() -> None:
    guard = _Guard()

    @asynccontextmanager
    async def lost_claim() -> AsyncIterator[None]:
        guard.calls.append("fence")
        raise RuntimeError("claim was replaced")
        yield

    guard.fenced_graph_mutation = lost_claim  # type: ignore[method-assign]

    with pytest.raises(CombinedTemporalPipelineError):
        _pipeline(guard).execute(
            nodes=(_node("local-source"), _node("local-target")),
            edges=(_edge(),),
            receipt={"provider_attempt_number": 1},
        )

    assert "persist" not in guard.calls
    assert guard.calls[-1] == "rollback"


def test_completed_ingest_replays_without_another_provider_leaf() -> None:
    case = fixture("pair-current")
    first_guard = _Guard()
    first_transport = _Transport(case.gold)
    first = extract_combined_temporal(
        case.revision,
        transport=first_transport,
        pipeline=_pipeline(first_guard),
    )

    assert first.outcome is CombinedTemporalOutcome.TERMINAL_SUCCESS_WITH_PROPOSALS
    assert first_transport.calls == 1
    assert first_guard.completed_receipt is not None

    replay_guard = _Guard(GuardState.COMPLETE)
    replay_guard.completed_receipt = first_guard.completed_receipt
    replay_transport = _Transport(RuntimeError("provider must not run"))
    replayed = extract_combined_temporal(
        case.revision,
        transport=replay_transport,
        pipeline=_pipeline(replay_guard),
    )

    assert replay_transport.calls == 0
    assert replayed.outcome is CombinedTemporalOutcome.TERMINAL_SUCCESS_WITH_PROPOSALS
    assert replayed.payload == first.payload
    assert replayed.payload_digest == first.payload_digest
    assert [node.uuid for node in replayed.nodes] == [node.uuid for node in first.nodes]
    assert [edge.uuid for edge in replayed.edges] == [edge.uuid for edge in first.edges]
    assert replay_guard.calls == ["begin", "completed"]


def test_zero_result_is_completed_and_replayed_without_another_leaf() -> None:
    case = fixture("zero-result")
    first_guard = _Guard()
    first_transport = _Transport(case.gold)
    first = extract_combined_temporal(
        case.revision,
        transport=first_transport,
        pipeline=_pipeline(first_guard),
    )

    assert first.outcome is CombinedTemporalOutcome.TERMINAL_SUCCESS_ZERO_PROPOSALS
    assert first_transport.calls == 1
    assert first_guard.completed_receipt is not None
    assert "resolve" not in first_guard.calls
    assert "persist" not in first_guard.calls

    replay_guard = _Guard(GuardState.COMPLETE)
    replay_guard.completed_receipt = first_guard.completed_receipt
    replay_transport = _Transport(RuntimeError("provider must not run"))
    replayed = extract_combined_temporal(
        case.revision,
        transport=replay_transport,
        pipeline=_pipeline(replay_guard),
    )

    assert replayed.outcome is CombinedTemporalOutcome.TERMINAL_SUCCESS_ZERO_PROPOSALS
    assert replay_transport.calls == 0
    assert replay_guard.calls == ["begin", "completed"]


def test_pending_marker_is_recovered_before_another_provider_leaf() -> None:
    case = fixture("pair-current")
    guard = _Guard(GuardState.PENDING)
    transport = _Transport(case.gold)

    with pytest.raises(CombinedTemporalPipelineError) as captured:
        extract_combined_temporal(
            case.revision,
            transport=transport,
            pipeline=_pipeline(guard),
        )

    assert transport.calls == 0
    assert captured.value.rollback_completed is True
    assert guard.calls == ["begin", "rollback"]


def test_journal_starts_before_the_provider_leaf() -> None:
    case = fixture("pair-current")
    guard = _Guard()

    class Transport(_Transport):
        def generate_response(self, **kwargs: object) -> CombinedTemporalTransportResult:
            assert guard.calls == ["begin"]
            return super().generate_response(**kwargs)

    extract_combined_temporal(
        case.revision,
        transport=Transport(case.gold),
        pipeline=_pipeline(guard),
    )


def test_malformed_leaf_is_completed_and_replayed_without_provider_retry() -> None:
    case = fixture("pair-current")
    first_guard = _Guard()
    first_transport = _Transport({"entities": [], "facts": [{"bad": True}]})
    first = extract_combined_temporal(
        case.revision,
        transport=first_transport,
        pipeline=_pipeline(first_guard),
    )

    assert first.outcome is CombinedTemporalOutcome.TERMINAL_ATTEMPT_FAILURE
    assert first.journal_skipped is False
    assert first_guard.completed_receipt is not None

    replay_guard = _Guard(GuardState.COMPLETE)
    replay_guard.completed_receipt = first_guard.completed_receipt
    replay_transport = _Transport(RuntimeError("provider must not run"))
    replayed = extract_combined_temporal(
        case.revision,
        transport=replay_transport,
        pipeline=_pipeline(replay_guard),
    )

    assert replay_transport.calls == 0
    assert replayed.outcome is CombinedTemporalOutcome.TERMINAL_ATTEMPT_FAILURE
    assert replayed.failure_code is first.failure_code


@pytest.mark.parametrize(
    "retained",
    ("SNAPSHOTTING", GuardState.RECOVERED_AMBIGUOUS),
)
def test_preflight_uses_guard_begin_recovery(retained: object) -> None:
    case = fixture("pair-current")
    guard = _Guard(retained)

    async def recovered_begin() -> Any:
        guard.calls.append("begin")
        return SimpleNamespace(state=GuardState.CREATED)

    guard.begin = recovered_begin  # type: ignore[method-assign]
    transport = _Transport(case.gold)
    leaf = extract_combined_temporal(
        case.revision,
        transport=transport,
        pipeline=_pipeline(guard),
    )

    assert leaf.outcome is CombinedTemporalOutcome.TERMINAL_SUCCESS_WITH_PROPOSALS
    assert transport.calls == 1
    assert guard.calls[0] == "begin"


def test_completion_retains_canonical_proposals_and_passage_provenance() -> None:
    case = fixture("pair-current")
    guard = _Guard()
    leaf = extract_combined_temporal(
        case.revision,
        transport=_Transport(case.gold),
        pipeline=_pipeline(guard),
    )

    assert leaf.outcome is CombinedTemporalOutcome.TERMINAL_SUCCESS_WITH_PROPOSALS
    assert guard.completed_receipt is not None
    assert guard.completed_receipt["invocation_count"] == 2
    assert guard.completed_receipt["pipeline_chat_invocations"] == [
        {"model": "composer-2.5"}
    ]
    assert guard.completed_receipt["embedding_usage"] == {
        "usage_basis": "PROVIDER_REPORTED"
    }
    assert len(guard.completed_receipt["transport_calls"]) == 1
    proposal = guard.completed_receipt["proposal_receipt"]
    assert isinstance(proposal, dict)
    assert proposal["contract"] == "NewsroomCombinedTemporalExtractionV1"
    assert proposal["source_revision_id"] == case.revision.revision_id
    assert proposal["reference_time"] == case.revision.reference_time
    assert proposal["entity_mentions"]
    assert proposal["relation_proposals"]
    assert proposal["evidence_passages"]
    assert proposal["evidence_passages"][0]["segments"][0]["text"] in case.revision.body
    relation = proposal["relation_proposals"][0]
    assert relation["source_identity"] == "existing-source"
    assert relation["target_identity"] == "existing-target"
    assert relation["fact"] == case.gold["facts"][0]["fact"]
    assert "valid_at" in relation
    assert "invalid_at" in relation
    assert leaf.invocation_count == 2
    assert leaf.pipeline_chat_invocations == ({"model": "composer-2.5"},)
    assert leaf.embedding_usage == {"usage_basis": "PROVIDER_REPORTED"}


def test_replay_revalidates_exact_evidence_passages() -> None:
    case = fixture("pair-current")
    first_guard = _Guard()
    extract_combined_temporal(
        case.revision,
        transport=_Transport(case.gold),
        pipeline=_pipeline(first_guard),
    )
    assert first_guard.completed_receipt is not None
    proposal = first_guard.completed_receipt["proposal_receipt"]
    assert isinstance(proposal, dict)
    proposal["evidence_passages"] = []

    replay_guard = _Guard(GuardState.COMPLETE)
    replay_guard.completed_receipt = first_guard.completed_receipt
    with pytest.raises(CombinedTemporalPipelineError, match="evidence"):
        extract_combined_temporal(
            case.revision,
            transport=_Transport(RuntimeError("provider must not run")),
            pipeline=_pipeline(replay_guard),
        )


def test_existing_pipeline_rolls_back_embedding_failure() -> None:
    guard = _Guard()
    with pytest.raises(CombinedTemporalPipelineError) as captured:
        _pipeline(guard, fail_embedding=True).execute(
            nodes=(_node("local-source"), _node("local-target")),
            edges=(_edge(),),
            receipt={"provider_attempt_number": 1},
        )

    assert captured.value.graph_effect_attempted is True
    assert captured.value.rollback_completed is True
    assert guard.calls == ["begin", "resolve", "pointers", "embed", "rollback"]


def test_held_resolution_without_persistable_graph_seals_explicit_empty_effect() -> None:
    """COMPLETE embeddings + nothing to persist is explicit zero, not unmarked."""

    guard = _Guard()
    persist_calls: list[object] = []
    sealed: list[tuple[list[object], list[object]]] = []

    async def resolve_nodes(
        nodes: list[Any],
    ) -> tuple[list[Any], dict[str, str], list[tuple[Any, Any]]]:
        guard.calls.append("resolve")
        held = [
            SimpleNamespace(
                uuid=str(node.uuid),
                attributes={"resolution": "AMBIGUOUS_HOLD"},
            )
            for node in nodes
        ]
        return (
            held,
            {str(node.uuid): str(node.uuid) for node in nodes},
            [],
        )

    def resolve_pointers(edges: list[Any], _uuid_map: dict[str, str]) -> list[Any]:
        guard.calls.append("pointers")
        return edges

    async def create_embeddings(_embedder: Any, _edges: list[Any]) -> None:
        guard.calls.append("embed")
        raise RuntimeError("edge embeddings must not run for empty persistable effect")

    async def persist_graph(nodes: list[Any], edges: list[Any]) -> None:
        persist_calls.append((list(nodes), list(edges)))
        raise RuntimeError("persist must not run for empty persistable effect")

    pipeline = ExistingGraphitiPipeline(
        guard=guard,  # type: ignore[arg-type]
        resolve_nodes=resolve_nodes,
        resolve_pointers=resolve_pointers,
        create_embeddings=create_embeddings,
        persist_graph=persist_graph,
        embedder=object(),
        run_async=asyncio.run,
        chat_receipt=lambda: [
            {"model": "composer-2.5", "outcome": "COMPLETE"}
        ],
        embedding_receipt=lambda: {
            "usage_basis": "PROVIDER_REPORTED",
            "request_count": 5,
            "embedding_tokens": 43,
            "cost_usd_microunits": 4,
            "requests": [{"outcome": "COMPLETE"}] * 5,
        },
        complete_receipt=lambda nodes, edges, receipt: (
            sealed.append((list(nodes), list(edges))) or dict(receipt)
        ),
    )
    result = pipeline.execute(
        nodes=(_node("local-entity"),),
        edges=(),
        receipt={
            "provider_attempt_number": 1,
            "proposal_receipt": {
                "entity_mentions": [{"local_id": 0}],
                "relation_proposals": [{"local_id": 0}],
            },
        },
    )

    assert persist_calls == []
    assert sealed == [([], [])]
    assert result.graph_effect_attempted is False
    assert result.nodes == ()
    assert result.edges == ()
    assert result.embedding_skipped is True
    assert guard.calls == ["begin", "resolve", "telemetry", "complete"]
    assert result.completed_receipt is not None
    assert result.completed_receipt["zero_proposal_effect"] == "EXPLICIT"
    assert result.completed_receipt["embedding_usage"]["request_count"] == 5
    assert result.completed_receipt["proposal_receipt"]["entity_mentions"] == [
        {"local_id": 0}
    ]


def test_new_nodes_without_persistable_edges_seal_explicit_empty_effect() -> None:
    """Live 13677: leftover NEW nodes and no persistable relation is explicit zero."""

    guard = _Guard()
    persist_calls: list[object] = []
    sealed: list[tuple[list[object], list[object]]] = []

    async def resolve_nodes(
        nodes: list[Any],
    ) -> tuple[list[Any], dict[str, str], list[tuple[Any, Any]]]:
        guard.calls.append("resolve")
        created = [
            SimpleNamespace(
                uuid=str(node.uuid),
                attributes={"resolution": "DETERMINISTIC_NEW_NODE"},
            )
            for node in nodes
        ]
        return (
            created,
            {str(node.uuid): str(node.uuid) for node in nodes},
            [],
        )

    def resolve_pointers(edges: list[Any], _uuid_map: dict[str, str]) -> list[Any]:
        guard.calls.append("pointers")
        return edges

    async def create_embeddings(_embedder: Any, _edges: list[Any]) -> None:
        guard.calls.append("embed")
        raise RuntimeError("edge embeddings must not run without persistable edges")

    async def persist_graph(nodes: list[Any], edges: list[Any]) -> None:
        persist_calls.append((list(nodes), list(edges)))
        raise RuntimeError("persist must not run without persistable edges")

    pipeline = ExistingGraphitiPipeline(
        guard=guard,  # type: ignore[arg-type]
        resolve_nodes=resolve_nodes,
        resolve_pointers=resolve_pointers,
        create_embeddings=create_embeddings,
        persist_graph=persist_graph,
        embedder=object(),
        run_async=asyncio.run,
        chat_receipt=lambda: [
            {"model": "composer-2.5", "outcome": "COMPLETE"}
        ],
        embedding_receipt=lambda: {
            "usage_basis": "PROVIDER_REPORTED",
            "request_count": 4,
            "embedding_tokens": 44,
            "cost_usd_microunits": 6,
            "requests": [{"outcome": "COMPLETE"}] * 4,
        },
        complete_receipt=lambda nodes, edges, receipt: (
            sealed.append((list(nodes), list(edges))) or dict(receipt)
        ),
    )
    result = pipeline.execute(
        nodes=(_node("local-entity"),),
        edges=(),
        receipt={
            "provider_attempt_number": 1,
            "proposal_receipt": {
                "entity_mentions": [{"local_id": 0}],
                "relation_proposals": [],
            },
        },
    )

    assert persist_calls == []
    assert sealed == [([], [])]
    assert result.graph_effect_attempted is False
    assert result.nodes == ()
    assert result.edges == ()
    assert result.embedding_skipped is True
    assert guard.calls == ["begin", "resolve", "telemetry", "complete"]
    assert result.completed_receipt is not None
    assert result.completed_receipt["zero_proposal_effect"] == "EXPLICIT"
    assert result.completed_receipt["embedding_usage"]["request_count"] == 4
    assert result.completed_receipt["proposal_receipt"]["entity_mentions"] == [
        {"local_id": 0}
    ]


def test_persistable_float_fact_embedding_remains_nonzero_and_canonical() -> None:
    """A derivative float embedding must not erase a persistable relation."""

    guard = _StatefulGuard()
    persist_calls: list[object] = []
    sealed: list[tuple[list[object], list[object]]] = []
    embedding_usage = {"request_count": 1}
    pipeline = _pipeline(guard)

    async def persist_graph(nodes: list[Any], edges: list[Any]) -> None:
        persist_calls.append((list(nodes), list(edges)))
        embedding_usage["request_count"] = 2
        guard.calls.append("persist")

    def complete_receipt(
        nodes: list[Any], edges: list[Any], receipt: object
    ) -> dict[str, object]:
        payload = dict(receipt)  # type: ignore[arg-type]
        sealed.append((list(nodes), list(edges)))
        canonical_json_bytes(payload)
        return payload

    pipeline.persist_graph = persist_graph
    pipeline.embedding_receipt = lambda: dict(embedding_usage)
    pipeline.complete_receipt = complete_receipt
    edge = _edge()
    edge.uuid = "edge-1"
    result = pipeline.execute(
        nodes=(_node("local-source"), _node("local-target")),
        edges=(edge,),
        receipt={
            "provider_attempt_number": 1,
            "proposal_receipt": {
                "entity_mentions": [{"local_id": 0}, {"local_id": 1}],
                "relation_proposals": [{"local_id": 0}],
            },
        },
    )

    assert len(persist_calls) == 1
    assert "rollback" not in guard.calls
    assert guard.claim_state is GuardState.COMPLETE
    assert len(sealed[-1][0]) == 2
    assert len(sealed[-1][1]) == 1
    assert result.graph_effect_attempted is True
    assert len(result.nodes) == 2
    assert len(result.edges) == 1
    assert result.completed_receipt is not None
    relation = result.completed_receipt["proposal_receipt"]["relation_proposals"][0]
    assert relation["proposal_status"] == "PROPOSED"
    assert "fact_embedding" not in relation
    assert result.completed_receipt["embedding_usage"]["request_count"] == 2
    assert guard.pending_telemetry is not None
    assert guard.pending_telemetry["embedding_usage"]["request_count"] == 2
    canonical_json_bytes(dict(result.completed_receipt))


def test_non_fact_embedding_canonicalization_error_stays_fail_closed() -> None:
    """Other CanonicalizationError must not be sealed as explicit zero."""

    guard = _Guard()
    pipeline = _pipeline(guard)
    edge = _edge()
    edge.uuid = "edge-1"

    def complete_receipt(
        _nodes: list[Any], _edges: list[Any], _receipt: object
    ) -> dict[str, object]:
        raise CanonicalizationError("unsupported value type at $.other")

    pipeline.complete_receipt = complete_receipt
    with pytest.raises(CombinedTemporalPipelineError):
        pipeline.execute(
            nodes=(_node("local-source"), _node("local-target")),
            edges=(edge,),
            receipt={
                "provider_attempt_number": 1,
                "proposal_receipt": {
                    "entity_mentions": [{"local_id": 0}, {"local_id": 1}],
                    "relation_proposals": [{"local_id": 0}],
                },
            },
        )

    assert "rollback" in guard.calls
    assert "complete" not in guard.calls


def test_existing_pipeline_rejects_a_malformed_complete_marker_without_effect() -> None:
    guard = _Guard(GuardState.COMPLETE)
    with pytest.raises(CombinedTemporalPipelineError) as captured:
        _pipeline(guard).execute(
            nodes=(_node("local-source"), _node("local-target")),
            edges=(_edge(),),
            receipt={},
        )

    assert captured.value.graph_effect_attempted is False
    assert captured.value.rollback_completed is False
    assert guard.calls == ["begin", "completed"]


@pytest.mark.parametrize("malformed_batch", (False, True))
def test_real_factory_uses_graphiti_types_and_bulk_persistence(
    monkeypatch: pytest.MonkeyPatch,
    malformed_batch: bool,
) -> None:
    guard = _Guard()
    calls: list[str] = []
    driver = object()
    guard.driver = driver
    guard.group_id = GRAPHITI_WORKSPACE_GROUP
    guard.episode_uuid = "episode"
    guard.input_digest = "ingest"

    class RuntimeObject:
        def __init__(self, **values: Any) -> None:
            self.name_embedding = None
            vars(self).update(values)

        @classmethod
        async def get_by_group_ids(
            cls, *_args: Any, **_kwargs: Any
        ) -> list[Any]:
            calls.append("retrieve")
            return []

    def resolve_pointers(edges: list[Any], _uuid_map: dict[str, str]) -> list[Any]:
        calls.append("pointers")
        return edges

    async def create_embeddings(_embedder: Any, edges: list[Any]) -> None:
        calls.append("embed")
        for edge in edges:
            edge.fact_embedding = [0.5]

    runtime = SimpleNamespace(
        EntityNode=RuntimeObject,
        EntityEdge=RuntimeObject,
        resolve_edge_pointers=resolve_pointers,
        create_entity_edge_embeddings=create_embeddings,
    )
    monkeypatch.setattr(real, "_load_graphiti", lambda: runtime)

    class Embedder:
        async def create_batch(self, names: list[str]) -> list[list[float]]:
            calls.append("node-embed")
            embeddings = [[float(index)] for index, _name in enumerate(names)]
            return embeddings[:-1] if malformed_batch else embeddings

        def receipt(self) -> dict[str, object]:
            return {"usage_basis": "TEST"}

    class Graphiti:
        clients = SimpleNamespace(
            llm_client=SimpleNamespace(invocations=[]),
            embedder=Embedder(),
        )

        async def _process_episode_data(self, *args: Any) -> None:
            assert isinstance(args[1][0], RuntimeObject)
            assert isinstance(args[2][0], RuntimeObject)
            calls.append("persist")

    Graphiti.driver = driver

    now = datetime.now(tz=UTC)
    proposal_nodes = tuple(
        SimpleNamespace(
            uuid=value,
            name=value,
            group_id=GRAPHITI_WORKSPACE_GROUP,
            labels=["Entity"],
            created_at=now,
            summary="",
            attributes={"entity_type_id": 0},
        )
        for value in ("source", "target")
    )
    proposal_edge = SimpleNamespace(
        uuid="edge",
        group_id=GRAPHITI_WORKSPACE_GROUP,
        source_node_uuid="source",
        target_node_uuid="target",
        created_at=now,
        name="ASKED_ABOUT",
        fact="asked about",
        fact_embedding=None,
        episodes=["episode"],
        expired_at=None,
        valid_at=None,
        invalid_at=None,
        reference_time=now,
        attributes={},
    )
    pipeline = real.combined_temporal_pipeline_for(
        configuration=evaluation_attempt_for(("Alice asked Bob.",)).configuration,
        graphiti=Graphiti(),
        guard=guard,  # type: ignore[arg-type]
        episode=SimpleNamespace(group_id=GRAPHITI_WORKSPACE_GROUP, uuid="episode"),
    )
    receipt = {
        "ingest_id": "ingest",
        "proposal_receipt": {
            "episode_id": "episode",
            "entity_mentions": [{}, {}],
            "relation_proposals": [{}],
        },
    }
    if malformed_batch:
        with pytest.raises(CombinedTemporalPipelineError) as caught:
            pipeline.execute(
                nodes=proposal_nodes,
                edges=(proposal_edge,),
                receipt=receipt,
            )
        assert isinstance(
            caught.value.__cause__, real.GraphitiAdapterContractError
        )
        assert calls == ["retrieve", "pointers", "embed", "node-embed"]
        assert guard.calls[-1] == "rollback"
        return

    result = pipeline.execute(
        nodes=proposal_nodes,
        edges=(proposal_edge,),
        receipt=receipt,
    )

    assert calls == ["retrieve", "pointers", "embed", "node-embed", "persist"]
    assert isinstance(result.nodes[0], RuntimeObject)
    assert [node.name_embedding for node in result.nodes] == [[0.0], [1.0]]
    assert isinstance(result.edges[0], RuntimeObject)
    assert result.edges[0].fact_embedding == [0.5]

    proposal_nodes[0].group_id = "other-generation"
    with pytest.raises(CombinedTemporalPipelineError, match="identity differs"):
        pipeline.execute(nodes=proposal_nodes, edges=(proposal_edge,), receipt=receipt)


def test_real_factory_rejects_a_different_journal_driver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(real, "_load_graphiti", lambda: SimpleNamespace())
    guard = _Guard()
    guard.driver = object()
    guard.group_id = "test"
    guard.episode_uuid = "episode"
    guard.input_digest = "ingest"

    with pytest.raises(real.GraphitiAdapterContractError, match="identity differ"):
        real.combined_temporal_pipeline_for(
            configuration=evaluation_attempt_for(("Alice asked Bob.",)).configuration,
            graphiti=SimpleNamespace(driver=object()),
            guard=guard,  # type: ignore[arg-type]
            episode=SimpleNamespace(group_id="test", uuid="episode"),
        )


def test_real_factory_requires_the_existing_evaluation_authority_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(real, "_load_graphiti", lambda: pytest.fail("must not load"))

    with pytest.raises(real.GraphitiAdapterContractError, match="typed configuration"):
        real.combined_temporal_pipeline_for(
            configuration=SimpleNamespace(),  # type: ignore[arg-type]
            graphiti=SimpleNamespace(driver=object()),
            guard=SimpleNamespace(),  # type: ignore[arg-type]
            episode=SimpleNamespace(),
        )
