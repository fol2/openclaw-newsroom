"""Provider-free proof of graphiti-core 0.29.3's non-zero combined call shape.

The first #739 calibration returned no edges, so upstream combined extraction
made only one LLM request. A relation-bearing result makes a second
BatchEdgeTimestamps request. This fixture pins that conditional behaviour and
validates the owner-gated second-stage experiment manifest.
"""

from __future__ import annotations

import asyncio
import importlib.metadata
import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("graphiti_core")

from graphiti_core.llm_client.client import LLMClient
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.nodes import EpisodeType, EpisodicNode
from graphiti_core.prompts.extract_edges import BatchEdgeTimestamps
from graphiti_core.prompts.extract_nodes_and_edges import CombinedExtraction
from graphiti_core.utils.maintenance.combined_extraction import (
    extract_nodes_and_edges,
)

from newsroom.graphiti_adapter.cli_client import messages_to_prompt
from newsroom.graphiti_adapter.evaluation_packet import (
    GRAPHITI_EXTRACTION_INSTRUCTIONS,
)

pytestmark = pytest.mark.skipif(
    importlib.metadata.version("graphiti-core") != "0.29.3",
    reason="call-shape fixtures are pinned to graphiti-core 0.29.3",
)

_PLAN_PATH = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "research"
    / "2026-08-21-graphiti-token-effectiveness-experiment-plan.json"
)


class _NonZeroCombinedRecordingLlm(LLMClient):
    def __init__(self) -> None:
        super().__init__(
            LLMConfig(model="composer-2.5", small_model="composer-2.5"),
            cache=False,
        )
        self.calls: list[dict[str, object]] = []

    async def _generate_response(
        self,
        messages: list[Any],
        response_model: type[Any] | None = None,
        max_tokens: int = 0,
        model_size: object = None,
    ) -> dict[str, Any]:
        prompt = messages_to_prompt(messages)
        self.calls.append(
            {
                "response_model": (
                    None if response_model is None else response_model.__name__
                ),
                "max_tokens": max_tokens,
                "model_size": getattr(model_size, "value", model_size),
                "prompt_chars": len(prompt),
            }
        )

        if response_model is CombinedExtraction:
            return {
                "extracted_entities": [
                    {
                        "name": "Legislative Council",
                        "entity_type_id": 0,
                    },
                    {
                        "name": "Technology and Living curriculum",
                        "entity_type_id": 0,
                    },
                ],
                "edges": [
                    {
                        "source_entity_name": "Legislative Council",
                        "target_entity_name": "Technology and Living curriculum",
                        "relation_type": "ASKED_ABOUT",
                        "fact": (
                            "The Legislative Council asked about the Technology "
                            "and Living curriculum."
                        ),
                        "episode_indices": [0],
                    }
                ],
            }

        if response_model is BatchEdgeTimestamps:
            return {
                "timestamps": [
                    {
                        "valid_at": "2026-08-20T00:00:00Z",
                        "invalid_at": None,
                    }
                ]
            }

        raise AssertionError(f"unexpected response model: {response_model!r}")


def _episode() -> EpisodicNode:
    return EpisodicNode(
        name="nonzero-combined-fixture",
        group_id="newsroom-call-shape",
        labels=[],
        source=EpisodeType.text,
        source_description="newsroom-eval-proposal",
        content=(
            "The Legislative Council asked about the Technology and Living curriculum."
        ),
        created_at=datetime(2026, 8, 21, tzinfo=UTC),
        valid_at=datetime(2026, 8, 20, tzinfo=UTC),
    )


def test_upstream_nonzero_combined_path_dispatches_timestamp_request() -> None:
    llm = _NonZeroCombinedRecordingLlm()
    clients = SimpleNamespace(llm_client=llm)

    async def run() -> tuple[list[Any], list[Any], dict[str, list[int]]]:
        return await extract_nodes_and_edges(
            clients,
            _episode(),
            [],
            custom_extraction_instructions=GRAPHITI_EXTRACTION_INSTRUCTIONS,
        )

    nodes, edges, node_episode_index_map = asyncio.run(run())

    assert [call["response_model"] for call in llm.calls] == [
        "CombinedExtraction",
        "BatchEdgeTimestamps",
    ]
    assert len(nodes) == 2
    assert len(edges) == 1
    assert edges[0].name == "ASKED_ABOUT"
    assert edges[0].valid_at == datetime(2026, 8, 20, tzinfo=UTC)
    assert edges[0].invalid_at is None
    assert sorted(node_episode_index_map.values()) == [[0], [0]]


def test_second_stage_experiment_plan_is_serial_bounded_and_owner_gated() -> None:
    payload = json.loads(_PLAN_PATH.read_text(encoding="utf-8"))

    assert payload["schema_version"] == (
        "newsroom.graphiti-token-effectiveness-experiment-plan.v1"
    )
    assert payload["parent_issue"] == 739
    assert payload["pull_request"] == 745
    assert payload["serial_issues"] == [746, 747, 748]
    assert payload["status"] == "COMPLETED_RETAINED_PACKET_REJECT"
    assert payload["serial_state"] == {
        "746": "COMPLETE_RETAINED_REJECT",
        "747": "COMPLETE_MERGED_PROVIDER_FREE_QUALIFICATION",
        "748": "READY_PROVIDER_FREE",
    }

    authority = payload["live_authority"]
    assert authority["authorised"] is False
    assert authority["requires_explicit_owner_instruction"] is True
    assert authority["maximum_cursor_sdk_model_leaves"] == 8
    assert authority["maximum_grok_model_leaves"] == 0
    assert authority["maximum_openrouter_calls"] == 0
    assert authority["adaptive_extra_calls_permitted"] is False
    assert authority["unchanged_request_retry_permitted"] is False
    assert authority["neo4j_mutation_permitted"] is False
    assert authority["publication_permitted"] is False

    sdk = payload["sdk_candidate"]
    assert sdk["model"] == "composer-2.5"
    assert sdk["runtime"] == "local"
    assert sdk["lifecycle"] == "Agent.prompt"
    assert sdk["tools"] == []
    assert sdk["mcp_servers"] == {}
    assert sdk["agents"] == {}
    assert sdk["custom_tools"] == {}
    assert sdk["setting_sources"] == "OMITTED"
    assert sdk["prior_messages"] == 0
    assert sdk["required_stream_tool_call_count"] is None
    assert sdk["agent_prompt_stream_exposure"] == (
        "UNAVAILABLE_IN_CURSOR_SDK_1_0_28_RUN_RESULT"
    )
    assert sdk["future_qualification_requires_observable_event_lifecycle"] is True

    shape = payload["call_shape"]
    assert shape["graphiti_core_version"] == "0.29.3"
    assert shape["upstream_combined_zero_edge_chat_leaves"] == 1
    assert shape["upstream_combined_nonzero_edge_chat_leaves"] == 2
    assert shape["upstream_nonzero_second_class"] == "BatchEdgeTimestamps"
    assert shape["candidate_compact_combined_temporal_zero_edge_chat_leaves"] == 1
    assert shape["candidate_compact_combined_temporal_nonzero_edge_chat_leaves"] == 1

    experiments = payload["experiments"]
    assert [item["ordinal"] for item in experiments] == list(range(1, 9))
    assert len({item["label"] for item in experiments}) == 8

    average = payload["average_token_model"]
    assert average["bernoulli_condition_terms_are_sufficient"] is False
    assert (
        average["primary_count_includes_expected_chunks_per_effective_revision"] is True
    )

    outcome = payload["retained_live_outcome"]
    assert outcome["provider_calls"] == 8
    assert outcome["tiny_input_tokens"] == 3_430
    assert outcome["semantic_pass_count"] == 4
    assert outcome["semantic_fail_count"] == 4
    assert outcome["invalid_zero_expectation_leaf_count"] == 2
    assert outcome["recommendation"] == "REJECT"

    over_limit = payload["provider_free_over_limit_proof"]
    assert over_limit["fixture"] == "MAX_EPISODE_BYTES + 50"
    assert over_limit["expected_chunk_count"] == 2
    assert over_limit["complete_ordered_reconstruction_required"] is True
    assert over_limit["provider_calls"] == 0

    decision = payload["decision_rule"]
    assert decision["minimum_tiny_input_reduction_fraction"] == 0.5
    assert decision["preferred_tiny_input_reduction_fraction"] == 0.75
    assert decision["maximum_tiny_input_tokens_for_minimum_effect"] == 10_051
    assert decision["quality_must_not_regress"] is True
    assert decision["source_content_must_not_be_truncated"] is True
    assert decision["tool_call_count_must_equal"] == 0
    assert decision["tool_call_execution_must_be_observed_for_qualification"] is True
    assert decision["missing_usage_is_zero"] is False


def test_missing_node_names_are_batched_before_pinned_bulk_persistence() -> None:
    import newsroom.graphiti_adapter.real as real
    from graphiti_core.driver.driver import GraphProvider
    from graphiti_core.nodes import EntityNode
    from graphiti_core.utils.bulk_utils import add_nodes_and_edges_bulk_tx
    from newsroom.graphiti_adapter.embedding_meter import MeteredOpenAIEmbedder

    provider_inputs: list[list[str]] = []

    class Embeddings:
        async def create(self, **values: object) -> object:
            inputs = list(values["input"])
            provider_inputs.append(inputs)
            return SimpleNamespace(
                id="batch-1",
                data=[
                    SimpleNamespace(embedding=[float(index), 0.5, 99.0])
                    for index, _item in enumerate(inputs)
                ],
                usage={
                    "prompt_tokens": len(inputs),
                    "total_tokens": len(inputs),
                    "cost": "0",
                },
            )

    delegate = SimpleNamespace(
        client=SimpleNamespace(embeddings=Embeddings()),
        config=SimpleNamespace(embedding_model="model", embedding_dim=2),
    )
    embedder = MeteredOpenAIEmbedder(delegate)
    nodes = [
        EntityNode(name=f"node\n{index}", group_id="group")
        for index in range(7)
    ]
    retained = EntityNode(
        name="retained",
        group_id="group",
        name_embedding=[8.0, 8.0],
    )
    nodes.append(retained)
    persisted: list[dict[str, object]] = []

    class Operations:
        async def episodic_node_save_bulk(self, *_args: object) -> None:
            return None

        async def node_save_bulk(
            self,
            _name: object,
            _driver: object,
            _tx: object,
            values: list[dict[str, object]],
        ) -> None:
            persisted.extend(values)

        async def episodic_edge_save_bulk(self, *_args: object) -> None:
            return None

        async def edge_save_bulk(self, *_args: object) -> None:
            return None

    driver = SimpleNamespace(
        provider=GraphProvider.NEO4J,
        graph_operations_interface=Operations(),
    )

    async def exercise() -> None:
        await real._batch_missing_node_name_embeddings(embedder, nodes)
        await add_nodes_and_edges_bulk_tx(
            object(), [], [], nodes, [], embedder, driver
        )

    asyncio.run(exercise())

    assert provider_inputs == [[f"node {index}" for index in range(7)]]
    assert [node.name_embedding for node in nodes[:-1]] == [
        [float(index), 0.5] for index in range(7)
    ]
    assert retained.name_embedding == [8.0, 8.0]
    assert [item["name_embedding"] for item in persisted] == [
        *[[float(index), 0.5] for index in range(7)],
        [8.0, 8.0],
    ]
    assert embedder.receipt()["request_count"] == 1
