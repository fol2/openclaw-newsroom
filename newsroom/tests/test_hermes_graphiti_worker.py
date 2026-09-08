from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from newsroom.authority.canonical import digest_canonical
from newsroom.control_plane.graphiti_events import GraphitiProcessResult


GRAPH_DESTINATION_ID = "sha256:" + "9" * 64


def _arguments(*extra: str) -> list[str]:
    return [
        "--event-id",
        "event-1",
        "--ledger-seq",
        "7",
        "--max-reserved-gbp-microunits",
        "500000",
        *extra,
    ]


def test_exact_event_is_preflighted_time_bounded_and_fallback_free(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from scripts import hermes_graphiti_worker as worker

    captured: dict[str, object] = {}

    class Runner:
        pass

    def qualify(**kwargs: object) -> dict[str, object]:
        captured["preflight"] = kwargs
        return {
            "evidence_digest": "sha256:" + "a" * 64,
            "resolved_units": [{"ingest_id": "ingest-1"}],
        }

    def consume(**kwargs: object) -> GraphitiProcessResult:
        captured["consume"] = kwargs
        return GraphitiProcessResult("event-1", 7, "TERMINAL", 1)

    monkeypatch.setattr(worker, "qualify_fresh_graphiti_event", qualify)
    monkeypatch.setattr(worker, "consume_next_graphiti_event", consume)
    monkeypatch.setattr(worker, "ensure_control_plane_state_root", lambda: None)

    runtime = worker._mint_graphiti_campaign_runtime(
        graphiti=Runner(),
        admission_factory=lambda _connection: object(),
        bind_unit_authority=lambda unit: unit,
        graph_state_fence=lambda _campaign: {},
        graph_destination_id=GRAPH_DESTINATION_ID,
        authority_store_source_path="/authority.sqlite3",
        authority_store_descriptor_digest="sha256:" + "a" * 64,
    )
    assert (
        worker.main(
            _arguments("--max-runtime-seconds", "45"),
            runtime=runtime,
        )
        == 0
    )

    assert captured["preflight"] == {
        "proving_store": str(worker.CANONICAL_PROVING_STORE),
        "unpublished_store": str(worker.CANONICAL_UNPUBLISHED_STORE),
        "event_id": "event-1",
        "ledger_seq": 7,
    }
    consumed = captured["consume"]
    assert isinstance(consumed, dict)
    assert consumed["event_id"] == "event-1"
    assert consumed["require_fresh"] is True
    assert consumed["recover_model_usage"] is False
    assert consumed["max_dispatch_seconds"] == 45
    assert consumed["prepared_event_preflight"] == {
        "evidence_digest": "sha256:" + "a" * 64,
        "resolved_units": [{"ingest_id": "ingest-1"}],
    }
    assert consumed["max_reserved_gbp_microunits"] == 500000
    assert consumed["graphiti"] is runtime.graphiti
    assert consumed["graphiti_admission_factory"] is runtime.admission_factory
    assert consumed["unit_authority_resolver"] is runtime.bind_unit_authority
    assert consumed["require_graphiti_admission"] is True
    bodies = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [body["event"] for body in bodies] == [
        "GRAPHITI_EVENT_PREFLIGHT",
        "GRAPHITI_EVENT_RESULT",
        "GRAPHITI_WORKER_STOPPED",
    ]
    assert bodies[-1]["reason"] == "EXACT_EVENT_TERMINAL"
    assert bodies[-1]["completed_events"] == 1


def test_runtime_composes_existing_4a_4d_4b_4c_4e_authorities(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from scripts import hermes_graphiti_worker as worker
    from newsroom.tests.authority_helpers import proof
    from newsroom.tests.test_graphiti_increment4_system import (
        TrackingMemoryNeo4jAdapter,
        _open,
    )

    captured: dict[str, object] = {}

    def runner(**kwargs: object) -> object:
        captured["runner"] = kwargs
        return object()

    def admission(connection: object, **kwargs: object) -> object:
        captured["connection"] = connection
        captured["admission"] = kwargs
        return object()

    monkeypatch.setattr(worker, "EvaluationGraphitiRunner", runner)
    monkeypatch.setattr(
        worker,
        "compose_existing_graphiti_admission_consumer",
        admission,
    )
    authority_system = _open(tmp_path, TrackingMemoryNeo4jAdapter())
    authority_proof = proof()
    bind_unit_authority = lambda unit: unit
    try:
        with pytest.raises(ValueError, match="path differs"):
            worker.compose_governed_graphiti_worker_runtime(
                authority_system=authority_system,
                expected_authority_store_path=str(tmp_path / "different.sqlite3"),
                authority_store_descriptor_digest="sha256:" + "a" * 64,
                proof=authority_proof,
                bind_unit_authority=bind_unit_authority,
                max_attempts=1,
            )

        runtime = worker.compose_governed_graphiti_worker_runtime(
            authority_system=authority_system,
            expected_authority_store_path=str(tmp_path / "authority.sqlite3"),
            authority_store_descriptor_digest="sha256:" + "a" * 64,
            proof=authority_proof,
            bind_unit_authority=bind_unit_authority,
            max_attempts=1,
        )
        connection = sqlite3.connect(":memory:")
        assert runtime.admission_factory(connection) is not None
        assert runtime.authority_store_source_path == str(
            (tmp_path / "authority.sqlite3").resolve()
        )
        assert runtime.authority_store_descriptor_digest == "sha256:" + "a" * 64
        assert runtime.graph_destination_id == authority_system.graph_destination_id
        assert runtime.bind_unit_authority is bind_unit_authority

        assert captured["runner"] == {
            "fallback_permitted": False,
            "proposal_adapter": authority_system.graphiti,
            "extraction_records": authority_system.extraction,
            "proof": authority_proof,
        }
        assert captured["connection"] is connection
        assert captured["admission"] == {
            "adapter": authority_system.graphiti,
            "extraction": authority_system.extraction,
            "objects": authority_system.objects,
            "entities": authority_system.entities,
            "relations": authority_system.relations,
            "increment4": authority_system.increment4,
            "proof": authority_proof,
            "max_attempts": 1,
        }
        connection.close()
    finally:
        authority_system.close()


def test_exact_event_not_claimed_is_fail_closed(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from scripts import hermes_graphiti_worker as worker

    assert worker._run(consume=lambda: None) == 2
    bodies = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert bodies[-1]["reason"] == "EXACT_EVENT_NOT_CLAIMED"


def test_exact_event_execution_refusal_is_structured_and_terminal(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from scripts import hermes_graphiti_worker as worker

    assert worker._run(consume=lambda: (_ for _ in ()).throw(ValueError("drift"))) == 2
    bodies = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert bodies[0]["reason"] == "EXACT_EVENT_EXECUTION_REFUSED"
    assert bodies[1]["failure_type"] == "ValueError"


@pytest.mark.parametrize(
    "state",
    ["RIGHTS_HELD", "RETRY_HELD", "CONFIGURATION_HELD", "DEAD_LETTER"],
)
def test_non_terminal_result_stops_without_another_selection(state: str) -> None:
    from scripts import hermes_graphiti_worker as worker

    calls: list[object] = []

    def consume() -> GraphitiProcessResult:
        calls.append("consume")
        return GraphitiProcessResult("event-1", 1, state, 1)

    assert worker._run(consume=consume) == 2
    assert calls == ["consume"]


def test_preflight_refusal_has_no_runner_or_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from scripts import hermes_graphiti_worker as worker

    monkeypatch.setattr(worker, "ensure_control_plane_state_root", lambda: None)
    monkeypatch.setattr(
        worker,
        "qualify_fresh_graphiti_event",
        lambda **_kwargs: (_ for _ in ()).throw(ValueError("not fresh")),
    )
    monkeypatch.setattr(
        worker,
        "consume_next_graphiti_event",
        lambda **_kwargs: pytest.fail("dispatch reached after refused preflight"),
    )

    runtime = worker._mint_graphiti_campaign_runtime(
        graphiti=object(),
        admission_factory=lambda _connection: object(),
        bind_unit_authority=lambda unit: unit,
        graph_state_fence=lambda _campaign: {},
        graph_destination_id=GRAPH_DESTINATION_ID,
        authority_store_source_path="/authority.sqlite3",
        authority_store_descriptor_digest="sha256:" + "a" * 64,
    )
    assert worker.main(_arguments(), runtime=runtime) == 2
    bodies = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert bodies[0]["reason"] == "PREFLIGHT_REFUSED"
    assert bodies[1]["failure_type"] == "ValueError"


def test_conservative_spend_bound_refuses_before_runner_or_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from scripts import hermes_graphiti_worker as worker

    monkeypatch.setattr(worker, "ensure_control_plane_state_root", lambda: None)
    monkeypatch.setattr(
        worker,
        "qualify_fresh_graphiti_event",
        lambda **_kwargs: {
            "resolved_units": [
                {"ingest_id": "ingest-1"},
                {"ingest_id": "ingest-2"},
            ]
        },
    )
    runtime = worker._mint_graphiti_campaign_runtime(
        graphiti=object(),
        admission_factory=lambda _connection: object(),
        bind_unit_authority=lambda unit: unit,
        graph_state_fence=lambda _campaign: {},
        graph_destination_id=GRAPH_DESTINATION_ID,
        authority_store_source_path="/authority.sqlite3",
        authority_store_descriptor_digest="sha256:" + "a" * 64,
    )
    assert worker.main(_arguments(), runtime=runtime) == 2
    bodies = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert bodies == [
        {
            "event": "GRAPHITI_WORKER_STOPPED",
            "reason": "RESERVED_SPEND_BOUND_EXCEEDED",
            "completed_events": 0,
            "result": None,
            "public_dispatch": False,
            "auto_publish": False,
        }
    ]


@pytest.mark.parametrize(
    "argv",
    (
        [],
        ["--event-id", "event-1"],
        ["--ledger-seq", "1"],
        _arguments("--max-runtime-seconds", "0"),
        _arguments("--max-runtime-seconds", "nan"),
        _arguments("--max-runtime-seconds", "inf"),
        [
            "--event-id",
            "event-1",
            "--ledger-seq",
            "0",
            "--max-reserved-gbp-microunits",
            "500000",
        ],
        [
            "--event-id",
            "event-1",
            "--ledger-seq",
            "1",
            "--max-reserved-gbp-microunits",
            "0",
        ],
    ),
)
def test_worker_refuses_missing_or_invalid_bounds(argv: list[str]) -> None:
    from scripts import hermes_graphiti_worker as worker

    with pytest.raises(SystemExit):
        worker.main(argv)


def test_unconfigured_authority_stops_before_preflight_or_provider(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from scripts import hermes_graphiti_worker as worker

    monkeypatch.setattr(
        worker,
        "qualify_fresh_graphiti_event",
        lambda **_kwargs: pytest.fail("preflight reached without authority runtime"),
    )

    assert worker.main(_arguments()) == 2
    assert json.loads(capsys.readouterr().out)["reason"] == (
        "AUTHORITY_COMPOSITION_UNCONFIGURED"
    )


def test_campaign_cli_requires_injected_runtime_and_f4_fence_before_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from scripts import hermes_graphiti_worker as worker

    packet_path = tmp_path / "missing.json"
    monkeypatch.setattr(
        worker,
        "_current_git_identity",
        lambda: pytest.fail("git identity reached without campaign authority"),
    )

    assert worker.main(["--campaign-packet", str(packet_path)]) == 2
    assert json.loads(capsys.readouterr().out)["reason"] == (
        "CAMPAIGN_AUTHORITY_UNCONFIGURED"
    )


def test_campaign_cli_executes_exact_packet_through_injected_fence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from scripts import hermes_graphiti_worker as worker

    packet = {"packet_digest": "sha256:" + "a" * 64}
    packet_path = tmp_path / "campaign.json"
    packet_path.write_text(json.dumps(packet), encoding="utf-8")
    runtime = worker._mint_graphiti_campaign_runtime(
        graphiti=object(),
        admission_factory=lambda _connection: object(),
        bind_unit_authority=lambda unit: unit,
        graph_state_fence=lambda _campaign: {},
        graph_destination_id=GRAPH_DESTINATION_ID,
        authority_store_source_path="/authority.sqlite3",
        authority_store_descriptor_digest="sha256:" + "a" * 64,
    )
    fence = lambda _packet: None
    captured: dict[str, object] = {}

    monkeypatch.setattr(worker, "_current_git_identity", lambda: ("head", "tree"))

    def run(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"state": "CAMPAIGN_COMPLETE"}

    monkeypatch.setattr(worker, "run_bounded_campaign", run)

    assert (
        worker.main(
            ["--campaign-packet", str(packet_path)],
            runtime=runtime,
            owner_f4_fence=fence,
        )
        == 0
    )
    assert captured == {
        "packet": packet,
        "proving_store": str(worker.CANONICAL_PROVING_STORE),
        "unpublished_store": str(worker.CANONICAL_UNPUBLISHED_STORE),
        "runtime": runtime,
        "head_sha": "head",
        "tree_sha": "tree",
        "owner_f4_fence": fence,
    }
    assert json.loads(capsys.readouterr().out) == {
        "event": "GRAPHITI_CAMPAIGN_RESULT",
        "result": {"state": "CAMPAIGN_COMPLETE"},
        "public_dispatch": False,
        "auto_publish": False,
    }


def test_campaign_cli_stop_reports_durable_partial_progress(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from scripts import hermes_graphiti_worker as worker

    unpublished = tmp_path / "unpublished.sqlite3"
    connection = worker.connect(str(unpublished))
    connection.executemany(
        "INSERT INTO unpublished_graphiti_revision_events("
        "event_id,ledger_seq,ledger_digest,source_id,item_key,revision_digest,"
        "landed_at,manifest_json,manifest_digest,unit_count,projector_version,"
        "projection_generation,state,attempt_count,available_at,"
        "provider_dispatched,terminal_at,proposal_count,last_failure_code"
        ") VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            (
                "event-1",
                1,
                "ledger-1",
                "source",
                "item-1",
                "revision-1",
                "2026-09-01T12:00:00Z",
                "{}",
                "manifest-1",
                1,
                "projector",
                "generation",
                "TERMINAL",
                1,
                "2026-09-01T12:00:00Z",
                1,
                "2026-09-01T12:00:01Z",
                1,
                None,
            ),
            (
                "event-2",
                2,
                "ledger-2",
                "source",
                "item-2",
                "revision-2",
                "2026-09-01T12:00:00Z",
                "{}",
                "manifest-2",
                1,
                "projector",
                "generation",
                "RUNNING",
                1,
                "2026-09-01T12:00:00Z",
                1,
                None,
                None,
                None,
            ),
        ),
    )
    connection.execute(
        "INSERT INTO unpublished_graphiti_spend("
        "spend_id,ingest_id,attempt_number,proving_run_id,generation_id,"
        "reserved_gbp_microunits,actual_usd_microunits,"
        "actual_gbp_microunits,usage_basis,status,provider_usage_json,"
        "dispatch_owner,dispatch_lease_expires_at,at"
        ") VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            "spend-1",
            "ingest-1",
            1,
            "run",
            "generation",
            100,
            23,
            23,
            "PROVIDER_REPORTED",
            "RECONCILED",
            "{}",
            None,
            None,
            "2026-09-01T12:00:01Z",
        ),
    )
    connection.commit()
    connection.close()
    packet = {
        "packet_digest": "sha256:" + "a" * 64,
        "bounded_campaign": {
            "cohort": {
                "events": [
                    {
                        "event_id": "event-1",
                        "ledger_seq": 1,
                        "ingest_ids": ["ingest-1"],
                    },
                    {
                        "event_id": "event-2",
                        "ledger_seq": 2,
                        "ingest_ids": ["ingest-2"],
                    },
                ]
            }
        },
    }
    packet_path = tmp_path / "campaign.json"
    packet_path.write_text(json.dumps(packet), encoding="utf-8")
    runtime = worker._mint_graphiti_campaign_runtime(
        graphiti=object(),
        admission_factory=lambda _connection: object(),
        bind_unit_authority=lambda unit: unit,
        graph_state_fence=lambda _campaign: {},
        graph_destination_id=GRAPH_DESTINATION_ID,
        authority_store_source_path="/authority.sqlite3",
        authority_store_descriptor_digest="sha256:" + "a" * 64,
    )
    monkeypatch.setattr(worker, "_current_git_identity", lambda: ("head", "tree"))
    monkeypatch.setattr(
        worker,
        "run_bounded_campaign",
        lambda **_kwargs: (_ for _ in ()).throw(
            worker.GraphitiCampaignStop(
                "event 2 stopped",
                evidence={
                    "arrival_count": 1,
                    "actionable_gaps": [
                        {
                            "ledger_seq": 3,
                            "event_id": "event-3",
                            "kind": "PROJECT_EVENT_GAP",
                        }
                    ],
                },
            )
        ),
    )

    assert (
        worker.main(
            [
                "--campaign-packet",
                str(packet_path),
                "--unpublished",
                str(unpublished),
            ],
            runtime=runtime,
            owner_f4_fence=lambda _packet: None,
        )
        == 2
    )
    body = json.loads(capsys.readouterr().out)
    assert body["event"] == "GRAPHITI_CAMPAIGN_STOPPED"
    report = body["result"]
    assert report["packet_digest"] == packet["packet_digest"]
    assert report["stage"] == "EXTRACTION_RECORDED"
    assert report["selected_event_count"] == 2
    assert report["completed_event_count"] == 1
    assert report["terminal_event_count"] == 1
    assert report["attempted_event_count"] == 2
    assert report["provider_dispatched_event_count"] == 2
    assert report["failure_evidence"] == {
        "arrival_count": 1,
        "actionable_gaps": [
            {
                "ledger_seq": 3,
                "event_id": "event-3",
                "kind": "PROJECT_EVENT_GAP",
            }
        ],
    }
    assert report["spend"] == {
        "row_count": 1,
        "status_counts": {"RECONCILED": 1},
        "reconciled_actual_gbp_microunits": 23,
        "actual_gbp_complete": False,
        "actual_gbp_microunits": None,
    }
    assert [item["state"] for item in report["events"]] == [
        "TERMINAL",
        "RUNNING",
    ]


def test_campaign_stop_report_attributes_all_hold_reconciliation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import hermes_graphiti_worker as worker
    from newsroom.control_plane.graphiti_admission import (
        graphiti_admission_generation_identity,
    )

    unpublished = tmp_path / "unpublished.sqlite3"
    connection = worker.connect(str(unpublished))
    connection.execute(
        "INSERT INTO unpublished_graphiti_revision_events("
        "event_id,ledger_seq,ledger_digest,source_id,item_key,revision_digest,"
        "landed_at,manifest_json,manifest_digest,unit_count,projector_version,"
        "projection_generation,state,attempt_count,available_at,"
        "provider_dispatched,terminal_at,proposal_count"
        ") VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            "event-1",
            1,
            "ledger-1",
            "source",
            "item-1",
            "revision-1",
            "2026-09-01T12:00:00Z",
            "{}",
            "manifest-1",
            1,
            "projector",
            "generation",
            "TERMINAL",
            1,
            "2026-09-01T12:00:00Z",
            1,
            "2026-09-01T12:00:01Z",
            1,
        ),
    )
    connection.execute(
        "INSERT INTO unpublished_graphiti_spend("
        "spend_id,ingest_id,attempt_number,proving_run_id,generation_id,"
        "reserved_gbp_microunits,actual_usd_microunits,"
        "actual_gbp_microunits,usage_basis,status,provider_usage_json,"
        "dispatch_owner,dispatch_lease_expires_at,at"
        ") VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            "spend-1",
            "ingest-1",
            1,
            "run",
            "generation",
            100,
            0,
            0,
            "PROVIDER_REPORTED",
            "RECONCILED",
            "{}",
            None,
            None,
            "2026-09-01T12:00:01Z",
        ),
    )
    connection.execute(
        "INSERT INTO unpublished_graphiti_admission_queue("
        "proposal_key,ingest_id,source_revision_id,source_receipt_digest,"
        "proposal_digest,proposal_kind,request_json,request_digest,state,"
        "created_at,updated_at"
        ") VALUES(?,?,?,?,?,?,?,?,?,?,?)",
        (
            "proposal-1",
            "ingest-1",
            "revision-1",
            "receipt-1",
            "proposal-digest-1",
            "ENTITY_MENTION",
            "{}",
            "request-digest-1",
            "TERMINAL",
            "2026-09-01T12:00:01Z",
            "2026-09-01T12:00:01Z",
        ),
    )
    connection.execute(
        "INSERT INTO unpublished_graphiti_admission_decisions("
        "proposal_key,action,decision_id,authority_ledger_seq,reason_code,"
        "authority_receipt_digest,decision_json,decision_digest,decided_at"
        ") VALUES(?,?,?,?,?,?,?,?,?)",
        (
            "proposal-1",
            "HOLD",
            "decision-1",
            1,
            "AMBIGUOUS",
            "authority-receipt-1",
            "{}",
            "decision-digest-1",
            "2026-09-01T12:00:01Z",
        ),
    )
    cohort_digest, generation_id = graphiti_admission_generation_identity(
        ingest_ids=("ingest-1",),
        source_receipts=(
            {
                "ingest_id": "ingest-1",
                "receipt_digest": "sha256:" + "1" * 64,
                "proposal_count": 1,
            },
        ),
        members=(
            {
                "ingest_id": "ingest-1",
                "proposal_key": "proposal-1",
                "proposal_envelope_id": "envelope-1",
                "decision_digest": "sha256:" + "2" * 64,
                "decision": {},
            },
        ),
    )
    reconciliation_digest = "sha256:" + "c" * 64
    raw_reconciliation = {
        "generation_id": generation_id,
        "expected_effect_ids": [],
        "actual_effect_ids": [],
        "authority_watermark": 1,
        "receipt_digest": reconciliation_digest,
        "projector_family_id": "graph.increment4.admitted",
        "provider_model_calls": 0,
    }
    reconciliation = {
        "schema_version": (
            worker.GRAPHITI_ADMISSION_RECONCILIATION_SCHEMA_VERSION
        ),
        "cohort_digest": cohort_digest,
        "ingest_ids": ["ingest-1"],
        "raw_receipt": raw_reconciliation,
    }
    connection.execute(
        "INSERT INTO unpublished_graphiti_projection_reconciliations("
        "receipt_digest,projector_family_id,generation_id,authority_watermark,"
        "receipt_json,reconciled_at"
        ") VALUES(?,?,?,?,?,?)",
        (
            reconciliation_digest,
            "graph.increment4.admitted",
            generation_id,
            1,
            worker.canonical_json_bytes(reconciliation).decode("utf-8"),
            "2026-09-01T12:00:02Z",
        ),
    )
    connection.commit()
    connection.close()
    monkeypatch.setattr(
        worker,
        "graphiti_decided_cohort_generation_identity",
        lambda _connection, *, ingest_ids: (
            cohort_digest,
            generation_id,
        ),
    )
    packet = {
        "packet_digest": "sha256:" + "a" * 64,
        "bounded_campaign": {
            "cohort": {
                "events": [
                    {
                        "event_id": "event-1",
                        "ledger_seq": 1,
                        "ingest_ids": ["ingest-1"],
                    }
                ]
            }
        },
    }

    report = worker._campaign_stop_report(
        packet=packet,
        unpublished_store=str(unpublished),
        failure=worker.GraphitiCampaignStop("wall time reached"),
    )

    assert report["stage"] == "PROJECTION_RECORDED"
    assert report["admission"]["projection_receipt_count"] == 0
    assert report["generation"] == {
        "cohort_digest": cohort_digest,
        "generation_id": generation_id,
        "reconciliation_count": 1,
        "reconciliations": [
            {
                "receipt_digest": reconciliation_digest,
                "projector_family_id": "graph.increment4.admitted",
                "generation_id": generation_id,
                "authority_watermark": 1,
                "expected_effect_ids": [],
                "actual_effect_ids": [],
                "reconciled_at": "2026-09-01T12:00:02Z",
            }
        ],
        "reconciliation_attribution_complete": True,
    }


def test_campaign_receipt_requires_exact_attempt_and_reconciled_spend(
    tmp_path: Path,
) -> None:
    from scripts import hermes_graphiti_worker as worker

    path = tmp_path / "unpublished.sqlite3"
    connection = worker.connect(str(path))
    embedding = {
        "usage_basis": "PROVIDER_REPORTED",
        "cost_usd_microunits": 3,
        "embedding_tokens": 5,
        "request_count": 1,
        "requests": [
            {
                "provider": "embedding-provider",
                "model": "embedding",
                "outcome": "COMPLETE",
                "cost_reported": True,
                "cost_usd_microunits": 3,
                "model_invocation_id": "embedding-invocation",
            }
        ],
    }
    receipt = {
        "ingest_id": "ingest-1",
        "outcome": "COMPLETE",
        "failure_code": "NONE",
        "proposal_count": 0,
        "attempt_number": 1,
        "provider_attempt_number": 1,
        "chat_invocations": [
            {
                "provider": "cursor-agent-cli",
                "model": "model",
                "outcome": "COMPLETE",
                "model_invocation_id": "chat-invocation",
                "usage": {"usage_basis": "PROVIDER_REPORTED"},
                "transport_qualification": {
                    "transport": "CURSOR_SDK",
                    "max_retries": 0,
                },
            }
        ],
        "embedding_usage": embedding,
        "accounting": {
            "spend_id": "ingest-1:1",
            "status": "RECONCILED",
            "usage_basis": "PROVIDER_REPORTED",
            "actual_usd_microunits": 3,
            "actual_gbp_microunits": 3,
            "unused_reservation_released": True,
        },
    }
    receipt_digest = digest_canonical(receipt)
    receipt = {**receipt, "receipt_digest": receipt_digest}
    encoded = json.dumps(receipt, ensure_ascii=False, sort_keys=True)
    connection.execute(
        "INSERT INTO unpublished_graphiti_ingest VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            "ingest-1",
            "source",
            "item",
            "COMPLETE",
            0,
            0,
            0,
            "NONE",
            "PUBLICATION_TIME",
            "2026-09-01T12:00:00Z",
            "generation",
            receipt_digest,
            "2026-09-01T12:00:00Z",
        ),
    )
    connection.execute(
        "INSERT INTO unpublished_graphiti_receipts VALUES(?,?)",
        ("ingest-1", encoded),
    )
    connection.execute(
        "INSERT INTO unpublished_graphiti_attempt_receipts VALUES(?,?,?,?,?,?)",
        (
            "ingest-1",
            1,
            "COMPLETE",
            receipt_digest,
            encoded,
            "2026-09-01T12:00:00Z",
        ),
    )
    connection.execute(
        "INSERT INTO unpublished_graphiti_spend VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            "ingest-1:1",
            "ingest-1",
            1,
            "run",
            "generation",
            500_000,
            3,
            3,
            "PROVIDER_REPORTED",
            "RECONCILED",
            json.dumps(embedding, sort_keys=True),
            None,
            None,
            "2026-09-01T12:00:00Z",
        ),
    )
    connection.commit()
    connection.close()

    evidence = worker._campaign_receipt_evidence(
        str(path),
        ingest_ids=("ingest-1",),
        provider={
            "provider_id": "cursor-agent-cli",
            "transport_id": "CURSOR_SDK",
            "model_id": "model",
            "embedding_provider_id": "embedding-provider",
            "embedding_model_id": "embedding",
        },
    )
    assert evidence == {
        "proposal_count": 0,
        "chat_invocation_count": 1,
        "embedding_request_count": 1,
        "fallback_count": 0,
        "retry_count": 0,
        "actual_gbp_microunits": 3,
    }
    with pytest.raises(
        worker.GraphitiCampaignStop,
        match="already have retained effects",
    ):
        worker._assert_fresh_campaign_ingests(
            str(path), ingest_ids=("ingest-1",)
        )

    with pytest.raises(worker.GraphitiCampaignStop, match="transport"):
        worker._campaign_receipt_evidence(
            str(path),
            ingest_ids=("ingest-1",),
            provider={
                "provider_id": "cursor-agent-cli",
                "transport_id": "RETIRED_TRANSPORT",
                "model_id": "model",
                "embedding_provider_id": "embedding-provider",
                "embedding_model_id": "embedding",
            },
        )

    connection = sqlite3.connect(path)
    connection.execute(
        "UPDATE unpublished_graphiti_spend SET status='UNRECONCILED'"
    )
    connection.commit()
    connection.close()
    with pytest.raises(worker.GraphitiCampaignStop, match="spend accounting drifted"):
        worker._campaign_receipt_evidence(
            str(path),
            ingest_ids=("ingest-1",),
            provider={
                "provider_id": "cursor-agent-cli",
                "transport_id": "CURSOR_SDK",
                "model_id": "model",
                "embedding_provider_id": "embedding-provider",
                "embedding_model_id": "embedding",
            },
        )


def _insert_campaign_event(
    connection: sqlite3.Connection,
    *,
    ledger_seq: int,
    state: str,
    landed_at: datetime,
    available_at: datetime | None = None,
    terminal_at: datetime | None = None,
    project_event: bool = True,
) -> str:
    from scripts import hermes_graphiti_worker as worker

    event_id = "sha256:" + f"{ledger_seq:064x}"
    previous = connection.execute(
        "SELECT digest FROM ledger ORDER BY seq DESC LIMIT 1"
    ).fetchone()
    connection.execute(
        "INSERT INTO ledger("
        "seq,at,kind,payload_digest,payload_json,prev_digest,digest) "
        "VALUES(?,?,?,?,?,?,?)",
        (
            ledger_seq,
            landed_at.isoformat().replace("+00:00", "Z"),
            "EFFECTIVE_REVISION_LANDED",
            "sha256:" + f"{ledger_seq + 1000:064x}",
            "{}",
            previous[0] if previous else "sha256:" + "0" * 64,
            event_id,
        ),
    )
    connection.execute(
        "INSERT INTO unpublished_effective_revision_landed("
        "source_id,item_key,revision_digest,published_at,updated_at,"
        "first_observed_at,ingest_ids_json,legacy_v10,payload_digest,"
        "ledger_digest,at) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
        (
            f"source-{ledger_seq}",
            f"item-{ledger_seq}",
            f"revision-{ledger_seq}",
            "",
            "",
            landed_at.isoformat().replace("+00:00", "Z"),
            "[]",
            0,
            "sha256:" + f"{ledger_seq + 1000:064x}",
            event_id,
            landed_at.isoformat().replace("+00:00", "Z"),
        ),
    )
    if project_event:
        connection.execute(
            "INSERT INTO unpublished_graphiti_revision_events("
            "event_id,ledger_seq,ledger_digest,source_id,item_key,revision_digest,"
            "published_at,updated_at,landed_at,manifest_json,manifest_digest,"
            "unit_count,projector_version,projection_generation,state,attempt_count,"
            "available_at,provider_dispatched,terminal_at,proposal_count) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                event_id,
                ledger_seq,
                event_id,
                f"source-{ledger_seq}",
                f"item-{ledger_seq}",
                f"revision-{ledger_seq}",
                "",
                "",
                landed_at.isoformat().replace("+00:00", "Z"),
                "{}",
                "sha256:" + f"{ledger_seq + 2000:064x}",
                1,
                "projector",
                "generation",
                state,
                1 if state == "TERMINAL" else 0,
                (available_at or landed_at).isoformat().replace("+00:00", "Z"),
                1 if state == "TERMINAL" else 0,
                (
                    None
                    if terminal_at is None
                    else terminal_at.isoformat().replace("+00:00", "Z")
                ),
                0 if state == "TERMINAL" else None,
            ),
        )
    return event_id


def _complete_campaign_event(
    connection: sqlite3.Connection,
    *,
    event_id: str,
    terminal_at: datetime,
) -> None:
    connection.execute(
        "UPDATE unpublished_graphiti_revision_events SET state='TERMINAL',"
        "attempt_count=1,provider_dispatched=1,terminal_at=?,proposal_count=1 "
        "WHERE event_id=?",
        (terminal_at.isoformat().replace("+00:00", "Z"), event_id),
    )


def _insert_all_hold_generation(
    connection: sqlite3.Connection,
    *,
    binding_cohort_digest: str | None = None,
) -> tuple[str, str]:
    from scripts import hermes_graphiti_worker as worker
    from newsroom.control_plane.graphiti_admission import (
        graphiti_admission_generation_identity,
    )

    connection.execute(
        "INSERT INTO unpublished_graphiti_admission_queue("
        "proposal_key,ingest_id,source_revision_id,source_receipt_digest,"
        "proposal_digest,proposal_kind,request_json,request_digest,state,"
        "created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
        (
            "proposal-1",
            "ingest-1",
            "revision-1",
            "sha256:" + "1" * 64,
            "sha256:" + "2" * 64,
            "ENTITY_MENTION",
            "{}",
            "sha256:" + "3" * 64,
            "TERMINAL",
            "2026-09-01T12:00:01Z",
            "2026-09-01T12:00:01Z",
        ),
    )
    connection.execute(
        "INSERT INTO unpublished_graphiti_admission_decisions("
        "proposal_key,action,decision_id,authority_ledger_seq,reason_code,"
        "authority_receipt_digest,decision_json,decision_digest,decided_at) "
        "VALUES(?,?,?,?,?,?,?,?,?)",
        (
            "proposal-1",
            "HOLD",
            "decision-1",
            1,
            "AMBIGUOUS",
            "sha256:" + "4" * 64,
            "{}",
            "sha256:" + "5" * 64,
            "2026-09-01T12:00:02Z",
        ),
    )
    cohort_digest, generation_id = graphiti_admission_generation_identity(
        ingest_ids=("ingest-1",),
        source_receipts=(
            {
                "ingest_id": "ingest-1",
                "receipt_digest": "sha256:" + "1" * 64,
                "proposal_count": 1,
            },
        ),
        members=(
            {
                "ingest_id": "ingest-1",
                "proposal_key": "proposal-1",
                "proposal_envelope_id": "envelope-1",
                "decision_digest": "sha256:" + "5" * 64,
                "decision": {},
            },
        ),
    )
    receipt_digest = "sha256:" + "6" * 64
    raw_receipt = {
        "generation_id": generation_id,
        "expected_effect_ids": [],
        "actual_effect_ids": [],
        "authority_watermark": 1,
        "receipt_digest": receipt_digest,
        "projector_family_id": "graph.increment4.admitted",
        "provider_model_calls": 0,
    }
    receipt = {
        "schema_version": (
            worker.GRAPHITI_ADMISSION_RECONCILIATION_SCHEMA_VERSION
        ),
        "cohort_digest": binding_cohort_digest or cohort_digest,
        "ingest_ids": ["ingest-1"],
        "raw_receipt": raw_receipt,
    }
    connection.execute(
        "INSERT INTO unpublished_graphiti_projection_reconciliations "
        "VALUES(?,?,?,?,?,?)",
        (
            receipt_digest,
            "graph.increment4.admitted",
            generation_id,
            1,
            worker.canonical_json_bytes(receipt).decode("utf-8"),
            "2026-09-01T12:00:03Z",
        ),
    )
    return cohort_digest, generation_id


def _campaign_event(
    event_id: str,
    *,
    ledger_seq: int = 7,
) -> dict[str, object]:
    return {
        "event_id": event_id,
        "ledger_seq": ledger_seq,
        "manifest_digest": f"manifest-{ledger_seq}",
        "ingest_ids": [f"ingest-{ledger_seq}"],
    }


def _fresh_actionable(
    event: Mapping[str, object],
    *,
    landed_at: datetime,
) -> dict[str, object]:
    return {
        "ledger_seq": event["ledger_seq"],
        "event_id": event["event_id"],
        "landed_at": landed_at.isoformat().replace("+00:00", "Z"),
        "kind": "FRESH_EVENT",
        "manifest_digest": event["manifest_digest"],
        "ingest_ids": event["ingest_ids"],
    }


def _operational_snapshot(
    *,
    observed_at: datetime,
    actionable: list[dict[str, object]],
    holds: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    value = {
        "observed_at": observed_at.isoformat().replace("+00:00", "Z"),
        "partition_digest": digest_canonical(actionable),
        "actionable": actionable,
        "holds": [] if holds is None else holds,
    }
    return {**value, "snapshot_digest": digest_canonical(value)}


def _campaign_completion_fixture(
    tmp_path: Path,
    *,
    with_pre_frontier_hold: bool = False,
) -> tuple[
    sqlite3.Connection,
    datetime,
    dict[str, object],
    dict[str, object],
    dict[str, object],
]:
    from scripts import hermes_graphiti_worker as worker

    start = datetime(2026, 9, 1, 12, 0, tzinfo=UTC)
    connection = worker.connect(str(tmp_path / "unpublished.sqlite3"))
    if with_pre_frontier_hold:
        _insert_campaign_event(
            connection,
            ledger_seq=6,
            state="QUEUED",
            landed_at=start - timedelta(seconds=6),
        )
    event_id = _insert_campaign_event(
        connection,
        ledger_seq=7,
        state="QUEUED",
        landed_at=start - timedelta(seconds=5),
    )
    event = _campaign_event(event_id)
    before = _operational_snapshot(
        observed_at=start,
        actionable=[
            _fresh_actionable(event, landed_at=start - timedelta(seconds=5))
        ],
    )
    _complete_campaign_event(
        connection,
        event_id=event_id,
        terminal_at=start + timedelta(seconds=5),
    )
    after = _operational_snapshot(
        observed_at=start + timedelta(seconds=10),
        actionable=[],
    )
    return connection, start, event, before, after


def _completion_evidence(
    connection: sqlite3.Connection,
    *,
    event: dict[str, object],
    before: dict[str, object],
    after: dict[str, object],
    expected_generation_identity: tuple[str, str] | None = None,
    proposal_count: int = 0,
    elapsed_seconds: float = 10.0,
    max_oldest_eligible_seconds: int = 60,
) -> dict[str, object]:
    from scripts import hermes_graphiti_worker as worker

    return worker._campaign_completion_evidence(
        connection,
        events=[event],
        operational_before=before,
        operational_after=after,
        reconciliation_ids_before=frozenset(),
        expected_generation_identity=expected_generation_identity,
        expected_ingest_ids=("ingest-1",),
        proposal_count=proposal_count,
        elapsed_seconds=elapsed_seconds,
        wall_time_cap=60,
        max_oldest_eligible_seconds=max_oldest_eligible_seconds,
    )


def test_campaign_completion_proves_exact_all_hold_operational_objectives(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from scripts import hermes_graphiti_worker as worker

    connection, _start, event, before, after = _campaign_completion_fixture(
        tmp_path
    )
    generation_identity = _insert_all_hold_generation(connection)
    exact_admission = {
        "total": True,
        "disjoint": True,
        "latest_generation_id": generation_identity[1],
        "cohorts": [
            {
                "cohort_digest": generation_identity[0],
                "generation_id": generation_identity[1],
                "ingest_ids": ["ingest-1"],
            }
        ],
    }
    monkeypatch.setattr(
        worker,
        "_exact_admission_reconciliation",
        lambda _connection: exact_admission,
    )
    evidence = _completion_evidence(
        connection,
        event=event,
        before=before,
        after=after,
        expected_generation_identity=generation_identity,
        proposal_count=1,
        elapsed_seconds=6.0,
        max_oldest_eligible_seconds=30,
    )

    assert evidence["watermark"]["terminal_ledger_seq"] == 7
    assert evidence["watermark"]["observed_operational_ledger_seq"] == 7
    assert evidence["backlog"]["remaining_actionable_at_watermark"] == []
    assert evidence["velocity"]["arrival_count"] == 0
    assert evidence["velocity"]["service_count"] == 1
    assert evidence["lag"]["oldest_post_watermark_eligible_seconds"] == 0
    assert evidence["reconciliation"]["new_generation_ids"] == [
        generation_identity[1]
    ]
    assert evidence["reconciliation"]["receipt"]["effect_ids"] == []
    assert evidence["reconciliation"]["exact_admission"] == exact_admission
    assert evidence["reconciliation"]["global_admission"][
        "projection_reconciled"
    ] is True
    connection.close()


def test_campaign_completion_rechecks_terminal_queue_in_final_transaction(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from scripts import hermes_graphiti_worker as worker

    connection, _start, event, before, after = _campaign_completion_fixture(
        tmp_path
    )
    generation_identity = _insert_all_hold_generation(connection)
    connection.execute(
        "UPDATE unpublished_graphiti_admission_queue SET state='READY' "
        "WHERE proposal_key='proposal-1'"
    )
    observed: dict[str, object] = {}

    def exact_admission(candidate: sqlite3.Connection) -> dict[str, object]:
        observed["in_transaction"] = candidate.in_transaction
        observed["queue_state"] = candidate.execute(
            "SELECT state FROM unpublished_graphiti_admission_queue "
            "WHERE proposal_key='proposal-1'"
        ).fetchone()[0]
        raise RuntimeError("non-terminal admission queue state")

    monkeypatch.setattr(
        worker,
        "_exact_admission_reconciliation",
        exact_admission,
    )
    with pytest.raises(
        worker.GraphitiCampaignStop,
        match="terminal admission reconciliation differs",
    ):
        _completion_evidence(
            connection,
            event=event,
            before=before,
            after=after,
            expected_generation_identity=generation_identity,
            proposal_count=1,
        )

    assert observed == {"in_transaction": True, "queue_state": "READY"}
    connection.close()


def test_campaign_rejects_unrelated_all_hold_reconciliation(
    tmp_path: Path,
) -> None:
    from scripts import hermes_graphiti_worker as worker

    connection, _start, event, before, after = _campaign_completion_fixture(
        tmp_path
    )
    generation_identity = _insert_all_hold_generation(
        connection,
        binding_cohort_digest="sha256:" + "8" * 64,
    )
    with pytest.raises(
        worker.GraphitiCampaignStop,
        match="projection reconciliation is malformed",
    ):
        _completion_evidence(
            connection,
            event=event,
            before=before,
            after=after,
            expected_generation_identity=generation_identity,
            proposal_count=1,
        )
    connection.close()


def test_campaign_completion_stops_when_arrival_velocity_exceeds_service(
    tmp_path: Path,
) -> None:
    from scripts import hermes_graphiti_worker as worker

    connection, start, event, before, _after = _campaign_completion_fixture(
        tmp_path
    )
    arrivals = [
        _campaign_event("sha256:" + f"{sequence:064x}", ledger_seq=sequence)
        for sequence in (8, 9)
    ]
    after = _operational_snapshot(
        observed_at=start + timedelta(seconds=10),
        actionable=[
            _fresh_actionable(
                arrival,
                landed_at=start + timedelta(seconds=int(arrival["ledger_seq"]) - 7),
            )
            for arrival in arrivals
        ],
    )
    with pytest.raises(
        worker.GraphitiCampaignStop,
        match="service velocity is below arrival velocity",
    ):
        _completion_evidence(
            connection,
            event=event,
            before=before,
            after=after,
        )
    connection.close()


def test_campaign_completion_counts_one_true_new_arrival(
    tmp_path: Path,
) -> None:
    connection, start, event, before, _after = _campaign_completion_fixture(
        tmp_path
    )
    arrival = _campaign_event("sha256:" + f"{8:064x}", ledger_seq=8)
    after = _operational_snapshot(
        observed_at=start + timedelta(seconds=10),
        actionable=[
            _fresh_actionable(arrival, landed_at=start + timedelta(seconds=1))
        ],
    )

    evidence = _completion_evidence(
        connection,
        event=event,
        before=before,
        after=after,
    )

    assert evidence["velocity"]["arrival_count"] == 1
    assert evidence["velocity"]["service_count"] == 1
    connection.close()


def test_campaign_completion_stops_on_oldest_post_watermark_lag(
    tmp_path: Path,
) -> None:
    from scripts import hermes_graphiti_worker as worker

    connection, start, event, before, _after = _campaign_completion_fixture(
        tmp_path
    )
    waiting = _campaign_event("sha256:" + f"{8:064x}", ledger_seq=8)
    after = _operational_snapshot(
        observed_at=start + timedelta(seconds=10),
        actionable=[
            _fresh_actionable(waiting, landed_at=start - timedelta(seconds=61))
        ],
    )
    with pytest.raises(
        worker.GraphitiCampaignStop,
        match="oldest eligible lag objective",
    ):
        _completion_evidence(
            connection,
            event=event,
            before=before,
            after=after,
            max_oldest_eligible_seconds=60,
        )
    connection.close()


def test_campaign_completion_reports_legitimate_hold_without_failing(
    tmp_path: Path,
) -> None:
    connection, start, event, before, _after = _campaign_completion_fixture(
        tmp_path
    )
    hold = {
        "ledger_seq": 8,
        "event_id": "sha256:" + f"{8:064x}",
        "reason": "CURRENT_RIGHTS_OR_INPUT_HELD",
    }
    after = _operational_snapshot(
        observed_at=start + timedelta(seconds=10),
        actionable=[],
        holds=[hold],
    )

    evidence = _completion_evidence(
        connection,
        event=event,
        before=before,
        after=after,
    )

    assert evidence["backlog"]["holds"] == [hold]
    assert evidence["velocity"]["arrival_count"] == 0
    assert evidence["lag"]["oldest_post_watermark_eligible_seconds"] == 0
    connection.close()


def test_campaign_completion_accepts_retained_pre_frontier_queued_hold(
    tmp_path: Path,
) -> None:
    from scripts import hermes_graphiti_worker as worker

    connection, start, event, before, _after = _campaign_completion_fixture(
        tmp_path,
        with_pre_frontier_hold=True,
    )
    hold = {
        "ledger_seq": 6,
        "event_id": "sha256:" + f"{6:064x}",
        "reason": worker.PRE_FRONTIER_BACKLOG_HOLD_REASON,
    }
    before = _operational_snapshot(
        observed_at=start,
        actionable=before["actionable"],
        holds=[hold],
    )
    after = _operational_snapshot(
        observed_at=start + timedelta(seconds=10),
        actionable=[],
        holds=[hold],
    )

    evidence = _completion_evidence(
        connection,
        event=event,
        before=before,
        after=after,
    )

    states = connection.execute(
        "SELECT ledger_seq,state FROM unpublished_graphiti_revision_events "
        "ORDER BY ledger_seq"
    ).fetchall()
    assert states == [(6, "QUEUED"), (7, "TERMINAL")]
    assert evidence["watermark"]["terminal_ledger_seq"] == 7
    assert evidence["backlog"]["remaining_actionable_at_watermark"] == []
    assert evidence["backlog"]["holds"] == [hold]
    connection.close()


@pytest.mark.parametrize("kind", ["PROJECT_EVENT_GAP", "UNCLASSIFIED_GAP"])
def test_campaign_completion_stops_on_actionable_event_gap(
    tmp_path: Path,
    kind: str,
) -> None:
    from scripts import hermes_graphiti_worker as worker

    connection, start, event, before, _after = _campaign_completion_fixture(
        tmp_path
    )
    after = _operational_snapshot(
        observed_at=start + timedelta(seconds=10),
        actionable=[
            {
                "ledger_seq": 8,
                "event_id": "sha256:" + f"{8:064x}",
                "landed_at": (start + timedelta(seconds=1))
                .isoformat()
                .replace("+00:00", "Z"),
                "kind": kind,
            }
        ],
    )
    with pytest.raises(
        worker.GraphitiCampaignStop,
        match="projectable or unclassified Graphiti event gap",
    ) as stopped:
        _completion_evidence(
            connection,
            event=event,
            before=before,
            after=after,
        )
    assert stopped.value.evidence is not None
    assert stopped.value.evidence["arrival_count"] == 1
    assert stopped.value.evidence["arrivals"] == after["actionable"]
    assert stopped.value.evidence["actionable_gaps"] == after["actionable"]
    connection.close()


def test_campaign_completion_requires_zero_actionable_backlog_at_watermark(
    tmp_path: Path,
) -> None:
    from scripts import hermes_graphiti_worker as worker

    connection, start, event, before, _after = _campaign_completion_fixture(
        tmp_path
    )
    old = _campaign_event("sha256:" + f"{6:064x}", ledger_seq=6)
    after = _operational_snapshot(
        observed_at=start + timedelta(seconds=10),
        actionable=[
            _fresh_actionable(old, landed_at=start - timedelta(seconds=6))
        ],
    )
    with pytest.raises(
        worker.GraphitiCampaignStop,
        match="backlog at watermark is non-zero",
    ):
        _completion_evidence(
            connection,
            event=event,
            before=before,
            after=after,
        )
    connection.close()


def test_campaign_completion_requires_zero_global_admission_backlog(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from scripts import hermes_graphiti_worker as worker

    connection, _start, event, before, after = _campaign_completion_fixture(
        tmp_path
    )
    connection.execute(
        "INSERT INTO unpublished_graphiti_admission_queue("
        "proposal_key,ingest_id,source_revision_id,source_receipt_digest,"
        "proposal_digest,proposal_kind,request_json,request_digest,state,"
        "created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
        (
            "proposal-pending",
            "ingest-pending",
            "revision-pending",
            "sha256:" + "1" * 64,
            "sha256:" + "2" * 64,
            "ENTITY_MENTION",
            "{}",
            "sha256:" + "3" * 64,
            "READY",
            "2026-09-01T12:00:01Z",
            "2026-09-01T12:00:01Z",
        ),
    )
    monkeypatch.setattr(
        worker,
        "_exact_admission_reconciliation",
        lambda _connection: {"total": True, "disjoint": True, "cohorts": []},
    )
    with pytest.raises(
        worker.GraphitiCampaignStop,
        match="admission backlog is non-zero",
    ):
        _completion_evidence(
            connection,
            event=event,
            before=before,
            after=after,
        )
    connection.close()


def test_zero_proposal_campaign_requires_no_generation(tmp_path: Path) -> None:
    connection, _start, event, before, after = _campaign_completion_fixture(
        tmp_path
    )
    evidence = _completion_evidence(
        connection,
        event=event,
        before=before,
        after=after,
    )

    assert evidence["reconciliation"]["new_generation_ids"] == []
    assert evidence["reconciliation"]["receipt"] is None
    connection.close()


def test_campaign_velocity_requires_positive_observation_window(
    tmp_path: Path,
) -> None:
    from scripts import hermes_graphiti_worker as worker

    connection, start, event, before, _after = _campaign_completion_fixture(
        tmp_path
    )
    after = _operational_snapshot(observed_at=start, actionable=[])
    with pytest.raises(
        worker.GraphitiCampaignStop,
        match="observation window differs",
    ):
        _completion_evidence(
            connection,
            event=event,
            before=before,
            after=after,
            elapsed_seconds=1.0,
        )
    connection.close()


def _campaign() -> dict[str, object]:
    return {
        "source_snapshot_digests": {
            "proving": "proving-snapshot",
            "unpublished": "unpublished-snapshot",
            "authority": "sha256:" + "a" * 64,
        },
        "cohort": {
            "events": [
                {
                    "event_id": "event-1",
                    "ledger_seq": 1,
                    "manifest_digest": "manifest-1",
                    "ingest_ids": ["ingest-1"],
                },
                {
                    "event_id": "event-2",
                    "ledger_seq": 2,
                    "manifest_digest": "manifest-2",
                    "ingest_ids": ["ingest-2"],
                },
            ]
        },
        "provider": {
            "provider_id": "provider",
            "transport_id": "CURSOR_SDK",
            "model_id": "model",
            "embedding_provider_id": "embedding-provider",
            "embedding_model_id": "embedding",
        },
        "graph": {
            "destination_id": GRAPH_DESTINATION_ID,
            "family_id": "graph.increment4.admitted",
        },
        "graph_destination_readback": {"generation_id": "generation"},
        "caps": {
            "per_event": {
                "proposals": 2,
                "entity_admits": 1,
                "relation_admits": 1,
                "effects": 2,
                "retries": 0,
                "fallbacks": 0,
            },
            "total": {
                "events": 2,
                "proposals": 2,
                "entity_admits": 1,
                "relation_admits": 1,
                "effects": 2,
                "retries": 0,
                "fallbacks": 0,
                "wall_time_seconds": 120,
                "spend_gbp_microunits": 1_000_000,
            },
            "rate": {"events_per_minute": 60},
        },
        "ramp": {
            "phases": [
                {
                    "phase_id": "one",
                    "event_limit": 1,
                    "entry_conditions": [
                        "EXACT_SNAPSHOT_AND_IDENTITY_RECONFIRMED",
                        "OWNER_F4_GO_RETAINED",
                    ],
                    "advance_conditions": [
                        "ALL_EXACT_RECEIPTS_RECONCILED",
                        "CAPS_AND_ACCOUNTING_RECONCILED",
                        "NO_STOP_CONDITION_OBSERVED",
                    ],
                },
                {
                    "phase_id": "two",
                    "event_limit": 2,
                    "entry_conditions": [
                        "EXACT_SNAPSHOT_AND_IDENTITY_RECONFIRMED",
                        "OWNER_F4_GO_RETAINED",
                    ],
                    "advance_conditions": [
                        "ALL_EXACT_RECEIPTS_RECONCILED",
                        "CAPS_AND_ACCOUNTING_RECONCILED",
                        "NO_STOP_CONDITION_OBSERVED",
                    ],
                },
            ]
        },
        "success_objectives": {
            "watermark": "selected cohort terminal",
            "backlog": 0,
            "velocity": "service_at_least_arrival",
            "lag": {"max_oldest_eligible_seconds": 300},
            "reconciliation": "exact",
        },
    }


def _campaign_resolved_units(campaign: Mapping[str, object]):
    cohort = campaign["cohort"]
    assert isinstance(cohort, Mapping)
    events = cohort["events"]
    assert isinstance(events, list)
    return tuple(
        SimpleNamespace(ingest_id=ingest_id)
        for event in events
        for ingest_id in event["ingest_ids"]
    )


@pytest.mark.parametrize(
    (
        "extra_fresh_candidate",
        "pre_frontier_hold",
        "processing_seconds",
        "expected_sleeps",
        "expected_dispatch_budgets",
        "expected_partition_watermarks",
    ),
    [
        (False, False, 0.25, [0.75], [120.0, 119.0], [None, None]),
        (False, False, 1.25, [], [120.0, 118.75], [None, None]),
        (True, False, 0.0, [], [], [None]),
        (False, True, 0.0, [1.0], [120.0, 119.0], [None, 3]),
    ],
    ids=[
        "fast-events",
        "slow-events",
        "fresh-candidate-race",
        "retained-pre-frontier-hold",
    ],
)
def test_bounded_campaign_requires_exact_candidate_snapshot_before_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    extra_fresh_candidate: bool,
    pre_frontier_hold: bool,
    processing_seconds: float,
    expected_sleeps: list[float],
    expected_dispatch_budgets: list[float],
    expected_partition_watermarks: list[int | None],
) -> None:
    from scripts import hermes_graphiti_worker as worker

    campaign = _campaign()
    if pre_frontier_hold:
        for event in campaign["cohort"]["events"]:
            event["ledger_seq"] += 1
    partition_events = list(campaign["cohort"]["events"])
    if extra_fresh_candidate:
        partition_events.append(
            {
                "event_id": "event-3",
                "ledger_seq": 3,
                "manifest_digest": "manifest-3",
                "ingest_ids": ["ingest-3"],
            }
        )
    packet = {
        "packet_digest": "sha256:" + "a" * 64,
        "code_identity": {"head_sha": "head", "tree_sha": "tree"},
        "store_snapshots": {
            "authority": {
                "source_path": "/authority.sqlite3",
                "descriptor_digest": "sha256:" + "a" * 64,
            }
        },
    }
    calls: list[tuple[str, object]] = []
    dispatch_budgets: list[float] = []
    partition_watermarks: list[int | None] = []
    corpus_loads = 0

    class FakeMonotonic:
        def __init__(self) -> None:
            self.value = 0.0
            self.sleeps: list[float] = []

        def __call__(self) -> float:
            return self.value

        def advance(self, seconds: float) -> None:
            self.value += seconds

        def sleep(self, seconds: float) -> None:
            self.sleeps.append(seconds)
            self.advance(seconds)

    monotonic = FakeMonotonic()
    monkeypatch.setattr(
        worker,
        "validate_graphiti_campaign_packet",
        lambda value: campaign if value is packet else pytest.fail("wrong packet"),
    )

    def qualify(**kwargs: object) -> dict[str, object]:
        event_id = str(kwargs["event_id"])
        suffix = event_id.removeprefix("event-")
        campaign_event = next(
            event
            for event in campaign["cohort"]["events"]
            if event["event_id"] == event_id
        )
        return {
            "event_id": event_id,
            "ledger_seq": campaign_event["ledger_seq"],
            "event_manifest_digest": f"manifest-{suffix}",
            "resolved_units": [{"ingest_id": f"ingest-{suffix}"}],
        }

    def load_corpus(**_kwargs: object):
        nonlocal corpus_loads
        corpus_loads += 1
        return _campaign_resolved_units(campaign)

    def consume(**kwargs: object) -> GraphitiProcessResult:
        event_id = str(kwargs["event_id"])
        campaign_event = next(
            event
            for event in campaign["cohort"]["events"]
            if event["event_id"] == event_id
        )
        calls.append(("extract", event_id))
        dispatch_budgets.append(float(kwargs["max_dispatch_seconds"]))
        assert kwargs["defer_graphiti_admission"] is True
        assert kwargs["require_graphiti_admission"] is True
        assert kwargs["unit_authority_resolver"] is runtime.bind_unit_authority
        assert [
            unit.ingest_id for unit in kwargs["prepared_event_units"]
        ] == [f"ingest-{event_id.removeprefix('event-')}"]
        monotonic.advance(processing_seconds)
        return GraphitiProcessResult(
            event_id,
            int(campaign_event["ledger_seq"]),
            "TERMINAL",
            1,
        )

    monkeypatch.setattr(worker, "qualify_fresh_graphiti_event", qualify)
    monkeypatch.setattr(worker, "load_graphiti_units", load_corpus)
    monkeypatch.setattr(
        worker,
        "qualify_campaign_adapter_runtime",
        lambda: None,
    )
    monkeypatch.setattr(
        worker,
        "graphiti_store_snapshot_digests",
        lambda **_kwargs: campaign["source_snapshot_digests"],
    )
    monkeypatch.setattr(
        worker,
        "_assert_fresh_campaign_ingests",
        lambda *_args, **_kwargs: None,
    )

    def operational_partition(**kwargs: object) -> dict[str, object]:
        watermark = kwargs.get("pre_frontier_hold_watermark")
        assert watermark is None or isinstance(watermark, int)
        if watermark is not None:
            assert [item for item in calls if item[0] == "extract"] == [
                ("extract", "event-1"),
                ("extract", "event-2"),
            ]
        partition_watermarks.append(watermark)
        holds = []
        if len(partition_watermarks) == 1 and pre_frontier_hold:
            holds.append(
                {
                    "ledger_seq": 1,
                    "event_id": "held-event",
                    "reason": worker.PRE_FRONTIER_BACKLOG_HOLD_REASON,
                }
            )
        return _operational_snapshot(
            observed_at=datetime(2026, 9, 1, 12, 0, tzinfo=UTC),
            actionable=[
                _fresh_actionable(
                    event,
                    landed_at=datetime(2026, 9, 1, 11, 59, tzinfo=UTC),
                )
                for event in partition_events
            ],
            holds=holds,
        )

    monkeypatch.setattr(
        worker,
        "_campaign_operational_partition_snapshot",
        operational_partition,
    )
    monkeypatch.setattr(worker, "consume_next_graphiti_event", consume)
    monkeypatch.setattr(
        worker,
        "_campaign_receipt_evidence",
        lambda *_args, **_kwargs: {
            "proposal_count": 1,
            "chat_invocation_count": 1,
            "embedding_request_count": 1,
            "fallback_count": 0,
            "retry_count": 0,
            "actual_gbp_microunits": 1,
        },
    )
    before = (
        ("ingest-1", "ENTITY_MENTION", None),
        ("ingest-2", "RELATION", None),
    )
    after = (
        ("ingest-1", "ENTITY_MENTION", "ADMIT"),
        ("ingest-2", "RELATION", "ADMIT"),
    )
    admission_reads = iter((before, after))
    monkeypatch.setattr(
        worker,
        "_campaign_admission_rows",
        lambda *_args, **_kwargs: next(admission_reads),
    )
    monkeypatch.setattr(
        worker,
        "_campaign_reconciliation_ids",
        lambda _connection: frozenset(),
    )
    monkeypatch.setattr(
        worker,
        "graphiti_decided_cohort_generation_identity",
        lambda *_args, **_kwargs: (
            "sha256:" + "7" * 64,
            "00000000-0000-4000-8000-000000000895",
        ),
    )
    objective_evidence = {
        "watermark": {"passed": True},
        "backlog": {"passed": True},
        "velocity": {"passed": True},
        "lag": {"passed": True},
        "reconciliation": {"passed": True},
    }
    monkeypatch.setattr(
        worker,
        "_campaign_completion_evidence",
        lambda *_args, **_kwargs: objective_evidence,
    )
    connection = sqlite3.connect(":memory:")
    monkeypatch.setattr(worker, "connect", lambda _path: connection)

    class Admission:
        def enqueue_complete_receipts(self, *, ingest_ids):
            calls.append(("enqueue", ingest_ids))

        def drain(self, **kwargs: object):
            assert [item for item in calls if item[0] == "extract"] == [
                ("extract", "event-1"),
                ("extract", "event-2"),
            ]
            assert kwargs["stop_on_failure"] is True
            calls.append(("drain", kwargs["ingest_ids"]))
            return SimpleNamespace(failed=0, dead_lettered=0)

        def finalise_decided_cohort(self, *, ingest_ids):
            calls.append(("generation", ingest_ids))
            return SimpleNamespace(failed=0, dead_lettered=0, projected=2)

    runtime = worker._mint_graphiti_campaign_runtime(
        graphiti=object(),
        admission_factory=lambda _connection: Admission(),
        bind_unit_authority=lambda unit: unit,
        graph_state_fence=lambda _campaign: (
            calls.append(("gate", "graph")) or {}
        ),
        graph_destination_id=GRAPH_DESTINATION_ID,
        authority_store_source_path="/authority.sqlite3",
        authority_store_descriptor_digest="sha256:" + "a" * 64,
    )

    def owner_fence(value: object) -> None:
        assert value is packet
        calls.append(("gate", "owner"))

    arguments = {
        "packet": packet,
        "proving_store": "proving",
        "unpublished_store": "unpublished",
        "runtime": runtime,
        "head_sha": "head",
        "tree_sha": "tree",
        "owner_f4_fence": owner_fence,
        "monotonic": monotonic,
        "sleep": monotonic.sleep,
    }
    if extra_fresh_candidate:
        with pytest.raises(
            worker.GraphitiCampaignStop,
            match="snapshot or candidate identity drifted",
        ) as stopped:
            worker.run_bounded_campaign(**arguments)
        assert calls == [("gate", "owner"), ("gate", "graph")]
        assert stopped.value.evidence is not None
        assert stopped.value.evidence["operational_candidates"][-1] == {
            "ledger_seq": 3,
            "event_id": "event-3",
            "manifest_digest": "manifest-3",
            "ingest_ids": ["ingest-3"],
        }
        assert monotonic.sleeps == expected_sleeps
        assert dispatch_budgets == expected_dispatch_budgets
        assert partition_watermarks == expected_partition_watermarks
        assert corpus_loads == 1
        connection.close()
        return

    report = worker.run_bounded_campaign(**arguments)

    assert calls == [
        ("gate", "owner"),
        ("gate", "graph"),
        ("extract", "event-1"),
        ("gate", "owner"),
        ("gate", "graph"),
        ("extract", "event-2"),
        ("gate", "owner"),
        ("gate", "graph"),
        ("enqueue", ("ingest-1", "ingest-2")),
        ("drain", ("ingest-1", "ingest-2")),
        ("generation", ("ingest-1", "ingest-2")),
    ]
    assert report["state"] == "CAMPAIGN_COMPLETE"
    assert report["event_count"] == 2
    assert report["entity_admits"] == 1
    assert report["relation_admits"] == 1
    assert report["success_objectives"] == objective_evidence
    assert monotonic.sleeps == expected_sleeps
    assert dispatch_budgets == expected_dispatch_budgets
    assert partition_watermarks == expected_partition_watermarks
    assert corpus_loads == 1
    assert [item["state"] for item in report["events"]] == [
        "EXTRACTION_TERMINAL_CAMPAIGN_PENDING",
        "EXTRACTION_TERMINAL_CAMPAIGN_PENDING",
    ]


def test_bounded_campaign_checks_owner_f4_before_graph_readback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import hermes_graphiti_worker as worker

    campaign = _campaign()
    packet = {
        "packet_digest": "sha256:" + "a" * 64,
        "code_identity": {"head_sha": "head", "tree_sha": "tree"},
        "store_snapshots": {
            "authority": {
                "source_path": "/authority.sqlite3",
                "descriptor_digest": "sha256:" + "a" * 64,
            }
        },
    }
    monkeypatch.setattr(
        worker,
        "validate_graphiti_campaign_packet",
        lambda value: campaign if value is packet else pytest.fail("wrong packet"),
    )
    monkeypatch.setattr(
        worker,
        "graphiti_store_snapshot_digests",
        lambda **_kwargs: campaign["source_snapshot_digests"],
    )
    monkeypatch.setattr(
        worker,
        "_assert_fresh_campaign_ingests",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        worker,
        "qualify_fresh_graphiti_event",
        lambda **kwargs: {
            "event_manifest_digest": (
                f"manifest-{str(kwargs['event_id']).removeprefix('event-')}"
            ),
            "resolved_units": [
                {
                    "ingest_id": (
                        f"ingest-{str(kwargs['event_id']).removeprefix('event-')}"
                    )
                }
            ],
        },
    )
    monkeypatch.setattr(worker, "qualify_campaign_adapter_runtime", lambda: None)
    monkeypatch.setattr(
        worker,
        "load_graphiti_units",
        lambda **_kwargs: _campaign_resolved_units(campaign),
    )
    monkeypatch.setattr(
        worker,
        "consume_next_graphiti_event",
        lambda **_kwargs: pytest.fail("dispatch reached after F4 stop"),
    )
    runtime = worker._mint_graphiti_campaign_runtime(
        graphiti=object(),
        admission_factory=lambda _connection: object(),
        bind_unit_authority=lambda unit: unit,
        graph_state_fence=lambda _campaign: pytest.fail(
            "graph readback reached before F4"
        ),
        graph_destination_id=GRAPH_DESTINATION_ID,
        authority_store_source_path="/authority.sqlite3",
        authority_store_descriptor_digest="sha256:" + "a" * 64,
    )

    def stopped(_packet: Mapping[str, object]) -> None:
        raise worker.GraphitiCampaignStop("owner F4 stopped")

    with pytest.raises(worker.GraphitiCampaignStop, match="owner F4 stopped"):
        worker.run_bounded_campaign(
            packet=packet,
            proving_store="proving",
            unpublished_store="unpublished",
            runtime=runtime,
            head_sha="head",
            tree_sha="tree",
            owner_f4_fence=stopped,
            monotonic=lambda: 0.0,
            sleep=lambda _delay: None,
        )


def test_campaign_cap_stops_before_any_canonical_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import hermes_graphiti_worker as worker

    campaign = _campaign()
    campaign["caps"]["per_event"]["entity_admits"] = 0
    packet = {
        "packet_digest": "sha256:" + "a" * 64,
        "code_identity": {"head_sha": "head", "tree_sha": "tree"},
        "store_snapshots": {
            "authority": {
                "source_path": "/authority.sqlite3",
                "descriptor_digest": "sha256:" + "a" * 64,
            }
        },
    }
    monkeypatch.setattr(
        worker, "validate_graphiti_campaign_packet", lambda _value: campaign
    )
    monkeypatch.setattr(
        worker,
        "graphiti_store_snapshot_digests",
        lambda **_kwargs: campaign["source_snapshot_digests"],
    )
    monkeypatch.setattr(
        worker,
        "_assert_fresh_campaign_ingests",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        worker,
        "_campaign_operational_partition_snapshot",
        lambda **_kwargs: _operational_snapshot(
            observed_at=datetime(2026, 9, 1, 12, 0, tzinfo=UTC),
            actionable=[
                _fresh_actionable(
                    event,
                    landed_at=datetime(2026, 9, 1, 11, 59, tzinfo=UTC),
                )
                for event in campaign["cohort"]["events"]
            ],
        ),
    )
    monkeypatch.setattr(
        worker,
        "qualify_fresh_graphiti_event",
        lambda **kwargs: {
            "event_manifest_digest": (
                f"manifest-{str(kwargs['event_id']).removeprefix('event-')}"
            ),
            "resolved_units": [
                {
                    "ingest_id": (
                        f"ingest-{str(kwargs['event_id']).removeprefix('event-')}"
                    )
                }
            ],
        },
    )
    monkeypatch.setattr(worker, "qualify_campaign_adapter_runtime", lambda: None)
    monkeypatch.setattr(
        worker,
        "load_graphiti_units",
        lambda **_kwargs: _campaign_resolved_units(campaign),
    )
    monkeypatch.setattr(
        worker,
        "consume_next_graphiti_event",
        lambda **kwargs: GraphitiProcessResult(
            str(kwargs["event_id"]),
            int(str(kwargs["event_id"]).removeprefix("event-")),
            "TERMINAL",
            1,
        ),
    )
    monkeypatch.setattr(
        worker,
        "_campaign_receipt_evidence",
        lambda *_args, **_kwargs: {
            "proposal_count": 1,
            "chat_invocation_count": 1,
            "embedding_request_count": 0,
            "fallback_count": 0,
            "retry_count": 0,
            "actual_gbp_microunits": 0,
        },
    )
    monkeypatch.setattr(
        worker,
        "_campaign_admission_rows",
        lambda *_args, **_kwargs: (
            ("ingest-1", "ENTITY_MENTION", None),
            ("ingest-2", "RELATION", None),
        ),
    )
    monkeypatch.setattr(
        worker,
        "_campaign_reconciliation_ids",
        lambda _connection: frozenset(),
    )
    connection = sqlite3.connect(":memory:")
    monkeypatch.setattr(worker, "connect", lambda _path: connection)

    class Admission:
        def enqueue_complete_receipts(self, *, ingest_ids):
            assert ingest_ids == ("ingest-1", "ingest-2")

        def drain(self, **_kwargs: object):
            pytest.fail("canonical admission occurred after cap stop")

    runtime = worker._mint_graphiti_campaign_runtime(
        graphiti=object(),
        admission_factory=lambda _connection: Admission(),
        bind_unit_authority=lambda unit: unit,
        graph_state_fence=lambda _campaign: {},
        graph_destination_id=GRAPH_DESTINATION_ID,
        authority_store_source_path="/authority.sqlite3",
        authority_store_descriptor_digest="sha256:" + "a" * 64,
    )
    with pytest.raises(worker.GraphitiCampaignStop, match="entity_admits cap"):
        worker.run_bounded_campaign(
            packet=packet,
            proving_store="proving",
            unpublished_store="unpublished",
            runtime=runtime,
            head_sha="head",
            tree_sha="tree",
            owner_f4_fence=lambda _packet: None,
            monotonic=lambda: 0.0,
            sleep=lambda _delay: None,
        )


def test_qualify_campaign_adapter_runtime_refuses_missing_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import newsroom.graphiti_adapter.real as real
    from newsroom.graphiti_adapter.types import GraphitiAdapterContractError
    from scripts import hermes_graphiti_worker as worker

    monkeypatch.setattr(
        real,
        "_load_graphiti",
        lambda: (_ for _ in ()).throw(
            GraphitiAdapterContractError(
                "graphiti extra (graphiti-core 0.29.3) is required "
                "for real Graphiti execution",
                reason_code="GRAPHITI_EXTRA_REQUIRED",
            )
        ),
    )

    with pytest.raises(
        worker.GraphitiCampaignStop,
        match="adapter runtime is unavailable",
    ) as stopped:
        worker.qualify_campaign_adapter_runtime()

    assert stopped.value.evidence == {
        "stage": "PRE_DISPATCH_ADAPTER_RUNTIME",
        "failure_type": "GraphitiAdapterContractError",
        "setup_failure_detail": "GRAPHITI_EXTRA_REQUIRED",
        "provider_dispatched": False,
    }


def test_bounded_campaign_refuses_missing_adapter_runtime_before_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import newsroom.graphiti_adapter.real as real
    from newsroom.graphiti_adapter.types import GraphitiAdapterContractError
    from scripts import hermes_graphiti_worker as worker

    campaign = _campaign()
    packet = {
        "packet_digest": "sha256:" + "a" * 64,
        "code_identity": {"head_sha": "head", "tree_sha": "tree"},
        "store_snapshots": {
            "authority": {
                "source_path": "/authority.sqlite3",
                "descriptor_digest": "sha256:" + "a" * 64,
            }
        },
    }
    monkeypatch.setattr(
        real,
        "_load_graphiti",
        lambda: (_ for _ in ()).throw(
            GraphitiAdapterContractError(
                "graphiti extra (graphiti-core 0.29.3) is required "
                "for real Graphiti execution",
                reason_code="GRAPHITI_EXTRA_REQUIRED",
            )
        ),
    )
    monkeypatch.setattr(
        worker, "validate_graphiti_campaign_packet", lambda _value: campaign
    )
    monkeypatch.setattr(
        worker,
        "graphiti_store_snapshot_digests",
        lambda **_kwargs: campaign["source_snapshot_digests"],
    )
    monkeypatch.setattr(
        worker,
        "load_graphiti_units",
        lambda **_kwargs: pytest.fail("corpus loaded after adapter runtime refusal"),
    )
    monkeypatch.setattr(
        worker,
        "qualify_fresh_graphiti_event",
        lambda **_kwargs: pytest.fail("event preflight after adapter runtime refusal"),
    )
    monkeypatch.setattr(
        worker,
        "consume_next_graphiti_event",
        lambda **_kwargs: pytest.fail("event claimed after adapter runtime refusal"),
    )
    runtime = worker._mint_graphiti_campaign_runtime(
        graphiti=object(),
        admission_factory=lambda _connection: object(),
        bind_unit_authority=lambda unit: unit,
        graph_state_fence=lambda _campaign: pytest.fail(
            "graph fence after adapter runtime refusal"
        ),
        graph_destination_id=GRAPH_DESTINATION_ID,
        authority_store_source_path="/authority.sqlite3",
        authority_store_descriptor_digest="sha256:" + "a" * 64,
    )

    with pytest.raises(
        worker.GraphitiCampaignStop,
        match="adapter runtime is unavailable",
    ) as stopped:
        worker.run_bounded_campaign(
            packet=packet,
            proving_store="proving",
            unpublished_store="unpublished",
            runtime=runtime,
            head_sha="head",
            tree_sha="tree",
            owner_f4_fence=lambda _packet: pytest.fail(
                "owner F4 fence after adapter runtime refusal"
            ),
            monotonic=lambda: 0.0,
            sleep=lambda _delay: None,
        )

    assert stopped.value.evidence == {
        "stage": "PRE_DISPATCH_ADAPTER_RUNTIME",
        "failure_type": "GraphitiAdapterContractError",
        "setup_failure_detail": "GRAPHITI_EXTRA_REQUIRED",
        "provider_dispatched": False,
    }


def test_bounded_campaign_stops_on_pre_dispatch_configuration_hold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import hermes_graphiti_worker as worker

    campaign = _campaign()
    packet = {
        "packet_digest": "sha256:" + "a" * 64,
        "code_identity": {"head_sha": "head", "tree_sha": "tree"},
        "store_snapshots": {
            "authority": {
                "source_path": "/authority.sqlite3",
                "descriptor_digest": "sha256:" + "a" * 64,
            }
        },
    }
    claimed: list[str] = []
    monkeypatch.setattr(
        worker, "validate_graphiti_campaign_packet", lambda _value: campaign
    )
    monkeypatch.setattr(
        worker,
        "graphiti_store_snapshot_digests",
        lambda **_kwargs: campaign["source_snapshot_digests"],
    )
    monkeypatch.setattr(
        worker,
        "_assert_fresh_campaign_ingests",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        worker,
        "_campaign_operational_partition_snapshot",
        lambda **_kwargs: _operational_snapshot(
            observed_at=datetime(2026, 9, 1, 12, 0, tzinfo=UTC),
            actionable=[
                _fresh_actionable(
                    event,
                    landed_at=datetime(2026, 9, 1, 11, 59, tzinfo=UTC),
                )
                for event in campaign["cohort"]["events"]
            ],
        ),
    )
    monkeypatch.setattr(worker, "qualify_campaign_adapter_runtime", lambda: None)
    monkeypatch.setattr(
        worker,
        "qualify_fresh_graphiti_event",
        lambda **kwargs: {
            "event_manifest_digest": (
                f"manifest-{str(kwargs['event_id']).removeprefix('event-')}"
            ),
            "resolved_units": [
                {
                    "ingest_id": (
                        f"ingest-{str(kwargs['event_id']).removeprefix('event-')}"
                    )
                }
            ],
        },
    )
    monkeypatch.setattr(
        worker,
        "load_graphiti_units",
        lambda **_kwargs: _campaign_resolved_units(campaign),
    )

    def consume(**kwargs: object) -> GraphitiProcessResult:
        event_id = str(kwargs["event_id"])
        claimed.append(event_id)
        return GraphitiProcessResult(event_id, 1, "CONFIGURATION_HELD", 1)

    monkeypatch.setattr(worker, "consume_next_graphiti_event", consume)
    monkeypatch.setattr(
        worker,
        "_campaign_receipt_evidence",
        lambda *_args, **_kwargs: pytest.fail(
            "terminal receipt required after configuration hold"
        ),
    )
    runtime = worker._mint_graphiti_campaign_runtime(
        graphiti=object(),
        admission_factory=lambda _connection: object(),
        bind_unit_authority=lambda unit: unit,
        graph_state_fence=lambda _campaign: {},
        graph_destination_id=GRAPH_DESTINATION_ID,
        authority_store_source_path="/authority.sqlite3",
        authority_store_descriptor_digest="sha256:" + "a" * 64,
    )

    with pytest.raises(
        worker.GraphitiCampaignStop,
        match="pre-dispatch configuration refusal",
    ):
        worker.run_bounded_campaign(
            packet=packet,
            proving_store="proving",
            unpublished_store="unpublished",
            runtime=runtime,
            head_sha="head",
            tree_sha="tree",
            owner_f4_fence=lambda _packet: None,
            monotonic=lambda: 0.0,
            sleep=lambda _delay: None,
        )

    assert claimed == ["event-1"]
