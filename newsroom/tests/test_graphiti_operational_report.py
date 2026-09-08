from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import pytest

from scripts import graphiti_steady_state_report


@pytest.mark.parametrize(
    "failure",
    [None, "consumer", "identity", "validation", "evaluator", "preparation", "no_go"],
)
def test_operational_packet_composes_existing_contracts_once_without_live_io(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    failure: str | None,
) -> None:
    events: list[str] = []
    proving = object()
    unpublished = object()
    proof = object()

    class Plan:
        plan_digest = "sha256:" + "3" * 64
        cohort_digest = "sha256:" + "4" * 64

    plan = Plan()
    binder = object()
    reconciliation = {"status": "RECONCILED"}
    runtime = object()
    graph_destination_id = "sha256:" + "d" * 64
    authority_descriptor = "sha256:" + "a" * 64

    class AuthorityConnection:
        row_factory = None

        def close(self) -> None:
            events.append("authority-close")

    class System:
        def __init__(self, destination_id: str) -> None:
            self.graph_destination_id = destination_id

        def close(self) -> None:
            events.append("system-close")

    class Bootstrap:
        candidate_event_count = 3

        def canonical_value(self) -> dict[str, object]:
            return {"unit_count": 3, "provider_calls": 0}

    system = System(graph_destination_id)
    bootstrap = Bootstrap()

    @contextmanager
    def locked_inputs():
        events.append("lock-enter")
        try:
            yield proving, unpublished
        finally:
            events.append("lock-exit")

    monkeypatch.setattr(
        graphiti_steady_state_report,
        "require_canonical_proving_store",
        lambda _path: events.append("require-proving"),
    )
    monkeypatch.setattr(
        graphiti_steady_state_report,
        "require_canonical_unpublished_store",
        lambda _path: events.append("require-unpublished"),
    )
    monkeypatch.setattr(
        graphiti_steady_state_report,
        "_locked_operational_inputs",
        locked_inputs,
    )
    monkeypatch.setattr(
        graphiti_steady_state_report,
        "_authority_backup_identity",
        lambda output_dir: events.append("backup") or "sha256:" + "b" * 64,
    )
    monkeypatch.setattr(
        graphiti_steady_state_report.secrets,
        "token_urlsafe",
        lambda _length: "ephemeral-credential",
    )

    def open_system(*, credential: str):
        assert credential == "ephemeral-credential"
        events.append("open-system")
        return system, proof

    monkeypatch.setattr(
        graphiti_steady_state_report,
        "open_operational_graphiti_authority_system",
        open_system,
    )
    monkeypatch.setattr(
        graphiti_steady_state_report.sqlite3,
        "connect",
        lambda _path: events.append("authority-connect") or AuthorityConnection(),
    )
    monkeypatch.setattr(
        graphiti_steady_state_report,
        "apply_control_plane_sqlite_profile",
        lambda _connection, **_kwargs: events.append("authority-profile"),
    )

    def plan_bootstrap(
        supplied_proving,
        supplied_unpublished,
        _authority,
        *,
        observed_at,
    ):
        assert (supplied_proving, supplied_unpublished) == (proving, unpublished)
        assert observed_at == datetime(2026, 9, 2, 12, tzinfo=UTC)
        events.append("plan")
        return plan

    monkeypatch.setattr(
        graphiti_steady_state_report,
        "plan_operational_authority_bootstrap",
        plan_bootstrap,
    )

    def apply_bootstrap(supplied_system, *, proof: object, plan: object):
        assert supplied_system is system
        assert proof is globals_proof
        assert plan is globals_plan
        events.append("bootstrap")
        return bootstrap, binder

    globals_proof = proof
    globals_plan = plan
    monkeypatch.setattr(
        graphiti_steady_state_report,
        "bootstrap_operational_authority",
        apply_bootstrap,
    )

    def reconcile(
        supplied_system,
        *,
        proof: object,
        plan: object,
        bootstrap: object,
    ):
        assert supplied_system is system
        assert proof is globals_proof
        assert plan is globals_plan
        assert bootstrap is globals_bootstrap
        events.append("reconcile")
        return reconciliation

    monkeypatch.setattr(
        graphiti_steady_state_report,
        "build_and_reconcile_operational_generation",
        reconcile,
    )

    def descriptors(**paths: object) -> dict[str, str]:
        assert set(paths) == {
            "proving_store",
            "unpublished_store",
            "authority_store",
        }
        events.append("descriptor")
        return {
            "proving": "sha256:" + "1" * 64,
            "unpublished": "sha256:" + "2" * 64,
            "authority": authority_descriptor,
        }

    monkeypatch.setattr(
        graphiti_steady_state_report,
        "graphiti_store_snapshot_digests",
        descriptors,
    )

    def compose_runtime(**kwargs: object):
        assert kwargs["authority_system"] is system
        assert kwargs["authority_store_descriptor_digest"] == authority_descriptor
        assert kwargs["proof"] is proof
        assert kwargs["bind_unit_authority"] is binder
        events.append("runtime")
        return runtime

    monkeypatch.setattr(
        graphiti_steady_state_report,
        "compose_governed_graphiti_worker_runtime",
        compose_runtime,
    )

    graph_readback = {"destination_id": graph_destination_id, "exact": True}

    def readback(*, destination_id: str, reconciliation: object):
        assert destination_id == graph_destination_id
        assert reconciliation is globals_reconciliation
        events.append("readback")
        return graph_readback

    globals_reconciliation = reconciliation
    globals_bootstrap = bootstrap
    monkeypatch.setattr(
        graphiti_steady_state_report,
        "graphiti_graph_destination_readback",
        readback,
    )

    campaign = {"campaign_authorised": False}

    def campaign_input(**kwargs: object):
        assert kwargs["graph_destination_id"] == graph_destination_id
        assert kwargs["candidate_event_count"] == 3
        assert str(kwargs["recovery_identity"]).startswith("sha256:")
        events.append("campaign")
        return campaign

    monkeypatch.setattr(
        graphiti_steady_state_report,
        "build_operational_campaign_input",
        campaign_input,
    )

    calls = 0
    packet = {"verdict": "READY_FOR_OWNER_DECISION"}

    def evaluate(**kwargs: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        assert kwargs["campaign_input"] == {"campaign_authorised": False}
        assert kwargs["graph_destination_reconciliation"] is reconciliation
        assert kwargs["governed_runtime"] is runtime
        events.append("evaluator")
        return packet

    monkeypatch.setattr(
        graphiti_steady_state_report,
        "build_graphiti_steady_state_packet",
        evaluate,
    )
    monkeypatch.setattr(
        graphiti_steady_state_report,
        "validate_graphiti_campaign_packet",
        lambda _supplied: events.append("validate"),
    )

    def code_identity():
        events.append("identity")
        return ("changed", "tree") if failure == "identity" else ("h" * 40, "t" * 40)

    def fail(*_args, **_kwargs):
        raise RuntimeError("injected failure")

    monkeypatch.setattr(graphiti_steady_state_report, "_exact_main_identity", code_identity)
    if failure in {"validation", "evaluator", "preparation"}:
        symbol = {
            "validation": "validate_graphiti_campaign_packet",
            "evaluator": "build_graphiti_steady_state_packet",
            "preparation": "plan_operational_authority_bootstrap",
        }[failure]
        monkeypatch.setattr(graphiti_steady_state_report, symbol, fail)
    elif failure == "no_go":
        packet["verdict"] = "NO_GO"

    yielded = False
    try:
        with graphiti_steady_state_report.sealed_operational_campaign_runtime(
            head_sha="h" * 40,
            tree_sha="t" * 40,
            focus_manifest_digest="sha256:" + "f" * 64,
            output_dir=tmp_path,
            observed_at=datetime(2026, 9, 2, 12, tzinfo=UTC),
        ) as (result, supplied_runtime):
            yielded = True
            assert "lock-exit" in events
            assert "system-close" not in events
            assert events[-1] == "identity"
            if failure in {"validation", "evaluator", "preparation", "no_go"}:
                assert result["verdict"] == "NO_GO"
                assert supplied_runtime is None
            else:
                assert supplied_runtime is runtime
            events.append("consumer")
            if failure == "consumer":
                raise RuntimeError("consumer failure")
    except RuntimeError as error:
        assert failure in {"consumer", "identity"}
        assert str(error) == (
            "consumer failure" if failure == "consumer"
            else "code identity changed while building steady-state evidence"
        )
    assert yielded is (failure != "identity")
    assert events.count("open-system") == 1
    assert events.count("system-close") == 1
    assert events[-1] == "system-close"
    assert events.index("lock-exit") < events.index("identity")
    if failure is not None:
        return

    assert result["verdict"] == "READY_FOR_OWNER_DECISION"
    assert result["non_effects_scope"] == "READ_ONLY_EVALUATOR_ONLY"
    assert result["operational_reconciliation"]["status"] == "COMPLETE"
    stderr = capsys.readouterr().err
    assert "operational_stage\tBACKUP\tbegin" in stderr
    assert "operational_stage\tBACKUP\telapsed_s=" in stderr
    assert "operational_stage\tRECOVERY_IDENTITY\telapsed_s=" in stderr
    assert "operational_stage\tREADINESS_EVALUATOR\tbegin" in stderr
    assert "operational_stage\tREADINESS_EVALUATOR\telapsed_s=" in stderr
    assert "operational_stage\tSEAL_OPERATIONAL_RESULT\telapsed_s=" in stderr
    assert "operational_stage\tREADY_PACKET_VALIDATION\telapsed_s=" in stderr
    assert "operational_stage\tTOTAL\telapsed_s=" in stderr
    assert "stage_timings" not in result["operational_reconciliation"]
    assert result["operational_reconciliation"]["provider_calls"] == 0
    assert calls == 1
    assert events == [
        "require-proving",
        "require-unpublished",
        "lock-enter",
        "backup",
        "open-system",
        "authority-connect",
        "authority-profile",
        "plan",
        "authority-close",
        "bootstrap",
        "reconcile",
        "descriptor",
        "runtime",
        "readback",
        "campaign",
        "evaluator",
        "validate",
        "lock-exit",
        "identity",
        "consumer",
        "system-close",
    ]


def test_operational_cli_rejects_incomplete_authority_shape_before_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        graphiti_steady_state_report,
        "_exact_main_identity",
        lambda: ("head", "tree"),
    )
    monkeypatch.setattr(
        graphiti_steady_state_report,
        "sealed_operational_campaign_runtime",
        lambda **_kwargs: pytest.fail("invalid operational shape was executed"),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "graphiti_steady_state_report.py",
            "--proving",
            "proving.sqlite3",
            "--unpublished",
            "unpublished.sqlite3",
            "--operational",
        ],
    )

    with pytest.raises(ValueError, match="requires output and Focus evidence only"):
        graphiti_steady_state_report.main()


def test_operational_preparation_failure_seals_no_go_without_known_failed_evaluator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    @contextmanager
    def locked_inputs():
        yield object(), object()

    monkeypatch.setattr(
        graphiti_steady_state_report,
        "require_canonical_proving_store",
        lambda _path: None,
    )
    monkeypatch.setattr(
        graphiti_steady_state_report,
        "require_canonical_unpublished_store",
        lambda _path: None,
    )
    monkeypatch.setattr(
        graphiti_steady_state_report,
        "_locked_operational_inputs",
        locked_inputs,
    )
    monkeypatch.setattr(
        graphiti_steady_state_report,
        "_authority_backup_identity",
        lambda _output: "sha256:" + "b" * 64,
    )
    monkeypatch.setattr(
        graphiti_steady_state_report,
        "open_operational_graphiti_authority_system",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("projector missing")),
    )
    monkeypatch.setattr(
        graphiti_steady_state_report,
        "build_graphiti_steady_state_packet",
        lambda **_kwargs: pytest.fail("known-incomplete evaluator was called"),
    )

    monkeypatch.setattr(
        graphiti_steady_state_report, "_exact_main_identity",
        lambda: ("h" * 40, "t" * 40),
    )
    with graphiti_steady_state_report.sealed_operational_campaign_runtime(
        head_sha="h" * 40,
        tree_sha="t" * 40,
        focus_manifest_digest="sha256:" + "f" * 64,
        output_dir=tmp_path,
        observed_at=datetime(2026, 9, 2, 12, tzinfo=UTC),
    ) as (result, runtime):
        assert runtime is None

    assert result["verdict"] == "NO_GO"
    assert result["operational_reconciliation"]["status"] == "FAILED"
    assert result["evaluator"] == {"attempted": False, "completed": False}
    assert result["blockers"] == ["OPERATIONAL_PREPARATION_FAILED"]


def test_authority_backup_identity_does_not_copy_the_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "authority.sqlite3"
    sqlite3.connect(source).close()
    monkeypatch.setattr(
        graphiti_steady_state_report,
        "CANONICAL_INCREMENT4_AUTHORITY_STORE",
        source,
    )
    output_dir = tmp_path / "out"
    digest = graphiti_steady_state_report._authority_backup_identity(output_dir)
    assert digest.startswith("sha256:")
    assert not (output_dir / "increment4-authority-pre-bootstrap.sqlite3").exists()
    assert graphiti_steady_state_report._authority_backup_identity(output_dir) == digest
    monkeypatch.setattr(
        graphiti_steady_state_report,
        "CANONICAL_INCREMENT4_AUTHORITY_STORE",
        tmp_path / "missing.sqlite3",
    )
    missing = graphiti_steady_state_report._authority_backup_identity(output_dir)
    assert missing != digest
    assert not (output_dir / "increment4-authority-pre-bootstrap.sqlite3").exists()
