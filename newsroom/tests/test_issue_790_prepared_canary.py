"""PreparedCanary is the unique pre-dispatch authority after #870."""

from __future__ import annotations

import ast
import inspect
import json
import os
import shlex
import sqlite3
import sys
from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from newsroom.authority.canonical import digest_canonical
from newsroom.control_plane import issue_790_disposition as disposition
from newsroom.control_plane.issue_790_disposition import (
    CANARY_BACKUP_DESTINATION_ALREADY_EXISTS,
    ISSUE_790_STEP16_PRE_DISPATCH_PATH,
    ISSUE_790_STEP22_PENDING_PLAN_PATH,
    Issue790DispositionError,
    _require_step16_code_identity,
    _require_step16_runtime_semantics,
    activate_issue_790_step16_plan,
    finalise_issue_790_step16_plan,
    issue_790_checked_approval,
    seal_issue_790_step16_plan,
)
from newsroom.control_plane.issue_790_prepared_canary import (
    BOUNDED_CANARY_AUTHORITY_CONSUMED,
    CANDIDATE_EVENT_ID,
    CANDIDATE_LEDGER_SEQ,
    FAIL_BRANCH_INVENTORY,
    FIELD_CLASSIFICATION,
    LIVE_ONLY_PREDISPATCH_GATES,
    PREPARED_CANARY_ABSENT,
    PREPARED_CANARY_DIGEST_DRIFT,
    PREPARED_CANARY_RECORD_INVALID,
    PreparedCanaryError,
    consume_prepared_canary,
    prepare_issue_790_canary,
    prepared_canary_from_record,
    prepared_canary_record,
    unused_queued_attempt_zero_candidates,
    _candidate_from_plan,
)
from newsroom.control_plane.issue_790_rehearsal import (
    RehearsalEvaluationGraphitiRunner,
    RehearsalRealGraphitiAdapter,
    live_issue_790_store_paths,
    refuse_live_issue_790_store_paths,
    run_prepared_canary_rehearsal,
    sqlite_backup_copy,
)
from newsroom.tests.test_issue_790_rehearsal_fixtures import (
    EVENT_13361,
    EVENT_13677,
    EVENT_13683,
    EVENT_13689,
    EVENT_13690,
    EXACT_HEAD,
    LEDGER_13677,
    LEDGER_13683,
    LEDGER_13689,
    LEDGER_13690,
    LIVE_13361_AVAILABLE_AT,
    OBSERVED_AT,
    SEALED_13361_AVAILABLE_AT,
    SUCCESSOR_EVENT_ID,
    SUCCESSOR_LEDGER_SEQ,
    build_rehearsal_stores,
    candidate_identity,
    dispatch_started_count,
    event_identity,
    file_digest,
    insert_unused_queued_attempt_zero,
    mutate_retry_field,
    retry_available_at,
    transfer_proving_identity,
)
from newsroom.tests.test_issue_790_step16_activation import (
    _COMMENT_ID,
    _FakeGitHub,
    _comment,
    _payload,
)

_TEST_FILE = Path(__file__)
_ROOT = _TEST_FILE.resolve().parents[2]
_STEP22_ACTIVATION_SHA = "f7946e8a53620b56a09bb4ae923a8003b92da760"
_STEP22_ACTIVATION_TREE = "8c443bde47adc34d8e9a38dd6ba80359fd56fa16"
_STEP22_ACTIVATION_FG_RUN = 33387559058
_SUCCESSOR_SHA = "a1f24f4e069af95ac94e8744f8f94c3e28e10d32"
_SUCCESSOR_TREE = "a59e5c620cb9a15a509bced75b96e9a44da940ff"
_SUCCESSOR_FG_RUN = 33404219327
_ACTIVATION_PARENT_SHA = "00f7df954c21816e9be13d783871186efaa84073"
_LIVE_13671_AVAILABLE_AT = "2026-08-31T16:17:39.354162Z"
_LIVE_OBSERVED_AT = datetime(2026, 8, 31, 17, 39, 23, 783082, tzinfo=UTC)
_PRODUCTION_DISPOSITION = "sha256:" + "cd" * 32


def _prepare(stores, *, store=None, role="preflight", **kwargs):
    return prepare_issue_790_canary(
        store=store or stores.work_unpublished,
        proving_store=stores.proving,
        plan=stores.plan,
        observed_at=OBSERVED_AT,
        exact_head=EXACT_HEAD,
        role=role,
        **kwargs,
    )


def _prepare_production(
    stores,
    *,
    plan: Mapping[str, object],
    event_id: str,
    ledger_seq: int,
):
    return prepared_canary_from_record(
        prepared_canary_record(
            prepare_issue_790_canary(
                store=stores.work_unpublished,
                proving_store=stores.proving,
                plan=plan,
                observed_at=OBSERVED_AT,
                exact_head=EXACT_HEAD,
                event_id=event_id,
                ledger_seq=ledger_seq,
                role="canary",
            )
        )
    )


def _bind_qualified_candidate(
    monkeypatch: pytest.MonkeyPatch, *, event_id: str, ledger_seq: int
) -> None:
    """Bind a downstream-path fixture to the exact candidate it exercises."""

    qualification = dict(
        disposition._step18_candidate_qualification(_step22_activated_plan())
    )
    qualification["event_id"] = event_id
    qualification["ledger_seq"] = ledger_seq
    monkeypatch.setattr(
        disposition,
        "_step18_candidate_qualification",
        lambda _plan: qualification,
    )


def _remove_snapshot_binding_from_canary_consumption(
    store: Path,
    *,
    ledger_seq: int,
) -> str:
    """Rewrite a no-outcome fixture to the exact legacy preflight shape."""

    connection = sqlite3.connect(store)
    try:
        row = connection.execute(
            "SELECT consumption_digest,record_json "
            "FROM issue_790_bounded_canary_consumptions WHERE ledger_seq=?",
            (ledger_seq,),
        ).fetchone()
        assert row is not None
        old_digest = str(row[0])
        record = json.loads(str(row[1]))
        preflight = dict(record["preflight_evidence"])
        for field in (
            "pre_operation_snapshot_digest",
            "prepared_canary_decision_digest",
            "prepared_canary_record_digest",
        ):
            preflight.pop(field, None)
        preflight.pop("evidence_digest", None)
        preflight["evidence_digest"] = digest_canonical(preflight)
        record["preflight_evidence"] = preflight
        record["preflight_evidence_digest"] = preflight["evidence_digest"]
        record.pop("consumption_digest", None)
        new_digest = digest_canonical(record)
        record["consumption_digest"] = new_digest
        connection.execute("PRAGMA foreign_keys=OFF")
        connection.execute(
            "UPDATE issue_790_bounded_canary_consumptions "
            "SET consumption_digest=?,record_json=? WHERE consumption_digest=?",
            (
                new_digest,
                json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                old_digest,
            ),
        )
        connection.commit()
        return new_digest
    finally:
        connection.close()


def _step22_activated_plan() -> dict[str, object]:
    pending = json.loads((_ROOT / ISSUE_790_STEP22_PENDING_PLAN_PATH).read_text())
    pre_dispatch = json.loads((_ROOT / ISSUE_790_STEP16_PRE_DISPATCH_PATH).read_text())
    candidate = seal_issue_790_step16_plan(
        pending,
        issue_790_checked_approval(str(pending["canonical_digest"])),
        pre_dispatch=pre_dispatch,
    )
    plan = finalise_issue_790_step16_plan(
        candidate,
        {
            "approved_by": "github:fol2",
            "approval_reference": (
                "https://github.com/fol2/newsroom/issues/790#issuecomment-5477950294"
            ),
            "approved_at": "2026-08-31T11:55:32.000000Z",
            "scope": "CONSERVATIVE_SUBSCRIPTION_CLI_USAGE_DISPOSITION",
            "reviewed_correction_revision": _STEP22_ACTIVATION_SHA,
            "reviewed_correction_tree": _STEP22_ACTIVATION_TREE,
        },
        pre_dispatch=pre_dispatch,
    )
    sequence = dict(plan["sequence"])
    binding = dict(sequence["owner_activation"])
    binding["focus_gate_run_id"] = _STEP22_ACTIVATION_FG_RUN
    binding["focus_gate_run_url"] = (
        f"https://github.com/fol2/newsroom/actions/runs/{_STEP22_ACTIVATION_FG_RUN}"
    )
    sequence["owner_activation"] = binding
    plan["sequence"] = sequence
    return plan


def _closed_circuit() -> dict[str, object]:
    return {
        "state": "CLOSED",
        "opened_at": None,
        "available_at": None,
        "failure_code": None,
    }


def _sqlite_canary_event(store: Path, *, ledger_seq: int) -> dict[str, object]:
    connection = sqlite3.connect(store)
    try:
        row = connection.execute(
            "SELECT event_id,ledger_seq,state,attempt_count,provider_dispatched,"
            "claim_owner,claim_expires_at,terminal_at,available_at "
            "FROM unpublished_graphiti_revision_events WHERE ledger_seq=?",
            (ledger_seq,),
        ).fetchone()
    finally:
        connection.close()
    assert row is not None
    return {
        "event_id": row[0],
        "ledger_seq": row[1],
        "state": row[2],
        "attempt_count": row[3],
        "provider_dispatched": row[4],
        "claim_owner": row[5],
        "claim_expires_at": row[6],
        "terminal_at": row[7],
        "available_at": row[8],
    }


def _expire_interrupted_canary_claim(store: Path, *, ledger_seq: int) -> None:
    connection = sqlite3.connect(store)
    try:
        connection.execute(
            "UPDATE unpublished_graphiti_revision_events SET claim_expires_at=? "
            "WHERE ledger_seq=?",
            ((OBSERVED_AT - timedelta(seconds=1)).isoformat(), ledger_seq),
        )
        connection.commit()
    finally:
        connection.close()


def _runtime_semantics_kwargs(
    event: dict[str, object],
    *,
    observed_at: datetime = OBSERVED_AT,
) -> dict[str, object]:
    return {
        "evidence": _successor_evidence(),
        "route_state": {"state": "OPEN", "reason": "SYSTEMIC_TRANSPORT"},
        "circuit_state": _closed_circuit(),
        "observed_at": observed_at,
        "canary_event": event,
    }


def _production_runtime_evidence() -> dict[str, object]:
    return {
        "revision": EXACT_HEAD,
        "store_quick_check": "ok",
        "worker": {
            "label": "com.jamesto.newsroom-graphiti-worker",
            "launchctl_loaded": False,
            "process_ids": [],
        },
    }


def _activate_step22(store: Path) -> dict[str, object]:
    pending = json.loads((_ROOT / ISSUE_790_STEP22_PENDING_PLAN_PATH).read_text())
    pre_dispatch = json.loads((_ROOT / ISSUE_790_STEP16_PRE_DISPATCH_PATH).read_text())
    candidate = seal_issue_790_step16_plan(
        pending,
        issue_790_checked_approval(str(pending["canonical_digest"])),
        pre_dispatch=pre_dispatch,
    )
    payload = _payload(candidate)
    github = _FakeGitHub(_comment(payload))
    activated = activate_issue_790_step16_plan(
        candidate,
        comment_id=_COMMENT_ID,
        pre_dispatch=pre_dispatch,
        store=store,
        github_api=github,
    )
    return {**activated, "github": github, "candidate": candidate}


def _seed_production_disposition(store: Path, plan: Mapping[str, object]) -> None:
    target = plan["target"]
    record = {
        "exact_usage_remains_unknown": True,
        "unknown_spend_released": False,
    }
    connection = sqlite3.connect(store)
    try:
        connection.execute("PRAGMA foreign_keys=OFF")
        connection.execute(
            "INSERT INTO model_usage_conservative_dispositions("
            "disposition_digest,invocation_id,terminal_digest,allocation_digest,"
            "policy_digest,approved_plan_digest,authority_digest,approved_by,"
            "approval_reference,approved_at,observed_at,usage_status,record_json"
            ") VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                _PRODUCTION_DISPOSITION,
                target["invocation_id"],
                target["terminal_digest"],
                target["allocation_digest"],
                "sha256:" + "11" * 32,
                plan["canonical_digest"],
                "sha256:" + "22" * 32,
                "github:fol2",
                "https://github.com/fol2/newsroom/issues/790#issuecomment-1",
                "2026-08-31T11:55:32.000000Z",
                "2026-08-31T11:55:33.000000Z",
                "ESTIMATED",
                json.dumps(record, sort_keys=True, separators=(",", ":")),
            ),
        )
        connection.execute(
            "INSERT INTO model_usage_route_circuit_events("
            "event_digest,route,state,reason,invocation_id,recorded_at,record_json"
            ") VALUES(?,?,?,?,?,?,?)",
            (
                "sha256:" + "33" * 32,
                "GRAPHITI_CHAT_PRIMARY",
                "CLOSED",
                f"AUTHORISED_OPERATOR_RESET:{_PRODUCTION_DISPOSITION}",
                target["invocation_id"],
                "2026-08-31T17:39:23.783082Z",
                "{}",
            ),
        )
        connection.commit()
    finally:
        connection.close()


def _patch_production_predispatch(
    monkeypatch: pytest.MonkeyPatch,
    *,
    plan: dict[str, object],
) -> None:
    retry = plan["retry_forbidden_events"]
    monkeypatch.setattr(disposition, "_assert_exact_target", lambda *a, **k: None)
    monkeypatch.setattr(
        disposition, "_require_sequence_predecessor", lambda *a, **k: None
    )
    monkeypatch.setattr(disposition, "_require_retry_exclusions", lambda *a, **k: [])
    monkeypatch.setattr(
        disposition, "_retain_retry_exclusions_for_plan", lambda *a, **k: []
    )
    monkeypatch.setattr(
        disposition, "_require_retry_events_unchanged", lambda *a, **k: retry
    )
    monkeypatch.setattr(disposition, "_retry_event_snapshots", lambda *a, **k: retry)
    monkeypatch.setattr(
        disposition,
        "collect_issue_790_operational_evidence",
        lambda **k: _production_runtime_evidence(),
    )
    monkeypatch.setattr(
        disposition,
        "_validate_operational_evidence",
        lambda evidence, **k: evidence,
    )
    monkeypatch.setattr(
        disposition,
        "_worker_state",
        lambda: _production_runtime_evidence()["worker"],
    )
    monkeypatch.setattr(
        disposition, "_require_issue_790_canary_route", lambda **k: None
    )
    monkeypatch.setattr(
        disposition.ModelUsageService,
        "route_state",
        lambda self, route: {
            "state": "CLOSED",
            "reason": f"AUTHORISED_OPERATOR_RESET:{_PRODUCTION_DISPOSITION}",
        },
    )
    unsigned = {
        "schema_version": "newsroom.issue-790.graphiti-runtime-readiness.v1",
        "framework": "graphiti-core",
        "framework_version": "0.29.3",
        "provider_calls": 0,
        "credential_resolution": False,
    }
    monkeypatch.setattr(
        disposition,
        "_qualify_real_graphiti_runtime",
        lambda: {**unsigned, "runtime_digest": digest_canonical(unsigned)},
    )


def _successor_evidence() -> dict[str, object]:
    return {
        "revision": _SUCCESSOR_SHA,
        "tree": _SUCCESSOR_TREE,
        "github_main_revision": _SUCCESSOR_SHA,
        "repository_root": str(_ROOT),
        "store_quick_check": "ok",
        "worker": {
            "label": "com.jamesto.newsroom-graphiti-worker",
            "launchctl_loaded": False,
            "process_ids": [],
        },
        "ci_test": {
            "name": "focus-gates",
            "status": "completed",
            "conclusion": "success",
            "head_sha": _SUCCESSOR_SHA,
            "url": (
                "https://github.com/fol2/newsroom/actions/runs/"
                f"{_SUCCESSOR_FG_RUN}"
            ),
        },
    }


def test_field_classification_keeps_available_at_as_audit_only() -> None:
    assert FIELD_CLASSIFICATION["available_at"] == "C"
    assert FIELD_CLASSIFICATION["exact_head"] == "A"
    assert FIELD_CLASSIFICATION["retry_safety_states"] == "B"
    assert FIELD_CLASSIFICATION["provider_response"] == "D"


def test_prepared_canary_accepts_13361_available_at_drift(tmp_path: Path) -> None:
    stores = build_rehearsal_stores(tmp_path)
    assert retry_available_at(stores.work_unpublished, 13361) == LIVE_13361_AVAILABLE_AT
    sealed = next(
        item
        for item in stores.plan["retry_forbidden_events"]
        if item["ledger_seq"] == 13361
    )
    assert sealed["available_at"] == SEALED_13361_AVAILABLE_AT
    prepared = _prepare(stores)
    assert prepared.candidate_identity["event_id"] == CANDIDATE_EVENT_ID
    assert prepared.candidate_identity["ledger_seq"] == CANDIDATE_LEDGER_SEQ
    assert prepared.decision_digest.startswith("sha256:")
    assert candidate_identity(stores.sealed_unpublished)[0] == CANDIDATE_EVENT_ID


def test_ready_digest_is_stable_for_unchanged_copy(tmp_path: Path) -> None:
    stores = build_rehearsal_stores(tmp_path)
    first = _prepare(stores)
    second = _prepare(stores, role="canary")
    assert first.decision_digest == second.decision_digest
    assert first.as_decision_payload() == second.as_decision_payload()
    assert file_digest(stores.sealed_unpublished) == stores.sealed_digest
    assert file_digest(stores.work_unpublished) == stores.sealed_digest


def test_ready_implies_dispatch_started(tmp_path: Path) -> None:
    stores = build_rehearsal_stores(tmp_path)
    before = file_digest(stores.work_unpublished)
    prepared = _prepare(stores)
    assert file_digest(stores.work_unpublished) == before
    result = run_prepared_canary_rehearsal(
        store=stores.work_unpublished,
        proving_store=stores.proving,
        plan=stores.plan,
        observed_at=OBSERVED_AT,
        exact_head=EXACT_HEAD,
        prepared=prepared,
    )
    assert result["decision_digest"] == prepared.decision_digest
    assert result["dispatch_started"] is True
    assert result["provider_calls"] == 0
    assert RehearsalRealGraphitiAdapter.provider_calls == 0
    assert dispatch_started_count(stores.work_unpublished) >= 1
    assert file_digest(stores.sealed_unpublished) == stores.sealed_digest
    assert candidate_identity(stores.sealed_unpublished)[0] == CANDIDATE_EVENT_ID


def test_ready_implies_dispatch_started_for_successor_unused_attempt_zero(
    tmp_path: Path,
) -> None:
    stores = build_rehearsal_stores(tmp_path, successor=True)
    before = file_digest(stores.work_unpublished)
    with pytest.raises(PreparedCanaryError) as caught:
        _prepare(stores)
    assert caught.value.failure_code == "CANDIDATE_NOT_FRESH"
    assert file_digest(stores.work_unpublished) == before
    assert candidate_identity(stores.sealed_unpublished) == (
        CANDIDATE_EVENT_ID,
        CANDIDATE_LEDGER_SEQ,
        "CONFIGURATION_HELD",
    )
    assert event_identity(stores.sealed_unpublished, SUCCESSOR_LEDGER_SEQ)[0] == (
        SUCCESSOR_EVENT_ID
    )


def test_ready_successor_exact_head_implies_dispatch_started(tmp_path: Path) -> None:
    stores = build_rehearsal_stores(tmp_path)
    plan = _step22_activated_plan()
    sequence = plan["sequence"]
    assert sequence["reviewed_correction_revision"] == _STEP22_ACTIVATION_SHA
    assert sequence["owner_activation"]["focus_gate_run_id"] == _STEP22_ACTIVATION_FG_RUN
    _require_step16_code_identity(plan, evidence=_successor_evidence())
    before = file_digest(stores.work_unpublished)
    prepared = prepare_issue_790_canary(
        store=stores.work_unpublished,
        proving_store=stores.proving,
        plan=plan,
        observed_at=OBSERVED_AT,
        exact_head=_SUCCESSOR_SHA,
        role="preflight",
    )
    assert file_digest(stores.work_unpublished) == before
    result = run_prepared_canary_rehearsal(
        store=stores.work_unpublished,
        proving_store=stores.proving,
        plan=plan,
        observed_at=OBSERVED_AT,
        event_id=CANDIDATE_EVENT_ID,
        ledger_seq=CANDIDATE_LEDGER_SEQ,
        prepared=prepared,
        exact_head=_SUCCESSOR_SHA,
    )
    assert result["decision_digest"] == prepared.decision_digest
    assert result["dispatch_started"] is True
    assert result["provider_calls"] == 0
    assert RehearsalRealGraphitiAdapter.provider_calls == 0
    assert dispatch_started_count(stores.work_unpublished) >= 1
    assert file_digest(stores.sealed_unpublished) == stores.sealed_digest
    assert candidate_identity(stores.sealed_unpublished) == (
        CANDIDATE_EVENT_ID,
        CANDIDATE_LEDGER_SEQ,
        "QUEUED",
    )


def test_step22_activation_rejects_non_ancestor_exact_head() -> None:
    plan = _step22_activated_plan()
    evidence = _successor_evidence()
    evidence["revision"] = _ACTIVATION_PARENT_SHA
    evidence["github_main_revision"] = _ACTIVATION_PARENT_SHA
    evidence["ci_test"] = dict(evidence["ci_test"])
    evidence["ci_test"]["head_sha"] = _ACTIVATION_PARENT_SHA
    with pytest.raises(
        Issue790DispositionError, match="reviewed correction identity"
    ):
        _require_step16_code_identity(plan, evidence=evidence)


def test_step22_activation_rejects_focus_gates_not_on_current_exact_head() -> None:
    plan = _step22_activated_plan()
    evidence = _successor_evidence()
    evidence["ci_test"] = {
        "name": "focus-gates",
        "status": "completed",
        "conclusion": "success",
        "head_sha": _STEP22_ACTIVATION_SHA,
        "url": (
            "https://github.com/fol2/newsroom/actions/runs/"
            f"{_STEP22_ACTIVATION_FG_RUN}"
        ),
    }
    with pytest.raises(Issue790DispositionError, match="focus gate evidence differs"):
        _require_step16_code_identity(plan, evidence=evidence)


def test_successor_safety_drift_fail_closes_before_dispatch(tmp_path: Path) -> None:
    stores = build_rehearsal_stores(tmp_path)
    plan = _step22_activated_plan()
    mutate_retry_field(
        stores.work_unpublished, ledger_seq=13361, field="state", value="QUEUED"
    )
    RehearsalRealGraphitiAdapter.provider_calls = 0
    RehearsalRealGraphitiAdapter.dispatch_started = False
    with pytest.raises(PreparedCanaryError) as caught:
        prepare_issue_790_canary(
            store=stores.work_unpublished,
            proving_store=stores.proving,
            plan=plan,
            observed_at=OBSERVED_AT,
            exact_head=_SUCCESSOR_SHA,
            role="preflight",
        )
    assert caught.value.failure_code == "RETRY_FORBIDDEN_SAFETY_STATE"
    assert RehearsalRealGraphitiAdapter.dispatch_started is False
    assert RehearsalRealGraphitiAdapter.provider_calls == 0
    assert dispatch_started_count(stores.work_unpublished) == 0
    assert candidate_identity(stores.sealed_unpublished)[0] == CANDIDATE_EVENT_ID
    assert candidate_identity(stores.work_unpublished)[0] == CANDIDATE_EVENT_ID


def test_successor_digest_drift_fail_closes_before_dispatch(tmp_path: Path) -> None:
    stores = build_rehearsal_stores(tmp_path)
    plan = _step22_activated_plan()
    prepared = prepare_issue_790_canary(
        store=stores.work_unpublished,
        proving_store=stores.proving,
        plan=plan,
        observed_at=OBSERVED_AT,
        exact_head=_SUCCESSOR_SHA,
        role="preflight",
    )
    drifted = replace(prepared, decision_digest="sha256:" + "00" * 32)
    RehearsalRealGraphitiAdapter.provider_calls = 0
    RehearsalRealGraphitiAdapter.dispatch_started = False
    with pytest.raises(PreparedCanaryError) as caught:
        run_prepared_canary_rehearsal(
            store=stores.work_unpublished,
            proving_store=stores.proving,
            plan=plan,
            observed_at=OBSERVED_AT,
            event_id=CANDIDATE_EVENT_ID,
            ledger_seq=CANDIDATE_LEDGER_SEQ,
            prepared=drifted,
            exact_head=_SUCCESSOR_SHA,
        )
    assert caught.value.failure_code == PREPARED_CANARY_DIGEST_DRIFT
    assert RehearsalRealGraphitiAdapter.dispatch_started is False
    assert dispatch_started_count(stores.work_unpublished) == 0
    assert candidate_identity(stores.work_unpublished)[0] == CANDIDATE_EVENT_ID


def test_event_13665_identity_is_not_mutated(tmp_path: Path) -> None:
    stores = build_rehearsal_stores(tmp_path)
    _prepare(stores)
    mutate_retry_field(
        stores.work_unpublished, ledger_seq=13361, field="state", value="QUEUED"
    )
    with pytest.raises(PreparedCanaryError) as caught:
        _prepare(stores)
    assert caught.value.failure_code == "RETRY_FORBIDDEN_SAFETY_STATE"
    event_id, ledger_seq, _state = candidate_identity(stores.work_unpublished)
    assert event_id == CANDIDATE_EVENT_ID
    assert ledger_seq == CANDIDATE_LEDGER_SEQ
    assert candidate_identity(stores.sealed_unpublished)[0] == CANDIDATE_EVENT_ID


def test_spent_13665_is_retry_forbidden_target(tmp_path: Path) -> None:
    stores = build_rehearsal_stores(tmp_path, successor=True)
    RehearsalRealGraphitiAdapter.provider_calls = 0
    RehearsalRealGraphitiAdapter.dispatch_started = False
    with pytest.raises(PreparedCanaryError) as caught:
        prepare_issue_790_canary(
            store=stores.work_unpublished,
            proving_store=stores.proving,
            plan=stores.plan,
            observed_at=OBSERVED_AT,
            exact_head=EXACT_HEAD,
            event_id=CANDIDATE_EVENT_ID,
            ledger_seq=CANDIDATE_LEDGER_SEQ,
            role="canary",
        )
    assert caught.value.failure_code == "RETRY_FORBIDDEN_TARGET"
    assert RehearsalRealGraphitiAdapter.dispatch_started is False
    assert RehearsalRealGraphitiAdapter.provider_calls == 0
    assert dispatch_started_count(stores.work_unpublished) == 0
    assert candidate_identity(stores.work_unpublished) == (
        CANDIDATE_EVENT_ID,
        CANDIDATE_LEDGER_SEQ,
        "CONFIGURATION_HELD",
    )
    assert candidate_identity(stores.sealed_unpublished) == (
        CANDIDATE_EVENT_ID,
        CANDIDATE_LEDGER_SEQ,
        "CONFIGURATION_HELD",
    )


def test_cli_flags_disagree_with_unused_candidate_fail_closes(tmp_path: Path) -> None:
    stores = build_rehearsal_stores(tmp_path)
    with pytest.raises(PreparedCanaryError) as caught:
        prepare_issue_790_canary(
            store=stores.work_unpublished,
            proving_store=stores.proving,
            plan=stores.plan,
            observed_at=OBSERVED_AT,
            exact_head=EXACT_HEAD,
            event_id=SUCCESSOR_EVENT_ID,
            ledger_seq=SUCCESSOR_LEDGER_SEQ,
            role="canary",
        )
    assert caught.value.failure_code == "CANDIDATE_IDENTITY"
    assert candidate_identity(stores.work_unpublished)[2] == "QUEUED"


def test_step22_spent_13665_successor_unused_attempt_zero_survives_full_path(
    tmp_path: Path,
) -> None:
    """A qualified spent candidate cannot drift to an unused successor."""

    stores = build_rehearsal_stores(tmp_path, successor=True)
    named = unused_queued_attempt_zero_candidates(stores.work_unpublished, stores.plan)
    assert named[0] == (SUCCESSOR_EVENT_ID, SUCCESSOR_LEDGER_SEQ)
    with pytest.raises(PreparedCanaryError) as implicit:
        _prepare(stores)
    assert implicit.value.failure_code == "CANDIDATE_NOT_FRESH"
    with pytest.raises(PreparedCanaryError) as explicit:
        _candidate_from_plan(
            stores.plan,
            event_id=SUCCESSOR_EVENT_ID,
            ledger_seq=SUCCESSOR_LEDGER_SEQ,
            role="canary",
            store=stores.work_unpublished,
        )
    assert explicit.value.failure_code == "CANDIDATE_IDENTITY"


def test_step22_sqlite_integer_zero_provider_dispatched_is_untouched(
    tmp_path: Path,
) -> None:
    stores = build_rehearsal_stores(tmp_path)
    plan = _step22_activated_plan()
    assert plan["sequence"]["candidate_event_qualification"]["ledger_seq"] == (
        CANDIDATE_LEDGER_SEQ
    )
    event = _sqlite_canary_event(
        stores.work_unpublished, ledger_seq=CANDIDATE_LEDGER_SEQ
    )
    assert event["state"] == "QUEUED"
    assert event["attempt_count"] == 0
    assert type(event["attempt_count"]) is int
    assert event["provider_dispatched"] == 0
    assert type(event["provider_dispatched"]) is int
    _require_step16_runtime_semantics(
        plan, **_runtime_semantics_kwargs(event)
    )
    event["provider_dispatched"] = 1
    with pytest.raises(Issue790DispositionError, match="not untouched"):
        _require_step16_runtime_semantics(
            plan, **_runtime_semantics_kwargs(event)
        )


def test_step22_sqlite_successor_attempt_zero_is_pre_dispatch_untouched(
    tmp_path: Path,
) -> None:
    stores = build_rehearsal_stores(tmp_path, successor=True)
    plan = _step22_activated_plan()
    qualification = plan["sequence"]["candidate_event_qualification"]
    assert qualification["ledger_seq"] == CANDIDATE_LEDGER_SEQ
    assert qualification["event_id"] == CANDIDATE_EVENT_ID
    connection = sqlite3.connect(stores.work_unpublished)
    try:
        connection.execute(
            "UPDATE unpublished_graphiti_revision_events SET available_at=? "
            "WHERE ledger_seq=?",
            (_LIVE_13671_AVAILABLE_AT, SUCCESSOR_LEDGER_SEQ),
        )
        connection.commit()
    finally:
        connection.close()
    event = _sqlite_canary_event(
        stores.work_unpublished, ledger_seq=SUCCESSOR_LEDGER_SEQ
    )
    assert event["event_id"] == SUCCESSOR_EVENT_ID
    assert event["ledger_seq"] == SUCCESSOR_LEDGER_SEQ
    assert event["state"] == "QUEUED"
    assert event["attempt_count"] == 0
    assert event["provider_dispatched"] == 0
    assert type(event["provider_dispatched"]) is int
    assert event["available_at"] == _LIVE_13671_AVAILABLE_AT
    assert event["claim_owner"] is None
    _require_step16_runtime_semantics(
        plan,
        **_runtime_semantics_kwargs(event, observed_at=_LIVE_OBSERVED_AT),
    )
    spent = _sqlite_canary_event(
        stores.work_unpublished, ledger_seq=CANDIDATE_LEDGER_SEQ
    )
    assert spent["state"] == "CONFIGURATION_HELD"
    assert spent["provider_dispatched"] == 1
    with pytest.raises(Issue790DispositionError, match="not untouched"):
        _require_step16_runtime_semantics(
            plan,
            **_runtime_semantics_kwargs(spent, observed_at=_LIVE_OBSERVED_AT),
        )


def test_step22_unused_13671_survives_production_pre_dispatch_untouched(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fails on 6c101627 with 'pre-dispatch event is not untouched'."""

    stores = build_rehearsal_stores(tmp_path, successor=True)
    activated = _activate_step22(stores.work_unpublished)
    plan = activated["plan"]
    _bind_qualified_candidate(
        monkeypatch, event_id=SUCCESSOR_EVENT_ID, ledger_seq=SUCCESSOR_LEDGER_SEQ
    )
    _seed_production_disposition(stores.work_unpublished, plan)
    assert plan["sequence"]["candidate_event_qualification"]["ledger_seq"] == (
        CANDIDATE_LEDGER_SEQ
    )
    event = _sqlite_canary_event(
        stores.work_unpublished, ledger_seq=SUCCESSOR_LEDGER_SEQ
    )
    assert event["provider_dispatched"] == 0
    assert type(event["provider_dispatched"]) is int
    assert event["attempt_count"] == 0
    _patch_production_predispatch(monkeypatch, plan=plan)
    consume_calls: list[str] = []

    def consume_successor(**values: object) -> None:
        consume_calls.append(str(values["event_id"]))
        assert values["event_id"] == SUCCESSOR_EVENT_ID
        return None

    monkeypatch.setattr(disposition, "_consume_issue_790_event", consume_successor)
    backup = tmp_path / "pre-dispatch-13671.sqlite3"
    prepared = _prepare_production(
        stores,
        plan=plan,
        event_id=SUCCESSOR_EVENT_ID,
        ledger_seq=SUCCESSOR_LEDGER_SEQ,
    )
    receipt = disposition.run_issue_790_canary(
        store=stores.work_unpublished,
        proving_store=stores.proving,
        backup_path=backup,
        plan=plan,
        observed_at=OBSERVED_AT,
        repository_root=tmp_path,
        event_id=SUCCESSOR_EVENT_ID,
        ledger_seq=SUCCESSOR_LEDGER_SEQ,
        disposition_digest=_PRODUCTION_DISPOSITION,
        prepared=prepared,
        github_api=activated["github"],
    )
    assert backup.is_file()
    assert consume_calls == [SUCCESSOR_EVENT_ID]
    assert receipt["resumed_zero_io_finalisation"] is False
    assert receipt["consumption"]["event_id"] == SUCCESSOR_EVENT_ID
    assert receipt["consumption"]["ledger_seq"] == SUCCESSOR_LEDGER_SEQ
    with pytest.raises(PreparedCanaryError) as caught:
        _candidate_from_plan(
            plan,
            event_id=CANDIDATE_EVENT_ID,
            ledger_seq=CANDIDATE_LEDGER_SEQ,
            role="canary",
            store=stores.work_unpublished,
        )
    assert caught.value.failure_code == BOUNDED_CANARY_AUTHORITY_CONSUMED
    resumed = disposition.run_issue_790_canary(
        store=stores.work_unpublished,
        proving_store=stores.proving,
        backup_path=backup,
        plan=plan,
        observed_at=OBSERVED_AT,
        repository_root=tmp_path,
        event_id=SUCCESSOR_EVENT_ID,
        ledger_seq=SUCCESSOR_LEDGER_SEQ,
        disposition_digest=_PRODUCTION_DISPOSITION,
        github_api=activated["github"],
    )
    assert resumed["resumed_zero_io_finalisation"] is True
    assert consume_calls == [SUCCESSOR_EVENT_ID]


def test_step22_consumed_13671_brokererror_setup_survives_full_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Live 13671: consume then BrokerError setup stays NOT_DISPATCHED.

    Fails on 47673c01 because this covering node is absent.
    """

    from newsroom.control_plane.broker import BrokerError
    from newsroom.control_plane.graphiti import EvaluationGraphitiRunner
    from newsroom.graphiti_adapter import real as real_adapter

    stores = build_rehearsal_stores(tmp_path, successor=True)
    activated = _activate_step22(stores.work_unpublished)
    plan = activated["plan"]
    _bind_qualified_candidate(
        monkeypatch, event_id=SUCCESSOR_EVENT_ID, ledger_seq=SUCCESSOR_LEDGER_SEQ
    )
    _seed_production_disposition(stores.work_unpublished, plan)
    _patch_production_predispatch(monkeypatch, plan=plan)
    monkeypatch.setattr(
        EvaluationGraphitiRunner,
        "requires_canonical_control_plane_stores",
        False,
    )
    def refuse_keychain() -> str:
        raise BrokerError("Keychain class OPENROUTER_API lookup failed")

    monkeypatch.setattr(real_adapter, "_load_graphiti", lambda: object())
    monkeypatch.setattr(real_adapter, "openrouter_api_key", refuse_keychain)

    async def must_not_dispatch(**_values: object) -> object:
        raise AssertionError("BrokerError setup reached provider")

    monkeypatch.setattr(real_adapter, "_add_episode", must_not_dispatch)
    consume_calls: list[str] = []
    real_consume = disposition._consume_issue_790_event

    def consume_successor(**values: object) -> object:
        consume_calls.append(str(values["event_id"]))
        assert values["event_id"] == SUCCESSOR_EVENT_ID
        values.setdefault("clock", lambda: OBSERVED_AT)
        return real_consume(**values)

    monkeypatch.setattr(disposition, "_consume_issue_790_event", consume_successor)
    prepared = _prepare_production(
        stores,
        plan=plan,
        event_id=SUCCESSOR_EVENT_ID,
        ledger_seq=SUCCESSOR_LEDGER_SEQ,
    )
    backup = tmp_path / "brokererror-13671.sqlite3"
    receipt = disposition.run_issue_790_canary(
        store=stores.work_unpublished,
        proving_store=stores.proving,
        backup_path=backup,
        plan=plan,
        observed_at=OBSERVED_AT,
        repository_root=tmp_path,
        event_id=SUCCESSOR_EVENT_ID,
        ledger_seq=SUCCESSOR_LEDGER_SEQ,
        disposition_digest=_PRODUCTION_DISPOSITION,
        prepared=prepared,
        github_api=activated["github"],
    )
    event = _sqlite_canary_event(
        stores.work_unpublished, ledger_seq=SUCCESSOR_LEDGER_SEQ
    )
    after = receipt["event_after"]["event"]
    usage = receipt["usage_evidence"]
    connection = sqlite3.connect(stores.work_unpublished)
    try:
        failure_code = connection.execute(
            "SELECT last_failure_code FROM unpublished_graphiti_revision_events "
            "WHERE ledger_seq=?",
            (SUCCESSOR_LEDGER_SEQ,),
        ).fetchone()[0]
        attempt_row = connection.execute(
            "SELECT outcome,receipt_json FROM unpublished_graphiti_attempt_receipts"
        ).fetchone()
    finally:
        connection.close()
    assert attempt_row is not None
    ingest = json.loads(attempt_row[1])
    unused = unused_queued_attempt_zero_candidates(stores.work_unpublished, plan)
    assert consume_calls == [SUCCESSOR_EVENT_ID]
    assert receipt["consumption"]["event_id"] == SUCCESSOR_EVENT_ID
    assert receipt["consumption"]["ledger_seq"] == SUCCESSOR_LEDGER_SEQ
    assert receipt["outcome"]["failure_code_after_seal"] == (
        "BOUNDED_CANARY_AUTHORITY_EXHAUSTED:BrokerError"
    )
    assert receipt["resumed_zero_io_finalisation"] is False
    assert receipt["provider_dispatch_attempted_this_run"] is True
    assert receipt["canary_evidence_passed"] is False
    assert receipt["retry_authorised"] is False
    assert receipt["publication_performed"] is False
    assert receipt["exception"] is None
    assert receipt["runtime_readiness"]["credential_resolution"] is False
    assert receipt["runtime_readiness"]["provider_calls"] == 0
    assert usage["committed_dispatch_observations"] == []
    assert usage["provider_backed_terminal_count"] == 0
    assert dispatch_started_count(stores.work_unpublished) == 0
    assert after["state"] == "CONFIGURATION_HELD"
    assert after["attempt_count"] == 1
    assert after["last_failure_code"] == (
        "BOUNDED_CANARY_AUTHORITY_EXHAUSTED:BrokerError"
    )
    assert failure_code == "BOUNDED_CANARY_AUTHORITY_EXHAUSTED:BrokerError"
    assert event["provider_dispatched"] == 0
    assert type(event["provider_dispatched"]) is int
    assert receipt["event_after"]["circuit"]["state"] == "OPEN"
    assert receipt["event_after"]["circuit"]["failure_code"] == "BrokerError"
    assert attempt_row[0] == "FAILED"
    assert ingest["failure_code"] == "PRODUCER_INTERNAL_ERROR"
    assert ingest["setup_failure"] == "BrokerError"
    assert ingest["dispatch_state"] == "NOT_DISPATCHED"
    assert ingest["chat_invocations"] == []
    assert ingest["embedding_usage"]["request_count"] == 0
    assert ingest["usage_basis"] == "NO_EMBEDDING_CALL"
    assert SUCCESSOR_EVENT_ID not in {item[0] for item in unused}
    assert SUCCESSOR_LEDGER_SEQ not in {item[1] for item in unused}
    with pytest.raises(PreparedCanaryError) as caught:
        _candidate_from_plan(
            plan,
            event_id=CANDIDATE_EVENT_ID,
            ledger_seq=CANDIDATE_LEDGER_SEQ,
            role="canary",
            store=stores.work_unpublished,
        )
    assert caught.value.failure_code == BOUNDED_CANARY_AUTHORITY_CONSUMED
    resumed = disposition.run_issue_790_canary(
        store=stores.work_unpublished,
        proving_store=stores.proving,
        backup_path=backup,
        plan=plan,
        observed_at=OBSERVED_AT,
        repository_root=tmp_path,
        event_id=SUCCESSOR_EVENT_ID,
        ledger_seq=SUCCESSOR_LEDGER_SEQ,
        disposition_digest=_PRODUCTION_DISPOSITION,
        github_api=activated["github"],
    )
    assert resumed["resumed_zero_io_finalisation"] is True
    assert consume_calls == [SUCCESSOR_EVENT_ID]
    assert dispatch_started_count(stores.work_unpublished) == 0


def test_step22_consumed_13677_zero_after_embeddings_survives_full_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Live 13677: leftover NEW nodes and no persistable relation is TERMINAL.

    Provider COMPLETE + four embeddings + 0/0/0 was AMBIGUOUS_EFFECT before
    the empty-effect seal covered leftover NEW nodes. Fails on be1e1895
    because this covering node is absent and that seal did not apply.
    """

    from types import SimpleNamespace

    from newsroom.authority.types import UtcTimestamp
    from newsroom.control_plane.graphiti import EvaluationGraphitiRunner
    from newsroom.graphiti_adapter import real as real_adapter
    from newsroom.graphiti_adapter.combined_temporal_pipeline import (
        ExistingGraphitiPipeline,
    )
    from newsroom.graphiti_adapter.evaluation_packet import (
        CURSOR_AGENT_MODEL_ID,
        OPENROUTER_EMBEDDING_SLUG,
    )
    from newsroom.graphiti_adapter.neo4j_guard import GuardState

    persist_calls: list[object] = []
    executions: list[object] = []
    stores = build_rehearsal_stores(tmp_path, unused_13677=True)
    activated = _activate_step22(stores.work_unpublished)
    plan = activated["plan"]
    _bind_qualified_candidate(
        monkeypatch, event_id=EVENT_13677, ledger_seq=LEDGER_13677
    )
    _seed_production_disposition(stores.work_unpublished, plan)
    _patch_production_predispatch(monkeypatch, plan=plan)
    monkeypatch.setattr(
        EvaluationGraphitiRunner,
        "requires_canonical_control_plane_stores",
        False,
    )
    adapter_type = real_adapter.RealGraphitiAdapter

    class ClockedAdapter:
        def __init__(self, **values: object) -> None:
            self.delegate = adapter_type(
                clock=lambda: UtcTimestamp(OBSERVED_AT),
                **values,
            )

        def execute(self, **values: object) -> object:
            execution = self.delegate.execute(**values)
            executions.append(execution)
            return execution

    monkeypatch.setattr(real_adapter, "RealGraphitiAdapter", ClockedAdapter)
    monkeypatch.setattr(real_adapter, "_load_graphiti", lambda: SimpleNamespace())
    monkeypatch.setattr(real_adapter, "openrouter_api_key", lambda: "fixture-key")
    monkeypatch.setattr(
        real_adapter, "neo4j_community_password", lambda: "fixture-password"
    )

    class Guard:
        async def begin(self) -> object:
            return SimpleNamespace(state=GuardState.CREATED)

        async def record_pending_telemetry(self, **_values: object) -> None:
            return None

        async def complete(self, _receipt: object) -> None:
            return None

        async def rollback_pending(self, **_values: object) -> bool:
            return True

    def _record_live_13677_leaves(observer: object) -> None:
        if observer is None or not hasattr(observer, "before_cli_invocation"):
            return
        chat = observer.before_cli_invocation(
            provider="cursor-agent-cli",
            model=CURSOR_AGENT_MODEL_ID,
            prompt="live-13677 leftover NEW without persistable edges",
            schema=None,
        )
        observer.transport_dispatch_started(chat)
        observer.after_cli_invocation(
            chat,
            outcome="COMPLETE",
            usage={
                "usage_basis": "PROVIDER_REPORTED",
                "input_tokens": 5_400,
                "cached_read_tokens": 500,
                "output_tokens": 3_046,
                "total_tokens": 8_446,
            },
        )
        for index, (tokens, cost) in enumerate(((36, 5), (4, 1), (2, 0), (2, 0))):
            embedding = observer.before_embedding_invocation(
                provider="openrouter",
                model=OPENROUTER_EMBEDDING_SLUG,
                input_data=[f"live-13677-embedding-{index}"],
            )
            observer.transport_dispatch_started(embedding)
            observer.after_embedding_invocation(
                embedding,
                outcome="COMPLETE",
                usage={
                    "usage_basis": "PROVIDER_REPORTED",
                    "input_tokens": tokens,
                    "output_tokens": 0,
                    "cached_read_tokens": 0,
                    "cached_write_tokens": 0,
                    "reasoning_tokens": 0,
                    "total_tokens": tokens,
                    "provider_telemetry": {
                        "request_id": f"live-13677-embedding-{index}",
                        "prompt_tokens": tokens,
                        "total_tokens": tokens,
                        "cost_usd_microunits": cost,
                    },
                },
            )

    async def leftover_new_nodes_without_edges(**values: object) -> object:
        telemetry = values["telemetry"]
        telemetry.chat_invocations = [
            {
                "provider": "cursor-agent-cli",
                "model": CURSOR_AGENT_MODEL_ID,
                "outcome": "COMPLETE",
                "usage": {
                    "usage_basis": "PROVIDER_REPORTED",
                    "input_tokens": 5_400,
                    "cached_read_tokens": 500,
                    "output_tokens": 3_046,
                    "total_tokens": 8_446,
                },
            }
        ]
        telemetry.embedding_usage = {
            "usage_basis": "PROVIDER_REPORTED",
            "request_count": 4,
            "embedding_tokens": 44,
            "cost_usd_microunits": 6,
            "requests": [
                {
                    "provider": "openrouter",
                    "model": OPENROUTER_EMBEDDING_SLUG,
                    "request_id": f"live-13677-embedding-{index}",
                    "prompt_tokens": tokens,
                    "total_tokens": tokens,
                    "cost_usd_microunits": cost,
                    "cost_reported": True,
                    "outcome": "COMPLETE",
                }
                for index, (tokens, cost) in enumerate(
                    ((36, 5), (4, 1), (2, 0), (2, 0))
                )
            ],
        }
        _record_live_13677_leaves(values.get("invocation_observer"))

        async def resolve_nodes(
            nodes: list[object],
        ) -> tuple[list[object], dict[str, str], list[tuple[object, object]]]:
            created = [
                SimpleNamespace(
                    uuid=str(getattr(node, "uuid", "new")),
                    attributes={"resolution": "DETERMINISTIC_NEW_NODE"},
                )
                for node in nodes
            ]
            return (
                created,
                {
                    str(getattr(node, "uuid", "new")): str(
                        getattr(node, "uuid", "new")
                    )
                    for node in nodes
                },
                [],
            )

        async def persist_graph(nodes: list[object], edges: list[object]) -> None:
            persist_calls.append((list(nodes), list(edges)))
            raise RuntimeError("persist must not run without persistable edges")

        async def create_embeddings(_embedder: object, _edges: list[object]) -> None:
            raise AssertionError(
                "edge embeddings must not run without persistable edges"
            )

        pipeline = ExistingGraphitiPipeline(
            guard=Guard(),  # type: ignore[arg-type]
            resolve_nodes=resolve_nodes,
            resolve_pointers=lambda edges, _uuid_map: edges,
            create_embeddings=create_embeddings,
            persist_graph=persist_graph,
            embedder=object(),
            run_async=lambda awaitable: awaitable,
            chat_receipt=lambda: list(telemetry.chat_invocations),
            embedding_receipt=lambda: dict(telemetry.embedding_usage),
            complete_receipt=lambda nodes, edges, receipt: values["validate_result"](
                SimpleNamespace(
                    episode=None,
                    nodes=tuple(nodes),
                    edges=tuple(edges),
                ),
                telemetry,
                receipt,
            ),
        )
        await pipeline._prepare_attempt()
        sealed = await pipeline._execute(
            nodes=(SimpleNamespace(uuid="entity-1", attributes={}),),
            edges=(),
            receipt={"provider_attempt_number": 1},
        )
        assert persist_calls == []
        assert sealed.graph_effect_attempted is False
        return SimpleNamespace(episode=None, nodes=(), edges=())

    monkeypatch.setattr(
        real_adapter, "_add_episode", leftover_new_nodes_without_edges
    )
    consume_calls: list[str] = []
    real_consume = disposition._consume_issue_790_event

    def consume_13677(**values: object) -> object:
        consume_calls.append(str(values["event_id"]))
        assert values["event_id"] == EVENT_13677
        values.setdefault("clock", lambda: OBSERVED_AT)
        values.setdefault(
            "graphiti",
            EvaluationGraphitiRunner(
                clock=lambda: OBSERVED_AT,
                fallback_permitted=False,
            ),
        )
        return real_consume(**values)

    monkeypatch.setattr(disposition, "_consume_issue_790_event", consume_13677)
    prepared = _prepare_production(
        stores,
        plan=plan,
        event_id=EVENT_13677,
        ledger_seq=LEDGER_13677,
    )
    backup = tmp_path / "zero-13677.sqlite3"
    receipt = disposition.run_issue_790_canary(
        store=stores.work_unpublished,
        proving_store=stores.proving,
        backup_path=backup,
        plan=plan,
        observed_at=OBSERVED_AT,
        repository_root=tmp_path,
        event_id=EVENT_13677,
        ledger_seq=LEDGER_13677,
        disposition_digest=_PRODUCTION_DISPOSITION,
        prepared=prepared,
        github_api=activated["github"],
    )
    event = _sqlite_canary_event(stores.work_unpublished, ledger_seq=LEDGER_13677)
    after = receipt["event_after"]["event"]
    connection = sqlite3.connect(stores.work_unpublished)
    try:
        attempt_row = connection.execute(
            "SELECT outcome,receipt_json FROM unpublished_graphiti_attempt_receipts"
        ).fetchone()
        ingest = connection.execute(
            "SELECT outcome,proposal_count,entity_count,relation_count "
            "FROM unpublished_graphiti_ingest"
        ).fetchone()
        spent_13665 = connection.execute(
            "SELECT state,attempt_count,provider_dispatched FROM "
            "unpublished_graphiti_revision_events WHERE ledger_seq=?",
            (CANDIDATE_LEDGER_SEQ,),
        ).fetchone()
        spent_13671 = connection.execute(
            "SELECT state,attempt_count,provider_dispatched FROM "
            "unpublished_graphiti_revision_events WHERE ledger_seq=?",
            (SUCCESSOR_LEDGER_SEQ,),
        ).fetchone()
    finally:
        connection.close()
    assert attempt_row is not None
    attempt = json.loads(attempt_row[1])
    execution = executions[0]
    raw = execution.produced.raw_output_value
    combined = None if not isinstance(raw, dict) else raw.get("combined_temporal_receipt")
    unused = unused_queued_attempt_zero_candidates(stores.work_unpublished, plan)
    assert persist_calls == []
    assert consume_calls == [EVENT_13677]
    assert receipt["consumption"]["event_id"] == EVENT_13677
    assert receipt["consumption"]["ledger_seq"] == LEDGER_13677
    assert receipt["exception"] is None
    assert receipt["resumed_zero_io_finalisation"] is False
    assert receipt["provider_dispatch_attempted_this_run"] is True
    assert receipt["publication_performed"] is False
    assert receipt["retry_authorised"] is False
    assert receipt["process_result"]["state"] == "TERMINAL"
    assert receipt["process_result"]["attempt_count"] == 1
    assert after["state"] == "TERMINAL"
    assert after["attempt_count"] == 1
    assert after["last_failure_code"] is None
    assert event["state"] == "TERMINAL"
    assert event["attempt_count"] == 1
    assert event["provider_dispatched"] == 1
    assert type(event["provider_dispatched"]) is int
    assert ingest == ("COMPLETE", 0, 0, 0)
    assert attempt["outcome"] == "COMPLETE"
    assert attempt["proposal_count"] == 0
    assert attempt.get("entity_count") == 0
    assert attempt.get("relation_count") == 0
    assert execution.outcome.value == "COMPLETE"
    assert execution.produced.outcome.value == "SUCCESS"
    assert isinstance(combined, dict)
    assert combined["zero_proposal_effect"] == "EXPLICIT"
    assert attempt["embedding_usage"]["request_count"] == 4
    assert attempt["chat_invocations"][0]["usage"]["total_tokens"] == 8_446
    assert spent_13665 == ("CONFIGURATION_HELD", 1, 1)
    assert spent_13671 == ("CONFIGURATION_HELD", 1, 0)
    assert EVENT_13677 not in {item[0] for item in unused}
    assert LEDGER_13677 not in {item[1] for item in unused}
    with pytest.raises(PreparedCanaryError) as spent_13665_target:
        _candidate_from_plan(
            plan,
            event_id=CANDIDATE_EVENT_ID,
            ledger_seq=CANDIDATE_LEDGER_SEQ,
            role="canary",
            store=stores.work_unpublished,
        )
    assert spent_13665_target.value.failure_code == BOUNDED_CANARY_AUTHORITY_CONSUMED
    with pytest.raises(PreparedCanaryError) as spent_13671_target:
        _candidate_from_plan(
            plan,
            event_id=SUCCESSOR_EVENT_ID,
            ledger_seq=SUCCESSOR_LEDGER_SEQ,
            role="canary",
            store=stores.work_unpublished,
        )
    assert spent_13671_target.value.failure_code == BOUNDED_CANARY_AUTHORITY_CONSUMED
    resumed = disposition.run_issue_790_canary(
        store=stores.work_unpublished,
        proving_store=stores.proving,
        backup_path=backup,
        plan=plan,
        observed_at=OBSERVED_AT,
        repository_root=tmp_path,
        event_id=EVENT_13677,
        ledger_seq=LEDGER_13677,
        disposition_digest=_PRODUCTION_DISPOSITION,
        github_api=activated["github"],
    )
    assert resumed["resumed_zero_io_finalisation"] is True
    assert resumed["provider_dispatch_attempted_this_run"] is False
    assert consume_calls == [EVENT_13677]
    assert persist_calls == []


def test_step22_consumed_13683_unmarked_zero_after_embeddings_survives_full_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Live 13683: persistable leftover NEW + float fact_embedding is TERMINAL.

    Provider COMPLETE + embeddings + persistable edges bind float
    fact_embedding for persistence. The derivative vector is omitted from the
    canonical receipt without erasing the accepted relation.
    """

    from contextlib import asynccontextmanager
    from datetime import UTC, datetime
    from types import SimpleNamespace

    from newsroom.authority.types import UtcTimestamp
    from newsroom.control_plane.corpus import CorpusIngestUnit
    from newsroom.control_plane.graphiti import EvaluationGraphitiRunner
    from newsroom.graphiti_adapter import real as real_adapter
    from newsroom.graphiti_adapter.combined_temporal_fixtures import (
        PAIR_BODY,
        _pair_entities,
        _pair_fact,
    )
    from newsroom.graphiti_adapter.evaluation_packet import (
        CURSOR_AGENT_MODEL_ID,
        GRAPHITI_WORKSPACE_GROUP,
    )
    from newsroom.graphiti_adapter.neo4j_guard import (
        GuardError,
        GuardMarker,
        GuardState,
    )

    persist_calls: list[str] = []
    guard_calls: list[str] = []
    guards: list[object] = []
    executions: list[object] = []
    existing_nodes: list[object] = []
    stores = build_rehearsal_stores(tmp_path, unused_13683=True)
    activated = _activate_step22(stores.work_unpublished)
    plan = activated["plan"]
    _bind_qualified_candidate(
        monkeypatch, event_id=EVENT_13683, ledger_seq=LEDGER_13683
    )
    _seed_production_disposition(stores.work_unpublished, plan)
    _patch_production_predispatch(monkeypatch, plan=plan)
    monkeypatch.setattr(
        EvaluationGraphitiRunner,
        "requires_canonical_control_plane_stores",
        False,
    )
    monkeypatch.setattr(CorpusIngestUnit, "episode_body", PAIR_BODY)
    adapter_type = real_adapter.RealGraphitiAdapter

    class ClockedAdapter:
        def __init__(self, **values: object) -> None:
            self.delegate = adapter_type(
                clock=lambda: UtcTimestamp(OBSERVED_AT),
                **values,
            )

        def execute(self, **values: object) -> object:
            execution = self.delegate.execute(**values)
            executions.append(execution)
            return execution

    class Missing(Exception):
        pass

    retained: dict[str, object] = {}

    class Episode:
        def __init__(self, **values: object) -> None:
            self.__dict__.update(values)
            self.entity_edges: list[object] = []

        @classmethod
        async def get_by_uuid(cls, _driver: object, episode_id: str) -> object:
            if episode_id not in retained:
                raise Missing(episode_id)
            return retained[episode_id]

        async def save(self, _driver: object) -> None:
            retained[str(self.uuid)] = self

    class EntityNode:
        def __init__(self, **values: object) -> None:
            self.name_embedding = None
            self.__dict__.update(values)

        @classmethod
        async def get_by_group_ids(
            cls, _driver: object, _groups: object, with_embeddings: bool = False
        ) -> list[object]:
            del with_embeddings
            return list(existing_nodes)

    class EntityEdge:
        def __init__(self, **values: object) -> None:
            self.__dict__.update(values)

    class Driver:
        _database = "neo4j"

        def clone(self, **_values: object) -> object:
            raise AssertionError("must not clone driver")

    class Graphiti:
        def __init__(self, *_args: object, **values: object) -> None:
            self.driver = Driver()
            self.clients = SimpleNamespace(
                driver=self.driver,
                llm_client=values["llm_client"],
                embedder=values["embedder"],
            )

        async def retrieve_episodes(self, *_a: object, **_k: object) -> list[object]:
            return []

        async def _process_episode_data(self, *_a: object, **_k: object) -> None:
            persist_calls.append("process")

        async def close(self) -> None:
            return None

    class Guard:
        def __init__(self, *_a: object, **values: object) -> None:
            self.driver = _a[0] if _a else None
            self.group_id = values.get("group_id")
            self.episode_uuid = values.get("episode_uuid")
            self.input_digest = values.get("input_digest")
            self.state = GuardState.CREATED
            guards.append(self)

        async def begin(self) -> object:
            marker = GuardMarker(
                state=GuardState.CREATED,
                attempt_number=1,
                input_digest=str(self.input_digest or "sha256:" + "0" * 64),
            )
            self.state = GuardState.PENDING
            return marker

        def require_pending(self, operation: str) -> None:
            if self.state is not GuardState.PENDING:
                raise GuardError(f"Graphiti {operation} lost its pending claim")

        async def record_pending_telemetry(self, **_values: object) -> None:
            self.require_pending("telemetry")
            guard_calls.append("telemetry")

        async def complete(self, _receipt: object) -> None:
            self.require_pending("completion")
            self.state = GuardState.COMPLETE
            guard_calls.append("complete")

        async def rollback_pending(self, **_values: object) -> bool:
            self.require_pending("rollback")
            self.state = GuardState.RECOVERED_AMBIGUOUS
            guard_calls.append("rollback")
            return True

        async def restore_preexisting(self) -> None:
            return None

        @asynccontextmanager
        async def fenced_graph_mutation(self):
            yield

    class OpenAIEmbedder:
        def __init__(self, config: object) -> None:
            self.config = config
            if not hasattr(self.config, "embedding_dim"):
                self.config.embedding_dim = 2
            if not hasattr(self.config, "embedding_model"):
                self.config.embedding_model = "openai/text-embedding-3-large"
            self.client = SimpleNamespace(
                embeddings=SimpleNamespace(create=self._create)
            )

        async def _create(self, input=None, model=None, **_k: object) -> object:
            del model
            count = len(input) if isinstance(input, list) else 1
            return SimpleNamespace(
                id="emb-1",
                data=[
                    SimpleNamespace(embedding=[0.0, 1.0])
                    for _index in range(count)
                ],
                usage={
                    "prompt_tokens": 2,
                    "total_tokens": 2,
                    "cost": 0,
                },
            )

    class LlmClient:
        def __init__(
            self,
            invocation_observer=None,
            fallback_permitted=True,
            **_k: object,
        ) -> None:
            del fallback_permitted
            self.invocations: list[dict[str, object]] = []
            self._observer = invocation_observer

        async def _generate_response(self, *_a: object, **_k: object) -> object:
            if self._observer is not None:
                token = self._observer.before_cli_invocation(
                    provider="cursor-agent-cli",
                    model=CURSOR_AGENT_MODEL_ID,
                    prompt="live-13683 unmarked zero after embeddings",
                    schema=None,
                )
                self._observer.transport_dispatch_started(token)
                self._observer.after_cli_invocation(
                    token,
                    outcome="COMPLETE",
                    usage={
                        "usage_basis": "PROVIDER_REPORTED",
                        "input_tokens": 4_000,
                        "cached_read_tokens": 0,
                        "output_tokens": 2_374,
                        "total_tokens": 6_374,
                    },
                )
            self.invocations.append(
                {
                    "provider": "cursor-agent-cli",
                    "model": CURSOR_AGENT_MODEL_ID,
                    "outcome": "COMPLETE",
                    "usage": {
                        "usage_basis": "PROVIDER_REPORTED",
                        "input_tokens": 4_000,
                        "cached_read_tokens": 0,
                        "output_tokens": 2_374,
                        "total_tokens": 6_374,
                    },
                }
            )
            return {
                "entities": _pair_entities(),
                "facts": [_pair_fact(valid_at=None, invalid_at=None)],
            }

    async def create_entity_edge_embeddings(
        _embedder: object, edges: list[object]
    ) -> None:
        for edge in edges:
            edge.fact_embedding = [0.0, 1.0]

    def resolve_edge_pointers(
        edges: list[object], uuid_map: dict[str, str]
    ) -> list[object]:
        for edge in edges:
            edge.source_node_uuid = uuid_map.get(
                str(edge.source_node_uuid), edge.source_node_uuid
            )
            edge.target_node_uuid = uuid_map.get(
                str(edge.target_node_uuid), edge.target_node_uuid
            )
        return edges

    runtime = SimpleNamespace(
        Graphiti=Graphiti,
        OpenAIEmbedder=OpenAIEmbedder,
        OpenAIEmbedderConfig=lambda **values: SimpleNamespace(**values),
        MeteredOpenAIEmbedder=real_adapter.MeteredOpenAIEmbedder,
        IdentityCrossEncoder=lambda: object(),
        EpisodeType=SimpleNamespace(text="text"),
        EpisodicNode=Episode,
        EntityNode=EntityNode,
        EntityEdge=EntityEdge,
        NodeNotFoundError=Missing,
        MutationGuard=Guard,
        create_entity_edge_embeddings=create_entity_edge_embeddings,
        resolve_edge_pointers=resolve_edge_pointers,
    )
    original_factory = real_adapter.combined_temporal_pipeline_for

    def wrapped_factory(**values: object) -> object:
        source_id = str(values["source_id"])
        existing_nodes[:] = [
            SimpleNamespace(
                uuid=f"leftover-{index}",
                name=f"Leftover workspace entity {index}",
                group_id=GRAPHITI_WORKSPACE_GROUP,
                labels=["Entity"],
                created_at=datetime(2026, 8, 31, tzinfo=UTC),
                summary="",
                attributes={
                    "entity_type_id": 0,
                    "permitted_source_ids": (source_id,),
                    "source_id": source_id,
                },
                name_embedding=[1.0, 0.0],
            )
            for index in (1, 2)
        ]
        return original_factory(**values)

    monkeypatch.setattr(real_adapter, "RealGraphitiAdapter", ClockedAdapter)
    monkeypatch.setattr(real_adapter, "_load_graphiti", lambda: runtime)
    monkeypatch.setattr(real_adapter, "openrouter_api_key", lambda: "fixture-key")
    monkeypatch.setattr(
        real_adapter, "neo4j_community_password", lambda: "fixture-password"
    )
    monkeypatch.setattr(real_adapter, "build_cli_llm_client", LlmClient)
    monkeypatch.setattr(real_adapter, "combined_temporal_pipeline_for", wrapped_factory)

    consume_calls: list[str] = []
    real_consume = disposition._consume_issue_790_event

    def consume_13683(**values: object) -> object:
        consume_calls.append(str(values["event_id"]))
        assert values["event_id"] == EVENT_13683
        values.setdefault("clock", lambda: OBSERVED_AT)
        values.setdefault(
            "graphiti",
            EvaluationGraphitiRunner(
                clock=lambda: OBSERVED_AT,
                fallback_permitted=False,
            ),
        )
        return real_consume(**values)

    monkeypatch.setattr(disposition, "_consume_issue_790_event", consume_13683)
    prepared = _prepare_production(
        stores,
        plan=plan,
        event_id=EVENT_13683,
        ledger_seq=LEDGER_13683,
    )
    backup = tmp_path / "zero-13683.sqlite3"
    receipt = disposition.run_issue_790_canary(
        store=stores.work_unpublished,
        proving_store=stores.proving,
        backup_path=backup,
        plan=plan,
        observed_at=OBSERVED_AT,
        repository_root=tmp_path,
        event_id=EVENT_13683,
        ledger_seq=LEDGER_13683,
        disposition_digest=_PRODUCTION_DISPOSITION,
        prepared=prepared,
        github_api=activated["github"],
    )
    event = _sqlite_canary_event(stores.work_unpublished, ledger_seq=LEDGER_13683)
    after = receipt["event_after"]["event"]
    connection = sqlite3.connect(stores.work_unpublished)
    try:
        attempt_row = connection.execute(
            "SELECT outcome,receipt_json FROM unpublished_graphiti_attempt_receipts"
        ).fetchone()
        ingest = connection.execute(
            "SELECT outcome,proposal_count,entity_count,relation_count "
            "FROM unpublished_graphiti_ingest"
        ).fetchone()
        spent_13665 = connection.execute(
            "SELECT state,attempt_count,provider_dispatched FROM "
            "unpublished_graphiti_revision_events WHERE ledger_seq=?",
            (CANDIDATE_LEDGER_SEQ,),
        ).fetchone()
        spent_13671 = connection.execute(
            "SELECT state,attempt_count,provider_dispatched FROM "
            "unpublished_graphiti_revision_events WHERE ledger_seq=?",
            (SUCCESSOR_LEDGER_SEQ,),
        ).fetchone()
        spent_13677 = connection.execute(
            "SELECT state,attempt_count,provider_dispatched FROM "
            "unpublished_graphiti_revision_events WHERE ledger_seq=?",
            (LEDGER_13677,),
        ).fetchone()
    finally:
        connection.close()
    assert attempt_row is not None
    attempt = json.loads(attempt_row[1])
    execution = executions[0]
    raw = execution.produced.raw_output_value
    combined = None if not isinstance(raw, dict) else raw.get("combined_temporal_receipt")
    unused = unused_queued_attempt_zero_candidates(stores.work_unpublished, plan)
    assert persist_calls == ["process"]
    assert guard_calls == ["telemetry", "complete"]
    assert getattr(guards[-1], "state") is GuardState.COMPLETE
    assert consume_calls == [EVENT_13683]
    assert receipt["consumption"]["event_id"] == EVENT_13683
    assert receipt["consumption"]["ledger_seq"] == LEDGER_13683
    assert receipt["exception"] is None
    assert receipt["resumed_zero_io_finalisation"] is False
    assert receipt["provider_dispatch_attempted_this_run"] is True
    assert receipt["publication_performed"] is False
    assert receipt["retry_authorised"] is False
    assert receipt["process_result"]["state"] == "TERMINAL"
    assert receipt["process_result"]["attempt_count"] == 1
    assert after["state"] == "TERMINAL"
    assert after["attempt_count"] == 1
    assert after["last_failure_code"] is None
    assert event["state"] == "TERMINAL"
    assert event["attempt_count"] == 1
    assert event["provider_dispatched"] == 1
    assert type(event["provider_dispatched"]) is int
    assert ingest == ("COMPLETE", 2, 2, 1)
    assert attempt["outcome"] == "COMPLETE"
    assert attempt["proposal_count"] == 2
    assert attempt.get("entity_count") == 2
    assert attempt.get("relation_count") == 1
    assert execution.outcome.value == "COMPLETE"
    assert execution.produced.outcome.value == "SUCCESS"
    assert len(raw["proposals"]) == 2
    assert raw["relations"][0]["proposal_status"] == "HELD_NO_EXACT_EVIDENCE"
    assert isinstance(combined, dict)
    relation = combined["proposal_receipt"]["relation_proposals"][0]
    assert relation["proposal_status"] == "PROPOSED"
    assert "fact_embedding" not in relation
    assert "zero_proposal_effect" not in combined
    assert attempt["chat_invocations"][0]["usage"]["total_tokens"] == 6_374
    assert spent_13665 == ("CONFIGURATION_HELD", 1, 1)
    assert spent_13671 == ("CONFIGURATION_HELD", 1, 0)
    assert spent_13677 == ("CONFIGURATION_HELD", 1, 1)
    assert EVENT_13683 not in {item[0] for item in unused}
    assert LEDGER_13683 not in {item[1] for item in unused}
    with pytest.raises(PreparedCanaryError) as spent_13665_target:
        _candidate_from_plan(
            plan,
            event_id=CANDIDATE_EVENT_ID,
            ledger_seq=CANDIDATE_LEDGER_SEQ,
            role="canary",
            store=stores.work_unpublished,
        )
    assert spent_13665_target.value.failure_code == BOUNDED_CANARY_AUTHORITY_CONSUMED
    with pytest.raises(PreparedCanaryError) as spent_13671_target:
        _candidate_from_plan(
            plan,
            event_id=SUCCESSOR_EVENT_ID,
            ledger_seq=SUCCESSOR_LEDGER_SEQ,
            role="canary",
            store=stores.work_unpublished,
        )
    assert spent_13671_target.value.failure_code == BOUNDED_CANARY_AUTHORITY_CONSUMED
    with pytest.raises(PreparedCanaryError) as spent_13677_target:
        _candidate_from_plan(
            plan,
            event_id=EVENT_13677,
            ledger_seq=LEDGER_13677,
            role="canary",
            store=stores.work_unpublished,
        )
    assert spent_13677_target.value.failure_code == BOUNDED_CANARY_AUTHORITY_CONSUMED
    resumed = disposition.run_issue_790_canary(
        store=stores.work_unpublished,
        proving_store=stores.proving,
        backup_path=backup,
        plan=plan,
        observed_at=OBSERVED_AT,
        repository_root=tmp_path,
        event_id=EVENT_13683,
        ledger_seq=LEDGER_13683,
        disposition_digest=_PRODUCTION_DISPOSITION,
        github_api=activated["github"],
    )
    assert resumed["resumed_zero_io_finalisation"] is True
    assert resumed["provider_dispatch_attempted_this_run"] is False
    assert consume_calls == [EVENT_13683]


class _CanarySpawnAborted(BaseException):
    """Process-kill analogue: not Exception, so complete() does not run."""


def test_step22_aborted_spawn_13689_backup_dest_exists_does_not_strand_running(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Live 13689 abort + dest-exists: both halves of the demonstrated path.

    Watcher: PID 56093 was still running, then exited with no LIVE_EXIT and
    no receipt after writing backup and claiming RUNNING. Colliding spawn
    then EXIT 2 `canary backup destination already exists` /
    CANARY_BACKUP_DESTINATION_ALREADY_EXISTS while sqlite last_failure_code
    stayed NULL. Artefact `/Users/jamesto/issue-790-step22-live/20260831T211335Z`.
    Dispatcher recap 5484862486; watcher supplement 5484885253.

    Half 1: first spawn abort after backup + consume + RUNNING claim, no
    receipt, no provider I/O. Half 2: same dest colliding re-entry must not
    claim again and must not leave RUNNING with last_failure_code NULL.
    Dest exists without that strand still fail-closes before claim with the
    named code. Fails on fdef41da because dest-exists raised before recovery.
    """

    from newsroom.control_plane.graphiti_events import GraphitiEventQueue

    stores = build_rehearsal_stores(tmp_path, unused_13689=True)
    activated = _activate_step22(stores.work_unpublished)
    plan = activated["plan"]
    _bind_qualified_candidate(
        monkeypatch, event_id=EVENT_13689, ledger_seq=LEDGER_13689
    )
    _seed_production_disposition(stores.work_unpublished, plan)
    _patch_production_predispatch(monkeypatch, plan=plan)
    consume_calls: list[str] = []

    def abort_after_claim(**values: object) -> None:
        consume_calls.append(str(values["event_id"]))
        assert values["event_id"] == EVENT_13689
        queue = GraphitiEventQueue(
            str(values["unpublished_store"]),
            clock=lambda: OBSERVED_AT,
        )
        claimed = queue.claim(
            owner_id=str(values["owner_id"]),
            lease_for=timedelta(minutes=15),
            event_id=str(values["event_id"]),
            require_fresh=True,
            canary_consumption_digest=str(values["canary_consumption_digest"]),
        )
        assert claimed is not None
        assert queue._start(
            str(values["event_id"]),
            owner_id=str(values["owner_id"]),
        ) == 1
        raise _CanarySpawnAborted("canary spawn aborted after claim")

    monkeypatch.setattr(disposition, "_consume_issue_790_event", abort_after_claim)
    backup = tmp_path / "aborted-spawn-13689.sqlite3"
    prepared = _prepare_production(
        stores,
        plan=plan,
        event_id=EVENT_13689,
        ledger_seq=LEDGER_13689,
    )
    canary_kwargs = {
        "store": stores.work_unpublished,
        "proving_store": stores.proving,
        "backup_path": backup,
        "plan": plan,
        "observed_at": OBSERVED_AT,
        "repository_root": tmp_path,
        "event_id": EVENT_13689,
        "ledger_seq": LEDGER_13689,
        "disposition_digest": _PRODUCTION_DISPOSITION,
        "prepared": prepared,
        "github_api": activated["github"],
    }
    with pytest.raises(_CanarySpawnAborted):
        disposition.run_issue_790_canary(**canary_kwargs)
    stranded = _sqlite_canary_event(
        stores.work_unpublished, ledger_seq=LEDGER_13689
    )
    connection = sqlite3.connect(stores.work_unpublished)
    try:
        consumption_row = connection.execute(
            "SELECT consumption_digest,owner_id FROM "
            "issue_790_bounded_canary_consumptions WHERE ledger_seq=?",
            (LEDGER_13689,),
        ).fetchone()
        outcomes = connection.execute(
            "SELECT COUNT(*) FROM issue_790_bounded_canary_outcomes "
            "WHERE ledger_seq=?",
            (LEDGER_13689,),
        ).fetchone()[0]
        failure_code = connection.execute(
            "SELECT last_failure_code FROM unpublished_graphiti_revision_events "
            "WHERE ledger_seq=?",
            (LEDGER_13689,),
        ).fetchone()[0]
    finally:
        connection.close()
    abort_backup_digest = "sha256:" + file_digest(backup)
    assert backup.is_file()
    assert abort_backup_digest.startswith("sha256:")
    assert len(abort_backup_digest) == 71
    assert stranded["state"] == "RUNNING"
    assert stranded["attempt_count"] == 1
    assert stranded["provider_dispatched"] == 0
    assert type(stranded["provider_dispatched"]) is int
    assert stranded["claim_owner"] is not None
    assert str(stranded["claim_owner"]).startswith("issue-790-canary:")
    assert stranded["claim_expires_at"] is not None
    assert stranded["terminal_at"] is None
    assert consumption_row is not None
    assert str(consumption_row[0]).startswith("sha256:")
    assert consumption_row[1] == stranded["claim_owner"]
    assert outcomes == 0
    assert failure_code is None
    assert dispatch_started_count(stores.work_unpublished) == 0

    # The retained live 13689 consumption predates durable snapshot binding.
    legacy_consumption_digest = _remove_snapshot_binding_from_canary_consumption(
        stores.work_unpublished,
        ledger_seq=LEDGER_13689,
    )

    _expire_interrupted_canary_claim(
        stores.work_unpublished, ledger_seq=LEDGER_13689
    )
    canary_kwargs["prepared"] = None
    canary_kwargs["recover_interrupted"] = True
    expected_backup_digest = "sha256:" + file_digest(backup)
    canary_kwargs["expected_backup_digest"] = expected_backup_digest
    receipt = disposition.run_issue_790_canary(**canary_kwargs)
    sealed = _sqlite_canary_event(
        stores.work_unpublished, ledger_seq=LEDGER_13689
    )
    connection = sqlite3.connect(stores.work_unpublished)
    try:
        failure_code = connection.execute(
            "SELECT last_failure_code FROM unpublished_graphiti_revision_events "
            "WHERE ledger_seq=?",
            (LEDGER_13689,),
        ).fetchone()[0]
        outcomes = connection.execute(
            "SELECT COUNT(*) FROM issue_790_bounded_canary_outcomes "
            "WHERE ledger_seq=?",
            (LEDGER_13689,),
        ).fetchone()[0]
    finally:
        connection.close()
    unused = unused_queued_attempt_zero_candidates(stores.work_unpublished, plan)
    after = receipt["event_after"]["event"]
    assert consume_calls == [EVENT_13689]
    assert receipt["resumed_zero_io_finalisation"] is True
    assert receipt["provider_dispatch_attempted_this_run"] is False
    assert receipt["canary_evidence_passed"] is False
    assert receipt["retry_authorised"] is False
    assert receipt["publication_performed"] is False
    assert receipt["consumption"]["event_id"] == EVENT_13689
    assert receipt["consumption"]["ledger_seq"] == LEDGER_13689
    assert receipt["consumption"]["consumption_digest"] == (
        legacy_consumption_digest
    )
    assert receipt["pre_operation_snapshot_digest"] == expected_backup_digest
    assert receipt["pre_operation_snapshot_digest"] == abort_backup_digest
    assert "pre_operation_snapshot_digest" not in receipt["consumption"][
        "preflight_evidence"
    ]
    assert receipt["prepared_canary_decision_digest"] is None
    assert receipt["prepared_canary_record_digest"] is None
    assert receipt["interrupted_recovery_evidence"] == {
        "checked_at": receipt["interrupted_recovery_evidence"]["checked_at"],
        "expected_backup_digest": expected_backup_digest,
        "other_canary_process_ids": [],
        "single_executor_lock_held": True,
        "verified_backup_digest": expected_backup_digest,
    }
    assert receipt["outcome"]["failure_code_after_seal"] == (
        "BOUNDED_CANARY_AUTHORITY_EXHAUSTED:NO_EVENT_RESULT"
    )
    assert receipt["outcome"]["state_after_seal"] == "CONFIGURATION_HELD"
    assert after["state"] == "CONFIGURATION_HELD"
    assert after["attempt_count"] == 1
    assert sealed["state"] == "CONFIGURATION_HELD"
    assert sealed["attempt_count"] == 1
    assert sealed["provider_dispatched"] == 0
    assert type(sealed["provider_dispatched"]) is int
    assert sealed["claim_owner"] is None
    assert failure_code == "BOUNDED_CANARY_AUTHORITY_EXHAUSTED:NO_EVENT_RESULT"
    assert outcomes == 1
    assert dispatch_started_count(stores.work_unpublished) == 0
    assert EVENT_13689 not in {item[0] for item in unused}
    assert LEDGER_13689 not in {item[1] for item in unused}
    with pytest.raises(PreparedCanaryError) as caught:
        _candidate_from_plan(
            plan,
            event_id=CANDIDATE_EVENT_ID,
            ledger_seq=CANDIDATE_LEDGER_SEQ,
            role="canary",
            store=stores.work_unpublished,
        )
    assert caught.value.failure_code == BOUNDED_CANARY_AUTHORITY_CONSUMED
    canary_kwargs.pop("recover_interrupted")
    canary_kwargs.pop("expected_backup_digest")
    with pytest.raises(
        Issue790DispositionError,
        match="legacy canary replay requires the expected backup digest",
    ):
        disposition.run_issue_790_canary(**canary_kwargs)
    wrong_backup = tmp_path / "valid-but-unbound-completed-replay.sqlite3"
    disposition._sqlite_backup(backup, wrong_backup)
    connection = sqlite3.connect(wrong_backup)
    try:
        connection.execute("PRAGMA user_version=1")
        connection.commit()
    finally:
        connection.close()
    canary_kwargs["backup_path"] = wrong_backup
    canary_kwargs["expected_backup_digest"] = expected_backup_digest
    with pytest.raises(Issue790DispositionError, match="canary backup digest differs"):
        disposition.run_issue_790_canary(**canary_kwargs)
    canary_kwargs["backup_path"] = backup
    replayed = disposition.run_issue_790_canary(**canary_kwargs)
    still = _sqlite_canary_event(
        stores.work_unpublished, ledger_seq=LEDGER_13689
    )
    assert still["state"] == "CONFIGURATION_HELD"
    assert still["attempt_count"] == 1
    assert still["provider_dispatched"] == 0
    assert still["claim_owner"] is None
    assert consume_calls == [EVENT_13689]
    assert "sha256:" + file_digest(backup) == abort_backup_digest
    assert replayed["resumed_zero_io_finalisation"] is True
    assert replayed["provider_dispatch_attempted_this_run"] is False
    assert replayed["pre_operation_snapshot_digest"] == expected_backup_digest
    assert "pre_operation_snapshot_digest" not in replayed["consumption"][
        "preflight_evidence"
    ]


def test_step22_backup_dest_exists_without_consumption_fail_closes_before_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same dest with unused QUEUED and no consumption must not claim."""

    stores = build_rehearsal_stores(tmp_path, unused_13689=True)
    activated = _activate_step22(stores.work_unpublished)
    plan = activated["plan"]
    _bind_qualified_candidate(
        monkeypatch, event_id=EVENT_13689, ledger_seq=LEDGER_13689
    )
    _seed_production_disposition(stores.work_unpublished, plan)
    _patch_production_predispatch(monkeypatch, plan=plan)
    consume_calls: list[str] = []

    def must_not_consume(**_values: object) -> None:
        consume_calls.append("called")
        raise AssertionError("dest-exists without consumption claimed")

    monkeypatch.setattr(disposition, "_consume_issue_790_event", must_not_consume)
    backup = tmp_path / "leftover-dest-13689.sqlite3"
    backup.write_bytes(b"leftover")
    prepared = _prepare_production(
        stores,
        plan=plan,
        event_id=EVENT_13689,
        ledger_seq=LEDGER_13689,
    )
    with pytest.raises(Issue790DispositionError) as dest_exists:
        disposition.run_issue_790_canary(
            store=stores.work_unpublished,
            proving_store=stores.proving,
            backup_path=backup,
            plan=plan,
            observed_at=OBSERVED_AT,
            repository_root=tmp_path,
            event_id=EVENT_13689,
            ledger_seq=LEDGER_13689,
            disposition_digest=_PRODUCTION_DISPOSITION,
            prepared=prepared,
            github_api=activated["github"],
        )
    event = _sqlite_canary_event(
        stores.work_unpublished, ledger_seq=LEDGER_13689
    )
    assert str(dest_exists.value) == "canary backup destination already exists"
    assert dest_exists.value.failure_code == (
        CANARY_BACKUP_DESTINATION_ALREADY_EXISTS
    )
    assert consume_calls == []
    assert event["state"] == "QUEUED"
    assert event["attempt_count"] == 0
    assert event["provider_dispatched"] == 0
    assert event["claim_owner"] is None
    assert dispatch_started_count(stores.work_unpublished) == 0


def test_step22_fresh_missing_prepared_fails_before_any_effect(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fresh production claim requires the cross-process prepared artefact."""

    stores = build_rehearsal_stores(tmp_path, unused_13689=True)
    activated = _activate_step22(stores.work_unpublished)
    plan = activated["plan"]
    _bind_qualified_candidate(
        monkeypatch, event_id=EVENT_13689, ledger_seq=LEDGER_13689
    )
    _seed_production_disposition(stores.work_unpublished, plan)
    _patch_production_predispatch(monkeypatch, plan=plan)
    consume_calls: list[str] = []

    def must_not_consume(**_values: object) -> None:
        consume_calls.append("called")
        raise AssertionError("missing prepared artefact reached consumption")

    monkeypatch.setattr(disposition, "_consume_issue_790_event", must_not_consume)
    backup = tmp_path / "missing-prepared-must-not-exist.sqlite3"
    with pytest.raises(PreparedCanaryError) as caught:
        disposition.run_issue_790_canary(
            store=stores.work_unpublished,
            proving_store=stores.proving,
            backup_path=backup,
            plan=plan,
            observed_at=OBSERVED_AT,
            repository_root=tmp_path,
            event_id=EVENT_13689,
            ledger_seq=LEDGER_13689,
            disposition_digest=_PRODUCTION_DISPOSITION,
            github_api=activated["github"],
        )
    event = _sqlite_canary_event(
        stores.work_unpublished, ledger_seq=LEDGER_13689
    )
    connection = sqlite3.connect(stores.work_unpublished)
    try:
        consumptions = connection.execute(
            "SELECT COUNT(*) FROM issue_790_bounded_canary_consumptions "
            "WHERE ledger_seq=?",
            (LEDGER_13689,),
        ).fetchone()[0]
        outcomes = connection.execute(
            "SELECT COUNT(*) FROM issue_790_bounded_canary_outcomes "
            "WHERE ledger_seq=?",
            (LEDGER_13689,),
        ).fetchone()[0]
    finally:
        connection.close()
    assert caught.value.failure_code == PREPARED_CANARY_ABSENT
    assert backup.exists() is False
    assert consume_calls == []
    assert consumptions == 0
    assert outcomes == 0
    assert event["state"] == "QUEUED"
    assert event["attempt_count"] == 0
    assert event["provider_dispatched"] == 0
    assert event["claim_owner"] is None
    assert dispatch_started_count(stores.work_unpublished) == 0


def test_step22_fresh_in_memory_prepared_fails_before_any_effect(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A caller cannot substitute an unpersisted in-memory decision."""

    stores = build_rehearsal_stores(tmp_path, unused_13689=True)
    activated = _activate_step22(stores.work_unpublished)
    plan = activated["plan"]
    _bind_qualified_candidate(
        monkeypatch, event_id=EVENT_13689, ledger_seq=LEDGER_13689
    )
    _seed_production_disposition(stores.work_unpublished, plan)
    prepared = prepare_issue_790_canary(
        store=stores.work_unpublished,
        proving_store=stores.proving,
        plan=plan,
        observed_at=OBSERVED_AT,
        exact_head=EXACT_HEAD,
        event_id=EVENT_13689,
        ledger_seq=LEDGER_13689,
        role="canary",
    )
    backup = tmp_path / "in-memory-prepared-must-not-exist.sqlite3"
    with pytest.raises(PreparedCanaryError) as caught:
        disposition.run_issue_790_canary(
            store=stores.work_unpublished,
            proving_store=stores.proving,
            backup_path=backup,
            plan=plan,
            observed_at=OBSERVED_AT,
            repository_root=tmp_path,
            event_id=EVENT_13689,
            ledger_seq=LEDGER_13689,
            disposition_digest=_PRODUCTION_DISPOSITION,
            prepared=prepared,
            github_api=activated["github"],
        )
    assert caught.value.failure_code == PREPARED_CANARY_RECORD_INVALID
    assert backup.exists() is False
    assert dispatch_started_count(stores.work_unpublished) == 0


def test_step22_fresh_recovery_flag_fails_before_any_effect(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recovery authority never converts a fresh candidate into a recovery."""

    stores = build_rehearsal_stores(tmp_path, unused_13689=True)
    activated = _activate_step22(stores.work_unpublished)
    plan = activated["plan"]
    _bind_qualified_candidate(
        monkeypatch, event_id=EVENT_13689, ledger_seq=LEDGER_13689
    )
    _seed_production_disposition(stores.work_unpublished, plan)
    _patch_production_predispatch(monkeypatch, plan=plan)
    prepared = _prepare_production(
        stores,
        plan=plan,
        event_id=EVENT_13689,
        ledger_seq=LEDGER_13689,
    )
    backup = tmp_path / "fresh-recovery-must-not-exist.sqlite3"
    with pytest.raises(
        Issue790DispositionError,
        match="interrupted canary recovery authority is absent",
    ):
        disposition.run_issue_790_canary(
            store=stores.work_unpublished,
            proving_store=stores.proving,
            backup_path=backup,
            plan=plan,
            observed_at=OBSERVED_AT,
            repository_root=tmp_path,
            event_id=EVENT_13689,
            ledger_seq=LEDGER_13689,
            disposition_digest=_PRODUCTION_DISPOSITION,
            prepared=prepared,
            recover_interrupted=True,
            expected_backup_digest="sha256:" + "00" * 32,
            github_api=activated["github"],
        )
    event = _sqlite_canary_event(
        stores.work_unpublished, ledger_seq=LEDGER_13689
    )
    assert backup.exists() is False
    assert event["state"] == "QUEUED"
    assert event["attempt_count"] == 0
    assert event["provider_dispatched"] == 0
    assert event["claim_owner"] is None
    assert dispatch_started_count(stores.work_unpublished) == 0


def test_step22_full_executor_rehearsal_reaches_fixture_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The zero-provider runner traverses the full public production wrapper."""

    stores = build_rehearsal_stores(tmp_path, unused_13689=True)
    activated = _activate_step22(stores.work_unpublished)
    plan = activated["plan"]
    _bind_qualified_candidate(
        monkeypatch, event_id=EVENT_13689, ledger_seq=LEDGER_13689
    )
    _seed_production_disposition(stores.work_unpublished, plan)
    _patch_production_predispatch(monkeypatch, plan=plan)
    prepared = _prepare_production(
        stores,
        plan=plan,
        event_id=EVENT_13689,
        ledger_seq=LEDGER_13689,
    )
    RehearsalRealGraphitiAdapter.provider_calls = 0
    RehearsalRealGraphitiAdapter.dispatch_started = False
    backup = tmp_path / "full-executor-rehearsal.sqlite3"
    receipt = disposition.run_issue_790_canary(
        store=stores.work_unpublished,
        proving_store=stores.proving,
        backup_path=backup,
        plan=plan,
        observed_at=OBSERVED_AT,
        repository_root=tmp_path,
        event_id=EVENT_13689,
        ledger_seq=LEDGER_13689,
        disposition_digest=_PRODUCTION_DISPOSITION,
        prepared=prepared,
        graphiti=RehearsalEvaluationGraphitiRunner(clock=lambda: OBSERVED_AT),
        github_api=activated["github"],
    )
    connection = sqlite3.connect(stores.work_unpublished)
    try:
        consumptions = connection.execute(
            "SELECT COUNT(*) FROM issue_790_bounded_canary_consumptions "
            "WHERE ledger_seq=?",
            (LEDGER_13689,),
        ).fetchone()[0]
    finally:
        connection.close()
    assert backup.is_file()
    assert consumptions == 1
    assert receipt["pre_operation_snapshot_digest"] == (
        receipt["consumption"]["preflight_evidence"][
            "pre_operation_snapshot_digest"
        ]
    )
    assert receipt["prepared_canary_decision_digest"] == prepared.decision_digest
    assert receipt["prepared_canary_record_digest"] == prepared.record_digest
    assert RehearsalRealGraphitiAdapter.dispatch_started is True
    assert RehearsalRealGraphitiAdapter.provider_calls == 0
    assert dispatch_started_count(stores.work_unpublished) >= 1
    assert receipt["provider_dispatch_attempted_this_run"] is True
    assert receipt["exception"] is None
    assert receipt["process_result"]["state"] == "TERMINAL"
    assert receipt["outcome"]["result_class"] == "TRUTHFUL_PROVIDER_SUCCESS"
    assert receipt["canary_evidence_passed"] is True
    assert receipt["publication_performed"] is False


def test_step22_complete_preflight_emits_prepared_bound_live_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """READY emits the sole live command with its exact PreparedCanary path."""

    import scripts.issue_790_live_canary_preflight as preflight

    ops_source = inspect.getsource(preflight._ops_gates)
    digest_drift_message = (
        "ledger 13690 prepared canary decision digest differs: "
        "preflight must prepare from the activated plan consumed by the live command"
    )
    assert "activated_plan=activated_plan" in ops_source, digest_drift_message
    prepare_helper = getattr(preflight, "_prepare_preflight_canary", None)
    if prepare_helper is not None:
        assert "plan=activated_plan" in inspect.getsource(prepare_helper), (
            digest_drift_message
        )

    stores_root = tmp_path / "stores"
    stores_root.mkdir()
    stores = build_rehearsal_stores(stores_root, unused_13689=True)
    activated = _activate_step22(stores.work_unpublished)
    plan = activated["plan"]
    _bind_qualified_candidate(
        monkeypatch, event_id=EVENT_13689, ledger_seq=LEDGER_13689
    )
    prepared = preflight._prepare_preflight_canary(
        store=stores.work_unpublished,
        proving_store=stores.proving,
        activated_plan=plan,
        observed_at=OBSERVED_AT,
        exact_head=EXACT_HEAD,
    )
    activated_plan_path = tmp_path / "activated-plan.json"
    prepared_path = tmp_path / "prepared-canary.json"
    command_path = tmp_path / "live-canary.sh"
    monkeypatch.setattr(
        preflight,
        "_ops_gates",
        lambda **_kwargs: (
            [("O01", True, "ready")],
            (EVENT_13689, LEDGER_13689),
            prepared,
            plan,
        ),
    )
    monkeypatch.setattr(
        preflight,
        "_blocker_smokes",
        lambda _root: [("B01", True, "ready")],
    )

    result = preflight.main(
        [
            "--ops-root",
            str(_ROOT),
            "--code-root",
            str(_ROOT),
            "--tip-plan",
            str(prepared.plan_identity["pending_digest"]),
            "--plan-rel",
            str(ISSUE_790_STEP22_PENDING_PLAN_PATH),
            "--prepared-canary-out",
            str(prepared_path),
        ]
    )

    assert result == 0
    retained = prepared_canary_from_record(
        json.loads(prepared_path.read_text(encoding="utf-8"))
    )
    assert retained.decision_digest == prepared.decision_digest
    assert json.loads(activated_plan_path.read_text(encoding="utf-8")) == plan
    assert command_path.stat().st_mode & 0o777 == 0o700
    script = command_path.read_text(encoding="utf-8")
    assert "set -eu" in script
    assert f"cd {shlex.quote(str(_ROOT))}" in script
    exec_line = next(line for line in script.splitlines() if line.startswith("exec "))
    tokens = shlex.split(exec_line)

    def option(name: str) -> str:
        assert tokens.count(name) == 1
        return tokens[tokens.index(name) + 1]

    assert tokens[:5] == [
        "exec",
        str(Path(sys.executable).absolute()),
        "-m",
        "scripts.issue_790_conservative_disposition",
        "canary",
    ]
    assert option("--store") == str(
        _ROOT / "data/newsroom/unpublished_store.sqlite3"
    )
    assert option("--proving-store") == str(
        _ROOT / "data/newsroom/proving_store.sqlite3"
    )
    assert option("--plan") == str(activated_plan_path)
    assert option("--observed-at") == "$OBSERVED_AT"
    assert option("--receipt") == str(tmp_path / "canary-receipt.json")
    assert option("--backup") == str(
        tmp_path / "unpublished_store.pre-canary.sqlite3"
    )
    assert option("--repository-root") == str(_ROOT)
    assert option("--canary-event-id") == EVENT_13689
    assert option("--canary-ledger-seq") == str(LEDGER_13689)
    assert option("--disposition-digest") == preflight.DISP
    assert option("--prepared-canary") == str(prepared_path)
    assert (tmp_path / "canary-receipt.json").exists() is False
    assert (tmp_path / "unpublished_store.pre-canary.sqlite3").exists() is False

    with pytest.raises(Issue790DispositionError, match="canary command already exists"):
        preflight._write_live_canary_command(
            root=_ROOT,
            activated_plan=activated_plan_path,
            prepared_canary=prepared_path,
            command_out=command_path,
            event_id=EVENT_13689,
            ledger_seq=LEDGER_13689,
        )
    existing_receipt = tmp_path / "canary-receipt.json"
    existing_receipt.write_text("already used\n", encoding="utf-8")
    blocked_command = tmp_path / "blocked-live-canary.sh"
    with pytest.raises(Issue790DispositionError, match="canary receipt already exists"):
        preflight._write_live_canary_command(
            root=_ROOT,
            activated_plan=activated_plan_path,
            prepared_canary=prepared_path,
            command_out=blocked_command,
            event_id=EVENT_13689,
            ledger_seq=LEDGER_13689,
        )
    assert blocked_command.exists() is False

    RehearsalRealGraphitiAdapter.provider_calls = 0
    RehearsalRealGraphitiAdapter.dispatch_started = False
    rehearsal = run_prepared_canary_rehearsal(
        store=stores.work_unpublished,
        proving_store=stores.proving,
        plan=plan,
        observed_at=OBSERVED_AT,
        exact_head=EXACT_HEAD,
        prepared=retained,
        event_id=EVENT_13689,
        ledger_seq=LEDGER_13689,
    )
    assert rehearsal["decision_digest"] == retained.decision_digest
    assert rehearsal["dispatch_started"] is True
    assert rehearsal["provider_calls"] == 0


def test_fresh_canary_rejects_backup_replacement_before_consumption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The digest is bound to the backup fd, never a replaced path."""

    stores = build_rehearsal_stores(tmp_path, unused_13689=True)
    activated = _activate_step22(stores.work_unpublished)
    plan = activated["plan"]
    _bind_qualified_candidate(
        monkeypatch, event_id=EVENT_13689, ledger_seq=LEDGER_13689
    )
    _seed_production_disposition(stores.work_unpublished, plan)
    _patch_production_predispatch(monkeypatch, plan=plan)
    prepared = _prepare_production(
        stores,
        plan=plan,
        event_id=EVENT_13689,
        ledger_seq=LEDGER_13689,
    )
    replacement = tmp_path / "replacement-pristine.sqlite3"
    disposition._sqlite_backup(stores.work_unpublished, replacement)
    connection = sqlite3.connect(replacement)
    try:
        connection.execute("PRAGMA user_version=1")
        connection.commit()
    finally:
        connection.close()
    publish = disposition._publish_file_no_replace

    def publish_then_replace(temporary: Path, destination: Path) -> None:
        publish(temporary, destination)
        os.replace(replacement, destination)

    monkeypatch.setattr(
        disposition,
        "_publish_file_no_replace",
        publish_then_replace,
    )
    backup = tmp_path / "replaced-before-consumption.sqlite3"
    with pytest.raises(
        Issue790DispositionError,
        match="backup published identity changed",
    ):
        disposition.run_issue_790_canary(
            store=stores.work_unpublished,
            proving_store=stores.proving,
            backup_path=backup,
            plan=plan,
            observed_at=OBSERVED_AT,
            repository_root=tmp_path,
            event_id=EVENT_13689,
            ledger_seq=LEDGER_13689,
            disposition_digest=_PRODUCTION_DISPOSITION,
            prepared=prepared,
            github_api=activated["github"],
        )
    connection = sqlite3.connect(stores.work_unpublished)
    try:
        consumptions = connection.execute(
            "SELECT COUNT(*) FROM issue_790_bounded_canary_consumptions "
            "WHERE ledger_seq=?",
            (LEDGER_13689,),
        ).fetchone()[0]
        outcomes = connection.execute(
            "SELECT COUNT(*) FROM issue_790_bounded_canary_outcomes "
            "WHERE ledger_seq=?",
            (LEDGER_13689,),
        ).fetchone()[0]
    finally:
        connection.close()
    event = _sqlite_canary_event(
        stores.work_unpublished,
        ledger_seq=LEDGER_13689,
    )
    assert consumptions == 0
    assert outcomes == 0
    assert event["state"] == "QUEUED"
    assert event["attempt_count"] == 0
    assert dispatch_started_count(stores.work_unpublished) == 0


def test_step22_backup_replacement_after_execution_fails_before_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The receipt never claims retention after the checked snapshot is replaced."""

    stores = build_rehearsal_stores(tmp_path, unused_13689=True)
    activated = _activate_step22(stores.work_unpublished)
    plan = activated["plan"]
    _bind_qualified_candidate(
        monkeypatch, event_id=EVENT_13689, ledger_seq=LEDGER_13689
    )
    _seed_production_disposition(stores.work_unpublished, plan)
    _patch_production_predispatch(monkeypatch, plan=plan)
    prepared = _prepare_production(
        stores,
        plan=plan,
        event_id=EVENT_13689,
        ledger_seq=LEDGER_13689,
    )
    backup = tmp_path / "replaced-after-execution.sqlite3"
    replacement = tmp_path / "different-valid-snapshot.sqlite3"
    disposition._sqlite_backup(stores.work_unpublished, replacement)
    connection = sqlite3.connect(replacement)
    try:
        connection.execute("PRAGMA user_version=1")
        connection.commit()
    finally:
        connection.close()
    usage_evidence = disposition._issue_790_canary_usage_evidence
    replaced = False

    def replace_before_receipt(*args, **kwargs):
        nonlocal replaced
        if not replaced and backup.exists():
            replacement.replace(backup)
            replaced = True
        return usage_evidence(*args, **kwargs)

    monkeypatch.setattr(
        disposition,
        "_issue_790_canary_usage_evidence",
        replace_before_receipt,
    )
    with pytest.raises(Issue790DispositionError, match="canary backup digest differs"):
        disposition.run_issue_790_canary(
            store=stores.work_unpublished,
            proving_store=stores.proving,
            backup_path=backup,
            plan=plan,
            observed_at=OBSERVED_AT,
            repository_root=tmp_path,
            event_id=EVENT_13689,
            ledger_seq=LEDGER_13689,
            disposition_digest=_PRODUCTION_DISPOSITION,
            prepared=prepared,
            graphiti=RehearsalEvaluationGraphitiRunner(clock=lambda: OBSERVED_AT),
            github_api=activated["github"],
        )
    assert replaced is True


@pytest.mark.parametrize(
    ("mutation", "failure"),
    (
        (
            "corrupt_backup",
            "canary backup digest differs",
        ),
        (
            "claimed_backup_snapshot",
            "canary backup digest differs",
        ),
        (
            "missing_transport_table",
            "canary backup destination already exists",
        ),
        (
            "consumption_authority_drift",
            "canary backup destination already exists",
        ),
        (
            "prepared_binding_tamper",
            "canary consumption digest differs",
        ),
        (
            "active_claim",
            "canary backup destination already exists",
        ),
        (
            "ownerless_active_expiry",
            "canary backup destination already exists",
        ),
        (
            "missing_recovery_flag",
            "interrupted canary recovery requires explicit authority",
        ),
        (
            "missing_backup_digest",
            "interrupted canary recovery requires the expected backup digest",
        ),
        (
            "absent_recovery_backup",
            "interrupted canary backup destination is absent",
        ),
    ),
)
def test_step22_aborted_spawn_recovery_rejects_unproven_backup_or_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    failure: str,
) -> None:
    """Recovery re-entry must prove both the backup and durable authority."""

    from newsroom.control_plane.graphiti_events import GraphitiEventQueue

    stores = build_rehearsal_stores(tmp_path, unused_13689=True)
    activated = _activate_step22(stores.work_unpublished)
    plan = activated["plan"]
    _bind_qualified_candidate(
        monkeypatch, event_id=EVENT_13689, ledger_seq=LEDGER_13689
    )
    _seed_production_disposition(stores.work_unpublished, plan)
    _patch_production_predispatch(monkeypatch, plan=plan)
    consume_calls: list[str] = []

    def abort_after_claim(**values: object) -> None:
        consume_calls.append(str(values["event_id"]))
        queue = GraphitiEventQueue(
            str(values["unpublished_store"]),
            clock=lambda: OBSERVED_AT,
        )
        claimed = queue.claim(
            owner_id=str(values["owner_id"]),
            lease_for=timedelta(minutes=15),
            event_id=str(values["event_id"]),
            require_fresh=True,
            canary_consumption_digest=str(values["canary_consumption_digest"]),
        )
        assert claimed is not None
        assert queue._start(
            str(values["event_id"]), owner_id=str(values["owner_id"])
        ) == 1
        raise _CanarySpawnAborted("canary spawn aborted after claim")

    monkeypatch.setattr(disposition, "_consume_issue_790_event", abort_after_claim)
    backup = tmp_path / f"aborted-spawn-{mutation}.sqlite3"
    prepared = _prepare_production(
        stores,
        plan=plan,
        event_id=EVENT_13689,
        ledger_seq=LEDGER_13689,
    )
    canary_kwargs = {
        "store": stores.work_unpublished,
        "proving_store": stores.proving,
        "backup_path": backup,
        "plan": plan,
        "observed_at": OBSERVED_AT,
        "repository_root": tmp_path,
        "event_id": EVENT_13689,
        "ledger_seq": LEDGER_13689,
        "disposition_digest": _PRODUCTION_DISPOSITION,
        "prepared": prepared,
        "github_api": activated["github"],
    }
    with pytest.raises(_CanarySpawnAborted):
        disposition.run_issue_790_canary(**canary_kwargs)

    if mutation != "active_claim":
        _expire_interrupted_canary_claim(
            stores.work_unpublished, ledger_seq=LEDGER_13689
        )
    canary_kwargs["prepared"] = None
    if mutation != "missing_recovery_flag":
        canary_kwargs["recover_interrupted"] = True
    if mutation not in {"missing_recovery_flag", "missing_backup_digest"}:
        canary_kwargs["expected_backup_digest"] = "sha256:" + file_digest(backup)
    before = _sqlite_canary_event(
        stores.work_unpublished, ledger_seq=LEDGER_13689
    )
    assert before["state"] == "RUNNING"
    assert before["attempt_count"] == 1
    assert before["provider_dispatched"] == 0
    assert before["claim_owner"] is not None
    assert dispatch_started_count(stores.work_unpublished) == 0

    if mutation == "corrupt_backup":
        backup.write_bytes(b"not a sqlite database")
    elif mutation == "claimed_backup_snapshot":
        connection = sqlite3.connect(backup)
        try:
            connection.execute(
                "UPDATE unpublished_graphiti_revision_events "
                "SET state='CLAIMED',attempt_count=1,claim_owner='other-owner' "
                "WHERE ledger_seq=?",
                (LEDGER_13689,),
            )
            connection.commit()
        finally:
            connection.close()
    elif mutation == "missing_transport_table":
        connection = sqlite3.connect(stores.work_unpublished)
        try:
            connection.execute("DROP TABLE model_transport_observations")
            connection.commit()
        finally:
            connection.close()
    elif mutation == "consumption_authority_drift":
        canary_kwargs["disposition_digest"] = "sha256:" + "ff" * 32
    elif mutation == "prepared_binding_tamper":
        connection = sqlite3.connect(stores.work_unpublished)
        try:
            row = connection.execute(
                "SELECT record_json FROM issue_790_bounded_canary_consumptions "
                "WHERE ledger_seq=?",
                (LEDGER_13689,),
            ).fetchone()
            assert row is not None
            record = json.loads(row[0])
            record["preflight_evidence"]["prepared_canary_record_digest"] = (
                "sha256:" + "00" * 32
            )
            connection.execute(
                "UPDATE issue_790_bounded_canary_consumptions SET record_json=? "
                "WHERE ledger_seq=?",
                (json.dumps(record, sort_keys=True), LEDGER_13689),
            )
            connection.commit()
        finally:
            connection.close()
    elif mutation == "absent_recovery_backup":
        canary_kwargs["backup_path"] = tmp_path / "absent-recovery.sqlite3"
    else:
        assert mutation in {
            "active_claim",
            "missing_backup_digest",
            "missing_recovery_flag",
            "ownerless_active_expiry",
        }
        if mutation == "active_claim":
            connection = sqlite3.connect(stores.work_unpublished)
            try:
                connection.execute(
                    "UPDATE unpublished_graphiti_revision_events "
                    "SET claim_expires_at='2099-01-01T00:00:00+00:00' "
                    "WHERE ledger_seq=?",
                    (LEDGER_13689,),
                )
                connection.commit()
            finally:
                connection.close()
            before = _sqlite_canary_event(
                stores.work_unpublished, ledger_seq=LEDGER_13689
            )
        elif mutation == "ownerless_active_expiry":
            connection = sqlite3.connect(stores.work_unpublished)
            try:
                connection.execute(
                    "UPDATE unpublished_graphiti_revision_events "
                    "SET claim_owner=NULL,"
                    "claim_expires_at='2099-01-01T00:00:00+00:00' "
                    "WHERE ledger_seq=?",
                    (LEDGER_13689,),
                )
                connection.commit()
            finally:
                connection.close()
            before = _sqlite_canary_event(
                stores.work_unpublished, ledger_seq=LEDGER_13689
            )

    with pytest.raises(Issue790DispositionError, match=failure):
        disposition.run_issue_790_canary(**canary_kwargs)

    after = _sqlite_canary_event(
        stores.work_unpublished, ledger_seq=LEDGER_13689
    )
    connection = sqlite3.connect(stores.work_unpublished)
    try:
        consumptions = connection.execute(
            "SELECT COUNT(*) FROM issue_790_bounded_canary_consumptions "
            "WHERE ledger_seq=?",
            (LEDGER_13689,),
        ).fetchone()[0]
        outcomes = connection.execute(
            "SELECT COUNT(*) FROM issue_790_bounded_canary_outcomes "
            "WHERE ledger_seq=?",
            (LEDGER_13689,),
        ).fetchone()[0]
        allocations = connection.execute(
            "SELECT COUNT(*) FROM model_invocation_allocations WHERE cycle_id=?",
            (EVENT_13689,),
        ).fetchone()[0]
    finally:
        connection.close()
    assert consume_calls == [EVENT_13689]
    assert consumptions == 1
    assert outcomes == 0
    assert allocations == 0
    assert after == before


def _abort_13689_unmatched(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[object, dict[str, object], Path]:
    """Reproduce live 13689: consume, claim RUNNING, abort, no outcome."""

    from newsroom.control_plane.graphiti_events import GraphitiEventQueue

    stores = build_rehearsal_stores(tmp_path, unused_13689=True)
    activated = _activate_step22(stores.work_unpublished)
    plan = activated["plan"]
    _bind_qualified_candidate(
        monkeypatch, event_id=EVENT_13689, ledger_seq=LEDGER_13689
    )
    _seed_production_disposition(stores.work_unpublished, plan)
    _patch_production_predispatch(monkeypatch, plan=plan)

    def abort_after_claim(**values: object) -> None:
        assert values["event_id"] == EVENT_13689
        queue = GraphitiEventQueue(
            str(values["unpublished_store"]),
            clock=lambda: OBSERVED_AT,
        )
        claimed = queue.claim(
            owner_id=str(values["owner_id"]),
            lease_for=timedelta(minutes=15),
            event_id=str(values["event_id"]),
            require_fresh=True,
            canary_consumption_digest=str(values["canary_consumption_digest"]),
        )
        assert claimed is not None
        assert queue._start(
            str(values["event_id"]),
            owner_id=str(values["owner_id"]),
        ) == 1
        raise _CanarySpawnAborted("canary spawn aborted after claim")

    monkeypatch.setattr(disposition, "_consume_issue_790_event", abort_after_claim)
    backup = tmp_path / "aborted-spawn-13689.sqlite3"
    prepared = _prepare_production(
        stores,
        plan=plan,
        event_id=EVENT_13689,
        ledger_seq=LEDGER_13689,
    )
    canary_kwargs = {
        "store": stores.work_unpublished,
        "proving_store": stores.proving,
        "backup_path": backup,
        "plan": plan,
        "observed_at": OBSERVED_AT,
        "repository_root": tmp_path,
        "event_id": EVENT_13689,
        "ledger_seq": LEDGER_13689,
        "disposition_digest": _PRODUCTION_DISPOSITION,
        "prepared": prepared,
        "github_api": activated["github"],
    }
    with pytest.raises(_CanarySpawnAborted):
        disposition.run_issue_790_canary(**canary_kwargs)
    return stores, canary_kwargs, backup


def test_step22_unmatched_13689_consumption_blocks_successor_ready_before_backup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Live 13690: unmatched 13689 consumption must not READY then write dest.

    Watcher: 13689 consumed with no outcome. Preflight still READY 45/45 for
    unused 13690, DISPATCH_STARTED, then EXIT 2
    `bounded canary authority is already consumed` after writing
    unpublished_store.pre-canary.sqlite3. Event 13690 stayed QUEUED attempt 0.
    Recap 5485653227. Hole belongs in prepare, not a post-READY gate.
    """

    stores, aborted, _backup = _abort_13689_unmatched(tmp_path, monkeypatch)
    insert_unused_queued_attempt_zero(
        stores.work_unpublished,
        source_ledger_seq=LEDGER_13689,
        event_id=EVENT_13690,
        ledger_seq=LEDGER_13690,
    )
    unused = unused_queued_attempt_zero_candidates(
        stores.work_unpublished, stores.plan
    )
    assert unused[0] == (EVENT_13690, LEDGER_13690)
    with pytest.raises(PreparedCanaryError) as caught:
        prepare_issue_790_canary(
            store=stores.work_unpublished,
            proving_store=stores.proving,
            plan=aborted["plan"],
            observed_at=OBSERVED_AT,
            exact_head=EXACT_HEAD,
            role="preflight",
        )
    assert caught.value.failure_code == BOUNDED_CANARY_AUTHORITY_CONSUMED
    assert str(caught.value) == "bounded canary authority is already consumed"
    successor = _sqlite_canary_event(
        stores.work_unpublished, ledger_seq=LEDGER_13690
    )
    assert successor["state"] == "QUEUED"
    assert successor["attempt_count"] == 0
    assert successor["provider_dispatched"] == 0
    assert successor["claim_owner"] is None

    consume_calls: list[str] = []

    def must_not_consume(**_values: object) -> None:
        consume_calls.append("called")
        raise AssertionError("unmatched consumption dispatched a successor")

    monkeypatch.setattr(disposition, "_consume_issue_790_event", must_not_consume)
    dest = tmp_path / "successor-13690.sqlite3"
    with pytest.raises(PreparedCanaryError) as dispatched:
        disposition.run_issue_790_canary(
            **{
                **aborted,
                "backup_path": dest,
                "event_id": EVENT_13690,
                "ledger_seq": LEDGER_13690,
            }
        )
    assert dispatched.value.failure_code == BOUNDED_CANARY_AUTHORITY_CONSUMED
    assert str(dispatched.value) == "bounded canary authority is already consumed"
    assert consume_calls == []
    assert dest.exists() is False
    still = _sqlite_canary_event(stores.work_unpublished, ledger_seq=LEDGER_13690)
    assert still["state"] == "QUEUED"
    assert still["attempt_count"] == 0
    assert still["provider_dispatched"] == 0
    assert dispatch_started_count(stores.work_unpublished) == 0


def test_step22_sealed_13689_abort_exhausts_plan_before_successor_prepare(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A sealed consumption exhausts Step 22 before a dynamic successor."""

    stores = build_rehearsal_stores(tmp_path)
    insert_unused_queued_attempt_zero(
        stores.work_unpublished,
        source_ledger_seq=CANDIDATE_LEDGER_SEQ,
        event_id=EVENT_13690,
        ledger_seq=LEDGER_13690,
    )
    plan_digest = str(stores.plan["canonical_digest"])
    connection = sqlite3.connect(stores.work_unpublished)
    try:
        connection.execute(
            "INSERT INTO issue_790_bounded_canary_consumptions VALUES(?,?,?,?,?,?,?,?)",
            (
                "sha256:" + "71" * 32,
                plan_digest,
                "sha256:" + "72" * 32,
                EVENT_13689,
                LEDGER_13689,
                "issue-790-canary:sealed",
                "2026-08-31T23:00:00.000000Z",
                "{}",
            ),
        )
        connection.execute(
            "INSERT INTO issue_790_bounded_canary_outcomes VALUES(?,?,?,?,?,?)",
            (
                "sha256:" + "73" * 32,
                "sha256:" + "71" * 32,
                EVENT_13689,
                LEDGER_13689,
                "2026-08-31T23:01:00.000000Z",
                "{}",
            ),
        )
        connection.commit()
    finally:
        connection.close()
    with pytest.raises(PreparedCanaryError) as exhausted:
        _prepare(stores)
    assert exhausted.value.failure_code == BOUNDED_CANARY_AUTHORITY_CONSUMED
    successor = _sqlite_canary_event(
        stores.work_unpublished, ledger_seq=LEDGER_13690
    )
    assert successor["state"] == "QUEUED"
    assert successor["attempt_count"] == 0
    assert successor["provider_dispatched"] == 0


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("state", "QUEUED"),
        ("attempt_count", 2),
        ("provider_dispatched", 0),
        (
            "last_failure_code",
            "BOUNDED_CANARY_AUTHORITY_EXHAUSTED:PRODUCER_INTERNAL_ERROR",
        ),
    ),
)
def test_exhausted_safety_mutation_fail_closes(
    tmp_path: Path, field: str, value: object
) -> None:
    stores = build_rehearsal_stores(tmp_path)
    mutate_retry_field(
        stores.work_unpublished, ledger_seq=13361, field=field, value=value
    )
    with pytest.raises(PreparedCanaryError) as caught:
        _prepare(stores)
    assert caught.value.failure_code == "RETRY_FORBIDDEN_SAFETY_STATE"
    assert RehearsalRealGraphitiAdapter.provider_calls == 0
    assert candidate_identity(stores.work_unpublished)[0] == CANDIDATE_EVENT_ID


def test_claim_or_lease_fail_closes(tmp_path: Path) -> None:
    stores = build_rehearsal_stores(tmp_path)
    mutate_retry_field(
        stores.work_unpublished,
        ledger_seq=13361,
        field="claim_owner",
        value="issue-790-canary:test",
    )
    with pytest.raises(PreparedCanaryError) as caught:
        _prepare(stores)
    assert caught.value.failure_code == "RETRY_FORBIDDEN_SAFETY_STATE"


def test_candidate_claim_fail_closes(tmp_path: Path) -> None:
    stores = build_rehearsal_stores(tmp_path)
    mutate_retry_field(
        stores.work_unpublished,
        ledger_seq=CANDIDATE_LEDGER_SEQ,
        field="claim_owner",
        value="issue-790-canary:test",
    )
    with pytest.raises(PreparedCanaryError) as caught:
        _prepare(stores)
    assert caught.value.failure_code == "CANDIDATE_NOT_FRESH"


def test_alias_proving_live_paths_fail_close(tmp_path: Path) -> None:
    stores = build_rehearsal_stores(tmp_path)
    with pytest.raises(PreparedCanaryError) as caught:
        prepare_issue_790_canary(
            store=stores.work_unpublished,
            proving_store=stores.work_unpublished,
            plan=stores.plan,
            observed_at=OBSERVED_AT,
            exact_head=EXACT_HEAD,
            role="preflight",
        )
    assert caught.value.failure_code == "PATHS_ALIAS"


def test_missing_store_fail_closes(tmp_path: Path) -> None:
    stores = build_rehearsal_stores(tmp_path)
    with pytest.raises(PreparedCanaryError) as caught:
        prepare_issue_790_canary(
            store=tmp_path / "missing.sqlite3",
            proving_store=stores.proving,
            plan=stores.plan,
            observed_at=OBSERVED_AT,
            exact_head=EXACT_HEAD,
            role="preflight",
        )
    assert caught.value.failure_code == "STORE_ABSENT"


def test_missing_exact_head_fail_closes(tmp_path: Path) -> None:
    stores = build_rehearsal_stores(tmp_path)
    with pytest.raises(PreparedCanaryError) as caught:
        prepare_issue_790_canary(
            store=stores.work_unpublished,
            proving_store=stores.proving,
            plan=stores.plan,
            observed_at=OBSERVED_AT,
            exact_head="",
            role="preflight",
        )
    assert caught.value.failure_code == "EXACT_HEAD_ABSENT"


def test_retry_forbidden_target_fail_closes(tmp_path: Path) -> None:
    stores = build_rehearsal_stores(tmp_path)
    plan = dict(stores.plan)
    sequence = dict(plan["sequence"])
    sequence["sequence_ordinal"] = 16
    sequence.pop("candidate_event_qualification", None)
    plan["sequence"] = sequence
    with pytest.raises(PreparedCanaryError) as caught:
        prepare_issue_790_canary(
            store=stores.work_unpublished,
            proving_store=stores.proving,
            plan=plan,
            observed_at=OBSERVED_AT,
            exact_head=EXACT_HEAD,
            event_id=EVENT_13361,
            ledger_seq=13361,
            role="preflight",
        )
    assert caught.value.failure_code == "RETRY_FORBIDDEN_TARGET"


def test_unique_prepare_is_consumed_by_preflight_apply_and_canary() -> None:
    root = _TEST_FILE.resolve().parents[2]
    sources = (
        (root / "scripts/issue_790_live_canary_preflight.py").read_text(encoding="utf-8"),
        inspect.getsource(disposition._execute_issue_790_plan),
        inspect.getsource(disposition._run_issue_790_canary_locked),
    )
    for source in sources:
        assert "prepare_issue_790_canary" in source
    definitions = [
        node
        for node in ast.parse(
            (
                root / "newsroom/control_plane/issue_790_prepared_canary.py"
            ).read_text(encoding="utf-8")
        ).body
        if isinstance(node, ast.FunctionDef) and node.name == "prepare_issue_790_canary"
    ]
    assert len(definitions) == 1


def test_event_identity_invalid_fail_closes(tmp_path: Path) -> None:
    stores = build_rehearsal_stores(tmp_path)
    plan = dict(stores.plan)
    sequence = dict(plan["sequence"])
    sequence["sequence_ordinal"] = 16
    sequence.pop("candidate_event_qualification", None)
    plan["sequence"] = sequence
    with pytest.raises(PreparedCanaryError) as caught:
        prepare_issue_790_canary(
            store=stores.work_unpublished,
            proving_store=stores.proving,
            plan=plan,
            observed_at=OBSERVED_AT,
            exact_head=EXACT_HEAD,
            event_id="not-a-digest",
            ledger_seq=2000,
            role="canary",
        )
    assert caught.value.failure_code == "EVENT_IDENTITY_INVALID"


def test_missing_prepared_canary_fail_closes_before_dispatch(tmp_path: Path) -> None:
    stores = build_rehearsal_stores(tmp_path)
    RehearsalRealGraphitiAdapter.provider_calls = 0
    RehearsalRealGraphitiAdapter.dispatch_started = False
    with pytest.raises(PreparedCanaryError) as caught:
        run_prepared_canary_rehearsal(
            store=stores.work_unpublished,
            proving_store=stores.proving,
            plan=stores.plan,
            observed_at=OBSERVED_AT,
            exact_head=EXACT_HEAD,
            prepared=None,
        )
    assert caught.value.failure_code == PREPARED_CANARY_ABSENT
    assert RehearsalRealGraphitiAdapter.dispatch_started is False
    assert RehearsalRealGraphitiAdapter.provider_calls == 0
    assert dispatch_started_count(stores.work_unpublished) == 0
    assert candidate_identity(stores.work_unpublished)[2] == "QUEUED"


def test_digest_drift_fail_closes_before_dispatch(tmp_path: Path) -> None:
    stores = build_rehearsal_stores(tmp_path)
    prepared = _prepare(stores)
    drifted = replace(prepared, decision_digest="sha256:" + "00" * 32)
    with pytest.raises(PreparedCanaryError) as caught:
        run_prepared_canary_rehearsal(
            store=stores.work_unpublished,
            proving_store=stores.proving,
            plan=stores.plan,
            observed_at=OBSERVED_AT,
            exact_head=EXACT_HEAD,
            prepared=drifted,
        )
    assert caught.value.failure_code == PREPARED_CANARY_DIGEST_DRIFT
    assert RehearsalRealGraphitiAdapter.dispatch_started is False
    assert dispatch_started_count(stores.work_unpublished) == 0
    assert candidate_identity(stores.work_unpublished)[0] == CANDIDATE_EVENT_ID


def test_dispatch_before_crash_leaves_candidate_unconsumed(tmp_path: Path) -> None:
    stores = build_rehearsal_stores(tmp_path)
    prepared = _prepare(stores)
    with pytest.raises(PreparedCanaryError) as caught:
        run_prepared_canary_rehearsal(
            store=stores.work_unpublished,
            proving_store=stores.proving,
            plan=stores.plan,
            observed_at=OBSERVED_AT,
            exact_head=EXACT_HEAD,
            prepared=prepared,
            crash_before_dispatch=True,
        )
    assert caught.value.failure_code == "REHEARSAL_CRASH_BEFORE_DISPATCH"
    event_id, ledger_seq, state = candidate_identity(stores.work_unpublished)
    assert event_id == CANDIDATE_EVENT_ID
    assert ledger_seq == CANDIDATE_LEDGER_SEQ
    assert state == "QUEUED"
    assert dispatch_started_count(stores.work_unpublished) == 0
    assert RehearsalRealGraphitiAdapter.provider_calls == 0


def test_rehearsal_refuses_canonical_live_store_paths() -> None:
    forbidden = next(iter(live_issue_790_store_paths()))
    with pytest.raises(PreparedCanaryError) as caught:
        refuse_live_issue_790_store_paths(forbidden)
    assert caught.value.failure_code == "LIVE_STORE_WRITE_REFUSED"


def test_sqlite_backup_refuses_overwrite(tmp_path: Path) -> None:
    stores = build_rehearsal_stores(tmp_path)
    with pytest.raises(PreparedCanaryError) as caught:
        sqlite_backup_copy(stores.sealed_unpublished, stores.work_unpublished)
    assert caught.value.failure_code == "LIVE_STORE_WRITE_REFUSED"


def test_canary_consumes_prepared_digest_only() -> None:
    source = inspect.getsource(disposition._run_issue_790_canary_locked)
    pre_consume, _sep, _post = source.partition("_consume_issue_790_event")
    assert "prepare_issue_790_canary" in pre_consume
    assert "consume_prepared_canary" in pre_consume
    assert source.index("_candidate_from_plan") < source.index(
        "_resolve_canary_backup_destination"
    )
    assert "_require_retry_events_unchanged" not in pre_consume
    assert "retry_forbidden_safety_states_match" not in pre_consume
    assert "validate_retry_forbidden_safety_state" not in pre_consume
    assert "_validate_operational_evidence" in source
    assert "_require_step16_code_identity" in inspect.getsource(
        disposition._validate_operational_evidence
    )
    identity_src = inspect.getsource(disposition._require_step16_code_identity)
    assert "_git_commit_is_ancestor" in identity_src
    assert "ci_test.get(\"head_sha\") != exact_head" in identity_src
    assert "merge-base" in inspect.getsource(disposition._git_commit_is_ancestor)
    resume = source.split("prior_consumption = canary_repository.existing_consumption", 1)[1]
    resume = resume.split("resuming_zero_io_finalisation", 1)[0]
    assert "event_id=event_id" in resume
    assert "ledger_seq=ledger_seq" in resume


def test_fail_branch_inventory_has_named_parity_tests() -> None:
    sources = (
        _TEST_FILE,
        _TEST_FILE.with_name("test_issue_790_prepared_canary_artifact.py"),
    )
    names = {
        node.name
        for source in sources
        for node in ast.parse(source.read_text(encoding="utf-8")).body
        if isinstance(node, ast.FunctionDef)
    }
    for branch in FAIL_BRANCH_INVENTORY:
        assert branch.positive_test in names, branch.invariant
        assert branch.negative_test in names, branch.invariant
        assert branch.zero_provider_calls is True


def test_prepared_failure_codes_are_inventoried() -> None:
    root = Path(__file__).resolve().parents[2]
    modules = (
        root / "newsroom/control_plane/issue_790_prepared_canary.py",
        root / "newsroom/control_plane/issue_790_rehearsal.py",
    )
    codes: set[str] = set()
    for module in modules:
        tree = ast.parse(module.read_text(encoding="utf-8"))
        codes.update(
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value.isupper()
            and "_" in node.value
            and node.value
            not in {
                "UTC",
                "QUEUED",
                "EXPLICIT_QUEUED_ATTEMPT_ZERO_EVENT",
                "DISABLED_BEFORE_PROVIDER_DISPATCH",
                "PREDISPATCH_BINDING_FAILURE",
                "DISPATCH_STARTED",
                "NO_PROVIDER_CALL",
                "UNSTRUCTURED",
            }
        )
    inventoried = {branch.failure_code for branch in FAIL_BRANCH_INVENTORY}
    named = {
        "PREPARED_CANARY_ABSENT",
        "PREPARED_CANARY_DIGEST_DRIFT",
        "EXACT_HEAD_ABSENT",
        "CANDIDATE_IDENTITY",
        "STORE_ABSENT",
        "PATHS_ALIAS",
        "RETRY_FORBIDDEN_SAFETY_STATE",
        "RETRY_FORBIDDEN_TARGET",
        "EVENT_IDENTITY_INVALID",
        "CANDIDATE_NOT_FRESH",
        "LIVE_STORE_WRITE_REFUSED",
        "REHEARSAL_CRASH_BEFORE_DISPATCH",
        "BOUNDED_CANARY_AUTHORITY_CONSUMED",
    }
    assert named <= inventoried
    assert named <= codes | inventoried
    local_raise_codes = {
        code
        for code in codes
        if code.endswith(
            (
                "_ABSENT",
                "_DRIFT",
                "_STATE",
                "_TARGET",
                "_INVALID",
                "_FRESH",
                "_REFUSED",
                "_DISPATCH",
                "_ALIAS",
                "_IDENTITY",
                "_CONSUMED",
            )
        )
        or code in named
    }
    assert local_raise_codes <= inventoried | named


def test_rehearsal_skips_live_only_predispatch_gates() -> None:
    source = inspect.getsource(run_prepared_canary_rehearsal)
    assert LIVE_ONLY_PREDISPATCH_GATES
    for gate in (
        "_require_approved_plan",
        "_validate_operational_evidence",
        "_require_issue_790_canary_route",
        "_require_step16_runtime_semantics",
        "_require_worker_unloaded",
        "_assert_exact_target",
        "_require_sequence_predecessor",
    ):
        assert gate not in source


def test_full_rehearsal_uses_production_executor_not_parallel_branch() -> None:
    production_source = inspect.getsource(disposition.run_issue_790_canary)
    locked_source = inspect.getsource(disposition._run_issue_790_canary_locked)
    script_source = (
        _ROOT / "scripts/issue_790_prepared_canary_rehearsal.py"
    ).read_text()
    signature = inspect.signature(disposition.run_issue_790_canary)
    assert {"rehearsal", "exact_head", "crash_before_dispatch"}.isdisjoint(
        signature.parameters
    )
    assert "run_prepared_canary_rehearsal" not in production_source
    assert "run_prepared_canary_rehearsal" not in locked_source
    assert "run_issue_790_canary(" in script_source
    assert "graphiti=RehearsalEvaluationGraphitiRunner(" in script_source
