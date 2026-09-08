"""Emit a provider-free Graphiti steady-state evidence packet."""

from __future__ import annotations

import argparse
import json
import secrets
import sqlite3
import subprocess
import sys
import time
from collections.abc import Iterator, Mapping
from contextlib import ExitStack, contextmanager
from datetime import UTC, datetime
from pathlib import Path

from newsroom.authority.canonical import (
    digest_canonical,
    validate_sha256_digest,
)
from newsroom.control_plane.graphiti_operational_readiness import (
    bootstrap_operational_authority,
    build_and_reconcile_operational_generation,
    build_operational_campaign_input,
    open_operational_graphiti_authority_system,
    plan_operational_authority_bootstrap,
)
from newsroom.control_plane.graphiti_steady_state import (
    build_graphiti_steady_state_packet,
    graphiti_graph_destination_readback,
    graphiti_store_snapshot_digests,
    validate_graphiti_campaign_packet,
    write_content_addressed_packet,
)
from newsroom.control_plane.paths import (
    CANONICAL_INCREMENT4_AUTHORITY_STORE,
    CANONICAL_PROVING_STORE,
    CANONICAL_UNPUBLISHED_STORE,
    require_canonical_proving_store,
    require_canonical_unpublished_store,
)
from newsroom.control_plane.sqlite_profile import apply_control_plane_sqlite_profile
from scripts.hermes_graphiti_worker import (
    GovernedGraphitiWorkerRuntime,
    compose_governed_graphiti_worker_runtime,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
OPERATIONAL_RESULT_SCHEMA_VERSION = "newsroom.graphiti-operational-readiness-result.v1"


def _utc_text(value: datetime) -> str:
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _begin_operational_stage(name: str) -> float:
    """Mark stage entry on stderr so an unfinished stage is visible."""

    print(f"operational_stage\t{name}\tbegin", file=sys.stderr, flush=True)
    return time.monotonic()


def _log_operational_stage(name: str, started: float) -> None:
    """Emit stage elapsed time on stderr; not part of packet identity."""

    elapsed = time.monotonic() - started
    print(
        f"operational_stage\t{name}\telapsed_s={elapsed:.3f}",
        file=sys.stderr,
        flush=True,
    )


def _git(*args: str) -> str:
    return subprocess.run(
        ("git", *args),
        check=True,
        capture_output=True,
        text=True,
        cwd=REPOSITORY_ROOT,
    ).stdout.strip()


def _exact_main_identity() -> tuple[str, str]:
    if _git("status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("steady-state evidence requires a clean worktree")
    head_sha = _git("rev-parse", "HEAD")
    if head_sha != _git("rev-parse", "origin/main"):
        raise RuntimeError("steady-state evidence requires exact origin/main")
    return head_sha, _git("rev-parse", "HEAD^{tree}")


@contextmanager
def _locked_operational_inputs() -> Iterator[
    tuple[sqlite3.Connection, sqlite3.Connection]
]:
    connections: list[sqlite3.Connection] = []
    try:
        for path in (CANONICAL_PROVING_STORE, CANONICAL_UNPUBLISHED_STORE):
            connection = sqlite3.connect(path, isolation_level=None, timeout=30)
            connection.row_factory = sqlite3.Row
            apply_control_plane_sqlite_profile(connection, wal=None)
            connection.execute("BEGIN IMMEDIATE")
            connections.append(connection)
        yield connections[0], connections[1]
    finally:
        for connection in reversed(connections):
            if connection.in_transaction:
                connection.rollback()
            connection.close()


def _authority_backup_identity(output_dir: Path) -> str:
    """Record pre-bootstrap presence without copying the authority store."""

    source = CANONICAL_INCREMENT4_AUTHORITY_STORE
    output_dir.mkdir(parents=True, exist_ok=True)
    if not source.exists():
        return digest_canonical(
            {"source_path": str(source), "pre_bootstrap_state": "ABSENT"}
        )
    return digest_canonical(
        {
            "source_path": str(source.resolve()),
            "pre_bootstrap_state": "PRESENT",
            "backup_omitted": True,
        }
    )


def _failure_evidence(
    reason_code: str,
    error: Exception,
    *,
    stage: str,
) -> dict[str, str]:
    """Describe a failure without retaining exception text or credentials."""

    return {
        "reason_code": reason_code,
        "stage": stage,
        "exception_type": type(error).__name__,
        "detail_digest": digest_canonical(
            {
                "exception_type": type(error).__name__,
                "message": str(error),
                "stage": stage,
            }
        ),
    }


def _seal_operational_result(
    packet: Mapping[str, object] | None,
    *,
    head_sha: str,
    tree_sha: str,
    observed_at: datetime,
    operational_evidence: Mapping[str, object],
    failure: Mapping[str, str] | None = None,
    evaluator_failure: Mapping[str, str] | None = None,
    evaluator_attempted: bool = True,
) -> dict[str, object]:
    """Bind authorised apply evidence to one truthful terminal result."""

    if packet is None:
        body: dict[str, object] = {
            "schema_version": OPERATIONAL_RESULT_SCHEMA_VERSION,
            "code_identity": {"head_sha": head_sha, "tree_sha": tree_sha},
            "observed_at": _utc_text(observed_at),
            "operational_reconciliation": dict(operational_evidence),
            "evaluator": {
                "attempted": evaluator_attempted,
                "completed": False,
                **(
                    {"failure": dict(evaluator_failure or {})}
                    if evaluator_attempted
                    else {}
                ),
            },
            "blockers": sorted(
                {
                    *(("READINESS_EVALUATOR_FAILED",) if evaluator_attempted else ()),
                    *(() if failure is None else (str(failure["reason_code"]),)),
                }
            ),
            "verdict": "NO_GO",
            "readiness": "ENGINEERING_PREPARATION_ONLY",
        }
        return {**body, "packet_digest": digest_canonical(body)}

    body = {key: value for key, value in packet.items() if key != "packet_digest"}
    body["operational_reconciliation"] = dict(operational_evidence)
    body["non_effects_scope"] = "READ_ONLY_EVALUATOR_ONLY"
    if failure is not None:
        blockers = body.get("blockers")
        blocker_values = (
            [str(item) for item in blockers] if isinstance(blockers, list) else []
        )
        blocker_values.append(str(failure["reason_code"]))
        body["blockers"] = sorted(set(blocker_values))
        body["verdict"] = "NO_GO"
        body["readiness"] = "ENGINEERING_PREPARATION_ONLY"
    return {**body, "packet_digest": digest_canonical(body)}


def _confirm_code_identity(head_sha: str, tree_sha: str) -> None:
    stage_started = _begin_operational_stage("CODE_IDENTITY_CONFIRM")
    if _exact_main_identity() != (head_sha, tree_sha):
        raise RuntimeError("code identity changed while building steady-state evidence")
    _log_operational_stage("CODE_IDENTITY_CONFIRM", stage_started)


@contextmanager
def sealed_operational_campaign_runtime(
    *,
    head_sha: str,
    tree_sha: str,
    focus_manifest_digest: str,
    output_dir: Path,
    observed_at: datetime,
) -> Iterator[tuple[dict[str, object], GovernedGraphitiWorkerRuntime | None]]:
    """Seal once and retain its authority owner for immediate guarded dispatch.

    Input snapshot locks end before handover; the authority writer remains owned
    until context exit. Non-READY results never expose a dispatchable runtime.
    Separate-process consumers still use the full reopen/currentness path.
    """

    with ExitStack() as owner:
        packet, runtime = _prepare_operational_packet(
            head_sha=head_sha,
            tree_sha=tree_sha,
            focus_manifest_digest=focus_manifest_digest,
            output_dir=output_dir,
            observed_at=observed_at,
            owner=owner,
        )
        _confirm_code_identity(head_sha, tree_sha)
        yield packet, runtime


def _prepare_operational_packet(
    *,
    head_sha: str,
    tree_sha: str,
    focus_manifest_digest: str,
    output_dir: Path,
    observed_at: datetime,
    owner: ExitStack,
) -> tuple[dict[str, object], GovernedGraphitiWorkerRuntime | None]:
    require_canonical_proving_store(str(CANONICAL_PROVING_STORE))
    require_canonical_unpublished_store(str(CANONICAL_UNPUBLISHED_STORE))
    validate_sha256_digest(
        focus_manifest_digest,
        field="exact-main Focus Gate manifest digest",
    )
    with _locked_operational_inputs() as (proving, unpublished):
        stage = "BACKUP"
        system = None
        campaign = None
        reconciliation = None
        runtime = None
        preparation_failure: dict[str, str] | None = None
        completed_steps: list[str] = []
        operational_evidence: dict[str, object] = {
            "schema_version": "newsroom.graphiti-operational-reconciliation.v1",
            "status": "INCOMPLETE",
            "completed_steps": completed_steps,
            "provider_calls": 0,
            "graphiti_dispatches": 0,
            "service_loads": 0,
            "publication_effects": 0,
            "production_admission_effects": 0,
        }
        overall_started = _begin_operational_stage("TOTAL")
        try:
            try:
                stage_started = _begin_operational_stage("BACKUP")
                backup_identity = _authority_backup_identity(output_dir)
                operational_evidence["backup_identity"] = backup_identity
                completed_steps.append("BACKUP")
                _log_operational_stage("BACKUP", stage_started)

                stage = "CANONICAL_AUTHORITY_OPEN"
                stage_started = _begin_operational_stage(stage)
                system, proof = open_operational_graphiti_authority_system(
                    credential=secrets.token_urlsafe(32)
                )
                owner.callback(system.close)
                completed_steps.append(stage)
                _log_operational_stage(stage, stage_started)

                stage = "CURRENT_COHORT_PLAN"
                stage_started = _begin_operational_stage(stage)
                authority = sqlite3.connect(CANONICAL_INCREMENT4_AUTHORITY_STORE)
                authority.row_factory = sqlite3.Row
                apply_control_plane_sqlite_profile(authority, query_only=True, wal=None)
                try:
                    plan = plan_operational_authority_bootstrap(
                        proving,
                        unpublished,
                        authority,
                        observed_at=observed_at,
                    )
                finally:
                    authority.close()
                operational_evidence["plan_digest"] = plan.plan_digest
                operational_evidence["cohort_digest"] = plan.cohort_digest
                completed_steps.append(stage)
                _log_operational_stage(stage, stage_started)

                stage = "SOURCE_AND_OBJECT_BOOTSTRAP"
                stage_started = _begin_operational_stage(stage)
                bootstrap, binder = bootstrap_operational_authority(
                    system,
                    proof=proof,
                    plan=plan,
                )
                operational_evidence["bootstrap"] = bootstrap.canonical_value()
                completed_steps.append(stage)
                _log_operational_stage(stage, stage_started)

                stage = "ACTIVE_GENERATION_RECONCILIATION"
                stage_started = _begin_operational_stage(stage)
                reconciliation = build_and_reconcile_operational_generation(
                    system,
                    proof=proof,
                    plan=plan,
                    bootstrap=bootstrap,
                )
                completed_steps.append(stage)
                _log_operational_stage(stage, stage_started)

                stage = "STORE_IDENTITY_SNAPSHOT"
                stage_started = _begin_operational_stage(stage)
                snapshots = graphiti_store_snapshot_digests(
                    proving_store=CANONICAL_PROVING_STORE,
                    unpublished_store=CANONICAL_UNPUBLISHED_STORE,
                    authority_store=CANONICAL_INCREMENT4_AUTHORITY_STORE,
                )
                operational_evidence["store_snapshot_digests"] = dict(snapshots)
                completed_steps.append(stage)
                _log_operational_stage(stage, stage_started)

                stage = "DORMANT_RUNTIME_COMPOSITION"
                stage_started = _begin_operational_stage(stage)
                runtime = compose_governed_graphiti_worker_runtime(
                    authority_system=system,
                    authority_store_descriptor_digest=snapshots["authority"],
                    proof=proof,
                    bind_unit_authority=binder,
                    expected_authority_store_path=str(
                        CANONICAL_INCREMENT4_AUTHORITY_STORE
                    ),
                )
                completed_steps.append(stage)
                _log_operational_stage(stage, stage_started)

                stage = "AUTHENTICATED_GRAPH_READBACK"
                stage_started = _begin_operational_stage(stage)
                graph_readback = graphiti_graph_destination_readback(
                    destination_id=system.graph_destination_id,
                    reconciliation=reconciliation,
                )
                operational_evidence["graph_readback"] = graph_readback
                completed_steps.append(stage)
                _log_operational_stage(stage, stage_started)

                stage = "RECOVERY_IDENTITY"
                stage_started = _begin_operational_stage(stage)
                recovery_identity = digest_canonical(
                    {
                        "pre_bootstrap_backup": backup_identity,
                        "bootstrap": bootstrap.canonical_value(),
                        "authority_store": snapshots["authority"],
                        "active_graph": graph_readback,
                    }
                )
                _log_operational_stage(stage, stage_started)
                stage = "DORMANT_CAMPAIGN_INPUT"
                stage_started = _begin_operational_stage(stage)
                campaign = build_operational_campaign_input(
                    head_sha=head_sha,
                    tree_sha=tree_sha,
                    focus_manifest_digest=focus_manifest_digest,
                    graph_destination_id=system.graph_destination_id,
                    candidate_event_count=bootstrap.candidate_event_count,
                    recovery_identity=recovery_identity,
                )
                operational_evidence["campaign_input_digest"] = digest_canonical(
                    campaign
                )
                operational_evidence["campaign_authorised"] = False
                completed_steps.append(stage)
                _log_operational_stage(stage, stage_started)
                operational_evidence["status"] = "COMPLETE"
            except Exception as error:
                _log_operational_stage(stage, stage_started)
                preparation_failure = _failure_evidence(
                    "OPERATIONAL_PREPARATION_FAILED",
                    error,
                    stage=stage,
                )
                operational_evidence["status"] = "FAILED"
                operational_evidence["failure"] = preparation_failure

            if preparation_failure is not None:
                stage_started = _begin_operational_stage("SEAL_OPERATIONAL_RESULT")
                sealed = _seal_operational_result(
                    None,
                    head_sha=head_sha,
                    tree_sha=tree_sha,
                    observed_at=datetime.now(tz=UTC),
                    operational_evidence=operational_evidence,
                    failure=preparation_failure,
                    evaluator_attempted=False,
                )
                _log_operational_stage("SEAL_OPERATIONAL_RESULT", stage_started)
                return sealed, None

            evaluator_observed_at = datetime.now(tz=UTC)
            stage = "READINESS_EVALUATOR"
            stage_started = _begin_operational_stage(stage)
            try:
                packet = build_graphiti_steady_state_packet(
                    proving_store=CANONICAL_PROVING_STORE,
                    unpublished_store=CANONICAL_UNPUBLISHED_STORE,
                    head_sha=head_sha,
                    tree_sha=tree_sha,
                    observed_at=evaluator_observed_at,
                    authority_store=(
                        CANONICAL_INCREMENT4_AUTHORITY_STORE
                        if CANONICAL_INCREMENT4_AUTHORITY_STORE.exists()
                        else None
                    ),
                    campaign_input=campaign,
                    graph_destination_reconciliation=reconciliation,
                    governed_runtime=runtime,
                )
            except Exception as error:
                _log_operational_stage(stage, stage_started)
                stage_started = _begin_operational_stage("SEAL_OPERATIONAL_RESULT")
                sealed = _seal_operational_result(
                    None,
                    head_sha=head_sha,
                    tree_sha=tree_sha,
                    observed_at=evaluator_observed_at,
                    operational_evidence=operational_evidence,
                    failure=preparation_failure,
                    evaluator_failure=_failure_evidence(
                        "READINESS_EVALUATOR_FAILED",
                        error,
                        stage="READINESS_EVALUATOR",
                    ),
                )
                _log_operational_stage("SEAL_OPERATIONAL_RESULT", stage_started)
                return sealed, None
            _log_operational_stage(stage, stage_started)

            stage = "SEAL_OPERATIONAL_RESULT"
            stage_started = _begin_operational_stage(stage)
            result = _seal_operational_result(
                packet,
                head_sha=head_sha,
                tree_sha=tree_sha,
                observed_at=evaluator_observed_at,
                operational_evidence=operational_evidence,
                failure=preparation_failure,
            )
            _log_operational_stage(stage, stage_started)
            try:
                if result.get("verdict") == "READY_FOR_OWNER_DECISION":
                    stage = "READY_PACKET_VALIDATION"
                    stage_started = _begin_operational_stage(stage)
                    validate_graphiti_campaign_packet(result)
                    _log_operational_stage(stage, stage_started)
            except Exception as error:
                _log_operational_stage(stage, stage_started)
                failure = _failure_evidence(
                    "READY_PACKET_VALIDATION_FAILED",
                    error,
                    stage="READY_PACKET_VALIDATION",
                )
                operational_evidence["status"] = "FAILED"
                operational_evidence["failure"] = failure
                stage_started = _begin_operational_stage("SEAL_OPERATIONAL_RESULT")
                result = _seal_operational_result(
                    result,
                    head_sha=head_sha,
                    tree_sha=tree_sha,
                    observed_at=evaluator_observed_at,
                    operational_evidence=operational_evidence,
                    failure=failure,
                )
                _log_operational_stage("SEAL_OPERATIONAL_RESULT", stage_started)
            return result, (
                runtime if result.get("verdict") == "READY_FOR_OWNER_DECISION" else None
            )
        finally:
            _log_operational_stage("TOTAL", overall_started)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proving", required=True)
    parser.add_argument("--unpublished", required=True)
    parser.add_argument("--authority")
    parser.add_argument("--campaign-input", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--operational", action="store_true")
    parser.add_argument("--focus-manifest-digest")
    args = parser.parse_args()
    head_sha, tree_sha = _exact_main_identity()
    observed_at = datetime.now(tz=UTC)
    if args.operational:
        if (
            args.campaign_input is not None
            or args.authority not in (None, str(CANONICAL_INCREMENT4_AUTHORITY_STORE))
            or args.focus_manifest_digest is None
            or args.output_dir is None
        ):
            raise ValueError("operational mode requires output and Focus evidence only")
        require_canonical_proving_store(args.proving)
        require_canonical_unpublished_store(args.unpublished)
        output_dir = args.output_dir.expanduser().resolve()
        if output_dir == REPOSITORY_ROOT or REPOSITORY_ROOT in output_dir.parents:
            raise ValueError("operational evidence must be outside the repository")
        with sealed_operational_campaign_runtime(
            head_sha=head_sha,
            tree_sha=tree_sha,
            focus_manifest_digest=args.focus_manifest_digest,
            output_dir=output_dir,
            observed_at=observed_at,
        ) as (packet, _runtime):
            pass
    else:
        campaign_input = (
            json.loads(args.campaign_input.read_text(encoding="utf-8"))
            if args.campaign_input is not None
            else None
        )
        if campaign_input is not None and not isinstance(campaign_input, dict):
            raise ValueError("campaign input must be a JSON object")
        packet = build_graphiti_steady_state_packet(
            proving_store=args.proving,
            unpublished_store=args.unpublished,
            head_sha=head_sha,
            tree_sha=tree_sha,
            observed_at=observed_at,
            authority_store=args.authority,
            campaign_input=campaign_input,
        )
        _confirm_code_identity(head_sha, tree_sha)
    if args.output_dir is None:
        print(json.dumps(packet, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        stage_started = _begin_operational_stage("PACKET_OUTPUT")
        print(write_content_addressed_packet(packet, args.output_dir))
        _log_operational_stage("PACKET_OUTPUT", stage_started)
    return 0 if packet["verdict"] == "READY_FOR_OWNER_DECISION" else 2


if __name__ == "__main__":
    raise SystemExit(main())
