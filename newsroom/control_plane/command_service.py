"""Authenticated Control Plane reconciliation command service (ADR 0002)."""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol

from newsroom.authority.auth import AuthenticationProof
from newsroom.authority.canonical import (
    canonical_json_bytes,
    digest_bytes,
    digest_canonical,
    validate_sha256_digest,
)
from newsroom.authority.types import UtcTimestamp
from newsroom.control_plane import backlog_reconciliation as backlog
from newsroom.control_plane.command_auth import HERMES_COMMAND_PRINCIPAL
from newsroom.control_plane.graphiti_admission import (
    GraphitiAdmissionConsumerError,
    graphiti_admission_request_from_value,
)
from newsroom.control_plane.graphiti_event_reconciliation import (
    GRAPHITI_EVENT_REPAIR_COMMAND_TYPE,
    GraphitiEventReconciliationReceipt,
    _GraphitiEventReconciliationCommand,
    _apply_graphiti_event_reconciliation,
)
from newsroom.control_plane.graphiti_spend_reconciliation import (
    GRAPHITI_SPEND_RECONCILE_COMMAND_TYPE,
    GraphitiSpendReconciliationReceipt,
    _GraphitiSpendReconciliationCommand,
    _apply_graphiti_spend_reconciliation,
)
from newsroom.control_plane.sqlite_profile import apply_control_plane_sqlite_profile
from newsroom.control_plane.veto import assert_private_store
from newsroom.graphiti_adapter.evaluation_packet import GRAPHITI_WORKSPACE_GROUP


_GRAPHITI_ADMISSION_AUTHORIZATION_RECOVERY_COMMAND_TYPE = (
    "RECOVER_GRAPHITI_ADMISSION_AUTHORIZATION"
)
_GRAPHITI_ADMISSION_AUTHORIZATION_RECOVERY_SCHEMA = (
    "newsroom.control-plane.graphiti-admission-authorization-recovery.v1"
)
_GRAPHITI_ADMISSION_AUTHORIZATION_ERROR = (
    "AuthorizationDenied: authorization denied: AUTHZ_SCOPE_MISSING"
)


class _VerifiedAuthentication(Protocol):
    principal_id: str

    def require_current(self, now: UtcTimestamp) -> None: ...


class _Authenticator(Protocol):
    def authenticate(
        self, proof: object, *, now: UtcTimestamp
    ) -> _VerifiedAuthentication: ...


def _validate_graphiti_admission_recovery_receipt(
    receipt_json: str,
    *,
    idempotency_key: str,
    command_digest: str,
    caller_principal: str,
    writer_principal: str,
) -> dict[str, object]:
    try:
        receipt = json.loads(receipt_json)
        if not isinstance(receipt, dict):
            raise TypeError("receipt must be an object")
        unsigned = dict(receipt)
        supplied_digest = unsigned.pop("receipt_digest", None)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise GraphitiAdmissionConsumerError(
            "retained Graphiti admission recovery receipt is invalid"
        ) from exc
    if (
        receipt.get("schema")
        != _GRAPHITI_ADMISSION_AUTHORIZATION_RECOVERY_SCHEMA
        or receipt.get("idempotency_key") != idempotency_key
        or receipt.get("command_digest") != command_digest
        or receipt.get("command_type")
        != _GRAPHITI_ADMISSION_AUTHORIZATION_RECOVERY_COMMAND_TYPE
        or receipt.get("authenticated_principal") != caller_principal
        or receipt.get("writer_principal") != writer_principal
        or supplied_digest != digest_canonical(unsigned)
    ):
        raise GraphitiAdmissionConsumerError(
            "retained Graphiti admission recovery receipt integrity differs"
        )
    return receipt


class ControlPlaneCommandService:
    """Sole direct writer for Control Plane canonical mutation."""

    principal = backlog.COMMAND_SERVICE_PRINCIPAL

    def __init__(
        self,
        *,
        authenticator: _Authenticator,
        clock: Callable[[], UtcTimestamp] = UtcTimestamp.now,
    ) -> None:
        if authenticator is None:
            raise ValueError("command service requires an authenticator")
        self._authenticator = authenticator
        self._clock = clock

    def reconcile_graphiti_spend(
        self,
        *,
        unpublished_store: str,
        dry_run_plan: Mapping[str, object],
        evaluated_at: datetime,
        idempotency_key: str,
        expected_plan_digest: str,
        proof: AuthenticationProof,
        graph_journal_evidence: Mapping[str, Mapping[str, object]] | None = None,
    ) -> GraphitiSpendReconciliationReceipt:
        """Authenticate and apply one provider-free spend reconciliation plan."""

        now = self._clock()
        authentication = self._authenticator.authenticate(proof, now=now)
        authentication.require_current(now)
        if authentication.principal_id != HERMES_COMMAND_PRINCIPAL:
            raise PermissionError(
                "Graphiti spend reconciliation requires the Hermes principal"
            )
        return _apply_graphiti_spend_reconciliation(
            unpublished_store,
            dry_run_plan=dry_run_plan,
            evaluated_at=evaluated_at,
            graph_journal_evidence=graph_journal_evidence,
            command=_GraphitiSpendReconciliationCommand(
                caller_principal=authentication.principal_id,
                writer_principal=self.principal,
                command_type=GRAPHITI_SPEND_RECONCILE_COMMAND_TYPE,
                idempotency_key=idempotency_key,
                expected_plan_digest=expected_plan_digest,
            ),
        )

    def reconcile_graphiti_events(
        self,
        *,
        proving_store: str,
        unpublished_store: str,
        dry_run_plan: Mapping[str, object],
        evaluated_at: datetime,
        idempotency_key: str,
        expected_plan_digest: str,
        proof: AuthenticationProof,
    ) -> GraphitiEventReconciliationReceipt:
        """Authenticate and apply one provider-free missing-event repair plan."""

        now = self._clock()
        authentication = self._authenticator.authenticate(proof, now=now)
        authentication.require_current(now)
        if authentication.principal_id != HERMES_COMMAND_PRINCIPAL:
            raise PermissionError("Graphiti event repair requires the Hermes principal")
        return _apply_graphiti_event_reconciliation(
            proving_store,
            unpublished_store,
            dry_run_plan=dry_run_plan,
            evaluated_at=evaluated_at,
            applied_at=now.value,
            command=_GraphitiEventReconciliationCommand(
                caller_principal=authentication.principal_id,
                writer_principal=self.principal,
                command_type=GRAPHITI_EVENT_REPAIR_COMMAND_TYPE,
                idempotency_key=idempotency_key,
                expected_plan_digest=expected_plan_digest,
            ),
        )

    def recover_graphiti_admission_authorization(
        self,
        *,
        unpublished_store: str,
        ingest_id: str,
        proposal_key: str,
        expected_request_digest: str,
        remediation_evidence_digest: str,
        proof: AuthenticationProof,
    ) -> dict[str, object]:
        """Return one exact first-attempt authorisation failure to READY."""

        now = self._clock()
        authentication = self._authenticator.authenticate(proof, now=now)
        authentication.require_current(now)
        if authentication.principal_id != HERMES_COMMAND_PRINCIPAL:
            raise PermissionError(
                "Graphiti admission authorisation recovery requires the "
                "Hermes principal"
            )
        if not ingest_id or not proposal_key:
            raise GraphitiAdmissionConsumerError(
                "Graphiti admission recovery identity is incomplete"
            )
        try:
            validate_sha256_digest(
                expected_request_digest, field="admission recovery request digest"
            )
            validate_sha256_digest(
                remediation_evidence_digest,
                field="admission recovery remediation evidence digest",
            )
        except (TypeError, ValueError) as exc:
            raise GraphitiAdmissionConsumerError(str(exc)) from exc

        command_type = _GRAPHITI_ADMISSION_AUTHORIZATION_RECOVERY_COMMAND_TYPE
        idempotency_key = digest_canonical(
            {"command_type": command_type, "proposal_key": proposal_key}
        )
        command_digest = digest_canonical(
            {
                "ingest_id": ingest_id,
                "proposal_key": proposal_key,
                "expected_request_digest": expected_request_digest,
                "remediation_evidence_digest": remediation_evidence_digest,
            }
        )
        recovered_at = now.to_text()
        store_path = (
            Path(assert_private_store(unpublished_store)).expanduser().resolve()
        )
        assert_private_store(str(store_path))
        connection = sqlite3.connect(store_path)
        try:
            apply_control_plane_sqlite_profile(connection)
            connection.execute("BEGIN IMMEDIATE")
            retained = connection.execute(
                "SELECT caller_principal,writer_principal,command_type,"
                "expected_mapping_digest,receipt_json FROM "
                "unpublished_reconciliation_commands WHERE idempotency_key=?",
                (idempotency_key,),
            ).fetchone()
            if retained is not None:
                if tuple(retained[:4]) != (
                    authentication.principal_id,
                    self.principal,
                    command_type,
                    command_digest,
                ):
                    raise GraphitiAdmissionConsumerError(
                        "Graphiti admission recovery identity was reused"
                    )
                receipt = _validate_graphiti_admission_recovery_receipt(
                    str(retained[4]),
                    idempotency_key=idempotency_key,
                    command_digest=command_digest,
                    caller_principal=authentication.principal_id,
                    writer_principal=self.principal,
                )
                connection.commit()
                return receipt

            row = connection.execute(
                """
                SELECT queue.queue_seq,queue.source_revision_id,
                       queue.source_receipt_digest,queue.proposal_digest,
                       queue.proposal_kind,queue.request_json,queue.request_digest,
                       ingest.source_id,ingest.item_key,receipt.receipt_json
                FROM unpublished_graphiti_admission_queue AS queue
                JOIN unpublished_graphiti_ingest AS ingest USING(ingest_id)
                JOIN unpublished_graphiti_receipts AS receipt USING(ingest_id)
                WHERE queue.proposal_key=? AND queue.ingest_id=?
                  AND queue.request_digest=? AND queue.state='DEAD_LETTER'
                  AND queue.attempt_count=1 AND queue.last_error=?
                  AND queue.claim_owner IS NULL AND queue.claim_until IS NULL
                  AND ingest.outcome='COMPLETE'
                  AND ingest.receipt_digest=queue.source_receipt_digest
                  AND NOT EXISTS(
                    SELECT 1 FROM unpublished_graphiti_admission_decisions
                    WHERE proposal_key=queue.proposal_key)
                  AND NOT EXISTS(
                    SELECT 1 FROM unpublished_graphiti_projection_receipts
                    WHERE proposal_key=queue.proposal_key)
                  AND NOT EXISTS(
                    SELECT 1 FROM unpublished_graphiti_projection_tombstones
                    WHERE proposal_key=queue.proposal_key)
                  AND NOT EXISTS(
                    SELECT 1 FROM unpublished_graphiti_admission_receipt_failures
                    WHERE ingest_id=queue.ingest_id)
                """,
                (
                    proposal_key, ingest_id, expected_request_digest,
                    _GRAPHITI_ADMISSION_AUTHORIZATION_ERROR,
                ),
            ).fetchone()
            if row is None:
                raise GraphitiAdmissionConsumerError(
                    "Graphiti admission recovery preconditions differ"
                )
            (
                queue_seq, source_revision_id, source_receipt_digest,
                proposal_digest, proposal_kind, request_json, request_digest,
                source_id, item_key, terminal_json,
            ) = row
            try:
                retained_request = json.loads(str(request_json))
                terminal = json.loads(str(terminal_json))
                if not isinstance(retained_request, dict) or not isinstance(
                    terminal, dict
                ):
                    raise TypeError("retained evidence must be objects")
                request = graphiti_admission_request_from_value(retained_request)
                unsigned_terminal = dict(terminal)
                terminal_digest = unsigned_terminal.pop("receipt_digest", None)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise GraphitiAdmissionConsumerError(
                    "Graphiti admission recovery evidence is invalid"
                ) from exc
            if (
                digest_bytes(canonical_json_bytes(retained_request)) != request_digest
                or request.queue_seq != queue_seq
                or request.proposal_key != proposal_key
                or request.source_receipt_digest != source_receipt_digest
                or request.proposal.digest != proposal_digest
                or request.proposal.kind.value != proposal_kind
                or request.source_lineage.get("ingest_id") != ingest_id
                or request.source_lineage.get("source_id") != source_id
                or request.source_lineage.get("item_key") != item_key
                or request.source_lineage.get("revision_id") != source_revision_id
                or terminal_digest != source_receipt_digest
                or digest_bytes(canonical_json_bytes(unsigned_terminal))
                != source_receipt_digest
                or terminal.get("ingest_id") != ingest_id
                or terminal.get("source_id") != source_id
                or terminal.get("item_key") != item_key
                or terminal.get("revision_id") != source_revision_id
                or terminal.get("outcome") != "COMPLETE"
                or terminal.get("workspace_group") != GRAPHITI_WORKSPACE_GROUP
                or terminal.get("profile") != "EVALUATION"
            ):
                raise GraphitiAdmissionConsumerError(
                    "Graphiti admission recovery evidence binding differs"
                )

            receipt: dict[str, object] = {
                "schema": _GRAPHITI_ADMISSION_AUTHORIZATION_RECOVERY_SCHEMA,
                "idempotency_key": idempotency_key,
                "command_digest": command_digest,
                "command_type": command_type,
                "authenticated_principal": authentication.principal_id,
                "writer_principal": self.principal,
                "ingest_id": ingest_id,
                "proposal_key": proposal_key,
                "request_digest": request_digest,
                "source_receipt_digest": source_receipt_digest,
                "proposal_digest": proposal_digest,
                "remediation_evidence_digest": remediation_evidence_digest,
                "prior_failure": {
                    "state": "DEAD_LETTER",
                    "attempt_count": 1,
                    "last_error": _GRAPHITI_ADMISSION_AUTHORIZATION_ERROR,
                    "claim_owner": None,
                    "claim_until": None,
                },
                "recovered_state": "READY",
                "recovered_at": recovered_at,
            }
            receipt["receipt_digest"] = digest_canonical(receipt)
            updated = connection.execute(
                "UPDATE unpublished_graphiti_admission_queue "
                "SET state='READY',updated_at=? WHERE proposal_key=? "
                "AND state='DEAD_LETTER' AND attempt_count=1 AND last_error=? "
                "AND claim_owner IS NULL AND claim_until IS NULL",
                (recovered_at, proposal_key, _GRAPHITI_ADMISSION_AUTHORIZATION_ERROR),
            )
            if updated.rowcount != 1:
                raise GraphitiAdmissionConsumerError(
                    "Graphiti admission recovery lost its exact failure state"
                )
            connection.execute(
                "INSERT INTO unpublished_reconciliation_commands("
                "idempotency_key,caller_principal,writer_principal,command_type,"
                "expected_mapping_digest,receipt_json,at) VALUES(?,?,?,?,?,?,?)",
                (
                    idempotency_key, authentication.principal_id, self.principal,
                    command_type, command_digest,
                    canonical_json_bytes(receipt).decode("utf-8"), recovered_at,
                ),
            )
            connection.commit()
            return receipt
        except Exception:
            if connection.in_transaction:
                connection.rollback()
            raise
        finally:
            connection.close()

    def reconcile_effective_revision_backlog(
        self,
        *,
        proving_store: str,
        unpublished_store: str,
        dry_run_receipt: Mapping[str, object],
        receipt_path: Path | None = None,
        backup_dir: Path | None = None,
        allow_canonical_mutation: bool = False,
        evaluated_at: datetime | None = None,
        idempotency_key: str,
        expected_mapping_digest: str,
        proof: AuthenticationProof,
        mode: Literal["live"] = "live",
    ) -> backlog.BacklogReconciliationReceipt:
        if mode != "live":
            raise ValueError("command-service mutation is live-only")
        now = self._clock()
        authentication = self._authenticator.authenticate(proof, now=now)
        authentication.require_current(now)
        command = backlog._ReconciliationCommand(
            caller_principal=authentication.principal_id,
            writer_principal=backlog.COMMAND_SERVICE_PRINCIPAL,
            command_type=backlog.RECONCILE_COMMAND_TYPE,
            idempotency_key=idempotency_key,
            expected_mapping_digest=expected_mapping_digest,
        )
        backlog.refuse_canonical_write(
            proving_store, allow_canonical_mutation=allow_canonical_mutation
        )
        backlog.refuse_canonical_write(
            unpublished_store, allow_canonical_mutation=allow_canonical_mutation
        )
        if backup_dir is None:
            raise backlog.BacklogReconciliationError(
                "G3: live migration requires a backup directory"
            )
        backlog._assert_command_authority(command)
        proving_path = Path(proving_store)
        unpublished_path = Path(unpublished_store)
        backup_root = Path(backup_dir)
        store_binding = backlog._store_pair_identity(
            proving_store, unpublished_store
        )

        # Recovery is itself a mutation, so it stays behind authentication.
        backlog._restore_incomplete_dual_store(
            proving_path, unpublished_path, backup_root
        )
        evaluated = backlog._as_utc(evaluated_at or datetime.now(tz=UTC))
        plan, _proving_before, _unpublished_before = backlog._plan_reconciliation(
            proving_store, unpublished_store, evaluated_at=evaluated
        )
        backlog._assert_g2(plan, dry_run_receipt, store_binding=store_binding)
        backlog._assert_command(command, plan, dry_run_receipt)
        completed = backlog._load_completed_command(
            unpublished_store, command, store_binding=store_binding
        )
        if completed is not None:
            backlog._write_receipt(receipt_path, completed)
            return completed

        proving_backup = backup_root / "proving_store.sqlite3"
        unpublished_backup = backup_root / "unpublished_store.sqlite3"
        proving_backup_result = backlog._backup_store(proving_path, proving_backup)
        unpublished_backup_result = backlog._backup_store(
            unpublished_path, unpublished_backup
        )
        coordinator: dict[str, object] = {
            "mapping_digest": plan.mapping_digest,
            "idempotency_key": command.idempotency_key,
            "proving_store": backlog._store_identity(proving_path),
            "unpublished_store": backlog._store_identity(unpublished_path),
            "proving_backup": str(proving_backup.resolve()),
            "unpublished_backup": str(unpublished_backup.resolve()),
            "proving_backup_digest": proving_backup_result["digest"],
            "unpublished_backup_digest": unpublished_backup_result["digest"],
        }
        coordinator_path = backup_root / backlog.COORDINATOR_NAME
        backlog._write_coordinator(
            coordinator_path, {**coordinator, "status": "STARTED"}
        )

        def apply_mutations() -> tuple[int, backlog.BacklogReconciliationReceipt]:
            conn: sqlite3.Connection | None = sqlite3.connect(str(unpublished_path))
            try:
                backlog.apply_control_plane_sqlite_profile(conn, wal=False)
                conn.execute("ATTACH DATABASE ? AS proving", (str(proving_path),))
                backlog.apply_control_plane_sqlite_profile(
                    conn, wal=False, schema=backlog.PROVING_ATTACH_SCHEMA
                )

                def versions() -> tuple[int, int]:
                    return (
                        int(conn.execute("PRAGMA main.data_version").fetchone()[0]),
                        int(conn.execute("PRAGMA proving.data_version").fetchone()[0]),
                    )

                before_plan_versions = versions()
                live_plan = backlog._build_plan(
                    conn,
                    conn,
                    evaluated_at=evaluated,
                    proving_schema=backlog.PROVING_ATTACH_SCHEMA,
                )
                if versions() != before_plan_versions:
                    raise backlog.BacklogReconciliationError(
                        "G2: stores changed while planning"
                    )
                backlog._assert_g1(live_plan)
                backlog._assert_g2(
                    live_plan, dry_run_receipt, store_binding=store_binding
                )
                backlog._assert_g5(live_plan)
                backlog._assert_command(command, live_plan, dry_run_receipt)
                proving_before = backlog._census_proving(
                    conn, schema=backlog.PROVING_ATTACH_SCHEMA
                )
                unpublished_before = backlog._census_unpublished(conn)
                conn.execute("BEGIN IMMEDIATE")
                if versions() != before_plan_versions:
                    raise backlog.BacklogReconciliationError(
                        "G2: stores changed before mutation"
                    )
                deadline = time.monotonic() + backlog.LIVE_TRANSACTION_TIMEOUT_SECONDS
                conn.set_progress_handler(lambda: time.monotonic() >= deadline, 1_000)
                backlog._ensure_landed_schema(conn)
                remapped = backlog._apply_proving(
                    conn, live_plan, schema=backlog.PROVING_ATTACH_SCHEMA
                )
                remapped += backlog._apply_remap_rows(conn, live_plan)
                no_loss = backlog._no_loss_proof(
                    proving_before=proving_before,
                    unpublished_before=unpublished_before,
                    proving_after=backlog._census_proving(
                        conn, schema=backlog.PROVING_ATTACH_SCHEMA
                    ),
                    unpublished_after=backlog._census_unpublished(conn),
                )
                if no_loss["lost"]:
                    raise backlog.BacklogReconciliationError(
                        "G3: append-only census lost records"
                    )
                rerun_changes = backlog._apply_proving(
                    conn, live_plan, schema=backlog.PROVING_ATTACH_SCHEMA
                ) + backlog._apply_remap_rows(conn, live_plan)
                if rerun_changes:
                    raise backlog.BacklogReconciliationError(
                        "G4: rerun produced further remapping"
                    )
                receipt = backlog._receipt_from_plan(
                    live_plan,
                    mode="live",
                    mutated=True,
                    remapped_count=remapped,
                    no_loss_proof=no_loss,
                    gates={key: "pass" for key in ("G1", "G2", "G3", "G4", "G5")},
                    store_binding=store_binding,
                    command=command.as_dict(),
                )
                backlog._retain_receipt(conn, receipt.as_dict())
                backlog._record_command(conn, command, receipt)
                conn.commit()
                return remapped, receipt
            except Exception as exc:
                if conn is not None and conn.in_transaction:
                    conn.rollback()
                if conn is not None:
                    conn.close()
                    conn = None
                if isinstance(exc, sqlite3.OperationalError) and "interrupted" in str(
                    exc
                ):
                    raise backlog.BacklogReconciliationError(
                        "live reconciliation exceeded the five-second transaction limit"
                    ) from exc
                raise
            finally:
                if conn is not None:
                    conn.set_progress_handler(None, 0)
                    conn.close()

        try:
            backlog._set_journal_mode(proving_path, "DELETE")
            backlog._set_journal_mode(unpublished_path, "DELETE")
            remapped, receipt = apply_mutations()
        except Exception:
            backlog._restore_wal_profiles(proving_path, unpublished_path)
            backlog._write_coordinator(
                coordinator_path, {**coordinator, "status": "ABORTED"}
            )
            raise
        coordinator["remapped_count"] = remapped
        backlog._write_coordinator(
            coordinator_path, {**coordinator, "status": "COMMITTED"}
        )
        backlog._restore_wal_profiles(proving_path, unpublished_path)
        backlog._write_coordinator(
            coordinator_path, {**coordinator, "status": "COMPLETE"}
        )
        backlog._write_receipt(receipt_path, receipt)
        return receipt


__all__ = ["ControlPlaneCommandService"]
