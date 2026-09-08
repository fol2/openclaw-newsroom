from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from newsroom.authority.auth import (
    AuthenticationProof,
    StaticAuthenticator,
    StaticPrincipal,
)
from newsroom.authority.canonical import digest_canonical
from newsroom.authority.types import UtcTimestamp
from newsroom.control_plane.command_service import ControlPlaneCommandService
from newsroom.control_plane.graphiti_admission import GraphitiAdmissionConsumerError
from newsroom.control_plane.store import connect
from newsroom.extraction.types import ExtractionProposalKind

from .test_graphiti_admission_consumer import (
    _Authority,
    _Projector,
    _Rights,
    _consumer,
    _draft,
    _seed_receipt,
)


NOW = datetime(2026, 9, 8, 12, tzinfo=UTC)
PROOF = AuthenticationProof(method="STATIC_TOKEN", credential="operator-token")
ERROR = "AuthorizationDenied: authorization denied: AUTHZ_SCOPE_MISSING"
REMEDIATION_DIGEST = digest_canonical({"remediation": "scope composition fixed"})


def _service(*, principal: str = "newsroom.hermes") -> ControlPlaneCommandService:
    return ControlPlaneCommandService(
        authenticator=StaticAuthenticator(
            credentials={"operator-token": StaticPrincipal(principal_id=principal)},
            authority_domain="newsroom.control-plane",
        ),
        clock=lambda: UtcTimestamp(NOW),
    )


def _failed_request(
    path: Path, *, attempt_count: int = 1, error: str = ERROR
) -> tuple[str, str, str]:
    connection = connect(str(path))
    draft = _draft("entity.0001", ExtractionProposalKind.ENTITY_MENTION)
    receipt = _seed_receipt(connection, draft)
    consumer = _consumer(
        connection,
        _Authority({draft.local_id: "ADMIT"}),
        _Projector(),
        _Rights(),
    )
    assert consumer.enqueue_complete_receipts() == 1
    proposal_key, request_digest = connection.execute(
        "SELECT proposal_key,request_digest "
        "FROM unpublished_graphiti_admission_queue"
    ).fetchone()
    connection.execute(
        "UPDATE unpublished_graphiti_admission_queue "
        "SET state='DEAD_LETTER',attempt_count=?,last_error=?",
        (attempt_count, error),
    )
    connection.commit()
    connection.close()
    return str(receipt["ingest_id"]), str(proposal_key), str(request_digest)


def _recover(
    path: Path,
    ingest_id: str,
    proposal_key: str,
    request_digest: str,
    *,
    service: ControlPlaneCommandService | None = None,
    remediation_digest: str = REMEDIATION_DIGEST,
) -> dict[str, object]:
    return (service or _service()).recover_graphiti_admission_authorization(
        unpublished_store=str(path),
        ingest_id=ingest_id,
        proposal_key=proposal_key,
        expected_request_digest=request_digest,
        remediation_evidence_digest=remediation_digest,
        proof=PROOF,
    )


def _resume(
    path: Path,
    ingest_id: str,
    proposal_key: str,
    request_digest: str,
    *,
    attempt_count: int,
    error: str,
    remediation_digest: str = REMEDIATION_DIGEST,
) -> dict[str, object]:
    return _service().resume_graphiti_admission(
        unpublished_store=str(path),
        ingest_id=ingest_id,
        proposal_key=proposal_key,
        expected_request_digest=request_digest,
        expected_attempt_count=attempt_count,
        expected_error=error,
        remediation_evidence_digest=remediation_digest,
        proof=PROOF,
    )


def test_exact_authorization_failure_recovers_once_and_replay_is_read_only(
    tmp_path: Path,
) -> None:
    path = tmp_path / "unpublished.sqlite3"
    ingest_id, proposal_key, request_digest = _failed_request(path)

    receipt = _recover(path, ingest_id, proposal_key, request_digest)

    connection = sqlite3.connect(path)
    recovered = connection.execute(
        "SELECT state,attempt_count,last_error,claim_owner,claim_until,updated_at "
        "FROM unpublished_graphiti_admission_queue WHERE proposal_key=?",
        (proposal_key,),
    ).fetchone()
    command = connection.execute(
        "SELECT caller_principal,writer_principal,command_type,receipt_json "
        "FROM unpublished_reconciliation_commands"
    ).fetchone()
    assert recovered == (
        "READY",
        1,
        ERROR,
        None,
        None,
        UtcTimestamp(NOW).to_text(),
    )
    assert command[:3] == (
        "newsroom.hermes",
        "newsroom.control-plane.command-service",
        "RECOVER_GRAPHITI_ADMISSION_AUTHORIZATION",
    )
    assert json.loads(command[3]) == receipt
    assert receipt["request_digest"] == request_digest
    assert receipt["remediation_evidence_digest"] == REMEDIATION_DIGEST
    assert receipt["prior_failure"] == {
        "state": "DEAD_LETTER",
        "attempt_count": 1,
        "last_error": ERROR,
        "claim_owner": None,
        "claim_until": None,
    }

    connection.execute(
        "UPDATE unpublished_graphiti_admission_queue "
        "SET state='DEAD_LETTER',attempt_count=2,updated_at='later' "
        "WHERE proposal_key=?",
        (proposal_key,),
    )
    connection.commit()
    connection.close()
    assert _recover(path, ingest_id, proposal_key, request_digest) == receipt
    connection = sqlite3.connect(path)
    assert connection.execute(
        "SELECT state,attempt_count,updated_at "
        "FROM unpublished_graphiti_admission_queue WHERE proposal_key=?",
        (proposal_key,),
    ).fetchone() == ("DEAD_LETTER", 2, "later")
    assert connection.execute(
        "SELECT COUNT(*) FROM unpublished_reconciliation_commands"
    ).fetchone()[0] == 1
    connection.close()

    with pytest.raises(GraphitiAdmissionConsumerError, match="identity was reused"):
        _recover(
            path,
            ingest_id,
            proposal_key,
            request_digest,
            remediation_digest=digest_canonical({"remediation": "different"}),
        )


def test_resume_binds_second_failure_epoch_and_replay_is_read_only(
    tmp_path: Path,
) -> None:
    path = tmp_path / "resume.sqlite3"
    error = (
        "GraphitiAdmissionConsumerError: entity decision command differs "
        "from the retained authority proposal"
    )
    ingest_id, proposal_key, request_digest = _failed_request(path)
    _recover(path, ingest_id, proposal_key, request_digest)
    connection = sqlite3.connect(path)
    connection.execute(
        "UPDATE unpublished_graphiti_admission_queue "
        "SET state='DEAD_LETTER',attempt_count=2,last_error=?",
        (error,),
    )
    connection.commit()
    connection.close()

    receipt = _resume(
        path,
        ingest_id,
        proposal_key,
        request_digest,
        attempt_count=2,
        error=error,
    )

    connection = sqlite3.connect(path)
    assert connection.execute(
        "SELECT state,attempt_count,last_error FROM "
        "unpublished_graphiti_admission_queue"
    ).fetchone() == ("READY", 2, error)
    assert receipt["schema"] == "newsroom.control-plane.graphiti-admission-resume.v1"
    assert receipt["command_type"] == "RESUME_GRAPHITI_ADMISSION"
    assert receipt["prior_failure"] == {
        "state": "DEAD_LETTER",
        "attempt_count": 2,
        "last_error": error,
        "claim_owner": None,
        "claim_until": None,
    }
    assert connection.execute(
        "SELECT COUNT(*) FROM unpublished_reconciliation_commands"
    ).fetchone()[0] == 2
    connection.execute(
        "UPDATE unpublished_graphiti_admission_queue "
        "SET state='DEAD_LETTER',attempt_count=3,last_error='later',updated_at='later'"
    )
    connection.commit()
    connection.close()

    assert _resume(
        path,
        ingest_id,
        proposal_key,
        request_digest,
        attempt_count=2,
        error=error,
    ) == receipt
    connection = sqlite3.connect(path)
    assert connection.execute(
        "SELECT state,attempt_count,last_error,updated_at FROM "
        "unpublished_graphiti_admission_queue"
    ).fetchone() == ("DEAD_LETTER", 3, "later", "later")
    connection.close()

    with pytest.raises(GraphitiAdmissionConsumerError, match="identity was reused"):
        _resume(
            path,
            ingest_id,
            proposal_key,
            request_digest,
            attempt_count=2,
            error=error,
            remediation_digest=digest_canonical({"remediation": "changed"}),
        )


@pytest.mark.parametrize("stale", ("attempt", "error", "request"))
def test_resume_rejects_stale_failure_epoch(tmp_path: Path, stale: str) -> None:
    path = tmp_path / f"stale-{stale}.sqlite3"
    error = "GraphitiAdmissionConsumerError: fixed failure"
    ingest_id, proposal_key, request_digest = _failed_request(
        path, attempt_count=2, error=error
    )

    with pytest.raises(GraphitiAdmissionConsumerError):
        _resume(
            path,
            ingest_id,
            proposal_key,
            (
                digest_canonical({"request": "stale"})
                if stale == "request"
                else request_digest
            ),
            attempt_count=1 if stale == "attempt" else 2,
            error="stale" if stale == "error" else error,
        )

    connection = sqlite3.connect(path)
    assert connection.execute(
        "SELECT state,attempt_count,last_error FROM "
        "unpublished_graphiti_admission_queue"
    ).fetchone() == ("DEAD_LETTER", 2, error)
    assert connection.execute(
        "SELECT COUNT(*) FROM unpublished_reconciliation_commands"
    ).fetchone()[0] == 0
    connection.close()


def test_recovery_requires_authenticated_hermes(tmp_path: Path) -> None:
    path = tmp_path / "unpublished.sqlite3"
    ingest_id, proposal_key, request_digest = _failed_request(path)

    with pytest.raises(PermissionError, match="Hermes principal"):
        _recover(
            path,
            ingest_id,
            proposal_key,
            request_digest,
            service=_service(principal="newsroom.other"),
        )

    connection = sqlite3.connect(path)
    assert connection.execute(
        "SELECT state FROM unpublished_graphiti_admission_queue"
    ).fetchone()[0] == "DEAD_LETTER"
    connection.close()


@pytest.mark.parametrize(
    "variation",
    (
        "different_key",
        "different_digest",
        "different_error",
        "different_state",
        "different_attempt",
        "active_claim",
        "decision",
        "projection",
        "receipt_integrity_hold",
    ),
)
def test_recovery_denies_any_non_exact_failure(
    tmp_path: Path, variation: str
) -> None:
    path = tmp_path / f"{variation}.sqlite3"
    ingest_id, proposal_key, request_digest = _failed_request(path)
    requested_key = proposal_key
    requested_digest = request_digest
    connection = sqlite3.connect(path)
    if variation == "different_key":
        requested_key = "missing-proposal-key"
    elif variation == "different_digest":
        requested_digest = digest_canonical({"request": "different"})
    elif variation == "different_error":
        connection.execute(
            "UPDATE unpublished_graphiti_admission_queue SET last_error='other'"
        )
    elif variation == "different_state":
        connection.execute(
            "UPDATE unpublished_graphiti_admission_queue SET state='READY'"
        )
    elif variation == "different_attempt":
        connection.execute(
            "UPDATE unpublished_graphiti_admission_queue SET attempt_count=2"
        )
    elif variation == "active_claim":
        connection.execute(
            "UPDATE unpublished_graphiti_admission_queue "
            "SET claim_owner='worker',claim_until='2026-09-08T13:00:00Z'"
        )
    elif variation == "decision":
        connection.execute(
            "INSERT INTO unpublished_graphiti_admission_decisions VALUES("
            "?,'HOLD','decision-1',1,'FIXTURE',?,?,?,?)",
            (
                proposal_key,
                digest_canonical({"authority": "receipt"}),
                "{}",
                digest_canonical({}),
                "2026-09-08T11:00:00Z",
            ),
        )
    elif variation == "projection":
        connection.execute(
            "INSERT INTO unpublished_graphiti_projection_receipts VALUES("
            "?,'effect-1',1,'family','generation','v1','ADMITTED',?,?,?)",
            (
                proposal_key,
                "{}",
                digest_canonical({}),
                "2026-09-08T11:00:00Z",
            ),
        )
    else:
        connection.execute(
            "INSERT INTO unpublished_graphiti_admission_receipt_failures "
            "VALUES(?,?,'QUEUE_INTEGRITY_INVALID','fixture',1,?,?)",
            (
                ingest_id,
                digest_canonical({"receipt": "failed"}),
                "2026-09-08T11:00:00Z",
                "2026-09-08T11:00:00Z",
            ),
        )
    connection.commit()
    connection.close()

    with pytest.raises(GraphitiAdmissionConsumerError):
        _recover(path, ingest_id, requested_key, requested_digest)

    connection = sqlite3.connect(path)
    assert connection.execute(
        "SELECT COUNT(*) FROM unpublished_reconciliation_commands"
    ).fetchone()[0] == 0
    connection.close()


def test_receipt_insert_failure_rolls_back_queue_recovery(tmp_path: Path) -> None:
    path = tmp_path / "rollback.sqlite3"
    error = "GraphitiAdmissionConsumerError: fixed failure"
    ingest_id, proposal_key, request_digest = _failed_request(
        path, attempt_count=2, error=error
    )
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TRIGGER reject_recovery_receipt "
        "BEFORE INSERT ON unpublished_reconciliation_commands "
        "BEGIN SELECT RAISE(ABORT,'fixture receipt failure'); END"
    )
    connection.commit()
    connection.close()

    with pytest.raises(sqlite3.IntegrityError, match="fixture receipt failure"):
        _resume(
            path,
            ingest_id,
            proposal_key,
            request_digest,
            attempt_count=2,
            error=error,
        )

    connection = sqlite3.connect(path)
    assert connection.execute(
        "SELECT state,attempt_count,last_error "
        "FROM unpublished_graphiti_admission_queue"
    ).fetchone() == ("DEAD_LETTER", 2, error)
    assert connection.execute(
        "SELECT COUNT(*) FROM unpublished_reconciliation_commands"
    ).fetchone()[0] == 0
    connection.close()
