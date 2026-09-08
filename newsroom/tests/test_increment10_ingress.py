from __future__ import annotations

import sqlite3
from hashlib import sha256
from pathlib import Path

import pytest

from newsroom.authority.canonical import canonical_json_bytes
from newsroom.authority.story_candidate_system import _create_story_candidate_read_port
from newsroom.increment10.ingress import (
    NON_PUBLIC_EVIDENCE_INTAKE_BOUNDARY,
    EvidenceIntakeError,
    EvidenceIntakeIngress,
    IntakeAcknowledgement,
    open_evidence_intake_ingress,
)
from newsroom.tests import test_increment6e2_candidate_store as candidate_fixture


def _candidates(tmp_path: Path, *record_ids: str):
    adapter = candidate_fixture._Adapter(tmp_path)
    location = adapter.create_location()
    handle = adapter.open_handle(location)
    versions = []
    for record_id in record_ids:
        handle.submit(candidate_fixture._generic(record_id))
        row = handle._row(record_id)
        assert row is not None
        versions.append(handle._opened().load_version(str(row[1])))
    handle.close()

    collaborators = candidate_fixture._collaborators(location.seed)
    connection = sqlite3.connect(location.seed[1], isolation_level=None)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    port = _create_story_candidate_read_port(
        connection,
        retrieval_authority=collaborators["retrieval_authority"],
        authenticator=collaborators["authenticator"],
        command_registry=collaborators["command_registry"],
        payload_schemas=collaborators["payload_schemas"],
        clock=collaborators["clock"],
    )
    return connection, port, tuple(versions)


def _candidate(tmp_path: Path):
    connection, port, versions = _candidates(tmp_path, "record-1")
    return connection, port, versions[0]


def _receive(ingress, connection, port, version, *, request_id: str, at: int = 1):
    connection.execute("BEGIN IMMEDIATE")
    try:
        return ingress.receive(
            port,
            candidate_version_id=version.version_id,
            expected_governing_manifest_digest=(
                version.governing_manifest.canonical_digest
            ),
            boundary_id=NON_PUBLIC_EVIDENCE_INTAKE_BOUNDARY,
            request_id=request_id,
            received_epoch_seconds=at,
        )
    finally:
        connection.execute("ROLLBACK")


def test_receive_binds_exact_authority_version_and_is_receipt_only(
    tmp_path: Path,
) -> None:
    candidate_connection, port, version = _candidate(tmp_path)
    ingress = open_evidence_intake_ingress(tmp_path / "intake.sqlite3")

    acknowledgement = _receive(
        ingress, candidate_connection, port, version, request_id="request-1"
    )

    assert acknowledgement.candidate_version_id == version.version_id
    assert (
        acknowledgement.governing_manifest_digest
        == version.governing_manifest.canonical_digest
    )
    assert acknowledgement.receipt_only is True
    assert acknowledgement.evidence_authority is False
    assert acknowledgement.publication_authority is False
    assert acknowledgement.runtime_authority is False
    assert ingress.receipt(acknowledgement.receipt_id) == acknowledgement
    ingress.close()
    candidate_connection.close()


def test_ingress_construction_is_factory_private() -> None:
    connection = sqlite3.connect(":memory:", isolation_level=None)
    with pytest.raises(EvidenceIntakeError, match="construction is private"):
        EvidenceIntakeIngress(
            object(), connection, NON_PUBLIC_EVIDENCE_INTAKE_BOUNDARY
        )
    connection.close()


def test_factory_rejects_ephemeral_sqlite() -> None:
    with pytest.raises(EvidenceIntakeError, match="file-backed"):
        open_evidence_intake_ingress(":memory:")


def test_acknowledgement_rejects_forged_handoff_and_receipt_identity() -> None:
    value = {
        "request_id": "request-1",
        "handoff_id": "intake-handoff:sha256:" + "0" * 64,
        "receipt_id": "intake-receipt:sha256:" + "1" * 64,
        "candidate_version_id": "candidate-version-1",
        "candidate_version_digest": "sha256:" + "2" * 64,
        "governing_manifest_digest": "sha256:" + "3" * 64,
        "boundary_id": NON_PUBLIC_EVIDENCE_INTAKE_BOUNDARY,
        "received_epoch_seconds": 1,
    }
    acknowledgement_id = (
        "intake-acknowledgement:sha256:"
        + sha256(
            canonical_json_bytes(
                {
                    name: value[name]
                    for name in (
                        "request_id",
                        "handoff_id",
                        "receipt_id",
                        "received_epoch_seconds",
                    )
                }
            )
        ).hexdigest()
    )

    with pytest.raises(EvidenceIntakeError, match="Handoff identity"):
        IntakeAcknowledgement(acknowledgement_id, **value)


def test_manifest_mismatch_and_public_boundary_fail_before_receipt(
    tmp_path: Path,
) -> None:
    candidate_connection, port, version = _candidate(tmp_path)
    ingress = open_evidence_intake_ingress(tmp_path / "intake.sqlite3")

    candidate_connection.execute("BEGIN IMMEDIATE")
    with pytest.raises(EvidenceIntakeError, match="manifest"):
        ingress.receive(
            port,
            candidate_version_id=version.version_id,
            expected_governing_manifest_digest="sha256:" + "0" * 64,
            boundary_id=NON_PUBLIC_EVIDENCE_INTAKE_BOUNDARY,
            request_id="request-1",
            received_epoch_seconds=1,
        )
    with pytest.raises(EvidenceIntakeError, match="non-public boundary"):
        ingress.receive(
            port,
            candidate_version_id=version.version_id,
            expected_governing_manifest_digest=(
                version.governing_manifest.canonical_digest
            ),
            boundary_id="public://publisher",
            request_id="request-1",
            received_epoch_seconds=1,
        )
    candidate_connection.execute("ROLLBACK")

    assert ingress.receipt_count == 0
    ingress.close()
    candidate_connection.close()


def test_reopen_reconciles_lost_ack_and_replays_requests_idempotently(
    tmp_path: Path,
) -> None:
    candidate_connection, port, version = _candidate(tmp_path)
    database = tmp_path / "intake.sqlite3"
    ingress = open_evidence_intake_ingress(database)
    lost = _receive(
        ingress, candidate_connection, port, version, request_id="request-lost", at=1
    )
    assert (
        _receive(
            ingress,
            candidate_connection,
            port,
            version,
            request_id="request-lost",
            at=2,
        )
        == lost
    )
    ingress.close()

    reopened = open_evidence_intake_ingress(database)
    reconciled = _receive(
        reopened,
        candidate_connection,
        port,
        version,
        request_id="request-retry",
        at=3,
    )
    assert reconciled == lost
    assert reopened.receipt_count == 1
    assert reopened.attempt_count == 2
    reopened.close()
    candidate_connection.close()


def test_replay_chronology_cannot_make_reopen_fail(tmp_path: Path) -> None:
    candidate_connection, port, version = _candidate(tmp_path)
    database = tmp_path / "intake.sqlite3"
    ingress = open_evidence_intake_ingress(database)
    _receive(
        ingress, candidate_connection, port, version, request_id="request-1", at=2
    )

    with pytest.raises(EvidenceIntakeError, match="precedes"):
        _receive(
            ingress,
            candidate_connection,
            port,
            version,
            request_id="request-2",
            at=1,
        )

    assert ingress.attempt_count == 1
    ingress.close()
    open_evidence_intake_ingress(database).close()
    candidate_connection.close()


def test_request_identity_cannot_cross_candidate_versions(tmp_path: Path) -> None:
    candidate_connection, port, versions = _candidates(
        tmp_path, "record-1", "record-2"
    )
    ingress = open_evidence_intake_ingress(tmp_path / "intake.sqlite3")
    _receive(
        ingress,
        candidate_connection,
        port,
        versions[0],
        request_id="request-1",
    )

    with pytest.raises(EvidenceIntakeError, match="request identity conflicts"):
        _receive(
            ingress,
            candidate_connection,
            port,
            versions[1],
            request_id="request-1",
            at=2,
        )

    assert ingress.receipt_count == 1
    assert ingress.attempt_count == 1
    ingress.close()
    candidate_connection.close()


def test_open_handle_rejects_cross_receipt_corruption_without_history_change(
    tmp_path: Path,
) -> None:
    candidate_connection, port, versions = _candidates(
        tmp_path, "record-1", "record-2"
    )
    database = tmp_path / "intake.sqlite3"
    ingress = open_evidence_intake_ingress(database)
    first = _receive(
        ingress,
        candidate_connection,
        port,
        versions[0],
        request_id="request-1",
        at=1,
    )
    second = _receive(
        ingress,
        candidate_connection,
        port,
        versions[1],
        request_id="request-2",
        at=2,
    )

    external = sqlite3.connect(database)
    second_bytes = external.execute(
        "SELECT canonical_bytes FROM evidence_intake_acknowledgements "
        "WHERE acknowledgement_id=?",
        (second.acknowledgement_id,),
    ).fetchone()[0]
    external.execute(
        "UPDATE evidence_intake_acknowledgements SET canonical_bytes=? "
        "WHERE acknowledgement_id=?",
        (second_bytes, first.acknowledgement_id),
    )
    external.commit()
    before = tuple(
        external.execute(
            "SELECT request_id,handoff_id,acknowledgement_id,"
            "observed_epoch_seconds FROM evidence_intake_attempts "
            "ORDER BY request_id"
        )
    )
    external.close()

    with pytest.raises(EvidenceIntakeError, match="retained acknowledgement"):
        ingress.receipt(first.receipt_id)
    with pytest.raises(EvidenceIntakeError, match="retained acknowledgement"):
        _receive(
            ingress,
            candidate_connection,
            port,
            versions[0],
            request_id="request-1",
            at=3,
        )
    with pytest.raises(EvidenceIntakeError, match="retained acknowledgement"):
        _receive(
            ingress,
            candidate_connection,
            port,
            versions[0],
            request_id="request-retry",
            at=3,
        )

    assert ingress.receipt_count == 2
    assert ingress.attempt_count == 2
    unchanged = sqlite3.connect(database)
    after = tuple(
        unchanged.execute(
            "SELECT request_id,handoff_id,acknowledgement_id,"
            "observed_epoch_seconds FROM evidence_intake_attempts "
            "ORDER BY request_id"
        )
    )
    unchanged.close()
    assert after == before
    ingress.close()
    candidate_connection.close()


def test_reopen_rejects_retained_logical_corruption(tmp_path: Path) -> None:
    candidate_connection, port, version = _candidate(tmp_path)
    database = tmp_path / "intake.sqlite3"
    ingress = open_evidence_intake_ingress(database)
    acknowledgement = _receive(
        ingress, candidate_connection, port, version, request_id="request-1"
    )
    ingress.close()

    connection = sqlite3.connect(database)
    connection.execute(
        "UPDATE evidence_intake_handoffs SET governing_manifest_digest=? "
        "WHERE handoff_id=?",
        ("sha256:" + "f" * 64, acknowledgement.handoff_id),
    )
    connection.commit()
    connection.close()

    with pytest.raises(EvidenceIntakeError, match="retained intake"):
        open_evidence_intake_ingress(database)
    candidate_connection.close()


def test_reopen_rejects_handoff_without_acknowledgement(tmp_path: Path) -> None:
    candidate_connection, port, version = _candidate(tmp_path)
    database = tmp_path / "intake.sqlite3"
    ingress = open_evidence_intake_ingress(database)
    _receive(ingress, candidate_connection, port, version, request_id="request-1")
    ingress.close()

    connection = sqlite3.connect(database)
    connection.execute("DELETE FROM evidence_intake_attempts")
    connection.execute("DELETE FROM evidence_intake_acknowledgements")
    connection.commit()
    connection.close()

    with pytest.raises(EvidenceIntakeError, match="acknowledgement coverage"):
        open_evidence_intake_ingress(database)
    candidate_connection.close()


def test_out_of_range_timestamp_cannot_poison_following_receive(tmp_path: Path) -> None:
    candidate_connection, port, version = _candidate(tmp_path)
    ingress = open_evidence_intake_ingress(tmp_path / "intake.sqlite3")
    for index, invalid in enumerate((True, -1, 1.5, 2**53, 2**63)):
        with pytest.raises(EvidenceIntakeError, match="received_epoch_seconds"):
            _receive(
                ingress, candidate_connection, port, version,
                request_id=f"invalid-{index}", at=invalid,
            )
        assert ingress.receipt_count == ingress.attempt_count == 0
    accepted = _receive(
        ingress, candidate_connection, port, version, request_id="valid", at=1
    )
    assert ingress.receipt(accepted.receipt_id) == accepted
    ingress.close()
    candidate_connection.close()


def test_unexpected_acknowledgement_failure_rolls_back_handoff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from newsroom.increment10 import ingress as module

    candidate_connection, port, version = _candidate(tmp_path)
    database = tmp_path / "intake.sqlite3"
    receiver = open_evidence_intake_ingress(database)
    original_identity = module._identity
    fail_once = True

    def interrupted_identity(prefix: str, value: object) -> str:
        nonlocal fail_once
        if prefix == "intake-acknowledgement" and fail_once:
            fail_once = False
            raise RuntimeError("interrupted after Handoff insert")
        return original_identity(prefix, value)

    monkeypatch.setattr(module, "_identity", interrupted_identity)
    with pytest.raises(RuntimeError, match="after Handoff insert"):
        _receive(receiver, candidate_connection, port, version, request_id="failed")
    assert receiver.receipt_count == receiver.attempt_count == 0
    accepted = _receive(
        receiver, candidate_connection, port, version, request_id="valid", at=2
    )
    receiver.close()
    reopened = open_evidence_intake_ingress(database)
    assert reopened.receipt(accepted.receipt_id) == accepted
    assert reopened.receipt_count == reopened.attempt_count == 1
    reopened.close()
    candidate_connection.close()


def test_unexpected_receipt_failure_releases_transaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from newsroom.increment10 import ingress as module

    candidate_connection, port, version = _candidate(tmp_path)
    database = tmp_path / "intake.sqlite3"
    receiver = open_evidence_intake_ingress(database)
    accepted = _receive(receiver, candidate_connection, port, version, request_id="first")
    original_identity = module._identity

    def interrupted_identity(prefix: str, value: object) -> str:
        raise RuntimeError("interrupted receipt validation")

    monkeypatch.setattr(module, "_identity", interrupted_identity)
    with pytest.raises(RuntimeError, match="interrupted receipt"):
        receiver.receipt(accepted.receipt_id)
    monkeypatch.setattr(module, "_identity", original_identity)
    other = sqlite3.connect(database, timeout=0)
    other.execute("BEGIN IMMEDIATE")
    other.rollback()
    other.close()
    assert receiver.receipt(accepted.receipt_id) == accepted
    assert _receive(
        receiver, candidate_connection, port, version, request_id="next", at=2
    ) == accepted
    assert receiver.receipt_count == 1
    assert receiver.attempt_count == 2
    receiver.close()
    candidate_connection.close()


def test_retained_secondary_attempt_rejects_out_of_range_timestamp(tmp_path: Path) -> None:
    candidate_connection, port, version = _candidate(tmp_path)
    database = tmp_path / "intake.sqlite3"
    receiver = open_evidence_intake_ingress(database)
    accepted = _receive(receiver, candidate_connection, port, version, request_id="first")
    other = sqlite3.connect(database)
    other.execute(
        "INSERT INTO evidence_intake_attempts VALUES (?,?,?,?)",
        ("corrupt", accepted.handoff_id, accepted.acknowledgement_id, 2**63 - 1),
    )
    other.commit()
    other.close()
    with pytest.raises(EvidenceIntakeError, match="received_epoch_seconds"):
        _receive(receiver, candidate_connection, port, version, request_id="corrupt", at=2)
    receiver.close()
    with pytest.raises(EvidenceIntakeError, match="received_epoch_seconds"):
        open_evidence_intake_ingress(database)
    candidate_connection.close()
