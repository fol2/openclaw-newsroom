from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from newsroom.authority.story_candidate_system import _create_story_candidate_read_port
from newsroom.increment10.ingress import (
    NON_PUBLIC_EVIDENCE_INTAKE_BOUNDARY,
    EvidenceIntakeError,
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
