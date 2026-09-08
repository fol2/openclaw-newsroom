from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from newsroom.authority import ObjectAdmissionRequest
from newsroom.authority.canonical import canonical_json_bytes
from newsroom.control_plane.native_publication import (
    NativePublicationBindings,
    NativePublicationController,
)
from newsroom.increment10.editorial import EditorialHold
from newsroom.increment10.ingress import open_evidence_intake_ingress
from newsroom.increment10.private_serving import open_private_serving_read_port
from newsroom.tests.authority_helpers import proof
from newsroom.tests.test_increment10_editorial import (
    _decision,
    _evidence_facade,
    _ready_package,
)
from newsroom.tests.test_increment10_ingress import _candidate, _receive
from newsroom.tests.test_increment10_private_serving import _open


def _bindings(tmp_path: Path, registries, hydration, definitions, commands):
    return NativePublicationBindings(
        target_path=tmp_path / "private-serving.sqlite3",
        reader_principal_id="principal.alpha",
        authority_domain="newsroom.authority",
        editorial_controller_principal_id="principal.alpha",
        story_principal_id="principal.alpha",
        publication_controller_principal_id="principal.alpha",
        serving_adapter_principal_id="principal.alpha",
        editorial_policy_bundle_digest="sha256:" + "a" * 64,
        editorial_decision_hydration_policy_digest=(
            registries[3][0].contract_digest
        ),
        editorial_story_hydration_policy_digest=registries[3][1].contract_digest,
        editorial_decision_admission_definition_digest=registries[4][0].digest,
        editorial_decision_command_definition_digest=registries[5].digest,
        editorial_story_command_definition_digest=registries[6].digest,
        editorial_story_admission_definition_digest=registries[4][1].digest,
        publication_authorisation_policy_digest="sha256:" + "8" * 64,
        target_id="newsroom-app-serving-launch",
        target_policy_digest="sha256:" + "9" * 64,
        publication_surface_hydration_policy_digest=(
            registries[7][0].contract_digest
        ),
        publication_transaction_hydration_policy_digest=(
            registries[7][1].contract_digest
        ),
        publication_surface_admission_definition_digest=registries[8][0].digest,
        publication_transaction_admission_definition_digest=registries[8][1].digest,
        publication_command_definition_digest=registries[9].digest,
        target_context_digest="sha256:" + "a" * 64,
        serving_attempt_hydration_policy_digest=hydration[0].contract_digest,
        serving_evidence_hydration_policy_digest=hydration[1].contract_digest,
        serving_attempt_admission_definition_digest=definitions[0].digest,
        serving_evidence_admission_definition_digest=definitions[1].digest,
        serving_attempt_command_definition_digest=commands[0].digest,
        serving_evidence_command_definition_digest=commands[1].digest,
    )


def test_native_publication_replays_to_exact_ack_only_rows(tmp_path: Path) -> None:
    candidate_connection, candidate_port, version = _candidate(tmp_path)
    ingress = open_evidence_intake_ingress(tmp_path / "intake.sqlite3")
    acknowledgement = _receive(
        ingress,
        candidate_connection,
        candidate_port,
        version,
        request_id="request-1",
    )
    system, registries, hydration, definitions, commands = _open(
        tmp_path / "objects.sqlite3"
    )
    evidence_packages = _evidence_facade(system, ingress, registries)
    passage, package, records = _ready_package(version)
    source = system.objects.admit(
        ObjectAdmissionRequest("evidence.source", "source-1"),
        passage.encode(),
        proof=proof(),
    ).admission
    record_ids = tuple(
        system.objects.admit(
            ObjectAdmissionRequest("evidence.record", f"record-{index}"),
            canonical_json_bytes(record),
            proof=proof(),
        ).admission.admission_id
        for index, record in enumerate(records)
    )
    candidate_connection.execute("BEGIN IMMEDIATE")
    retained = evidence_packages.retain(
        package,
        receipt_id=acknowledgement.receipt_id,
        candidate_port=candidate_port,
        source_admission_ids=(source.admission_id,),
        record_admission_ids=record_ids,
        proof=proof(),
    )
    bindings = _bindings(tmp_path, registries, hydration, definitions, commands)
    controller = NativePublicationController(
        objects=system.objects,
        commands=system.commands,
        events=system.events,
        candidate_port=candidate_port,
        evidence_packages=evidence_packages,
        bindings=bindings,
    )
    request = {
        "expected_story_version": 0,
        "expected_publication_version": 0,
        "expected_delivery_evidence_version": 0,
        "applied_at": "2026-07-16T11:00:00Z",
        "observed_at": "2026-07-16T11:30:00Z",
        "proof": proof(),
    }
    with pytest.raises(EditorialHold):
        controller.advance(
            retained.package_admission_id,
            _decision(retained, source.admission_id, result="HOLD"),
            **request,
        )
    target = sqlite3.connect(bindings.target_path)
    count = target.execute(
        "SELECT COUNT(*) FROM private_serving_payloads"
    ).fetchone()[0]
    assert count == 0
    target.close()

    decision = _decision(retained, source.admission_id)
    first = controller.advance(retained.package_admission_id, decision, **request)
    replay = controller.advance(retained.package_admission_id, decision, **request)
    assert replay == first

    reader = open_private_serving_read_port(
        bindings.target_path,
        target_id=bindings.target_id,
        target_context_digest=bindings.target_context_digest,
        proof=first.read_proof,
    )
    acknowledged = reader.acknowledged_rows()
    assert acknowledged is not None
    assert tuple(row.surface_kind for row in acknowledged.rows) == (
        "ARTICLE",
        "FEED_CARD",
    )
    assert reader._connection.total_changes == 0
    reader.close()

    controller.close()
    reopened = NativePublicationController(
        objects=system.objects,
        commands=system.commands,
        events=system.events,
        candidate_port=candidate_port,
        evidence_packages=evidence_packages,
        bindings=bindings,
    )
    assert reopened.advance(retained.package_admission_id, decision, **request) == first
    reopened.close()
    candidate_connection.rollback()
    system.close()
    ingress.close()
    candidate_connection.close()
