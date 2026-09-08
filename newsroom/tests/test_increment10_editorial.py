from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from newsroom.authority import (
    AggregateId,
    CommandDefinition,
    CommandRegistry,
    HydrationPolicyContract,
    HydrationPolicyRegistry,
    MetadataClass,
    ObjectAdmissionDefinition,
    ObjectAdmissionDescriptor,
    ObjectAdmissionId,
    ObjectAdmissionPayload,
    ObjectAdmissionRegistry,
    ObjectAdmissionRequest,
    ObjectLimits,
    PayloadGoldenVector,
    PayloadMode,
    PayloadSchemaContract,
    PayloadSchemaRegistry,
    SemanticCommand,
    StaticAuthenticator,
    StaticAuthorizer,
    StaticPrincipal,
    TrustScope,
)
from newsroom.authority.canonical import canonical_json_bytes, digest_bytes
from newsroom.authority.policy import PayloadSchemaValidationError
from newsroom.control_plane.evidence import (
    EVID_012_POLICY_VERSION,
    Evid012QualificationTest,
    GovernedClaimEvidence,
    QualificationEvidence,
)
from newsroom.increment10.editorial import (
    DECISION_ADMISSION_TYPE,
    DECISION_CLASS,
    DECISION_COMMAND,
    DECISION_EVENT,
    DECISION_PURPOSE,
    DECISION_USE,
    EDITORIAL_RETENTION_SCOPE,
    EDITORIAL_SECURITY_SCOPE,
    STORY_ADMISSION_TYPE,
    STORY_CLASS,
    STORY_COMMAND,
    STORY_EVENT,
    STORY_PURPOSE,
    STORY_USE,
    DecisionReference,
    EditorialError,
    EditorialHold,
    EditorialPolicyDecision,
    NativeEditorial,
    SourceCurrentness,
    SourceIntegrity,
    StoryVersionRequest,
)
from newsroom.increment10.evidence import GovernedEvidencePackages
from newsroom.increment10.ingress import open_evidence_intake_ingress
from newsroom.tests.authority_a2b_helpers import open_object_system
from newsroom.tests.authority_event_helpers import fixture_read_policy
from newsroom.tests.authority_helpers import FIXED_NOW, proof
from newsroom.tests.test_increment10_evidence import (
    _package_and_records,
    _policies,
)
from newsroom.tests.test_increment10_ingress import _candidate, _receive


def _object_reference_bytes(value: object) -> bytes:
    if not isinstance(value, ObjectAdmissionDescriptor):
        raise PayloadSchemaValidationError("object descriptor required")
    return canonical_json_bytes(
        {
            "admission_id": str(value.admission_id),
            "blob_digest": value.blob_digest,
            "object_class": value.object_class,
            "allowed_use": value.allowed_use,
            "security_scope": value.security_scope,
            "retention_scope": value.retention_scope,
        }
    )


def _command_contract() -> PayloadSchemaContract:
    vector = ObjectAdmissionDescriptor(
        ObjectAdmissionId.parse("00000000-0000-4000-8000-000000000001"),
        "sha256:" + "a" * 64,
        DECISION_CLASS,
        DECISION_USE,
        EDITORIAL_SECURITY_SCOPE,
        EDITORIAL_RETENTION_SCOPE,
        True,
    )
    return PayloadSchemaContract(
        schema_version="editorial_object_reference_v1",
        payload_mode=PayloadMode.OBJECT_ADMISSION,
        contract_version="editorial-object-reference-v1",
        canonicalizer_implementation_version="editorial-object-canonicalizer-v1",
        canonicalizer=_object_reference_bytes,
        golden_vectors=(
            PayloadGoldenVector(
                "editorial-reference",
                "editorial-reference-v1",
                vector,
                _object_reference_bytes(vector),
            ),
        ),
    )


def _command_definition(
    contract: PayloadSchemaContract,
    *,
    command: str,
    event: str,
    aggregate: str,
    object_class: str,
    allowed_use: str,
    required_scope: str,
) -> CommandDefinition:
    return CommandDefinition(
        command_type=command,
        definition_version="editorial-command-v1",
        aggregate_type=aggregate,
        event_type=event,
        event_schema_version=1,
        payload_mode=PayloadMode.OBJECT_ADMISSION,
        payload_schema_version=contract.schema_version,
        payload_schema_contract_version=contract.contract_version,
        payload_schema_contract_digest=contract.contract_digest,
        payload_canonicalizer_version=contract.canonicalizer_implementation_version,
        trust_scope=TrustScope.ADMITTED,
        security_scope=EDITORIAL_SECURITY_SCOPE,
        retention_scope=EDITORIAL_RETENTION_SCOPE,
        required_scope=required_scope,
        required_object_class=object_class,
        required_allowed_use=allowed_use,
    )


def _registries():
    rights, hydration, admissions, evidence_hydration, evidence_definitions = (
        _policies()
    )
    editorial_hydration = tuple(
        HydrationPolicyContract(
            policy_id=f"{purpose}-read-v1",
            contract_version="hydration-v1",
            implementation_version="hydration-static-v1",
            purpose=purpose,
            required_scope="authority.objects.read",
            allowed_principal_ids=frozenset({"principal.alpha"}),
            allowed_authority_domains=frozenset({"newsroom.authority"}),
            allowed_object_classes=frozenset({object_class}),
            allowed_uses=frozenset({allowed_use}),
            allowed_security_scopes=frozenset({EDITORIAL_SECURITY_SCOPE}),
            allowed_retention_scopes=frozenset({EDITORIAL_RETENTION_SCOPE}),
            max_bytes=1024 * 1024,
        )
        for purpose, object_class, allowed_use in (
            (DECISION_PURPOSE, DECISION_CLASS, DECISION_USE),
            (STORY_PURPOSE, STORY_CLASS, STORY_USE),
        )
    )
    all_hydration = HydrationPolicyRegistry(
        (*hydration.contracts(), *editorial_hydration)
    )
    editorial_definitions = tuple(
        ObjectAdmissionDefinition(
            admission_type=admission_type,
            definition_version="admission-v1",
            object_class=object_class,
            allowed_use=allowed_use,
            security_scope=EDITORIAL_SECURITY_SCOPE,
            retention_scope=EDITORIAL_RETENTION_SCOPE,
            required_write_scope=write_scope,
            required_read_scope="authority.objects.read",
            required_manage_scope="authority.objects.manage",
            rights_policy_contract_digest=rights.contracts()[0].contract_digest,
            hydration_policy_contract_digests=frozenset(
                {hydration_contract.contract_digest}
            ),
        )
        for (
            admission_type,
            object_class,
            allowed_use,
            write_scope,
        ), hydration_contract in zip(
            (
                (
                    DECISION_ADMISSION_TYPE,
                    DECISION_CLASS,
                    DECISION_USE,
                    "authority.editorial.decide",
                ),
                (
                    STORY_ADMISSION_TYPE,
                    STORY_CLASS,
                    STORY_USE,
                    "authority.editorial.story.write",
                ),
            ),
            editorial_hydration,
            strict=True,
        )
    )
    all_admissions = ObjectAdmissionRegistry(
        (*admissions.definitions(), *editorial_definitions),
        rights_policies=rights,
        hydration_policies=all_hydration,
    )
    contract = _command_contract()
    decision_command = _command_definition(
        contract,
        command=DECISION_COMMAND,
        event=DECISION_EVENT,
        aggregate="editorial_package_decision",
        object_class=DECISION_CLASS,
        allowed_use=DECISION_USE,
        required_scope="authority.editorial.decide",
    )
    story_command = _command_definition(
        contract,
        command=STORY_COMMAND,
        event=STORY_EVENT,
        aggregate="story",
        object_class=STORY_CLASS,
        allowed_use=STORY_USE,
        required_scope="authority.editorial.story.admit",
    )
    return (
        (rights, all_hydration, all_admissions),
        evidence_hydration,
        evidence_definitions,
        editorial_hydration,
        editorial_definitions,
        decision_command,
        story_command,
        CommandRegistry((decision_command, story_command)),
        PayloadSchemaRegistry((contract,)),
    )


def _open_editorial_system(path: Path):
    registries = _registries()
    read_policy = fixture_read_policy(
        allowed_security_scopes=frozenset(
            {EDITORIAL_SECURITY_SCOPE, "authority.object_lifecycle"}
        ),
        allowed_trust_scopes=frozenset({TrustScope.ADMITTED}),
        metadata_classes=frozenset(
            {MetadataClass.ROUTING, MetadataClass.PROVENANCE, MetadataClass.RESULT}
        ),
        max_results=1000,
    )
    scopes = frozenset(
        {
            "authority.evidence.admit",
            "authority.editorial.decide",
            "authority.editorial.story.write",
            "authority.editorial.story.admit",
            "authority.objects.read",
            "authority.objects.manage",
            "authority.objects.lifecycle.write",
            read_policy.required_scope,
        }
    )
    system = open_object_system(
        path,
        policy_registries=registries[0],
        authenticator=StaticAuthenticator(
            credentials={"token-1": StaticPrincipal("principal.alpha")},
            authority_domain="newsroom.authority",
        ),
        authorizer=StaticAuthorizer(
            policy_version="authz-v1",
            grants_by_principal={"principal.alpha": scopes},
        ),
        command_registry=registries[-2],
        payload_schema_registry=registries[-1],
        object_limits=ObjectLimits(
            global_max_bytes=1024 * 1024,
            class_max_bytes={
                "evidence_source": 1024 * 1024,
                "evidence_record": 1024 * 1024,
                "evidence_package": 1024 * 1024,
                DECISION_CLASS: 1024 * 1024,
                STORY_CLASS: 1024 * 1024,
            },
            max_read_bytes=1024 * 1024,
            min_free_bytes=0,
            io_chunk_bytes=64,
            max_staging_bytes=1024 * 1024,
            max_range_bytes=1024 * 1024,
        ),
    )
    return system, registries


def _ready_package(version):
    passage = "The deadline changed.\nThe official deadline changed."
    package, records = _package_and_records(version, passage)
    headline = package.governed_claims[0]
    substantive = replace(
        headline,
        claim_id="claim-2",
        claim="The official deadline changed.",
        supporting_excerpt="The official deadline changed.",
        source_authority_decision_ids=("authority-2",),
        rendered_assertion_zh_hant_hk="官方限期安排已經更新。",
        claim_role="SUBSTANTIVE",
        semantic_relation_evidence_id="semantic-2",
    )
    evidence = (
        ("action_class", "OFFICIAL_DEADLINE"),
        ("event_polarity", "AFFIRMED"),
        ("action_relation", "NEW_OR_CHANGED_OFFICIAL_ACTION"),
        ("material_relation_span", headline.claim),
        ("reader_action", headline.claim),
    )
    qualification = QualificationEvidence(
        Evid012QualificationTest.OFFICIAL_ACTION_OR_DEADLINE,
        headline.claim_id,
        "qualification-1",
        evidence,
    )
    package = replace(
        package,
        substantive_new_information=(headline.claim, substantive.claim),
        governed_claims=(headline, substantive),
        qualification_evidence=(qualification,),
        selection_rationale="A verified official deadline has changed.",
        geography=("UK",),
        categories=("Politics and law",),
        evidence_gate_results=tuple((gate, "HOLD") for gate in (
            "CLAIM_TRACEABILITY", "EVIDENCE_SUFFICIENCY", "SOURCE_AUTHORITY"
        )),
        freshness_result="HOLD",
        integrity_result="HOLD",
    )
    common = {
        "candidate_id": version.candidate_id,
        "base_package_digest": records[0]["base_package_digest"],
        "status": "CURRENT",
    }
    added = (
        {
            **common,
            "record_id": "authority-2",
            "record_type": "SOURCE_AUTHORITY_DECISION",
            "source_id": "source-1",
            "decision": "ADMITTED",
            "authority_class": "RESPONSIBLE_PRIMARY",
            "authority_scope": "Own deadline",
            "governed_claim_id": substantive.claim_id,
            "claim_digest": digest_bytes(substantive.claim.encode()),
        },
        {
            **common,
            "record_id": "semantic-2",
            "record_type": "SEMANTIC_RELATION_EVIDENCE",
            "governed_claim_id": substantive.claim_id,
            "source_modality": "ASSERTED",
            "rendered_modality": "ASSERTED",
            "source_polarity": "AFFIRMED",
            "rendered_polarity": "AFFIRMED",
            "relation": "SEMANTICALLY_EQUIVALENT",
            "claim_digest": digest_bytes(substantive.claim.encode()),
            "rendered_assertion_digest": digest_bytes(
                substantive.rendered_assertion_zh_hant_hk.encode()
            ),
        },
        {
            **common,
            "record_id": qualification.qualification_record_id,
            "record_type": "QUALIFICATION_EVIDENCE",
            "governed_claim_id": headline.claim_id,
            "test": qualification.test.value,
            "test_evidence": [list(item) for item in evidence],
            "policy_version": EVID_012_POLICY_VERSION,
            "evidence_span_digest": digest_bytes(headline.supporting_excerpt.encode()),
            "source_record_ids": list(headline.source_record_ids),
        },
    )
    return passage, package, (*records, *added)


def _evidence_facade(system, ingress, registries):
    evidence_hydration, evidence_definitions = registries[1:3]
    return GovernedEvidencePackages(
        objects=system.objects,
        ingress=ingress,
        reader_principal_id="principal.alpha",
        reader_authority_domain="newsroom.authority",
        source_hydration_policy_digest=evidence_hydration[0].contract_digest,
        record_hydration_policy_digest=evidence_hydration[1].contract_digest,
        package_hydration_policy_digest=evidence_hydration[2].contract_digest,
        package_admission_definition_digest=evidence_definitions[2].digest,
    )


def _native(system, evidence, registries):
    editorial_hydration = registries[3]
    editorial_definitions = registries[4]
    return NativeEditorial(
        objects=system.objects,
        commands=system.commands,
        events=system.events,
        evidence=evidence,
        reader_principal_id="principal.alpha",
        reader_authority_domain="newsroom.authority",
        controller_principal_id="principal.alpha",
        story_principal_id="principal.alpha",
        policy_bundle_digest="sha256:" + "a" * 64,
        decision_hydration_policy_digest=editorial_hydration[0].contract_digest,
        story_hydration_policy_digest=editorial_hydration[1].contract_digest,
        decision_command_definition_digest=registries[5].digest,
        story_command_definition_digest=registries[6].digest,
        story_admission_definition_digest=editorial_definitions[1].digest,
    )


def _decision(
    retained,
    source_admission_id,
    *,
    result="PASS",
    evaluated_at="2026-09-08T12:02:00Z",
):
    return EditorialPolicyDecision.create(
        candidate_version_id=retained.candidate_version_id,
        candidate_version_digest=retained.candidate_version_digest,
        governing_manifest_digest=retained.governing_manifest_digest,
        package_admission_id=retained.package_admission_id,
        package_digest=retained.package.digest,
        policy_bundle_digest="sha256:" + "a" * 64,
        evaluated_at=evaluated_at,
        currentness=(
            SourceCurrentness(
                "source-1",
                "source-definition-1",
                "sha256:" + "b" * 64,
                "CURRENT_VERSION",
                "2026-09-08T12:00:00Z",
                "2026-09-08T12:01:00Z",
                None,
                "instrument-version=2026-09-08",
                "sha256:" + "d" * 64,
                "sha256:" + "c" * 64,
                result,
                "CURRENT_VERSION_CONFIRMED" if result == "PASS" else "POLICY_MISSING",
            ),
        ),
        integrity=(
            SourceIntegrity(
                "source-1",
                source_admission_id,
                retained.package.observation_digests[0],
                (
                    "ACCESS_COMPLETE",
                    "ENCODING_VALID",
                    "EXTRACTION_COMPLETE",
                    "NOT_PAYWALL_FRAGMENT",
                    "NOT_TRUNCATED",
                    "VERSION_UNAMBIGUOUS",
                ),
                result,
                "INTEGRITY_CONFIRMED" if result == "PASS" else "POLICY_MISSING",
            ),
        ),
        evidence_gate_results=tuple((gate, result) for gate in (
            "CLAIM_TRACEABILITY", "EVIDENCE_SUFFICIENCY", "SOURCE_AUTHORITY"
        )),
    )


def _record_decision(system, decision):
    admission = system.objects.admit(
        ObjectAdmissionRequest(DECISION_ADMISSION_TYPE, decision.decision_id),
        decision.canonical_bytes(),
        proof=proof(),
    ).admission
    committed = system.commands.execute(
        SemanticCommand(
            DECISION_COMMAND,
            AggregateId.new(),
            0,
            ObjectAdmissionPayload(admission.admission_id),
            decision.decision_id,
        ),
        proof=proof(),
    )
    return DecisionReference(committed.event_id, admission.admission_id)


def test_observed_state_currentness_enforces_window_boundary_and_missing_rule() -> None:
    admission_id = ObjectAdmissionId.parse(
        "00000000-0000-4000-8000-000000000001"
    )
    integrity = SourceIntegrity(
        "source-1",
        admission_id,
        "sha256:" + "1" * 64,
        (
            "ACCESS_COMPLETE",
            "ENCODING_VALID",
            "EXTRACTION_COMPLETE",
            "NOT_PAYWALL_FRAGMENT",
            "NOT_TRUNCATED",
            "VERSION_UNAMBIGUOUS",
        ),
        "PASS",
        "INTEGRITY_CONFIRMED",
    )

    def currentness(window, result="PASS"):
        return SourceCurrentness(
            "source-1",
            "source-definition-1",
            "sha256:" + "2" * 64,
            "OBSERVED_STATE",
            "2026-09-08T12:00:00Z",
            "2026-09-08T12:00:30Z",
            window,
            None,
            None,
            "sha256:" + "3" * 64,
            result,
            "CURRENT" if result == "PASS" else "CURRENCY_RULE_MISSING",
        )

    def decision(item, evaluated_at):
        return EditorialPolicyDecision.create(
            candidate_version_id="candidate-version-1",
            candidate_version_digest="sha256:" + "4" * 64,
            governing_manifest_digest="sha256:" + "5" * 64,
            package_admission_id=admission_id,
            package_digest="sha256:" + "6" * 64,
            policy_bundle_digest="sha256:" + "7" * 64,
            evaluated_at=evaluated_at,
            currentness=(item,),
            integrity=(integrity,),
            evidence_gate_results=tuple((gate, "PASS") for gate in (
                "CLAIM_TRACEABILITY", "EVIDENCE_SUFFICIENCY", "SOURCE_AUTHORITY"
            )),
        )

    assert decision(currentness(60), "2026-09-08T12:01:00Z")
    with pytest.raises(EditorialError, match="stale"):
        decision(currentness(60), "2026-09-08T12:01:00.000001Z")
    with pytest.raises(EditorialError, match="observed-state"):
        currentness(None)
    assert currentness(None, "HOLD").result == "HOLD"


def test_authenticated_policy_admits_and_reopens_native_story_version(
    tmp_path: Path,
) -> None:
    candidate_connection, candidate_port, version = _candidate(tmp_path)
    ingress_path = tmp_path / "intake.sqlite3"
    ingress = open_evidence_intake_ingress(ingress_path)
    acknowledgement = _receive(
        ingress, candidate_connection, candidate_port, version, request_id="request-1"
    )
    object_path = tmp_path / "objects.sqlite3"
    system, registries = _open_editorial_system(object_path)
    evidence = _evidence_facade(system, ingress, registries)
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
    try:
        retained = evidence.retain(
            package,
            receipt_id=acknowledgement.receipt_id,
            candidate_port=candidate_port,
            source_admission_ids=(source.admission_id,),
            record_admission_ids=record_ids,
            proof=proof(),
        )
        decision_reference = _record_decision(
            system, _decision(retained, source.admission_id)
        )
        native = _native(system, evidence, registries)
        request = StoryVersionRequest(AggregateId.new(), 0, "story-1")

        class ChangingRequest:
            expected_aggregate_version = 0
            idempotency_key = "changing-request"

            def __init__(self):
                self.first = AggregateId.new()
                self.later = AggregateId.new()
                self.calls = 0

            @property
            def story_id(self):
                self.calls += 1
                return self.first if self.calls == 1 else self.later

        changing = ChangingRequest()
        with pytest.raises(EditorialError, match="immutable Story Version request"):
            native.admit_story_version(
                changing,
                package_admission_id=retained.package_admission_id,
                decision_reference=decision_reference,
                candidate_port=candidate_port,
                proof=proof(),
            )
        assert changing.calls == 0
        with pytest.raises(EditorialError, match="differ from Candidate"):
            native.admit_story_version(
                StoryVersionRequest(AggregateId.parse(version.candidate_id), 0, "alias"),
                package_admission_id=retained.package_admission_id,
                decision_reference=decision_reference,
                candidate_port=candidate_port,
                proof=proof(),
            )
        receipt, story = native.admit_story_version(
            request,
            package_admission_id=retained.package_admission_id,
            decision_reference=decision_reference,
            candidate_port=candidate_port,
            proof=proof(),
        )
        replay_receipt, replay_story = native.admit_story_version(
            request,
            package_admission_id=retained.package_admission_id,
            decision_reference=decision_reference,
            candidate_port=candidate_port,
            proof=proof(),
        )
        changed_reference = _record_decision(
            system,
            _decision(
                retained,
                source.admission_id,
                evaluated_at="2026-09-08T12:03:00Z",
            ),
        )
        story_events_before = tuple(
            event
            for event in system.events.after(0, limit=1000, proof=proof())
            if event.event_type == STORY_EVENT
        )
        with pytest.raises(EditorialError, match="object admission differs"):
            native.admit_story_version(
                request,
                package_admission_id=retained.package_admission_id,
                decision_reference=changed_reference,
                candidate_port=candidate_port,
                proof=proof(),
            )
        story_events_after = tuple(
            event
            for event in system.events.after(0, limit=1000, proof=proof())
            if event.event_type == STORY_EVENT
        )
        with pytest.raises(EditorialError, match="event binding"):
            native.read_story_version(
                replace(receipt, story_version_digest="sha256:" + "0" * 64),
                candidate_port=candidate_port,
                proof=proof(),
            )
    finally:
        candidate_connection.rollback()
    assert story.write_admission.decision == "WRITE_READY"
    assert all(item.result == "PASS" for item in story.validators)
    assert story.copy.title.startswith("【未出版】")
    assert story.story_id != version.candidate_id
    assert replay_receipt == receipt
    assert replay_story == story
    assert story_events_after == story_events_before
    system.close()
    ingress.close()

    reopened_system, reopened_registries = _open_editorial_system(object_path)
    reopened_ingress = open_evidence_intake_ingress(ingress_path)
    reopened = _native(
        reopened_system,
        _evidence_facade(reopened_system, reopened_ingress, reopened_registries),
        reopened_registries,
    )
    candidate_connection.execute("BEGIN IMMEDIATE")
    try:
        assert reopened.read_story_version(
            receipt, candidate_port=candidate_port, proof=proof()
        ) == story
    finally:
        candidate_connection.rollback()
    reopened_system.close()
    reopened_ingress.close()
    candidate_connection.close()


def test_non_pass_controller_decision_holds_without_story_event(tmp_path: Path) -> None:
    candidate_connection, candidate_port, version = _candidate(tmp_path)
    ingress = open_evidence_intake_ingress(tmp_path / "intake.sqlite3")
    acknowledgement = _receive(
        ingress, candidate_connection, candidate_port, version, request_id="request-1"
    )
    system, registries = _open_editorial_system(tmp_path / "objects.sqlite3")
    evidence = _evidence_facade(system, ingress, registries)
    passage, package, records = _ready_package(version)
    source = system.objects.admit(
        ObjectAdmissionRequest("evidence.source", "source-1"), passage.encode(), proof=proof()
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
    try:
        retained = evidence.retain(
            package,
            receipt_id=acknowledgement.receipt_id,
            candidate_port=candidate_port,
            source_admission_ids=(source.admission_id,),
            record_admission_ids=record_ids,
            proof=proof(),
        )
        reference = _record_decision(
            system, _decision(retained, source.admission_id, result="HOLD")
        )
        before = system.events.after(0, limit=1000, proof=proof())
        with pytest.raises(EditorialHold) as held:
            _native(system, evidence, registries).admit_story_version(
                StoryVersionRequest(AggregateId.new(), 0, "held-story"),
                package_admission_id=retained.package_admission_id,
                decision_reference=reference,
                candidate_port=candidate_port,
                proof=proof(),
            )
        with pytest.raises(EditorialHold, match="DECISION_MISSING") as missing:
            _native(system, evidence, registries).admit_story_version(
                StoryVersionRequest(AggregateId.new(), 0, "missing-policy"),
                package_admission_id=retained.package_admission_id,
                decision_reference=DecisionReference(
                    str(AggregateId.new()), ObjectAdmissionId.new()
                ),
                candidate_port=candidate_port,
                proof=proof(),
            )
        after = system.events.after(0, limit=1000, proof=proof())
    finally:
        candidate_connection.rollback()
    assert held.value.decision.decision == "HOLD"
    assert missing.value.decision is None
    assert after == before
    system.close()
    ingress.close()
    candidate_connection.close()
