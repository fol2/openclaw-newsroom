from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from newsroom.authority import (
    AuthenticationProof,
    HydrationRequest,
    HydrationPolicyContract,
    HydrationPolicyRegistry,
    ObjectAdmissionDefinition,
    ObjectAdmissionRegistry,
    ObjectAdmissionRequest,
    ObjectHydrationDenied,
    ObjectLimits,
    RightsPolicyContract,
    RightsPolicyRegistry,
    StaticAuthenticator,
    StaticAuthorizer,
    StaticPrincipal,
)
from newsroom.authority.canonical import canonical_json_bytes, digest_bytes
from newsroom.control_plane.evidence import (
    ClaimAuthorityClass,
    EvidencePackage,
    GovernedClaimEvidence,
    GovernedClaimStatus,
    validate_governed_evidence_records,
)
from newsroom.increment10.evidence import EvidencePackageError, GovernedEvidencePackages
from newsroom.increment10.ingress import open_evidence_intake_ingress
from newsroom.tests.authority_a2b_helpers import open_object_system
from newsroom.tests.authority_helpers import proof
from newsroom.tests.test_increment10_ingress import _candidate, _receive


def _policies():
    rights_contract = RightsPolicyContract(
        "evidence-permitted", "rights-v1", "rights-static-v1", True, "PERMITTED"
    )
    rights = RightsPolicyRegistry((rights_contract,))
    specifications = (
        (
            "evidence.source",
            "evidence_source",
            "evidence.source",
            "publication_evidence",
        ),
        (
            "evidence.record",
            "evidence_record",
            "evidence.record",
            "publication_evidence",
        ),
        (
            "evidence.package",
            "evidence_package",
            "evidence.package.retained",
            "evidence_package_retention",
        ),
    )
    hydration_contracts = tuple(
        HydrationPolicyContract(
            policy_id=f"{admission_type}-read-v1",
            contract_version="hydration-v1",
            implementation_version="hydration-static-v1",
            purpose=purpose,
            required_scope="authority.objects.read",
            allowed_principal_ids=frozenset({"principal.alpha", "principal.reader"}),
            allowed_authority_domains=frozenset({"newsroom.authority"}),
            allowed_object_classes=frozenset({object_class}),
            allowed_uses=frozenset({allowed_use}),
            allowed_security_scopes=frozenset({"authority.protected"}),
            allowed_retention_scopes=frozenset({"evidence.retained"}),
            max_bytes=1024 * 1024,
        )
        for admission_type, object_class, purpose, allowed_use in specifications
    )
    hydration = HydrationPolicyRegistry(hydration_contracts)
    definitions = tuple(
        ObjectAdmissionDefinition(
            admission_type=admission_type,
            definition_version="admission-v1",
            object_class=object_class,
            allowed_use=allowed_use,
            security_scope="authority.protected",
            retention_scope="evidence.retained",
            required_write_scope="authority.evidence.admit",
            required_read_scope="authority.objects.read",
            required_manage_scope="authority.objects.manage",
            rights_policy_contract_digest=rights_contract.contract_digest,
            hydration_policy_contract_digests=frozenset(
                {hydration_contract.contract_digest}
            ),
        )
        for (admission_type, object_class, _, allowed_use), hydration_contract in zip(
            specifications, hydration_contracts, strict=True
        )
    )
    admissions = ObjectAdmissionRegistry(
        definitions, rights_policies=rights, hydration_policies=hydration
    )
    return rights, hydration, admissions, hydration_contracts, definitions


def _open_objects(path: Path, *, unprivileged: bool = False):
    policies = _policies()
    authenticator = StaticAuthenticator(
        credentials={
            "token-1": StaticPrincipal("principal.alpha"),
            "reader-token": StaticPrincipal("principal.reader"),
        },
        authority_domain="newsroom.authority",
    )
    grants = {
        "principal.alpha": frozenset(
            {
                "authority.observed.write",
                "authority.admitted.write",
                "authority.objects.read",
                "authority.objects.manage",
                "authority.objects.lifecycle.write",
                "authority.evidence.admit",
                "authority.events.read",
            }
        ),
        "principal.reader": frozenset({"authority.objects.read"}),
    }
    system = open_object_system(
        path,
        policy_registries=policies[:3],
        authenticator=authenticator,
        authorizer=StaticAuthorizer(
            policy_version="authz-v1", grants_by_principal=grants
        ),
        object_limits=ObjectLimits(
            global_max_bytes=1024 * 1024,
            class_max_bytes={
                "evidence_source": 1024 * 1024,
                "evidence_record": 1024 * 1024,
                "evidence_package": 1024 * 1024,
            },
            max_read_bytes=1024 * 1024,
            min_free_bytes=0,
            io_chunk_bytes=64,
            max_staging_bytes=1024 * 1024,
            max_range_bytes=1024 * 1024,
        ),
    )
    return system, policies


def _admit(system, admission_type: str, data: bytes, key: str, *, reader=False):
    selected_proof = (
        AuthenticationProof("reader-token", "proof-reader") if reader else proof()
    )
    return system.objects.admit(
        ObjectAdmissionRequest(admission_type, key), data, proof=selected_proof
    ).admission.admission_id


def _package_and_records(version, passage: str):
    bindings = version.governing_manifest.lead_signal_bindings
    base = EvidencePackage(
        candidate_id=version.candidate_id,
        hypothesis_id=version.governing_manifest.hypothesis_id,
        signal_ids=tuple(item.signal_id for item in bindings),
        lead_ids=tuple(item.lead_id for item in bindings),
        source_ids=("source-1",),
        observation_digests=(digest_bytes(passage.encode()),),
        passages=(passage,),
    )
    claim = GovernedClaimEvidence(
        claim_id="claim-1",
        claim="The deadline changed.",
        passage_index=0,
        supporting_excerpt="The deadline changed.",
        source_ids=("source-1",),
        source_record_ids=("source-record-1",),
        source_authority_decision_ids=("authority-1",),
        rights_decision_ids=("rights-1",),
        dependency_evidence_ids=("dependency-1",),
        evidential_origin_ids=("origin-1",),
        authority_class=ClaimAuthorityClass.RESPONSIBLE_PRIMARY,
        authority_scope="Own deadline",
        status=GovernedClaimStatus.CONFIRMED_FACT,
        attribution="Official source",
        rendered_assertion_zh_hant_hk="官方確認限期已經更改。",
        claim_role="HEADLINE",
        semantic_relation_evidence_id="semantic-1",
    )
    package = replace(base, governed_claims=(claim,))
    common = {
        "candidate_id": version.candidate_id,
        "base_package_digest": base.digest,
        "status": "CURRENT",
    }
    records = (
        {
            **common,
            "record_id": "source-record-1",
            "record_type": "SOURCE_RECORD",
            "source_id": "source-1",
            "canonical_url": "https://example.test/current",
            "publisher": "Example Authority",
            "responsible_body": "Example Authority",
            "source_type": "PRIMARY_OFFICIAL",
            "authority_class": "RESPONSIBLE_PRIMARY",
            "publication_time": "2026-09-08T12:00:00+00:00",
            "retrieval_time": "2026-09-08T12:01:00+00:00",
            "geography": "UK",
            "language": "en-GB",
            "extraction_status": "COMPLETE",
            "rights_decision_id": "rights-1",
            "originating_report_id": "origin-1",
            "originating_artefact_digest": digest_bytes(passage.encode()),
            "dependency_evidence_ids": ["dependency-1"],
        },
        {
            **common,
            "record_id": "authority-1",
            "record_type": "SOURCE_AUTHORITY_DECISION",
            "source_id": "source-1",
            "decision": "ADMITTED",
            "authority_class": "RESPONSIBLE_PRIMARY",
            "authority_scope": "Own deadline",
            "governed_claim_id": "claim-1",
            "claim_digest": digest_bytes(claim.claim.encode()),
        },
        {
            **common,
            "record_id": "rights-1",
            "record_type": "RIGHTS_DECISION",
            "source_id": "source-1",
            "decision": "PERMITTED",
            "permitted_use": "PUBLICATION_EVIDENCE",
        },
        {
            **common,
            "record_id": "dependency-1",
            "record_type": "DEPENDENCY_EVIDENCE",
            "source_id": "source-1",
            "dependency_status": "RESOLVED",
            "evidential_origin_id": "origin-1",
            "originating_report_id": "origin-1",
        },
        {
            **common,
            "record_id": "semantic-1",
            "record_type": "SEMANTIC_RELATION_EVIDENCE",
            "governed_claim_id": "claim-1",
            "source_modality": "ASSERTED",
            "rendered_modality": "ASSERTED",
            "source_polarity": "AFFIRMED",
            "rendered_polarity": "AFFIRMED",
            "relation": "SEMANTICALLY_EQUIVALENT",
            "claim_digest": digest_bytes(claim.claim.encode()),
            "rendered_assertion_digest": digest_bytes(
                claim.rendered_assertion_zh_hant_hk.encode()
            ),
        },
    )
    return package, records


def _facade(system, ingress, policies):
    _, _, _, hydration, definitions = policies
    return GovernedEvidencePackages(
        objects=system.objects,
        ingress=ingress,
        reader_principal_id="principal.alpha",
        reader_authority_domain="newsroom.authority",
        source_hydration_policy_digest=hydration[0].contract_digest,
        record_hydration_policy_digest=hydration[1].contract_digest,
        package_hydration_policy_digest=hydration[2].contract_digest,
        package_admission_definition_digest=definitions[2].digest,
    )


def test_native_candidate_receipt_to_governed_package_reopens(tmp_path: Path) -> None:
    candidate_connection, candidate_port, version = _candidate(tmp_path)
    ingress_path = tmp_path / "intake.sqlite3"
    ingress = open_evidence_intake_ingress(ingress_path)
    acknowledgement = _receive(
        ingress, candidate_connection, candidate_port, version, request_id="request-1"
    )
    object_path = tmp_path / "objects.sqlite3"
    objects, policies = _open_objects(object_path)
    passage = "The deadline changed."
    package, records = _package_and_records(version, passage)
    source_id = _admit(objects, "evidence.source", passage.encode(), "source-1")
    record_ids = tuple(
        _admit(
            objects,
            "evidence.record",
            canonical_json_bytes(record),
            f"record-{index}",
        )
        for index, record in enumerate(records)
    )
    opaque_package_id = _admit(
        objects, "evidence.package", b"{}", "opaque-package"
    )
    extra_record_id = _admit(
        objects,
        "evidence.record",
        canonical_json_bytes({"record_id": "extra", "record_type": "RIGHTS_DECISION"}),
        "extra-record",
    )
    facade = _facade(objects, ingress, policies)

    candidate_connection.execute("BEGIN IMMEDIATE")
    try:
        with pytest.raises(EvidencePackageError, match="schema"):
            facade.read(
                opaque_package_id, candidate_port=candidate_port, proof=proof()
            )
        with pytest.raises(EvidencePackageError, match="Candidate binding"):
            facade.retain(
                replace(package, candidate_id="another-candidate"),
                receipt_id=acknowledgement.receipt_id,
                candidate_port=candidate_port,
                source_admission_ids=(source_id,),
                record_admission_ids=record_ids,
                proof=proof(),
            )
        with pytest.raises(EvidencePackageError, match="governed bytes"):
            facade.retain(
                replace(package, passages=("Changed after acquisition.",)),
                receipt_id=acknowledgement.receipt_id,
                candidate_port=candidate_port,
                source_admission_ids=(source_id,),
                record_admission_ids=record_ids,
                proof=proof(),
            )
        with pytest.raises(EvidencePackageError, match="governed evidence records"):
            facade.retain(
                package,
                receipt_id=acknowledgement.receipt_id,
                candidate_port=candidate_port,
                source_admission_ids=(source_id,),
                record_admission_ids=record_ids + (extra_record_id,),
                proof=proof(),
            )
        mutable_source_ids = list(package.source_ids)
        mutable_package = replace(
            package, source_ids=mutable_source_ids, passages=list(package.passages)
        )
        retained = facade.retain(
            mutable_package,
            receipt_id=acknowledgement.receipt_id,
            candidate_port=candidate_port,
            source_admission_ids=(source_id,),
            record_admission_ids=record_ids,
            proof=proof(),
        )
        mutable_source_ids.append("mutated-after-retain")
        assert retained.package.source_ids == package.source_ids
        assert retained.package.passages == package.passages
        replayed = facade.retain(
            package,
            receipt_id=acknowledgement.receipt_id,
            candidate_port=candidate_port,
            source_admission_ids=(source_id,),
            record_admission_ids=record_ids,
            proof=proof(),
        )
    finally:
        candidate_connection.rollback()
    assert retained.package.resolved_evidence_records
    assert replayed == retained
    assert retained.drafting_authority is False
    assert retained.publication_authority is False
    assert retained.package.freshness_result == "MISSING"
    package_id = retained.package_admission_id
    objects.close()
    ingress.close()

    reopened_objects, reopened_policies = _open_objects(object_path)
    reopened_ingress = open_evidence_intake_ingress(ingress_path)
    reopened = _facade(reopened_objects, reopened_ingress, reopened_policies)
    candidate_connection.execute("BEGIN IMMEDIATE")
    try:
        loaded = reopened.read(package_id, candidate_port=candidate_port, proof=proof())
    finally:
        candidate_connection.rollback()
    assert loaded == retained
    reopened_objects.close()
    reopened_ingress.close()
    candidate_connection.close()


def test_record_requires_governed_record_class_and_privileged_admission(
    tmp_path: Path,
) -> None:
    objects, _ = _open_objects(tmp_path / "objects.sqlite3")
    opaque = canonical_json_bytes(
        {"record_id": "opaque", "record_type": "SOURCE_RECORD"}
    )
    with pytest.raises(PermissionError):
        _admit(objects, "evidence.record", opaque, "unprivileged", reader=True)

    source_class_id = _admit(objects, "evidence.source", opaque, "wrong-class")
    with pytest.raises(ObjectHydrationDenied):
        objects.objects.hydrate(
            # The configured record policy cannot hydrate a source-class object.
            HydrationRequest(source_class_id, "evidence.record"),
            proof=proof(),
        )
    objects.close()


@pytest.mark.parametrize("remove_last", (False, True))
def test_shared_record_validator_rejects_duplicate_or_missing_rows(
    tmp_path: Path, remove_last: bool
) -> None:
    candidate_connection, _, version = _candidate(tmp_path)
    package, records = _package_and_records(version, "The deadline changed.")
    rows = tuple(
        (
            record["record_id"],
            record["record_type"],
            canonical_json_bytes(record).decode(),
            digest_bytes(canonical_json_bytes(record)),
        )
        for record in records
    )
    invalid_rows = rows[:-1] if remove_last else (*rows[:-1], rows[0])

    assert (
        validate_governed_evidence_records(
            candidate_id=version.candidate_id,
            source_inventory=(("source-1", "https://example.test/current"),),
            base_package_digest=records[0]["base_package_digest"],
            package=package,
            retained_records=invalid_rows,
        )
        is None
    )
    candidate_connection.close()
