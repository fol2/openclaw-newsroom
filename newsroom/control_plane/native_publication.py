"""One governed native Story-to-private-serving publication transaction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from newsroom.authority import (
    AggregateId,
    AuthenticationProof,
    AuthorityCommands,
    AuthorityEvents,
    GovernedObjects,
    ObjectAdmissionId,
    ObjectAdmissionPayload,
    ObjectAdmissionRequest,
    SemanticCommand,
)
from newsroom.authority.canonical import (
    canonical_json_bytes,
    digest_bytes,
    validate_sha256_digest,
)
from newsroom.increment6.candidates import StoryCandidateReadPort
from newsroom.increment10.editorial import (
    DECISION_ADMISSION_TYPE,
    DECISION_CLASS,
    DECISION_COMMAND,
    DECISION_USE,
    DecisionReference,
    EditorialPolicyDecision,
    NativeEditorial,
    StoryVersionReceipt,
    StoryVersionRequest,
)
from newsroom.increment10.evidence import GovernedEvidencePackages
from newsroom.increment10.private_serving import (
    AttemptReceipt,
    EvidenceReceipt,
    PrivateServingDelivery,
    PrivateServingReadProof,
    open_private_serving_delivery,
)
from newsroom.increment10.publication import (
    LAUNCH_CAPABILITIES,
    OfflinePublication,
    PublicationReceipt,
    PublicationRequest,
)


class NativePublicationError(ValueError):
    """Raised when the connected native publication transaction cannot advance."""


@dataclass(frozen=True, slots=True)
class NativePublicationBindings:
    target_path: Path
    reader_principal_id: str
    authority_domain: str
    editorial_controller_principal_id: str
    story_principal_id: str
    publication_controller_principal_id: str
    serving_adapter_principal_id: str
    editorial_policy_bundle_digest: str
    editorial_decision_hydration_policy_digest: str
    editorial_story_hydration_policy_digest: str
    editorial_decision_admission_definition_digest: str
    editorial_decision_command_definition_digest: str
    editorial_story_command_definition_digest: str
    editorial_story_admission_definition_digest: str
    publication_authorisation_policy_digest: str
    target_id: str
    target_policy_digest: str
    publication_surface_hydration_policy_digest: str
    publication_transaction_hydration_policy_digest: str
    publication_surface_admission_definition_digest: str
    publication_transaction_admission_definition_digest: str
    publication_command_definition_digest: str
    target_context_digest: str
    serving_attempt_hydration_policy_digest: str
    serving_evidence_hydration_policy_digest: str
    serving_attempt_admission_definition_digest: str
    serving_evidence_admission_definition_digest: str
    serving_attempt_command_definition_digest: str
    serving_evidence_command_definition_digest: str
    source_licence_policy: tuple[tuple[str, str, str], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.target_path, Path):
            raise NativePublicationError("native publication target path is required")
        for name in (
            "reader_principal_id",
            "authority_domain",
            "editorial_controller_principal_id",
            "story_principal_id",
            "publication_controller_principal_id",
            "serving_adapter_principal_id",
            "target_id",
        ):
            value = getattr(self, name)
            if type(value) is not str or not value.strip():
                raise NativePublicationError(
                    f"native publication {name} is required"
                )
        for name in self.__dataclass_fields__:
            if name.endswith("_digest"):
                try:
                    validate_sha256_digest(getattr(self, name))
                except Exception as exc:
                    raise NativePublicationError(
                        f"native publication {name} differs"
                    ) from exc


@dataclass(frozen=True, slots=True)
class NativePublicationResult:
    story_receipt: StoryVersionReceipt
    publication_receipt: PublicationReceipt
    attempt_receipt: AttemptReceipt
    evidence_receipt: EvidenceReceipt
    read_proof: PrivateServingReadProof


class NativePublicationController:
    """Compose existing native authorities into one retry-stable transaction."""

    def __init__(
        self,
        *,
        objects: GovernedObjects,
        commands: AuthorityCommands,
        events: AuthorityEvents,
        candidate_port: StoryCandidateReadPort,
        evidence_packages: GovernedEvidencePackages,
        bindings: NativePublicationBindings,
    ) -> None:
        if not all(
            type(value) is expected
            for value, expected in (
                (objects, GovernedObjects),
                (commands, AuthorityCommands),
                (events, AuthorityEvents),
                (candidate_port, StoryCandidateReadPort),
                (evidence_packages, GovernedEvidencePackages),
                (bindings, NativePublicationBindings),
            )
        ):
            raise NativePublicationError(
                "exact native publication authorities required"
            )
        self._objects = objects
        self._commands = commands
        self._candidate_port = candidate_port
        self._evidence = evidence_packages
        self._bindings = bindings
        self._editorial = NativeEditorial(
            objects=objects,
            commands=commands,
            events=events,
            evidence=evidence_packages,
            reader_principal_id=bindings.reader_principal_id,
            reader_authority_domain=bindings.authority_domain,
            controller_principal_id=bindings.editorial_controller_principal_id,
            story_principal_id=bindings.story_principal_id,
            policy_bundle_digest=bindings.editorial_policy_bundle_digest,
            decision_hydration_policy_digest=(
                bindings.editorial_decision_hydration_policy_digest
            ),
            story_hydration_policy_digest=(
                bindings.editorial_story_hydration_policy_digest
            ),
            decision_command_definition_digest=(
                bindings.editorial_decision_command_definition_digest
            ),
            story_command_definition_digest=(
                bindings.editorial_story_command_definition_digest
            ),
            story_admission_definition_digest=(
                bindings.editorial_story_admission_definition_digest
            ),
        )
        self._publication = OfflinePublication(
            objects=objects,
            commands=commands,
            events=events,
            editorial=self._editorial,
            evidence=evidence_packages,
            reader_principal_id=bindings.reader_principal_id,
            authority_domain=bindings.authority_domain,
            controller_principal_id=bindings.publication_controller_principal_id,
            authorisation_policy_digest=(
                bindings.publication_authorisation_policy_digest
            ),
            target_id=bindings.target_id,
            target_policy_digest=bindings.target_policy_digest,
            target_capabilities=LAUNCH_CAPABILITIES,
            surface_hydration_policy_digest=(
                bindings.publication_surface_hydration_policy_digest
            ),
            transaction_hydration_policy_digest=(
                bindings.publication_transaction_hydration_policy_digest
            ),
            surface_admission_definition_digest=(
                bindings.publication_surface_admission_definition_digest
            ),
            transaction_admission_definition_digest=(
                bindings.publication_transaction_admission_definition_digest
            ),
            command_definition_digest=bindings.publication_command_definition_digest,
            source_licence_policy=bindings.source_licence_policy,
        )
        self._delivery = open_private_serving_delivery(
            bindings.target_path,
            objects=objects,
            commands=commands,
            events=events,
            publication=self._publication,
            adapter_principal_id=bindings.serving_adapter_principal_id,
            authority_domain=bindings.authority_domain,
            target_id=bindings.target_id,
            target_context_digest=bindings.target_context_digest,
            attempt_hydration_policy_digest=(
                bindings.serving_attempt_hydration_policy_digest
            ),
            evidence_hydration_policy_digest=(
                bindings.serving_evidence_hydration_policy_digest
            ),
            attempt_admission_definition_digest=(
                bindings.serving_attempt_admission_definition_digest
            ),
            evidence_admission_definition_digest=(
                bindings.serving_evidence_admission_definition_digest
            ),
            attempt_command_definition_digest=(
                bindings.serving_attempt_command_definition_digest
            ),
            evidence_command_definition_digest=(
                bindings.serving_evidence_command_definition_digest
            ),
        )

    def close(self) -> None:
        self._delivery.close()

    def advance(
        self,
        package_admission_id: ObjectAdmissionId,
        editorial_decision: EditorialPolicyDecision,
        *,
        expected_story_version: int,
        expected_publication_version: int,
        expected_delivery_evidence_version: int,
        applied_at: str,
        observed_at: str,
        proof: AuthenticationProof,
    ) -> NativePublicationResult:
        if (
            type(package_admission_id) is not ObjectAdmissionId
            or type(editorial_decision) is not EditorialPolicyDecision
            or any(
                type(value) is not int or value < 0
                for value in (
                    expected_story_version,
                    expected_publication_version,
                    expected_delivery_evidence_version,
                )
            )
        ):
            raise NativePublicationError("native publication request differs")
        retained = self._evidence.read(
            package_admission_id,
            candidate_port=self._candidate_port,
            proof=proof,
        )
        identity = retained.package.candidate_id
        story_id = _aggregate("story", identity)
        publication_id = _aggregate("publication", identity)
        decision_reference = self._record_decision(editorial_decision, proof=proof)
        story_receipt, _story = self._editorial.admit_story_version(
            StoryVersionRequest(
                story_id,
                expected_story_version,
                f"native-story:{identity}:{expected_story_version + 1}",
            ),
            package_admission_id=package_admission_id,
            decision_reference=decision_reference,
            candidate_port=self._candidate_port,
            proof=proof,
        )
        publication_receipt, _transaction = self._publication.decide(
            PublicationRequest(
                publication_id,
                expected_publication_version,
                f"native-publication:{identity}:{expected_publication_version + 1}",
                "AUTO_PUBLISH",
                ("NATIVE_STORY_WRITE_READY",),
                editorial_decision.evaluated_at,
            ),
            story_receipt=story_receipt,
            candidate_port=self._candidate_port,
            proof=proof,
        )
        attempt_receipt, _batch = self._delivery.begin(
            publication_receipt,
            story_receipt=story_receipt,
            candidate_port=self._candidate_port,
            proof=proof,
        )
        self._delivery.apply(
            attempt_receipt,
            publication_receipt=publication_receipt,
            story_receipt=story_receipt,
            candidate_port=self._candidate_port,
            applied_at=applied_at,
            proof=proof,
        )
        evidence = self._delivery.observe(
            attempt_receipt,
            publication_receipt=publication_receipt,
            story_receipt=story_receipt,
            candidate_port=self._candidate_port,
            observed_at=observed_at,
            proof=proof,
        )
        evidence_receipt = self._delivery.record(
            evidence,
            attempt_receipt,
            expected_version=expected_delivery_evidence_version,
            proof=proof,
        )
        read_proof = self._delivery.acknowledged_read_proof(
            evidence_receipt,
            attempt_receipt,
            publication_receipt=publication_receipt,
            story_receipt=story_receipt,
            candidate_port=self._candidate_port,
            proof=proof,
        )
        if read_proof is None:
            raise NativePublicationError("private delivery is not acknowledged")
        return NativePublicationResult(
            story_receipt,
            publication_receipt,
            attempt_receipt,
            evidence_receipt,
            read_proof,
        )

    def _record_decision(
        self,
        decision: EditorialPolicyDecision,
        *,
        proof: AuthenticationProof,
    ) -> DecisionReference:
        raw = decision.canonical_bytes()
        admission = self._objects.admit(
            ObjectAdmissionRequest(DECISION_ADMISSION_TYPE, decision.decision_id),
            raw,
            proof=proof,
        ).admission
        if (
            admission.definition_digest
            != self._bindings.editorial_decision_admission_definition_digest
            or admission.blob.blob_digest != digest_bytes(raw)
            or admission.object_class != DECISION_CLASS
            or admission.allowed_use != DECISION_USE
            or not admission.active
        ):
            raise NativePublicationError("editorial decision admission differs")
        committed = self._commands.execute(
            SemanticCommand(
                DECISION_COMMAND,
                _aggregate("editorial-decision", decision.decision_id),
                0,
                ObjectAdmissionPayload(admission.admission_id),
                f"native-editorial-decision:{decision.decision_id}",
            ),
            proof=proof,
        )
        return DecisionReference(committed.event_id, admission.admission_id)


def _aggregate(kind: str, identity: str) -> AggregateId:
    digest = digest_bytes(canonical_json_bytes([kind, identity]))
    raw = bytearray.fromhex(
        digest.removeprefix("sha256:")[:32]
    )
    raw[6] = (raw[6] & 0x0F) | 0x40
    raw[8] = (raw[8] & 0x3F) | 0x80
    return AggregateId(UUID(bytes=bytes(raw)))
