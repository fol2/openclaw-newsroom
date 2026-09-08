"""Strict, provider-independent Triage Proposal contract for Increment 6C1.

A proposal is immutable untrusted fixture output.  Its content digest is not a
decision: parsing or retaining it creates no Hypothesis or Candidate and grants
no evidence, publication, operational, or other editorial authority.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from enum import StrEnum
from typing import Mapping

from newsroom.authority.canonical import (
    CanonicalizationError,
    canonical_json_bytes,
    digest_bytes,
    validate_sha256_digest,
)
from newsroom.increment5.retrieval_context import RETRIEVAL_CONTEXT_CONTRACT_DIGEST


PROPOSAL_SCHEMA_VERSION = "newsroom.increment6.triage-proposal.v1"
TRIAGE_PROPOSAL = PROPOSAL_SCHEMA_VERSION
PROPOSAL_CONTENT_IDENTITY = "SHA256_CANONICAL_SCHEMA_AND_PROPOSAL"
PROPOSAL_NO_AUTHORITY_BOUNDARY = (
    "NO_HYPOTHESIS_OR_CANDIDATE_MUTATION;"
    "NO_PUBLICATION_EVIDENCE_OR_OPERATIONAL_AUTHORITY"
)

PROBABILITY_SCALE = 1_000_000
MAX_DECISION_LEADS = 32
MAX_CONTEXT_LEADS = 64
MAX_INPUT_CITATIONS = 64
MAX_LIST_ITEMS = 32
MAX_RATIONALE_BYTES = 4_096
MAX_TEXT_BYTES = 1_024
MAX_CITATION_SPAN_BYTES = 262_144

_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:\-]{0,255}\Z")
_DECIMAL = re.compile(r"(?:0\.[0-9]{6}|1\.000000)\Z")


class ProposalContractError(ValueError):
    """Untrusted proposal bytes do not satisfy the exact v1 contract."""


class ProposalRoute(StrEnum):
    EDITORIAL_REJECT = "EDITORIAL_REJECT"
    WATCH_DEFER = "WATCH_DEFER"
    ASSOCIATE_WITHOUT_CANDIDATE = "ASSOCIATE_WITHOUT_CANDIDATE"
    SUPPLEMENTAL_DISCOVERY = "SUPPLEMENTAL_DISCOVERY"
    OPERATIONAL_HOLD = "OPERATIONAL_HOLD"
    NEW_EVENT_CANDIDATE = "NEW_EVENT_CANDIDATE"
    DEVELOPMENT_CANDIDATE = "DEVELOPMENT_CANDIDATE"
    CORRECTION_CANDIDATE = "CORRECTION_CANDIDATE"


class WorkerKind(StrEnum):
    FAKE = "FAKE"
    REPLAY = "REPLAY"
    AUTONOMOUS_DETERMINISTIC = "AUTONOMOUS_DETERMINISTIC"


# Retained import compatibility for the accepted fixture contracts.  New
# production code names the complete enum truthfully as WorkerKind.
FixtureWorkerKind = WorkerKind


class CitationSourceKind(StrEnum):
    DECISION_LEAD = "DECISION_LEAD"
    CONTEXT_LEAD = "CONTEXT_LEAD"
    RETRIEVAL_MATCH = "RETRIEVAL_MATCH"
    RETRIEVAL_CONTRADICTION = "RETRIEVAL_CONTRADICTION"


class HypothesisRelationship(StrEnum):
    SAME_STATE = "SAME_STATE"
    DEVELOPMENT_OF = "DEVELOPMENT_OF"
    CORRECTION_REVERSAL_OF = "CORRECTION_REVERSAL_OF"
    RELATED_DISTINCT = "RELATED_DISTINCT"
    NO_ADEQUATE_PRIOR_MATCH = "NO_ADEQUATE_PRIOR_MATCH"
    UNCERTAIN = "UNCERTAIN"


class CandidateManifestKind(StrEnum):
    NEW_EVENT = "NEW_EVENT"
    DEVELOPMENT = "DEVELOPMENT"
    CORRECTION = "CORRECTION"


def _exact_keys(value: object, expected: set[str], field: str) -> Mapping[str, object]:
    if not isinstance(value, dict) or set(value) != expected:
        raise ProposalContractError(f"{field} keys are not exact")
    return value


def _digest(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise ProposalContractError(f"{field} must be a canonical SHA-256 digest")
    try:
        return validate_sha256_digest(value, field=field)
    except CanonicalizationError as exc:
        raise ProposalContractError(str(exc)) from exc


def _text(value: object, field: str, maximum_bytes: int = MAX_TEXT_BYTES) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value.encode("utf-8")) > maximum_bytes
    ):
        raise ProposalContractError(f"{field} must be bounded canonical text")
    return value


def _token(value: object, field: str) -> str:
    if not isinstance(value, str) or _TOKEN.fullmatch(value) is None:
        raise ProposalContractError(f"{field} must be a bounded canonical token")
    return value


def _uuid(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise ProposalContractError(f"{field} must be a canonical UUID")
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError) as exc:
        raise ProposalContractError(f"{field} must be a canonical UUID") from exc
    if str(parsed) != value:
        raise ProposalContractError(f"{field} must be a canonical UUID")
    return value


def _optional_uuid(value: object, field: str) -> str | None:
    return None if value is None else _uuid(value, field)


def _optional_text(value: object, field: str) -> str | None:
    return None if value is None else _text(value, field)


def _optional_token(value: object, field: str) -> str | None:
    return None if value is None else _token(value, field)


def _uint(value: object, field: str, *, minimum: int = 0, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ProposalContractError(f"{field} must be a bounded non-negative integer")
    if maximum is not None and value > maximum:
        raise ProposalContractError(f"{field} must be a bounded non-negative integer")
    return value


def _enum(enum_type: type[StrEnum], value: object, field: str) -> StrEnum:
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        raise ProposalContractError(f"{field} is unsupported") from exc


def _string_list(
    value: object,
    field: str,
    *,
    maximum_items: int = MAX_LIST_ITEMS,
    allow_empty: bool = True,
    tokens: bool = False,
) -> tuple[str, ...]:
    if not isinstance(value, list) or len(value) > maximum_items:
        raise ProposalContractError(f"{field} must be a bounded array")
    if not allow_empty and not value:
        raise ProposalContractError(f"{field} must not be empty")
    converter = _token if tokens else _text
    result = tuple(converter(item, field) for item in value)
    if result != tuple(sorted(set(result))):
        raise ProposalContractError(f"{field} must be sorted and unique")
    return result


def _uuid_list(
    value: object,
    field: str,
    *,
    maximum_items: int,
    allow_empty: bool,
) -> tuple[str, ...]:
    if not isinstance(value, list) or len(value) > maximum_items:
        raise ProposalContractError(f"{field} must be a bounded array")
    if not allow_empty and not value:
        raise ProposalContractError(f"{field} must not be empty")
    result = tuple(_uuid(item, field) for item in value)
    if result != tuple(sorted(set(result))):
        raise ProposalContractError(f"{field} must be sorted and unique")
    return result


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for name, item in pairs:
        if name in value:
            raise ProposalContractError(f"duplicate object name: {name}")
        value[name] = item
    return value


def _decode_canonical(raw: bytes) -> dict[str, object]:
    if not isinstance(raw, bytes):
        raise ProposalContractError("proposal input must be immutable bytes")
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ProposalContractError(f"unsupported JSON constant: {constant}")
            ),
        )
    except ProposalContractError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProposalContractError("proposal is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ProposalContractError("proposal document must be an object")
    try:
        expected = canonical_json_bytes(value)
    except CanonicalizationError as exc:
        raise ProposalContractError("proposal is outside canonical JSON") from exc
    if raw != expected:
        raise ProposalContractError("proposal is not exact canonical JSON")
    return value


def _content_identity(proposal: Mapping[str, object]) -> str:
    return digest_bytes(
        canonical_json_bytes(
            {"schema_version": PROPOSAL_SCHEMA_VERSION, "proposal": proposal}
        )
    )


@dataclass(frozen=True, slots=True)
class FixedProbability:
    decimal: str
    millionths: int

    @classmethod
    def from_value(cls, value: object, field: str) -> "FixedProbability":
        item = _exact_keys(value, {"decimal", "millionths"}, field)
        millionths = _uint(
            item["millionths"], field + ".millionths", maximum=PROBABILITY_SCALE
        )
        decimal = item["decimal"]
        if not isinstance(decimal, str) or _DECIMAL.fullmatch(decimal) is None:
            raise ProposalContractError(
                f"{field}.decimal must use exact six-place [0,1] text"
            )
        expected = f"{millionths // PROBABILITY_SCALE}.{millionths % PROBABILITY_SCALE:06d}"
        if decimal != expected:
            raise ProposalContractError(f"{field} decimal and integer representations differ")
        return cls(decimal=decimal, millionths=millionths)

    def canonical_value(self) -> dict[str, object]:
        return {"decimal": self.decimal, "millionths": self.millionths}


@dataclass(frozen=True, slots=True)
class WorkItemBinding:
    work_item_id: str
    work_item_version_id: str
    work_item_version_digest: str

    @classmethod
    def from_value(cls, value: object) -> "WorkItemBinding":
        item = _exact_keys(
            value,
            {"work_item_id", "work_item_version_id", "work_item_version_digest"},
            "work_item_binding",
        )
        return cls(
            work_item_id=_uuid(item["work_item_id"], "work_item_id"),
            work_item_version_id=_uuid(
                item["work_item_version_id"], "work_item_version_id"
            ),
            work_item_version_digest=_digest(
                item["work_item_version_digest"], "work_item_version_digest"
            ),
        )

    def canonical_value(self) -> dict[str, object]:
        return {
            "work_item_id": self.work_item_id,
            "work_item_version_id": self.work_item_version_id,
            "work_item_version_digest": self.work_item_version_digest,
        }


@dataclass(frozen=True, slots=True)
class RetrievalContextBinding:
    context_id: str
    context_digest: str
    contract_digest: str

    @classmethod
    def from_value(cls, value: object) -> "RetrievalContextBinding":
        item = _exact_keys(
            value,
            {"context_id", "context_digest", "contract_digest"},
            "retrieval_context_binding",
        )
        contract_digest = _digest(item["contract_digest"], "contract_digest")
        if contract_digest != RETRIEVAL_CONTEXT_CONTRACT_DIGEST:
            raise ProposalContractError("retrieval context contract digest differs")
        return cls(
            context_id=_uuid(item["context_id"], "context_id"),
            context_digest=_digest(item["context_digest"], "context_digest"),
            contract_digest=contract_digest,
        )

    def canonical_value(self) -> dict[str, object]:
        return {
            "context_id": self.context_id,
            "context_digest": self.context_digest,
            "contract_digest": self.contract_digest,
        }


@dataclass(frozen=True, slots=True)
class WorkerAttemptBinding:
    attempt_id: str
    attempt_digest: str
    worker_kind: WorkerKind
    worker_version: str
    input_digest: str
    work_item_version_digest: str
    retrieval_context_digest: str

    @classmethod
    def from_value(cls, value: object) -> "WorkerAttemptBinding":
        item = _exact_keys(
            value,
            {
                "attempt_id",
                "attempt_digest",
                "worker_kind",
                "worker_version",
                "input_digest",
                "work_item_version_digest",
                "retrieval_context_digest",
            },
            "worker_attempt_binding",
        )
        worker_kind = _enum(WorkerKind, item["worker_kind"], "worker_kind")
        assert isinstance(worker_kind, WorkerKind)
        return cls(
            attempt_id=_token(item["attempt_id"], "attempt_id"),
            attempt_digest=_digest(item["attempt_digest"], "attempt_digest"),
            worker_kind=worker_kind,
            worker_version=_token(item["worker_version"], "worker_version"),
            input_digest=_digest(item["input_digest"], "input_digest"),
            work_item_version_digest=_digest(
                item["work_item_version_digest"], "worker_work_item_version_digest"
            ),
            retrieval_context_digest=_digest(
                item["retrieval_context_digest"], "worker_retrieval_context_digest"
            ),
        )

    def canonical_value(self) -> dict[str, object]:
        return {
            "attempt_id": self.attempt_id,
            "attempt_digest": self.attempt_digest,
            "worker_kind": self.worker_kind.value,
            "worker_version": self.worker_version,
            "input_digest": self.input_digest,
            "work_item_version_digest": self.work_item_version_digest,
            "retrieval_context_digest": self.retrieval_context_digest,
        }


@dataclass(frozen=True, slots=True)
class InputCitation:
    citation_id: str
    source_kind: CitationSourceKind
    source_id: str
    source_digest: str
    field_path: str
    byte_start: int
    byte_end: int
    quote_digest: str
    target_hypothesis_id: str | None

    @classmethod
    def from_value(cls, value: object) -> "InputCitation":
        item = _exact_keys(
            value,
            {
                "citation_id",
                "source_kind",
                "source_id",
                "source_digest",
                "field_path",
                "byte_start",
                "byte_end",
                "quote_digest",
                "target_hypothesis_id",
            },
            "input citation",
        )
        source_kind = _enum(CitationSourceKind, item["source_kind"], "source_kind")
        assert isinstance(source_kind, CitationSourceKind)
        byte_start = _uint(item["byte_start"], "citation byte_start")
        byte_end = _uint(item["byte_end"], "citation byte_end")
        if byte_end <= byte_start or byte_end - byte_start > MAX_CITATION_SPAN_BYTES:
            raise ProposalContractError("citation byte range is invalid or unbounded")
        source_id = (
            _uuid(item["source_id"], "citation source_id")
            if source_kind
            in {CitationSourceKind.DECISION_LEAD, CitationSourceKind.CONTEXT_LEAD}
            else _token(item["source_id"], "citation source_id")
        )
        target_hypothesis_id = _optional_uuid(
            item["target_hypothesis_id"], "citation target_hypothesis_id"
        )
        if (
            source_kind
            in {CitationSourceKind.DECISION_LEAD, CitationSourceKind.CONTEXT_LEAD}
            and target_hypothesis_id is not None
        ):
            raise ProposalContractError(
                "a Lead citation cannot bind a retrieved Hypothesis target"
            )
        return cls(
            citation_id=_token(item["citation_id"], "citation_id"),
            source_kind=source_kind,
            source_id=source_id,
            source_digest=_digest(item["source_digest"], "citation source_digest"),
            field_path=_text(item["field_path"], "citation field_path", 256),
            byte_start=byte_start,
            byte_end=byte_end,
            quote_digest=_digest(item["quote_digest"], "citation quote_digest"),
            target_hypothesis_id=target_hypothesis_id,
        )

    def canonical_value(self) -> dict[str, object]:
        return {
            "citation_id": self.citation_id,
            "source_kind": self.source_kind.value,
            "source_id": self.source_id,
            "source_digest": self.source_digest,
            "field_path": self.field_path,
            "byte_start": self.byte_start,
            "byte_end": self.byte_end,
            "quote_digest": self.quote_digest,
            "target_hypothesis_id": self.target_hypothesis_id,
        }


@dataclass(frozen=True, slots=True)
class ProposedHypothesis:
    proposal_local_id: str
    summary: str
    relationship_kind: HypothesisRelationship
    target_hypothesis_id: str | None

    @classmethod
    def from_value(cls, value: object) -> "ProposedHypothesis":
        item = _exact_keys(
            value,
            {
                "proposal_local_id",
                "summary",
                "relationship_kind",
                "target_hypothesis_id",
            },
            "hypothesis",
        )
        relationship = _enum(
            HypothesisRelationship, item["relationship_kind"], "relationship_kind"
        )
        assert isinstance(relationship, HypothesisRelationship)
        target = _optional_uuid(item["target_hypothesis_id"], "target_hypothesis_id")
        target_required = relationship is not HypothesisRelationship.NO_ADEQUATE_PRIOR_MATCH
        if target_required != (target is not None):
            raise ProposalContractError(
                "hypothesis relationship target does not match relationship kind"
            )
        return cls(
            proposal_local_id=_token(item["proposal_local_id"], "proposal_local_id"),
            summary=_text(item["summary"], "hypothesis summary"),
            relationship_kind=relationship,
            target_hypothesis_id=target,
        )

    def canonical_value(self) -> dict[str, object]:
        return {
            "proposal_local_id": self.proposal_local_id,
            "summary": self.summary,
            "relationship_kind": self.relationship_kind.value,
            "target_hypothesis_id": self.target_hypothesis_id,
        }


@dataclass(frozen=True, slots=True)
class WatchAction:
    condition_kind: str
    condition: str
    next_action: str

    @classmethod
    def from_value(cls, value: object) -> "WatchAction":
        item = _exact_keys(
            value, {"condition_kind", "condition", "next_action"}, "watch action"
        )
        allowed = {
            "SOURCE_UPDATE",
            "CORROBORATING_LEAD",
            "OCCURRENCE",
            "DEADLINE",
            "EXPIRY",
            "REVIEW",
        }
        condition_kind = _token(item["condition_kind"], "watch condition_kind")
        if condition_kind not in allowed:
            raise ProposalContractError("watch condition_kind is unsupported")
        return cls(
            condition_kind=condition_kind,
            condition=_text(item["condition"], "watch condition"),
            next_action=_token(item["next_action"], "watch next_action"),
        )

    def canonical_value(self) -> dict[str, object]:
        return {
            "condition_kind": self.condition_kind,
            "condition": self.condition,
            "next_action": self.next_action,
        }


@dataclass(frozen=True, slots=True)
class SupplementalAction:
    action_kind: str
    scope: str
    maximum_attempts: int
    requires_approval: bool

    @classmethod
    def from_value(cls, value: object) -> "SupplementalAction":
        item = _exact_keys(
            value,
            {"action_kind", "scope", "maximum_attempts", "requires_approval"},
            "supplemental action",
        )
        if item["requires_approval"] is not True:
            raise ProposalContractError(
                "a supplemental proposal must retain the approval boundary"
            )
        return cls(
            action_kind=_token(item["action_kind"], "supplemental action_kind"),
            scope=_text(item["scope"], "supplemental scope"),
            maximum_attempts=_uint(
                item["maximum_attempts"],
                "supplemental maximum_attempts",
                minimum=1,
                maximum=10,
            ),
            requires_approval=True,
        )

    def canonical_value(self) -> dict[str, object]:
        return {
            "action_kind": self.action_kind,
            "scope": self.scope,
            "maximum_attempts": self.maximum_attempts,
            "requires_approval": self.requires_approval,
        }


@dataclass(frozen=True, slots=True)
class OperationalAction:
    action_kind: str
    owner_id: str | None
    dependency: str | None
    retry_condition: str | None
    review_condition: str | None
    expiry_condition: str | None

    @classmethod
    def from_value(cls, value: object) -> "OperationalAction":
        item = _exact_keys(
            value,
            {
                "action_kind",
                "owner_id",
                "dependency",
                "retry_condition",
                "review_condition",
                "expiry_condition",
            },
            "operational action",
        )
        action_kind = _optional_token(item["action_kind"], "operational action_kind")
        owner_id = _optional_token(item["owner_id"], "operational owner_id")
        dependency = _optional_token(
            item["dependency"], "operational dependency"
        )
        retry_condition = _optional_text(
            item["retry_condition"], "operational retry_condition"
        )
        review_condition = _optional_text(
            item["review_condition"], "operational review_condition"
        )
        expiry_condition = _optional_text(
            item["expiry_condition"], "operational expiry_condition"
        )
        if action_kind is None or not any(
            (
                owner_id,
                dependency,
                retry_condition,
                review_condition,
                expiry_condition,
            )
        ):
            raise ProposalContractError(
                "operational hold requires an inspectable action and condition"
            )
        return cls(
            action_kind=action_kind,
            owner_id=owner_id,
            dependency=dependency,
            retry_condition=retry_condition,
            review_condition=review_condition,
            expiry_condition=expiry_condition,
        )

    def canonical_value(self) -> dict[str, object]:
        return {
            "action_kind": self.action_kind,
            "owner_id": self.owner_id,
            "dependency": self.dependency,
            "retry_condition": self.retry_condition,
            "review_condition": self.review_condition,
            "expiry_condition": self.expiry_condition,
        }


@dataclass(frozen=True, slots=True)
class CandidateManifest:
    manifest_kind: CandidateManifestKind
    contributing_lead_ids: tuple[str, ...]
    proposed_geography: str
    proposed_category: str
    urgency: str
    likely_new_information: str
    reader_utility_basis: str
    uncertainties: tuple[str, ...]
    evidence_objectives: tuple[str, ...]
    governing_versions: tuple[str, ...]

    @classmethod
    def from_value(cls, value: object) -> "CandidateManifest":
        item = _exact_keys(
            value,
            {
                "manifest_kind",
                "contributing_lead_ids",
                "proposed_geography",
                "proposed_category",
                "urgency",
                "likely_new_information",
                "reader_utility_basis",
                "uncertainties",
                "evidence_objectives",
                "governing_versions",
            },
            "Candidate manifest",
        )
        kind = _enum(CandidateManifestKind, item["manifest_kind"], "manifest_kind")
        assert isinstance(kind, CandidateManifestKind)
        return cls(
            manifest_kind=kind,
            contributing_lead_ids=_uuid_list(
                item["contributing_lead_ids"],
                "contributing_lead_ids",
                maximum_items=MAX_DECISION_LEADS + MAX_CONTEXT_LEADS,
                allow_empty=False,
            ),
            proposed_geography=_token(item["proposed_geography"], "proposed_geography"),
            proposed_category=_token(item["proposed_category"], "proposed_category"),
            urgency=_token(item["urgency"], "urgency"),
            likely_new_information=_text(
                item["likely_new_information"], "Candidate likely_new_information"
            ),
            reader_utility_basis=_text(
                item["reader_utility_basis"], "Candidate reader_utility_basis"
            ),
            uncertainties=_string_list(
                item["uncertainties"], "Candidate uncertainties", allow_empty=False
            ),
            evidence_objectives=_string_list(
                item["evidence_objectives"],
                "Candidate evidence_objectives",
                allow_empty=False,
            ),
            governing_versions=_string_list(
                item["governing_versions"],
                "Candidate governing_versions",
                allow_empty=False,
                tokens=True,
            ),
        )

    def canonical_value(self) -> dict[str, object]:
        return {
            "manifest_kind": self.manifest_kind.value,
            "contributing_lead_ids": list(self.contributing_lead_ids),
            "proposed_geography": self.proposed_geography,
            "proposed_category": self.proposed_category,
            "urgency": self.urgency,
            "likely_new_information": self.likely_new_information,
            "reader_utility_basis": self.reader_utility_basis,
            "uncertainties": list(self.uncertainties),
            "evidence_objectives": list(self.evidence_objectives),
            "governing_versions": list(self.governing_versions),
        }


@dataclass(frozen=True, slots=True)
class LeadRecommendation:
    decision_lead_id: str
    route: ProposalRoute
    confidence: FixedProbability
    uncertainty: FixedProbability
    input_citations: tuple[InputCitation, ...]
    likely_new_information: str
    materiality_basis: str
    missing_context: tuple[str, ...]
    retrieval_incompleteness: tuple[str, ...]
    hypothesis: ProposedHypothesis | None
    watch_action: WatchAction | None
    supplemental_action: SupplementalAction | None
    operational_action: OperationalAction | None
    candidate_manifest: CandidateManifest | None

    @classmethod
    def from_value(cls, value: object) -> "LeadRecommendation":
        item = _exact_keys(
            value,
            {
                "decision_lead_id",
                "route",
                "confidence",
                "uncertainty",
                "input_citations",
                "likely_new_information",
                "materiality_basis",
                "missing_context",
                "retrieval_incompleteness",
                "hypothesis",
                "watch_action",
                "supplemental_action",
                "operational_action",
                "candidate_manifest",
            },
            "recommendation",
        )
        route = _enum(ProposalRoute, item["route"], "proposal route")
        assert isinstance(route, ProposalRoute)
        raw_citations = item["input_citations"]
        if (
            not isinstance(raw_citations, list)
            or not 1 <= len(raw_citations) <= MAX_INPUT_CITATIONS
        ):
            raise ProposalContractError("input citations must be a non-empty bounded array")
        citations = tuple(InputCitation.from_value(citation) for citation in raw_citations)
        citation_ids = tuple(citation.citation_id for citation in citations)
        if citation_ids != tuple(sorted(set(citation_ids))):
            raise ProposalContractError("input citations must be sorted and unique")

        hypothesis = (
            None
            if item["hypothesis"] is None
            else ProposedHypothesis.from_value(item["hypothesis"])
        )
        watch = (
            None
            if item["watch_action"] is None
            else WatchAction.from_value(item["watch_action"])
        )
        supplemental = (
            None
            if item["supplemental_action"] is None
            else SupplementalAction.from_value(item["supplemental_action"])
        )
        operational = (
            None
            if item["operational_action"] is None
            else OperationalAction.from_value(item["operational_action"])
        )
        candidate = (
            None
            if item["candidate_manifest"] is None
            else CandidateManifest.from_value(item["candidate_manifest"])
        )

        if (route is ProposalRoute.WATCH_DEFER) != (watch is not None):
            raise ProposalContractError("watch action must exist only for WATCH_DEFER")
        if (route is ProposalRoute.SUPPLEMENTAL_DISCOVERY) != (supplemental is not None):
            raise ProposalContractError(
                "supplemental action must exist only for SUPPLEMENTAL_DISCOVERY"
            )
        if (route is ProposalRoute.OPERATIONAL_HOLD) != (operational is not None):
            raise ProposalContractError(
                "operational action must exist only for OPERATIONAL_HOLD"
            )
        candidate_kinds = {
            ProposalRoute.NEW_EVENT_CANDIDATE: CandidateManifestKind.NEW_EVENT,
            ProposalRoute.DEVELOPMENT_CANDIDATE: CandidateManifestKind.DEVELOPMENT,
            ProposalRoute.CORRECTION_CANDIDATE: CandidateManifestKind.CORRECTION,
        }
        expected_kind = candidate_kinds.get(route)
        if (expected_kind is None) != (candidate is None):
            raise ProposalContractError(
                "Candidate manifest must exist only for a Candidate route"
            )
        if candidate is not None and candidate.manifest_kind is not expected_kind:
            raise ProposalContractError("Candidate manifest kind differs from route")

        hypothesis_routes = {
            ProposalRoute.ASSOCIATE_WITHOUT_CANDIDATE,
            *candidate_kinds,
        }
        if (route in hypothesis_routes) != (hypothesis is not None):
            raise ProposalContractError(
                "hypothesis must exist exactly for association or Candidate routes"
            )
        required_relationships = {
            ProposalRoute.NEW_EVENT_CANDIDATE: HypothesisRelationship.NO_ADEQUATE_PRIOR_MATCH,
            ProposalRoute.DEVELOPMENT_CANDIDATE: HypothesisRelationship.DEVELOPMENT_OF,
            ProposalRoute.CORRECTION_CANDIDATE: HypothesisRelationship.CORRECTION_REVERSAL_OF,
        }
        required_relationship = required_relationships.get(route)
        if (
            required_relationship is not None
            and hypothesis is not None
            and hypothesis.relationship_kind is not required_relationship
        ):
            raise ProposalContractError("hypothesis relationship differs from Candidate route")
        if route is ProposalRoute.ASSOCIATE_WITHOUT_CANDIDATE and (
            hypothesis is None
            or hypothesis.target_hypothesis_id is None
            or hypothesis.relationship_kind
            not in {
                HypothesisRelationship.SAME_STATE,
                HypothesisRelationship.RELATED_DISTINCT,
                HypothesisRelationship.UNCERTAIN,
            }
        ):
            raise ProposalContractError(
                "association must name an allowed prior Hypothesis relationship"
            )

        return cls(
            decision_lead_id=_uuid(item["decision_lead_id"], "decision_lead_id"),
            route=route,
            confidence=FixedProbability.from_value(item["confidence"], "confidence"),
            uncertainty=FixedProbability.from_value(item["uncertainty"], "uncertainty"),
            input_citations=citations,
            likely_new_information=_text(
                item["likely_new_information"], "likely_new_information"
            ),
            materiality_basis=_text(item["materiality_basis"], "materiality_basis"),
            missing_context=_string_list(item["missing_context"], "missing_context"),
            retrieval_incompleteness=_string_list(
                item["retrieval_incompleteness"], "retrieval_incompleteness"
            ),
            hypothesis=hypothesis,
            watch_action=watch,
            supplemental_action=supplemental,
            operational_action=operational,
            candidate_manifest=candidate,
        )

    def canonical_value(self) -> dict[str, object]:
        return {
            "decision_lead_id": self.decision_lead_id,
            "route": self.route.value,
            "confidence": self.confidence.canonical_value(),
            "uncertainty": self.uncertainty.canonical_value(),
            "input_citations": [citation.canonical_value() for citation in self.input_citations],
            "likely_new_information": self.likely_new_information,
            "materiality_basis": self.materiality_basis,
            "missing_context": list(self.missing_context),
            "retrieval_incompleteness": list(self.retrieval_incompleteness),
            "hypothesis": None if self.hypothesis is None else self.hypothesis.canonical_value(),
            "watch_action": (
                None
                if self.watch_action is None
                else self.watch_action.canonical_value()
            ),
            "supplemental_action": (
                None
                if self.supplemental_action is None
                else self.supplemental_action.canonical_value()
            ),
            "operational_action": (
                None
                if self.operational_action is None
                else self.operational_action.canonical_value()
            ),
            "candidate_manifest": (
                None
                if self.candidate_manifest is None
                else self.candidate_manifest.canonical_value()
            ),
        }


@dataclass(frozen=True, slots=True)
class ProposalAuthority:
    effect: str = "NONE"
    creates_hypothesis: bool = False
    creates_candidate: bool = False
    mutates_editorial_state: bool = False
    publication_authority: bool = False
    evidence_authority: bool = False
    operational_authority: bool = False

    @classmethod
    def from_value(cls, value: object) -> "ProposalAuthority":
        fields = {
            "effect",
            "creates_hypothesis",
            "creates_candidate",
            "mutates_editorial_state",
            "publication_authority",
            "evidence_authority",
            "operational_authority",
        }
        item = _exact_keys(value, fields, "authority")
        if item["effect"] != "NONE" or any(
            item[name] is not False for name in fields - {"effect"}
        ):
            raise ProposalContractError(
                "a Triage Proposal grants no authority and has no editorial effect"
            )
        return cls()

    def canonical_value(self) -> dict[str, object]:
        return {
            "effect": self.effect,
            "creates_hypothesis": self.creates_hypothesis,
            "creates_candidate": self.creates_candidate,
            "mutates_editorial_state": self.mutates_editorial_state,
            "publication_authority": self.publication_authority,
            "evidence_authority": self.evidence_authority,
            "operational_authority": self.operational_authority,
        }


@dataclass(frozen=True, slots=True)
class TriageProposal:
    proposal_id: str
    work_item: WorkItemBinding
    retrieval_context: RetrievalContextBinding
    worker_attempt: WorkerAttemptBinding
    decision_lead_ids: tuple[str, ...]
    context_lead_ids: tuple[str, ...]
    recommendations: tuple[LeadRecommendation, ...]
    rationale: str
    authority: ProposalAuthority
    content_identity: str

    @classmethod
    def from_canonical_bytes(cls, raw: bytes) -> "TriageProposal":
        document = _decode_canonical(raw)
        root = _exact_keys(
            document,
            {"schema_version", "content_identity", "proposal"},
            "proposal document",
        )
        if root["schema_version"] != PROPOSAL_SCHEMA_VERSION:
            raise ProposalContractError("proposal schema version is unsupported")
        content_identity = _digest(root["content_identity"], "content_identity")
        proposal = _exact_keys(
            root["proposal"],
            {
                "proposal_id",
                "work_item_binding",
                "retrieval_context_binding",
                "worker_attempt_binding",
                "decision_lead_ids",
                "context_lead_ids",
                "recommendations",
                "rationale",
                "authority",
            },
            "proposal",
        )
        if _content_identity(proposal) != content_identity:
            raise ProposalContractError("proposal content identity differs")

        work_item = WorkItemBinding.from_value(proposal["work_item_binding"])
        retrieval_context = RetrievalContextBinding.from_value(
            proposal["retrieval_context_binding"]
        )
        worker_attempt = WorkerAttemptBinding.from_value(
            proposal["worker_attempt_binding"]
        )
        if worker_attempt.work_item_version_digest != work_item.work_item_version_digest:
            raise ProposalContractError("worker attempt work item digest differs")
        if worker_attempt.retrieval_context_digest != retrieval_context.context_digest:
            raise ProposalContractError("worker attempt retrieval context digest differs")

        decision_leads = _uuid_list(
            proposal["decision_lead_ids"],
            "decision_lead_ids",
            maximum_items=MAX_DECISION_LEADS,
            allow_empty=False,
        )
        context_leads = _uuid_list(
            proposal["context_lead_ids"],
            "context_lead_ids",
            maximum_items=MAX_CONTEXT_LEADS,
            allow_empty=True,
        )
        if set(decision_leads) & set(context_leads):
            raise ProposalContractError("decision and context Lead manifests overlap")

        raw_recommendations = proposal["recommendations"]
        if (
            not isinstance(raw_recommendations, list)
            or not 1 <= len(raw_recommendations) <= MAX_DECISION_LEADS
        ):
            raise ProposalContractError("recommendations must be a non-empty bounded array")
        recommendations = tuple(
            LeadRecommendation.from_value(value) for value in raw_recommendations
        )
        recommendation_leads = tuple(item.decision_lead_id for item in recommendations)
        if recommendation_leads != decision_leads:
            raise ProposalContractError(
                "every decision Lead must be recommended exactly once in manifest order"
            )

        admitted_leads = set(decision_leads) | set(context_leads)
        recommendations_by_lead = {
            recommendation.decision_lead_id: recommendation
            for recommendation in recommendations
        }
        hypotheses_by_local_id: dict[str, dict[str, object]] = {}
        for recommendation in recommendations:
            hypothesis = recommendation.hypothesis
            if hypothesis is None:
                continue
            hypothesis_value = hypothesis.canonical_value()
            prior = hypotheses_by_local_id.setdefault(
                hypothesis.proposal_local_id, hypothesis_value
            )
            if prior != hypothesis_value:
                raise ProposalContractError(
                    "proposal_local_id conflicts across Hypothesis recommendations"
                )
            if hypothesis.target_hypothesis_id is not None and not any(
                citation.source_kind
                in {
                    CitationSourceKind.RETRIEVAL_MATCH,
                    CitationSourceKind.RETRIEVAL_CONTRADICTION,
                }
                and citation.target_hypothesis_id
                == hypothesis.target_hypothesis_id
                for citation in recommendation.input_citations
            ):
                raise ProposalContractError(
                    "relationship target lacks an exact Retrieval Context citation"
                )

        for recommendation in recommendations:
            for citation in recommendation.input_citations:
                if (
                    citation.source_kind is CitationSourceKind.DECISION_LEAD
                    and citation.source_id not in decision_leads
                ):
                    raise ProposalContractError(
                        "citation does not name the decision Lead manifest"
                    )
                if (
                    citation.source_kind is CitationSourceKind.CONTEXT_LEAD
                    and citation.source_id not in context_leads
                ):
                    raise ProposalContractError(
                        "citation does not name the context Lead manifest"
                    )
            if not any(
                citation.source_kind is CitationSourceKind.DECISION_LEAD
                and citation.source_id == recommendation.decision_lead_id
                for citation in recommendation.input_citations
            ):
                raise ProposalContractError(
                    "recommendation must cite its exact decision Lead input"
                )
            manifest = recommendation.candidate_manifest
            if manifest is not None:
                if (
                    manifest.likely_new_information
                    != recommendation.likely_new_information
                ):
                    raise ProposalContractError(
                        "Candidate manifest likely new information differs"
                    )
                if recommendation.decision_lead_id not in manifest.contributing_lead_ids:
                    raise ProposalContractError(
                        "Candidate manifest omits its decision Lead"
                    )
                if not set(manifest.contributing_lead_ids) <= admitted_leads:
                    raise ProposalContractError(
                        "Candidate manifest contains an unbound Lead"
                    )
                if not set(manifest.contributing_lead_ids) <= set(decision_leads):
                    raise ProposalContractError(
                        "Candidate contributor cannot be a context-only Lead"
                    )
                assert recommendation.hypothesis is not None
                for contributor_id in manifest.contributing_lead_ids:
                    contributor = recommendations_by_lead[contributor_id]
                    if (
                        contributor.candidate_manifest is None
                        or contributor.hypothesis is None
                        or contributor.hypothesis.proposal_local_id
                        != recommendation.hypothesis.proposal_local_id
                        or contributor.candidate_manifest.canonical_value()
                        != manifest.canonical_value()
                    ):
                        raise ProposalContractError(
                            "Candidate contributor is unrelated or has a non-Candidate route"
                        )

        result = cls(
            proposal_id=_uuid(proposal["proposal_id"], "proposal_id"),
            work_item=work_item,
            retrieval_context=retrieval_context,
            worker_attempt=worker_attempt,
            decision_lead_ids=decision_leads,
            context_lead_ids=context_leads,
            recommendations=recommendations,
            rationale=_text(proposal["rationale"], "rationale", MAX_RATIONALE_BYTES),
            authority=ProposalAuthority.from_value(proposal["authority"]),
            content_identity=content_identity,
        )
        if result.canonical_bytes != raw:
            raise ProposalContractError("proposal typed replay differs")
        return result

    @property
    def grants_authority(self) -> bool:
        return False

    @property
    def creates_hypothesis(self) -> bool:
        return False

    @property
    def creates_candidate(self) -> bool:
        return False

    def proposal_value(self) -> dict[str, object]:
        return {
            "proposal_id": self.proposal_id,
            "work_item_binding": self.work_item.canonical_value(),
            "retrieval_context_binding": self.retrieval_context.canonical_value(),
            "worker_attempt_binding": self.worker_attempt.canonical_value(),
            "decision_lead_ids": list(self.decision_lead_ids),
            "context_lead_ids": list(self.context_lead_ids),
            "recommendations": [item.canonical_value() for item in self.recommendations],
            "rationale": self.rationale,
            "authority": self.authority.canonical_value(),
        }

    def canonical_value(self) -> dict[str, object]:
        return {
            "schema_version": PROPOSAL_SCHEMA_VERSION,
            "content_identity": self.content_identity,
            "proposal": self.proposal_value(),
        }

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.canonical_value())


__all__ = [
    "PROPOSAL_CONTENT_IDENTITY",
    "PROPOSAL_NO_AUTHORITY_BOUNDARY",
    "PROPOSAL_SCHEMA_VERSION",
    "TRIAGE_PROPOSAL",
    "CandidateManifest",
    "CandidateManifestKind",
    "CitationSourceKind",
    "FixedProbability",
    "FixtureWorkerKind",
    "HypothesisRelationship",
    "InputCitation",
    "LeadRecommendation",
    "OperationalAction",
    "ProposalAuthority",
    "ProposalContractError",
    "ProposalRoute",
    "ProposedHypothesis",
    "RetrievalContextBinding",
    "SupplementalAction",
    "TriageProposal",
    "WatchAction",
    "WorkerKind",
    "WorkerAttemptBinding",
    "WorkItemBinding",
]
