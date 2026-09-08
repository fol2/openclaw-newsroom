"""Observed GOV.UK text-reuse policy for the autonomous private evidence route.

No fixture reviews or reviewer quorum are minted. The two official licence
pages were observed over HTTPS on 8 September 2026. A changed substantive terms
text is a source-local HOLD, not an implicit new licence or a daemon-wide stop.
"""

from __future__ import annotations

import ssl
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from urllib.parse import urlsplit

from lxml import html

from newsroom.authority import AuthenticationProof, GovernedObjects, ObjectAdmissionRequest
from newsroom.authority.canonical import digest_bytes, digest_canonical
from newsroom.authority.types import ObjectAdmissionId

from .govuk_evidence import MAX_BODY_BYTES, TIMEOUT_SECONDS, _NoRedirect
from .native_evidence import NativeEvidenceHold, PublicationRightsAssessment

REUSE_URL = "https://www.gov.uk/help/reuse-govuk-content"
LICENCE_URL = "https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/"
REVIEWED_TEXT = {
    REUSE_URL: "sha256:b58998ef4b7ffc6754b6780bbec15f4b909f7662a47186329c175a86ea37d9ea",
    LICENCE_URL: "sha256:c9f8aa884c89702fc694ea97c91d8db796e6f2a966d93f08e05731c7ab088c61",
}
ATTRIBUTION = "Contains public sector information licensed under the Open Government Licence v3.0."
POLICY_DIGEST = digest_canonical({
    "version": "hermes-govuk-text-ogl-v1", "reviewed_terms": REVIEWED_TEXT,
    "use": "PUBLICATION_EVIDENCE", "scope": "GOVUK_PUBLISHED_TEXT_ONLY",
    "required_attribution": ATTRIBUTION,
    "exclusions": ["personal_data", "third_party_rights", "logos_and_insignia",
                   "unpublished_information", "identity_documents", "endorsement"],
    "public_exposure": False,
})


@dataclass(frozen=True, slots=True)
class GovUkLicenceEvidence:
    admission_ids: tuple[ObjectAdmissionId, ObjectAdmissionId]
    raw_digests: tuple[str, str]
    observed_at: str
    policy_digest: str

    def for_source(self, *, source_id: str, definition_url: str) -> PublicationRightsAssessment:
        permitted = (
            self.policy_digest == POLICY_DIGEST
            and source_id in {"UK-01", "UK-02", "UK-03", "UK-05"}
            and urlsplit(definition_url).scheme == "https"
            and urlsplit(definition_url).netloc == "www.gov.uk"
        )
        return PublicationRightsAssessment.create(
            decision="PERMITTED" if permitted else "HOLD",
            permitted_use="PUBLICATION_EVIDENCE", policy_digest=POLICY_DIGEST,
            evidence_digest=digest_canonical({
                "source_id": source_id, "definition_url": definition_url,
                "admission_ids": [str(item) for item in self.admission_ids],
                "raw_digests": self.raw_digests, "observed_at": self.observed_at,
                "reviewed_text": REVIEWED_TEXT, "policy_digest": POLICY_DIGEST,
            }),
        )


def licence_text_digest(raw: bytes) -> str:
    tree = html.fromstring(raw)
    roots = tree.xpath("//main") or [tree]
    text = " ".join(" ".join(root.text_content().split()) for root in roots)
    return digest_bytes(text.encode("utf-8"))


def retain_current_govuk_licence(
    *, objects: GovernedObjects, proof: AuthenticationProof,
    dispatch_fence: Callable[[], None],
    clock: Callable[[], datetime] = lambda: datetime.now(tz=UTC),
) -> GovUkLicenceEvidence:
    """Fetch once at runtime qualification, retain exact raw terms in governed CAS.

    This is acquisition permission for the reviewed text use, not blanket
    publication approval. The post-acquisition assessor must apply exclusions;
    the final private article/feed card must carry attribution and source links.
    """
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}), _NoRedirect(),
        urllib.request.HTTPSHandler(context=ssl.create_default_context()),
    )
    observations = []
    for url in (REUSE_URL, LICENCE_URL):
        dispatch_fence()
        request = urllib.request.Request(url, method="GET", headers={
            "User-Agent": "Newsroom-Hermes-Rights-Review/1.0", "Accept-Encoding": "identity",
        })
        try:
            with opener.open(request, timeout=TIMEOUT_SECONDS) as response:
                raw = response.read(MAX_BODY_BYTES + 1)
                if response.status != 200 or response.geturl() != url:
                    raise ValueError("licence response identity differs")
            if not raw or len(raw) > MAX_BODY_BYTES:
                raise ValueError("licence response length differs")
            if licence_text_digest(raw) != REVIEWED_TEXT[url]:
                raise ValueError("substantive licence terms changed")
        except (urllib.error.URLError, TimeoutError, OSError, ValueError):
            raise NativeEvidenceHold("GOVUK_LICENCE_REVIEW_HOLD", "UK-GOVUK") from None
        observations.append(raw)
    # Check every response before the first authority write; no partial licence.
    dispatch_fence()
    admissions = tuple(
        objects.admit(
            ObjectAdmissionRequest("evidence.source", f"govuk-licence:{digest_bytes(raw)}"),
            raw, proof=proof,
        ).admission for raw in observations
    )
    return GovUkLicenceEvidence(
        tuple(item.admission_id for item in admissions),
        tuple(item.blob.blob_digest for item in admissions),
        clock().astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        POLICY_DIGEST,
    )
