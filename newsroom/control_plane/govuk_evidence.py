"""Independent GOV.UK Content API acquisition for the approved source route.

The runtime binds the exact current Source Registry and its dispatch fence.
Only the fixed public HTTPS Content API is reachable; there are no credentials,
redirects, browser execution, model calls or caller-selected network backends.
See https://content-api.publishing.service.gov.uk/getting-started.html.
"""

from __future__ import annotations

import json
import re
import ssl
import urllib.error
import urllib.request
from collections.abc import Callable
from datetime import UTC, datetime
from urllib.parse import unquote, urlsplit

from lxml import etree, html

from newsroom.authority import AuthenticationProof
from newsroom.authority.canonical import digest_bytes, digest_canonical
from newsroom.sources import SourceDefinitionVersionId, SourceRevisionId

from .native_evidence import (
    AcquiredEvidence, EvidenceAcquisitionRequest, NativeEvidenceHold,
)

VERSION = "hermes-govuk-evidence-v1"
MAX_BODY_BYTES = 1_048_576
TIMEOUT_SECONDS = 20
POLICY_DIGEST = digest_canonical({
    "version": VERSION, "origin": "https://www.gov.uk",
    "api_prefix": "/api/content", "method": "GET", "redirects": 0,
    "max_bytes": MAX_BODY_BYTES, "timeout_seconds": TIMEOUT_SECONDS,
    "credentials": False,
})


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _instant(value: object) -> datetime:
    if type(value) is not str:
        raise ValueError("source publication time is missing")
    instant = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if instant.tzinfo is None:
        raise ValueError("source publication time lacks offset")
    return instant.astimezone(UTC)


def _utc(value: datetime) -> str:
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _api_url(canonical_url: str) -> str:
    parsed = urlsplit(canonical_url)
    path = unquote(parsed.path)
    if (
        parsed.scheme != "https" or parsed.netloc != "www.gov.uk"
        or parsed.query or parsed.fragment or not path.startswith("/")
        or path.startswith("//") or "\\" in path
        or any(part in {".", ".."} for part in path.split("/"))
        or any(ord(character) < 32 for character in canonical_url + path)
        or parsed.path.startswith("/api/")
    ):
        raise ValueError("canonical source URL is outside the fixed GOV.UK route")
    return "https://www.gov.uk/api/content" + parsed.path


class GovUkEvidenceAcquisition:
    """A concrete bounded GET, with current rights/stop rechecked before I/O."""

    def __init__(
        self, *, sources, proof: AuthenticationProof,
        dispatch_fence: Callable[[EvidenceAcquisitionRequest], None],
        clock: Callable[[], datetime] = lambda: datetime.now(tz=UTC),
        licence_evidence=None,
    ) -> None:
        self._sources = sources
        self._proof = proof
        self._fence = dispatch_fence
        self._clock = clock
        self._licence = licence_evidence

    def __call__(self, request: EvidenceAcquisitionRequest) -> AcquiredEvidence:
        def hold(reason: str):
            return NativeEvidenceHold(reason, request.source_id)

        if type(request) is not EvidenceAcquisitionRequest:
            raise TypeError("exact independent acquisition request required")
        if request.transport_policy_digest != POLICY_DIGEST:
            raise hold("TRANSPORT_POLICY_MISMATCH")
        try:
            url = _api_url(request.canonical_url)
            version = self._sources.version_details(
                SourceDefinitionVersionId.parse(request.source_definition_version_id),
                proof=self._proof,
            )
            revision = self._sources.revision(
                SourceRevisionId.parse(request.source_revision_id), proof=self._proof,
            )
            if (
                str(version.request.definition_id) != request.source_definition_id
                or version.canonical_digest != request.source_definition_version_digest
                or revision.request.definition_version_id != version.version_id
                or urlsplit(version.request.locator).netloc != "www.gov.uk"
            ):
                raise ValueError("exact source binding differs")
        except (ValueError, LookupError):
            raise hold("GOVUK_SOURCE_BINDING_HOLD") from None
        # The caller supplies the existing signed-stop/current-rights fence,
        # not a per-story human approval. No SQLite transaction spans this I/O.
        self._fence(request)
        opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}), _NoRedirect(),
            urllib.request.HTTPSHandler(context=ssl.create_default_context()),
        )
        http_request = urllib.request.Request(url, method="GET", headers={
            "User-Agent": "Newsroom-Hermes/1.0", "Accept": "application/json",
            "Accept-Encoding": "identity",
        })
        try:
            with opener.open(http_request, timeout=TIMEOUT_SECONDS) as response:
                status = response.status
                content_type = response.headers.get_content_type()
                response_url = response.geturl()
                raw = response.read(MAX_BODY_BYTES + 1)
        except (urllib.error.URLError, TimeoutError, OSError):
            raise hold("GOVUK_ACQUISITION_UNAVAILABLE") from None
        retrieved = self._clock()
        if (
            status != 200 or response_url != url or content_type != "application/json"
            or not raw or len(raw) > MAX_BODY_BYTES
        ):
            raise hold("GOVUK_ACQUISITION_INCOMPLETE")
        try:
            value = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object)
            if (
                type(value) is not dict
                or value.get("base_path") != urlsplit(request.canonical_url).path
                or value.get("locale") != "en"
                or value.get("document_type") not in {
                    "news_story", "press_release", "guidance", "detailed_guide",
                    "html_publication", "notice", "policy_paper", "written_statement", "guide",
                }
                or value.get("withdrawn_notice")
            ):
                raise ValueError("source schema or currentness differs")
            publication = _instant(value.get("first_published_at"))
            updated = _instant(value.get("public_updated_at"))
            if publication > updated or updated > retrieved:
                raise ValueError("source temporal order differs")
            title = value["title"]
            if type(title) is not str or not title.strip():
                raise ValueError("source title is absent")
            body_text = _document_text(value)
            organisations = value["links"]["organisations"]
            names = tuple(sorted({item["title"] for item in organisations}))
            if not names or any(type(name) is not str or not name.strip() for name in names):
                raise ValueError("responsible publisher is absent")
            body = (title.strip() + "\n\n" + body_text).encode("utf-8")
        except (ValueError, TypeError, KeyError, UnicodeError, etree.ParserError):
            raise hold("GOVUK_EVIDENCE_METADATA_HOLD") from None
        transport_digest = digest_canonical({
            "version": VERSION, "request_digest": request.digest,
            "url": url, "response_url": response_url, "http_status": status,
            "content_type": content_type, "response_digest": digest_bytes(raw),
            "extracted_body_digest": digest_bytes(body),
            "public_updated_at": _utc(updated), "retrieved_at": _utc(retrieved),
        })
        # These are observed acquisition facts, not six invented semantic PASS
        # decisions. Editorial claim checks still decide what may be rewritten.
        # Image/logo bytes are never acquired or retained by this text route.
        signals = _exclusion_signals(value, body_text)
        rights_digest = ""
        attribution = ""
        if self._licence is not None:
            from .govuk_rights import ATTRIBUTION, POLICY_DIGEST as RIGHTS_POLICY

            rights = self._licence.for_source(
                source_id=request.source_id, definition_url=version.request.locator,
            )
            if rights.decision == "PERMITTED" and rights.policy_digest == RIGHTS_POLICY:
                rights_digest = digest_canonical({
                    "rights_receipt": rights.record_id,
                    "body_digest": digest_bytes(body), "transport": transport_digest,
                    "exclusion_signals": signals, "text_only": True,
                })
                attribution = ATTRIBUTION
        return AcquiredEvidence.create(
            request_digest=request.digest, outcome="COMPLETE",
            canonical_url=request.canonical_url, body=body, body_digest=digest_bytes(body),
            publisher="; ".join(names), responsible_body="; ".join(names),
            source_type="OFFICIAL_PRIMARY", publication_time=_utc(publication), source_updated_time=_utc(updated),
            retrieval_time=_utc(retrieved), geography="UK", language="en-GB",
            transport_evidence_digest=transport_digest,
            currentness_basis="AUTHORITATIVE_CURRENT_CONTENT_ENDPOINT",
            rights_eligibility_digest=rights_digest,
            licence_attribution=attribution,
            exclusion_signals=signals, text_only=True,
        )


def _unique_object(pairs):
    value = dict(pairs)
    if len(value) != len(pairs):
        raise ValueError("source JSON has duplicate fields")
    return value


def _exclusion_signals(value: dict, body_text: str) -> tuple[str, ...]:
    """Retain explicit contrary rights signals; absence is not a legal finding."""
    details = value["details"]
    notices = " ".join(str(details.get(key, "")) for key in (
        "copyright_notice", "copyright", "licence", "license",
    ))
    text = (notices + " " + body_text).casefold()
    signals = set()
    if re.search(r"third[- ]party copyright|all rights reserved|permission.{0,30}copyright holder", text):
        signals.add("THIRD_PARTY_RIGHTS")
    if re.search(r"not (?:covered|available|licensed).{0,45}open government licen[cs]e", text):
        signals.add("NON_OGL_CONTENT")
    if details.get("personal_information") or details.get("identity_document"):
        signals.add("EXCLUDED_PERSONAL_OR_IDENTITY_CONTENT")
    return tuple(sorted(signals))


def _document_text(value: dict) -> str:
    """Read every supplied guide part, not just the first-page summary."""
    if value.get("document_type") == "guide":
        parts = value["details"]["parts"]
        if type(parts) is not list or not parts:
            raise ValueError("source guide parts are absent")
        slugs = set()
        sections = []
        for part in parts:
            if type(part) is not dict:
                raise ValueError("source guide part differs")
            slug, title = part.get("slug"), part.get("title")
            if (type(slug) is not str or not slug or slug in slugs
                    or type(title) is not str or not title.strip()):
                raise ValueError("source guide part identity differs")
            slugs.add(slug)
            sections.append(title.strip() + "\n" + _html_text(part.get("body")))
        return "\n\n".join(sections)
    return _html_text(value["details"].get("body"))


def _html_text(fragment: object) -> str:
    if type(fragment) is not str or not fragment.strip():
        raise ValueError("source document body is absent")
    document = html.fragment_fromstring(fragment, create_parent="div")
    if document.xpath(".//script | .//iframe | .//object"):
        raise ValueError("source body requires non-text resources")
    text = " ".join(document.text_content().split())
    if not text:
        raise ValueError("source content is empty")
    return text
