import io
import json
from dataclasses import replace
from datetime import UTC, datetime
from email.message import Message

import pytest

from newsroom.control_plane.govuk_evidence import (
    GovUkEvidenceAcquisition, MAX_BODY_BYTES, POLICY_DIGEST, _api_url,
)
from newsroom.control_plane.native_evidence import EvidenceAcquisitionRequest, NativeEvidenceHold
from newsroom.sources import SourceDefinitionVersionId
from newsroom.tests.check_3c_authority_helpers import proof
from newsroom.tests.discovery_3d_authority_helpers import open_discovery_system
from newsroom.tests.test_graphiti_operational_readiness import _unit
from newsroom.tests.test_native_discovery import _seed, NOW


class Response(io.BytesIO):
    status = 200

    def __init__(self, raw, url):
        super().__init__(raw)
        self.url = url
        self.headers = Message()
        self.headers["Content-Type"] = "application/json; charset=utf-8"

    def geturl(self):
        return self.url


def _document(path):
    return {
        "base_path": path, "locale": "en", "document_type": "news_story",
        "title": "Official update", "details": {"body": "<p>Exact independent source text.</p>"},
        "first_published_at": "2026-09-01T10:00:00Z",
        "public_updated_at": "2026-09-02T10:00:00Z",
        "links": {"organisations": [{"title": "Home Office"}]},
    }


def _request(system, unit):
    version = system.sources.version_details(
        SourceDefinitionVersionId.parse(unit.authority.definition_version_id), proof=proof(),
    )
    return EvidenceAcquisitionRequest(
        unit.source_id, unit.authority.definition_id, unit.authority.definition_version_id,
        version.canonical_digest, unit.authority.revision_id,
        "https://www.gov.uk/government/news/official-update", POLICY_DIGEST,
    )


@pytest.mark.parametrize("url", [
    "http://www.gov.uk/government/news/x", "https://www.gov.uk.evil.test/x",
    "https://www.gov.uk@127.0.0.1/x", "https://www.gov.uk:443/x",
    "https://www.gov.uk/x?url=http://127.0.0.1", "https://www.gov.uk/x#other",
    "https://www.gov.uk/api/content/x", "https://www.gov.uk/%2e%2e/x",
    "https://www.gov.uk/%2f%2fother/x", "https://www.gov.uk/x%0d%0ay",
])
def test_govuk_route_rejects_ambiguous_or_external_urls(url):
    with pytest.raises(ValueError):
        _api_url(url)


def test_exact_native_source_fetches_bounded_independent_content(tmp_path, monkeypatch):
    with open_discovery_system(tmp_path / "authority.sqlite3", clock=lambda: NOW) as system:
        unit = replace(_unit(), source_definition_url="https://www.gov.uk/government/organisations/home-office.atom")
        _seed(system, unit)
        request = _request(system, unit)
        calls = []
        raw = json.dumps(_document("/government/news/official-update")).encode()
        class Opener:
            def open(self, http, timeout):
                calls.append((http.full_url, timeout, http.get_method()))
                return Response(raw, http.full_url)
        monkeypatch.setattr("urllib.request.build_opener", lambda *args: Opener())
        fenced = []
        acquire = GovUkEvidenceAcquisition(
            sources=system.sources, proof=proof(), dispatch_fence=fenced.append,
            clock=lambda: datetime(2026, 9, 2, 12, 2, tzinfo=UTC),
        )
        result = acquire(request)
        assert fenced == [request]
        assert calls == [("https://www.gov.uk/api/content/government/news/official-update", 20, "GET")]
        assert result.body == b"Official update\n\nExact independent source text."
        assert result.request_digest == request.digest
        assert result.publisher == "Home Office"
        assert result.publication_time == "2026-09-01T10:00:00.000000Z"
        assert result.source_updated_time == "2026-09-02T10:00:00.000000Z"
        assert result.retrieval_time == "2026-09-02T12:02:00.000000Z"
        assert result.outcome == "COMPLETE"
        assert result.body != unit.body.encode()
        assert result.receipt_digest != result.transport_evidence_digest
        with pytest.raises(NativeEvidenceHold, match="POLICY_MISMATCH"):
            acquire(replace(request, transport_policy_digest="sha256:" + "0" * 64))
        assert len(calls) == 1


@pytest.mark.parametrize("failure", ["too_large", "wrong_path", "missing_date", "future_date", "no_body", "redirect", "duplicate_json"])
def test_incomplete_source_response_is_never_complete(tmp_path, monkeypatch, failure):
    with open_discovery_system(tmp_path / "authority.sqlite3", clock=lambda: NOW) as system:
        unit = replace(_unit(), source_definition_url="https://www.gov.uk/government/organisations/home-office.atom")
        _seed(system, unit)
        request = _request(system, unit)
        document = _document("/government/news/official-update")
        if failure == "wrong_path": document["base_path"] = "/other"
        if failure == "missing_date": document.pop("first_published_at")
        if failure == "future_date": document["public_updated_at"] = "2027-01-01T00:00:00Z"
        if failure == "no_body": document["details"] = {}
        raw = json.dumps(document).encode()
        if failure == "too_large": raw = b"x" * (MAX_BODY_BYTES + 1)
        if failure == "duplicate_json": raw = b'{"base_path":"a","base_path":"b"}'
        class Opener:
            def open(self, http, timeout):
                return Response(raw, "https://other.test" if failure == "redirect" else http.full_url)
        monkeypatch.setattr("urllib.request.build_opener", lambda *args: Opener())
        acquire = GovUkEvidenceAcquisition(
            sources=system.sources, proof=proof(), dispatch_fence=lambda _: None,
            clock=lambda: datetime(2026, 9, 2, 12, 2, tzinfo=UTC),
        )
        with pytest.raises(NativeEvidenceHold):
            acquire(request)


def test_explicit_exclusions_are_signals_not_invented_semantic_passes():
    from newsroom.control_plane.govuk_evidence import _exclusion_signals
    assert _exclusion_signals({"details": {}}, "Government published an update.") == ()
    assert _exclusion_signals({"details": {"copyright_notice": "Third-party copyright"}}, "Update") == ("THIRD_PARTY_RIGHTS",)
    assert _exclusion_signals({"details": {}}, "This material is not covered by the Open Government Licence.") == ("NON_OGL_CONTENT",)
    assert _exclusion_signals({"details": {"personal_information": True}}, "Details") == ("EXCLUDED_PERSONAL_OR_IDENTITY_CONTENT",)


def test_acquisition_facts_are_in_the_exact_retained_receipt():
    from newsroom.authority.canonical import digest_bytes
    from newsroom.control_plane.native_evidence import AcquiredEvidence, NativeEvidenceError
    result = AcquiredEvidence.create(
        request_digest=digest_bytes(b"request"), outcome="COMPLETE",
        canonical_url="https://www.gov.uk/government/news/update", body=b"Update",
        body_digest=digest_bytes(b"Update"), publisher="Home Office",
        responsible_body="Home Office", source_type="OFFICIAL_PRIMARY",
        publication_time="2026-09-01T10:00:00Z", source_updated_time="2026-09-01T10:00:00Z",
        retrieval_time="2026-09-01T11:00:00Z", geography="UK", language="en-GB",
        transport_evidence_digest=digest_bytes(b"transport"),
        currentness_basis="AUTHORITATIVE_CURRENT_CONTENT_ENDPOINT",
        rights_eligibility_digest=digest_bytes(b"reviewed terms and exact source"),
        licence_attribution="Observed attribution", exclusion_signals=(), text_only=True,
    )
    for changes in ({"currentness_basis": ""}, {"rights_eligibility_digest": ""},
                    {"licence_attribution": ""}, {"text_only": False},
                    {"exclusion_signals": ("THIRD_PARTY_RIGHTS",)}):
        with pytest.raises(NativeEvidenceError, match="receipt differs"):
            replace(result, **changes)


def test_native_evidence_origin_comparison_rejects_embedded_credentials():
    from newsroom.control_plane.native_evidence import _same_https_origin
    assert _same_https_origin("https://www.gov.uk/a", "https://www.gov.uk/feed")
    assert not _same_https_origin("https://user:password@www.gov.uk/a", "https://www.gov.uk/feed")
    assert not _same_https_origin("https://www.gov.uk/a", "https://user:password@www.gov.uk/feed")


def test_maintained_guide_includes_every_part_and_rejects_partial_parts():
    from newsroom.control_plane.govuk_evidence import _document_text
    value = {"document_type": "guide", "details": {"parts": [
        {"title": "Overview", "slug": "overview", "body": "<p>First part.</p>"},
        {"title": "Changed deadline", "slug": "deadline", "body": "<p>Second part.</p>"},
    ]}}
    assert _document_text(value) == "Overview\nFirst part.\n\nChanged deadline\nSecond part."
    value["details"]["parts"][1]["body"] = ""
    with pytest.raises(ValueError, match="absent"):
        _document_text(value)
    value["details"]["parts"][1]["body"] = "Text"
    value["details"]["parts"][1]["slug"] = "overview"
    with pytest.raises(ValueError, match="identity"):
        _document_text(value)
