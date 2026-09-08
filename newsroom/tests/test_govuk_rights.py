import io
from datetime import UTC, datetime

import pytest

from newsroom.control_plane import govuk_rights as rights
from newsroom.control_plane.native_evidence import NativeEvidenceHold
from newsroom.tests.test_native_runtime import _args
from newsroom.control_plane.native_runtime import open_native_runtime


def test_observed_licence_retained_once_and_unknown_scope_held(tmp_path, monkeypatch):
    # F1 fixture terms are pinned separately; never production licence evidence.
    responses = {
        rights.REUSE_URL: b"<main>Test GOV reuse policy</main>",
        rights.LICENCE_URL: b"<main>Test OGL terms</main>",
    }
    monkeypatch.setattr(rights, "REVIEWED_TEXT", {
        url: rights.licence_text_digest(raw) for url, raw in responses.items()
    })
    calls = []
    class Response(io.BytesIO):
        status = 200
        def __init__(self, url):
            super().__init__(responses[url]); self.url = url
        def geturl(self): return self.url
    class Opener:
        def open(self, request, timeout):
            calls.append((request.full_url, timeout))
            return Response(request.full_url)
    monkeypatch.setattr("urllib.request.build_opener", lambda *args: Opener())
    args = _args(tmp_path, monkeypatch)
    with open_native_runtime(**args) as runtime:
        fence_calls = []
        params = dict(objects=runtime.authority.objects, proof=runtime.proof,
                      dispatch_fence=lambda: fence_calls.append(True),
                      clock=lambda: datetime(2026, 9, 8, 10, tzinfo=UTC))
        first = rights.retain_current_govuk_licence(**params)
        second = rights.retain_current_govuk_licence(**params)
        assert first == second
        assert len(calls) == 4 and len(fence_calls) == 6
        assert len(first.admission_ids) == 2
        assert first.for_source(source_id="UK-01", definition_url="https://www.gov.uk/feed").decision == "PERMITTED"
        assert first.for_source(source_id="RAD-02", definition_url="https://www.gov.uk/feed").decision == "HOLD"
        assert first.for_source(source_id="UK-01", definition_url="https://www.gov.uk.evil.test/feed").decision == "HOLD"
        responses[rights.LICENCE_URL] = b"<main>Changed licence terms</main>"
        with pytest.raises(NativeEvidenceHold, match="LICENCE_REVIEW_HOLD"):
            rights.retain_current_govuk_licence(**params)
        assert len(fence_calls) == 8


def test_licence_substantive_text_changes_are_detected():
    first = rights.licence_text_digest(b"<main><p>Re-use is permitted.</p></main>")
    assert first == rights.licence_text_digest(b"<aside>menu</aside><main> Re-use  is permitted. </main>")
    assert first != rights.licence_text_digest(b"<main>Re-use is prohibited.</main>")
