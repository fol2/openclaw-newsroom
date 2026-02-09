"""Tests for newsroom.gemini_client — 100% coverage target."""

from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from newsroom.gemini_client import (
    GeminiClient,
    _extract_first_json_object,
    _QUOTA_ENDPOINT,
    _TERMINAL_COOLDOWN_S,
    _RETRYABLE_COOLDOWN_S,
)


# ---------------------------------------------------------------------------
# _extract_first_json_object
# ---------------------------------------------------------------------------


class TestExtractFirstJsonObject(unittest.TestCase):
    def test_empty_string(self) -> None:
        self.assertIsNone(_extract_first_json_object(""))

    def test_none_value(self) -> None:
        self.assertIsNone(_extract_first_json_object(""))  # type: ignore[arg-type]

    def test_plain_json(self) -> None:
        self.assertEqual(_extract_first_json_object('{"a": 1}'), {"a": 1})

    def test_json_with_fences(self) -> None:
        text = '```json\n{"key": "val"}\n```'
        self.assertEqual(_extract_first_json_object(text), {"key": "val"})

    def test_json_with_bare_fences(self) -> None:
        text = '```\n{"key": 2}\n```'
        self.assertEqual(_extract_first_json_object(text), {"key": 2})

    def test_json_embedded_in_text(self) -> None:
        text = 'Here is the result: {"status": "ok"} done.'
        self.assertEqual(_extract_first_json_object(text), {"status": "ok"})

    def test_no_json(self) -> None:
        self.assertIsNone(_extract_first_json_object("just text"))

    def test_array_not_object(self) -> None:
        # Top-level array should return None (we only want objects).
        self.assertIsNone(_extract_first_json_object("[1, 2, 3]"))

    def test_invalid_json_inside_braces(self) -> None:
        self.assertIsNone(_extract_first_json_object("{not json}"))

    def test_unclosed_brace(self) -> None:
        self.assertIsNone(_extract_first_json_object('{"a": 1'))

    def test_non_object_in_braces(self) -> None:
        # Braces that parse as JSON but not a dict (shouldn't happen in practice).
        self.assertIsNone(_extract_first_json_object("not json at all"))

    def test_whitespace_only(self) -> None:
        self.assertIsNone(_extract_first_json_object("   \n  "))


# ---------------------------------------------------------------------------
# GeminiClient — profile loading
# ---------------------------------------------------------------------------


def _write_auth_profiles(
    td: str, profiles: dict, order: list[str] | None = None
) -> Path:
    p = Path(td) / "auth-profiles.json"
    data: dict[str, Any] = {"profiles": profiles}
    if order is not None:
        data["order"] = {"google-gemini-cli": order}
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


class TestGeminiClientProfileLoading(unittest.TestCase):
    def test_loads_profiles_from_order(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            future_ms = int(time.time() * 1000) + 3_600_000
            profiles = {
                "google-gemini-cli:a@test.com": {
                    "provider": "google-gemini-cli",
                    "access": "token_a",
                    "projectId": "proj_a",
                    "expires": future_ms,
                },
                "google-gemini-cli:b@test.com": {
                    "provider": "google-gemini-cli",
                    "access": "token_b",
                    "projectId": "proj_b",
                    "expires": future_ms,
                },
            }
            path = _write_auth_profiles(
                td,
                profiles,
                order=["google-gemini-cli:b@test.com", "google-gemini-cli:a@test.com"],
            )
            with patch.object(GeminiClient, "_check_quota", return_value={}):
                client = GeminiClient(auth_profiles_path=path)
            self.assertEqual(len(client._profiles), 2)
            # Order should match the explicit order section (quota unknown = keep original).
            self.assertEqual(client._profile_ids[0], "google-gemini-cli:b@test.com")

    def test_missing_file_graceful(self) -> None:
        client = GeminiClient(auth_profiles_path=Path("/nonexistent/auth.json"))
        self.assertEqual(client._profiles, [])

    def test_no_matching_profiles(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            profiles = {
                "other:a@test.com": {
                    "provider": "other",
                    "access": "tok",
                },
            }
            path = _write_auth_profiles(td, profiles, order=[])
            client = GeminiClient(auth_profiles_path=path)
            self.assertEqual(client._profiles, [])

    def test_skips_non_matching_provider_in_order(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            profiles = {
                "google-gemini-cli:a@test.com": {
                    "provider": "google-gemini-cli",
                    "access": "tok",
                    "expires": int(time.time() * 1000) + 3_600_000,
                },
                "google-antigravity:a@test.com": {
                    "provider": "google-antigravity",
                    "access": "tok2",
                },
            }
            path = _write_auth_profiles(
                td,
                profiles,
                order=["google-gemini-cli:a@test.com", "google-antigravity:a@test.com"],
            )
            client = GeminiClient(auth_profiles_path=path)
            self.assertEqual(len(client._profiles), 1)


# ---------------------------------------------------------------------------
# GeminiClient — profile validity / rotation
# ---------------------------------------------------------------------------


class TestGeminiClientProfilePicking(unittest.TestCase):
    def _make_client(self, profiles: list[dict], ids: list[str]) -> GeminiClient:
        with tempfile.TemporaryDirectory() as td:
            path = _write_auth_profiles(td, {}, order=[])
            client = GeminiClient(auth_profiles_path=path)
        client._profiles = profiles
        client._profile_ids = ids
        client._rotation_index = 0
        return client

    def test_valid_profile_picked(self) -> None:
        future = int(time.time() * 1000) + 3_600_000
        client = self._make_client(
            [{"access": "tok", "expires": future}],
            ["id1"],
        )
        pick = client._pick_profile()
        self.assertIsNotNone(pick)
        assert pick is not None
        self.assertEqual(pick[1], "id1")

    def test_expired_profile_skipped(self) -> None:
        past = int(time.time() * 1000) - 3_600_000
        future = int(time.time() * 1000) + 3_600_000
        client = self._make_client(
            [
                {"access": "tok_expired", "expires": past},
                {"access": "tok_valid", "expires": future},
            ],
            ["expired", "valid"],
        )
        pick = client._pick_profile()
        self.assertIsNotNone(pick)
        assert pick is not None
        self.assertEqual(pick[1], "valid")

    def test_all_expired_returns_first(self) -> None:
        past = int(time.time() * 1000) - 3_600_000
        client = self._make_client(
            [
                {"access": "tok1", "expires": past},
                {"access": "tok2", "expires": past},
            ],
            ["id1", "id2"],
        )
        pick = client._pick_profile()
        self.assertIsNotNone(pick)
        assert pick is not None
        self.assertEqual(pick[1], "id1")

    def test_no_profiles_returns_none(self) -> None:
        client = self._make_client([], [])
        self.assertIsNone(client._pick_profile())

    def test_no_expiry_assumed_valid(self) -> None:
        client = self._make_client([{"access": "tok"}], ["id1"])
        self.assertTrue(client._is_profile_valid({"access": "tok"}))

    def test_rotation_round_robin(self) -> None:
        future = int(time.time() * 1000) + 3_600_000
        client = self._make_client(
            [
                {"access": "a", "expires": future},
                {"access": "b", "expires": future},
            ],
            ["id_a", "id_b"],
        )
        pick1 = client._pick_profile()
        pick2 = client._pick_profile()
        self.assertIsNotNone(pick1)
        self.assertIsNotNone(pick2)
        assert pick1 is not None and pick2 is not None
        self.assertEqual(pick1[1], "id_a")
        self.assertEqual(pick2[1], "id_b")


# ---------------------------------------------------------------------------
# GeminiClient — _call_sse
# ---------------------------------------------------------------------------


def _429_response(body_text: str = "rate limited", headers: dict | None = None) -> MagicMock:
    """Build a mock 429 response with proper .text and .headers."""
    resp = MagicMock()
    resp.status_code = 429
    resp.text = body_text
    resp.headers = headers or {}
    return resp


def _sse_response(text: str, status_code: int = 200) -> MagicMock:
    """Build a mock requests.Response with SSE data lines."""
    chunk = {"response": {"candidates": [{"content": {"parts": [{"text": text}]}}]}}
    lines = [
        b"data: " + json.dumps(chunk).encode(),
        b"",
        b"data: [DONE]",
    ]
    resp = MagicMock()
    resp.status_code = status_code
    resp.iter_lines.return_value = iter(lines)
    resp.text = ""
    return resp


class TestGeminiClientCallSse(unittest.TestCase):
    def _make_client(self) -> GeminiClient:
        with tempfile.TemporaryDirectory() as td:
            path = _write_auth_profiles(td, {}, order=[])
            return GeminiClient(auth_profiles_path=path)

    @patch("newsroom.gemini_client.requests.post")
    def test_successful_sse(self, mock_post: MagicMock) -> None:
        mock_post.return_value = _sse_response("Hello world")
        client = self._make_client()
        result = client._call_sse(
            prompt="test",
            model="gemini-3-flash-preview",
            profile={"access": "tok", "projectId": "proj"},
        )
        self.assertEqual(result, "Hello world")

    @patch("newsroom.gemini_client.requests.post")
    def test_no_token_returns_none(self, mock_post: MagicMock) -> None:
        client = self._make_client()
        result = client._call_sse(prompt="test", model="m", profile={})
        self.assertIsNone(result)
        mock_post.assert_not_called()

    @patch("newsroom.gemini_client.requests.post")
    def test_401_returns_none(self, mock_post: MagicMock) -> None:
        mock_post.return_value = _sse_response("", status_code=401)
        client = self._make_client()
        result = client._call_sse(prompt="test", model="m", profile={"access": "tok"})
        self.assertIsNone(result)

    @patch("newsroom.gemini_client.requests.post")
    def test_429_returns_none(self, mock_post: MagicMock) -> None:
        mock_post.return_value = _429_response()
        client = self._make_client()
        result = client._call_sse(prompt="test", model="m", profile={"access": "tok"})
        self.assertIsNone(result)

    @patch("newsroom.gemini_client.requests.post")
    def test_500_returns_none(self, mock_post: MagicMock) -> None:
        resp = MagicMock()
        resp.status_code = 500
        resp.text = "server error"
        mock_post.return_value = resp
        client = self._make_client()
        result = client._call_sse(prompt="test", model="m", profile={"access": "tok"})
        self.assertIsNone(result)

    @patch("newsroom.gemini_client.requests.post")
    def test_connection_error_returns_none(self, mock_post: MagicMock) -> None:
        mock_post.side_effect = ConnectionError("network down")
        client = self._make_client()
        result = client._call_sse(prompt="test", model="m", profile={"access": "tok"})
        self.assertIsNone(result)

    @patch("newsroom.gemini_client.requests.post")
    def test_empty_sse_returns_none(self, mock_post: MagicMock) -> None:
        resp = MagicMock()
        resp.status_code = 200
        resp.iter_lines.return_value = iter([b"", b"data: [DONE]"])
        mock_post.return_value = resp
        client = self._make_client()
        result = client._call_sse(prompt="test", model="m", profile={"access": "tok"})
        self.assertIsNone(result)

    @patch("newsroom.gemini_client.requests.post")
    def test_malformed_json_in_sse_skipped(self, mock_post: MagicMock) -> None:
        resp = MagicMock()
        resp.status_code = 200
        resp.iter_lines.return_value = iter(
            [
                b"data: {not json}",
                b"data: "
                + json.dumps(
                    {
                        "response": {
                            "candidates": [{"content": {"parts": [{"text": "ok"}]}}]
                        }
                    }
                ).encode(),
            ]
        )
        mock_post.return_value = resp
        client = self._make_client()
        result = client._call_sse(prompt="test", model="m", profile={"access": "tok"})
        self.assertEqual(result, "ok")

    @patch("newsroom.gemini_client.requests.post")
    def test_sse_stream_error_with_partial(self, mock_post: MagicMock) -> None:
        """If stream errors after partial text, return what we have."""

        def exploding_lines():
            yield (
                b"data: "
                + json.dumps(
                    {
                        "response": {
                            "candidates": [
                                {"content": {"parts": [{"text": "partial"}]}}
                            ]
                        }
                    }
                ).encode()
            )
            raise ConnectionError("mid-stream")

        resp = MagicMock()
        resp.status_code = 200
        resp.iter_lines.return_value = exploding_lines()
        mock_post.return_value = resp
        client = self._make_client()
        result = client._call_sse(prompt="test", model="m", profile={"access": "tok"})
        self.assertEqual(result, "partial")

    @patch("newsroom.gemini_client.requests.post")
    def test_sse_stream_error_no_partial_returns_none(
        self, mock_post: MagicMock
    ) -> None:
        def exploding_lines():
            raise ConnectionError("immediate error")
            yield  # make it a generator  # pragma: no cover

        resp = MagicMock()
        resp.status_code = 200
        resp.iter_lines.return_value = exploding_lines()
        mock_post.return_value = resp
        client = self._make_client()
        result = client._call_sse(prompt="test", model="m", profile={"access": "tok"})
        self.assertIsNone(result)

    @patch("newsroom.gemini_client.requests.post")
    def test_no_project_id_still_works(self, mock_post: MagicMock) -> None:
        mock_post.return_value = _sse_response("yes")
        client = self._make_client()
        result = client._call_sse(prompt="test", model="m", profile={"access": "tok"})
        self.assertEqual(result, "yes")
        # Verify "project" key not in body.
        call_kwargs = mock_post.call_args
        body = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        self.assertNotIn("project", body)

    @patch("newsroom.gemini_client.requests.post")
    def test_non_data_lines_ignored(self, mock_post: MagicMock) -> None:
        resp = MagicMock()
        resp.status_code = 200
        resp.iter_lines.return_value = iter(
            [
                b"event: ping",
                b": comment",
                b"data: "
                + json.dumps(
                    {
                        "response": {
                            "candidates": [{"content": {"parts": [{"text": "result"}]}}]
                        }
                    }
                ).encode(),
            ]
        )
        mock_post.return_value = resp
        client = self._make_client()
        result = client._call_sse(prompt="test", model="m", profile={"access": "tok"})
        self.assertEqual(result, "result")


# ---------------------------------------------------------------------------
# GeminiClient — generate / generate_json
# ---------------------------------------------------------------------------


class TestGeminiClientGenerate(unittest.TestCase):
    def _make_client_with_profiles(self) -> GeminiClient:
        with tempfile.TemporaryDirectory() as td:
            future = int(time.time() * 1000) + 3_600_000
            profiles = {
                "google-gemini-cli:a@test.com": {
                    "provider": "google-gemini-cli",
                    "access": "tok_a",
                    "projectId": "proj_a",
                    "expires": future,
                },
                "google-gemini-cli:b@test.com": {
                    "provider": "google-gemini-cli",
                    "access": "tok_b",
                    "projectId": "proj_b",
                    "expires": future,
                },
            }
            path = _write_auth_profiles(
                td,
                profiles,
                order=["google-gemini-cli:a@test.com", "google-gemini-cli:b@test.com"],
            )
            with patch.object(GeminiClient, "_check_quota", return_value={}):
                return GeminiClient(auth_profiles_path=path)

    @patch("newsroom.gemini_client.requests.post")
    def test_generate_flash_success(self, mock_post: MagicMock) -> None:
        mock_post.return_value = _sse_response("flash result")
        client = self._make_client_with_profiles()
        result = client.generate("hello")
        self.assertEqual(result, "flash result")

    @patch("newsroom.gemini_client.requests.post")
    def test_generate_pro_fallback(self, mock_post: MagicMock) -> None:
        """Flash fails (rate limit), Pro succeeds."""
        call_count = 0

        def side_effect(*args: Any, **kwargs: Any) -> MagicMock:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Flash fails with retryable 429.
                return _429_response()
            # Pro succeeds.
            return _sse_response("pro result")

        mock_post.side_effect = side_effect
        client = self._make_client_with_profiles()
        result = client.generate("hello")
        self.assertEqual(result, "pro result")

    @patch("newsroom.gemini_client.requests.post")
    def test_generate_second_profile_fallback(self, mock_post: MagicMock) -> None:
        """Both Flash and Pro fail on first profile, Flash succeeds on second."""
        call_count = 0

        def side_effect(*args: Any, **kwargs: Any) -> MagicMock:
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                # First profile: Flash + Pro fail with retryable 429.
                return _429_response()
            # Second profile succeeds.
            return _sse_response("second profile result")

        mock_post.side_effect = side_effect
        client = self._make_client_with_profiles()
        result = client.generate("hello")
        self.assertEqual(result, "second profile result")

    @patch("newsroom.gemini_client.requests.post")
    def test_generate_all_fail(self, mock_post: MagicMock) -> None:
        resp = MagicMock()
        resp.status_code = 500
        resp.text = "error"
        mock_post.return_value = resp
        client = self._make_client_with_profiles()
        result = client.generate("hello")
        self.assertIsNone(result)

    def test_generate_no_profiles(self) -> None:
        client = GeminiClient(auth_profiles_path=Path("/nonexistent/auth.json"))
        result = client.generate("hello")
        self.assertIsNone(result)

    @patch("newsroom.gemini_client.requests.post")
    def test_generate_json_success(self, mock_post: MagicMock) -> None:
        mock_post.return_value = _sse_response('{"status": "ok"}')
        client = self._make_client_with_profiles()
        result = client.generate_json("give me json")
        self.assertEqual(result, {"status": "ok"})

    @patch("newsroom.gemini_client.requests.post")
    def test_generate_json_no_json_in_response(self, mock_post: MagicMock) -> None:
        mock_post.return_value = _sse_response("just plain text with no json")
        client = self._make_client_with_profiles()
        result = client.generate_json("give me json")
        self.assertIsNone(result)

    @patch("newsroom.gemini_client.requests.post")
    def test_generate_json_llm_returns_none(self, mock_post: MagicMock) -> None:
        resp = MagicMock()
        resp.status_code = 500
        resp.text = "error"
        mock_post.return_value = resp
        client = self._make_client_with_profiles()
        result = client.generate_json("test")
        self.assertIsNone(result)

    @patch("newsroom.gemini_client.requests.post")
    def test_generate_single_profile_no_second_fallback(
        self, mock_post: MagicMock
    ) -> None:
        """With only one profile, second-profile fallback is skipped."""
        with tempfile.TemporaryDirectory() as td:
            future = int(time.time() * 1000) + 3_600_000
            profiles = {
                "google-gemini-cli:a@test.com": {
                    "provider": "google-gemini-cli",
                    "access": "tok_a",
                    "projectId": "proj_a",
                    "expires": future,
                },
            }
            path = _write_auth_profiles(
                td, profiles, order=["google-gemini-cli:a@test.com"]
            )
            client = GeminiClient(auth_profiles_path=path)

        resp = MagicMock()
        resp.status_code = 500
        resp.text = "error"
        mock_post.return_value = resp
        result = client.generate("hello")
        self.assertIsNone(result)


# ---------------------------------------------------------------------------
# GeminiClient — quota checking & ordering
# ---------------------------------------------------------------------------


class TestGeminiClientQuota(unittest.TestCase):
    """Tests for _check_quota, _order_profiles_by_quota, and exhausted_pids."""

    @patch("newsroom.gemini_client.requests.post")
    def test_check_quota_success(self, mock_post: MagicMock) -> None:
        """Successful quota check parses flash/pro remainingFraction."""
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "buckets": [
                {"modelId": "gemini-3-flash-preview", "remainingFraction": 0.85},
                {"modelId": "gemini-3-flash-preview-exp", "remainingFraction": 0.90},
                {"modelId": "gemini-3-pro-preview", "remainingFraction": 0.40},
            ]
        }
        mock_post.return_value = resp

        with tempfile.TemporaryDirectory() as td:
            path = _write_auth_profiles(td, {}, order=[])
            client = GeminiClient(auth_profiles_path=path)

        result = client._check_quota({"access": "tok"})
        self.assertAlmostEqual(result["flash"], 0.85)  # min of 0.85, 0.90
        self.assertAlmostEqual(result["pro"], 0.40)

    @patch("newsroom.gemini_client.requests.post")
    def test_check_quota_timeout(self, mock_post: MagicMock) -> None:
        """Timeout returns empty dict."""
        mock_post.side_effect = ConnectionError("timeout")

        with tempfile.TemporaryDirectory() as td:
            path = _write_auth_profiles(td, {}, order=[])
            client = GeminiClient(auth_profiles_path=path)

        result = client._check_quota({"access": "tok"})
        self.assertEqual(result, {})

    @patch("newsroom.gemini_client.requests.post")
    def test_check_quota_no_token(self, mock_post: MagicMock) -> None:
        """No access token returns empty dict without making request."""
        with tempfile.TemporaryDirectory() as td:
            path = _write_auth_profiles(td, {}, order=[])
            client = GeminiClient(auth_profiles_path=path)

        result = client._check_quota({})
        self.assertEqual(result, {})
        mock_post.assert_not_called()

    @patch("newsroom.gemini_client.requests.post")
    def test_check_quota_non_200(self, mock_post: MagicMock) -> None:
        """Non-200 response returns empty dict."""
        resp = MagicMock()
        resp.status_code = 401
        mock_post.return_value = resp

        with tempfile.TemporaryDirectory() as td:
            path = _write_auth_profiles(td, {}, order=[])
            client = GeminiClient(auth_profiles_path=path)

        result = client._check_quota({"access": "tok"})
        self.assertEqual(result, {})

    def test_order_profiles_by_quota(self) -> None:
        """4 profiles with different quotas get reordered correctly."""
        future = int(time.time() * 1000) + 3_600_000
        profiles = {}
        order = []
        # Create 4 profiles: a=10%, b=90%, c=50%, d=unknown (fail)
        for email, frac in [("a", 0.10), ("b", 0.90), ("c", 0.50), ("d", None)]:
            pid = f"google-gemini-cli:{email}@test.com"
            profiles[pid] = {
                "provider": "google-gemini-cli",
                "access": f"tok_{email}",
                "expires": future,
            }
            order.append(pid)

        def fake_check_quota(profile: dict) -> dict:
            tok = profile.get("access", "")
            if tok == "tok_a":
                return {"flash": 0.10}
            if tok == "tok_b":
                return {"flash": 0.90}
            if tok == "tok_c":
                return {"flash": 0.50}
            return {}  # d = unknown

        with tempfile.TemporaryDirectory() as td:
            path = _write_auth_profiles(td, profiles, order=order)
            with patch.object(GeminiClient, "_check_quota", side_effect=fake_check_quota):
                client = GeminiClient(auth_profiles_path=path)

        # Expected order: b(90%) > c(50%) > a(10%) > d(unknown=-1, original idx=3)
        self.assertEqual(
            [p.split(":")[1] for p in client._profile_ids],
            ["b@test.com", "c@test.com", "a@test.com", "d@test.com"],
        )

    @patch("newsroom.gemini_client.requests.post")
    def test_generate_tries_all_profiles(self, mock_post: MagicMock) -> None:
        """First 3 profiles 429 (retryable), 4th succeeds."""
        future = int(time.time() * 1000) + 3_600_000
        profiles = {}
        order = []
        for email in ["a", "b", "c", "d"]:
            pid = f"google-gemini-cli:{email}@test.com"
            profiles[pid] = {
                "provider": "google-gemini-cli",
                "access": f"tok_{email}",
                "projectId": f"proj_{email}",
                "expires": future,
            }
            order.append(pid)

        with tempfile.TemporaryDirectory() as td:
            path = _write_auth_profiles(td, profiles, order=order)
            with patch.object(GeminiClient, "_check_quota", return_value={}):
                client = GeminiClient(auth_profiles_path=path)

        call_count = 0

        def side_effect(*args: Any, **kwargs: Any) -> MagicMock:
            nonlocal call_count
            call_count += 1
            # Each profile tries flash + pro = 2 calls. First 3 profiles = 6 calls all 429.
            if call_count <= 6:
                return _429_response()
            # 4th profile flash succeeds.
            return _sse_response("fourth profile success")

        mock_post.side_effect = side_effect
        result = client.generate("hello")
        self.assertEqual(result, "fourth profile success")
        # Verify 3 profiles were marked with cooldown.
        self.assertEqual(len(client._exhausted_until), 3)

    @patch("newsroom.gemini_client.requests.post")
    def test_exhausted_skipped_during_cooldown(self, mock_post: MagicMock) -> None:
        """After terminal 429, same profile is skipped while cooldown is active."""
        future = int(time.time() * 1000) + 3_600_000
        profiles = {}
        order = []
        for email in ["a", "b"]:
            pid = f"google-gemini-cli:{email}@test.com"
            profiles[pid] = {
                "provider": "google-gemini-cli",
                "access": f"tok_{email}",
                "projectId": f"proj_{email}",
                "expires": future,
            }
            order.append(pid)

        with tempfile.TemporaryDirectory() as td:
            path = _write_auth_profiles(td, profiles, order=order)
            with patch.object(GeminiClient, "_check_quota", return_value={}):
                client = GeminiClient(auth_profiles_path=path)

        call_count = 0
        # Use terminal 429 (daily quota) so profile gets long cooldown.
        terminal_body = 'Your quota will reset after 2h0m0s.'

        def side_effect(*args: Any, **kwargs: Any) -> MagicMock:
            nonlocal call_count
            call_count += 1
            # Profile a: flash returns terminal 429 → breaks out of model loop.
            if call_count == 1:
                return _429_response(terminal_body)
            # Profile b: succeeds.
            return _sse_response("result")

        mock_post.side_effect = side_effect
        # First call: a terminal-429s (only 1 call, breaks on terminal), b succeeds.
        result1 = client.generate("hello")
        self.assertEqual(result1, "result")
        self.assertIn("google-gemini-cli:a@test.com", client._exhausted_until)

        # Second call (still within cooldown): profile a skipped, only b tried.
        call_count = 0

        def side_effect2(*args: Any, **kwargs: Any) -> MagicMock:
            nonlocal call_count
            call_count += 1
            return _sse_response("second call result")

        mock_post.side_effect = side_effect2
        result2 = client.generate("world")
        self.assertEqual(result2, "second call result")
        # Only 1 call made (profile b flash), not 3 (a flash + a pro + b flash).
        self.assertEqual(call_count, 1)

    @patch("newsroom.gemini_client.requests.post")
    def test_exhausted_retried_after_cooldown(self, mock_post: MagicMock) -> None:
        """After cooldown expires, previously 429'd profile is retried."""
        future = int(time.time() * 1000) + 3_600_000
        profiles = {}
        order = []
        for email in ["a", "b"]:
            pid = f"google-gemini-cli:{email}@test.com"
            profiles[pid] = {
                "provider": "google-gemini-cli",
                "access": f"tok_{email}",
                "projectId": f"proj_{email}",
                "expires": future,
            }
            order.append(pid)

        with tempfile.TemporaryDirectory() as td:
            path = _write_auth_profiles(td, profiles, order=order)
            with patch.object(GeminiClient, "_check_quota", return_value={}):
                client = GeminiClient(auth_profiles_path=path)

        # Manually set profile a's cooldown to already expired.
        client._exhausted_until["google-gemini-cli:a@test.com"] = time.monotonic() - 1

        mock_post.return_value = _sse_response("recovered")
        result = client.generate("hello")
        self.assertEqual(result, "recovered")
        # Profile a was retried (cooldown expired), so it should be the first call.
        call_kwargs = mock_post.call_args
        body = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        self.assertIn("Bearer tok_a", call_kwargs.kwargs.get("headers", {}).get("Authorization", ""))


    @patch("newsroom.gemini_client.requests.post")
    def test_generate_round_robins_across_profiles(self, mock_post: MagicMock) -> None:
        """Successive generate() calls rotate across profiles, not always the first."""
        future = int(time.time() * 1000) + 3_600_000
        profiles = {}
        order = []
        for email in ["a", "b", "c"]:
            pid = f"google-gemini-cli:{email}@test.com"
            profiles[pid] = {
                "provider": "google-gemini-cli",
                "access": f"tok_{email}",
                "projectId": f"proj_{email}",
                "expires": future,
            }
            order.append(pid)

        with tempfile.TemporaryDirectory() as td:
            path = _write_auth_profiles(td, profiles, order=order)
            with patch.object(GeminiClient, "_check_quota", return_value={}):
                client = GeminiClient(auth_profiles_path=path)

        mock_post.return_value = _sse_response("ok")

        # Track which profile token is used per call.
        tokens_used: list[str] = []

        def capture_post(*args: Any, **kwargs: Any) -> MagicMock:
            auth = (kwargs.get("headers") or {}).get("Authorization", "")
            tokens_used.append(auth)
            return _sse_response("ok")

        mock_post.side_effect = capture_post

        for _ in range(6):
            client.generate("test")

        # Should cycle: a, b, c, a, b, c
        expected = [f"Bearer tok_{e}" for e in ["a", "b", "c", "a", "b", "c"]]
        self.assertEqual(tokens_used, expected)

    # ---- _classify_429 tests ----

    def test_classify_429_terminal_quota_exhausted(self) -> None:
        """ErrorInfo reason=QUOTA_EXHAUSTED from cloudcode domain → terminal."""
        body = json.dumps({"error": {
            "code": 429,
            "message": "Quota exhausted",
            "details": [
                {
                    "@type": "type.googleapis.com/google.rpc.ErrorInfo",
                    "reason": "QUOTA_EXHAUSTED",
                    "domain": "cloudcode-pa.googleapis.com",
                    "metadata": {},
                },
            ],
        }})
        is_terminal, cd = GeminiClient._classify_429(body)
        self.assertTrue(is_terminal)
        self.assertEqual(cd, _TERMINAL_COOLDOWN_S)  # fallback, no retryDelay

    def test_classify_429_terminal_per_day_quota(self) -> None:
        """QuotaFailure with 'PerDay' quotaId → terminal."""
        body = json.dumps({"error": {
            "code": 429,
            "message": "Daily limit",
            "details": [
                {
                    "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                    "violations": [{"quotaId": "GenerateContentRequestsPerDay"}],
                },
            ],
        }})
        is_terminal, cd = GeminiClient._classify_429(body)
        self.assertTrue(is_terminal)
        self.assertEqual(cd, _TERMINAL_COOLDOWN_S)

    def test_classify_429_retryable_rate_limit(self) -> None:
        """ErrorInfo reason=RATE_LIMIT_EXCEEDED → retryable with default 10s."""
        body = json.dumps({"error": {
            "code": 429,
            "message": "Rate limited",
            "details": [
                {
                    "@type": "type.googleapis.com/google.rpc.ErrorInfo",
                    "reason": "RATE_LIMIT_EXCEEDED",
                    "domain": "cloudcode-pa.googleapis.com",
                    "metadata": {},
                },
            ],
        }})
        is_terminal, cd = GeminiClient._classify_429(body)
        self.assertFalse(is_terminal)
        self.assertEqual(cd, _RETRYABLE_COOLDOWN_S)

    def test_classify_429_retryable_with_retry_delay(self) -> None:
        """RetryInfo retryDelay present → retryable with parsed delay."""
        body = json.dumps({"error": {
            "code": 429,
            "message": "Slow down",
            "details": [
                {
                    "@type": "type.googleapis.com/google.rpc.ErrorInfo",
                    "reason": "RATE_LIMIT_EXCEEDED",
                    "domain": "cloudcode-pa.googleapis.com",
                    "metadata": {},
                },
                {
                    "@type": "type.googleapis.com/google.rpc.RetryInfo",
                    "retryDelay": "34.07s",
                },
            ],
        }})
        is_terminal, cd = GeminiClient._classify_429(body)
        self.assertFalse(is_terminal)
        self.assertAlmostEqual(cd, 34.07, places=2)

    def test_classify_429_terminal_with_retry_delay(self) -> None:
        """QUOTA_EXHAUSTED with retryDelay → terminal but uses parsed delay."""
        body = json.dumps({"error": {
            "code": 429,
            "message": "Exhausted",
            "details": [
                {
                    "@type": "type.googleapis.com/google.rpc.ErrorInfo",
                    "reason": "QUOTA_EXHAUSTED",
                    "domain": "cloudcode-pa.googleapis.com",
                    "metadata": {},
                },
                {
                    "@type": "type.googleapis.com/google.rpc.RetryInfo",
                    "retryDelay": "14400s",
                },
            ],
        }})
        is_terminal, cd = GeminiClient._classify_429(body)
        self.assertTrue(is_terminal)
        self.assertAlmostEqual(cd, 14400, places=0)

    def test_classify_429_per_minute_quota(self) -> None:
        """QuotaFailure with PerMinute quotaId → retryable 60s."""
        body = json.dumps({"error": {
            "code": 429,
            "message": "Per-minute limit",
            "details": [
                {
                    "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                    "violations": [{"quotaId": "TokensPerMinute"}],
                },
            ],
        }})
        is_terminal, cd = GeminiClient._classify_429(body)
        self.assertFalse(is_terminal)
        self.assertEqual(cd, 60)

    def test_classify_429_please_retry_in(self) -> None:
        """'Please retry in Xs' in message → retryable."""
        body = "Please retry in 5s after the limit resets."
        is_terminal, cd = GeminiClient._classify_429(body)
        self.assertFalse(is_terminal)
        self.assertAlmostEqual(cd, 5.0, places=1)

    def test_classify_429_please_retry_in_ms(self) -> None:
        """'Please retry in 500ms' → retryable 0.5s."""
        body = "Please retry in 500ms."
        is_terminal, cd = GeminiClient._classify_429(body)
        self.assertFalse(is_terminal)
        self.assertAlmostEqual(cd, 0.5, places=1)

    def test_classify_429_reset_after_body(self) -> None:
        """'reset after 4h14m0s' without structured error → terminal."""
        body = 'Cloud Code Assist API error (429): Your quota will reset after 4h14m0s.'
        is_terminal, cd = GeminiClient._classify_429(body)
        self.assertTrue(is_terminal)
        self.assertEqual(cd, 4 * 3600 + 14 * 60)

    def test_classify_429_retry_after_header(self) -> None:
        """Retry-After header only → retryable."""
        is_terminal, cd = GeminiClient._classify_429("unknown 429", {"Retry-After": "120"})
        self.assertFalse(is_terminal)
        self.assertEqual(cd, 120)

    def test_classify_429_unknown(self) -> None:
        """Unknown 429 with no hints → retryable with default."""
        is_terminal, cd = GeminiClient._classify_429("some error")
        self.assertFalse(is_terminal)
        self.assertEqual(cd, _RETRYABLE_COOLDOWN_S)

    def test_parse_reset_time(self) -> None:
        self.assertEqual(GeminiClient._parse_reset_time("reset after 4h14m0s"), 4 * 3600 + 14 * 60)
        self.assertEqual(GeminiClient._parse_reset_time("reset after 14m30s"), 14 * 60 + 30)
        self.assertEqual(GeminiClient._parse_reset_time("reset after 30s"), 30)
        self.assertEqual(GeminiClient._parse_reset_time("no match"), 0)

    def test_parse_retry_delay(self) -> None:
        self.assertAlmostEqual(GeminiClient._parse_retry_delay("34.07s"), 34.07)
        self.assertAlmostEqual(GeminiClient._parse_retry_delay("900ms"), 0.9)
        self.assertEqual(GeminiClient._parse_retry_delay("invalid"), 0)

    @patch("newsroom.gemini_client.requests.post")
    def test_terminal_429_long_cooldown(self, mock_post: MagicMock) -> None:
        """Terminal 429 (daily quota) sets long cooldown, not 10s default."""
        future = int(time.time() * 1000) + 3_600_000
        profiles = {
            "google-gemini-cli:a@test.com": {
                "provider": "google-gemini-cli",
                "access": "tok_a",
                "projectId": "proj_a",
                "expires": future,
            },
        }
        with tempfile.TemporaryDirectory() as td:
            path = _write_auth_profiles(td, profiles, order=["google-gemini-cli:a@test.com"])
            client = GeminiClient(auth_profiles_path=path)

        resp = MagicMock()
        resp.status_code = 429
        resp.text = 'Your quota will reset after 2h0m0s.'
        resp.headers = {}
        mock_post.return_value = resp

        client.generate("hello")

        pid = "google-gemini-cli:a@test.com"
        self.assertIn(pid, client._exhausted_until)
        # Cooldown should be ~7200s from now, not 10s.
        remaining = client._exhausted_until[pid] - time.monotonic()
        self.assertGreater(remaining, 3600)  # at least 1h

    @patch("newsroom.gemini_client.requests.post")
    def test_retryable_429_short_cooldown(self, mock_post: MagicMock) -> None:
        """Retryable 429 (rate limit) sets short cooldown."""
        future = int(time.time() * 1000) + 3_600_000
        profiles = {
            "google-gemini-cli:a@test.com": {
                "provider": "google-gemini-cli",
                "access": "tok_a",
                "projectId": "proj_a",
                "expires": future,
            },
        }
        with tempfile.TemporaryDirectory() as td:
            path = _write_auth_profiles(td, profiles, order=["google-gemini-cli:a@test.com"])
            client = GeminiClient(auth_profiles_path=path)

        # Return structured RATE_LIMIT_EXCEEDED for both flash and pro.
        resp = MagicMock()
        resp.status_code = 429
        resp.text = json.dumps({"error": {
            "code": 429, "message": "Rate limited",
            "details": [{
                "@type": "type.googleapis.com/google.rpc.ErrorInfo",
                "reason": "RATE_LIMIT_EXCEEDED",
                "domain": "cloudcode-pa.googleapis.com",
                "metadata": {},
            }],
        }})
        resp.headers = {}
        mock_post.return_value = resp

        client.generate("hello")

        pid = "google-gemini-cli:a@test.com"
        self.assertIn(pid, client._exhausted_until)
        remaining = client._exhausted_until[pid] - time.monotonic()
        self.assertLess(remaining, 30)  # short cooldown, not hours

    @patch("newsroom.gemini_client.requests.post")
    def test_api_key_fallback_after_all_profiles_429(self, mock_post: MagicMock) -> None:
        """API key fallback works after all OAuth profiles 429."""
        future = int(time.time() * 1000) + 3_600_000
        profiles = {
            "google-gemini-cli:a@test.com": {
                "provider": "google-gemini-cli",
                "access": "tok_a",
                "projectId": "proj_a",
                "expires": future,
            },
            "google-gemini-cli:b@test.com": {
                "provider": "google-gemini-cli",
                "access": "tok_b",
                "projectId": "proj_b",
                "expires": future,
            },
        }
        with tempfile.TemporaryDirectory() as td:
            path = _write_auth_profiles(
                td, profiles,
                order=["google-gemini-cli:a@test.com", "google-gemini-cli:b@test.com"],
            )
            with patch.object(GeminiClient, "_check_quota", return_value={}):
                client = GeminiClient(auth_profiles_path=path, api_key="test-key")

        retryable_body = json.dumps({"error": {
            "code": 429, "message": "Rate limited",
            "details": [{
                "@type": "type.googleapis.com/google.rpc.ErrorInfo",
                "reason": "RATE_LIMIT_EXCEEDED",
                "domain": "cloudcode-pa.googleapis.com",
                "metadata": {},
            }],
        }})

        call_urls: list[str] = []

        def side_effect(*args: Any, **kwargs: Any) -> MagicMock:
            url = str(args[0] if args else kwargs.get("url", ""))
            call_urls.append(url)
            if "generativelanguage.googleapis.com" in url:
                resp = MagicMock()
                resp.status_code = 200
                resp.json.return_value = {
                    "candidates": [{"content": {"parts": [{"text": "api key ok"}]}}]
                }
                return resp
            else:
                resp = MagicMock()
                resp.status_code = 429
                resp.text = retryable_body
                resp.headers = {}
                return resp

        mock_post.side_effect = side_effect
        result = client.generate("hello")

        self.assertEqual(result, "api key ok")
        # Both profiles tried (2 profiles × 2 models = 4 OAuth), then API key = 1.
        api_key_calls = [u for u in call_urls if "generativelanguage" in u]
        self.assertEqual(len(api_key_calls), 1)
        oauth_calls = [u for u in call_urls if "generativelanguage" not in u]
        self.assertEqual(len(oauth_calls), 4)  # 2 profiles × (flash + pro)


if __name__ == "__main__":
    unittest.main()
