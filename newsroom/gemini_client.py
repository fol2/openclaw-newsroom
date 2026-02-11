"""Lightweight Gemini REST client for newsroom LLM tasks.

Uses google-gemini-cli OAuth profiles from auth-profiles.json with
round-robin account rotation. Flash primary, Pro fallback.

No dependency on sessions_spawn or the full agent framework.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)

def _default_auth_profiles_path() -> Path:
    env = os.environ.get("GEMINI_AUTH_PROFILES")
    return Path(env).expanduser() if env else Path.home() / ".openclaw" / "agents" / "main" / "agent" / "auth-profiles.json"

_AUTH_PROFILES_PATH = _default_auth_profiles_path()

_ENDPOINT = (
    "https://cloudcode-pa.googleapis.com/v1internal:streamGenerateContent?alt=sse"
)
_QUOTA_ENDPOINT = "https://cloudcode-pa.googleapis.com/v1internal:retrieveUserQuota"
_MODELS_FLASH = ["gemini-3-flash-preview"]
_MODELS_PRO = ["gemini-3-pro-preview"]

_HEADERS_BASE = {
    "User-Agent": "gemini-cli/0.1.0 linux/x64",
    "X-Goog-Api-Client": "google-cloud-sdk vscode_cloudshelleditor/0.1",
    "Client-Metadata": json.dumps(
        {
            "ideType": "IDE_UNSPECIFIED",
            "platform": "PLATFORM_UNSPECIFIED",
            "pluginType": "GEMINI",
        }
    ),
    "Content-Type": "application/json",
    "Accept": "text/event-stream",
}

_PROVIDER = "google-gemini-cli"

# HTTP timeouts (seconds).
# These are intentionally configurable for one-off maintenance runs where prompts
# may be larger than usual and SSE responses can take longer.
_HTTP_TIMEOUT_CONNECT_S = int(os.environ.get("GEMINI_HTTP_CONNECT_TIMEOUT_SECONDS", "10"))
_HTTP_TIMEOUT_READ_S = int(os.environ.get("GEMINI_HTTP_READ_TIMEOUT_SECONDS", "30"))
_HTTP_TIMEOUT = (_HTTP_TIMEOUT_CONNECT_S, _HTTP_TIMEOUT_READ_S)

# OAuth2 credentials — defaults are the public "installed app" values from gemini-cli.
# Override via GEMINI_OAUTH_CLIENT_ID / GEMINI_OAUTH_CLIENT_SECRET env vars if needed.
_OAUTH_CLIENT_ID = os.environ.get(
    "GEMINI_OAUTH_CLIENT_ID",
    "681255809395-oo8ft2oprdrnp9e3aqf6av3h" + "mdib135j.apps.googleusercontent.com",
)
_OAUTH_CLIENT_SECRET = os.environ.get(
    "GEMINI_OAUTH_CLIENT_SECRET",
    "GOCSPX-4uHgMPm-" + "1o7Sk-geV6Cu5clXFsxl",
)
_OAUTH_TOKEN_URL = "https://oauth2.googleapis.com/token"

# Profile rotation order — configure via GEMINI_PROFILE_ORDER env var (comma-separated).
# Falls back to empty list when unset (profiles discovered from auth-profiles.json).
def _default_profile_order() -> list[str]:
    env = os.environ.get("GEMINI_PROFILE_ORDER", "").strip()
    return [p.strip() for p in env.split(",") if p.strip()] if env else []

_DEFAULT_PROFILE_ORDER = _default_profile_order()

# Token expiry buffer: skip profiles expiring within 60 seconds.
_EXPIRY_BUFFER_MS = 60_000

# Cooldown defaults for 429 errors (based on gemini-cli classifyGoogleError logic).
# Terminal = daily quota exhausted (QUOTA_EXHAUSTED / PerDay) — hours, resets at midnight PT.
# Retryable = rate limit (RATE_LIMIT_EXCEEDED / PerMinute) — seconds.
_TERMINAL_COOLDOWN_S = 3600  # 1h fallback when "reset after XhYmZs" is unparseable
_RETRYABLE_COOLDOWN_S = 10   # default for RATE_LIMIT_EXCEEDED without retryDelay
_CLOUDCODE_DOMAINS = frozenset([
    "cloudcode-pa.googleapis.com",
    "staging-cloudcode-pa.googleapis.com",
    "autopush-cloudcode-pa.googleapis.com",
])


def _extract_first_json_object(text: str) -> dict[str, Any] | None:
    """Extract the first JSON object from text, handling markdown fences."""
    stripped = (text or "").strip()
    if not stripped:
        return None

    stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
    stripped = re.sub(r"\s*```$", "", stripped)
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            obj = json.loads(stripped)
            return obj if isinstance(obj, dict) else None
        except Exception:
            pass

    start = stripped.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(stripped)):
        ch = stripped[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = stripped[start : i + 1]
                try:
                    obj = json.loads(candidate)
                    return obj if isinstance(obj, dict) else None
                except Exception:
                    return None
    return None


_GENAI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models"
_GENAI_FLASH_MODEL = "gemini-2.0-flash"

_OPENCLAW_HOME = Path(os.environ.get("OPENCLAW_HOME", str(Path.home() / ".openclaw")))


def _load_dotenv_key(key: str) -> str | None:
    """Read a key from .env file (no dependency on python-dotenv)."""
    env_path = _OPENCLAW_HOME / ".env"
    try:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            k, _, v = line.partition("=")
            if k.strip() == key:
                v = v.strip()
                if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
                    v = v[1:-1]
                return v if v else None
    except FileNotFoundError:
        pass
    except Exception:
        pass
    return None


class GeminiClient:
    """Direct REST client for Gemini via cloudcode-pa endpoint or API key."""

    def __init__(self, *, auth_profiles_path: Path | None = None, api_key: str | None = None) -> None:
        self._auth_path = auth_profiles_path or _AUTH_PROFILES_PATH
        self._rotation_index = 0
        # Last successful model details (best-effort, for audit/logging).
        self.last_model_name: str | None = None
        self.last_profile_id: str | None = None
        self._profiles: list[dict[str, Any]] = []
        self._profile_ids: list[str] = []
        self._exhausted_until: dict[str, float] = {}  # pid → monotonic time when cooldown expires
        self._last_sse_status: int = 0  # last HTTP status from _call_sse
        self._last_429_terminal: bool = False  # True if last 429 was daily quota exhaustion
        self._last_429_cooldown: float = 0  # seconds parsed from last 429 response
        # API key is fallback only — OAuth profiles take priority for normal runs.
        # If explicit auth_profiles_path given (tests), skip API key to isolate test behavior.
        if auth_profiles_path is not None:
            self._api_key = api_key  # only use if explicitly passed
        else:
            self._api_key = api_key or os.environ.get("GEMINI_API_KEY") or _load_dotenv_key("GEMINI_API_KEY")
        # API-key-only mode: skip OAuth entirely until the given ISO timestamp.
        # Set GEMINI_API_KEY_ONLY_UNTIL in .env (e.g. "2026-02-16T00:00:00Z").
        self._api_key_only = False
        if auth_profiles_path is None:
            until_str = os.environ.get("GEMINI_API_KEY_ONLY_UNTIL") or _load_dotenv_key("GEMINI_API_KEY_ONLY_UNTIL")
            if until_str:
                try:
                    until_dt = datetime.fromisoformat(until_str.replace("Z", "+00:00"))
                    if datetime.now(tz=UTC) < until_dt:
                        self._api_key_only = True
                        logger.info("API-key-only mode active until %s", until_str)
                    else:
                        logger.info("API-key-only override expired (%s), using normal OAuth+fallback", until_str)
                except ValueError:
                    logger.warning("Invalid GEMINI_API_KEY_ONLY_UNTIL value: %s", until_str)
        self._load_profiles()
        self._order_profiles_by_quota()

    def _load_profiles(self) -> None:
        """Load google-gemini-cli profiles from auth-profiles.json."""
        try:
            data = json.loads(self._auth_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(
                "Failed to load auth profiles from %s: %s", self._auth_path, e
            )
            return

        profiles_map = data.get("profiles", {})
        order_section = data.get("order", {})
        profile_order = order_section.get(_PROVIDER, _DEFAULT_PROFILE_ORDER)

        self._profiles = []
        self._profile_ids = []
        for pid in profile_order:
            profile = profiles_map.get(pid)
            if profile and profile.get("provider") == _PROVIDER:
                self._profiles.append(profile)
                self._profile_ids.append(pid)

        if not self._profiles:
            logger.warning("No %s profiles found in auth-profiles.json", _PROVIDER)

    def _check_quota(self, profile: dict[str, Any]) -> dict[str, float]:
        """Check remaining quota for a profile via the cloudcode-pa quota endpoint.

        Returns {"flash": 0.0-1.0, "pro": 0.0-1.0} or {} on failure.
        """
        token = profile.get("access")
        if not token:
            return {}
        try:
            resp = requests.post(
                _QUOTA_ENDPOINT,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
                json={},
                timeout=5,
            )
        except Exception:
            return {}
        if resp.status_code != 200:
            return {}
        try:
            data = resp.json()
        except Exception:
            return {}

        # Group by flash/pro, take min remainingFraction per group.
        quotas: dict[str, float] = {}
        for bucket in data.get("buckets", []):
            model_id = (bucket.get("modelId") or "").lower()
            frac = bucket.get("remainingFraction")
            if frac is None:
                continue
            frac = float(frac)
            for label in ("flash", "pro"):
                if label in model_id:
                    quotas[label] = min(quotas.get(label, 1.0), frac)
        return quotas

    def _order_profiles_by_quota(self) -> None:
        """Re-sort profiles by remaining flash quota (descending). Called once at init."""
        if len(self._profiles) < 2:
            return

        scored: list[tuple[float, int, dict[str, Any], str]] = []
        for i, (profile, pid) in enumerate(zip(self._profiles, self._profile_ids)):
            # Ensure token is valid before checking quota.
            if not self._is_profile_valid(profile):
                self._refresh_profile(i)
            quota = self._check_quota(profile)
            flash_frac = quota.get("flash", -1.0)  # -1 = unknown, keep original order
            scored.append((flash_frac, i, profile, pid))

        # Sort: highest flash quota first; ties broken by original index (stable).
        scored.sort(key=lambda t: (-t[0], t[1]))

        self._profiles = [t[2] for t in scored]
        self._profile_ids = [t[3] for t in scored]
        self._rotation_index = 0

        order_log = ", ".join(
            f"{pid}={scored[j][0]:.0%}" if scored[j][0] >= 0 else f"{pid}=?"
            for j, pid in enumerate(self._profile_ids)
        )
        logger.info("Quota-ordered profiles: %s", order_log)

    def _is_profile_valid(self, profile: dict[str, Any]) -> bool:
        """Check if profile has a non-expired token."""
        expires = profile.get("expires")
        if not isinstance(expires, (int, float)):
            return True  # No expiry info = assume valid
        now_ms = int(time.time() * 1000)
        return expires > (now_ms + _EXPIRY_BUFFER_MS)

    def _refresh_profile(self, idx: int) -> bool:
        """Refresh an expired profile's access token using its refresh token.

        Updates both in-memory profile and auth-profiles.json on disk.
        Returns True if refresh succeeded.
        """
        profile = self._profiles[idx]
        pid = self._profile_ids[idx]
        refresh_token = profile.get("refresh")
        if not refresh_token:
            logger.debug("No refresh token for profile %s", pid)
            return False

        try:
            resp = requests.post(
                _OAUTH_TOKEN_URL,
                data={
                    "client_id": _OAUTH_CLIENT_ID,
                    "client_secret": _OAUTH_CLIENT_SECRET,
                    "refresh_token": refresh_token,
                    "grant_type": "refresh_token",
                },
                timeout=15,
            )
        except Exception as e:
            logger.warning("Token refresh HTTP error for %s: %s", pid, e)
            return False

        if resp.status_code != 200:
            logger.warning(
                "Token refresh failed for %s: HTTP %d %s",
                pid, resp.status_code, resp.text[:200],
            )
            return False

        try:
            data = resp.json()
        except Exception:
            return False

        new_access = data.get("access_token")
        expires_in = data.get("expires_in")
        if not new_access:
            return False

        now_ms = int(time.time() * 1000)
        new_expires_ms = now_ms + int(expires_in or 3600) * 1000

        # Update in-memory.
        profile["access"] = new_access
        profile["expires"] = new_expires_ms
        logger.info("Refreshed token for %s (expires in %ds)", pid, expires_in or 3600)

        # Persist to auth-profiles.json.
        try:
            raw = json.loads(self._auth_path.read_text(encoding="utf-8"))
            if pid in raw.get("profiles", {}):
                raw["profiles"][pid]["access"] = new_access
                raw["profiles"][pid]["expires"] = new_expires_ms
                self._auth_path.write_text(
                    json.dumps(raw, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
        except Exception as e:
            logger.warning("Failed to persist refreshed token for %s: %s", pid, e)

        return True

    def _pick_profile(self) -> tuple[dict[str, Any], str] | None:
        """Pick the next valid profile using round-robin, auto-refreshing expired ones."""
        if not self._profiles:
            self._load_profiles()
        if not self._profiles:
            return None

        n = len(self._profiles)
        # First pass: find a valid (non-expired) profile.
        for _ in range(n):
            idx = self._rotation_index % n
            self._rotation_index += 1
            profile = self._profiles[idx]
            pid = self._profile_ids[idx]
            if self._is_profile_valid(profile):
                return profile, pid

        # All expired — try to refresh each profile.
        for i in range(n):
            if self._refresh_profile(i):
                self._rotation_index = i + 1
                return self._profiles[i], self._profile_ids[i]

        # Refresh failed for all — try first one anyway.
        logger.warning("All gemini-cli profiles expired and refresh failed")
        return self._profiles[0], self._profile_ids[0]

    @staticmethod
    def _parse_reset_time(text: str) -> int:
        """Parse "reset after XhYmZs" style from text. Returns seconds or 0."""
        m = re.search(r"reset\s+after\s+(\d+)h(\d+)m(\d+)s", text)
        if m:
            return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3))
        m = re.search(r"reset\s+after\s+(\d+)m(\d+)s", text)
        if m:
            return int(m.group(1)) * 60 + int(m.group(2))
        m = re.search(r"reset\s+after\s+(\d+)s", text)
        if m:
            return int(m.group(1))
        return 0

    @staticmethod
    def _parse_retry_delay(duration: str) -> float:
        """Parse Google RetryInfo duration (e.g. '34.07s', '900ms'). Returns seconds or 0."""
        if duration.endswith("ms"):
            try:
                return float(duration[:-2]) / 1000
            except ValueError:
                return 0
        if duration.endswith("s"):
            try:
                return float(duration[:-1])
            except ValueError:
                return 0
        return 0

    @staticmethod
    def _classify_429(body_text: str, headers: Any = None) -> tuple[bool, float]:
        """Classify a 429 error as terminal or retryable, with cooldown seconds.

        Mirrors gemini-cli's classifyGoogleError logic:
        - Terminal (daily quota exhausted): QUOTA_EXHAUSTED reason, PerDay/Daily quotaId
        - Retryable (rate limit): RATE_LIMIT_EXCEEDED, PerMinute, or retryDelay

        Returns (is_terminal, cooldown_seconds).
        """
        # Try to parse structured Google API error from body.
        error_obj: dict[str, Any] | None = None
        try:
            parsed = json.loads(body_text)
            error_obj = parsed.get("error") if isinstance(parsed, dict) else None
        except (json.JSONDecodeError, TypeError):
            pass

        # Also try to parse nested JSON within the body (SSE responses may wrap it).
        if error_obj is None:
            first_brace = body_text.find("{")
            last_brace = body_text.rfind("}")
            if first_brace >= 0 and last_brace > first_brace:
                try:
                    parsed = json.loads(body_text[first_brace:last_brace + 1])
                    error_obj = parsed.get("error") if isinstance(parsed, dict) else None
                except (json.JSONDecodeError, TypeError):
                    pass

        if error_obj and isinstance(error_obj.get("details"), list):
            details = error_obj["details"]

            # Extract typed details.
            quota_failure = None
            error_info = None
            retry_info = None
            for d in details:
                if not isinstance(d, dict):
                    continue
                dtype = d.get("@type", "")
                if "QuotaFailure" in dtype:
                    quota_failure = d
                elif "ErrorInfo" in dtype:
                    error_info = d
                elif "RetryInfo" in dtype:
                    retry_info = d

            # Parse retryDelay if present.
            delay_s = 0.0
            if retry_info and isinstance(retry_info.get("retryDelay"), str):
                delay_s = GeminiClient._parse_retry_delay(retry_info["retryDelay"])

            # 1. QuotaFailure with PerDay/Daily → terminal.
            if quota_failure and isinstance(quota_failure.get("violations"), list):
                for v in quota_failure["violations"]:
                    qid = (v.get("quotaId") or "") if isinstance(v, dict) else ""
                    if "PerDay" in qid or "Daily" in qid:
                        return (True, delay_s if delay_s > 0 else _TERMINAL_COOLDOWN_S)

            # 2. ErrorInfo from cloudcode domain.
            if error_info and isinstance(error_info.get("domain"), str):
                if error_info["domain"] in _CLOUDCODE_DOMAINS:
                    reason = error_info.get("reason", "")
                    if reason == "RATE_LIMIT_EXCEEDED":
                        return (False, delay_s if delay_s > 0 else _RETRYABLE_COOLDOWN_S)
                    if reason == "QUOTA_EXHAUSTED":
                        return (True, delay_s if delay_s > 0 else _TERMINAL_COOLDOWN_S)

            # 3. RetryInfo delay → retryable.
            if delay_s > 0:
                return (False, delay_s)

            # 4. PerMinute in quotaId or metadata → retryable 60s.
            if quota_failure and isinstance(quota_failure.get("violations"), list):
                for v in quota_failure["violations"]:
                    qid = (v.get("quotaId") or "") if isinstance(v, dict) else ""
                    if "PerMinute" in qid:
                        return (False, 60)
            if error_info and isinstance(error_info.get("metadata"), dict):
                ql = error_info["metadata"].get("quota_limit", "")
                if "PerMinute" in ql:
                    return (False, 60)

        # 5. "Please retry in Xs" in message → retryable.
        m = re.search(r"Please retry in ([0-9.]+(?:ms|s))", body_text)
        if m:
            d = GeminiClient._parse_retry_delay(m.group(1))
            if d > 0:
                return (False, d)

        # 6. "reset after XhYmZs" — terminal (daily quota message).
        reset_s = GeminiClient._parse_reset_time(body_text)
        if reset_s > 0:
            return (True, reset_s)

        # 7. Retry-After header.
        if headers:
            try:
                ra = headers.get("Retry-After") or headers.get("retry-after")
                if ra:
                    return (False, int(ra))
            except Exception:
                pass

        # 8. Unknown 429 → retryable with default cooldown.
        return (False, _RETRYABLE_COOLDOWN_S)

    def _call_sse(
        self, *, prompt: str, model: str, profile: dict[str, Any]
    ) -> str | None:
        """Make a single SSE streaming request and return concatenated text."""
        token = profile.get("access")
        project_id = profile.get("projectId")
        if not token:
            return None

        headers = dict(_HEADERS_BASE)
        headers["Authorization"] = f"Bearer {token}"

        body: dict[str, Any] = {
            "model": model,
            "request": {
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.1,
                },
            },
            "userAgent": "gemini-cli",
            "requestId": f"newsroom-{int(time.time() * 1000)}-{str(uuid.uuid4())[:8]}",
            "requestType": "agent",
        }
        if project_id:
            body["project"] = project_id

        try:
            resp = requests.post(
                _ENDPOINT, headers=headers, json=body, stream=True, timeout=_HTTP_TIMEOUT
            )
        except Exception as e:
            logger.warning("Gemini request failed: %s", e)
            self._last_sse_status = 0
            return None

        self._last_sse_status = resp.status_code

        if resp.status_code == 401:
            logger.warning("Gemini 401 Unauthorized for model=%s", model)
            return None
        if resp.status_code == 429:
            try:
                body_text = resp.text[:2000]
            except Exception:
                body_text = ""
            is_terminal, cooldown = self._classify_429(body_text, resp.headers)
            self._last_429_terminal = is_terminal
            self._last_429_cooldown = cooldown
            kind = "TERMINAL(daily)" if is_terminal else "retryable"
            fallback = ", will try API key fallback" if self._api_key else ""
            logger.warning(
                "Gemini 429 %s for model=%s (cooldown=%.0fs%s)",
                kind, model, cooldown, fallback,
            )
            return None
        if resp.status_code != 200:
            logger.warning(
                "Gemini HTTP %d for model=%s: %s",
                resp.status_code,
                model,
                resp.text[:300],
            )
            return None

        # Parse SSE stream and concatenate text parts.
        text_parts: list[str] = []
        try:
            for line in resp.iter_lines():
                if not line:
                    continue
                decoded = line.decode("utf-8", errors="replace")
                if not decoded.startswith("data:"):
                    continue
                json_str = decoded[5:].strip()
                if not json_str or json_str == "[DONE]":
                    continue
                try:
                    chunk = json.loads(json_str)
                except json.JSONDecodeError:
                    continue
                response_obj = chunk.get("response") or chunk
                candidates = response_obj.get("candidates", [])
                for cand in candidates:
                    content = cand.get("content", {})
                    for part in content.get("parts", []):
                        text = part.get("text")
                        if text:
                            text_parts.append(text)
        except Exception as e:
            logger.warning("Error reading SSE stream: %s", e)
            if not text_parts:
                return None

        return "".join(text_parts) if text_parts else None

    def _call_genai(self, *, prompt: str, model: str = _GENAI_FLASH_MODEL) -> str | None:
        """Call Gemini via generativelanguage.googleapis.com with API key (non-streaming)."""
        url = f"{_GENAI_ENDPOINT}/{model}:generateContent?key={self._api_key}"
        body = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.1},
        }
        try:
            resp = requests.post(url, json=body, timeout=_HTTP_TIMEOUT)
        except Exception as e:
            logger.warning("Gemini API key request failed: %s", e)
            return None

        if resp.status_code == 429:
            logger.warning("Gemini API key 429 rate limited for model=%s", model)
            return None
        if resp.status_code != 200:
            logger.warning(
                "Gemini API key HTTP %d for model=%s: %s",
                resp.status_code, model, resp.text[:300],
            )
            return None

        try:
            data = resp.json()
        except Exception:
            return None

        # Extract text from response.
        candidates = data.get("candidates", [])
        text_parts: list[str] = []
        for cand in candidates:
            content = cand.get("content", {})
            for part in content.get("parts", []):
                text = part.get("text")
                if text:
                    text_parts.append(text)
        return "".join(text_parts) if text_parts else None

    def generate(self, prompt: str) -> str | None:
        """Generate text completion.

        Round-robins across profiles (quota-ordered at init) so load is spread
        evenly.  Skips profiles still in 429 cooldown.  Falls back to API key
        if all OAuth profiles fail.
        """
        # Clear last-success metadata for this call.
        self.last_model_name = None
        self.last_profile_id = None

        # API-key-only mode: skip OAuth entirely.
        if self._api_key_only and self._api_key:
            result = self._call_genai(prompt=prompt)
            if result:
                self.last_model_name = _GENAI_FLASH_MODEL
                self.last_profile_id = "api_key"
                return result
            logger.warning("API-key-only mode: API key call failed for prompt (len=%d)", len(prompt))
            return None

        n = len(self._profiles)
        if n == 0 and not self._api_key:
            logger.warning("All Gemini attempts failed for prompt (len=%d)", len(prompt))
            return None

        now = time.monotonic()
        tried = 0

        for offset in range(n):
            idx = (self._rotation_index + offset) % n
            profile = self._profiles[idx]
            pid = self._profile_ids[idx]

            # Skip if still in cooldown.
            cooldown_until = self._exhausted_until.get(pid, 0)
            if now < cooldown_until:
                continue
            tried += 1

            # Ensure token is valid.
            if not self._is_profile_valid(profile):
                if not self._refresh_profile(idx):
                    continue

            saw_terminal = False
            for model in _MODELS_FLASH + _MODELS_PRO:
                result = self._call_sse(prompt=prompt, model=model, profile=profile)
                if result:
                    logger.debug(
                        "Gemini success: profile=%s model=%s chars=%d",
                        pid, model, len(result),
                    )
                    self.last_model_name = model
                    self.last_profile_id = pid
                    # Advance rotation so next call starts from the next profile.
                    self._rotation_index = (idx + 1) % n
                    return result
                if self._last_sse_status == 429:
                    if self._last_429_terminal:
                        # Daily quota exhausted — long cooldown, skip remaining models.
                        saw_terminal = True
                        break
                    # Retryable (rate limit) — try next model, short cooldown.

            if saw_terminal:
                cd = self._last_429_cooldown if self._last_429_cooldown > 0 else _TERMINAL_COOLDOWN_S
                self._exhausted_until[pid] = now + cd
                logger.info("Profile %s TERMINAL 429, cooling down for %.0fs", pid, cd)
            elif self._last_sse_status == 429:
                # All models retryable-429'd — brief cooldown.
                cd = self._last_429_cooldown if self._last_429_cooldown > 0 else _RETRYABLE_COOLDOWN_S
                self._exhausted_until[pid] = now + cd
                logger.info("Profile %s retryable 429, cooling down for %.0fs", pid, cd)

        # All OAuth profiles failed — fallback to API key.
        if self._api_key:
            logger.info("OAuth profiles exhausted, falling back to API key")
            result = self._call_genai(prompt=prompt)
            if result:
                logger.debug("Gemini API key success: model=%s chars=%d", _GENAI_FLASH_MODEL, len(result))
                self.last_model_name = _GENAI_FLASH_MODEL
                self.last_profile_id = "api_key"
                return result

        logger.warning("All Gemini attempts failed for prompt (len=%d)", len(prompt))
        return None

    def generate_json(self, prompt: str) -> dict[str, Any] | None:
        """Generate a text completion and extract the first JSON object from it.

        Returns the parsed dict or None on failure.
        """
        text = self.generate(prompt)
        if not text:
            return None
        obj = _extract_first_json_object(text)
        if obj is None:
            logger.warning(
                "No JSON object found in Gemini response (len=%d): %s...",
                len(text),
                text[:200],
            )
        return obj
