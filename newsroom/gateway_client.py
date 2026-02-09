from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


class ToolInvokeError(RuntimeError):
    def __init__(self, *, tool: str, action: str | None, status_code: int, payload: dict[str, Any], response_text: str):
        super().__init__(f"tools.invoke failed: tool={tool} action={action} status={status_code}")
        self.tool = tool
        self.action = action
        self.status_code = status_code
        self.payload = payload
        self.response_text = response_text


@dataclass(frozen=True)
class OpenClawGatewayConfig:
    http_url: str
    token: str


def _coerce_http_url(url: str) -> str:
    url = url.strip()
    if url.startswith("ws://"):
        return "http://" + url[len("ws://") :]
    if url.startswith("wss://"):
        return "https://" + url[len("wss://") :]
    return url


def load_gateway_config(openclaw_home: Path) -> OpenClawGatewayConfig:
    config_path = openclaw_home / "openclaw.json"
    raw = json.loads(config_path.read_text(encoding="utf-8"))

    gateway = raw.get("gateway", {}) or {}
    port = gateway.get("port")
    token = (
        os.environ.get("OPENCLAW_GATEWAY_TOKEN")
        or os.environ.get("CLAWDBOT_GATEWAY_TOKEN")
        or (gateway.get("auth", {}) or {}).get("token")
    )
    if not token:
        raise RuntimeError("Missing gateway token. Set OPENCLAW_GATEWAY_TOKEN or configure gateway.auth.token.")

    http_url = os.environ.get("OPENCLAW_GATEWAY_HTTP_URL") or os.environ.get("OPENCLAW_GATEWAY_URL")
    if http_url:
        http_url = _coerce_http_url(http_url)
    else:
        if not port:
            raise RuntimeError("Missing gateway port. Configure gateway.port in openclaw.json.")
        http_url = f"http://127.0.0.1:{port}"

    return OpenClawGatewayConfig(http_url=http_url.rstrip("/"), token=token)


class GatewayClient:
    def __init__(
        self,
        *,
        http_url: str,
        token: str,
        default_session_key: str,
        timeout_seconds: float = 30.0,
    ) -> None:
        self._http_url = http_url.rstrip("/")
        self._token = token
        self._default_session_key = default_session_key
        self._timeout_seconds = timeout_seconds

    @property
    def base_url(self) -> str:
        return self._http_url

    def invoke(
        self,
        *,
        tool: str,
        action: str | None = None,
        args: dict[str, Any] | None = None,
        session_key: str | None = None,
        extra_headers: dict[str, str] | None = None,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"tool": tool, "args": args or {}}
        if action is not None:
            payload["action"] = action
        payload["sessionKey"] = session_key or self._default_session_key

        headers = {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
        }
        if extra_headers:
            headers.update(extra_headers)

        timeout = timeout_seconds or self._timeout_seconds

        resp = requests.post(
            f"{self._http_url}/tools/invoke",
            headers=headers,
            json=payload,
            timeout=timeout,
        )

        # Gateway returns JSON for both success and error.
        try:
            data = resp.json()
        except Exception as e:
            raise ToolInvokeError(
                tool=tool, action=action, status_code=resp.status_code, payload=payload, response_text=resp.text
            ) from e

        if resp.status_code != 200 or not data.get("ok", False):
            raise ToolInvokeError(
                tool=tool, action=action, status_code=resp.status_code, payload=payload, response_text=resp.text
            )

        return data

    @staticmethod
    def result_text(resp: dict[str, Any]) -> str:
        result = (resp or {}).get("result", {}) or {}
        content = result.get("content", [])
        if not isinstance(content, list):
            return ""
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "\n".join(parts).strip()

    def invoke_result_json(
        self,
        *,
        tool: str,
        action: str | None = None,
        args: dict[str, Any] | None = None,
        session_key: str | None = None,
        extra_headers: dict[str, str] | None = None,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        resp = self.invoke(
            tool=tool,
            action=action,
            args=args,
            session_key=session_key,
            extra_headers=extra_headers,
            timeout_seconds=timeout_seconds,
        )
        text = self.result_text(resp)
        if not text:
            # Some tools return structured details; fall back to that.
            details = (resp.get("result", {}) or {}).get("details")
            if isinstance(details, dict):
                return details
            return {}

        try:
            return json.loads(text)
        except Exception:
            details = (resp.get("result", {}) or {}).get("details")
            if isinstance(details, dict):
                return details
            return {"text": text}

