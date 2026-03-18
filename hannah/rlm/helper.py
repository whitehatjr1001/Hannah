"""Optional helper utilities for probing local OpenAI-compatible runtimes."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Callable
from urllib import request


@dataclass(frozen=True)
class RuntimeHelperConfig:
    """Resolved local runtime helper configuration."""

    base_url: str
    api_key: str
    model: str
    timeout_s: float = 3.0


def resolve_runtime_helper_config(
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    timeout_s: float = 3.0,
) -> RuntimeHelperConfig:
    """Resolve optional runtime-helper config from explicit args or env."""
    return RuntimeHelperConfig(
        base_url=(base_url or os.getenv("HANNAH_RLM_API_BASE") or "http://localhost:9001").rstrip("/"),
        api_key=api_key or os.getenv("HANNAH_RLM_API_KEY") or "none",
        model=model or os.getenv("HANNAH_MODEL") or "rlm-f1-strategy-v1",
        timeout_s=timeout_s,
    )


def build_health_url(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/health"


def build_chat_url(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/v1/chat/completions"


def probe_runtime_helper(
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    timeout_s: float = 3.0,
    fetch_json: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Probe health and chat-completions endpoints without entering the main loop."""
    config = resolve_runtime_helper_config(
        base_url=base_url,
        api_key=api_key,
        model=model,
        timeout_s=timeout_s,
    )
    fetch = fetch_json or _fetch_json

    health_url = build_health_url(config.base_url)
    chat_url = build_chat_url(config.base_url)
    report: dict[str, Any] = {
        "base_url": config.base_url,
        "model": config.model,
        "health": {"ok": False},
        "chat": {"ok": False},
        "ok": False,
    }

    try:
        health_payload = fetch(
            health_url,
            method="GET",
            headers={"Authorization": f"Bearer {config.api_key}"},
            body=None,
            timeout=config.timeout_s,
        )
        report["health"] = {
            "ok": True,
            "status": health_payload.get("status", "unknown"),
            "model": health_payload.get("model"),
        }
    except Exception as err:
        report["health"] = {"ok": False, "error": str(err)}

    try:
        chat_payload = fetch(
            chat_url,
            method="POST",
            headers={
                "Authorization": f"Bearer {config.api_key}",
                "Content-Type": "application/json",
            },
            body={
                "model": config.model,
                "messages": [{"role": "user", "content": "Say ready."}],
                "temperature": 0.0,
                "max_tokens": 16,
            },
            timeout=config.timeout_s,
        )
        choices = chat_payload.get("choices", [])
        message = choices[0].get("message", {}) if choices and isinstance(choices[0], dict) else {}
        preview = str(message.get("content", "")).strip()
        report["chat"] = {
            "ok": bool(preview),
            "message_preview": preview,
        }
    except Exception as err:
        report["chat"] = {"ok": False, "error": str(err)}

    report["ok"] = bool(report["health"]["ok"] and report["chat"]["ok"])
    return report


def _fetch_json(
    url: str,
    *,
    method: str,
    headers: dict[str, str] | None,
    body: dict[str, Any] | None,
    timeout: float,
) -> dict[str, Any]:
    encoded_body = None if body is None else json.dumps(body).encode("utf-8")
    http_request = request.Request(url=url, method=method, headers=headers or {}, data=encoded_body)
    with request.urlopen(http_request, timeout=timeout) as response:
        payload = response.read().decode("utf-8")
    return json.loads(payload)
