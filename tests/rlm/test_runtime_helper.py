"""Optional runtime-helper contracts for local OpenAI-compatible endpoints."""

from __future__ import annotations

from hannah.rlm.helper import (
    build_chat_url,
    build_health_url,
    probe_runtime_helper,
    resolve_runtime_helper_config,
)


def test_runtime_helper_config_prefers_explicit_values_over_env(monkeypatch) -> None:
    monkeypatch.setenv("HANNAH_RLM_API_BASE", "http://env-base:9001")
    monkeypatch.setenv("HANNAH_RLM_API_KEY", "env-key")
    monkeypatch.setenv("HANNAH_MODEL", "env-model")

    env_config = resolve_runtime_helper_config()
    assert env_config.base_url == "http://env-base:9001"
    assert env_config.api_key == "env-key"
    assert env_config.model == "env-model"

    explicit = resolve_runtime_helper_config(
        base_url="http://explicit-base:9002",
        api_key="explicit-key",
        model="explicit-model",
    )
    assert explicit.base_url == "http://explicit-base:9002"
    assert explicit.api_key == "explicit-key"
    assert explicit.model == "explicit-model"


def test_runtime_helper_probe_reports_health_and_chat_success() -> None:
    calls: list[tuple[str, str]] = []

    def _fake_fetch(url: str, *, method: str, headers, body, timeout: float):
        del headers, timeout
        calls.append((url, method))
        if url.endswith("/health"):
            return {"status": "ok", "model": "rlm-f1-strategy-v1"}
        if url.endswith("/v1/chat/completions"):
            assert body["model"] == "rlm-f1-strategy-v1"
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "ready",
                        }
                    }
                ]
            }
        raise AssertionError(f"unexpected url: {url}")

    report = probe_runtime_helper(
        base_url="http://localhost:9001",
        api_key="none",
        model="rlm-f1-strategy-v1",
        fetch_json=_fake_fetch,
    )

    assert build_health_url(report["base_url"]) == "http://localhost:9001/health"
    assert build_chat_url(report["base_url"]) == "http://localhost:9001/v1/chat/completions"
    assert calls == [
        ("http://localhost:9001/health", "GET"),
        ("http://localhost:9001/v1/chat/completions", "POST"),
    ]
    assert report["ok"] is True
    assert report["health"]["ok"] is True
    assert report["chat"]["ok"] is True
    assert report["chat"]["message_preview"] == "ready"


def test_runtime_helper_probe_captures_failures_without_raising() -> None:
    def _failing_fetch(url: str, *, method: str, headers, body, timeout: float):
        del url, method, headers, body, timeout
        raise RuntimeError("connection refused")

    report = probe_runtime_helper(
        base_url="http://localhost:9001",
        api_key="none",
        model="rlm-f1-strategy-v1",
        fetch_json=_failing_fetch,
    )

    assert report["ok"] is False
    assert report["health"]["ok"] is False
    assert report["chat"]["ok"] is False
    assert "connection refused" in report["health"]["error"]
    assert "connection refused" in report["chat"]["error"]
