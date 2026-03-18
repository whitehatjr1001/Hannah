"""Provider credential handling tests for the LiteLLM wrapper."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from hannah.config.loader import load_config
from hannah.providers.litellm_provider import LiteLLMProvider


def test_litellm_provider_skips_hosted_call_when_matching_key_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("HANNAH_FORCE_LOCAL_PROVIDER", raising=False)

    class _FakeLiteLLM:
        suppress_debug_info = False
        api_base = None
        api_key = None

        @staticmethod
        async def acompletion(**kwargs):
            raise AssertionError("acompletion should not run without matching provider credentials")

    monkeypatch.setitem(__import__("sys").modules, "litellm", _FakeLiteLLM)
    monkeypatch.setenv("HANNAH_MODEL", "gpt-4o-mini")

    provider = LiteLLMProvider(config=load_config(path=Path("config.yaml")))
    result = asyncio.run(
        provider.complete(
            messages=[{"role": "user", "content": "hello"}],
            tools=None,
            temperature=0.1,
            max_tokens=32,
        )
    )

    assert hasattr(result, "choices")
