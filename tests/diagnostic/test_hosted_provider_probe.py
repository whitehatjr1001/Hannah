"""Diagnostic test: hit the hosted provider directly with current .env values.

Run with:
    .venv/bin/python -m pytest tests/diagnostic/test_hosted_provider_probe.py -v -s
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_env_file() -> dict[str, str]:
    env_path = REPO_ROOT / ".env"
    values: dict[str, str] = {}
    if not env_path.exists():
        return values
    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


@pytest.fixture(autouse=True)
def _apply_env_from_file() -> None:
    """Merge .env into os.environ so the provider sees real keys."""
    file_values = _load_env_file()
    saved: dict[str, str | None] = {}
    for key, value in file_values.items():
        saved[key] = os.environ.get(key)
        os.environ[key] = value
    yield
    for key, old in saved.items():
        if old is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = old


class TestHostedProviderProbe:
    """Probe the actual provider path that the agent loop uses."""

    def test_env_vars_loaded(self) -> None:
        """Confirm .env values are visible to os.environ."""
        model = os.getenv("HANNAH_MODEL", "")
        openai_key = os.getenv("OPENAI_API_KEY", "")
        anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
        force_local = os.getenv("HANNAH_FORCE_LOCAL_PROVIDER", "")

        assert model, "HANNAH_MODEL is not set in .env"
        assert openai_key, "OPENAI_API_KEY is not set in .env"
        assert force_local.lower() not in {"1", "true", "yes", "on"}, (
            f"HANNAH_FORCE_LOCAL_PROVIDER={force_local!r} will force local fallback"
        )
        # OPENAI_API_KEY must look like a real key
        assert openai_key.startswith("sk-"), (
            f"OPENAI_API_KEY does not look like a real key: {openai_key[:12]}..."
        )

    def test_config_resolves_model(self) -> None:
        """Config should pick up HANNAH_MODEL from env."""
        from hannah.config.loader import load_config

        cfg = load_config()
        assert cfg.agent.model == os.getenv("HANNAH_MODEL"), (
            f"config.agent.model={cfg.agent.model!r} != HANNAH_MODEL={os.getenv('HANNAH_MODEL')!r}"
        )

    def test_provider_detects_openai(self) -> None:
        """detect_provider_from_model should return 'openai' for gpt-4o-mini."""
        from hannah.config.provider_setup import detect_provider_from_model

        model = os.getenv("HANNAH_MODEL", "")
        provider = detect_provider_from_model(model)
        assert provider == "openai", f"detected {provider!r} for model {model!r}"

    def test_hosted_credentials_available(self) -> None:
        """LiteLLMProvider should NOT fall back to local for this config."""
        from hannah.config.loader import load_config
        from hannah.providers.litellm_provider import LiteLLMProvider

        cfg = load_config()
        provider = LiteLLMProvider(config=cfg)

        assert not provider._force_local(), (
            "HANNAH_FORCE_LOCAL_PROVIDER is blocking hosted calls"
        )
        assert provider._hosted_credentials_available(), (
            "_hosted_credentials_available returned False — provider will silently fall back to local"
        )

    def test_litellm_import(self) -> None:
        """LiteLLM must be importable."""
        try:
            import litellm  # noqa: F401
        except ImportError as exc:
            pytest.fail(f"litellm is not installed: {exc}")

    def test_litellm_direct_completion(self) -> None:
        """Hit LiteLLM directly with the same model + key the agent uses."""
        import litellm

        litellm.suppress_debug_info = True
        model = os.getenv("HANNAH_MODEL", "gpt-4o-mini")
        api_key = os.getenv("OPENAI_API_KEY", "")

        assert api_key.startswith("sk-"), "OPENAI_API_KEY not set or invalid"

        try:
            response = litellm.completion(
                model=model,
                messages=[{"role": "user", "content": "Say hello in one word."}],
                max_tokens=10,
                temperature=0.0,
                api_key=api_key,
            )
            text = response.choices[0].message.content
            assert text, "Model returned empty content"
        except Exception as exc:
            pytest.fail(f"LiteLLM direct call failed: {type(exc).__name__}: {exc}")

    def test_provider_complete_async(self) -> None:
        """Hit the full LiteLLMProvider.complete() path the agent loop uses."""
        from hannah.config.loader import load_config
        from hannah.providers.litellm_provider import LiteLLMProvider

        cfg = load_config()
        provider = LiteLLMProvider(config=cfg)

        result = asyncio.run(
            provider.complete(
                messages=[{"role": "user", "content": "Say hello in one word."}],
                tools=None,
                temperature=0.0,
                max_tokens=10,
            )
        )

        # If we got a LocalCompletion back, the hosted call was skipped
        from hannah.providers.local_fallback import LocalCompletion

        assert not isinstance(result, LocalCompletion), (
            "Provider returned LocalCompletion — hosted call was bypassed. "
            "Check _force_local() and _hosted_credentials_available()."
        )

        # Should have choices[0].message.content
        assert hasattr(result, "choices"), (
            f"Response has no 'choices' attr: {type(result)}"
        )
        assert len(result.choices) > 0, "Response has zero choices"
        text = result.choices[0].message.content
        assert text, "Model returned empty content"

    def test_full_agent_loop_one_turn(self) -> None:
        """Run one turn through AgentLoop and confirm it hits the hosted model."""
        from hannah.agent.loop import AgentLoop
        from hannah.providers.local_fallback import LocalCompletion

        loop = AgentLoop()

        # Verify provider is NOT local fallback before the turn
        assert not loop.provider._force_local(), "AgentLoop provider is force-local"
        assert loop.provider._hosted_credentials_available(), (
            "AgentLoop provider has no hosted credentials"
        )

        result = asyncio.run(loop.run_turn("Say hello in exactly one word."))

        # If the result is the local fallback message, the hosted call was skipped
        assert "No external model was available" not in result, (
            f"AgentLoop returned local fallback message: {result[:120]}..."
        )
        assert len(result.strip()) > 0, "AgentLoop returned empty result"
