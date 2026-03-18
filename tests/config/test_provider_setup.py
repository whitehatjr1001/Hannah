"""Provider setup helper tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from hannah.config.provider_setup import (
    apply_provider_configuration,
    get_provider_preset,
    load_env_context,
    summarize_provider_statuses,
)


def test_apply_provider_configuration_updates_env_file_and_clears_conflicts(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                "HANNAH_MODEL=claude-sonnet-4-6",
                "HANNAH_RLM_API_BASE=http://localhost:9001",
                "HANNAH_RLM_API_KEY=none",
                "HANNAH_FORCE_LOCAL_PROVIDER=1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    changes = apply_provider_configuration(
        env_path=env_path,
        provider=get_provider_preset("openai"),
        api_key="sk-test-openai",
        model="gpt-4o-mini",
    )

    content = env_path.read_text(encoding="utf-8")
    assert "HANNAH_MODEL=gpt-4o-mini" in content
    assert "OPENAI_API_KEY=sk-test-openai" in content
    assert "HANNAH_RLM_API_BASE" not in content
    assert "HANNAH_RLM_API_KEY" not in content
    assert "HANNAH_FORCE_LOCAL_PROVIDER" not in content
    assert changes["provider"] == "openai"
    assert changes["model"] == "gpt-4o-mini"


def test_summarize_provider_statuses_detects_google_api_key_aliases(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                "HANNAH_MODEL=gemini/gemini-2.0-flash",
                "GEMINI_API_KEY=google-test-key",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    env = load_env_context(env_path=env_path, include_process_env=False)
    statuses = summarize_provider_statuses(env)
    google = next(status for status in statuses if status.name == "google")

    assert google.configured is True
    assert google.active is True
    assert google.configured_env_var == "GEMINI_API_KEY"


def test_load_env_context_prefers_env_file_values(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                "HANNAH_MODEL=gpt-4o-mini",
                "OPENAI_API_KEY=sk-file-openai",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HANNAH_MODEL", "claude-sonnet-4-6")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-process-anthropic")

    env = load_env_context(env_path=env_path)

    assert env["HANNAH_MODEL"] == "gpt-4o-mini"
    assert env["OPENAI_API_KEY"] == "sk-file-openai"
    assert env["ANTHROPIC_API_KEY"] == "sk-process-anthropic"


def test_summarize_provider_statuses_ignores_example_placeholder_keys() -> None:
    env = {
        "HANNAH_MODEL": "gpt-4o-mini",
        "OPENAI_API_KEY": "sk-real-openai",
        "ANTHROPIC_API_KEY": "sk-ant-your-key-here",
        "GOOGLE_API_KEY": "your-google-key-here",
    }

    statuses = summarize_provider_statuses(env)
    anthropic = next(status for status in statuses if status.name == "anthropic")
    google = next(status for status in statuses if status.name == "google")

    assert anthropic.configured is False
    assert anthropic.configured_env_var is None
    assert google.configured is False
    assert google.configured_env_var is None
