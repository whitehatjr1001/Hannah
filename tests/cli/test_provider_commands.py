"""CLI tests for provider setup commands."""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

import hannah.cli.app as app_module


def test_configure_command_writes_provider_env_file(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    runner = CliRunner()

    result = runner.invoke(
        app_module.cli,
        [
            "configure",
            "--provider",
            "openai",
            "--api-key",
            "sk-openai-test",
            "--model",
            "gpt-4o-mini",
            "--env-file",
            str(env_path),
        ],
    )

    assert result.exit_code == 0
    assert "Configured OpenAI" in result.output
    content = env_path.read_text(encoding="utf-8")
    assert "HANNAH_MODEL=gpt-4o-mini" in content
    assert "OPENAI_API_KEY=sk-openai-test" in content


def test_providers_command_renders_status_from_env_file(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                "HANNAH_MODEL=claude-sonnet-4-6",
                "ANTHROPIC_API_KEY=sk-ant-test",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    runner = CliRunner()

    result = runner.invoke(app_module.cli, ["providers", "--env-file", str(env_path)])

    assert result.exit_code == 0
    assert "Anthropic" in result.output
    assert "configured" in result.output.lower()
    assert "active" in result.output.lower()


def test_configure_command_prompts_for_missing_values(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    runner = CliRunner()

    result = runner.invoke(
        app_module.cli,
        ["configure", "--env-file", str(env_path)],
        input="google\ngoogle-secret\ngoogle-secret\ngemini/gemini-2.0-flash\n",
    )

    assert result.exit_code == 0
    assert "Configured Google" in result.output
    content = env_path.read_text(encoding="utf-8")
    assert "GOOGLE_API_KEY=google-secret" in content
    assert "HANNAH_MODEL=gemini/gemini-2.0-flash" in content
