"""CLI tests for the primary agent command and compatibility wrappers."""

from __future__ import annotations

from click.testing import CliRunner

import hannah.cli.agent_command as agent_command_module
import hannah.cli.app as app_module
import hannah.cli.command_prompts as command_prompts_module


def test_agent_message_mode_invokes_shared_runtime(monkeypatch) -> None:
    runner = CliRunner()
    seen: list[tuple[str | None, bool, str, bool]] = []

    async def fake_run_agent_command(
        message: str | None,
        *,
        interactive: bool,
        session_id: str,
        new_session: bool,
    ) -> str:
        seen.append((message, interactive, session_id, new_session))
        return "ok"

    monkeypatch.setattr(agent_command_module, "run_agent_command", fake_run_agent_command)

    result = runner.invoke(app_module.cli, ["agent", "--message", "should we undercut"])

    assert result.exit_code == 0
    assert seen == [("should we undercut", False, "cli:direct", False)]


def test_ask_is_a_backward_compatible_wrapper(monkeypatch) -> None:
    runner = CliRunner()
    seen: list[tuple[str | None, bool]] = []

    async def fake_run_agent_command(
        message: str | None,
        *,
        interactive: bool,
        session_id: str,
        new_session: bool,
    ) -> str:
        del session_id, new_session
        seen.append((message, interactive))
        return "ok"

    monkeypatch.setattr(agent_command_module, "run_agent_command", fake_run_agent_command)

    result = runner.invoke(app_module.cli, ["ask", "who wins monza"])

    assert result.exit_code == 0
    assert seen == [("who wins monza", False)]


def test_simulate_is_a_wrapper_over_shared_agent_runtime(monkeypatch) -> None:
    runner = CliRunner()
    prompts: list[dict[str, object]] = []
    seen_messages: list[str | None] = []

    def fake_build_simulate_intent(
        *,
        race: str,
        year: int,
        driver: str | None,
        laps: int,
        weather: str,
    ) -> str:
        prompts.append(
            {
                "race": race,
                "year": year,
                "driver": driver,
                "laps": laps,
                "weather": weather,
            }
        )
        return "simulate prompt"

    async def fake_run_agent_command(
        message: str | None,
        *,
        interactive: bool,
        session_id: str,
        new_session: bool,
    ) -> str:
        del interactive, session_id, new_session
        seen_messages.append(message)
        return "ok"

    monkeypatch.setattr(command_prompts_module, "build_simulate_intent", fake_build_simulate_intent)
    monkeypatch.setattr(agent_command_module, "run_agent_command", fake_run_agent_command)

    result = runner.invoke(
        app_module.cli,
        ["simulate", "--race", "bahrain", "--year", "2026", "--driver", "VER", "--laps", "57", "--weather", "wet"],
    )

    assert result.exit_code == 0
    assert prompts == [
        {
            "race": "bahrain",
            "year": 2026,
            "driver": "VER",
            "laps": 57,
            "weather": "wet",
        }
    ]
    assert seen_messages == ["simulate prompt"]


def test_predict_and_strategy_use_centralized_wrapper_prompts(monkeypatch) -> None:
    runner = CliRunner()
    prompt_calls: list[tuple[str, dict[str, object]]] = []
    seen_messages: list[str | None] = []

    def fake_build_predict_intent(*, race: str, year: int) -> str:
        prompt_calls.append(("predict", {"race": race, "year": year}))
        return "predict prompt"

    def fake_build_strategy_intent(
        *,
        race: str,
        lap: int,
        driver: str,
        strategy_type: str,
    ) -> str:
        prompt_calls.append(
            (
                "strategy",
                {
                    "race": race,
                    "lap": lap,
                    "driver": driver,
                    "strategy_type": strategy_type,
                },
            )
        )
        return "strategy prompt"

    async def fake_run_agent_command(
        message: str | None,
        *,
        interactive: bool,
        session_id: str,
        new_session: bool,
    ) -> str:
        del interactive, session_id, new_session
        seen_messages.append(message)
        return "ok"

    monkeypatch.setattr(command_prompts_module, "build_predict_intent", fake_build_predict_intent)
    monkeypatch.setattr(command_prompts_module, "build_strategy_intent", fake_build_strategy_intent)
    monkeypatch.setattr(agent_command_module, "run_agent_command", fake_run_agent_command)

    predict_result = runner.invoke(app_module.cli, ["predict", "--race", "monza", "--year", "2025"])
    strategy_result = runner.invoke(
        app_module.cli,
        ["strategy", "--race", "bahrain", "--lap", "22", "--driver", "VER", "--type", "undercut"],
    )

    assert predict_result.exit_code == 0
    assert strategy_result.exit_code == 0
    assert prompt_calls == [
        ("predict", {"race": "monza", "year": 2025}),
        (
            "strategy",
            {
                "race": "bahrain",
                "lap": 22,
                "driver": "VER",
                "strategy_type": "undercut",
            },
        ),
    ]
    assert seen_messages == ["predict prompt", "strategy prompt"]


def test_sandbox_fetch_and_train_stay_off_shared_agent_runtime(monkeypatch) -> None:
    runner = CliRunner()
    direct_calls: list[str] = []

    async def fail_if_shared_runtime_used(
        message: str | None,
        *,
        interactive: bool,
        session_id: str,
        new_session: bool,
    ) -> str:
        del message, interactive, session_id, new_session
        raise AssertionError("shared runtime should not be used")

    class _FakeAgentLoop:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs

        async def run_command(self, command: str) -> None:
            direct_calls.append(command)

    monkeypatch.setattr(agent_command_module, "run_agent_command", fail_if_shared_runtime_used)
    monkeypatch.setattr(app_module, "AgentLoop", _FakeAgentLoop, raising=False)

    sandbox_result = runner.invoke(app_module.cli, ["sandbox", "--agents", "VER,NOR", "--race", "bahrain"])
    fetch_result = runner.invoke(app_module.cli, ["fetch", "--race", "monza", "--year", "2025", "--session", "R"])
    train_result = runner.invoke(app_module.cli, ["train", "all", "--years", "2024"])

    assert sandbox_result.exit_code == 0
    assert fetch_result.exit_code == 0
    assert train_result.exit_code == 0
    assert len(direct_calls) == 3
