"""CLI tests for the primary agent command and compatibility wrappers."""

from __future__ import annotations

from click.testing import CliRunner

import hannah.cli.agent_command as agent_command_module
import hannah.cli.app as app_module
import hannah.cli.command_prompts as command_prompts_module


def test_agent_message_mode_invokes_shared_runtime(monkeypatch) -> None:
    runner = CliRunner()
    seen: list[tuple[str | None, bool, str, bool, bool]] = []

    async def fake_run_agent_command(
        message: str | None,
        *,
        interactive: bool,
        session_id: str,
        new_session: bool,
        persist_session: bool,
    ) -> str:
        seen.append((message, interactive, session_id, new_session, persist_session))
        return "ok"

    monkeypatch.setattr(agent_command_module, "run_agent_command", fake_run_agent_command)

    result = runner.invoke(app_module.cli, ["agent", "--message", "should we undercut"])

    assert result.exit_code == 0
    assert seen == [("should we undercut", False, "cli:direct", False, True)]


def test_ask_is_a_backward_compatible_wrapper(monkeypatch) -> None:
    runner = CliRunner()
    seen: list[tuple[str | None, bool, bool]] = []

    async def fake_run_agent_command(
        message: str | None,
        *,
        interactive: bool,
        session_id: str,
        new_session: bool,
        persist_session: bool,
    ) -> str:
        del session_id, new_session
        seen.append((message, interactive, persist_session))
        return "ok"

    monkeypatch.setattr(agent_command_module, "run_agent_command", fake_run_agent_command)

    result = runner.invoke(app_module.cli, ["ask", "who wins monza"])

    assert result.exit_code == 0
    assert seen == [("who wins monza", False, False)]


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
        persist_session: bool,
    ) -> str:
        del interactive, session_id, new_session, persist_session
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
        persist_session: bool,
    ) -> str:
        del interactive, session_id, new_session, persist_session
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


class _OneShotAgentLoop:
    def __init__(self, memory=None, **kwargs) -> None:
        del kwargs
        self.memory = memory

    async def run_turn(self, user_input: str) -> str:
        if self.memory is not None:
            self.memory.add("user", user_input)
        return f"reply={user_input}"


def test_legacy_one_shot_wrappers_do_not_persist_or_render_session_ui(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HANNAH_SESSION_DIR", str(tmp_path))
    monkeypatch.setattr(agent_command_module, "AgentLoop", _OneShotAgentLoop)
    monkeypatch.setattr(agent_command_module, "make_hannah_panel", lambda text: text)

    runner = CliRunner()
    first = runner.invoke(app_module.cli, ["ask", "who wins monza"])
    second = runner.invoke(app_module.cli, ["simulate", "--race", "bahrain", "--driver", "VER"])

    assert first.exit_code == 0
    assert second.exit_code == 0
    assert "Session:" not in first.output
    assert "Session:" not in second.output
    assert list(tmp_path.glob("*.jsonl")) == []


def test_excluded_commands_stay_off_shared_agent_runtime(tmp_path, monkeypatch) -> None:
    runner = CliRunner()
    direct_calls: list[str] = []
    utility_calls: list[str] = []

    async def fail_if_shared_runtime_used(
        message: str | None,
        *,
        interactive: bool,
        session_id: str,
        new_session: bool,
        persist_session: bool,
    ) -> str:
        del message, interactive, session_id, new_session, persist_session
        raise AssertionError("shared runtime should not be used")

    class _FakeAgentLoop:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs

        async def run_command(self, command: str) -> None:
            direct_calls.append(command)

    monkeypatch.setattr(agent_command_module, "run_agent_command", fail_if_shared_runtime_used)
    monkeypatch.setattr(app_module, "AgentLoop", _FakeAgentLoop, raising=False)
    monkeypatch.setattr(app_module, "render_provider_status_table", lambda **kwargs: utility_calls.append("providers"))
    monkeypatch.setattr(app_module, "run_provider_configure_flow", lambda **kwargs: utility_calls.append("configure"))
    monkeypatch.setattr(app_module, "render_model_status", lambda **kwargs: utility_calls.append("model"))
    monkeypatch.setattr(app_module, "print_sessions", lambda **kwargs: utility_calls.append("sessions"))

    class _FakeToolRegistry:
        def list_tools(self) -> None:
            utility_calls.append("tools")

    monkeypatch.setattr(app_module, "ToolRegistry", _FakeToolRegistry)

    import hannah.rlm.helper as rlm_helper_module
    import hannah.tools.race_sim.tool as race_sim_tool_module

    async def fake_trace_run(**kwargs):
        del kwargs
        utility_calls.append("trace")
        return {"trace": {"summary": "ok"}}

    monkeypatch.setattr(race_sim_tool_module, "run", fake_trace_run)
    monkeypatch.setattr(
        rlm_helper_module,
        "probe_runtime_helper",
        lambda **kwargs: utility_calls.append("rlm-probe")
        or {"ok": True, "base_url": "http://localhost", "model": "test", "health": {}, "chat": {}},
    )

    sandbox_result = runner.invoke(app_module.cli, ["sandbox", "--agents", "VER,NOR", "--race", "bahrain"])
    fetch_result = runner.invoke(app_module.cli, ["fetch", "--race", "monza", "--year", "2025", "--session", "R"])
    train_result = runner.invoke(app_module.cli, ["train", "all", "--years", "2024"])
    trace_result = runner.invoke(app_module.cli, ["trace", "--race", "bahrain"])
    providers_result = runner.invoke(app_module.cli, ["providers"])
    configure_result = runner.invoke(
        app_module.cli,
        ["configure", "--provider", "openai", "--api-key", "sk-test", "--model", "gpt-test", "--env-file", str(tmp_path / ".env")],
    )
    tools_result = runner.invoke(app_module.cli, ["tools"])
    model_result = runner.invoke(app_module.cli, ["model"])
    sessions_result = runner.invoke(app_module.cli, ["sessions"])
    probe_result = runner.invoke(app_module.cli, ["rlm-probe"])

    assert sandbox_result.exit_code == 0
    assert fetch_result.exit_code == 0
    assert train_result.exit_code == 0
    assert trace_result.exit_code == 0
    assert providers_result.exit_code == 0
    assert configure_result.exit_code == 0
    assert tools_result.exit_code == 0
    assert model_result.exit_code == 0
    assert sessions_result.exit_code == 0
    assert probe_result.exit_code == 0
    assert len(direct_calls) == 3
    assert utility_calls == ["trace", "providers", "configure", "tools", "model", "sessions", "rlm-probe"]


def test_bare_tty_startup_dispatches_to_agent(monkeypatch) -> None:
    runner = CliRunner()
    seen: list[tuple[str | None, bool, str, bool, bool]] = []

    async def fake_run_agent_command(
        message: str | None,
        *,
        interactive: bool,
        session_id: str,
        new_session: bool,
        persist_session: bool,
    ) -> str:
        seen.append((message, interactive, session_id, new_session, persist_session))
        return "ok"

    monkeypatch.setattr(app_module, "is_interactive_terminal", lambda: True)
    monkeypatch.setattr(agent_command_module, "run_agent_command", fake_run_agent_command)

    result = runner.invoke(app_module.cli, [])

    assert result.exit_code == 0
    assert seen == [(None, True, "cli:direct", False, True)]
