"""CLI tests for Hannah chat sessions."""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

import hannah.cli.agent_command as agent_command_module
import hannah.cli.app as app_module
import hannah.cli.chat as chat_module
from hannah.session.manager import SessionManager
from hannah.utils.console import Console


class _FakeAgentLoop:
    def __init__(self, memory=None, **kwargs) -> None:
        del kwargs
        self.memory = memory

    async def run_turn(self, user_input: str) -> str:
        history = self.memory.get_recent(10) if self.memory is not None else []
        response = f"history={len(history)}"
        if self.memory is not None:
            self.memory.add("user", user_input)
            self.memory.add("assistant", response)
        return response


def test_chat_message_mode_persists_session_history(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HANNAH_SESSION_DIR", str(tmp_path))
    monkeypatch.setattr(agent_command_module, "AgentLoop", _FakeAgentLoop)
    monkeypatch.setattr(agent_command_module, "make_hannah_panel", lambda text: text)

    runner = CliRunner()
    first = runner.invoke(app_module.cli, ["chat", "--message", "first turn", "--session", "cli:test"])
    second = runner.invoke(app_module.cli, ["chat", "--message", "second turn", "--session", "cli:test"])

    assert first.exit_code == 0
    assert second.exit_code == 0

    session = SessionManager(sessions_dir=tmp_path).get_or_create("cli:test")
    assert [message["content"] for message in session.messages] == [
        "first turn",
        "history=0",
        "second turn",
        "history=2",
    ]


def test_chat_message_mode_routes_through_shared_agent_command(monkeypatch) -> None:
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

    result = runner.invoke(
        app_module.cli,
        ["chat", "--message", "session prompt", "--session", "cli:shared", "--new-session"],
    )

    assert result.exit_code == 0
    assert seen == [("session prompt", False, "cli:shared", True, True)]


def test_sessions_command_lists_saved_sessions(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HANNAH_SESSION_DIR", str(tmp_path))
    manager = SessionManager(sessions_dir=tmp_path)
    session = manager.get_or_create("cli:monaco")
    session.add_message("user", "box?")
    session.add_message("assistant", "pit now")
    manager.save(session)

    runner = CliRunner()
    result = runner.invoke(app_module.cli, ["sessions"])

    assert result.exit_code == 0
    assert "cli:monaco" in result.output
    assert "2" in result.output


def test_chat_local_provider_commands_are_handled(tmp_path, monkeypatch) -> None:
    manager = SessionManager(sessions_dir=tmp_path)
    console = Console()
    calls: list[tuple[str, str | None]] = []

    monkeypatch.setattr(
        chat_module,
        "render_provider_status_table",
        lambda *, console, env_file=Path(".env"): calls.append(("providers", str(env_file))),
    )
    monkeypatch.setattr(
        chat_module,
        "render_model_status",
        lambda *, console, env_file=Path(".env"): calls.append(("model", str(env_file))),
    )
    monkeypatch.setattr(
        chat_module,
        "run_provider_configure_flow",
        lambda *, console, provider=None, api_key=None, model=None, env_file=Path(".env"): calls.append(
            ("configure", provider)
        ),
    )

    providers_handled, session_id = chat_module._handle_local_command(
        command="/providers",
        session_id="cli:test",
        manager=manager,
        console=console,
    )
    model_handled, session_id = chat_module._handle_local_command(
        command="/model",
        session_id=session_id,
        manager=manager,
        console=console,
    )
    configure_handled, session_id = chat_module._handle_local_command(
        command="/configure openai",
        session_id=session_id,
        manager=manager,
        console=console,
    )

    assert providers_handled is True
    assert model_handled is True
    assert configure_handled is True
    assert session_id == "cli:test"
    assert calls == [
        ("providers", ".env"),
        ("model", ".env"),
        ("configure", "openai"),
        ("model", ".env"),
    ]
