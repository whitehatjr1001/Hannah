"""CLI tests for Hannah chat sessions."""

from __future__ import annotations

import asyncio
from pathlib import Path

from click.testing import CliRunner

import hannah.cli.agent_command as agent_command_module
import hannah.cli.app as app_module
import hannah.cli.chat as chat_module
import hannah.bus.queue as bus_queue_module
from hannah.runtime.events import EventEnvelope
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


def test_chat_runtime_event_handler_renders_and_persists_subagent_events(tmp_path, monkeypatch) -> None:
    manager = SessionManager(sessions_dir=tmp_path)
    session = manager.get_or_create("cli:test")
    manager.save(session)
    console = Console()
    rendered: list[str] = []

    monkeypatch.setattr(
        chat_module,
        "render_runtime_event",
        lambda envelope: rendered.append(f"{envelope.event_type}:{envelope.worker_id}") or rendered[-1],
    )

    handler = chat_module.build_runtime_event_handler(
        console=console,
        manager=manager,
        session_id="cli:test",
    )

    asyncio.run(
        handler(
            EventEnvelope.create(
                "subagent_progress",
                session_id="cli:test",
                message_id="msg-1",
                worker_id="strategy",
                payload={"message": "Running race_sim"},
            )
        )
    )

    assert rendered == ["subagent_progress:strategy"]
    lines = (tmp_path / "cli_test.jsonl").read_text(encoding="utf-8").splitlines()
    assert any('"record_type": "event"' in line for line in lines)


class _FakeEventBus:
    def __init__(self) -> None:
        self.handlers = []

    def subscribe(self, handler) -> None:
        self.handlers.append(handler)

    async def publish(self, envelope: EventEnvelope) -> None:
        for handler in list(self.handlers):
            await handler(envelope)


class _SessionAwareAgentLoop:
    def __init__(self, memory=None, **kwargs) -> None:
        del kwargs
        self.memory = memory
        self.event_bus = _FakeEventBus()

    async def run_turn(self, user_input: str, *, session_id: str = "default") -> str:
        if self.memory is not None:
            self.memory.add("user", user_input)
        await self.event_bus.publish(
            EventEnvelope.create(
                "subagent_progress",
                session_id=session_id,
                message_id="msg-chat",
                worker_id="strategy",
                payload={"message": "Running race_sim"},
            )
        )
        response = f"session={session_id}"
        if self.memory is not None:
            self.memory.add("assistant", response)
        return response


def test_chat_turn_threads_active_session_id_into_emitted_and_persisted_events(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("HANNAH_SESSION_DIR", str(tmp_path))
    console = Console()
    seen_event_session_ids: list[str] = []

    monkeypatch.setattr(
        chat_module,
        "render_runtime_event",
        lambda envelope: seen_event_session_ids.append(envelope.session_id) or envelope.event_type,
    )

    result = asyncio.run(
        chat_module.run_message_chat_session(
            message="compare strategies",
            console=console,
            panel_renderer=lambda text: text,
            agent_loop_cls=_SessionAwareAgentLoop,
            session_id="cli:active",
        )
    )

    assert result == "session=cli:active"
    assert seen_event_session_ids == ["cli:active"]
    reloaded = SessionManager(sessions_dir=tmp_path).get_or_create("cli:active")
    assert [message["content"] for message in reloaded.messages] == [
        "compare strategies",
        "session=cli:active",
    ]
    event_record = reloaded.event_records[0]
    assert event_record["session_id"] == "cli:active"
    assert event_record["payload"]["event_type"] == "subagent_progress"
    assert event_record["payload"]["worker_id"] == "strategy"
    assert event_record["payload"]["payload"]["message"] == "Running race_sim"


def test_interactive_chat_session_uses_async_prompt_api(monkeypatch) -> None:
    class _FakePromptSession:
        def __init__(self) -> None:
            self.awaited = False

        def prompt(self, *_args, **_kwargs) -> str:
            raise AssertionError("sync prompt() should not be used inside async chat")

        async def prompt_async(self, *_args, **_kwargs) -> str:
            self.awaited = True
            return "exit"

    fake_prompt_session = _FakePromptSession()

    monkeypatch.setattr(chat_module, "is_interactive_terminal", lambda: True)
    monkeypatch.setattr(chat_module, "_build_prompt_session", lambda: fake_prompt_session)
    monkeypatch.setattr(chat_module, "HTML", lambda value: value)
    monkeypatch.setattr(chat_module, "render_model_status", lambda **_kwargs: None)

    asyncio.run(
        chat_module.run_interactive_chat_session(
            console=Console(),
            panel_renderer=lambda text: text,
            agent_loop_cls=_FakeAgentLoop,
        )
    )

    assert fake_prompt_session.awaited is True


def test_chat_message_mode_publishes_bus_ingress_and_egress(monkeypatch) -> None:
    runner = CliRunner()
    published: list[str] = []

    class _RecordingBus(bus_queue_module.MessageBus):
        def __init__(self) -> None:
            super().__init__()
            published.append("created")

        async def publish(self, message):
            published.append(message.direction)
            await super().publish(message)

    class _FakeAgentLoop:
        def __init__(self, memory=None, **kwargs) -> None:
            del kwargs
            self.memory = memory

        async def run_turn(self, user_input: str, *, session_id: str = "default") -> str:
            del session_id
            return f"reply={user_input}"

    monkeypatch.setattr(bus_queue_module, "MessageBus", _RecordingBus)
    monkeypatch.setattr(agent_command_module, "AgentLoop", _FakeAgentLoop)
    monkeypatch.setattr(agent_command_module, "make_hannah_panel", lambda text: text)

    result = runner.invoke(app_module.cli, ["chat", "--message", "box the car", "--session", "cli:bus"])

    assert result.exit_code == 0
    assert published == ["created", "inbound", "outbound"]
