"""Interactive nanobot-style chat helpers for the Hannah CLI."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any, Callable

try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.formatted_text import HTML
    from prompt_toolkit.history import FileHistory
except Exception:  # pragma: no cover - fallback path is covered instead
    PromptSession = None  # type: ignore[assignment]
    HTML = None  # type: ignore[assignment]
    FileHistory = None  # type: ignore[assignment]

from hannah.cli.format import BANNER
from hannah.cli.provider_ui import (
    render_model_status,
    render_provider_status_table,
    run_provider_configure_flow,
)
from hannah.session.manager import SessionManager, SessionMemory, create_session_key
from hannah.utils.console import Console, Table

EXIT_COMMANDS = {"exit", "quit", "/exit", "/quit", ":q"}


def is_interactive_terminal() -> bool:
    try:
        return sys.stdin.isatty() and sys.stdout.isatty()
    except Exception:
        return False


def run_interactive_chat(
    *,
    console: Console,
    panel_renderer: Callable[[str], Any],
    agent_loop_cls: type,
    session_id: str = "cli:direct",
    new_session: bool = False,
) -> None:
    """Run the interactive chat TUI on a real terminal."""
    if not is_interactive_terminal():
        console.print("[yellow]Interactive chat requires a TTY. Use `hannah chat --message ...` instead.[/yellow]")
        return

    manager = SessionManager()
    active_session = _resolve_session_id(session_id=session_id, new_session=new_session)
    prompt_session = _build_prompt_session()

    console.print(BANNER)
    _print_session_status(console=console, session_id=active_session)
    render_model_status(console=console)
    console.print("[dim]  Commands: /new, /sessions, /use <id>, /clear, /model, /providers, /configure [provider], /help, exit[/dim]\n")

    while True:
        try:
            user_input = _read_user_input(prompt_session=prompt_session, session_id=active_session)
        except (EOFError, KeyboardInterrupt):
            console.print("\nGoodbye!\n")
            return

        command = user_input.strip()
        if not command:
            continue
        if command.lower() in EXIT_COMMANDS:
            console.print("\nGoodbye!\n")
            return

        handled, active_session = _handle_local_command(
            command=command,
            session_id=active_session,
            manager=manager,
            console=console,
        )
        if handled:
            continue

        response = asyncio.run(
            _run_chat_turn(
                message=user_input,
                session_id=active_session,
                manager=manager,
                agent_loop_cls=agent_loop_cls,
            )
        )
        console.print()
        console.print(panel_renderer(response))
        console.print()


def run_message_chat(
    *,
    message: str,
    console: Console,
    panel_renderer: Callable[[str], Any],
    agent_loop_cls: type,
    session_id: str = "cli:direct",
    new_session: bool = False,
) -> None:
    """Run a one-shot message through the chat-session path."""
    manager = SessionManager()
    active_session = _resolve_session_id(session_id=session_id, new_session=new_session)
    response = asyncio.run(
        _run_chat_turn(
            message=message,
            session_id=active_session,
            manager=manager,
            agent_loop_cls=agent_loop_cls,
        )
    )
    _print_session_status(console=console, session_id=active_session)
    console.print()
    console.print(panel_renderer(response))
    console.print()


def print_sessions(*, console: Console, manager: SessionManager | None = None) -> None:
    """Render the saved chat session list."""
    session_manager = manager or SessionManager()
    sessions = session_manager.list_sessions()
    table = Table(title="Hannah Sessions")
    table.add_column("Session", style="cyan")
    table.add_column("Messages", style="magenta")
    table.add_column("Updated", style="green")
    for session in sessions:
        table.add_row(
            str(session.get("key", "")),
            str(session.get("message_count", 0)),
            str(session.get("updated_at", "")),
        )
    if not sessions:
        console.print("\n[dim]No saved chat sessions yet.[/dim]\n")
        return
    console.print()
    console.print(table)
    console.print()


def _resolve_session_id(*, session_id: str, new_session: bool) -> str:
    return create_session_key("cli") if new_session else session_id


async def _run_chat_turn(
    *,
    message: str,
    session_id: str,
    manager: SessionManager,
    agent_loop_cls: type,
) -> str:
    session = manager.get_or_create(session_id)
    memory = SessionMemory(manager=manager, session=session)
    agent_loop = agent_loop_cls(memory=memory)
    run_turn = getattr(agent_loop, "run_turn", None)
    if not callable(run_turn):
        raise RuntimeError("chat mode requires AgentLoop.run_turn()")
    return await run_turn(message)


def _handle_local_command(
    *,
    command: str,
    session_id: str,
    manager: SessionManager,
    console: Console,
) -> tuple[bool, str]:
    if not command.startswith("/"):
        return False, session_id

    if command == "/help":
        console.print(
            "[dim]  Commands: /new, /sessions, /use <id>, /clear, /model, /providers, /configure [provider], /help, exit[/dim]\n"
        )
        return True, session_id

    if command == "/sessions":
        print_sessions(console=console, manager=manager)
        _print_session_status(console=console, session_id=session_id)
        return True, session_id

    if command == "/model":
        render_model_status(console=console, env_file=Path(".env"))
        _print_session_status(console=console, session_id=session_id)
        return True, session_id

    if command == "/providers":
        render_provider_status_table(console=console, env_file=Path(".env"))
        _print_session_status(console=console, session_id=session_id)
        return True, session_id

    if command.startswith("/configure"):
        provider = command.split(None, 1)[1].strip() if " " in command else None
        run_provider_configure_flow(
            console=console,
            provider=provider or None,
            env_file=Path(".env"),
        )
        render_model_status(console=console, env_file=Path(".env"))
        _print_session_status(console=console, session_id=session_id)
        return True, session_id

    if command == "/new":
        new_session_id = create_session_key("cli")
        manager.save(manager.get_or_create(new_session_id))
        _print_session_status(console=console, session_id=new_session_id)
        return True, new_session_id

    if command == "/clear":
        memory = SessionMemory(manager=manager, session=manager.get_or_create(session_id))
        memory.clear()
        console.print(f"[dim]  Cleared session [cyan]{session_id}[/cyan].[/dim]\n")
        return True, session_id

    if command.startswith("/use "):
        target_session = command.split(None, 1)[1].strip()
        if not target_session:
            console.print("[yellow]Usage: /use <session-id>[/yellow]\n")
            return True, session_id
        manager.save(manager.get_or_create(target_session))
        _print_session_status(console=console, session_id=target_session)
        return True, target_session

    console.print(f"[yellow]Unknown command:[/yellow] {command}\n")
    return True, session_id


def _print_session_status(*, console: Console, session_id: str) -> None:
    console.print(f"\n[dim]  Session:[/dim] [cyan]{session_id}[/cyan]")


def _build_prompt_session():
    if PromptSession is None or FileHistory is None:
        return None
    history_path = Path(os.getenv("HANNAH_HISTORY_FILE", "data/history/cli_history"))
    history_path.parent.mkdir(parents=True, exist_ok=True)
    return PromptSession(
        history=FileHistory(str(history_path)),
        enable_open_in_editor=False,
        multiline=False,
    )


def _read_user_input(*, prompt_session, session_id: str) -> str:
    if prompt_session is None or HTML is None:
        return input(f"You [{session_id}] > ")
    return prompt_session.prompt(HTML("<b fg='ansiblue'>You</b> <style fg='ansibrightblack'>›</style> "))
