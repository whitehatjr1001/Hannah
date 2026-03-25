"""Shared CLI execution path for the primary `agent` surface and wrappers."""

from __future__ import annotations

from hannah.agent.loop import AgentLoop
from hannah.cli.chat import run_interactive_chat_session, run_message_chat_session
from hannah.cli.format import make_hannah_panel
from hannah.utils.console import Console

DEFAULT_SESSION_ID = "cli:direct"

console = Console()


async def run_agent_command(
    message: str | None,
    *,
    interactive: bool,
    session_id: str = DEFAULT_SESSION_ID,
    new_session: bool = False,
) -> str:
    """Execute a CLI turn through the shared session-aware runtime path."""
    if interactive:
        await run_interactive_chat_session(
            console=console,
            panel_renderer=make_hannah_panel,
            agent_loop_cls=AgentLoop,
            session_id=session_id,
            new_session=new_session,
        )
        return ""

    return await run_message_chat_session(
        message=message or "",
        console=console,
        panel_renderer=make_hannah_panel,
        agent_loop_cls=AgentLoop,
        session_id=session_id,
        new_session=new_session,
    )
