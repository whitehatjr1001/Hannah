"""Shared CLI execution path for the primary `agent` surface and wrappers."""

from __future__ import annotations

from dataclasses import dataclass, field

from hannah.agent.loop import AgentLoop
from hannah.cli.chat import run_interactive_chat_session, run_message_chat_session
from hannah.cli.format import make_hannah_panel
from hannah.utils.console import Console

DEFAULT_SESSION_ID = "cli:direct"

console = Console()


@dataclass
class _EphemeralMemory:
    messages: list[dict[str, str]] = field(default_factory=list)

    def add(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})

    def get_recent(self, n: int = 10) -> list[dict[str, str]]:
        if n <= 0:
            return []
        return list(self.messages[-n:])

    def clear(self) -> None:
        self.messages.clear()


async def run_agent_command(
    message: str | None,
    *,
    interactive: bool,
    session_id: str = DEFAULT_SESSION_ID,
    new_session: bool = False,
    persist_session: bool = True,
) -> str:
    """Execute a CLI turn through the shared runtime path."""
    if interactive:
        await run_interactive_chat_session(
            console=console,
            panel_renderer=make_hannah_panel,
            agent_loop_cls=AgentLoop,
            session_id=session_id,
            new_session=new_session,
        )
        return ""

    if not persist_session:
        console.print(f"\n[dim]  ❯[/dim] [white]{message or ''}[/white]\n")
        final_text = await AgentLoop(memory=_EphemeralMemory()).run_turn(message or "")
        console.print()
        console.print(make_hannah_panel(final_text))
        console.print()
        return final_text

    return await run_message_chat_session(
        message=message or "",
        console=console,
        panel_renderer=make_hannah_panel,
        agent_loop_cls=AgentLoop,
        session_id=session_id,
        new_session=new_session,
    )
