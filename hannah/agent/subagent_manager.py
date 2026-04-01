"""Background subagent orchestration and bus announcements."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Protocol, Sequence

from hannah.agent.context import RaceContext
from hannah.runtime.events import EventEnvelope
from hannah.utils.console import Console


class _BackgroundSubAgent(Protocol):
    name: str

    async def run(self, ctx: RaceContext) -> Any: ...


@dataclass(slots=True)
class SubagentManager:
    """Run subagents as background workers and announce lifecycle events."""

    event_bus: object | None = None
    console: Console | None = None
    session_id: str = "default"

    def __post_init__(self) -> None:
        if self.console is None:
            self.console = Console()

    async def run_all(
        self,
        ctx: RaceContext,
        subagents: Sequence[_BackgroundSubAgent],
    ) -> dict[str, dict[str, Any]]:
        self.console.print(
            f"  [dim]spawning {len(subagents)} background sub-agents concurrently...[/dim]"
        )
        tasks = [asyncio.create_task(self._run_one(agent, ctx)) for agent in subagents]
        results = await asyncio.gather(*tasks)

        output: dict[str, dict[str, Any]] = {}
        for result in results:
            if getattr(result, "success", False):
                output[str(getattr(result, "agent", ""))] = self._result_data(result)
                self.console.print(f"  [green]✓ {getattr(result, 'agent', 'subagent')}[/green]")
            else:
                error = getattr(result, "error", "unknown error")
                self.console.print(
                    f"  [yellow]⚠ {getattr(result, 'agent', 'subagent')}: {error}[/yellow]"
                )
        return output

    async def _run_one(self, agent: _BackgroundSubAgent, ctx: RaceContext) -> Any:
        worker_id = self._worker_id(agent)
        await self._publish(
            "subagent_spawned",
            worker_id=worker_id,
            payload={
                "task": self._task_description(agent, ctx),
            },
        )
        await self._publish(
            "subagent_progress",
            worker_id=worker_id,
            payload={
                "message": "Background worker running",
            },
        )

        try:
            result = await agent.run(ctx)
        except Exception as err:
            result = SimpleNamespace(agent=worker_id, success=False, data={}, error=str(err))
        await self._publish(
            "subagent_progress",
            worker_id=worker_id,
            payload={
                "message": "Background worker completed",
            },
        )
        await self._publish(
            "subagent_completed",
            worker_id=worker_id,
            payload=self.build_completion_payload(result),
        )
        return result

    def build_completion_payload(self, result: Any) -> dict[str, Any]:
        worker_id = self._worker_id(result)
        success = bool(getattr(result, "success", False))
        payload: dict[str, Any] = {
            "worker_id": worker_id,
            "status": "completed" if success else "error",
            "result": self._result_data(result),
        }
        error = getattr(result, "error", None)
        if not success and error:
            payload["error"] = str(error)
        return payload

    def build_result_message(self, result: Any) -> dict[str, Any]:
        payload = self.build_completion_payload(result)
        return {
            "role": "system",
            "name": "subagent_result",
            "worker_id": payload["worker_id"],
            "content": json.dumps(payload, default=str),
        }

    async def _publish(
        self,
        event_type: str,
        *,
        worker_id: str,
        payload: dict[str, Any],
    ) -> None:
        bus = self._event_bus()
        if bus is None:
            return
        await bus.publish(
            EventEnvelope.create(
                event_type=event_type,
                session_id=self.session_id,
                message_id=worker_id,
                worker_id=worker_id,
                payload=payload,
            )
        )

    def _event_bus(self) -> object | None:
        if self.event_bus is None:
            return None
        publish = getattr(self.event_bus, "publish", None)
        if callable(publish):
            return self.event_bus
        return None

    def _worker_id(self, agent: Any) -> str:
        candidate = getattr(agent, "agent", None) or getattr(agent, "name", None)
        if isinstance(candidate, str) and candidate.strip():
            return candidate
        return agent.__class__.__name__.lower()

    def _result_data(self, result: Any) -> dict[str, Any]:
        data = getattr(result, "data", {})
        if isinstance(data, dict):
            return dict(data)
        if data is None:
            return {}
        return {"value": data}

    def _task_description(self, agent: Any, ctx: RaceContext) -> str:
        worker_id = self._worker_id(agent)
        return (
            f"Run {worker_id} as a background worker for "
            f"{ctx.race.title()} {ctx.year}, {ctx.laps} laps, {ctx.weather} conditions."
        )
