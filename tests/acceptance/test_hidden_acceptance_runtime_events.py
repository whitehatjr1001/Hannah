"""Masked acceptance scenarios for streamed runtime-event ordering."""

from __future__ import annotations

import asyncio
import inspect
import json
from dataclasses import dataclass, field
from typing import Any

import pytest

import hannah.cli.chat as chat_module
from hannah.agent.tool_registry import normalize_tool_args
from hannah.agent.worker_runtime import SPAWN_TOOL_SPEC
from hannah.runtime.core import RuntimeCore
from hannah.session.manager import SessionManager
from hannah.utils.console import Console


@dataclass
class _AcceptanceRuntimeRegistry:
    handlers: dict[str, Any] = field(default_factory=dict)
    allowed_names: tuple[str, ...] = ("spawn", "race_data")

    def get_tool_specs(self) -> list[dict[str, Any]]:
        specs: list[dict[str, Any]] = []
        for name in self.allowed_names:
            if name == "spawn":
                parameters = dict(SPAWN_TOOL_SPEC["parameters"])
                description = "Spawn a bounded worker."
            else:
                parameters = {
                    "type": "object",
                    "properties": {"race": {"type": "string"}},
                    "required": ["race"],
                    "additionalProperties": False,
                }
                description = "Fetch deterministic race data."
            specs.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": description,
                        "parameters": parameters,
                    },
                }
            )
        return specs

    def tool_names(self) -> set[str]:
        return set(self.allowed_names)

    def with_runtime_tools(self, handlers: dict[str, Any]) -> "_AcceptanceRuntimeRegistry":
        return _AcceptanceRuntimeRegistry(
            handlers=dict(handlers),
            allowed_names=self.allowed_names,
        )

    def subset(self, allowed_names: list[str] | set[str]) -> "_AcceptanceRuntimeRegistry":
        return _AcceptanceRuntimeRegistry(
            handlers=dict(self.handlers),
            allowed_names=tuple(name for name in self.allowed_names if name in set(allowed_names)),
        )

    def normalize_args(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        if name == "spawn":
            return normalize_tool_args(name, args, parameters=SPAWN_TOOL_SPEC["parameters"])
        return normalize_tool_args(
            name,
            args,
            parameters={
                "type": "object",
                "properties": {"race": {"type": "string"}},
                "required": ["race"],
                "additionalProperties": False,
            },
        )

    async def call(self, name: str, args: dict[str, Any], *, state: Any = None) -> dict[str, Any]:
        handler = self.handlers.get(name)
        if handler is not None:
            if "state" in inspect.signature(handler).parameters:
                result = handler(**args, state=state)
            else:
                result = handler(**args)
            if inspect.isawaitable(result):
                return await result
            return result
        return {"status": "ok", "tool": name, "args": args}


class _SpawnSequenceProvider:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        temperature: float,
        max_tokens: int,
    ) -> dict[str, Any]:
        del temperature, max_tokens
        snapshot = json.loads(json.dumps(messages))
        self.calls.append({"messages": snapshot, "tools": tools or []})
        if len(self.calls) == 1:
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "spawn-main",
                                    "type": "function",
                                    "function": {
                                        "name": "spawn",
                                        "arguments": {
                                            "task": "Summarize the Bahrain pit window.",
                                            "system_prompt": "You are a strategy worker.",
                                            "allowed_tools": ["race_data"],
                                            "result_contract": {"summary": "string"},
                                        },
                                    },
                                }
                            ],
                        }
                    }
                ]
            }
        if len(self.calls) == 2:
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": '{"summary":"Undercut window opens after lap 18."}',
                        }
                    }
                ]
            }
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Main answer uses the worker result.",
                    }
                }
            ]
        }


@pytest.mark.anyio
async def test_hidden_runtime_events_stream_and_persist_subagent_order(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = SessionManager(sessions_dir=tmp_path)
    manager.save(manager.get_or_create("cli:runtime"))
    rendered: list[str] = []

    def _render_runtime_event(event: Any) -> str | None:
        if not getattr(event, "event_type", "").startswith("subagent_"):
            return None
        rendered.append(str(event.event_type))
        return rendered[-1]

    monkeypatch.setattr(chat_module, "render_runtime_event", _render_runtime_event)

    handler = chat_module.build_runtime_event_handler(
        console=Console(),
        manager=manager,
        session_id="cli:runtime",
    )
    core = RuntimeCore(
        provider=_SpawnSequenceProvider(),
        registry=_AcceptanceRuntimeRegistry(),
    )
    core.event_bus.subscribe(handler)

    reply = await core.run_turn(
        messages=[{"role": "user", "content": "Delegate a Bahrain pit-window summary."}],
        session_id="cli:runtime",
    )

    assert reply["content"] == "Main answer uses the worker result."

    reloaded = SessionManager(sessions_dir=tmp_path).get_or_create("cli:runtime")
    subagent_records = [
        record
        for record in reloaded.event_records
        if str(record["payload"]["event_type"]).startswith("subagent_")
    ]
    event_types = [record["payload"]["event_type"] for record in subagent_records]
    worker_ids = [record["payload"]["worker_id"] for record in subagent_records]

    assert rendered == [
        "subagent_spawned",
        "subagent_progress",
        "subagent_progress",
        "subagent_completed",
    ]
    assert event_types == rendered
    assert len({worker_id for worker_id in worker_ids if worker_id}) == 1
    assert worker_ids[0].startswith("worker-")
