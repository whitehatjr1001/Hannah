"""Main Hannah tool-using loop."""

from __future__ import annotations

import asyncio
import inspect
import json
import re
from typing import Any

from hannah.agent.memory import Memory
from hannah.agent.persona import HANNAH_PERSONA
from hannah.agent.worker_runtime import WorkerRuntime, WorkerSpec, make_worker_id
from hannah.agent.tool_registry import ToolRegistry, normalize_tool_args
from hannah.cli.format import make_hannah_panel
from hannah.config.loader import load_config
from hannah.providers.base import (
    coerce_payload,
    coerce_provider_message,
    coerce_tool_calls,
    extract_first_message,
    flatten_content_blocks,
    message_to_provider_message,
    normalize_message_content,
    payload_to_provider_message,
)
from hannah.providers.registry import ProviderRegistry
from hannah.runtime.bus import AsyncEventBus
from hannah.runtime.context import MainAgentContext, RuntimeContextBuilder
from hannah.runtime.events import EventEnvelope
from hannah.runtime.turn_state import TurnState
from hannah.utils.console import Console

console = Console()
_TRAIN_MODEL_TOOL_NAME = "train_model"
_TOOL_MESSAGE_MAX_CHARS = 20_000
_EXPLICIT_TRAINING_HINTS = (
    "train ",
    "train the",
    "retrain",
    "training",
    "fine-tune",
    "fine tune",
)
_ANALYSIS_INTENT_HINTS = (
    "predict",
    "prediction",
    "strategy",
    "startegy",
    "startagy",
    "pit",
    "simulate",
    "simulation",
    "analysis",
    "analyze",
    "upcoming",
    "next race",
    "next grand prix",
    "ai model",
    "ai models",
    "model the",
    "models to predict",
)
_ANALYSIS_DEFERRAL_HINTS = (
    "let me know if you'd like",
    "let me know if you would like",
    "if you'd like me to proceed",
    "if you would like me to proceed",
    "i can analyze",
    "i can model",
)


class AgentLoop:
    """Minimal NanoChat-style agent loop."""

    def __init__(
        self,
        memory: Memory | None = None,
        registry: ToolRegistry | None = None,
        provider: object | None = None,
    ) -> None:
        self.config = load_config()
        self.memory = memory or Memory()
        self.registry = registry or ToolRegistry()
        self.provider = provider or ProviderRegistry.from_config(self.config)
        self.event_bus = AsyncEventBus()
        self.context_builder = RuntimeContextBuilder()
        self._worker_runtime: WorkerRuntime | None = None
        runtime_binding = getattr(self.registry, "with_runtime_tools", None)
        if callable(runtime_binding):
            self._worker_runtime = WorkerRuntime(
                provider=self.provider,
                registry=self.registry,
                event_bus=self.event_bus,
                context_builder=self.context_builder,
                temperature=self.config.agent.temperature,
                max_tokens=self.config.agent.max_tokens,
                console=console,
            )
            self.registry = runtime_binding({"spawn": self._handle_spawn_tool})
        self.tools = self.registry.get_tool_specs()

    async def run_turn(self, user_input: str, *, session_id: str = "default") -> str:
        turn_tools = self._select_tools_for_turn(user_input)
        messages = self.context_builder.build_main_turn(
            MainAgentContext(
                persona=HANNAH_PERSONA,
                dynamic_guidance=self._dynamic_turn_guidance(user_input),
                recent_messages=tuple(self.memory.get_recent(n=10)),
                user_input=user_input,
            )
        )
        state = TurnState(
            session_id=session_id,
            messages=self.context_builder.build_main_messages(messages),
        )
        retry_used = False

        await self._publish_event(
            "user_message_received",
            state,
            payload={"content": state.latest_user_content()},
        )

        while True:
            await self._publish_event(
                "provider_request_started",
                state,
                payload={
                    "message_count": len(state.messages),
                    "tool_names": [tool["function"]["name"] for tool in turn_tools],
                },
            )
            try:
                response = await self.provider.complete(
                    messages=state.snapshot_messages(),
                    tools=turn_tools if turn_tools else None,
                    temperature=self.config.agent.temperature,
                    max_tokens=self.config.agent.max_tokens,
                )
            except Exception as exc:
                await self._publish_event(
                    "error_emitted",
                    state,
                    payload={"stage": "provider", "error": str(exc)},
                )
                raise

            message = self._coerce_first_message(response)
            tool_calls = list(message.tool_calls or [])
            await self._publish_event(
                "provider_response_received",
                state,
                payload={"has_tool_calls": bool(tool_calls)},
            )

            if tool_calls:
                for tool_call in tool_calls:
                    console.print(
                        f"  [dim cyan]◆ calling tool:[/dim cyan] [cyan]{tool_call.function.name}[/cyan]"
                    )
                state.append_message(message.model_dump())
                tool_messages = await self._execute_tool_calls(
                    state.snapshot_messages(),
                    tool_calls,
                    state=state,
                )
                state.extend_messages(tool_messages)
                continue

            final_text = message.content or ""
            if self._should_retry_analysis_turn(
                user_input=user_input,
                final_text=final_text,
                retry_used=retry_used,
            ):
                state.append_message(message.model_dump())
                state.append_message({"role": "system", "content": self._analysis_retry_guidance()})
                retry_used = True
                continue

            state.append_message(message.model_dump())
            await self._publish_event(
                "final_answer_emitted",
                state,
                payload={"content": final_text},
            )
            break

        self.memory.add("user", user_input)
        self.memory.add("assistant", final_text)
        return final_text

    def _select_tools_for_turn(self, user_input: str) -> list[dict[str, Any]]:
        if not self._should_hide_train_model(user_input):
            return list(self.tools)
        return [
            tool
            for tool in self.tools
            if tool.get("function", {}).get("name") != _TRAIN_MODEL_TOOL_NAME
        ]

    def _dynamic_turn_guidance(self, user_input: str) -> str | None:
        if not self._should_hide_train_model(user_input):
            return None
        return (
            "This turn is a race analysis or prediction request for a specific event. "
            "Do not call train_model even if the user mentions models or AI. "
            "Treat the question as a direct request to do the analysis now, not a request for permission. "
            "You must call the relevant analysis tools before answering. "
            "Use race_data, race_sim, pit_strategy, and predict_winner as needed, then answer decisively."
        )

    def _should_hide_train_model(self, user_input: str) -> bool:
        lowered = user_input.lower()
        if any(token in lowered for token in _EXPLICIT_TRAINING_HINTS):
            return False
        has_race_context = re.search(r"\b(race|grand prix|prix|gp)\b", lowered) is not None
        has_analysis_intent = any(token in lowered for token in _ANALYSIS_INTENT_HINTS)
        return has_race_context and has_analysis_intent

    def _should_retry_analysis_turn(self, *, user_input: str, final_text: str, retry_used: bool) -> bool:
        if retry_used or not self._should_hide_train_model(user_input):
            return False
        lowered = final_text.lower()
        if not lowered:
            return False
        return any(token in lowered for token in _ANALYSIS_DEFERRAL_HINTS)

    def _analysis_retry_guidance(self) -> str:
        return (
            "The user already asked you to do the analysis. "
            "Do not ask for permission or defer. "
            "Call the relevant tools now and answer decisively."
        )

    async def run_command(self, user_input: str) -> None:
        console.print(f"\n[dim]  ❯[/dim] [white]{user_input}[/white]\n")
        final_text = await self.run_turn(user_input)
        console.print()
        console.print(make_hannah_panel(final_text))
        console.print()

    async def _execute_tool_calls(
        self,
        messages_or_tool_calls: list,
        tool_calls: list | None = None,
        *,
        state: Any = None,
    ) -> list[dict[str, str]]:
        pending_tool_calls = tool_calls if tool_calls is not None else messages_or_tool_calls
        results = await asyncio.gather(
            *(self._invoke_tool(tool_call, state=state) for tool_call in pending_tool_calls),
            return_exceptions=True,
        )

        messages: list[dict[str, Any]] = []
        for tool_call, result in zip(pending_tool_calls, results):
            if isinstance(result, Exception):
                console.print(f"  [red]✗ {tool_call.function.name}: {result}[/red]")
                content = self._serialize_tool_message(
                    {
                        "status": "error",
                        "tool": tool_call.function.name,
                        "error": str(result),
                    },
                    tool_name=tool_call.function.name,
                )
            else:
                console.print(f"  [green]✓ {tool_call.function.name} returned[/green]")
                content = self._serialize_tool_message(result, tool_name=tool_call.function.name)

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_call.function.name,
                    "content": content,
                }
            )
            subagent_result_message = self._build_subagent_result_message(
                tool_name=tool_call.function.name,
                result=result if not isinstance(result, Exception) else None,
            )
            if subagent_result_message is not None:
                messages.append(subagent_result_message)
        return messages

    async def _invoke_tool(
        self,
        tool_call: Any,
        *,
        state: TurnState | None = None,
    ) -> dict[str, Any]:
        arguments = self._load_tool_arguments(tool_call.function.arguments)
        if state is not None:
            await self._publish_event(
                "tool_call_started",
                state,
                payload={"tool_name": tool_call.function.name, "args": arguments},
            )

        try:
            result = await self._call_tool_with_optional_state(tool_call, state=state)
        except Exception as exc:
            if state is not None:
                await self._publish_event(
                    "tool_call_finished",
                    state,
                    payload={
                        "tool_name": tool_call.function.name,
                        "status": "error",
                        "error": str(exc),
                    },
                )
            raise

        if state is not None:
            await self._publish_event(
                "tool_call_finished",
                state,
                payload={"tool_name": tool_call.function.name, "status": "ok"},
            )
        return result

    async def _call_tool_with_optional_state(
        self,
        tool_call: Any,
        *,
        state: TurnState | None = None,
    ) -> dict[str, Any]:
        if "state" in inspect.signature(self._call_tool).parameters:
            result = self._call_tool(tool_call, state=state)
        else:
            result = self._call_tool(tool_call)
        if inspect.isawaitable(result):
            return await result
        return result

    def _serialize_tool_message(self, payload: Any, *, tool_name: str) -> str:
        if isinstance(payload, str):
            return payload
        return json.dumps(self._compact_tool_payload(payload, tool_name=tool_name), default=str)

    def _compact_tool_payload(self, payload: Any, *, tool_name: str) -> Any:
        serialized = json.dumps(payload, default=str)
        if len(serialized) <= _TOOL_MESSAGE_MAX_CHARS:
            return payload
        if tool_name == "race_data" and isinstance(payload, dict):
            return self._summarize_race_data_payload(payload, raw_payload_chars=len(serialized))
        return {
            "summary": {
                "tool": tool_name,
                "raw_payload_chars": len(serialized),
                "message": "tool payload compacted for provider context",
            }
        }

    def _summarize_race_data_payload(
        self,
        payload: dict[str, Any],
        *,
        raw_payload_chars: int,
    ) -> dict[str, Any]:
        telemetry_counts = {
            "laps": self._record_count(payload.get("laps")),
            "stints": self._record_count(payload.get("stints")),
            "weather": self._record_count(payload.get("weather")),
        }
        available_telemetry = [
            name
            for name, count in telemetry_counts.items()
            if count > 0
        ]
        return {
            "session_info": payload.get("session_info", {}),
            "drivers": payload.get("drivers", []),
            "available_telemetry": available_telemetry,
            "telemetry_counts": telemetry_counts,
            "raw_payload_chars": raw_payload_chars,
            "note": "raw race_data payload compacted for provider context",
        }

    def _record_count(self, value: Any) -> int:
        if isinstance(value, list):
            return len(value)
        return 0

    async def _call_tool(self, tool_call, *, state: Any = None) -> dict:
        raw_arguments = tool_call.function.arguments
        arguments = self._load_tool_arguments(raw_arguments)
        normalizer = getattr(self.registry, "normalize_args", None)
        if callable(normalizer):
            arguments = normalizer(tool_call.function.name, arguments)
        else:
            arguments = self._normalize_tool_args_from_specs(tool_call.function.name, arguments)
        registry_call = self._resolve_registry_caller()
        if "state" in inspect.signature(registry_call).parameters:
            result = registry_call(tool_call.function.name, arguments, state=state)
        else:
            result = registry_call(tool_call.function.name, arguments)
        if inspect.isawaitable(result):
            return await result
        return result

    def _load_tool_arguments(self, raw_arguments: Any) -> dict[str, Any]:
        if isinstance(raw_arguments, dict):
            return raw_arguments
        try:
            parsed = json.loads(raw_arguments)
        except (TypeError, json.JSONDecodeError):
            return {}
        return parsed if isinstance(parsed, dict) else {}

    def _resolve_registry_caller(self) -> Any:
        execute = getattr(self.registry, "execute", None)
        if callable(execute):
            return execute
        call = getattr(self.registry, "call", None)
        if callable(call):
            return call
        raise AttributeError("registry must define either execute() or call()")

    def _normalize_tool_args_from_specs(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        for tool in self.tools:
            function = tool.get("function", {})
            if not isinstance(function, dict):
                continue
            if function.get("name") != name:
                continue
            parameters = function.get("parameters")
            return normalize_tool_args(name, args, parameters=parameters)
        return args

    def _coerce_first_message(self, response: object) -> Any:
        return coerce_provider_message(response)

    def _extract_first_message(self, response: object) -> object | None:
        return extract_first_message(response)

    def _message_to_adapter(self, message: object) -> Any:
        return message_to_provider_message(message)

    def _payload_to_message(self, message: dict[str, Any]) -> Any:
        return payload_to_provider_message(message)

    def _coerce_tool_calls(self, tool_calls_payload: Any) -> list[Any]:
        return coerce_tool_calls(tool_calls_payload)

    def _coerce_payload(self, payload: Any) -> dict[str, Any] | None:
        return coerce_payload(payload)

    def _normalize_message_content(self, content: Any, *, has_tool_calls: bool) -> str:
        return normalize_message_content(content, has_tool_calls=has_tool_calls)

    def _flatten_content_blocks(self, blocks: list[Any]) -> str:
        return flatten_content_blocks(blocks)

    async def _publish_event(
        self,
        event_type: str,
        state: TurnState,
        *,
        payload: dict[str, Any] | None = None,
        worker_id: str | None = None,
    ) -> None:
        await self.event_bus.publish(
            EventEnvelope.create(
                event_type=event_type,
                session_id=state.session_id,
                message_id=state.message_id,
                worker_id=worker_id,
                payload=payload or {},
            )
        )

    async def _handle_spawn_tool(
        self,
        task: str,
        system_prompt: str,
        allowed_tools: list[str],
        result_contract: dict[str, Any],
        *,
        state: TurnState | None = None,
    ) -> dict[str, Any]:
        if self._worker_runtime is None:
            raise ValueError("spawn tool is not available")

        spec = WorkerSpec(
            worker_id=make_worker_id("worker"),
            task=task,
            system_prompt=system_prompt,
            allowed_tools=list(allowed_tools),
            result_contract=dict(result_contract),
        )
        parent_session_id = state.session_id if state is not None else "default"
        return await self._worker_runtime.run_worker(
            spec,
            parent_session_id=parent_session_id,
        )

    def _build_subagent_result_message(
        self,
        *,
        tool_name: str,
        result: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if tool_name != "spawn" or not isinstance(result, dict):
            return None

        worker_id = result.get("worker_id")
        if not isinstance(worker_id, str) or not worker_id:
            return None

        return {
            "role": "system",
            "name": "subagent_result",
            "worker_id": worker_id,
            "content": json.dumps(result, default=str),
        }


AgentCore = AgentLoop
