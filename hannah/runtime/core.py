from __future__ import annotations

import asyncio
import inspect
import json
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from hannah.runtime.bus import AsyncEventBus
from hannah.runtime.context import RuntimeContextBuilder
from hannah.runtime.events import EventEnvelope
from hannah.runtime.turn_state import TurnState
from hannah.utils.console import Console

_TOOL_MESSAGE_MAX_CHARS = 20_000


@dataclass(frozen=True)
class _FunctionAdapter:
    name: str
    arguments: str


@dataclass(frozen=True)
class _ToolCallAdapter:
    id: str
    function: _FunctionAdapter
    type: str = "function"

    def model_dump(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "function": {"name": self.function.name, "arguments": self.function.arguments},
        }


@dataclass(frozen=True)
class _MessageAdapter:
    role: str
    content: str
    tool_calls: list[_ToolCallAdapter] | None = None
    name: str | None = None
    tool_call_id: str | None = None

    def model_dump(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            payload["tool_calls"] = [call.model_dump() for call in self.tool_calls]
        if self.name is not None:
            payload["name"] = self.name
        if self.tool_call_id is not None:
            payload["tool_call_id"] = self.tool_call_id
        return payload


RetryPolicy = Callable[[str, bool], bool]
ToolCaller = Callable[[Any], Awaitable[dict[str, Any]]]
ExecuteToolCallsHook = Callable[..., Awaitable[list[dict[str, str]]]]


class RuntimeCore:
    def __init__(
        self,
        provider: object,
        registry: object,
        event_bus: AsyncEventBus | None = None,
        *,
        memory: object | None = None,
        context_builder: RuntimeContextBuilder | None = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        console: Console | None = None,
        allow_spawn_tool: bool = True,
    ) -> None:
        self.provider = provider
        self.registry = registry
        self.event_bus = event_bus or AsyncEventBus()
        self.memory = memory
        self.context_builder = context_builder or RuntimeContextBuilder()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.console = console or Console()
        self._worker_runtime: object | None = None
        if allow_spawn_tool:
            runtime_binding = getattr(self.registry, "with_runtime_tools", None)
            if callable(runtime_binding):
                from hannah.agent.worker_runtime import WorkerRuntime

                self._worker_runtime = WorkerRuntime(
                    provider=self.provider,
                    registry=registry,
                    event_bus=self.event_bus,
                    context_builder=self.context_builder,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    console=self.console,
                )
                self.registry = runtime_binding({"spawn": self._handle_spawn_tool})

    async def run_turn(
        self,
        messages: list[dict[str, Any]],
        *,
        session_id: str = "default",
        turn_tools: list[dict[str, Any]] | None = None,
        should_retry: RetryPolicy | None = None,
        retry_guidance: str | None = None,
        execute_tool_calls: ExecuteToolCallsHook | None = None,
    ) -> dict[str, str]:
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
            tools = self._resolve_turn_tools(turn_tools)
            await self._publish_event(
                "provider_request_started",
                state,
                payload={
                    "message_count": len(state.messages),
                    "tool_names": [tool["function"]["name"] for tool in tools],
                },
            )
            try:
                response = await self.provider.complete(
                    messages=state.snapshot_messages(),
                    tools=tools if tools else None,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
            except Exception as exc:
                await self._publish_event(
                    "error_emitted",
                    state,
                    payload={"stage": "provider", "error": str(exc)},
                )
                raise

            message = self._coerce_first_message(response)
            tool_calls = getattr(message, "tool_calls", None)
            await self._publish_event(
                "provider_response_received",
                state,
                payload={"has_tool_calls": bool(tool_calls)},
            )

            if tool_calls:
                for tool_call in tool_calls:
                    self.console.print(
                        f"  [dim cyan]◆ calling tool:[/dim cyan] [cyan]{tool_call.function.name}[/cyan]"
                    )
                state.append_message(message.model_dump())
                if execute_tool_calls is None:
                    tool_messages = await self._execute_tool_calls(tool_calls, state=state)
                else:
                    tool_messages = await execute_tool_calls(tool_calls, state=state)
                state.extend_messages(tool_messages)
                continue

            final_text = message.content or ""
            if (
                should_retry is not None
                and retry_guidance is not None
                and should_retry(final_text, retry_used)
            ):
                state.append_message(message.model_dump())
                state.append_message({"role": "system", "content": retry_guidance})
                retry_used = True
                continue

            await self._publish_event(
                "final_answer_emitted",
                state,
                payload={"content": final_text},
            )
            return {"role": "assistant", "content": final_text}

    def _resolve_turn_tools(
        self,
        turn_tools: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]]:
        if turn_tools is not None:
            return list(turn_tools)
        tool_specs = getattr(self.registry, "get_tool_specs", None)
        if callable(tool_specs):
            return list(tool_specs())
        return []

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

    async def _execute_tool_calls(
        self,
        tool_calls: list[Any],
        *,
        state: TurnState | None = None,
        call_tool: ToolCaller | None = None,
    ) -> list[dict[str, str]]:
        results = await asyncio.gather(
            *(
                self._invoke_tool(tool_call, state=state, call_tool=call_tool)
                for tool_call in tool_calls
            ),
            return_exceptions=True,
        )

        messages: list[dict[str, str]] = []
        for tool_call, result in zip(tool_calls, results):
            if isinstance(result, Exception):
                self.console.print(f"  [red]✗ {tool_call.function.name}: {result}[/red]")
                content = self._serialize_tool_message(
                    {
                        "status": "error",
                        "tool": tool_call.function.name,
                        "error": str(result),
                    },
                    tool_name=tool_call.function.name,
                )
            else:
                self.console.print(f"  [green]✓ {tool_call.function.name} returned[/green]")
                content = self._serialize_tool_message(result, tool_name=tool_call.function.name)

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_call.function.name,
                    "content": content,
                }
            )
        return messages

    async def _invoke_tool(
        self,
        tool_call: Any,
        *,
        state: TurnState | None = None,
        call_tool: ToolCaller | None = None,
    ) -> dict[str, Any]:
        raw_arguments = tool_call.function.arguments
        arguments = self._load_tool_arguments(raw_arguments)

        if state is not None:
            await self._publish_event(
                "tool_call_started",
                state,
                payload={"tool_name": tool_call.function.name, "args": arguments},
            )

        try:
            if call_tool is None:
                result = await self._call_tool(tool_call, state=state)
            else:
                result = await self._call_tool_with_optional_state(
                    call_tool,
                    tool_call,
                    state=state,
                )
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

    async def _call_tool(self, tool_call: Any, *, state: TurnState | None = None) -> dict[str, Any]:
        raw_arguments = tool_call.function.arguments
        arguments = self._load_tool_arguments(raw_arguments)
        normalizer = getattr(self.registry, "normalize_args", None)
        if callable(normalizer):
            arguments = normalizer(tool_call.function.name, arguments)
        return await self._call_registry(tool_call.function.name, arguments, state=state)

    async def _call_registry(
        self,
        name: str,
        args: dict[str, Any],
        *,
        state: TurnState | None = None,
    ) -> dict[str, Any]:
        registry_call = getattr(self.registry, "call")
        if "state" in inspect.signature(registry_call).parameters:
            result = registry_call(name, args, state=state)
        else:
            result = registry_call(name, args)
        if inspect.isawaitable(result):
            return await result
        return result

    async def _call_tool_with_optional_state(
        self,
        tool_caller: ToolCaller,
        tool_call: Any,
        *,
        state: TurnState | None = None,
    ) -> dict[str, Any]:
        if "state" in inspect.signature(tool_caller).parameters:
            result = tool_caller(tool_call, state=state)
        else:
            result = tool_caller(tool_call)
        if inspect.isawaitable(result):
            return await result
        return result

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

        from hannah.agent.worker_runtime import WorkerSpec, make_worker_id

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

    def _load_tool_arguments(self, raw_arguments: Any) -> dict[str, Any]:
        if isinstance(raw_arguments, dict):
            return raw_arguments
        try:
            parsed = json.loads(raw_arguments)
        except (TypeError, json.JSONDecodeError):
            return {}
        return parsed if isinstance(parsed, dict) else {}

    def _normalize_tool_args_from_specs(
        self,
        name: str,
        args: dict[str, Any],
    ) -> dict[str, Any]:
        del name
        return args

    def _coerce_first_message(self, response: object) -> _MessageAdapter:
        message = self._extract_first_message(response)
        if message is None:
            return _MessageAdapter(
                role="assistant",
                content="I could not parse the model response.",
            )
        return self._message_to_adapter(message)

    def _extract_first_message(self, response: object) -> object | None:
        if hasattr(response, "choices") and getattr(response, "choices"):
            choice = response.choices[0]
            message = getattr(choice, "message", None)
            if message is not None:
                return message

        if isinstance(response, dict):
            choices = response.get("choices", [])
            if choices:
                first_choice = choices[0]
                if isinstance(first_choice, dict):
                    return first_choice.get("message")
                if hasattr(first_choice, "message"):
                    return getattr(first_choice, "message", None)
        return None

    def _message_to_adapter(self, message: object) -> _MessageAdapter:
        if isinstance(message, dict):
            payload = message
        elif hasattr(message, "model_dump"):
            payload = message.model_dump()
        else:
            payload = {
                "role": getattr(message, "role", "assistant"),
                "content": getattr(message, "content", ""),
                "tool_calls": getattr(message, "tool_calls", None),
                "name": getattr(message, "name", None),
                "tool_call_id": getattr(message, "tool_call_id", None),
            }

        if not isinstance(payload, dict):
            return _MessageAdapter(
                role="assistant",
                content="I could not parse the model response.",
            )
        return self._payload_to_message(payload)

    def _payload_to_message(self, message: dict[str, Any]) -> _MessageAdapter:
        tool_calls = self._coerce_tool_calls(message.get("tool_calls"))
        content = self._normalize_message_content(
            message.get("content"),
            has_tool_calls=bool(tool_calls),
        )
        return _MessageAdapter(
            role=str(message.get("role", "assistant")),
            content=content,
            tool_calls=tool_calls or None,
            name=message.get("name"),
            tool_call_id=message.get("tool_call_id"),
        )

    def _coerce_tool_calls(self, tool_calls_payload: Any) -> list[_ToolCallAdapter]:
        tool_calls: list[_ToolCallAdapter] = []
        for index, raw_call in enumerate(tool_calls_payload or []):
            call = self._coerce_payload(raw_call)
            if call is None:
                continue
            function = self._coerce_payload(call.get("function"))
            if function is None:
                continue
            name = str(function.get("name", "")).strip()
            if not name:
                continue
            arguments = function.get("arguments", "{}")
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments, default=str)
            tool_calls.append(
                _ToolCallAdapter(
                    id=str(call.get("id", f"dict-tool-call-{index + 1}")),
                    function=_FunctionAdapter(name=name, arguments=arguments),
                    type=str(call.get("type", "function")),
                )
            )
        return tool_calls

    def _coerce_payload(self, payload: Any) -> dict[str, Any] | None:
        if isinstance(payload, dict):
            return payload
        if hasattr(payload, "model_dump"):
            dumped = payload.model_dump()
            return dumped if isinstance(dumped, dict) else None
        if payload is None:
            return None
        result: dict[str, Any] = {}
        for key in ("id", "type", "function", "role", "content", "tool_calls", "name", "tool_call_id"):
            if hasattr(payload, key):
                result[key] = getattr(payload, key)
        return result or None

    def _normalize_message_content(self, content: Any, *, has_tool_calls: bool) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            if has_tool_calls:
                return ""
            return self._flatten_content_blocks(content)
        if isinstance(content, dict):
            if has_tool_calls:
                return ""
            return self._flatten_content_blocks([content])
        if has_tool_calls:
            return ""
        return str(content)

    def _flatten_content_blocks(self, blocks: list[Any]) -> str:
        parts: list[str] = []
        for block in blocks:
            if isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
                    continue
                if isinstance(text, dict):
                    nested = text.get("value") or text.get("text")
                    if isinstance(nested, str):
                        parts.append(nested)
                        continue
                value = block.get("value")
                if isinstance(value, str):
                    parts.append(value)
                    continue
            else:
                text = getattr(block, "text", None)
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(part for part in parts if part).strip()
