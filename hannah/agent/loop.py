"""Main Hannah tool-using loop."""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from typing import Any

from hannah.agent.memory import Memory
from hannah.agent.persona import HANNAH_PERSONA
from hannah.agent.tool_registry import ToolRegistry, normalize_tool_args
from hannah.cli.format import make_hannah_panel
from hannah.config.loader import load_config
from hannah.providers.registry import ProviderRegistry
from hannah.utils.console import Console

console = Console()
_TOOL_MESSAGE_MAX_CHARS = 20_000
_TRAIN_MODEL_TOOL_NAME = "train_model"
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


@dataclass(frozen=True)
class _FunctionAdapter:
    name: str
    arguments: str


@dataclass(frozen=True)
class _ToolCallAdapter:
    id: str
    function: _FunctionAdapter
    type: str = "function"

    def model_dump(self) -> dict:
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

    def model_dump(self) -> dict:
        payload: dict = {"role": self.role, "content": self.content}
        if self.tool_calls:
            payload["tool_calls"] = [call.model_dump() for call in self.tool_calls]
        if self.name is not None:
            payload["name"] = self.name
        if self.tool_call_id is not None:
            payload["tool_call_id"] = self.tool_call_id
        return payload


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
        self.tools = self.registry.get_tool_specs()
        self.provider = provider or ProviderRegistry.from_config(self.config)

    async def run_turn(self, user_input: str) -> str:
        turn_tools = self._select_tools_for_turn(user_input)
        analysis_retry_used = False
        messages = [
            {"role": "system", "content": HANNAH_PERSONA},
        ]
        dynamic_guidance = self._dynamic_turn_guidance(user_input)
        if dynamic_guidance is not None:
            messages.append({"role": "system", "content": dynamic_guidance})
        messages.extend([*self.memory.get_recent(n=10), {"role": "user", "content": user_input}])

        while True:
            response = await self.provider.complete(
                messages=messages,
                tools=turn_tools if turn_tools else None,
                temperature=self.config.agent.temperature,
                max_tokens=self.config.agent.max_tokens,
            )

            message = self._coerce_first_message(response)
            tool_calls = getattr(message, "tool_calls", None)

            if tool_calls:
                for tool_call in tool_calls:
                    console.print(
                        f"  [dim cyan]◆ calling tool:[/dim cyan] [cyan]{tool_call.function.name}[/cyan]"
                    )
                messages.append(message.model_dump())
                messages.extend(await self._execute_tool_calls(tool_calls))
                continue

            final_text = message.content or ""
            if self._should_retry_analysis_turn(
                user_input=user_input,
                final_text=final_text,
                retry_used=analysis_retry_used,
            ):
                messages.append(message.model_dump())
                messages.append({"role": "system", "content": self._analysis_retry_guidance()})
                analysis_retry_used = True
                continue
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

    async def _execute_tool_calls(self, tool_calls: list) -> list[dict[str, str]]:
        results = await asyncio.gather(
            *(self._call_tool(tool_call) for tool_call in tool_calls),
            return_exceptions=True,
        )

        messages: list[dict[str, str]] = []
        for tool_call, result in zip(tool_calls, results):
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
        return messages

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

    async def _call_tool(self, tool_call) -> dict:
        raw_arguments = tool_call.function.arguments
        arguments = self._load_tool_arguments(raw_arguments)
        normalizer = getattr(self.registry, "normalize_args", None)
        if callable(normalizer):
            arguments = normalizer(tool_call.function.name, arguments)
        else:
            arguments = self._normalize_tool_args_from_specs(tool_call.function.name, arguments)
        return await self.registry.call(tool_call.function.name, arguments)

    def _load_tool_arguments(self, raw_arguments: Any) -> dict[str, Any]:
        if isinstance(raw_arguments, dict):
            return raw_arguments
        try:
            parsed = json.loads(raw_arguments)
        except (TypeError, json.JSONDecodeError):
            return {}
        return parsed if isinstance(parsed, dict) else {}

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

    def _coerce_first_message(self, response: object) -> _MessageAdapter:
        message = self._extract_first_message(response)
        if message is None:
            return _MessageAdapter(role="assistant", content="I could not parse the model response.")
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
            return _MessageAdapter(role="assistant", content="I could not parse the model response.")
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


AgentCore = AgentLoop
