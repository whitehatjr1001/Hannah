"""Main Hannah tool-using loop."""

from __future__ import annotations

import re
from typing import Any

from hannah.agent.memory import Memory
from hannah.agent.persona import HANNAH_PERSONA
from hannah.agent.tool_registry import ToolRegistry, normalize_tool_args
from hannah.cli.format import make_hannah_panel
from hannah.config.loader import load_config
from hannah.providers.registry import ProviderRegistry
from hannah.runtime import AsyncEventBus, MainAgentContext, RuntimeContextBuilder, RuntimeCore
from hannah.utils.console import Console

console = Console()
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
        self.event_bus = AsyncEventBus()
        self.context_builder = RuntimeContextBuilder()
        self.runtime = RuntimeCore(
            provider=self.provider,
            registry=self.registry,
            event_bus=self.event_bus,
            memory=self.memory,
            context_builder=self.context_builder,
            temperature=self.config.agent.temperature,
            max_tokens=self.config.agent.max_tokens,
            console=console,
        )

    async def run_turn(self, user_input: str) -> str:
        turn_tools = self._select_tools_for_turn(user_input)
        messages = self.context_builder.build_main_turn(
            MainAgentContext(
                persona=HANNAH_PERSONA,
                dynamic_guidance=self._dynamic_turn_guidance(user_input),
                recent_messages=tuple(self.memory.get_recent(n=10)),
                user_input=user_input,
            )
        )
        reply = await self.runtime.run_turn(
            messages=messages,
            session_id="default",
            turn_tools=turn_tools,
            should_retry=lambda final_text, retry_used: self._should_retry_analysis_turn(
                user_input=user_input,
                final_text=final_text,
                retry_used=retry_used,
            ),
            retry_guidance=self._analysis_retry_guidance(),
            execute_tool_calls=self._execute_tool_calls,
        )
        final_text = reply.get("content", "")
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
        tool_calls: list,
        *,
        state: Any = None,
    ) -> list[dict[str, str]]:
        return await self.runtime._execute_tool_calls(
            tool_calls,
            state=state,
            call_tool=self._call_tool,
        )

    def _serialize_tool_message(self, payload: Any, *, tool_name: str) -> str:
        return self.runtime._serialize_tool_message(payload, tool_name=tool_name)

    def _compact_tool_payload(self, payload: Any, *, tool_name: str) -> Any:
        return self.runtime._compact_tool_payload(payload, tool_name=tool_name)

    def _summarize_race_data_payload(
        self,
        payload: dict[str, Any],
        *,
        raw_payload_chars: int,
    ) -> dict[str, Any]:
        return self.runtime._summarize_race_data_payload(
            payload,
            raw_payload_chars=raw_payload_chars,
        )

    def _record_count(self, value: Any) -> int:
        return self.runtime._record_count(value)

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
        return self.runtime._load_tool_arguments(raw_arguments)

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
        return self.runtime._coerce_first_message(response)

    def _extract_first_message(self, response: object) -> object | None:
        return self.runtime._extract_first_message(response)

    def _message_to_adapter(self, message: object) -> Any:
        return self.runtime._message_to_adapter(message)

    def _payload_to_message(self, message: dict[str, Any]) -> Any:
        return self.runtime._payload_to_message(message)

    def _coerce_tool_calls(self, tool_calls_payload: Any) -> list[Any]:
        return self.runtime._coerce_tool_calls(tool_calls_payload)

    def _coerce_payload(self, payload: Any) -> dict[str, Any] | None:
        return self.runtime._coerce_payload(payload)

    def _normalize_message_content(self, content: Any, *, has_tool_calls: bool) -> str:
        return self.runtime._normalize_message_content(content, has_tool_calls=has_tool_calls)

    def _flatten_content_blocks(self, blocks: list[Any]) -> str:
        return self.runtime._flatten_content_blocks(blocks)


AgentCore = AgentLoop
