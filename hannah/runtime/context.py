from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


@dataclass(frozen=True, slots=True)
class MainAgentContext:
    persona: str
    user_input: str
    recent_messages: tuple[Mapping[str, Any], ...] = field(default_factory=tuple)
    dynamic_guidance: str | None = None


class RuntimeContextBuilder:
    """Build runtime message stacks with instruction, context, and input kept separate."""

    def build_main_turn(self, context: MainAgentContext) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = [{"role": "system", "content": context.persona}]
        if context.dynamic_guidance:
            messages.append({"role": "system", "content": context.dynamic_guidance})
        messages.extend(self.build_main_messages(context.recent_messages))
        messages.append({"role": "user", "content": context.user_input})
        return messages

    def build_main_messages(
        self,
        messages: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        return [self._coerce_message(message) for message in messages]

    def _coerce_message(self, message: Mapping[str, Any] | Any) -> dict[str, Any]:
        if isinstance(message, dict):
            return dict(message)
        if hasattr(message, "model_dump"):
            dumped = message.model_dump()
            if isinstance(dumped, dict):
                return dumped
        if isinstance(message, Mapping):
            return dict(message)

        payload: dict[str, Any] = {}
        for key in ("role", "content", "tool_calls", "name", "tool_call_id"):
            if hasattr(message, key):
                payload[key] = getattr(message, key)
        if payload:
            return payload
        raise TypeError(f"Unsupported runtime message type: {type(message).__name__}")
