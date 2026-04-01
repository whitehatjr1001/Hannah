"""Command and race context objects for the agent runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from hannah.agent import prompts


@dataclass(frozen=True)
class AgentCommandContext:
    command: str
    race: str | None = None
    year: int | None = None
    driver: str | None = None
    laps: int | None = None
    weather: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class RaceContext:
    race: str
    year: int
    laps: int
    weather: str
    drivers: list[str]
    race_data: dict | None = None


@dataclass(frozen=True, slots=True)
class MainAgentContext:
    persona: str
    user_input: str
    recent_messages: tuple[Mapping[str, Any], ...] = field(default_factory=tuple)
    dynamic_guidance: str | None = None
    bootstrap_docs: tuple[str, ...] = field(default_factory=tuple)
    memory_context: str | None = None
    skills_summary_hook: Callable[[], str] | str | None = None


class NanobotContextBuilder:
    """Build nanobot-style context blocks for the main Hannah runtime turn."""

    def build_main_turn(self, context: MainAgentContext) -> list[dict[str, Any]]:
        skills_summary = self._resolve_skills_summary(context.skills_summary_hook)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": prompts.build_identity_runtime_block(dynamic_guidance=context.dynamic_guidance)},
            {"role": "system", "content": prompts.build_bootstrap_docs_block(context.bootstrap_docs)},
            {
                "role": "system",
                "content": prompts.build_memory_context_block(
                    context.recent_messages,
                    memory_context=context.memory_context,
                ),
            },
            {"role": "system", "content": prompts.build_skills_summary_block(skills_summary)},
            {"role": "system", "content": prompts.build_hannah_persona_block(context.persona)},
        ]
        messages.extend(self.build_main_messages(context.recent_messages))
        messages.append({"role": "user", "content": context.user_input})
        return messages

    def build_main_messages(
        self,
        messages: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        return [self._coerce_message(message) for message in messages]

    def _resolve_skills_summary(self, hook: Any | None) -> str | None:
        if hook is None:
            return None
        if callable(hook):
            try:
                result = hook()
            except Exception as exc:
                return f"skills summary hook failed: {exc}"
            if result is None:
                return None
            return str(result)
        return str(hook)

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


RuntimeContextBuilder = NanobotContextBuilder
