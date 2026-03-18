"""LiteLLM-backed provider wrapper."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from hannah.config.provider_setup import detect_provider_from_model, get_provider_preset
from hannah.config.schema import AppConfig
from hannah.providers.local_fallback import DeterministicFallbackPlanner, LocalCompletion

_ALLOWED_MESSAGE_KEYS = frozenset({"role", "content", "tool_calls", "tool_call_id", "name"})


@dataclass(frozen=True)
class LiteLLMProvider:
    """Thin provider wrapper so the agent does not depend on LiteLLM directly."""

    config: AppConfig

    async def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        temperature: float,
        max_tokens: int,
    ) -> object:
        if self._force_local():
            return self._local_complete(messages=messages, tools=tools)

        try:
            import litellm
        except Exception:
            return self._local_complete(messages=messages, tools=tools)

        litellm.suppress_debug_info = True
        if not self._hosted_credentials_available():
            return self._local_complete(messages=messages, tools=tools)

        if self.config.rlm.enabled or os.getenv("HANNAH_RLM_API_BASE"):
            litellm.api_base = os.getenv("HANNAH_RLM_API_BASE", self.config.rlm.api_base)
            litellm.api_key = os.getenv("HANNAH_RLM_API_KEY", self.config.rlm.api_key)
        try:
            return await litellm.acompletion(
                model=self.config.agent.model,
                messages=self._sanitize_messages(messages),
                tools=tools,
                tool_choice="auto" if tools else None,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception:
            return self._local_complete(messages=messages, tools=tools)

    def _local_complete(self, messages: list[dict], tools: list[dict] | None) -> LocalCompletion:
        planner = DeterministicFallbackPlanner()
        return planner.complete(messages=messages, tools=tools)

    def _sanitize_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        sanitized: list[dict[str, Any]] = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            clean = {key: value for key, value in message.items() if key in _ALLOWED_MESSAGE_KEYS}
            if clean.get("role") == "assistant" and "content" not in clean:
                clean["content"] = ""
            sanitized.append(clean)
        return sanitized

    def _force_local(self) -> bool:
        value = os.getenv("HANNAH_FORCE_LOCAL_PROVIDER", "").strip().lower()
        return value in {"1", "true", "yes", "on"}

    def _hosted_credentials_available(self) -> bool:
        if self.config.rlm.enabled or os.getenv("HANNAH_RLM_API_BASE"):
            return True

        provider_name = detect_provider_from_model(self.config.agent.model)
        if provider_name is None:
            return True
        preset = get_provider_preset(provider_name)
        return any(os.getenv(key) for key in preset.api_key_env_vars)
