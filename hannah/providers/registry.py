"""Provider selection for Hannah."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol

from hannah.config.schema import AppConfig
from hannah.providers.litellm_provider import LiteLLMProvider


class CompletionProvider(Protocol):
    """Provider contract consumed by the agent loop and sub-agents."""

    async def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        temperature: float,
        max_tokens: int,
    ) -> object:
        ...


@dataclass(frozen=True)
class ProviderSelection:
    """Resolved provider metadata for diagnostics and tests."""

    provider_name: str
    model: str
    local_fallback_enabled: bool
    rlm_enabled: bool


class ProviderRegistry:
    """Resolve the runtime provider from config."""

    @staticmethod
    def from_config(config: AppConfig) -> CompletionProvider:
        return LiteLLMProvider(config=config)

    @staticmethod
    def describe(config: AppConfig) -> ProviderSelection:
        force_local = os.getenv("HANNAH_FORCE_LOCAL_PROVIDER", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        return ProviderSelection(
            provider_name="litellm",
            model=config.agent.model,
            local_fallback_enabled=force_local,
            rlm_enabled=config.rlm.enabled or bool(os.getenv("HANNAH_RLM_API_BASE")),
        )
