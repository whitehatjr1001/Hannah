"""Focused seam tests for sub-agent aggregation and provider registry metadata."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

import hannah.agent.subagents as subagents
from hannah.agent.context import RaceContext
from hannah.agent.worker_runtime import WorkerSpec
from hannah.config.loader import load_config
from hannah.providers.litellm_provider import LiteLLMProvider
from hannah.providers.registry import ProviderRegistry


def test_spawn_all_keeps_successes_and_contains_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    class _SimAgent:
        async def run(self, ctx: RaceContext) -> subagents.SubAgentResult:
            del ctx
            return subagents.SubAgentResult(agent="sim_agent", success=True, data={"winner": "VER"})

    class _StrategyAgent:
        async def run(self, ctx: RaceContext) -> subagents.SubAgentResult:
            del ctx
            return subagents.SubAgentResult(
                agent="strategy_agent",
                success=False,
                error="provider timeout",
            )

    class _PredictAgent:
        async def run(self, ctx: RaceContext) -> subagents.SubAgentResult:
            del ctx
            raise RuntimeError("prediction crash")

    class _RivalAgent:
        def __init__(self, driver: str) -> None:
            self.driver = driver

        async def run(self, ctx: RaceContext) -> subagents.SubAgentResult:
            del ctx
            if self.driver == "NOR":
                return subagents.SubAgentResult(
                    agent="rival_nor",
                    success=True,
                    data={"decision": "cover undercut"},
                )
            return subagents.SubAgentResult(
                agent="rival_lec",
                success=False,
                error="missing state",
            )

    monkeypatch.setattr(subagents, "SimAgent", _SimAgent)
    monkeypatch.setattr(subagents, "StrategyAgent", _StrategyAgent)
    monkeypatch.setattr(subagents, "PredictAgent", _PredictAgent)
    monkeypatch.setattr(subagents, "RivalAgent", _RivalAgent)
    monkeypatch.setattr(subagents.console, "print", lambda *args, **kwargs: None)

    ctx = RaceContext(
        race="bahrain",
        year=2025,
        laps=57,
        weather="dry",
        drivers=["VER", "NOR", "LEC"],
        race_data={"session_info": {"current_lap": 20}},
    )
    output = asyncio.run(subagents.spawn_all(ctx))

    assert output == {
        "sim_agent": {"winner": "VER"},
        "rival_nor": {"decision": "cover undercut"},
    }


def test_provider_registry_describe_reflects_env_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    config = load_config(path=Path("/Users/deepedge/Desktop/projects/files/config.yaml"))

    monkeypatch.delenv("HANNAH_FORCE_LOCAL_PROVIDER", raising=False)
    monkeypatch.delenv("HANNAH_RLM_API_BASE", raising=False)
    baseline = ProviderRegistry.describe(config)
    assert baseline.provider_name == "litellm"
    assert baseline.model == config.agent.model
    assert baseline.local_fallback_enabled is False
    assert baseline.rlm_enabled is config.rlm.enabled

    monkeypatch.setenv("HANNAH_FORCE_LOCAL_PROVIDER", "true")
    monkeypatch.setenv("HANNAH_RLM_API_BASE", "http://localhost:9001")
    overridden = ProviderRegistry.describe(config)
    assert overridden.local_fallback_enabled is True
    assert overridden.rlm_enabled is True


def test_provider_registry_returns_litellm_provider_instance() -> None:
    config = load_config(path=Path("/Users/deepedge/Desktop/projects/files/config.yaml"))
    provider = ProviderRegistry.from_config(config)
    assert isinstance(provider, LiteLLMProvider)


def test_build_legacy_worker_specs_maps_fixed_roster_to_bounded_specs() -> None:
    ctx = RaceContext(
        race="bahrain",
        year=2025,
        laps=57,
        weather="dry",
        drivers=["VER", "NOR", "LEC"],
        race_data={"session_info": {"current_lap": 20}},
    )

    specs = subagents.build_legacy_worker_specs(ctx)

    assert all(isinstance(spec, WorkerSpec) for spec in specs)
    assert [spec.worker_id for spec in specs] == [
        "sim_agent",
        "strategy_agent",
        "predict_agent",
        "rival_nor",
        "rival_lec",
    ]
    assert all(spec.allowed_tools for spec in specs)
    assert all("spawn" not in spec.allowed_tools for spec in specs)
