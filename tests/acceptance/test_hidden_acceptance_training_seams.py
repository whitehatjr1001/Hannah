"""Masked acceptance scenarios for training and provider seams."""

from __future__ import annotations

import asyncio
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pytest

from hannah.agent.tool_registry import ToolRegistry
from hannah.config.loader import load_config
from hannah.providers.litellm_provider import LiteLLMProvider


@dataclass(frozen=True)
class HiddenTrainingScenario:
    scenario_id: str
    model_name: str
    expected_suffix: str


TRAINING_SCENARIOS: Final[tuple[HiddenTrainingScenario, ...]] = (
    HiddenTrainingScenario("HACC_T01", "tyre_model", "models/saved/tyre_deg_v1.pkl"),
    HiddenTrainingScenario("HACC_T02", "laptime_model", "models/saved/laptime_v1.pkl"),
    HiddenTrainingScenario("HACC_T03", "pit_rl", "models/saved/pit_rl_v1.zip"),
    HiddenTrainingScenario("HACC_T04", "pit_policy_q", "models/saved/pit_policy_q_v1.pkl"),
    HiddenTrainingScenario("HACC_T05", "winner_ensemble", "models/saved/winner_ensemble_v1.pkl"),
)


def _run_tool(registry: ToolRegistry, name: str, args: dict) -> dict:
    return asyncio.run(registry.call(name, args))


@pytest.mark.parametrize(
    "scenario",
    TRAINING_SCENARIOS,
    ids=[scenario.scenario_id for scenario in TRAINING_SCENARIOS],
)
def test_hidden_training_dispatch_contract(scenario: HiddenTrainingScenario) -> None:
    registry = ToolRegistry()
    result = _run_tool(
        registry,
        "train_model",
        {"model_name": scenario.model_name, "years": [2023, 2024], "races": ["bahrain"]},
    )

    assert set(result.keys()) == {"saved"}
    saved_path = Path(result["saved"])
    assert saved_path.as_posix().endswith(scenario.expected_suffix)
    assert saved_path.parent.exists()


def test_hidden_training_dispatch_all_models_contract() -> None:
    registry = ToolRegistry()
    result = _run_tool(
        registry,
        "train_model",
        {"model_name": "all", "years": [2022, 2023, 2024], "races": ["bahrain", "monaco"]},
    )

    assert set(result.keys()) == {"saved"}
    assert set(result["saved"].keys()) == {
        "tyre_model",
        "laptime_model",
        "pit_rl",
        "pit_policy_q",
        "winner_ensemble",
    }
    assert result["saved"]["tyre_model"].endswith("models/saved/tyre_deg_v1.pkl")
    assert result["saved"]["laptime_model"].endswith("models/saved/laptime_v1.pkl")
    assert result["saved"]["pit_rl"].endswith("models/saved/pit_rl_v1.zip")
    assert result["saved"]["pit_policy_q"].endswith("models/saved/pit_policy_q_v1.pkl")
    assert result["saved"]["winner_ensemble"].endswith("models/saved/winner_ensemble_v1.pkl")


def test_hidden_config_env_model_resolution_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HANNAH_MODEL", "openai/rlm-local")
    config = load_config(path=Path("config.yaml"))
    assert config.agent.model == "openai/rlm-local"


def test_hidden_provider_rlm_seam_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_litellm = types.SimpleNamespace()
    fake_litellm.api_base = None
    fake_litellm.api_key = None

    async def _fake_completion(**kwargs):
        return kwargs

    fake_litellm.acompletion = _fake_completion

    monkeypatch.setitem(sys.modules, "litellm", fake_litellm)
    monkeypatch.setenv("HANNAH_RLM_API_BASE", "http://localhost:9001")
    monkeypatch.setenv("HANNAH_RLM_API_KEY", "none")

    config = load_config(path=Path("config.yaml"))
    provider = LiteLLMProvider(config=config)
    result = asyncio.run(
        provider.complete(
            messages=[{"role": "user", "content": "health check"}],
            tools=None,
            temperature=0.1,
            max_tokens=32,
        )
    )

    assert fake_litellm.api_base == "http://localhost:9001"
    assert fake_litellm.api_key == "none"
    assert result["model"] == config.agent.model
    assert result["tool_choice"] is None
