"""Regression tests for the train_model tool contract."""

from __future__ import annotations

import asyncio

import pytest

from hannah.agent.tool_registry import ToolRegistry
import hannah.tools.train_model.tool as train_model_tool


def _train_model_spec(registry: ToolRegistry) -> dict:
    return next(tool for tool in registry.get_tool_specs() if tool["function"]["name"] == "train_model")


def test_train_model_spec_exposes_canonical_model_enum_and_offline_guidance() -> None:
    registry = ToolRegistry()

    spec = _train_model_spec(registry)["function"]
    model_schema = spec["parameters"]["properties"]["model_name"]

    assert set(model_schema["enum"]) == set(train_model_tool.SUPPORTED_MODEL_NAMES)
    assert "strategy_model" not in model_schema["enum"]
    description = spec["description"].lower()
    assert "offline" in description
    assert "strategy analysis" in description


@pytest.mark.parametrize(
    ("raw_name", "expected_name"),
    [
        ("strategy_model", "pit_policy_q"),
        ("race_strategy", "pit_policy_q"),
        ("race strategy", "pit_policy_q"),
        ("pit-strategy-model", "pit_policy_q"),
        ("winner_model", "winner_ensemble"),
        ("lap-time-model", "laptime_model"),
    ],
)
def test_train_model_aliases_are_normalized_before_schema_validation(
    raw_name: str,
    expected_name: str,
) -> None:
    registry = ToolRegistry()

    normalized = registry.normalize_args("train_model", {"model_name": raw_name})

    assert normalized == {"model_name": expected_name}


def test_train_model_alias_dispatches_to_strategy_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = ToolRegistry()
    seen: dict[str, object] = {}

    def _fake_train(years: list[int] | None = None, races: list[str] | None = None) -> str:
        seen["years"] = years
        seen["races"] = races
        return "models/saved/pit_policy_q_v1.pkl"

    monkeypatch.setattr(train_model_tool.train_pit_q, "train", _fake_train)

    result = asyncio.run(
        registry.call(
            "train_model",
            {
                "model_name": "strategy_model",
                "years": [2023, 2024],
                "races": ["japan"],
            },
        )
    )

    assert result == {"saved": "models/saved/pit_policy_q_v1.pkl"}
    assert seen == {"years": [2023, 2024], "races": ["japan"]}
