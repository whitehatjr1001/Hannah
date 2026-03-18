"""Public training-smoke scenario contract tests."""

from __future__ import annotations

import pytest

from tests.scenarios.contracts import get_scenarios_by_category
from tests.scenarios.harness import assert_required_keys, execute_public_contract

TRAINING_SCENARIOS = get_scenarios_by_category("training_smoke")


def test_training_smoke_contract_count() -> None:
    assert len(TRAINING_SCENARIOS) >= 6


@pytest.mark.parametrize("scenario", TRAINING_SCENARIOS, ids=[s.scenario_id for s in TRAINING_SCENARIOS])
def test_training_smoke_scenarios_follow_public_contract(scenario, monkeypatch) -> None:
    try:
        executed_path, outputs = execute_public_contract(scenario, monkeypatch)
    except Exception as err:
        pytest.fail(
            f"training smoke scenario {scenario.scenario_id} should execute through train_model: {err}"
        )
    assert tuple(executed_path) == scenario.expected_tool_path

    training_payload = outputs["train_model"]
    assert_required_keys(training_payload, scenario.expected_training_shape)

    model_name = str(scenario.input_context["model_name"])
    saved = training_payload["saved"]
    if model_name == "all":
        assert isinstance(saved, dict)
        assert set(saved.keys()) == {
            "tyre_model",
            "laptime_model",
            "pit_rl",
            "pit_policy_q",
            "winner_ensemble",
        }
        assert all(str(path) for path in saved.values())
    else:
        assert isinstance(saved, str)
        assert saved
