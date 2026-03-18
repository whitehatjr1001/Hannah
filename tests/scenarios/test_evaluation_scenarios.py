"""Public evaluation scenario contract tests for V2-S2."""

from __future__ import annotations

from tests.scenarios.contracts import get_scenarios_by_category
from tests.scenarios.harness import (
    assert_required_keys,
    execute_evaluation_contract,
    execute_public_contract,
)

EVALUATION_SCENARIOS = [
    scenario
    for scenario in get_scenarios_by_category("training_smoke")
    if scenario.expected_evaluation_shape
]


def test_evaluation_contract_count() -> None:
    assert len(EVALUATION_SCENARIOS) >= 4


def test_evaluation_contract_model_coverage() -> None:
    covered_models = {str(scenario.input_context["model_name"]) for scenario in EVALUATION_SCENARIOS}
    assert covered_models == {"tyre_model", "laptime_model", "pit_rl", "winner_ensemble"}


def test_evaluation_scenarios_follow_public_contract(monkeypatch) -> None:
    for scenario in EVALUATION_SCENARIOS:
        executed_path, outputs = execute_public_contract(scenario, monkeypatch)
        assert tuple(executed_path) == scenario.expected_tool_path

        training_payload = outputs["train_model"]
        assert_required_keys(training_payload, scenario.expected_training_shape)

        evaluation_payload = execute_evaluation_contract(scenario)
        assert_required_keys(evaluation_payload, scenario.expected_evaluation_shape)
        assert evaluation_payload["model"] == scenario.input_context["model_name"]
        assert evaluation_payload["artifact_exists"] is True
        assert evaluation_payload["meets_threshold"] is (
            float(evaluation_payload["score"]) >= float(evaluation_payload["threshold"])
        )
        depth = evaluation_payload["evaluation_depth"]
        assert isinstance(depth, dict)
        assert {"scenario_scorecard", "coverage", "stability"} <= set(depth.keys())
