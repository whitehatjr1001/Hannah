"""Public winner-prediction scenario contract tests."""

from __future__ import annotations

from tests.scenarios.contracts import get_scenarios_by_category
from tests.scenarios.harness import assert_required_keys, execute_public_contract

PREDICTION_SCENARIOS = get_scenarios_by_category("prediction")


def test_prediction_contract_count() -> None:
    assert len(PREDICTION_SCENARIOS) >= 5


def test_prediction_scenarios_follow_public_contract(monkeypatch) -> None:
    for scenario in PREDICTION_SCENARIOS:
        executed_path, outputs = execute_public_contract(scenario, monkeypatch)
        assert tuple(executed_path) == scenario.expected_tool_path

        race_data_payload = outputs["race_data"]
        assert_required_keys(race_data_payload, ("laps", "weather", "drivers", "session_info"))

        prediction_payload = outputs["predict_winner"]
        assert_required_keys(prediction_payload, scenario.expected_prediction_shape)
        winner_probs = prediction_payload["winner_probs"]
        assert isinstance(winner_probs, dict)
        assert winner_probs
        total = sum(float(probability) for probability in winner_probs.values())
        assert abs(total - 1.0) <= 1e-3
        assert all(float(probability) >= 0.0 for probability in winner_probs.values())
