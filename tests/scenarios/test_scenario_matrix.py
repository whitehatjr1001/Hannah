"""Scenario matrix integrity checks for Hannah v1 public contracts."""

from __future__ import annotations

from tests.scenarios.contracts import get_public_scenarios


def test_public_matrix_has_at_least_twenty_scenarios() -> None:
    scenarios = get_public_scenarios()
    assert len(scenarios) >= 20


def test_public_matrix_ids_are_unique() -> None:
    scenarios = get_public_scenarios()
    ids = [scenario.scenario_id for scenario in scenarios]
    assert len(ids) == len(set(ids))


def test_public_matrix_category_coverage() -> None:
    categories = {scenario.category for scenario in get_public_scenarios()}
    assert categories == {"strategy", "prediction", "training_smoke"}


def test_each_scenario_declares_contract_fields() -> None:
    allowed_tools = {"race_data", "race_sim", "pit_strategy", "predict_winner", "train_model"}
    for scenario in get_public_scenarios():
        assert scenario.expected_tool_path
        assert scenario.pass_fail_criteria
        assert scenario.available_telemetry
        for tool in scenario.expected_tool_path:
            assert tool in allowed_tools
            assert tool in scenario.tool_inputs

