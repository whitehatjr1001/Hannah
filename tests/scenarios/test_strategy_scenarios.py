"""Public strategy scenario contract tests."""

from __future__ import annotations

from tests.scenarios.contracts import get_scenarios_by_category
from tests.scenarios.harness import assert_required_keys, execute_public_contract

STRATEGY_SCENARIOS = get_scenarios_by_category("strategy")


def test_strategy_contract_count() -> None:
    assert len(STRATEGY_SCENARIOS) >= 8


def test_strategy_scenarios_follow_public_contract(monkeypatch) -> None:
    for scenario in STRATEGY_SCENARIOS:
        executed_path, outputs = execute_public_contract(scenario, monkeypatch)
        assert tuple(executed_path) == scenario.expected_tool_path

        race_data_payload = outputs["race_data"]
        assert_required_keys(race_data_payload, ("laps", "weather", "drivers", "session_info"))

        race_sim_payload = outputs["race_sim"]
        assert_required_keys(race_sim_payload, ("simulation", "strategy", "trace"))
        assert_required_keys(race_sim_payload["simulation"], scenario.expected_sim_output_shape)
        assert_required_keys(race_sim_payload["trace"], scenario.expected_trace_shape)
        trace_timeline = race_sim_payload["trace"]["timeline"]
        assert isinstance(trace_timeline, list)
        assert trace_timeline
        assert_required_keys(trace_timeline[0], scenario.expected_trace_timeline_entry_shape)
        timeline_laps = [int(event["lap"]) for event in trace_timeline]
        assert timeline_laps == sorted(timeline_laps)
        assert timeline_laps[0] >= 1
        trace_payload = race_sim_payload["trace"]
        assert trace_payload["focus_driver"] == scenario.input_context["driver"]

        events = trace_payload.get("events")
        assert isinstance(events, list)
        if events:
            first_event = events[0]
            assert_required_keys(first_event, ("kind", "start_lap", "end_lap", "intensity"))
            assert int(first_event["start_lap"]) <= int(first_event["end_lap"])
        valid_labels = {"pit_window", "pace_projection", *(str(event["kind"]) for event in events)}
        assert {str(event["event"]) for event in trace_timeline} <= valid_labels
        coherent_entries = [entry for entry in trace_timeline if str(entry.get("event")) in valid_labels - {"pit_window", "pace_projection"}]
        for entry in coherent_entries:
            window = entry.get("event_window")
            assert isinstance(window, dict)
            assert_required_keys(window, ("kind", "start_lap", "end_lap", "intensity", "window_index"))
            assert window["kind"] == entry["event"]
            assert int(window["start_lap"]) <= int(entry["lap"]) <= int(window["end_lap"])

        pit_plan = trace_payload.get("pit_plan")
        assert isinstance(pit_plan, list)
        assert pit_plan
        assert_required_keys(
            pit_plan[0],
            (
                "driver",
                "current_compound",
                "target_compound",
                "current_tyre_age",
                "optimal_pit_lap",
                "undercut_window",
                "projected_rejoin_position",
                "projected_rejoin_gap_s",
                "projected_car_ahead",
            ),
        )
        assert pit_plan[0]["driver"] == scenario.input_context["driver"]
        assert int(pit_plan[0]["projected_rejoin_position"]) >= 1
        assert float(pit_plan[0]["projected_rejoin_gap_s"]) >= 0.0

        decision = outputs["pit_strategy"]
        assert_required_keys(decision, scenario.expected_recommendation_shape)
        assert isinstance(decision["recommended_pit_lap"], int)
        assert decision["recommended_pit_lap"] >= 1
        assert isinstance(decision["rival_threats"], list)
        assert 0.0 <= float(decision["confidence"]) <= 1.0
