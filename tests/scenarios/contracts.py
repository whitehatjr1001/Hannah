"""Public scenario matrix for Hannah v1 contract tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

ScenarioCategory = Literal["strategy", "prediction", "training_smoke"]

STRATEGY_RECOMMENDATION_KEYS: tuple[str, ...] = (
    "recommended_pit_lap",
    "recommended_compound",
    "strategy_type",
    "confidence",
    "undercut_window",
    "rival_threats",
    "reasoning",
)

SIMULATION_KEYS: tuple[str, ...] = (
    "winner_probs",
    "optimal_pit_laps",
    "optimal_compounds",
    "p50_race_time_s",
    "undercut_windows",
)

TRACE_KEYS: tuple[str, ...] = (
    "race",
    "year",
    "weather",
    "seed",
    "focus_driver",
    "timeline",
)

TRACE_TIMELINE_ENTRY_KEYS: tuple[str, ...] = (
    "lap",
    "event",
    "recommended_pit_lap",
    "recommended_compound",
    "summary",
)

PREDICTION_KEYS: tuple[str, ...] = ("winner_probs",)
TRAINING_KEYS: tuple[str, ...] = ("saved",)
EVALUATION_KEYS: tuple[str, ...] = (
    "model",
    "score",
    "artifact",
    "artifact_exists",
    "threshold",
    "meets_threshold",
    "evaluation_depth",
)


@dataclass(frozen=True)
class ScenarioContract:
    scenario_id: str
    category: ScenarioCategory
    title: str
    input_context: dict[str, Any]
    available_telemetry: tuple[str, ...]
    expected_tool_path: tuple[str, ...]
    tool_inputs: dict[str, dict[str, Any]]
    expected_recommendation_shape: tuple[str, ...]
    expected_sim_output_shape: tuple[str, ...]
    expected_prediction_shape: tuple[str, ...]
    expected_training_shape: tuple[str, ...]
    pass_fail_criteria: tuple[str, ...]
    expected_evaluation_shape: tuple[str, ...] = ()
    expected_trace_shape: tuple[str, ...] = ()
    expected_trace_timeline_entry_shape: tuple[str, ...] = ()


def _strategy(
    scenario_id: str,
    title: str,
    race: str,
    lap: int,
    weather: str,
    driver: str = "VER",
    year: int = 2025,
) -> ScenarioContract:
    drivers = [driver, "NOR", "LEC"]
    return ScenarioContract(
        scenario_id=scenario_id,
        category="strategy",
        title=title,
        input_context={
            "race": race,
            "year": year,
            "lap": lap,
            "weather": weather,
            "driver": driver,
            "drivers": drivers,
        },
        available_telemetry=("lap_times", "stints", "weather", "positions"),
        expected_tool_path=("race_data", "race_sim", "pit_strategy"),
        tool_inputs={
            "race_data": {"race": race, "year": year, "session": "R", "driver": driver},
            "race_sim": {
                "race": race,
                "year": year,
                "weather": weather,
                "drivers": drivers,
                "laps": 57,
                "trace": True,
            },
            "pit_strategy": {"race": race, "driver": driver, "year": year, "lap": lap},
        },
        expected_recommendation_shape=STRATEGY_RECOMMENDATION_KEYS,
        expected_sim_output_shape=SIMULATION_KEYS,
        expected_prediction_shape=(),
        expected_training_shape=(),
        expected_trace_shape=TRACE_KEYS,
        expected_trace_timeline_entry_shape=TRACE_TIMELINE_ENTRY_KEYS,
        pass_fail_criteria=(
            "tool path follows race_data -> race_sim -> pit_strategy",
            "race_sim includes simulation and strategy sections",
            "race_sim includes deterministic trace payload for replay/debug",
            "pit_strategy returns a decisive recommendation shape",
        ),
    )


def _prediction(
    scenario_id: str,
    title: str,
    race: str,
    weather: str,
    drivers: tuple[str, ...] = ("VER", "NOR", "LEC"),
    year: int = 2025,
) -> ScenarioContract:
    return ScenarioContract(
        scenario_id=scenario_id,
        category="prediction",
        title=title,
        input_context={
            "race": race,
            "year": year,
            "weather": weather,
            "drivers": list(drivers),
        },
        available_telemetry=("qualifying", "practice_pace", "weather", "historical_race"),
        expected_tool_path=("race_data", "predict_winner"),
        tool_inputs={
            "race_data": {"race": race, "year": year, "session": "Q", "driver": None},
            "predict_winner": {"race": race, "year": year, "drivers": list(drivers)},
        },
        expected_recommendation_shape=(),
        expected_sim_output_shape=(),
        expected_prediction_shape=PREDICTION_KEYS,
        expected_training_shape=(),
        pass_fail_criteria=(
            "tool path follows race_data -> predict_winner",
            "predict_winner returns probability map shape",
            "winner_probs values are normalized and non-negative",
        ),
    )


def _training(
    scenario_id: str,
    title: str,
    model_name: str,
    years: list[int],
    races: list[str] | None,
) -> ScenarioContract:
    evaluation_supported_models = {"tyre_model", "laptime_model", "pit_rl", "winner_ensemble"}
    evaluation_shape: tuple[str, ...] = (
        EVALUATION_KEYS if model_name in evaluation_supported_models else ()
    )
    tool_inputs: dict[str, dict[str, Any]] = {
        "train_model": {"model_name": model_name, "years": years, "races": races}
    }
    if model_name in evaluation_supported_models:
        tool_inputs["evaluate_model"] = {"model_name": model_name}

    pass_fail_criteria = [
        "tool path uses train_model only",
        "training returns saved artifact path shape",
        "all-mode returns a dict keyed by each trainable model target",
    ]
    if model_name in evaluation_supported_models:
        pass_fail_criteria.append(
            "scenario includes evaluation contract inputs for post-training validation"
        )

    return ScenarioContract(
        scenario_id=scenario_id,
        category="training_smoke",
        title=title,
        input_context={
            "model_name": model_name,
            "years": years,
            "races": races or [],
        },
        available_telemetry=("openpitwall", "engineered_features"),
        expected_tool_path=("train_model",),
        tool_inputs=tool_inputs,
        expected_recommendation_shape=(),
        expected_sim_output_shape=(),
        expected_prediction_shape=(),
        expected_training_shape=TRAINING_KEYS,
        pass_fail_criteria=tuple(pass_fail_criteria),
        expected_evaluation_shape=evaluation_shape,
    )


ALL_PUBLIC_SCENARIOS: tuple[ScenarioContract, ...] = (
    _strategy("S01", "Bahrain dry one-stop baseline", race="bahrain", lap=18, weather="dry"),
    _strategy("S02", "Monaco track-position overcut", race="monaco", lap=30, weather="dry"),
    _strategy("S03", "Singapore safety-car pit window", race="singapore", lap=24, weather="dry"),
    _strategy("S04", "Wet-to-dry crossover call", race="silverstone", lap=20, weather="mixed"),
    _strategy("S05", "McLaren undercut threat", race="miami", lap=17, weather="dry"),
    _strategy("S06", "Ferrari undercut threat", race="barcelona", lap=19, weather="dry"),
    _strategy("S07", "Soft-tyre cliff approaching", race="bahrain", lap=16, weather="dry"),
    _strategy("S08", "Virtual safety car opportunity", race="jeddah", lap=34, weather="dry"),
    _strategy("S09", "Late-race rain hedge decision", race="spa", lap=40, weather="mixed"),
    _strategy("S10", "Double-stack risk under restart", race="imola", lap=22, weather="dry"),
    _prediction(
        "P01",
        "Winner prediction from partial Bahrain weekend",
        race="bahrain",
        weather="dry",
    ),
    _prediction(
        "P02",
        "Winner prediction from partial Monza weekend",
        race="monza",
        weather="dry",
    ),
    _prediction(
        "P03",
        "Rain-affected qualifying projection",
        race="interlagos",
        weather="wet",
    ),
    _prediction(
        "P04",
        "Three-way title fight podium probabilities",
        race="abu_dhabi",
        weather="dry",
    ),
    _prediction(
        "P05",
        "Safety-car heavy upset probability",
        race="singapore",
        weather="mixed",
    ),
    _training(
        "T01",
        "Training smoke: tyre model",
        model_name="tyre_model",
        years=[2023, 2024],
        races=["bahrain", "jeddah"],
    ),
    _training(
        "T02",
        "Training smoke: lap-time model",
        model_name="laptime_model",
        years=[2022, 2023, 2024],
        races=["monaco"],
    ),
    _training(
        "T03",
        "Training smoke: pit RL",
        model_name="pit_rl",
        years=[2024],
        races=["singapore", "monza"],
    ),
    _training(
        "T06",
        "Training smoke: Q-learning pit policy",
        model_name="pit_policy_q",
        years=[2023, 2024],
        races=["bahrain", "singapore"],
    ),
    _training(
        "T04",
        "Training smoke: winner ensemble",
        model_name="winner_ensemble",
        years=[2023, 2024],
        races=None,
    ),
    _training(
        "T05",
        "Training smoke: all v1 models",
        model_name="all",
        years=[2022, 2023, 2024],
        races=["bahrain", "monaco", "singapore"],
    ),
)


def get_public_scenarios() -> list[ScenarioContract]:
    return list(ALL_PUBLIC_SCENARIOS)


def get_scenarios_by_category(category: ScenarioCategory) -> list[ScenarioContract]:
    return [scenario for scenario in ALL_PUBLIC_SCENARIOS if scenario.category == category]


def get_scenario_by_id(scenario_id: str) -> ScenarioContract:
    for scenario in ALL_PUBLIC_SCENARIOS:
        if scenario.scenario_id == scenario_id:
            return scenario
    raise ValueError(f"unknown public scenario: {scenario_id}")
