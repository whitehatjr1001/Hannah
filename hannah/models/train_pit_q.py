"""Lightweight Q-learning pit-policy trainer and runtime helper."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hannah.models.artifact_paths import resolve_artifact_path
from hannah.models.artifacts import atomic_pickle_dump
from hannah.domain.tracks import get_track
from hannah.simulation.environment import EnvironmentConfig, StrategyEnvironment
from hannah.simulation.sandbox import RaceState

ARTIFACT_PATH = Path("models/saved/pit_policy_q_v1.pkl")
STATE_BUCKETS: tuple[int, int, int, int, int, int] = (58, 10, 10, 2, 2, 2)
ACTION_COUNT = 2


@dataclass(frozen=True)
class QPitPolicyArtifact:
    """Serialized Q-learning pit policy."""

    version: str
    buckets: tuple[int, int, int, int, int, int]
    q_table: np.ndarray
    years: tuple[int, ...]
    races: tuple[str, ...]


def _discretize(observation: list[float] | np.ndarray) -> tuple[int, int, int, int, int, int]:
    values = [float(value) for value in observation[:7]]
    lap, tyre_wear, traffic, _fuel_weight, rain_active, safety_car_active, vsc_active = values

    lap_bin = max(0, min(int(lap), STATE_BUCKETS[0] - 1))
    tyre_bin = max(0, min(int(tyre_wear / (100.0 / STATE_BUCKETS[1])), STATE_BUCKETS[1] - 1))
    traffic_bin = max(0, min(int(traffic * STATE_BUCKETS[2]), STATE_BUCKETS[2] - 1))
    rain_bin = max(0, min(int(rain_active), STATE_BUCKETS[3] - 1))
    safety_car_bin = max(0, min(int(safety_car_active), STATE_BUCKETS[4] - 1))
    vsc_bin = max(0, min(int(vsc_active), STATE_BUCKETS[5] - 1))
    return lap_bin, tyre_bin, traffic_bin, rain_bin, safety_car_bin, vsc_bin


def _build_training_env(track_name: str, weather: str, seed: int) -> StrategyEnvironment:
    track = get_track(track_name)
    return StrategyEnvironment(
        EnvironmentConfig(
            track=track_name,
            total_laps=min(track.laps, 36),
            weather=weather,
            seed=seed,
        )
    )


def _train_q_table(years: list[int], races: list[str] | None) -> np.ndarray:
    race_names = [str(race).lower().replace(" ", "_") for race in (races or ["bahrain", "singapore"])]
    q_table = np.zeros(STATE_BUCKETS + (ACTION_COUNT,), dtype=float)
    base_seed = int(sum(years) + len(race_names) * 17)
    rng = np.random.default_rng(base_seed)
    alpha = 0.12
    gamma = 0.95
    epsilon = 1.0
    epsilon_decay = 0.985
    min_epsilon = 0.05
    episodes = max(120, 40 * len(race_names))

    for episode in range(episodes):
        track_name = race_names[episode % len(race_names)]
        weather = "mixed" if episode % 5 == 0 else "dry"
        env = _build_training_env(track_name=track_name, weather=weather, seed=base_seed + episode)
        observation, _ = env.reset()
        state = _discretize(observation)
        done = False

        while not done:
            if float(rng.random()) < epsilon:
                action = int(rng.integers(0, ACTION_COUNT))
            else:
                action = int(np.argmax(q_table[state]))

            next_observation, reward, done, _truncated, _info = env.step(action)
            next_state = _discretize(next_observation)
            best_future = float(np.max(q_table[next_state]))
            current_value = float(q_table[state + (action,)])
            updated_value = current_value + alpha * (float(reward) + gamma * best_future - current_value)
            q_table[state + (action,)] = updated_value
            state = next_state

        epsilon = max(min_epsilon, epsilon * epsilon_decay)

    return q_table


def train(years: list[int], races: list[str] | None = None) -> str:
    """Train and persist a deterministic tabular Q-learning pit policy."""
    artifact = QPitPolicyArtifact(
        version="v1",
        buckets=STATE_BUCKETS,
        q_table=_train_q_table(years=years, races=races),
        years=tuple(years),
        races=tuple(races or []),
    )
    return atomic_pickle_dump(resolve_artifact_path("pit_policy_q"), artifact)


def load_artifact() -> QPitPolicyArtifact:
    """Load the persisted Q-policy artifact."""
    artifact_path = resolve_artifact_path("pit_policy_q")
    if not artifact_path.exists():
        raise FileNotFoundError(f"pit_policy_q artifact not found: {artifact_path}")
    with artifact_path.open("rb") as handle:
        artifact = pickle.load(handle)
    if not isinstance(artifact, QPitPolicyArtifact):
        raise TypeError(f"unexpected pit_policy_q artifact type: {type(artifact)!r}")
    return artifact


def _observation_from_race_state(
    race_state: RaceState,
    driver_code: str,
    current_lap: int,
) -> list[float]:
    try:
        index = race_state.drivers.index(driver_code)
    except ValueError:
        index = 0

    tyre_wear = min(float(race_state.tyre_ages[index]) * race_state.tyre_wear_factor * 4.2, 100.0)
    traffic = float(np.clip(max(race_state.positions[index] - 1, 0) * 0.18 + race_state.gaps[index] * 0.03, 0.0, 0.85))
    fuel_weight = max(0.0, float(race_state.fuel_load) - max(current_lap, 0) * 1.9)
    rain_active = 1.0 if race_state.weather in {"mixed", "wet"} else 0.0
    safety_car_active = 1.0 if race_state.event_at(current_lap, "safety_car") is not None else 0.0
    vsc_active = 1.0 if race_state.event_at(current_lap, "vsc") is not None else 0.0
    return [
        float(current_lap),
        tyre_wear,
        traffic,
        fuel_weight,
        rain_active,
        safety_car_active,
        vsc_active,
    ]


def choose_action(race_state: RaceState, driver_code: str, current_lap: int) -> int:
    """Choose a deterministic pit action for the current race snapshot."""
    artifact = load_artifact()
    observation = _observation_from_race_state(race_state, driver_code, current_lap)
    state = _discretize(observation)
    return int(np.argmax(artifact.q_table[state]))
