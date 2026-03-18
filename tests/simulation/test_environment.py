"""Unit tests for deterministic environment stepping."""

from __future__ import annotations

import pytest

from hannah.simulation.environment import EnvironmentConfig, StrategyEnvironment


def test_environment_reset_is_repeatable_for_same_seed() -> None:
    env = StrategyEnvironment(EnvironmentConfig(track="bahrain", total_laps=6, seed=11))
    obs1, info1 = env.reset()
    obs2, info2 = env.reset()
    assert obs1 == obs2
    assert info1["track"] == info2["track"] == "bahrain"


def test_environment_step_updates_wear_and_pit_behavior() -> None:
    env = StrategyEnvironment(EnvironmentConfig(track="bahrain", total_laps=6, seed=12))
    env.reset()

    obs_stay, reward_stay, done_stay, truncated_stay, info_stay = env.step(0)
    assert obs_stay[1] > 0.0
    assert reward_stay < 0.0
    assert done_stay is False
    assert truncated_stay is False
    assert info_stay["action"] == 0

    obs_pit, reward_pit, done_pit, truncated_pit, info_pit = env.step(1)
    assert obs_pit[1] == 0.0
    assert reward_pit < reward_stay
    assert done_pit is False
    assert truncated_pit is False
    assert info_pit["action"] == 1


def test_environment_rejects_invalid_actions() -> None:
    env = StrategyEnvironment(EnvironmentConfig(total_laps=4))
    env.reset()
    with pytest.raises(ValueError):
        env.step(3)
