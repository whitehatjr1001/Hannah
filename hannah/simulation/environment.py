"""Gym-compatible environment for RL smoke paths."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hannah.domain.tracks import get_track


@dataclass
class EnvironmentConfig:
    track: str = "bahrain"
    total_laps: int = 57
    pit_loss: float = 22.5
    weather: str = "dry"
    seed: int = 7


class StrategyEnvironment:
    """Small deterministic environment following the pit-stop-simulator state shape."""

    def __init__(self, config: EnvironmentConfig | None = None) -> None:
        self.config = config or EnvironmentConfig()
        self.track = get_track(self.config.track, self.config.total_laps)
        self.rng = np.random.default_rng(self.config.seed)
        self.current_lap = 0
        self.tire_wear = 0.0
        self.traffic = 0.12
        self.fuel_weight = 105.0
        self.rain_active = self.config.weather in {"mixed", "wet"}
        self.safety_car_active = False
        self.vsc_active = False
        self.current_tire = "MEDIUM"
        self.done = False

    def reset(self) -> tuple[list[float], dict]:
        self.current_lap = 0
        self.tire_wear = 0.0
        self.traffic = 0.1
        self.fuel_weight = 105.0
        self.safety_car_active = False
        self.vsc_active = False
        self.done = False
        self.current_tire = "INTER" if self.config.weather == "wet" else "MEDIUM"
        # Keep reset deterministic across repeated calls for tests.
        self.rng = np.random.default_rng(self.config.seed)
        return self._obs(), {"status": "reset", "track": self.track.name}

    def step(self, action: int) -> tuple[list[float], float, bool, bool, dict]:
        if self.done:
            raise RuntimeError("environment is finished; call reset()")
        if action not in {0, 1}:
            raise ValueError(f"unsupported action: {action}")
        self.current_lap += 1
        self._update_control_state()
        lap_time = self.track.base_lap_time
        lap_time += (self.tire_wear / 100.0) ** 1.6 * 8.0
        lap_time += self.traffic * 4.5
        lap_time += self.fuel_weight * 0.03
        if self.safety_car_active:
            lap_time = max(lap_time, self.track.base_lap_time * 1.45)
        elif self.vsc_active:
            lap_time = max(lap_time, self.track.base_lap_time * 1.2)
        if self.rain_active and self.current_tire not in {"INTER", "WET"}:
            lap_time += 6.0 if self.config.weather == "mixed" else 12.0
        if action == 1:
            lap_time += self.track.pit_loss
            self.tire_wear = 0.0
            self.current_tire = "HARD" if self.current_tire == "MEDIUM" else "MEDIUM"
        else:
            self.tire_wear = min(100.0, self.tire_wear + self.track.tyre_wear_factor * 4.2)
        self.traffic = float(np.clip(self.rng.normal(0.22, 0.08), 0.0, 0.85))
        self.fuel_weight = max(0.0, self.fuel_weight - 1.9)
        self.done = self.current_lap >= self.config.total_laps
        reward = -lap_time
        return self._obs(), reward, self.done, False, self._info(action, lap_time)

    def _obs(self) -> list[float]:
        return [
            float(self.current_lap),
            float(self.tire_wear),
            float(self.traffic),
            float(self.fuel_weight),
            float(self.rain_active),
            float(self.safety_car_active),
            float(self.vsc_active),
        ]

    def _update_control_state(self) -> None:
        # Deterministic, low-frequency control periods inspired by pit-stop-simulator.
        self.safety_car_active = False
        self.vsc_active = False
        if self.track.safety_car_risk >= 0.25 and self.current_lap % 17 == 0:
            self.safety_car_active = True
        elif self.track.safety_car_risk >= 0.18 and self.current_lap % 13 == 0:
            self.vsc_active = True

    def _info(self, action: int, lap_time: float) -> dict:
        return {
            "action": action,
            "lap_time": round(lap_time, 3),
            "lap": self.current_lap,
            "tire_wear": round(self.tire_wear, 2),
            "fuel_weight": round(self.fuel_weight, 2),
            "safety_car_active": self.safety_car_active,
            "vsc_active": self.vsc_active,
        }
