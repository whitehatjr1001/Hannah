"""Typed application configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from dataclasses import fields as dataclass_fields
from typing import Any


@dataclass(frozen=True)
class AgentSettings:
    model: str = field(default_factory=lambda: os.getenv("HANNAH_MODEL", "claude-sonnet-4-6"))
    temperature: float = 0.2
    max_tokens: int = 2048
    stream: bool = True


@dataclass(frozen=True)
class SimulationSettings:
    n_worlds: int = 1000
    prediction_mode: bool = True
    async_enabled: bool = True
    safety_car_prob: float = 0.18


@dataclass(frozen=True)
class FastF1Settings:
    cache_dir: str = "data/fastf1_cache"
    timeout: int = 30


@dataclass(frozen=True)
class OpenF1Settings:
    base_url: str = "https://api.openf1.org/v1"
    timeout: int = 10


@dataclass(frozen=True)
class ModelPaths:
    tyre_model: str = "models/saved/tyre_deg_v1.pkl"
    laptime_model: str = "models/saved/laptime_v1.pkl"
    pit_rl: str = "models/saved/pit_rl_v1.zip"
    pit_policy_q: str = "models/saved/pit_policy_q_v1.pkl"
    winner_ensemble: str = "models/saved/winner_ensemble_v1.pkl"

    @property
    def tyre_deg(self) -> str:
        return self.tyre_model

    @property
    def laptime(self) -> str:
        return self.laptime_model

    @property
    def winner(self) -> str:
        return self.winner_ensemble


@dataclass(frozen=True)
class RLMSettings:
    enabled: bool = False
    api_base: str = "http://localhost:8000"
    api_key: str = "none"


@dataclass(frozen=True)
class AppConfig:
    agent: AgentSettings = field(default_factory=AgentSettings)
    simulation: SimulationSettings = field(default_factory=SimulationSettings)
    fastf1: FastF1Settings = field(default_factory=FastF1Settings)
    openf1: OpenF1Settings = field(default_factory=OpenF1Settings)
    models: ModelPaths = field(default_factory=ModelPaths)
    rlm: RLMSettings = field(default_factory=RLMSettings)

    @classmethod
    def model_validate(cls, raw: dict[str, Any]) -> "AppConfig":
        """Build nested config objects from a plain dict."""
        return cls(
            agent=AgentSettings(**_filter_known_fields(AgentSettings, raw.get("agent", {}))),
            simulation=SimulationSettings(
                **_filter_known_fields(SimulationSettings, raw.get("simulation", {}))
            ),
            fastf1=FastF1Settings(**_filter_known_fields(FastF1Settings, raw.get("fastf1", {}))),
            openf1=OpenF1Settings(**_filter_known_fields(OpenF1Settings, raw.get("openf1", {}))),
            models=ModelPaths(**_filter_known_fields(ModelPaths, raw.get("models", {}))),
            rlm=RLMSettings(**_filter_known_fields(RLMSettings, raw.get("rlm", {}))),
        )


def _filter_known_fields(dataclass_type, raw: dict[str, Any]) -> dict[str, Any]:
    known = {field.name for field in dataclass_fields(dataclass_type)}
    return {key: value for key, value in raw.items() if key in known}
