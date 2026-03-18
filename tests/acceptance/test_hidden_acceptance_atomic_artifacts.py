"""Masked acceptance checks for atomic artifact writes."""

from __future__ import annotations

import pickle
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pytest

from hannah.models import train_laptime, train_pit_q, train_pit_rl, train_tyre_deg, train_winner

pytestmark = pytest.mark.filterwarnings("ignore:urllib3 v2 only supports OpenSSL 1.1.1+")


@dataclass(frozen=True)
class HiddenAtomicArtifactScenario:
    scenario_id: str
    model_name: str
    loader: str


ATOMIC_ARTIFACT_SCENARIOS: tuple[HiddenAtomicArtifactScenario, ...] = (
    HiddenAtomicArtifactScenario("HACC_V2S4_A01", "tyre_model", "pickle"),
    HiddenAtomicArtifactScenario("HACC_V2S4_A02", "laptime_model", "pickle"),
    HiddenAtomicArtifactScenario("HACC_V2S4_A03", "pit_rl", "zip"),
    HiddenAtomicArtifactScenario("HACC_V2S4_A04", "pit_policy_q", "pickle"),
    HiddenAtomicArtifactScenario("HACC_V2S4_A05", "winner_ensemble", "pickle"),
)


def _train(model_name: str) -> Path:
    train_map = {
        "tyre_model": train_tyre_deg.train,
        "laptime_model": train_laptime.train,
        "pit_rl": train_pit_rl.train,
        "pit_policy_q": train_pit_q.train,
        "winner_ensemble": train_winner.train,
    }
    return Path(train_map[model_name](years=[2023, 2024], races=["bahrain", "singapore"]))


@pytest.mark.parametrize(
    "scenario",
    ATOMIC_ARTIFACT_SCENARIOS,
    ids=[scenario.scenario_id for scenario in ATOMIC_ARTIFACT_SCENARIOS],
)
def test_hidden_atomic_artifact_writes_leave_loadable_outputs_and_no_temp_leaks(
    scenario: HiddenAtomicArtifactScenario,
) -> None:
    artifact_path = _train(scenario.model_name)
    assert artifact_path.exists()

    if scenario.loader == "zip":
        with zipfile.ZipFile(artifact_path, mode="r") as archive:
            assert archive.namelist()
    else:
        with artifact_path.open("rb") as handle:
            payload = pickle.load(handle)
        assert payload is not None

    temp_files = list(artifact_path.parent.glob("*.tmp"))
    assert not temp_files
