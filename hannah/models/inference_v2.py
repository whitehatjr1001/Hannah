"""Direct inference helpers for v2 trained model artifacts."""

from __future__ import annotations

import importlib.util
import pickle
import sys
import zipfile
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

FEATURE_STORE_ROOT = Path("data/feature_store")
MODEL_DIR = Path("models/saved")
REPO_ROOT = Path(__file__).resolve().parents[2]


def load_race_frame(year: int, race: str) -> pd.DataFrame:
    race_slug = race.lower().replace(" ", "_")
    path = FEATURE_STORE_ROOT / str(year) / race_slug / "race_features.parquet"
    if not path.exists():
        raise FileNotFoundError(f"no feature store parquet found at {path}")
    df = pd.read_parquet(path)
    if "compound" in df.columns:
        df = df[df["compound"].isin({"SOFT", "MEDIUM", "HARD"})]
    if "lap_time_s" in df.columns:
        df = df[df["lap_time_s"] > 0]
        df = df[df["lap_time_s"] < 200]
        df = df.dropna(subset=["lap_time_s"])
    if "is_pit_out_lap" in df.columns:
        df = df[~df["is_pit_out_lap"].astype(bool)]
    return df.reset_index(drop=True)


def available_races(year: int) -> list[str]:
    root = FEATURE_STORE_ROOT / str(year)
    if not root.exists():
        return []
    races: list[str] = []
    for path in sorted(root.glob("*/race_features.parquet")):
        races.append(path.parent.name)
    return races


def load_joblib_artifact(path: str | Path) -> Any:
    _bootstrap_script_artifact_classes()
    return joblib.load(Path(path))


def load_pickle_artifact(path: str | Path) -> Any:
    _bootstrap_script_artifact_classes()
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def _bootstrap_script_artifact_classes() -> None:
    """Expose script-defined dataclasses under __main__ for pickle/joblib loads.

    The v2 artifacts were trained and serialized from top-level scripts, so the
    dataclass types were recorded as living in ``__main__``. When we later load
    them from a library or another script, Python cannot resolve those class
    names unless we re-register them here.
    """
    main_module = sys.modules.get("__main__")
    if main_module is None:
        return

    for script_name, class_names in (
        ("train_tyre_v2.py", ("TyreDegV2Artifact",)),
        ("train_laptime_v2.py", ("LapTimeV2Artifact",)),
        ("train_pit_policy_v2.py", ("PitPolicyV2Artifact",)),
        ("train_winner_real.py", ("WinnerV2Artifact",)),
    ):
        script_path = REPO_ROOT / "scripts" / script_name
        spec = importlib.util.spec_from_file_location(f"_artifact_{script_name}", script_path)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        sys.modules.setdefault(spec.name, module)
        spec.loader.exec_module(module)
        for class_name in class_names:
            artifact_class = getattr(module, class_name, None)
            if artifact_class is not None and not hasattr(main_module, class_name):
                setattr(main_module, class_name, artifact_class)


def _age_col(df: pd.DataFrame) -> str:
    if " tyre_age_in_stint" in df.columns:
        return " tyre_age_in_stint"
    if "tyr_e_age_in_stint" in df.columns:
        return "tyr_e_age_in_stint"
    if "tyre_age_in_stint" in df.columns:
        return "tyre_age_in_stint"
    candidates = [c for c in df.columns if "age" in c.lower() and "stint" in c.lower()]
    if candidates:
        return candidates[0]
    raise KeyError("tyre age in stint column not found")


def build_tyre_features(df: pd.DataFrame, feature_names: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    frame = df.copy()
    age_col = _age_col(frame)
    race_medians = frame.groupby("race")["lap_time_s"].median()
    frame["race_median"] = frame["race"].map(race_medians)
    frame["stint_length"] = frame.groupby(["race", "driver_code", "stint_number"]).cumcount() + 1
    frame["position_change"] = frame.groupby(["race", "driver_code"])["position"].diff().fillna(0)
    frame["gap_normalized"] = frame["gap_to_leader_s"] / frame["race_median"].replace(0, 100)
    frame["gap_normalized"] = frame["gap_normalized"].fillna(0).clip(0, 10)
    features = pd.DataFrame(
        {
            "tyr_e_age_in_stint": frame[age_col].fillna(0),
            "stint_number": frame["stint_number"].fillna(1),
            "stint_length": frame["stint_length"].fillna(1),
            "lap_number": frame["lap_number"].fillna(1),
            "position": frame["position"].fillna(10),
            "position_change": frame["position_change"].fillna(0),
            "gap_normalized": frame["gap_normalized"].fillna(0),
            "track_temp": frame["track_temp"].fillna(30.0),
            "air_temp": frame["air_temp"].fillna(25.0),
            "rainfall": frame["rainfall"].fillna(0.0),
            "race_median": frame["race_median"].fillna(95.0),
        }
    )
    compound_dummies = pd.get_dummies(frame["compound"], prefix="compound", dtype=float)
    race_dummies = pd.get_dummies(frame["race"], prefix="race", dtype=float)
    features = pd.concat([features, compound_dummies, race_dummies], axis=1)
    features = features.reindex(columns=feature_names, fill_value=0.0)
    return features, frame["race_median"]


def build_laptime_features(df: pd.DataFrame, feature_names: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    frame = df.copy()
    age_col = _age_col(frame)
    frame["driver_race_median"] = frame.groupby(["race", "driver_code"])["lap_time_s"].transform("median")
    frame["stint_length"] = frame.groupby(["race", "driver_code", "stint_number"]).cumcount() + 1
    frame["position_change"] = frame.groupby(["race", "driver_code"])["position"].diff().fillna(0)
    frame["gap_normalized"] = frame["gap_to_leader_s"] / frame["driver_race_median"].replace(0, 100)
    frame["gap_normalized"] = frame["gap_normalized"].fillna(0).clip(0, 10)
    frame["sector_1_clean"] = frame["sector_1"].fillna(0).replace(0, np.nan)
    frame["sector_2_clean"] = frame["sector_2"].fillna(0).replace(0, np.nan)
    frame["sector_3_clean"] = frame["sector_3"].fillna(0).replace(0, np.nan)
    frame["sector_sum"] = frame["sector_1_clean"] + frame["sector_2_clean"] + frame["sector_3_clean"]
    frame["compound_encoded"] = frame["compound"].map({"SOFT": 2, "MEDIUM": 1, "HARD": 0})
    features = pd.DataFrame(
        {
            "tyr_e_age_in_stint": frame[age_col].fillna(0),
            "stint_number": frame["stint_number"].fillna(1),
            "stint_length": frame["stint_length"].fillna(1),
            "compound_encoded": frame["compound_encoded"].fillna(1),
            "lap_number": frame["lap_number"].fillna(1),
            "position": frame["position"].fillna(10),
            "position_change": frame["position_change"].fillna(0),
            "gap_normalized": frame["gap_normalized"].fillna(0),
            "track_temp": frame["track_temp"].fillna(30.0),
            "air_temp": frame["air_temp"].fillna(25.0),
            "rainfall": frame["rainfall"].fillna(0.0),
            "sector_1": frame["sector_1_clean"].fillna(0),
            "sector_2": frame["sector_2_clean"].fillna(0),
            "sector_3": frame["sector_3_clean"].fillna(0),
            "sector_sum": frame["sector_sum"].fillna(0),
        }
    )
    compound_dummies = pd.get_dummies(frame["compound"], prefix="compound", dtype=float)
    features = pd.concat([features, compound_dummies], axis=1)
    features = features.reindex(columns=feature_names, fill_value=0.0)
    return features, frame["driver_race_median"]


def build_pit_features(df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    frame = df.copy()
    age_col = _age_col(frame)
    stint_max = frame.groupby(["race", "driver_code", "stint_number"])["lap_number"].transform("max")
    frame["stint_progress"] = frame["lap_number"] / stint_max.clip(lower=1)
    race_total_laps = frame.groupby(["race"])["lap_number"].transform("max")
    frame["laps_remaining"] = (race_total_laps - frame["lap_number"]).clip(lower=0)
    frame["laps_remaining_pct"] = frame["laps_remaining"] / race_total_laps.clip(lower=1)
    frame["position_change"] = frame.groupby(["race", "driver_code"])["position"].diff().fillna(0)
    frame["is_leader"] = (frame["position"] == 1).astype(int)
    frame["top_3"] = (frame["position"] <= 3).astype(int)
    frame["gap_normalized"] = frame["gap_to_leader_s"].fillna(0) / 100
    frame["gap_to_ahead"] = frame["gap_to_leader_s"].fillna(0)
    frame["compound_encoded"] = frame["compound"].map({"SOFT": 0, "MEDIUM": 1, "HARD": 2}).fillna(1).astype(int)
    frame["safety_car_flag"] = frame.get("safety_car", 0).astype(int)
    frame["vsc_flag"] = frame.get("vsc", 0).astype(int)
    frame["race_phase"] = pd.cut(frame["lap_number"], bins=[0, 15, 35, 100], labels=[0, 1, 2]).astype(int)
    frame["compound_x_age"] = frame["compound_encoded"] * frame[age_col]
    features = pd.DataFrame(
        {
            "lap_number": frame["lap_number"].fillna(1),
            "tyr_e_age_in_stint": frame[age_col].fillna(0),
            "stint_progress": frame["stint_progress"].fillna(0),
            "compound_encoded": frame["compound_encoded"],
            "compound_x_age": frame["compound_x_age"].fillna(0),
            "position": frame["position"].fillna(10),
            "position_change": frame["position_change"].fillna(0),
            "is_leader": frame["is_leader"],
            "top_3": frame["top_3"],
            "gap_to_ahead": frame["gap_to_ahead"].fillna(0),
            "gap_normalized": frame["gap_normalized"].fillna(0),
            "laps_remaining": frame["laps_remaining"].fillna(0),
            "laps_remaining_pct": frame["laps_remaining_pct"].fillna(0),
            "race_phase": frame["race_phase"],
            "track_temp": frame["track_temp"].fillna(30.0),
            "rainfall": frame["rainfall"].fillna(0.0),
            "safety_car_flag": frame["safety_car_flag"],
            "vsc_flag": frame["vsc_flag"],
        }
    )
    return features.reindex(columns=feature_names, fill_value=0.0)


def build_winner_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    rows: list[dict[str, Any]] = []
    for driver, driver_df in df.groupby("driver_code"):
        final_laps = driver_df[driver_df["lap_number"] == driver_df["lap_number"].max()]
        if final_laps.empty:
            continue
        first_lap = driver_df[driver_df["lap_number"] == 1]
        grid_pos = int(first_lap.iloc[0].get("position", 10)) if not first_lap.empty else 10
        avg_lap = float(driver_df["lap_time_s"].mean())
        compounds = driver_df["compound"].value_counts()
        compound = compounds.index[0] if len(compounds) > 0 else "UNKNOWN"
        rows.append(
            {
                "driver": driver,
                "team": driver_df["team_name"].iloc[0] if "team_name" in driver_df.columns else "",
                "grid_position": grid_pos,
                "avg_lap_time": avg_lap,
                "compound": compound,
                "total_laps": len(driver_df),
            }
        )
    race_df = pd.DataFrame(rows)
    if race_df.empty:
        return race_df, []
    race_baseline = race_df["avg_lap_time"].median()
    race_df["lap_time_delta"] = race_df["avg_lap_time"] - race_baseline
    team_encoder = {team: idx for idx, team in enumerate(race_df["team"].unique())}
    race_df["team_encoded"] = race_df["team"].map(team_encoder).fillna(0)
    compound_map = {"SOFT": 0, "MEDIUM": 1, "HARD": 2, "UNKNOWN": 3}
    race_df["compound_encoded"] = race_df["compound"].map(compound_map).fillna(3)
    features = pd.DataFrame(
        {
            "grid_position": race_df["grid_position"].fillna(10),
            "lap_time_delta": race_df["lap_time_delta"].fillna(0),
            "team_encoded": race_df["team_encoded"],
            "compound_encoded": race_df["compound_encoded"],
            "total_laps": race_df["total_laps"].fillna(50),
        }
    )
    return pd.concat([race_df[["driver"]], features], axis=1), list(features.columns)


def infer_tyre(df: pd.DataFrame) -> dict[str, Any]:
    artifact = load_joblib_artifact(MODEL_DIR / "tyre_deg_v1.pkl")
    features, race_medians = build_tyre_features(df, list(artifact.feature_names))
    predicted_delta = artifact.model.predict(features)
    predicted_lap_time = predicted_delta + race_medians.to_numpy(dtype=float)
    return {
        "rows": int(len(df)),
        "rmse": float(np.sqrt(np.mean((df["lap_time_s"].to_numpy(dtype=float) - predicted_lap_time) ** 2))),
        "mean_predicted_delta": float(np.mean(predicted_delta)),
    }


def infer_laptime(df: pd.DataFrame) -> dict[str, Any]:
    artifact = load_joblib_artifact(MODEL_DIR / "laptime_v1.pkl")
    features, driver_race_medians = build_laptime_features(df, list(artifact.feature_names))
    predicted_delta = artifact.model.predict(features)
    predicted_lap_time = predicted_delta + driver_race_medians.to_numpy(dtype=float)
    return {
        "rows": int(len(df)),
        "rmse": float(np.sqrt(np.mean((df["lap_time_s"].to_numpy(dtype=float) - predicted_lap_time) ** 2))),
        "mean_predicted_lap_time": float(np.mean(predicted_lap_time)),
    }


def infer_pit(df: pd.DataFrame) -> dict[str, Any]:
    artifact = load_joblib_artifact(MODEL_DIR / "pit_rl_v1.zip")
    features = build_pit_features(df, list(artifact.feature_names))
    probabilities = artifact.model.predict_proba(features)[:, 1]
    scored = df[["driver_code", "lap_number"]].copy()
    scored["pit_probability"] = probabilities
    recommendations: list[dict[str, Any]] = []
    for driver_code, driver_df in scored.groupby("driver_code"):
        best = driver_df.sort_values("pit_probability", ascending=False).iloc[0]
        recommendations.append(
            {
                "driver": str(driver_code),
                "recommended_pit_lap": int(best["lap_number"]),
                "pit_probability": round(float(best["pit_probability"]), 4),
            }
        )
    recommendations.sort(key=lambda row: row["pit_probability"], reverse=True)
    return {
        "rows": int(len(df)),
        "mean_pit_probability": float(np.mean(probabilities)),
        "recommendations": recommendations,
    }


def infer_winner(df: pd.DataFrame) -> dict[str, Any]:
    artifact = load_pickle_artifact(MODEL_DIR / "winner_ensemble_v1.pkl")
    race_df, feature_names = build_winner_features(df)
    if race_df.empty:
        return {"winner_probs": {}}
    features = race_df[feature_names].to_numpy(dtype=float)
    ensemble = pickle.loads(artifact.ensemble)
    probabilities = ensemble.predict_proba(features)[:, 1]
    total = float(np.sum(probabilities))
    if total > 0:
        probabilities = probabilities / total
    payload = [
        {"driver": str(driver), "win_probability": round(float(probability), 4)}
        for driver, probability in zip(race_df["driver"], probabilities)
    ]
    payload.sort(key=lambda row: row["win_probability"], reverse=True)
    return {"winner_probs": payload}


def run_inference(year: int, race: str) -> dict[str, Any]:
    df = load_race_frame(year, race)
    return {
        "year": year,
        "race": race,
        "rows": int(len(df)),
        "tyre": infer_tyre(df),
        "laptime": infer_laptime(df),
        "pit": infer_pit(df),
        "winner": infer_winner(df),
    }
