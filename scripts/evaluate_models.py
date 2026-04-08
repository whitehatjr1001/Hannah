"""Evaluate trained models using real train/test split from feature store.

Phase 6: Real data evaluation - train on 2018-2024, test on 2025.

Usage:
    python3 scripts/evaluate_models.py --train-years 2022,2023 --test-year 2024
    python3 scripts/evaluate_models.py --all
"""

from __future__ import annotations

import pickle
from pathlib import Path

import click
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

console = Console()

FEATURE_STORE_ROOT = Path("data/feature_store")
ARTIFACT_DIR = Path("models/saved")

VALID_COMPOUNDS = {"SOFT", "MEDIUM", "HARD"}


def load_dataset_coverage() -> dict:
    """Load and report dataset coverage from feature store."""
    all_files = sorted(FEATURE_STORE_ROOT.rglob("race_features.parquet"))

    years = set()
    races = set()
    total_rows = 0

    for f in all_files:
        try:
            year = int(f.parts[-3])
            race = f.parts[-2]
            years.add(year)
            races.add(race)
            df = pd.read_parquet(f)
            total_rows += len(df)
        except (ValueError, IndexError):
            continue

    return {
        "years": sorted(years),
        "total_races": len(races),
        "total_rows": total_rows,
    }


def _load_feature_store(
    train_years: list[int], test_year: int, include_pit_laps: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load parquet files and split into train/test."""
    all_files = sorted(FEATURE_STORE_ROOT.rglob("race_features.parquet"))

    train_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []

    for f in all_files:
        try:
            year = int(f.parts[-3])
        except (ValueError, IndexError):
            continue

        df = pd.read_parquet(f)
        df = df[df["compound"].isin(VALID_COMPOUNDS)]
        df = df[df["lap_time_s"] > 0]
        df = df[df["lap_time_s"] < 200]
        df = df[~df["is_pit_out_lap"]]
        df = df.dropna(subset=["lap_time_s"])

        # Only filter pit laps for tyre/laptime models, not for pit policy
        if not include_pit_laps:
            df = df[~df["pit_lap"]]

        if year in train_years:
            train_parts.append(df)
        elif year == test_year:
            test_parts.append(df)

    train_df = (
        pd.concat(train_parts, ignore_index=True) if train_parts else pd.DataFrame()
    )
    test_df = pd.concat(test_parts, ignore_index=True) if test_parts else pd.DataFrame()

    return train_df, test_df


def _build_tyre_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build features for tyre degradation model."""
    features = [
        " tyre_age_in_stint",
        "track_temp",
        "air_temp",
        "rainfall",
        "stint_number",
    ]

    compound_dummies = pd.get_dummies(df["compound"], prefix="compound")
    df = pd.concat([df, compound_dummies], axis=1)

    feature_cols = features + list(compound_dummies.columns)
    available_cols = [c for c in feature_cols if c in df.columns]

    X = df[available_cols].fillna(0).values
    y = df["lap_time_s"].values

    return X, y, available_cols


def _load_tyre_model() -> tuple:
    """Load tyre degradation model artifact."""
    artifact_path = ARTIFACT_DIR / " tyre_deg_v1.pkl"
    if not artifact_path.exists():
        return None, None

    with open(artifact_path, "rb") as f:
        artifact = pickle.load(f)

    if hasattr(artifact, "gbr_model"):
        return artifact.gbr_model, artifact.feature_names
    return None, None


def evaluate_tyre_model(train_years: list[int], test_year: int) -> dict:
    """Evaluate tyre degradation model with real train/test split."""
    train_df, test_df = _load_feature_store(
        train_years, test_year, include_pit_laps=False
    )

    if train_df.empty or test_df.empty:
        return {"error": "No data available for evaluation"}

    X_train, y_train, feature_names = _build_tyre_features(train_df)
    X_test, y_test, _ = _build_tyre_features(test_df)

    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.metrics import mean_squared_error, r2_score

    model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    per_compound = {}
    for compound in VALID_COMPOUNDS:
        mask = test_df["compound"] == compound
        if mask.sum() > 0:
            compound_rmse = np.sqrt(mean_squared_error(y_test[mask], y_pred[mask]))
            per_compound[compound] = round(compound_rmse, 2)

    return {
        "rmse": round(rmse, 2),
        "r2": round(r2, 3),
        "per_compound": per_compound,
    }


def evaluate_laptime_model(train_years: list[int], test_year: int) -> dict:
    """Evaluate lap time model with real train/test split."""
    train_df, test_df = _load_feature_store(
        train_years, test_year, include_pit_laps=False
    )

    if train_df.empty or test_df.empty:
        return {"error": "No data available for evaluation"}

    features = [
        "lap_number",
        " tyre_age_in_stint",
        "track_temp",
        "air_temp",
        "rainfall",
        "position",
    ]
    compound_dummies = pd.get_dummies(train_df["compound"], prefix="compound")

    for col in compound_dummies.columns:
        if col not in test_df.columns:
            test_df[col] = 0
    test_compound_dummies = pd.get_dummies(test_df["compound"], prefix="compound")

    feature_cols = features + list(compound_dummies.columns)
    available_cols = [
        c for c in feature_cols if c in train_df.columns and c in test_df.columns
    ]

    X_train = train_df[available_cols].fillna(0).values
    y_train = train_df["lap_time_s"].values
    X_test = test_df[available_cols].fillna(0).values
    y_test = test_df["lap_time_s"].values

    try:
        import lightgbm as lgb

        model = lgb.LGBMRegressor(
            n_estimators=300,
            max_depth=7,
            learning_rate=0.03,
            random_state=42,
            verbosity=-1,
        )
    except ImportError:
        from sklearn.ensemble import GradientBoostingRegressor

        model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            random_state=42,
        )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
    r2 = 1 - np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)
    within_0_5 = np.mean(np.abs(y_test - y_pred) < 0.5) * 100

    return {
        "rmse": round(rmse, 2),
        "r2": round(r2, 3),
        "within_0_5s_pct": round(within_0_5, 1),
    }


def _build_winner_features(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build per-driver per-race features for winner prediction."""
    race_features = []
    labels = []

    for race, race_df in df.groupby("race"):
        for driver, driver_df in race_df.groupby("driver_code"):
            final_laps = driver_df[
                driver_df["lap_number"] == driver_df["lap_number"].max()
            ]
            if final_laps.empty:
                continue

            final_pos = int(final_laps.iloc[0].get("position", 0))
            first_lap = driver_df[driver_df["lap_number"] == 1]
            grid_pos = (
                int(first_lap.iloc[0].get("position", 10))
                if not first_lap.empty
                else 10
            )
            avg_lap = driver_df["lap_time_s"].mean()

            race_features.append([grid_pos, avg_lap])
            labels.append(1 if final_pos == 1 else 0)

    X = np.array(race_features)
    y = np.array(labels)
    feature_names = ["grid_position", "avg_lap_time"]

    return X, y, feature_names


def evaluate_winner_model(train_years: list[int], test_year: int) -> dict:
    """Evaluate winner prediction model with real train/test split."""
    train_df, test_df = _load_feature_store(
        train_years, test_year, include_pit_laps=True
    )

    if train_df.empty or test_df.empty:
        return {"error": "No data available for evaluation"}

    X_train, y_train, _ = _build_winner_features(train_df)
    X_test, y_test, _ = _build_winner_features(test_df)

    if len(X_test) == 0:
        return {"error": "No winner data available"}

    from sklearn.ensemble import RandomForestClassifier, VotingClassifier
    from xgboost import XGBClassifier
    from sklearn.metrics import accuracy_score, brier_score_loss

    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=15,
        random_state=42,
        eval_metric="logloss",
    )
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
    )

    ensemble = VotingClassifier(estimators=[("xgb", xgb), ("rf", rf)], voting="soft")
    ensemble.fit(X_train, y_train)

    y_pred = ensemble.predict(X_test)
    y_prob = ensemble.predict_proba(X_test)[:, 1]

    top1 = accuracy_score(y_test, y_pred)

    top3 = 0
    for i in range(len(y_test)):
        if y_test[i] == 1:
            if y_prob[i] >= sorted(y_prob, reverse=True)[min(2, len(y_prob) - 1)]:
                top3 += 1
    top3 = top3 / max(1, y_test.sum())

    brier = brier_score_loss(y_test, y_prob)

    return {
        "top1_accuracy": round(top1, 3),
        "top3_accuracy": round(top3, 3),
        "brier_score": round(brier, 4),
    }


def evaluate_pit_policy(train_years: list[int], test_year: int) -> dict:
    """Evaluate pit policy model with real train/test split."""
    train_df, test_df = _load_feature_store(
        train_years, test_year, include_pit_laps=True
    )

    if train_df.empty or test_df.empty:
        return {"error": "No data available for evaluation"}

    features = [
        "lap_number",
        " tyre_age_in_stint",
        "position",
        "track_temp",
        "rainfall",
    ]
    available_cols = [
        c for c in features if c in train_df.columns and c in test_df.columns
    ]

    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df["pit_label"] = train_df["pit_lap"].astype(int)
    test_df["pit_label"] = test_df["pit_lap"].astype(int)

    X_train = train_df[available_cols].fillna(0).values
    y_train = train_df["pit_label"].values
    X_test = test_df[available_cols].fillna(0).values
    y_test = test_df["pit_label"].values

    if y_test.sum() == 0 or y_train.sum() == 0:
        return {"error": "No pit events in data"}

    from xgboost import XGBClassifier
    from sklearn.metrics import roc_auc_score, f1_score, precision_score

    model = XGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=len(y_train) / max(1, y_train.sum()),
        random_state=42,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)

    return {
        "auc_roc": round(auc, 3),
        "f1": round(f1, 3),
        "precision": round(precision, 3),
    }


def evaluate_all(train_years: list[int], test_year: int) -> dict:
    """Run all model evaluations and return summary."""
    coverage = load_dataset_coverage()

    results = {
        "coverage": coverage,
        " tyre_deg": evaluate_tyre_model(train_years, test_year),
        "laptime": evaluate_laptime_model(train_years, test_year),
        "winner": evaluate_winner_model(train_years, test_year),
        "pit_policy": evaluate_pit_policy(train_years, test_year),
    }

    return results


def print_evaluation_results(results: dict) -> None:
    """Print evaluation results in rich table format."""
    console.print("\n[bold green]Model Evaluation Results (Phase 6)[/bold green]\n")

    coverage = results.get("coverage", {})
    console.print(
        f"[dim]Dataset: {coverage.get('years', [])}, {coverage.get('total_races', 0)} races, {coverage.get('total_rows', 0)} rows[/dim]\n"
    )

    table = Table(title="Model Performance")
    table.add_column("Model", style="cyan")
    table.add_column("Metric", style="magenta")
    table.add_column("Score", justify="right")
    table.add_column("Status", justify="center")

    tyre = results.get(" tyre_deg", {})
    if "rmse" in tyre:
        table.add_row(
            "Tyre Deg",
            "RMSE",
            f"{tyre['rmse']}s",
            "[green]✓[/green]" if tyre["rmse"] < 10 else "[yellow]~[/yellow]",
        )
        table.add_row(
            "",
            "R²",
            f"{tyre['r2']}",
            "[green]✓[/green]" if tyre["r2"] > 0.1 else "[yellow]~[/yellow]",
        )

    lap = results.get("laptime", {})
    if "rmse" in lap:
        table.add_row(
            "Lap Time",
            "RMSE",
            f"{lap['rmse']}s",
            "[green]✓[/green]" if lap["rmse"] < 2 else "[yellow]~[/yellow]",
        )
        table.add_row(
            "",
            "R²",
            f"{lap['r2']}",
            "[green]✓[/green]" if lap["r2"] > 0.3 else "[yellow]~[/yellow]",
        )
        table.add_row(
            "",
            "Within 0.5s",
            f"{lap['within_0_5s_pct']}%",
            "[green]✓[/green]" if lap["within_0_5s_pct"] > 50 else "[yellow]~[/yellow]",
        )

    winner = results.get("winner", {})
    if "top1_accuracy" in winner:
        table.add_row(
            "Winner",
            "Top-1",
            f"{winner['top1_accuracy']:.1%}",
            "[green]✓[/green]"
            if winner["top1_accuracy"] > 0.25
            else "[yellow]~[/yellow]",
        )
        table.add_row(
            "",
            "Top-3",
            f"{winner['top3_accuracy']:.1%}",
            "[green]✓[/green]"
            if winner["top3_accuracy"] > 0.5
            else "[yellow]~[/yellow]",
        )
        table.add_row(
            "",
            "Brier",
            f"{winner['brier_score']}",
            "[green]✓[/green]" if winner["brier_score"] < 0.2 else "[yellow]~[/yellow]",
        )

    pit = results.get("pit_policy", {})
    if "auc_roc" in pit:
        table.add_row(
            "Pit Policy",
            "AUC-ROC",
            f"{pit['auc_roc']}",
            "[green]✓[/green]" if pit["auc_roc"] > 0.7 else "[yellow]~[/yellow]",
        )
        table.add_row(
            "",
            "F1",
            f"{pit['f1']}",
            "[green]✓[/green]" if pit["f1"] > 0.3 else "[yellow]~[/yellow]",
        )

    console.print(table)


@click.command()
@click.option("--train-years", default="2022,2023,2024")
@click.option("--test-year", default="2025")
@click.option(
    "--all", "show_all", is_flag=True, help="Show all results without test split"
)
def main(train_years: str, test_year: str, show_all: bool) -> None:
    """Evaluate trained models with real train/test split."""
    train_yrs = [int(y) for y in train_years.split(",") if y.strip()]
    test_yr = int(test_year) if not show_all else None

    console.print(f"[bold]Phase 6: Real Data Evaluation[/bold]")
    console.print(f"[dim]Train: {train_yrs}, Test: {test_yr or 'N/A'}[/dim]\n")

    if show_all:
        results = evaluate_all(train_yrs, train_yrs[-1])
    else:
        results = evaluate_all(train_yrs, test_yr)

    print_evaluation_results(results)


if __name__ == "__main__":
    main()
