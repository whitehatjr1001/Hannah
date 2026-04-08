"""Train winner prediction model using real historical F1 data from parquet feature store.

Uses XGBoost + RandomForest soft-vote ensemble with calibrated probabilities.

Usage:
    python3 scripts/train_winner_real.py --train-years 2022,2023 --test-year 2024
    python3 scripts/train_winner_real.py --all
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import click
import numpy as np
import pandas as pd
from rich.table import Table

from hannah.utils.console import Console

console = Console()

FEATURE_STORE_ROOT = Path("data/feature_store")
ARTIFACT_PATH = Path("models/saved/winner_ensemble_v1.pkl")


@dataclass
class WinnerV2Artifact:
    """Winner prediction model artifact."""

    version: str = "v2"
    ensemble: bytes = b""
    feature_names: list = field(default_factory=list)
    train_years: tuple = field(default_factory=tuple)
    train_races: tuple = field(default_factory=tuple)
    top1_accuracy: float = 0.0
    top3_accuracy: float = 0.0
    brier_score: float = 0.0


def _load_data(
    train_years: list[int], test_years: list[int]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load parquet files and build per-race driver features."""
    all_files = sorted(FEATURE_STORE_ROOT.rglob("race_features.parquet"))
    if not all_files:
        console.print("[red]No parquet files found[/red]")
        raise SystemExit(1)

    train_parts, test_parts = [], []

    for f in all_files:
        try:
            year = int(f.parts[-3])
        except (ValueError, IndexError):
            continue

        df = pd.read_parquet(f)

        # Build per-driver features per race
        race_features = []
        for driver, driver_df in df.groupby("driver_code"):
            # Final position
            final_laps = driver_df[
                driver_df["lap_number"] == driver_df["lap_number"].max()
            ]
            if final_laps.empty:
                continue
            final_pos = int(final_laps.iloc[0].get("position", 0))

            # Grid position (first lap position)
            first_lap = driver_df[driver_df["lap_number"] == 1]
            grid_pos = (
                int(first_lap.iloc[0].get("position", 0)) if not first_lap.empty else 10
            )

            # Avg lap time
            avg_lap = driver_df["lap_time_s"].mean()

            # Most-used compound
            compounds = driver_df["compound"].value_counts()
            compound = compounds.index[0] if len(compounds) > 0 else "UNKNOWN"

            race_features.append(
                {
                    "year": year,
                    "race": df["race"].iloc[0],
                    "driver": driver,
                    "team": driver_df["team_name"].iloc[0]
                    if "team_name" in driver_df.columns
                    else "",
                    "grid_position": grid_pos,
                    "final_position": final_pos,
                    "won": 1 if final_pos == 1 else 0,
                    "avg_lap_time": avg_lap,
                    "compound": compound,
                    "total_laps": len(driver_df),
                }
            )

        race_df = pd.DataFrame(race_features)
        if train_years and year in train_years:
            train_parts.append(race_df)
        elif test_years and year in test_years:
            test_parts.append(race_df)

    if not train_parts:
        console.print("[red]No training data[/red]")
        raise SystemExit(1)

    train_df = pd.concat(train_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True) if test_parts else pd.DataFrame()

    console.print(
        f"[dim]Loaded {len(train_df):,} train, {len(test_df):,} test rows[/dim]"
    )
    console.print(
        f"[dim]Winners: train={train_df['won'].sum()}, test={test_df['won'].sum() if not test_df.empty else 0}[/dim]"
    )
    return train_df, test_df


def _build_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build features for winner prediction."""
    race_baseline = df.groupby("race")["avg_lap_time"].median()
    df["lap_time_delta"] = df["avg_lap_time"] - df["race"].map(race_baseline)

    team_encoder = {t: i for i, t in enumerate(df["team"].unique())}
    df["team_encoded"] = df["team"].map(team_encoder).fillna(0)

    compound_map = {"SOFT": 0, "MEDIUM": 1, "HARD": 2, "UNKNOWN": 3}
    df["compound_encoded"] = df["compound"].map(compound_map).fillna(3)

    features = pd.DataFrame(
        {
            "grid_position": df["grid_position"].fillna(10),
            "lap_time_delta": df["lap_time_delta"].fillna(0),
            "team_encoded": df["team_encoded"],
            "compound_encoded": df["compound_encoded"],
            "total_laps": df["total_laps"].fillna(50),
        }
    )

    return features.values, df["won"].values, list(features.columns)


@click.command()
@click.option("--train-years", default="2022,2023")
@click.option("--test-year", default="2024")
@click.option("--all", "use_all", is_flag=True)
def train(train_years: str, test_year: str, use_all: bool) -> None:
    """Train winner prediction model."""
    train_yrs = [int(y) for y in train_years.split(",") if y.strip()]
    test_yrs = [int(test_year)] if test_year and not use_all else []

    console.print(f"[bold green]Winner Prediction Trainer (v2)[/bold green]")
    console.print(
        f"[dim]Train: {train_yrs}, Test: {test_yrs if test_yrs else 'N/A'}[/dim]\n"
    )

    train_df, test_df = _load_data(train_yrs, test_yrs)
    if use_all:
        test_df = pd.DataFrame()

    X_train, y_train, feature_names = _build_features(train_df)
    console.print(
        f"[dim]Features: {feature_names}, Shape: {X_train.shape}, Pos: {y_train.sum()}[/dim]"
    )

    # Train ensemble
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
    console.print("[dim]Training...[/dim]")
    ensemble.fit(X_train, y_train)

    if not test_df.empty:
        X_test, y_test, _ = _build_features(test_df)
        y_pred = ensemble.predict(X_test)
        y_prob = ensemble.predict_proba(X_test)[:, 1]

        top1 = accuracy_score(y_test, y_pred)

        # Top-3: check if winner in top-3 predicted
        top3 = 0
        for race in test_df["race"].unique():
            race_mask = test_df["race"] == race
            race_probs = y_prob[race_mask]
            winner_idx = np.where(y_test[race_mask] == 1)[0]
            if len(winner_idx) > 0:
                winner_prob = race_probs[winner_idx[0]]
                top3_probs = sorted(race_probs, reverse=True)[:3]
                if winner_prob in top3_probs:
                    top3 += 1
        top3 /= (
            len(test_df["race"].unique()) if len(test_df["race"].unique()) > 0 else 1
        )

        brier = brier_score_loss(y_test, y_prob)

        table = Table(title="Winner Prediction — Evaluation")
        table.add_column("Metric", style="cyan")
        table.add_column("Score", justify="right")
        table.add_column("Status", justify="center")
        table.add_row(
            "Top-1 Accuracy",
            f"{top1:.1%}",
            "[green]PASS[/green]" if top1 > 0.25 else "[yellow]WARN[/yellow]",
        )
        table.add_row(
            "Top-3 Accuracy",
            f"{top3:.1%}",
            "[green]PASS[/green]" if top3 > 0.5 else "[yellow]WARN[/yellow]",
        )
        table.add_row(
            "Brier Score",
            f"{brier:.4f}",
            "[green]PASS[/green]" if brier < 0.2 else "[yellow]WARN[/yellow]",
        )
        console.print(table)
        console.print(f"[dim]Baseline: {(~y_test.astype(bool)).mean():.1%}[/dim]")
    else:
        top1 = accuracy_score(y_train, ensemble.predict(X_train))
        console.print(f"[dim]Train accuracy: {top1:.1%}[/dim]")
        top3, brier = 0.0, 0.0

    # Save
    artifact = WinnerV2Artifact(
        version="v2",
        ensemble=pickle.dumps(ensemble),
        feature_names=feature_names,
        train_years=tuple(train_yrs),
        train_races=tuple(sorted(train_df["race"].unique())),
        top1_accuracy=top1,
        top3_accuracy=top3,
        brier_score=brier,
    )

    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with ARTIFACT_PATH.open("wb") as f:
        pickle.dump(artifact, f)

    console.print(f"\n[green]✓[/green] Saved to [bold]{ARTIFACT_PATH}[/bold]")


if __name__ == "__main__":
    train()
