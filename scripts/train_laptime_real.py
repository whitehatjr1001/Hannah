"""Train lap-time predictor using real historical F1 data from parquet feature store.

Uses LightGBM for fast tabular prediction with SHAP explainability.

Usage:
    python3 scripts/train_laptime_real.py --train-years 2022,2023 --test-year 2024
    python3 scripts/train_laptime_real.py --all
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
ARTIFACT_PATH = Path("models/saved/laptime_v1.pkl")

VALID_COMPOUNDS = {"SOFT", "MEDIUM", "HARD"}


@dataclass
class LapTimeV2Artifact:
    """Lap-time prediction model artifact."""

    version: str = "v2"
    model: bytes = b""
    feature_names: list = field(default_factory=list)
    train_years: tuple = field(default_factory=tuple)
    train_races: tuple = field(default_factory=tuple)
    rmse: float = 0.0
    r2: float = 0.0
    shap_summary: bytes = b""


def _load_feature_store(
    train_years: list[int] | None = None,
    test_years: list[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load parquet files from feature store, split into train/test."""
    all_files = sorted(FEATURE_STORE_ROOT.rglob("race_features.parquet"))
    if not all_files:
        console.print("[red]No parquet files found in data/feature_store/[/red]")
        raise SystemExit(1)

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
        df = df[~df["pit_lap"]]
        df = df.dropna(subset=["lap_time_s"])

        if train_years and year in train_years:
            train_parts.append(df)
        elif test_years and year in test_years:
            test_parts.append(df)
        elif not train_years and not test_years:
            train_parts.append(df)

    if not train_parts:
        console.print("[red]No training data found[/red]")
        raise SystemExit(1)

    train_df = pd.concat(train_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True) if test_parts else pd.DataFrame()

    console.print(
        f"[dim]Loaded {len(train_df):,} train rows, {len(test_df):,} test rows[/dim]"
    )
    return train_df, test_df


def _build_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build feature matrix and target for lap-time prediction."""
    df = df.copy()

    # Race median as circuit baseline
    race_medians = df.groupby("race")["lap_time_s"].median()
    df["race_median"] = df["race"].map(race_medians)

    features = pd.DataFrame(
        {
            "tyre_age_in_stint": df["tyre_age_in_stint"].fillna(0),
            "stint_number": df["stint_number"].fillna(1),
            "lap_number": df["lap_number"].fillna(1),
            "track_temp": df["track_temp"].fillna(30.0),
            "air_temp": df["air_temp"].fillna(25.0),
            "rainfall": df["rainfall"].fillna(0.0),
            "race_median": df["race_median"].fillna(95.0),
        }
    )

    # One-hot encode compound
    compound_dummies = pd.get_dummies(df["compound"], prefix="compound", dtype=float)
    for col in compound_dummies.columns:
        features[col] = compound_dummies[col]

    target = df["lap_time_s"].values
    return features.values, target, list(features.columns)


@click.command()
@click.option(
    "--train-years", default="2022,2023", help="Comma-separated training years"
)
@click.option("--test-year", default="2024", help="Test year (held out)")
@click.option("--all", "use_all", is_flag=True, help="Use all available data")
def train(train_years: str, test_year: str, use_all: bool) -> None:
    """Train lap-time predictor on real F1 data."""
    train_yrs = [int(y.strip()) for y in train_years.split(",") if y.strip()]
    test_yrs = [int(test_year)] if test_year and not use_all else []

    console.print(
        f"[bold green]Lap-Time Predictor Trainer (v2 — LightGBM)[/bold green]"
    )
    console.print(
        f"[dim]Train years: {train_yrs}, Test years: {test_yrs if test_yrs else 'N/A'}[/dim]"
    )
    console.print()

    train_df, test_df = _load_feature_store(
        train_years=train_yrs if not use_all else None,
        test_years=test_yrs if not use_all else None,
    )

    if use_all:
        test_df = train_df.iloc[:0]

    X_train, y_train, feature_names = _build_features(train_df)
    console.print(f"[dim]Features: {feature_names}[/dim]")
    console.print(f"[dim]Train shape: {X_train.shape}[/dim]")

    # Train LightGBM
    try:
        import lightgbm as lgb
    except ImportError:
        console.print(
            "[yellow]lightgbm not installed, falling back to GradientBoosting[/yellow]"
        )
        from sklearn.ensemble import GradientBoostingRegressor

        model = GradientBoostingRegressor(
            n_estimators=500,
            max_depth=7,
            learning_rate=0.03,
            subsample=0.8,
            min_samples_leaf=20,
            random_state=42,
        )
        model.fit(X_train, y_train)
        model_bytes = pickle.dumps(model)
        shap_values = b""
    else:
        model = lgb.LGBMRegressor(
            n_estimators=500,
            max_depth=7,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=20,
            random_state=42,
            verbose=-1,
        )
        model.fit(X_train, y_train)
        model_bytes = pickle.dumps(model)

        # SHAP summary
        try:
            import shap

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_train[:1000])
            shap_summary = pickle.dumps(
                {
                    "values": shap_values,
                    "feature_names": feature_names,
                }
            )
        except Exception:
            shap_summary = b""

    # Evaluate
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    if not test_df.empty:
        X_test, y_test, _ = _build_features(test_df)
        y_pred = model.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae = float(mean_absolute_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred))
        pct_within_05 = float(np.mean(np.abs(y_test - y_pred) < 0.5))

        console.print()
        table = Table(title="Lap-Time Predictor — Evaluation")
        table.add_column("Metric", style="cyan")
        table.add_column("Score", justify="right")
        table.add_column("Status", justify="center")

        rmse_status = "[green]PASS[/green]" if rmse < 1.0 else "[yellow]WARN[/yellow]"
        table.add_row("RMSE (s)", f"{rmse:.4f}", rmse_status)
        table.add_row("MAE (s)", f"{mae:.4f}", "")
        r2_status = "[green]PASS[/green]" if r2 > 0.7 else "[yellow]WARN[/yellow]"
        table.add_row("R²", f"{r2:.4f}", r2_status)
        table.add_row(
            "% within 0.5s",
            f"{pct_within_05:.1%}",
            "[green]PASS[/green]" if pct_within_05 > 0.8 else "[yellow]WARN[/yellow]",
        )

        console.print(table)

        # Feature importance
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            console.print("[dim]Feature importance:[/dim]")
            for name, imp in sorted(
                zip(feature_names, importances), key=lambda x: -x[1]
            ):
                bar = "█" * int(imp / max(importances) * 30)
                console.print(f"  [dim]{name:25s} {imp:.0f} {bar}[/dim]")
    else:
        y_pred = model.predict(X_train)
        rmse = float(np.sqrt(mean_squared_error(y_train, y_pred)))
        r2 = float(r2_score(y_train, y_pred))
        console.print(f"[dim]Train RMSE: {rmse:.4f}, R²: {r2:.4f}[/dim]")

    # Save artifact
    artifact = LapTimeV2Artifact(
        version="v2",
        model=model_bytes,
        feature_names=feature_names,
        train_years=tuple(train_yrs),
        train_races=tuple(sorted(train_df["race"].unique().tolist())),
        rmse=rmse,
        r2=r2,
        shap_summary=shap_summary,
    )

    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with ARTIFACT_PATH.open("wb") as f:
        pickle.dump(artifact, f)

    console.print()
    console.print(f"[green]✓[/green] Artifact saved to [bold]{ARTIFACT_PATH}[/bold]")


if __name__ == "__main__":
    train()
