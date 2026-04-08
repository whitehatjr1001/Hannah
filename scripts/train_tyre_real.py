"""Train tyre degradation model using real historical F1 data from parquet feature store.

This model predicts lap time from tyre age, compound, weather, and circuit context.
The tyre degradation signal is learned as part of the full lap-time model.

Usage:
    python3 scripts/train_tyre_real.py --train-years 2022,2023 --test-year 2024
    python3 scripts/train_tyre_real.py --all
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import click
import numpy as np
import pandas as pd
from rich.table import Table

from hannah.utils.console import Console

console = Console()

FEATURE_STORE_ROOT = Path("data/feature_store")
ARTIFACT_PATH = Path("models/saved/tyre_deg_v1.pkl")

VALID_COMPOUNDS = {"SOFT", "MEDIUM", "HARD"}


@dataclass
class TyreDegV2Artifact:
    """Tyre degradation model artifact."""

    version: str = "v2"
    gbr_model: bytes = b""
    physics_params: dict = field(default_factory=dict)
    feature_names: list = field(default_factory=list)
    train_years: tuple = field(default_factory=tuple)
    train_races: tuple = field(default_factory=tuple)
    rmse: float = 0.0
    r2: float = 0.0
    median_lap_times: dict = field(default_factory=dict)


def _load_feature_store(
    train_years: list[int] | None = None,
    test_years: list[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load parquet files from feature store, split into train/test."""
    all_files = sorted(FEATURE_STORE_ROOT.rglob("race_features.parquet"))
    if not all_files:
        console.print("[red]No parquet files found in data/feature_store/[/red]")
        console.print(
            "[dim]Run: python3 scripts/fetch_openf1_features.py --years 2022,2023,2024[/dim]"
        )
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
        df = df.dropna(subset=["lap_time_s"])

        # Drop pit laps (outliers from pit entry/exit)
        df = df[~df["pit_lap"]]

        if train_years and year in train_years:
            train_parts.append(df)
        elif test_years and year in test_years:
            test_parts.append(df)
        elif not train_years and not test_years:
            train_parts.append(df)

    if not train_parts:
        console.print("[red]No training data found for specified years[/red]")
        raise SystemExit(1)

    train_df = pd.concat(train_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True) if test_parts else pd.DataFrame()

    console.print(
        f"[dim]Loaded {len(train_df):,} train rows, {len(test_df):,} test rows[/dim]"
    )
    return train_df, test_df


def _build_features(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, float]]:
    """Build feature matrix and target vector.

    Predicts absolute lap time. The model learns the combined effect of
    tyre degradation + fuel burn + weather + circuit baseline.
    """
    df = df.copy()

    # Circuit baseline: median lap time per race
    race_medians = df.groupby("race")["lap_time_s"].median()
    df["race_median"] = df["race"].map(race_medians)

    # Target: absolute lap time
    target = df["lap_time_s"].values

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

    feature_names = list(features.columns)

    # Store race medians for inference
    median_lap_times = race_medians.to_dict()

    return features.values, target, feature_names, median_lap_times


def _fit_physics_params(df: pd.DataFrame) -> dict:
    """Fit power-law tyre degradation parameters per compound.

    Model: penalty = (tyre_age / cliff_lap)^power * scale + offset

    Uses residuals from driver-race median to isolate degradation.
    """
    from scipy.optimize import curve_fit

    def power_law(age, cliff_lap, power, scale, offset):
        normalized = np.clip(age / max(cliff_lap, 1), 0, 10)
        return np.power(normalized, power) * scale + offset

    df = df.copy()
    medians = df.groupby(["driver_code", "race"])["lap_time_s"].median()
    df["residual"] = df.apply(
        lambda r: (
            r["lap_time_s"]
            - medians.get((r["driver_code"], r["race"]), r["lap_time_s"])
        ),
        axis=1,
    )

    physics_params: dict[str, dict[str, float]] = {}

    for compound in VALID_COMPOUNDS:
        mask = df["compound"] == compound
        if mask.sum() < 50:
            continue

        sub = df[mask]
        ages = sub["tyre_age_in_stint"].values.astype(float)
        residuals = sub["residual"].values.astype(float)
        # Only use positive residuals (degradation, not improvement from fuel)
        # Focus on later laps in stints where fuel effect is smaller
        late_mask = ages > 5
        if late_mask.sum() < 20:
            continue

        ages_late = ages[late_mask]
        residuals_late = residuals[late_mask]
        # Clip to remove outliers
        p5, p95 = np.percentile(residuals_late, [5, 95])
        clean_mask = (residuals_late >= p5) & (residuals_late <= p95)
        ages_clean = ages_late[clean_mask]
        residuals_clean = residuals_late[clean_mask]

        try:
            popt, _ = curve_fit(
                power_law,
                ages_clean,
                residuals_clean,
                p0=[20.0, 1.8, 3.0, 0.1],
                bounds=([5.0, 0.5, 0.1, -2.0], [60.0, 4.0, 20.0, 10.0]),
                maxfev=5000,
            )
            physics_params[compound] = {
                "cliff_lap": round(float(popt[0]), 2),
                "power": round(float(popt[1]), 3),
                "scale": round(float(popt[2]), 3),
                "offset": round(float(popt[3]), 4),
            }
        except Exception:
            physics_params[compound] = {
                "cliff_lap": 20.0,
                "power": 1.8,
                "scale": 3.0,
                "offset": 0.1,
            }

    return physics_params


def _evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, float]:
    """Compute evaluation metrics."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    y_pred = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))

    return {"rmse": rmse, "mae": mae, "r2": r2}


def _per_compound_rmse(
    model, X_test: np.ndarray, y_test: np.ndarray, df_test: pd.DataFrame
) -> dict[str, float]:
    """Compute RMSE per compound."""
    from sklearn.metrics import mean_squared_error

    y_pred = model.predict(X_test)
    results: dict[str, float] = {}
    for compound in VALID_COMPOUNDS:
        mask = df_test["compound"] == compound
        if mask.sum() < 10:
            continue
        rmse = float(np.sqrt(mean_squared_error(y_test[mask], y_pred[mask])))
        results[compound] = rmse
    return results


@click.command()
@click.option(
    "--train-years", default="2022,2023", help="Comma-separated training years"
)
@click.option("--test-year", default="2024", help="Test year (held out)")
@click.option(
    "--all",
    "use_all",
    is_flag=True,
    help="Use all available data for training (no test split)",
)
def train(train_years: str, test_year: str, use_all: bool) -> None:
    """Train tyre degradation model on real F1 data."""
    train_yrs = [int(y.strip()) for y in train_years.split(",") if y.strip()]
    test_yrs = [int(test_year)] if test_year and not use_all else []

    console.print(f"[bold green]Tyre Degradation Model Trainer (v2)[/bold green]")
    console.print(
        f"[dim]Train years: {train_yrs}, Test years: {test_yrs if test_yrs else 'N/A (all data)'}[/dim]"
    )
    console.print()

    # Load data
    train_df, test_df = _load_feature_store(
        train_years=train_yrs if not use_all else None,
        test_years=test_yrs if not use_all else None,
    )

    if use_all:
        test_df = train_df.iloc[:0]

    # Build features
    X_train, y_train, feature_names, median_lap_times = _build_features(train_df)
    console.print(f"[dim]Features: {feature_names}[/dim]")
    console.print(f"[dim]Train shape: {X_train.shape}[/dim]")
    console.print(f"[dim]Target: absolute lap time (s)[/dim]")
    console.print(
        f"[dim]Target range: [{y_train.min():.1f}, {y_train.max():.1f}], mean: {y_train.mean():.1f}[/dim]"
    )
    console.print(
        f"[dim]Target range: [{y_train.min():.2f}, {y_train.max():.2f}], mean: {y_train.mean():.2f}, std: {y_train.std():.2f}[/dim]"
    )

    # Fit physics parameters
    console.print("[dim]Fitting physics parameters...[/dim]")
    physics_params = _fit_physics_params(train_df)
    for compound, params in sorted(physics_params.items()):
        console.print(
            f"  [dim]{compound}: cliff_lap={params['cliff_lap']}, power={params['power']}, scale={params['scale']}, offset={params['offset']}[/dim]"
        )

    # Train GradientBoostingRegressor
    from sklearn.ensemble import GradientBoostingRegressor

    console.print("[dim]Training GradientBoostingRegressor...[/dim]")
    model = GradientBoostingRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=10,
        random_state=42,
    )
    model.fit(X_train, y_train)

    # Feature importance
    importances = model.feature_importances_
    console.print("[dim]Feature importance:[/dim]")
    for name, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
        bar = "█" * int(imp * 40)
        console.print(f"  [dim]{name:25s} {imp:.4f} {bar}[/dim]")

    # Evaluate
    metrics: dict[str, Any] = {}
    if not test_df.empty:
        X_test, y_test, _, _ = _build_features(test_df)
        metrics = _evaluate_model(model, X_test, y_test)
        per_compound = _per_compound_rmse(model, X_test, y_test, test_df)

        console.print()
        table = Table(title="Tyre Degradation Model — Evaluation (residuals)")
        table.add_column("Metric", style="cyan")
        table.add_column("Score", justify="right")
        table.add_column("Status", justify="center")

        rmse_status = (
            "[green]PASS[/green]" if metrics["rmse"] < 1.0 else "[yellow]WARN[/yellow]"
        )
        table.add_row("RMSE (s)", f"{metrics['rmse']:.4f}", rmse_status)
        table.add_row("MAE (s)", f"{metrics['mae']:.4f}", "")
        r2_status = (
            "[green]PASS[/green]" if metrics["r2"] > 0.3 else "[yellow]WARN[/yellow]"
        )
        table.add_row("R²", f"{metrics['r2']:.4f}", r2_status)

        for compound, crmse in sorted(per_compound.items()):
            table.add_row(f"RMSE {compound} (s)", f"{crmse:.4f}", "")

        console.print(table)
    else:
        metrics = _evaluate_model(model, X_train, y_train)
        console.print(
            f"[dim]Train RMSE: {metrics['rmse']:.4f}, R²: {metrics['r2']:.4f}[/dim]"
        )

    # Save artifact
    artifact = TyreDegV2Artifact(
        version="v2",
        gbr_model=pickle.dumps(model),
        physics_params=physics_params,
        feature_names=feature_names,
        train_years=tuple(train_yrs),
        train_races=tuple(sorted(train_df["race"].unique().tolist())),
        rmse=metrics.get("rmse", 0.0),
        r2=metrics.get("r2", 0.0),
        median_lap_times={str(k): float(v) for k, v in median_lap_times.items()},
    )

    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with ARTIFACT_PATH.open("wb") as f:
        pickle.dump(artifact, f)

    console.print()
    console.print(f"[green]✓[/green] Artifact saved to [bold]{ARTIFACT_PATH}[/bold]")
    console.print(
        f"[dim]  Races: {len(artifact.train_races)}, Years: {artifact.train_years}[/dim]"
    )


if __name__ == "__main__":
    train()
