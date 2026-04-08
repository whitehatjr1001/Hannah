"""Train pit stop policy using real historical F1 data from parquet feature store.

Two-stage approach:
  1. When to pit — XGBClassifier for binary pit probability
  2. What compound — Q-learning table trained on real pit outcomes

Usage:
    python3 scripts/train_pit_policy_real.py --train-years 2022,2023 --test-year 2024
    python3 scripts/train_pit_policy_real.py --all
"""

from __future__ import annotations

import json
import pickle
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import click
import numpy as np
import pandas as pd
from rich.table import Table

from hannah.utils.console import Console

console = Console()

FEATURE_STORE_ROOT = Path("data/feature_store")
ARTIFACT_PATH = Path("models/saved/pit_rl_v1.zip")

VALID_COMPOUNDS = {"SOFT", "MEDIUM", "HARD"}
COMPOUND_MAP = {"SOFT": 0, "MEDIUM": 1, "HARD": 2}

TYRE_AGE_BINS = 12  # 0-1, 2-3, 4-5, ..., 22+
SC_STATES = 2  # no safety car, safety car
Q_ACTIONS = ["stay_out", "pit_soft", "pit_medium", "pit_hard"]


@dataclass(frozen=True)
class PitPolicyArtifact:
    """Serialized pit stop policy (two-stage)."""

    version: str = "v1"
    train_years: tuple = field(default_factory=tuple)
    train_races: tuple = field(default_factory=tuple)
    classifier_bytes: bytes = b""
    feature_names: list = field(default_factory=list)
    q_table: bytes = b""
    q_state_dims: tuple = field(default_factory=tuple)
    auc_roc: float = 0.0
    f1_score: float = 0.0
    precision: float = 0.0
    recall: float = 0.0


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
        df = df.dropna(subset=["lap_time_s"])

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


def _build_pit_classifier_features(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build features for the pit-when classifier.

    Target: pit_lap (boolean) — was this lap a pit-in lap?
    Features: lap_number, tyre_age_in_stint, compound, track_temp, rainfall,
              stint_length_so_far, laps_remaining, position
    """
    df = df.copy()

    # Compute stint length so far: max tyre_age_in_stint per (driver, race, stint)
    stint_max = df.groupby(["driver_code", "race", "stint_number"])[
        "tyre_age_in_stint"
    ].transform("max")
    df["stint_length_so_far"] = df["tyre_age_in_stint"].clip(upper=stint_max)

    # Laps remaining: total laps in race minus current lap
    race_total_laps = df.groupby(["race"])["lap_number"].transform("max")
    df["laps_remaining"] = (race_total_laps - df["lap_number"]).clip(lower=0)

    # Compound encoding
    df["compound_encoded"] = df["compound"].map(COMPOUND_MAP).fillna(1).astype(int)

    # Safety car flag
    if "safety_car" in df.columns:
        df["safety_car_flag"] = df["safety_car"].astype(int)
    else:
        df["safety_car_flag"] = 0

    features = pd.DataFrame(
        {
            "lap_number": df["lap_number"].fillna(1),
            "tyre_age_in_stint": df["tyre_age_in_stint"].fillna(0),
            "compound_encoded": df["compound_encoded"],
            "track_temp": df["track_temp"].fillna(30.0),
            "rainfall": df["rainfall"].fillna(0.0),
            "stint_length_so_far": df["stint_length_so_far"].fillna(0),
            "laps_remaining": df["laps_remaining"].fillna(0),
            "position": df["position"].fillna(10),
            "safety_car_flag": df["safety_car_flag"],
        }
    )

    target = df["pit_lap"].astype(int).values
    feature_names = list(features.columns)

    return features.values, target, feature_names


def _extract_pit_transitions(df: pd.DataFrame) -> pd.DataFrame:
    """Extract real pit transitions for Q-table training.

    For each pit lap, capture the pre-pit state and the post-pit outcome.
    """
    df = df.copy()
    df = df.sort_values(["driver_code", "race", "lap_number"])

    transitions: list[dict] = []

    for (driver, race), group in df.groupby(["driver_code", "race"]):
        group = group.sort_values("lap_number").reset_index(drop=True)
        total_laps = int(group["lap_number"].max())

        for idx, row in group.iterrows():
            if not row.get("pit_lap", False):
                continue

            # Pre-pit state
            tyre_age = float(row.get("tyre_age_in_stint", 0))
            compound = str(row.get("compound", "UNKNOWN"))
            safety_car = bool(row.get("safety_car", False))
            position_before = int(row.get("position", 0))
            lap_before = int(row.get("lap_number", 0))
            lap_time_before = float(row.get("lap_time_s", 0))

            if compound not in VALID_COMPOUNDS:
                continue
            if lap_time_before <= 0 or lap_time_before > 200:
                continue

            # Find post-pit lap (next lap by same driver after pit)
            post_pit_rows = group[
                (group["lap_number"] > lap_before)
                & (group["stint_number"] > row.get("stint_number", 1))
            ]
            if post_pit_rows.empty:
                continue

            first_post = post_pit_rows.iloc[0]
            position_after = int(first_post.get("position", position_before))
            lap_time_after = float(first_post.get("lap_time_s", lap_time_before))
            new_compound = str(first_post.get("compound", "UNKNOWN"))

            # Reward: position gained + lap time improvement
            position_gain = (
                position_before - position_after
            )  # positive = gained positions
            if lap_time_after > 0 and lap_time_before > 0:
                time_improvement = (lap_time_before - lap_time_after) / lap_time_before
            else:
                time_improvement = 0.0

            reward = position_gain * 0.5 + time_improvement * 2.0

            action_idx = (
                COMPOUND_MAP.get(new_compound, 1) + 1
            )  # 1=soft, 2=medium, 3=hard

            transitions.append(
                {
                    "tyre_age": tyre_age,
                    "compound": compound,
                    "safety_car": safety_car,
                    "position_before": position_before,
                    "lap_before": lap_before,
                    "action_idx": action_idx,
                    "reward": reward,
                    "position_gain": position_gain,
                    "time_improvement": time_improvement,
                }
            )

    return pd.DataFrame(transitions)


def _build_q_table(
    transitions: pd.DataFrame,
    alpha: float = 0.1,
    gamma: float = 0.9,
    n_episodes: int = 50,
) -> np.ndarray:
    """Build Q-table from real pit transitions.

    State: (tyre_age_bin, compound, safety_car)
    Action: [stay_out, pit_soft, pit_medium, pit_hard]
    """
    q_table = np.zeros(
        (TYRE_AGE_BINS, len(COMPOUND_MAP), SC_STATES, len(Q_ACTIONS)),
        dtype=np.float64,
    )

    if transitions.empty:
        console.print(
            "[yellow]No pit transitions found — returning zero Q-table[/yellow]"
        )
        return q_table

    # Discretize and accumulate rewards
    for _, t in transitions.iterrows():
        age_bin = min(int(t["tyre_age"]), TYRE_AGE_BINS - 1)
        comp_idx = COMPOUND_MAP.get(t["compound"], 1)
        sc_idx = 1 if t["safety_car"] else 0
        action = int(t["action_idx"])

        q_table[age_bin, comp_idx, sc_idx, action] += t["reward"]

    # Normalize by visit counts
    visit_counts = np.maximum((q_table != 0).sum(axis=-1, keepdims=True), 1)
    q_table = np.where(q_table != 0, q_table / visit_counts, q_table)

    # Fill unseen states with small random exploration bonus
    rng = np.random.default_rng(42)
    unseen_mask = q_table.sum(axis=-1) == 0
    # For unseen states, set all actions to small random values
    for age in range(TYRE_AGE_BINS):
        for comp in range(len(COMPOUND_MAP)):
            for sc in range(SC_STATES):
                if unseen_mask[age, comp, sc]:
                    q_table[age, comp, sc] = rng.uniform(
                        -0.01, 0.01, size=len(Q_ACTIONS)
                    )

    # Run a few Q-learning sweeps to propagate values
    for _ in range(n_episodes):
        for age in range(TYRE_AGE_BINS):
            for comp in range(len(COMPOUND_MAP)):
                for sc in range(SC_STATES):
                    for action in range(len(Q_ACTIONS)):
                        if q_table[age, comp, sc, action] == 0:
                            continue
                        # Bootstrap to next age
                        next_age = min(age + 1, TYRE_AGE_BINS - 1)
                        best_future = np.max(q_table[next_age, comp, sc])
                        current = q_table[age, comp, sc, action]
                        q_table[age, comp, sc, action] = current + alpha * (
                            gamma * best_future - current
                        )

    return q_table


def _evaluate_classifier(
    model, X_test: np.ndarray, y_test: np.ndarray
) -> dict[str, float]:
    """Compute classification metrics."""
    from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    auc = float(roc_auc_score(y_test, y_prob))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))

    return {"auc_roc": auc, "f1": f1, "precision": prec, "recall": rec}


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
    """Train pit stop policy on real F1 data."""
    train_yrs = [int(y.strip()) for y in train_years.split(",") if y.strip()]
    test_yrs = [int(test_year)] if test_year and not use_all else []

    console.print(f"[bold green]Pit Policy Trainer (v1 — two-stage)[/bold green]")
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

    # === Stage 1: Pit classifier ===
    console.print("[bold]Stage 1: When to pit (XGBClassifier)[/bold]")

    X_train, y_train, feature_names = _build_pit_classifier_features(train_df)
    console.print(f"[dim]Features: {feature_names}[/dim]")
    console.print(f"[dim]Train shape: {X_train.shape}[/dim]")

    pit_ratio = y_train.mean()
    console.print(f"[dim]Pit-lap ratio: {pit_ratio:.3f}[/dim]")

    scale_pos_weight = (1 - pit_ratio) / max(pit_ratio, 1e-6)
    console.print(f"[dim]scale_pos_weight: {scale_pos_weight:.1f}[/dim]")

    try:
        import xgboost as xgb
    except ImportError:
        console.print("[red]xgboost not installed[/red]")
        raise SystemExit(1)

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="logloss",
        use_label_encoder=False,
    )
    model.fit(X_train, y_train)

    # Feature importance
    importances = model.feature_importances_
    console.print("[dim]Feature importance:[/dim]")
    for name, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
        bar = "█" * int(imp * 40)
        console.print(f"  [dim]{name:25s} {imp:.4f} {bar}[/dim]")

    # Evaluate
    metrics: dict[str, float] = {}
    if not test_df.empty:
        X_test, y_test, _ = _build_pit_classifier_features(test_df)
        metrics = _evaluate_classifier(model, X_test, y_test)

        console.print()
        table = Table(title="Pit Classifier — Evaluation")
        table.add_column("Metric", style="cyan")
        table.add_column("Score", justify="right")
        table.add_column("Status", justify="center")

        auc_status = (
            "[green]PASS[/green]"
            if metrics["auc_roc"] > 0.7
            else "[yellow]WARN[/yellow]"
        )
        table.add_row("AUC-ROC", f"{metrics['auc_roc']:.4f}", auc_status)

        f1_status = (
            "[green]PASS[/green]" if metrics["f1"] > 0.3 else "[yellow]WARN[/yellow]"
        )
        table.add_row("F1", f"{metrics['f1']:.4f}", f1_status)

        table.add_row("Precision", f"{metrics['precision']:.4f}", "")
        table.add_row("Recall", f"{metrics['recall']:.4f}", "")

        console.print(table)
    else:
        metrics = _evaluate_classifier(model, X_train, y_train)
        console.print(
            f"[dim]Train AUC-ROC: {metrics['auc_roc']:.4f}, F1: {metrics['f1']:.4f}[/dim]"
        )

    # === Stage 2: Q-table from real pit transitions ===
    console.print()
    console.print("[bold]Stage 2: What compound (Q-table from real transitions)[/bold]")

    all_data = (
        pd.concat([train_df, test_df], ignore_index=True)
        if not test_df.empty
        else train_df
    )
    transitions = _extract_pit_transitions(all_data)

    console.print(f"[dim]Extracted {len(transitions):,} pit transitions[/dim]")

    if not transitions.empty:
        console.print("[dim]Reward distribution:[/dim]")
        console.print(
            f"[dim]  mean={transitions['reward'].mean():.3f}, "
            f"std={transitions['reward'].std():.3f}, "
            f"median={transitions['reward'].median():.3f}[/dim]"
        )

        # Show transition breakdown by compound chosen
        action_counts = transitions["action_idx"].value_counts().sort_index()
        for action_idx, count in action_counts.items():
            action_name = (
                Q_ACTIONS[action_idx]
                if action_idx < len(Q_ACTIONS)
                else f"unknown_{action_idx}"
            )
            console.print(f"[dim]  {action_name}: {count} transitions[/dim]")

    q_table = _build_q_table(transitions)
    console.print(f"[dim]Q-table shape: {q_table.shape}[/dim]")

    # Save artifact
    classifier_bytes = pickle.dumps(model)
    q_table_bytes = q_table.tobytes()

    metadata = {
        "version": "v1",
        "train_years": train_yrs,
        "train_races": sorted(all_data["race"].unique().tolist()),
        "feature_names": feature_names,
        "q_state_dims": list(q_table.shape),
        "q_actions": Q_ACTIONS,
        "compound_map": COMPOUND_MAP,
        "tyre_age_bins": TYRE_AGE_BINS,
        "n_transitions": len(transitions),
        "metrics": metrics,
    }

    artifact = PitPolicyArtifact(
        version="v1",
        train_years=tuple(train_yrs),
        train_races=tuple(sorted(all_data["race"].unique().tolist())),
        classifier_bytes=classifier_bytes,
        feature_names=feature_names,
        q_table=q_table_bytes,
        q_state_dims=tuple(q_table.shape),
        auc_roc=metrics.get("auc_roc", 0.0),
        f1_score=metrics.get("f1", 0.0),
        precision=metrics.get("precision", 0.0),
        recall=metrics.get("recall", 0.0),
    )

    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(
        ARTIFACT_PATH, mode="w", compression=zipfile.ZIP_DEFLATED
    ) as zf:
        zf.writestr("pit_classifier.pkl", classifier_bytes)
        zf.writestr("q_table.npy", q_table_bytes)
        zf.writestr("metadata.json", json.dumps(metadata, indent=2, default=str))

    console.print()
    console.print(f"[green]✓[/green] Artifact saved to [bold]{ARTIFACT_PATH}[/bold]")
    console.print(
        f"[dim]  Races: {len(artifact.train_races)}, Years: {artifact.train_years}[/dim]"
    )
    console.print(
        f"[dim]  Contents: pit_classifier.pkl, q_table.npy, metadata.json[/dim]"
    )


if __name__ == "__main__":
    train()
