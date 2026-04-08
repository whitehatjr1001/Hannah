"""Tests for evaluate_models.py - Phase 6 real train/test evaluation."""

import pandas as pd
from pathlib import Path


def test_evaluate_loads_feature_store_coverage():
    """Evaluation reports dataset coverage from feature store."""
    from scripts.evaluate_models import load_dataset_coverage

    coverage = load_dataset_coverage()
    assert "years" in coverage
    assert "total_races" in coverage
    assert "total_rows" in coverage


def test_evaluate_tyre_model_returns_rmse_and_r2():
    """Tyre model evaluation returns RMSE, R², per-compound metrics."""
    from scripts.evaluate_models import evaluate_tyre_model

    result = evaluate_tyre_model(train_years=[2022, 2023], test_year=2024)

    assert "rmse" in result
    assert "r2" in result
    assert "per_compound" in result


def test_evaluate_laptime_model_returns_rmse_and_r2():
    """Lap time model evaluation returns RMSE, R², within_0_5s."""
    from scripts.evaluate_models import evaluate_laptime_model

    result = evaluate_laptime_model(train_years=[2022, 2023], test_year=2024)

    assert "rmse" in result
    assert "r2" in result
    assert "within_0_5s_pct" in result


def test_evaluate_winner_model_returns_top_accuracy_and_brier():
    """Winner model evaluation returns Top-1, Top-3, Brier score."""
    from scripts.evaluate_models import evaluate_winner_model

    result = evaluate_winner_model(train_years=[2022, 2023], test_year=2024)

    assert "top1_accuracy" in result
    assert "top3_accuracy" in result
    assert "brier_score" in result


def test_evaluate_pit_policy_returns_auc_roc_and_f1():
    """Pit policy evaluation returns AUC-ROC, F1, precision."""
    from scripts.evaluate_models import evaluate_pit_policy

    result = evaluate_pit_policy(train_years=[2022, 2023], test_year=2024)

    assert "auc_roc" in result
    assert "f1" in result
    assert "precision" in result


def test_evaluate_all_runs_all_models():
    """Evaluate all runs all 4 model evaluations and returns summary."""
    from scripts.evaluate_models import evaluate_all

    results = evaluate_all(train_years=[2022, 2023], test_year=2024)

    assert " tyre_deg" in results
    assert "laptime" in results
    assert "winner" in results
    assert "pit_policy" in results
    assert "coverage" in results
