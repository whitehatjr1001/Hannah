# Real Historical Data Training Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a cached historical-data pipeline for FastF1/OpenF1 session ingestion, normalize practice/qualifying/race features into parquet datasets, and retrain Hannah's baseline artifacts from those datasets instead of synthetic generators.

**Architecture:** Add a dedicated historical dataset layer that fetches and caches raw sessions, normalizes them into stable parquet tables, and exposes those tables to the existing trainer entrypoints. Keep runtime inference artifact-only, keep public model names unchanged, and gate historical windows so results-era and telemetry-era trainers use the right source coverage.

**Tech Stack:** Python, pandas, FastF1, OpenF1 REST, parquet, existing Hannah model artifact helpers, pytest.

---

### Task 1: Add Historical Cache And Manifest Primitives

**Files:**
- Create: `hannah/data/historical_cache.py`
- Modify: `config.yaml`
- Test: `tests/data/test_historical_cache.py`

- [ ] **Step 1: Write the failing test**

```python
def test_historical_cache_builds_deterministic_raw_and_normalized_paths() -> None:
    cache = HistoricalCache(root=Path("data/historical_cache"))
    assert cache.raw_session_path(year=2024, event="bahrain", session_type="R").as_posix().endswith(
        "data/historical_cache/raw/2024/bahrain/R.json"
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/data/test_historical_cache.py -v`
Expected: FAIL because the cache helper does not exist.

- [ ] **Step 3: Write minimal implementation**

```python
@dataclass(frozen=True)
class HistoricalCache:
    root: Path

    def raw_session_path(self, year: int, event: str, session_type: str) -> Path:
        return self.root / "raw" / str(year) / event / f"{session_type}.json"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/data/test_historical_cache.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/data/test_historical_cache.py hannah/data/historical_cache.py config.yaml
git commit -m "feat: add historical cache path primitives"
```

### Task 2: Build Raw Historical Session Fetcher

**Files:**
- Create: `hannah/data/historical_dataset.py`
- Modify: `hannah/_data_/fastf1_loader.py`
- Modify: `hannah/_data_/openf1_client.py`
- Test: `tests/data/test_historical_dataset_builder.py`

- [ ] **Step 1: Write the failing test**

```python
def test_builder_fetches_all_requested_session_types(monkeypatch, tmp_path) -> None:
    builder = HistoricalDatasetBuilder(cache=HistoricalCache(tmp_path))
    seen = []
    monkeypatch.setattr(builder, "_fetch_fastf1_session", lambda **kwargs: seen.append(kwargs["session_type"]) or {"ok": True})
    monkeypatch.setattr(builder, "_fetch_openf1_enrichment", lambda **kwargs: {"drivers": []})

    builder.build_raw(years=[2024], events=["bahrain"], session_types=["FP1", "FP2", "FP3", "Q", "R"])

    assert seen == ["FP1", "FP2", "FP3", "Q", "R"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/data/test_historical_dataset_builder.py::test_builder_fetches_all_requested_session_types -v`
Expected: FAIL because the builder does not exist.

- [ ] **Step 3: Write minimal implementation**

```python
class HistoricalDatasetBuilder:
    def build_raw(self, years: list[int], events: list[str], session_types: list[str]) -> None:
        for year in years:
            for event in events:
                for session_type in session_types:
                    payload = self._fetch_fastf1_session(year=year, event=event, session_type=session_type)
                    self.cache.write_raw_session(year=year, event=event, session_type=session_type, payload=payload)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/data/test_historical_dataset_builder.py::test_builder_fetches_all_requested_session_types -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/data/test_historical_dataset_builder.py hannah/data/historical_dataset.py hannah/_data_/fastf1_loader.py hannah/_data_/openf1_client.py
git commit -m "feat: add raw historical session builder"
```

### Task 3: Add OpenF1 Gating And Provenance Recording

**Files:**
- Modify: `hannah/data/historical_dataset.py`
- Modify: `hannah/_data_/openf1_client.py`
- Test: `tests/data/test_openf1_historical_gating.py`

- [ ] **Step 1: Write the failing test**

```python
def test_builder_skips_openf1_for_unsupported_historical_year(monkeypatch, tmp_path) -> None:
    builder = HistoricalDatasetBuilder(cache=HistoricalCache(tmp_path))
    calls = []
    monkeypatch.setattr(builder, "_fetch_openf1_enrichment", lambda **kwargs: calls.append(kwargs) or {"drivers": []})

    builder._maybe_fetch_openf1(year=2019, event="bahrain", session_type="R")

    assert calls == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/data/test_openf1_historical_gating.py -v`
Expected: FAIL because the gating helper does not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
def _maybe_fetch_openf1(self, year: int, event: str, session_type: str) -> dict[str, object] | None:
    if year < self.openf1_min_year:
        return None
    return self._fetch_openf1_enrichment(year=year, event=event, session_type=session_type)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/data/test_openf1_historical_gating.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/data/test_openf1_historical_gating.py hannah/data/historical_dataset.py hannah/_data_/openf1_client.py
git commit -m "feat: gate openf1 enrichment in historical builds"
```

### Task 4: Normalize Results Features Into Parquet

**Files:**
- Create: `hannah/data/feature_builders/results_features.py`
- Modify: `hannah/data/historical_dataset.py`
- Test: `tests/data/test_historical_schema_contracts.py`

- [ ] **Step 1: Write the failing test**

```python
def test_results_feature_builder_emits_expected_columns() -> None:
    frame = build_results_features(raw_session={"results": [{"Abbreviation": "VER"}]}, year=2024, event="bahrain", session_type="R")
    assert {"year", "event", "session_type", "driver", "team", "grid_position", "finish_position"} <= set(frame.columns)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/data/test_historical_schema_contracts.py::test_results_feature_builder_emits_expected_columns -v`
Expected: FAIL because the feature builder does not exist.

- [ ] **Step 3: Write minimal implementation**

```python
def build_results_features(raw_session: dict[str, object], year: int, event: str, session_type: str) -> pd.DataFrame:
    rows = raw_session.get("results", [])
    return pd.DataFrame(
        {
            "year": year,
            "event": event,
            "session_type": session_type,
            "driver": row.get("Abbreviation"),
            "team": row.get("TeamName"),
        }
        for row in rows
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/data/test_historical_schema_contracts.py::test_results_feature_builder_emits_expected_columns -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/data/test_historical_schema_contracts.py hannah/data/feature_builders/results_features.py hannah/data/historical_dataset.py
git commit -m "feat: normalize historical results features"
```

### Task 5: Normalize Telemetry Features Into Parquet

**Files:**
- Create: `hannah/data/feature_builders/telemetry_features.py`
- Modify: `hannah/data/historical_dataset.py`
- Test: `tests/data/test_historical_schema_contracts.py`

- [ ] **Step 1: Write the failing test**

```python
def test_telemetry_feature_builder_emits_lap_level_columns() -> None:
    frame = build_telemetry_features(raw_session={"laps": [{"Driver": "VER", "LapNumber": 1}]}, year=2024, event="bahrain", session_type="FP1")
    assert {"year", "event", "session_type", "driver", "lap_number", "lap_time_s", "compound"} <= set(frame.columns)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/data/test_historical_schema_contracts.py::test_telemetry_feature_builder_emits_lap_level_columns -v`
Expected: FAIL because the builder does not exist.

- [ ] **Step 3: Write minimal implementation**

```python
def build_telemetry_features(raw_session: dict[str, object], year: int, event: str, session_type: str) -> pd.DataFrame:
    return pd.DataFrame(...)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/data/test_historical_schema_contracts.py::test_telemetry_feature_builder_emits_lap_level_columns -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/data/test_historical_schema_contracts.py hannah/data/feature_builders/telemetry_features.py hannah/data/historical_dataset.py
git commit -m "feat: normalize historical telemetry features"
```

### Task 6: Normalize Pit Features Into Parquet

**Files:**
- Create: `hannah/data/feature_builders/pit_features.py`
- Modify: `hannah/data/historical_dataset.py`
- Test: `tests/data/test_historical_schema_contracts.py`

- [ ] **Step 1: Write the failing test**

```python
def test_pit_feature_builder_emits_strategy_columns() -> None:
    frame = build_pit_features(raw_session={"laps": [], "stints": [], "positions": []}, year=2024, event="bahrain", session_type="R")
    assert {"driver", "lap_number", "pit_in", "pit_out", "gap_ahead_s", "gap_behind_s"} <= set(frame.columns)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/data/test_historical_schema_contracts.py::test_pit_feature_builder_emits_strategy_columns -v`
Expected: FAIL because the pit builder does not exist.

- [ ] **Step 3: Write minimal implementation**

```python
def build_pit_features(raw_session: dict[str, object], year: int, event: str, session_type: str) -> pd.DataFrame:
    return pd.DataFrame(...)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/data/test_historical_schema_contracts.py::test_pit_feature_builder_emits_strategy_columns -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/data/test_historical_schema_contracts.py hannah/data/feature_builders/pit_features.py hannah/data/historical_dataset.py
git commit -m "feat: normalize historical pit features"
```

### Task 7: Wire Dataset Build CLI And Manifest Reporting

**Files:**
- Modify: `hannah/cli/app.py`
- Modify: `hannah/data/historical_dataset.py`
- Test: `tests/cli/test_historical_dataset_commands.py`

- [ ] **Step 1: Write the failing test**

```python
def test_dataset_build_command_accepts_full_session_matrix(runner) -> None:
    result = runner.invoke(app, ["dataset", "build", "--years", "2018,2019", "--sessions", "FP1,FP2,FP3,Q,R"])
    assert result.exit_code == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/cli/test_historical_dataset_commands.py -v`
Expected: FAIL because the CLI command does not exist.

- [ ] **Step 3: Write minimal implementation**

```python
@app.group()
def dataset() -> None:
    ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/cli/test_historical_dataset_commands.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/cli/test_historical_dataset_commands.py hannah/cli/app.py hannah/data/historical_dataset.py
git commit -m "feat: add historical dataset build cli"
```

### Task 8: Rework Winner Training To Consume Real Results Dataset

**Files:**
- Modify: `hannah/models/train_winner.py`
- Modify: `hannah/models/evaluate.py`
- Test: `tests/models/test_real_data_trainers.py`

- [ ] **Step 1: Write the failing test**

```python
def test_winner_trainer_reads_cached_results_dataset(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(train_winner, "load_results_dataset", lambda **_: fixture_results_frame())
    output = train_winner.train(years=[2022, 2023, 2024], races=["bahrain"])
    assert output.endswith("winner_ensemble_v1.pkl")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/models/test_real_data_trainers.py::test_winner_trainer_reads_cached_results_dataset -v`
Expected: FAIL because the trainer still uses the synthetic builder.

- [ ] **Step 3: Write minimal implementation**

```python
def train(years: list[int], races: list[str] | None = None) -> str:
    dataset = load_results_dataset(years=years, races=races)
    ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/models/test_real_data_trainers.py::test_winner_trainer_reads_cached_results_dataset -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/models/test_real_data_trainers.py hannah/models/train_winner.py hannah/models/evaluate.py
git commit -m "feat: train winner baseline from cached results data"
```

### Task 9: Rework Tyre And Lap-Time Trainers To Consume Real Telemetry Dataset

**Files:**
- Modify: `hannah/models/train_tyre_deg.py`
- Modify: `hannah/models/train_laptime.py`
- Modify: `hannah/models/evaluate.py`
- Test: `tests/models/test_real_data_trainers.py`

- [ ] **Step 1: Write the failing test**

```python
def test_tyre_and_laptime_trainers_read_cached_telemetry_dataset(monkeypatch) -> None:
    monkeypatch.setattr(train_tyre_deg, "load_telemetry_dataset", lambda **_: fixture_telemetry_frame())
    monkeypatch.setattr(train_laptime, "load_telemetry_dataset", lambda **_: fixture_telemetry_frame())
    assert train_tyre_deg.train(years=[2024]).endswith("tyre_deg_v1.pkl")
    assert train_laptime.train(years=[2024]).endswith("laptime_v1.pkl")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/models/test_real_data_trainers.py::test_tyre_and_laptime_trainers_read_cached_telemetry_dataset -v`
Expected: FAIL because the trainers still use `build_telemetry_baseline`.

- [ ] **Step 3: Write minimal implementation**

```python
def _load_training_frame(years: list[int], races: list[str] | None) -> pd.DataFrame:
    return load_telemetry_dataset(years=years, races=races)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/models/test_real_data_trainers.py::test_tyre_and_laptime_trainers_read_cached_telemetry_dataset -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/models/test_real_data_trainers.py hannah/models/train_tyre_deg.py hannah/models/train_laptime.py hannah/models/evaluate.py
git commit -m "feat: train tyre and laptime models from cached telemetry data"
```

### Task 10: Rework Pit Policy Trainers To Consume Real Pit Dataset

**Files:**
- Modify: `hannah/models/train_pit_q.py`
- Modify: `hannah/models/train_pit_rl.py`
- Test: `tests/models/test_real_data_trainers.py`

- [ ] **Step 1: Write the failing test**

```python
def test_pit_trainers_read_cached_pit_dataset(monkeypatch) -> None:
    monkeypatch.setattr(train_pit_q, "load_pit_dataset", lambda **_: fixture_pit_frame())
    monkeypatch.setattr(train_pit_rl, "load_pit_dataset", lambda **_: fixture_pit_frame())
    assert train_pit_q.train(years=[2024]).endswith("pit_policy_q_v1.pkl")
    assert train_pit_rl.train(years=[2024]).endswith("pit_rl_v1.zip")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/models/test_real_data_trainers.py::test_pit_trainers_read_cached_pit_dataset -v`
Expected: FAIL because those trainers do not consume cached pit features yet.

- [ ] **Step 3: Write minimal implementation**

```python
def load_pit_dataset(years: list[int], races: list[str] | None = None) -> pd.DataFrame:
    return load_cached_pit_features(...)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/models/test_real_data_trainers.py::test_pit_trainers_read_cached_pit_dataset -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/models/test_real_data_trainers.py hannah/models/train_pit_q.py hannah/models/train_pit_rl.py
git commit -m "feat: train pit models from cached pit data"
```

### Task 11: Add Coverage-Aware Evaluation And Dataset Summary

**Files:**
- Modify: `hannah/models/evaluate.py`
- Modify: `hannah/tools/train_model/tool.py`
- Test: `tests/models/test_real_data_evaluation.py`

- [ ] **Step 1: Write the failing test**

```python
def test_evaluation_reports_dataset_coverage(monkeypatch) -> None:
    monkeypatch.setattr(evaluate, "load_dataset_manifest", lambda: {"years": [2018, 2019, 2020], "sessions_built": 15})
    result = evaluate.evaluate_model("winner_ensemble")
    assert "coverage" in result["evaluation_depth"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/models/test_real_data_evaluation.py -v`
Expected: FAIL because evaluation only reports synthetic scenario depth today.

- [ ] **Step 3: Write minimal implementation**

```python
def _load_dataset_coverage() -> dict[str, object]:
    ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/models/test_real_data_evaluation.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/models/test_real_data_evaluation.py hannah/models/evaluate.py hannah/tools/train_model/tool.py
git commit -m "feat: report historical dataset coverage in evaluation"
```

### Task 12: Run End-To-End Historical Build, Train, And Eval Lane

**Files:**
- Modify: `docs/superpowers/specs/2026-04-02-real-historical-data-training-design.md`
- Modify: `docs/superpowers/plans/2026-04-02-real-historical-data-training.md`
- Test: `tests/data/`
- Test: `tests/models/`
- Test: `tests/cli/test_historical_dataset_commands.py`

- [ ] **Step 1: Run the focused historical data test lane**

Run: `pytest tests/data tests/models/test_real_data_trainers.py tests/models/test_real_data_evaluation.py tests/cli/test_historical_dataset_commands.py -q`
Expected: PASS

- [ ] **Step 2: Build a bounded historical dataset cache**

Run: `python hannah.py dataset build --years 2022,2023,2024 --sessions FP1,FP2,FP3,Q,R --events bahrain,monza,silverstone`
Expected: completes with a manifest and normalized parquet outputs under `data/historical_cache/`

- [ ] **Step 3: Train all baseline artifacts from cached datasets**

Run: `python hannah.py train all --years 2022,2023,2024 --races bahrain,monza,silverstone`
Expected: writes refreshed artifacts under `models/saved/`

- [ ] **Step 4: Run evaluation**

Run: `python -c "from hannah.models.evaluate import evaluate_model; print(evaluate_model('winner_ensemble'))"`
Expected: returns artifact metrics plus dataset coverage

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/specs/2026-04-02-real-historical-data-training-design.md docs/superpowers/plans/2026-04-02-real-historical-data-training.md
git commit -m "docs: finalize real historical data training plan"
```
