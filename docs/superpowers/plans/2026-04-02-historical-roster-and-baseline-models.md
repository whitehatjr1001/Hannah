# Historical Roster And Baseline Models Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a season-aware roster resolver backed by FastF1 and OpenF1, rewire the main F1 tools to use resolved historical entrants, and replace smoke-only shipped artifacts with real baseline model builders.

**Architecture:** Add a resolver layer in the existing `hannah/_data_/` package that merges FastF1 session facts, OpenF1 recent-session enrichment, and `teams.py` heuristic metadata into one runtime roster object. Propagate that object through the agent context, simulation state, and subagent paths before rewiring `race_data`, `race_sim`, `pit_strategy`, and `predict_winner`, then add dataset builders and baseline trainers that preserve the current public model/tool names while producing stronger shipped artifacts plus optional user-trained variants.

**Tech Stack:** Python, FastF1, OpenF1 REST, existing Hannah tool registry and agent loop, pytest, pickle/zip artifacts, deterministic simulation helpers.

---

### Task 1: Define The Resolver Types And Contracts

**Files:**
- Create: `hannah/domain/resolved_roster.py`
- Modify: `hannah/domain/__init__.py`
- Test: `tests/domain/test_resolved_roster.py`

- [ ] **Step 1: Write the failing test**

```python
from hannah.domain.resolved_roster import ResolvedDriverProfile, ResolvedRoster


def test_resolved_roster_groups_drivers_by_team() -> None:
    roster = ResolvedRoster(
        year=2025,
        race="bahrain",
        session="R",
        drivers=(
            ResolvedDriverProfile(code="VER", full_name="Max Verstappen", team="Red Bull Racing"),
            ResolvedDriverProfile(code="NOR", full_name="Lando Norris", team="McLaren"),
        ),
        source="fastf1",
    )

    assert roster.driver_codes == ("VER", "NOR")
    assert roster.team_names == ("McLaren", "Red Bull Racing")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/domain/test_resolved_roster.py -v`
Expected: FAIL with import or attribute errors because the types do not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
@dataclass(frozen=True)
class ResolvedDriverProfile:
    code: str
    full_name: str
    team: str
    teammate: str | None = None
    car_number: str | None = None
    source: str = "unknown"
    strategy_style: str | None = None


@dataclass(frozen=True)
class ResolvedRoster:
    year: int
    race: str
    session: str
    drivers: tuple[ResolvedDriverProfile, ...]
    source: str

    @property
    def driver_codes(self) -> tuple[str, ...]:
        return tuple(driver.code for driver in self.drivers)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/domain/test_resolved_roster.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/domain/test_resolved_roster.py hannah/domain/resolved_roster.py hannah/domain/__init__.py
git commit -m "feat: add resolved roster domain types"
```

### Task 2: Add Runtime Carrier Support For `resolved_roster`

**Files:**
- Modify: `hannah/agent/context.py`
- Modify: `hannah/agent/prompts.py`
- Modify: `hannah/domain/race_state.py`
- Test: `tests/agent/test_resolved_roster_context.py`

- [ ] **Step 1: Write the failing test**

```python
def test_race_context_carries_resolved_roster() -> None:
    roster = ResolvedRoster(year=2025, race="bahrain", session="R", drivers=(), source="fastf1")
    ctx = RaceContext(race="bahrain", year=2025, laps=57, weather="dry", drivers=["VER"], resolved_roster=roster)
    assert ctx.resolved_roster is roster
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/agent/test_resolved_roster_context.py -v`
Expected: FAIL because the runtime context does not carry `resolved_roster`.

- [ ] **Step 3: Write minimal implementation**

```python
@dataclass
class RaceContext:
    ...
    resolved_roster: ResolvedRoster | None = None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/agent/test_resolved_roster_context.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/agent/test_resolved_roster_context.py hannah/agent/context.py hannah/agent/prompts.py hannah/domain/race_state.py
git commit -m "feat: carry resolved roster through runtime context"
```

### Task 3: Add Season Roster Resolver With 2026-Only Fallback

**Files:**
- Create: `hannah/_data_/season_roster_resolver.py`
- Modify: `hannah/domain/teams.py`
- Test: `tests/data/test_season_roster_resolver.py`

- [ ] **Step 1: Write the failing test**

```python
def test_resolver_uses_teams_fallback_only_for_2026(monkeypatch) -> None:
    resolver = SeasonRosterResolver()

    monkeypatch.setattr(resolver, "_resolve_from_fastf1", lambda **_: None)
    monkeypatch.setattr(resolver, "_resolve_from_openf1", lambda **_: None)

    roster_2026 = resolver.resolve(year=2026, race="bahrain", session="R")
    assert roster_2026.source == "teams_fallback"

    with pytest.raises(RosterUnavailableError):
        resolver.resolve(year=2024, race="bahrain", session="R")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/data/test_season_roster_resolver.py -v`
Expected: FAIL because the resolver does not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
class SeasonRosterResolver:
    def resolve(self, year: int, race: str, session: str) -> ResolvedRoster:
        fastf1 = self._resolve_from_fastf1(year=year, race=race, session=session)
        if fastf1 is not None:
            return fastf1
        openf1 = self._resolve_from_openf1(year=year, race=race, session=session)
        if openf1 is not None:
            return openf1
        if year == 2026:
            return self._resolve_from_teams_fallback(year=year, race=race, session=session)
        raise RosterUnavailableError(f"no roster data for {year} {race} {session}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/data/test_season_roster_resolver.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/data/test_season_roster_resolver.py hannah/_data_/season_roster_resolver.py hannah/domain/teams.py
git commit -m "feat: add season roster resolver with 2026 fallback"
```

### Task 4: Enrich Resolved Roster With FastF1 And OpenF1 Metadata

**Files:**
- Modify: `hannah/_data_/season_roster_resolver.py`
- Modify: `hannah/_data_/openf1_client.py`
- Modify: `hannah/_data_/fastf1_loader.py`
- Test: `tests/data/test_season_roster_resolution_sources.py`

- [ ] **Step 1: Write the failing test**

```python
def test_resolver_merges_openf1_driver_numbers_into_fastf1_roster(monkeypatch) -> None:
    resolver = SeasonRosterResolver()
    monkeypatch.setattr(resolver, "_load_fastf1_results", lambda **_: [{"Abbreviation": "VER", "TeamName": "Red Bull Racing"}])
    monkeypatch.setattr(resolver, "_load_openf1_drivers", lambda **_: [{"name_acronym": "VER", "driver_number": 1}])

    roster = resolver.resolve(year=2025, race="bahrain", session="R")
    assert roster.get("VER").car_number == "1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/data/test_season_roster_resolution_sources.py -v`
Expected: FAIL because merge logic is incomplete.

- [ ] **Step 3: Write minimal implementation**

```python
def _merge_openf1_driver_metadata(self, roster: ResolvedRoster, openf1_drivers: list[dict]) -> ResolvedRoster:
    by_code = {str(row.get("name_acronym", "")).upper(): row for row in openf1_drivers}
    merged = []
    for driver in roster.drivers:
        row = by_code.get(driver.code, {})
        merged.append(replace(driver, car_number=str(row.get("driver_number")) if row.get("driver_number") else driver.car_number))
    return replace(roster, drivers=tuple(merged))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/data/test_season_roster_resolution_sources.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/data/test_season_roster_resolution_sources.py hannah/_data_/season_roster_resolver.py hannah/_data_/openf1_client.py hannah/_data_/fastf1_loader.py
git commit -m "feat: merge fastf1 and openf1 roster metadata"
```

### Task 5: Rework `teams.py` Into Metadata Catalog And 2026 Fallback

**Files:**
- Modify: `hannah/domain/teams.py`
- Modify: `hannah/domain/prompts.py`
- Test: `tests/domain/test_team_catalog.py`
- Test: `tests/agent/test_current_f1_grid.py`

- [ ] **Step 1: Write the failing test**

```python
def test_team_catalog_returns_prompt_metadata_without_owning_historical_pairings() -> None:
    profile = get_team_metadata("VER")
    assert profile.strategy_style
    assert profile.code == "VER"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/domain/test_team_catalog.py -v`
Expected: FAIL because the metadata accessor does not exist.

- [ ] **Step 3: Write minimal implementation**

```python
def get_team_metadata(driver_code: str) -> TeamInfo:
    canonical = canonical_driver_code(driver_code)
    return TEAM_METADATA[canonical]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/domain/test_team_catalog.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/domain/test_team_catalog.py tests/agent/test_current_f1_grid.py hannah/domain/teams.py hannah/domain/prompts.py
git commit -m "refactor: split team metadata from season roster ownership"
```

### Task 6: Rewire Simulation And Worker Surfaces Off Static Grid Assumptions

**Files:**
- Modify: `hannah/simulation/sandbox.py`
- Modify: `hannah/simulation/monte_carlo.py`
- Modify: `hannah/agent/subagents.py`
- Modify: `hannah/agent/worker_registry.py`
- Test: `tests/simulation/test_resolved_roster_runtime.py`

- [ ] **Step 1: Write the failing test**

```python
def test_rival_agent_uses_resolved_roster_teammate_data() -> None:
    roster = build_test_roster(...)
    ctx = RaceContext(..., resolved_roster=roster)
    result = asyncio.run(RivalAgent("NOR").run(ctx))
    assert result.success is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/simulation/test_resolved_roster_runtime.py -v`
Expected: FAIL because simulation and rival paths still assume the static grid.

- [ ] **Step 3: Write minimal implementation**

```python
# Thread resolved_roster through RaceState builders and rival/team persona selection.
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/simulation/test_resolved_roster_runtime.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/simulation/test_resolved_roster_runtime.py hannah/simulation/sandbox.py hannah/simulation/monte_carlo.py hannah/agent/subagents.py hannah/agent/worker_registry.py
git commit -m "feat: propagate resolved rosters through simulation and subagents"
```

### Task 7: Make `race_data` Return Resolved Roster And Session Context

**Files:**
- Modify: `hannah/tools/race_data/tool.py`
- Modify: `hannah/_data_/season_roster_resolver.py`
- Test: `tests/tools/test_race_data_tool.py`

- [ ] **Step 1: Write the failing test**

```python
@pytest.mark.asyncio
async def test_race_data_tool_returns_resolved_roster(monkeypatch) -> None:
    monkeypatch.setattr("hannah.tools.race_data.tool.resolve_roster", lambda **_: {"drivers": ["VER", "NOR"], "source": "fastf1"})
    payload = await run(race="bahrain", year=2025, session="R")
    assert payload["resolved_roster"]["source"] == "fastf1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/tools/test_race_data_tool.py -v`
Expected: FAIL because `resolved_roster` is missing.

- [ ] **Step 3: Write minimal implementation**

```python
resolved_roster = resolver.resolve(year=year, race=race, session=session)
return {
    "laps": ...,
    "weather": ...,
    "drivers": list(resolved_roster.driver_codes),
    "resolved_roster": resolved_roster.to_dict(),
    "session_info": session_info,
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/tools/test_race_data_tool.py -v`
Expected: PASS

Compatibility check:
- confirm the existing top-level payload keys still exist
- confirm `resolved_roster` is additive rather than replacing acceptance-facing fields

- [ ] **Step 5: Commit**

```bash
git add tests/tools/test_race_data_tool.py hannah/tools/race_data/tool.py hannah/_data_/season_roster_resolver.py
git commit -m "feat: return resolved roster from race_data tool"
```

### Task 8: Rewire `race_sim` To Use Resolved Entrants Instead Of Defaults

**Files:**
- Modify: `hannah/tools/race_sim/tool.py`
- Modify: `hannah/simulation/sandbox.py`
- Test: `tests/tools/test_race_sim_tool.py`

- [ ] **Step 1: Write the failing test**

```python
@pytest.mark.asyncio
async def test_race_sim_prefers_resolved_roster_drivers() -> None:
    replay = {"resolved_roster": {"drivers": [{"code": "VER"}, {"code": "NOR"}]}}
    payload = await run(race="bahrain", replay=replay)
    assert set(payload["simulation"]["winner_probs"]) <= {"VER", "NOR"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/tools/test_race_sim_tool.py -v`
Expected: FAIL because the tool still uses hardcoded defaults.

- [ ] **Step 3: Write minimal implementation**

```python
selected_drivers = _drivers_from_replay_or_context(replay=replay, drivers=drivers)
race_state = RaceState.from_resolved_context(...)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/tools/test_race_sim_tool.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/tools/test_race_sim_tool.py hannah/tools/race_sim/tool.py hannah/simulation/sandbox.py
git commit -m "feat: seed race_sim from resolved entrants"
```

### Task 9: Rewire `pit_strategy` To Use Real Session State

**Files:**
- Modify: `hannah/tools/pit_strategy/tool.py`
- Modify: `hannah/simulation/strategy_engine.py`
- Test: `tests/tools/test_pit_strategy_tool.py`

- [ ] **Step 1: Write the failing test**

```python
@pytest.mark.asyncio
async def test_pit_strategy_uses_resolved_driver_context(monkeypatch) -> None:
    monkeypatch.setattr("hannah.tools.pit_strategy.tool.load_session_context", lambda **_: {"drivers": ["VER", "NOR", "LEC"], "current_lap": 18})
    payload = await run(race="bahrain", driver="VER", year=2025, lap=18)
    assert payload["recommended_pit_lap"] >= 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/tools/test_pit_strategy_tool.py -v`
Expected: FAIL because the tool still synthesizes a toy state inline.

- [ ] **Step 3: Write minimal implementation**

```python
session_context = load_session_context(race=race, year=year, driver=driver, lap=lap)
state = RaceState.from_race_data(session_context)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/tools/test_pit_strategy_tool.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/tools/test_pit_strategy_tool.py hannah/tools/pit_strategy/tool.py hannah/simulation/strategy_engine.py
git commit -m "feat: seed pit strategy from resolved session state"
```

### Task 10: Add Centralized Artifact Path Resolution

**Files:**
- Modify: `hannah/config/schema.py`
- Modify: `hannah/config/loader.py`
- Create: `hannah/models/artifact_paths.py`
- Test: `tests/config/test_model_artifact_selection.py`

- [ ] **Step 1: Write the failing test**

```python
def test_artifact_paths_keep_current_public_model_names() -> None:
    paths = ModelArtifactPaths()
    assert paths.resolve("winner_ensemble").endswith("winner_ensemble_v1.pkl")
    assert paths.resolve("pit_policy_q").endswith("pit_policy_q_v1.pkl")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/config/test_model_artifact_selection.py -v`
Expected: FAIL because artifact resolution is still hardcoded in model modules.

- [ ] **Step 3: Write minimal implementation**

```python
@dataclass(frozen=True)
class ModelArtifactPaths:
    ...
    def resolve(self, model_name: str) -> str:
        ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/config/test_model_artifact_selection.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/config/test_model_artifact_selection.py hannah/config/schema.py hannah/config/loader.py hannah/models/artifact_paths.py
git commit -m "feat: centralize model artifact path resolution"
```

### Task 11: Rework Winner Training Into A Real Historical Baseline

**Files:**
- Modify: `hannah/models/train_winner.py`
- Create: `hannah/models/datasets/results_baseline.py`
- Test: `tests/models/test_train_winner_baseline.py`

- [ ] **Step 1: Write the failing test**

```python
def test_train_winner_uses_results_baseline_dataset(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("hannah.models.datasets.results_baseline.build_results_dataset", lambda **_: [{"driver": "VER", "grid_position": 1, "won": 1}])
    saved = train(years=[2023, 2024], races=["bahrain"])
    assert saved.endswith("winner_ensemble_v1.pkl")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/models/test_train_winner_baseline.py -v`
Expected: FAIL because the dataset builder does not exist and training is still a fixed prior.

- [ ] **Step 3: Write minimal implementation**

```python
rows = build_results_dataset(years=years, races=races)
artifact = WinnerArtifact(version="v1", rows_seen=len(rows), feature_schema=(...))
return atomic_pickle_dump(ARTIFACT_PATH, artifact)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/models/test_train_winner_baseline.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/models/test_train_winner_baseline.py hannah/models/train_winner.py hannah/models/datasets/results_baseline.py
git commit -m "feat: train winner baseline from historical results dataset"
```

### Task 12: Add Telemetry-Era Feature Builder For Tyre And Lap-Time Baselines

**Files:**
- Create: `hannah/models/datasets/telemetry_baseline.py`
- Modify: `hannah/models/train_tyre_deg.py`
- Modify: `hannah/models/train_laptime.py`
- Test: `tests/models/test_telemetry_baseline_dataset.py`

- [ ] **Step 1: Write the failing test**

```python
def test_build_telemetry_baseline_dataset_returns_expected_columns() -> None:
    df = build_telemetry_baseline_dataset(years=[2024], races=["bahrain"])
    assert {"driver", "lap_number", "compound", "tyre_age", "lap_time_s"} <= set(df.columns)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/models/test_telemetry_baseline_dataset.py -v`
Expected: FAIL because the dataset builder does not exist.

- [ ] **Step 3: Write minimal implementation**

```python
def build_telemetry_baseline_dataset(years: list[int], races: list[str] | None = None) -> pd.DataFrame:
    rows = []
    for year in years:
        ...
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/models/test_telemetry_baseline_dataset.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/models/test_telemetry_baseline_dataset.py hannah/models/datasets/telemetry_baseline.py hannah/models/train_tyre_deg.py hannah/models/train_laptime.py
git commit -m "feat: add telemetry baseline dataset builders"
```

### Task 13: Strengthen Pit-Policy Baseline Training And Remove Silent Auto-Train

**Files:**
- Modify: `hannah/models/train_pit_q.py`
- Modify: `hannah/models/train_pit_rl.py`
- Modify: `hannah/simulation/environment.py`
- Test: `tests/models/test_train_pit_policy_baselines.py`

- [ ] **Step 1: Write the failing test**

```python
def test_train_pit_q_records_training_window_metadata() -> None:
    saved = train(years=[2021, 2022, 2023, 2024], races=["bahrain", "monaco"])
    artifact = load_artifact()
    assert artifact.years == (2021, 2022, 2023, 2024)
```

```python
def test_load_artifact_does_not_train_when_missing(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("hannah.models.train_pit_q.ARTIFACT_PATH", tmp_path / "missing.pkl")
    with pytest.raises(FileNotFoundError):
        load_artifact()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/models/test_train_pit_policy_baselines.py -v`
Expected: FAIL if artifact metadata and missing-artifact behavior are incomplete.

- [ ] **Step 3: Write minimal implementation**

```python
artifact = QPitPolicyArtifact(
    version="v1",
    buckets=STATE_BUCKETS,
    q_table=_train_q_table(...),
    years=tuple(years),
    races=tuple(races or []),
)
```

```python
if not ARTIFACT_PATH.exists():
    raise FileNotFoundError(...)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/models/test_train_pit_policy_baselines.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/models/test_train_pit_policy_baselines.py hannah/models/train_pit_q.py hannah/models/train_pit_rl.py hannah/simulation/environment.py
git commit -m "feat: strengthen shipped pit policy baselines"
```

### Task 14: Preserve `pit_rl` As A First-Class Baseline Surface

**Files:**
- Modify: `hannah/tools/train_model/tool.py`
- Modify: `hannah/models/evaluate.py`
- Modify: `config.yaml`
- Test: `tests/models/test_pit_rl_contract.py`

- [ ] **Step 1: Write the failing test**

```python
def test_pit_rl_remains_a_supported_public_model_name() -> None:
    assert "pit_rl" in SUPPORTED_MODEL_NAMES
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/models/test_pit_rl_contract.py -v`
Expected: FAIL if any baseline refactor drops `pit_rl` from the public surface.

- [ ] **Step 3: Write minimal implementation**

```python
# Keep pit_rl in the supported config, train_model, and evaluation surfaces.
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/models/test_pit_rl_contract.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/models/test_pit_rl_contract.py hannah/tools/train_model/tool.py hannah/models/evaluate.py config.yaml
git commit -m "test: preserve pit_rl as first-class public model surface"
```

### Task 15: Add Artifact Selection For Shipped Vs User Models

**Files:**
- Modify: `hannah/tools/train_model/tool.py`
- Modify: `hannah/models/evaluate.py`
- Modify: `hannah/models/train_winner.py`
- Modify: `hannah/models/train_pit_q.py`
- Test: `tests/config/test_model_artifact_selection_runtime.py`

- [ ] **Step 1: Write the failing test**

```python
def test_runtime_prefers_user_winner_artifact_when_configured(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HANNAH_PREFER_USER_ARTIFACTS", "1")
    monkeypatch.setenv("HANNAH_WINNER_USER_ARTIFACT", str(tmp_path / "winner_user.pkl"))
    assert resolve_model_artifact("winner_ensemble").endswith("winner_user.pkl")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/config/test_model_artifact_selection_runtime.py -v`
Expected: FAIL because runtime loaders do not use the centralized resolver yet.

- [ ] **Step 3: Write minimal implementation**

```python
artifact_path = resolve_model_artifact("winner_ensemble")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/config/test_model_artifact_selection_runtime.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/config/test_model_artifact_selection_runtime.py hannah/tools/train_model/tool.py hannah/models/evaluate.py hannah/models/train_winner.py hannah/models/train_pit_q.py
git commit -m "feat: wire runtime model loaders to central artifact resolver"
```

### Task 16: Gate OpenF1 Enrichment To Supported Historical Windows

**Files:**
- Modify: `hannah/_data_/season_roster_resolver.py`
- Modify: `hannah/_data_/openf1_client.py`
- Test: `tests/data/test_openf1_enrichment_gate.py`

- [ ] **Step 1: Write the failing test**

```python
def test_openf1_enrichment_is_skipped_for_legacy_years(monkeypatch) -> None:
    resolver = SeasonRosterResolver()
    called = {"openf1": False}

    def _fake_openf1(**kwargs):
        called["openf1"] = True
        return None

    monkeypatch.setattr(resolver, "_resolve_from_openf1", _fake_openf1)
    monkeypatch.setattr(resolver, "_resolve_from_fastf1", lambda **_: build_test_roster(year=2019))

    resolver.resolve(year=2019, race="bahrain", session="R")
    assert called["openf1"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/data/test_openf1_enrichment_gate.py -v`
Expected: FAIL because the OpenF1 gate does not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
def _should_attempt_openf1(self, year: int, session: str) -> bool:
    return year >= 2023 and session in {"R", "Q", "FP1", "FP2", "FP3", "S", "SQ"}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/data/test_openf1_enrichment_gate.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/data/test_openf1_enrichment_gate.py hannah/_data_/season_roster_resolver.py hannah/_data_/openf1_client.py
git commit -m "feat: gate openf1 enrichment to supported historical windows"
```

### Task 17: Revalidate End-To-End Tool Contracts And Docs

**Files:**
- Modify: `tests/scenarios/contracts.py`
- Modify: `tests/scenarios/test_strategy_scenarios.py`
- Modify: `tests/scenarios/test_prediction_scenarios.py`
- Modify: `hannah/docs/ARCHITECTURE.md`
- Modify: `hannah/docs/PRD.md`
- Modify: `AGENTS.md`

- [ ] **Step 1: Write the failing test**

```python
def test_prediction_scenarios_expect_dynamic_entry_lists() -> None:
    scenario = build_prediction_scenario(...)
    assert "resolved_roster" in scenario.expected_payload_keys
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/scenarios/test_strategy_scenarios.py tests/scenarios/test_prediction_scenarios.py -v`
Expected: FAIL because scenario contracts still assume static defaults.

- [ ] **Step 3: Write minimal implementation**

```python
# Update scenario harnesses and docs to reflect resolved roster inputs and baseline model artifacts.
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/scenarios/test_strategy_scenarios.py tests/scenarios/test_prediction_scenarios.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/scenarios/contracts.py tests/scenarios/test_strategy_scenarios.py tests/scenarios/test_prediction_scenarios.py hannah/docs/ARCHITECTURE.md hannah/docs/PRD.md AGENTS.md
git commit -m "docs: align scenarios and architecture with resolved roster baselines"
```

### Task 18: Full Validation Pass

**Files:**
- Modify: none unless a bug is found during validation
- Test: `tests/domain/ tests/data/ tests/tools/ tests/models/ tests/scenarios/ tests/agent/`

- [ ] **Step 1: Run focused suites**

Run: `pytest tests/domain tests/data tests/tools tests/models -v`
Expected: PASS

- [ ] **Step 2: Run scenario and agent suites**

Run: `pytest tests/scenarios tests/agent -v`
Expected: PASS

- [ ] **Step 3: Run the full recommended smoke lane**

Run: `/Users/deepedge/Desktop/projects/files/.venv/bin/python -m pytest -q tests/bus tests/agent tests/providers/test_litellm_provider_credentials.py tests/mcp tests/runtime tests/acceptance/test_hidden_acceptance_agent_loop.py tests/cli/test_agent_command.py tests/cli/test_chat_sessions.py tests/domain tests/data tests/tools tests/models tests/scenarios`
Expected: PASS with only known warning noise

- [ ] **Step 4: Commit any validation-driven fixes**

```bash
git add <fixed files>
git commit -m "test: close validation gaps for roster resolver and baselines"
```

- [ ] **Step 5: Prepare execution handoff**

```bash
git status
```

Expected: clean working tree on the feature branch.
