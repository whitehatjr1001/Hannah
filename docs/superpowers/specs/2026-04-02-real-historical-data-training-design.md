# Real Historical Data Training Pipeline Design

## Goal

Replace Hannah's synthetic smoke-training path with a real offline historical-data pipeline that ingests FastF1 and OpenF1 session data, caches normalized parquet datasets, and trains shipped baseline artifacts from those datasets.

## Problem

The current model trainers are fast because they do not use real multi-year Formula 1 data. They build deterministic synthetic tables in memory and fit lightweight artifacts on top of those tables. That is useful for smoke coverage, but it is not a real baseline stack.

The next version needs to train from actual historical sessions across:

- practice sessions (`FP1`, `FP2`, `FP3`)
- qualifying (`Q`)
- race (`R`)

It also needs to stay operationally sane:

- data fetches must be cached
- historical builds must be restartable
- runtime inference must remain artifact-only
- tools must never retrain during a normal analysis turn

## Recommended Approach

Use one unified offline dataset cache that every trainer reads from.

Why this is the right choice:

- one ingestion path is easier to validate than one per model
- one feature-normalization layer prevents feature drift across trainers
- parquet caches make retraining fast after the initial build
- the runtime stays simple because inference still just loads artifacts

## Source Policy

### Canonical Historical Backbone

FastF1 is the primary historical source for:

- schedules
- session results
- laps
- compounds and stint-related lap context
- track status
- weather
- telemetry-era data where available

Use it as the canonical source of season/session identity and core timing facts.

### Secondary Enrichment

OpenF1 is used only as a recent-era enrichment layer for:

- driver metadata
- weather snapshots
- positions
- pit events
- stint-like session context

OpenF1 must never define the canonical roster or session identity. If OpenF1 is missing or unsupported for a historical window, the dataset build continues without it.

### Historical Windows

Use different windows for different model families:

- results-era models: approximately `2016+`
- telemetry and pit-policy models: approximately `2018+`
- OpenF1 enrichment: only on supported recent seasons, gated by year and session support

## Pipeline Architecture

### Stage 1: Raw Session Cache

For each requested `year + event + session_type`, fetch and cache raw session payloads under a deterministic cache path.

Raw cache should include:

- FastF1 session metadata
- FastF1 lap/result/weather/status extracts
- OpenF1 enrichment payloads when allowed
- dataset build manifest entries for provenance and retries

This stage should be idempotent. If the cache exists and the manifest says it is complete, the builder should reuse it.

### Stage 2: Normalized Session Tables

Transform raw session caches into normalized tabular outputs with stable schemas.

Core outputs:

- `results_baseline.parquet`
- `telemetry_baseline.parquet`
- `pit_events_baseline.parquet`
- optional per-session normalized parquet shards for incremental rebuilds

The normalized schema should preserve:

- season, event, session type
- driver code
- team
- lap and stint context
- track/weather context
- pit and control-period context
- provenance fields that record whether a feature came from FastF1, OpenF1, or derived logic

### Stage 3: Trainer Inputs

All training scripts should read normalized parquet datasets instead of fetching APIs directly.

That gives:

- reproducible builds
- restartability
- easier debugging
- easier train/eval separation

## Dataset Families

### Results Dataset

Use for winner and form priors.

Candidate columns:

- `year`
- `event`
- `session_type`
- `driver`
- `team`
- `grid_position`
- `finish_position`
- `qualifying_rank`
- `qualifying_delta_s`
- `track_type`
- `safety_car_prob_proxy`
- `dnf`
- `points`
- `recent_form_score`

### Telemetry Dataset

Use for tyre and lap-time models.

Candidate columns:

- `year`
- `event`
- `session_type`
- `driver`
- `lap_number`
- `lap_time_s`
- `sector1_s`
- `sector2_s`
- `sector3_s`
- `compound`
- `tyre_age`
- `stint_number`
- `track_temp`
- `air_temp`
- `rainfall`
- `track_status`
- `position`
- `gap_to_leader_s`
- `traffic_proxy`

### Pit Events Dataset

Use for pit-policy learning and evaluation.

Candidate columns:

- `year`
- `event`
- `session_type`
- `driver`
- `lap_number`
- `pit_in`
- `pit_out`
- `compound_in`
- `compound_out`
- `tyre_age_before_stop`
- `gap_ahead_s`
- `gap_behind_s`
- `safety_car_active`
- `vsc_active`
- `weather_state`
- `stint_number`

## Trainer Design

### Winner Model

Replace the current synthetic winner-baseline builder with a real historical trainer that consumes the normalized results dataset.

Requirements:

- keep the public artifact name `winner_ensemble`
- preserve deterministic loading at runtime
- allow a small, stable fallback when no artifact exists

### Tyre And Lap-Time Models

Replace synthetic dataset generation with normalized telemetry data.

Requirements:

- keep current artifact names
- support offline retraining from cached parquet
- preserve the runtime fallback path if the artifact is missing

### Pit Policy Models

Split the two pit-policy paths clearly:

- `pit_policy_q`: train from normalized historical features plus the strategy environment
- `pit_rl`: remain optional and offline-only, but train against real feature distributions instead of purely synthetic thresholds

The first production-grade slice should prioritize `pit_policy_q` over a full PPO rewrite.

## CLI And Tooling

Add explicit offline commands for data preparation and training.

Needed surfaces:

- dataset build command for year ranges and session types
- dataset inspect/summary command
- train command that consumes cached datasets
- evaluation command that reports artifact metrics plus dataset coverage

The normal runtime tools:

- `race_data`
- `race_sim`
- `pit_strategy`
- `predict_winner`

must continue to load artifacts only. They must not trigger dataset builds or training.

## Caching And Provenance

Every build should produce metadata describing:

- requested year range
- requested events and session types
- fetched source windows
- enrichment coverage
- skipped sessions
- failed sessions
- artifact versions trained from the build

Suggested outputs:

- `data/historical_cache/manifest.json`
- `data/historical_cache/raw/...`
- `data/historical_cache/normalized/...`

## Failure Handling

The pipeline must degrade gracefully:

- if one session fails, record it and continue
- if OpenF1 enrichment fails, continue with FastF1-only data
- if telemetry is unavailable for a year/session, exclude that session from telemetry-era training but keep results-era training alive
- if dataset coverage falls below a configured threshold, fail training with a clear reason instead of producing a misleading artifact

## Testing Strategy

Use layered tests:

1. dataset builder unit tests with fake FastF1/OpenF1 payloads
2. normalized schema contract tests
3. historical-window gating tests
4. trainer tests using fixture parquet files
5. evaluation tests proving artifacts read real cached datasets instead of synthetic builders

The acceptance target is not "large model accuracy" yet. The acceptance target is:

- the pipeline fetches and caches the right sessions
- normalized schemas are stable
- trainers consume cached real-data tables
- artifacts are produced deterministically enough for CI smoke evaluation

## File Map

Create:

- `hannah/data/historical_dataset.py`
- `hannah/data/historical_cache.py`
- `hannah/data/feature_builders/results_features.py`
- `hannah/data/feature_builders/telemetry_features.py`
- `hannah/data/feature_builders/pit_features.py`
- `tests/data/test_historical_dataset_builder.py`
- `tests/data/test_historical_schema_contracts.py`
- `tests/models/test_real_data_trainers.py`

Modify:

- `hannah/_data_/fastf1_loader.py`
- `hannah/_data_/openf1_client.py`
- `hannah/models/train_tyre_deg.py`
- `hannah/models/train_laptime.py`
- `hannah/models/train_pit_q.py`
- `hannah/models/train_pit_rl.py`
- `hannah/models/train_winner.py`
- `hannah/models/evaluate.py`
- `hannah/tools/train_model/tool.py`
- `hannah/cli/app.py`
- `config.yaml`

## Non-Goals

This slice does not aim to:

- make live runtime inference depend on API fetches
- introduce online learning during chat
- solve frontier model accuracy
- replace every fallback heuristic at once

## Success Criteria

This design is successful when:

- Hannah can build cached historical datasets for user-selected year ranges and session types
- the trainers consume those cached datasets instead of synthetic generators
- the shipped artifact interfaces remain unchanged
- retraining is explicit, offline, and reproducible
- evaluation reports dataset coverage and model metrics from real cached data
