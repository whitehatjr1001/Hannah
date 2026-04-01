# Historical Roster And Baseline Models Design

## Goal

Replace Hannah's static year-agnostic roster assumptions with a season-aware resolver built on FastF1 and OpenF1, then use that resolved data to create stronger shipped baseline model artifacts.

## Problem

Hannah currently mixes three different concerns:

- current-season aliases and persona metadata in `hannah/domain/teams.py`
- live and historical session facts in the data layer
- baseline model artifacts in `models/saved/`

That works for a current-season toy setup, but it breaks down for multi-year training. Team names, driver pairings, and entry lists change by season. A static roster file cannot be the source of truth for historical training and still stay correct.

The correct split is:

- `teams.py` owns reusable aliases, prompt styles, and the 2026 fallback catalog
- FastF1 owns season/session roster truth
- OpenF1 enriches recent sessions with extra session detail
- runtime tools, simulation state, and subagent contexts consume merged resolved objects instead of static team constants

## Data Source Policy

### Canonical Resolution Order

For a requested `year + race + session`, Hannah resolves roster and session facts in this order:

1. FastF1
2. OpenF1
3. `teams.py` fallback, but only for the 2026 season

### Why FastF1 Is Primary

FastF1 covers the most important multi-year surfaces:

- event schedule
- session results
- lap timing
- tyre and pit timing
- track status
- telemetry and position data from the telemetry era onward

FastF1 is therefore the right backbone for both runtime and model training.

### Why OpenF1 Is Secondary

OpenF1 is valuable for recent-session enrichment:

- stints
- weather
- positions
- driver metadata
- pit and control-period context

But it is not broad enough to be the only historical backbone for a ten-year baseline stack, so it should enrich, not replace, FastF1.

Hard gate:

- only attempt OpenF1 enrichment for years and session types where it is expected to have useful data
- skip OpenF1 during broad historical dataset builds for older seasons instead of paying for noisy empty calls
- treat OpenF1 failures as non-fatal enrichment misses

### Why `teams.py` Still Exists

`hannah/domain/teams.py` still has two important jobs:

- stable driver and team aliases for user input and prompt construction
- prompt/persona heuristics such as strategist style, wet-weather tendencies, and team voice

It should stop owning historical pairings. It should become a reusable metadata catalog plus a 2026 fallback roster.

Compatibility rule:

- keep `get_driver_info()` and `get_driver_codes()` working for current runtime paths
- preserve the current 2026-grid tests while moving historical roster ownership out of the static catalog

## Proposed Runtime Architecture

### New Runtime Objects

Add three merged runtime objects:

- `ResolvedDriverProfile`
- `ResolvedTeamProfile`
- `ResolvedRoster`

These should include:

- driver code
- full name
- car number when known
- team name
- teammate
- season and event identifiers
- source provenance (`fastf1`, `openf1`, `teams_fallback`)
- prompt/style metadata copied from `teams.py`

### New Resolver Layer

Add a season-aware resolver module in the existing data layout, for example:

- `hannah/domain/resolved_roster.py`
- `hannah/_data_/season_roster_resolver.py`

Responsibilities:

- resolve the season roster from FastF1 session results or schedule/session data
- enrich recent-session records with OpenF1 driver/session metadata
- merge prompt metadata from `teams.py`
- expose a deterministic fallback for 2026 only

This layer is responsible for making historical rosters correct without forcing the rest of the codebase to understand API differences.

### Runtime Carrier Changes

The resolver is not sufficient by itself. Hannah needs an explicit runtime carrier for merged roster objects.

Add `resolved_roster` support to the runtime context and downstream consumers:

- `hannah/agent/context.py`
- `hannah/agent/prompts.py`
- `hannah/simulation/sandbox.py`
- `hannah/domain/race_state.py`
- `hannah/simulation/monte_carlo.py`
- `hannah/agent/subagents.py`
- `hannah/agent/worker_registry.py`

Without that propagation layer, historical entrants would still collapse back into current-grid assumptions inside simulation and worker paths.

### Tool Integration

The following tools must stop inventing driver lists or teams:

- `hannah/tools/race_data/tool.py`
- `hannah/tools/race_sim/tool.py`
- `hannah/tools/pit_strategy/tool.py`
- `hannah/tools/predict_winner/tool.py`

Each tool should work off a resolved roster/session payload.

Tool contract compatibility rule:

- keep the existing top-level response keys for `race_data`, `race_sim`, and `pit_strategy`
- add `resolved_roster` as an additive field instead of replacing current acceptance-facing payload keys
- swap internal defaults for resolved entrants without breaking current external contracts

The most important changes are:

- `race_data` returns real season-aware roster and session metadata
- `race_sim` seeds state from resolved entrants, positions, compounds, and weather
- `pit_strategy` stops using a synthetic three-car state
- `predict_winner` uses the real entry list for the requested event

## Prompt And Persona Architecture

Prompt generation should use merged roster objects instead of hardcoded yearly assumptions.

`teams.py` should keep:

- aliases
- fallback display names
- strategist style metadata
- prompt heuristics

Prompt builders and worker/subagent builders in:

- `hannah/domain/prompts.py`
- `hannah/agent/subagents.py`
- `hannah/agent/worker_registry.py`

should accept resolved profiles and enrich them with `teams.py` metadata instead of assuming that static roster data is always correct.

## Dataset Building Strategy

The training stack should be split into two model families with different historical windows.

### Results-Era Baselines

Use approximately 2016-2025 season data for models that only need high-level event and result metadata.

Candidate features:

- grid position
- finish position
- qualifying rank or delta
- team strength proxy
- track type
- reliability prior
- recent form

Primary use:

- winner-probability baseline

### Telemetry-Era Baselines

Use 2018-2025 session data for models that require rich lap, stint, timing, or event-window features.

Candidate features:

- lap time
- tyre compound
- tyre age
- stint number
- pit in/out markers
- weather
- track status
- safety car or VSC windows
- traffic and gap proxies

Primary use:

- tyre degradation baseline
- lap-time baseline
- pit-policy baseline

### OpenPitWall Role

The OpenPitWall donor seam should be treated as optional supplemental training corpus material, not as the canonical source of truth.

Use it to:

- widen strategy examples
- enrich smoke-training corpora
- compare engineered feature shapes

Do not let it define roster truth or season metadata.

## Shipped Baseline Artifacts

Hannah should ship frozen baseline artifacts in `models/saved/`.

Baseline families:

- `winner_ensemble`
- `tyre_model`
- `laptime_model`
- `pit_policy_q`
- `pit_rl`

These shipped artifacts should be:

- created offline
- versioned
- deterministic to load
- replaceable by user-trained artifacts

The runtime must never silently retrain during a normal analysis turn.

Backwards compatibility requirement:

- keep the existing public tool and CLI model names
- keep the current artifact filenames unless a compatibility alias is added
- do not introduce new public names such as `winner_ensemble_baseline` without aliasing them to the current contract surface

## User Model Workflow

Users should have two supported modes:

1. Use shipped baselines
2. Train and select custom artifacts

The `train_model` tool remains explicit and offline. It should build user artifacts from the same dataset builder and feature pipeline used for shipped baselines.

Artifact selection should be config-driven, not prompt-driven.

This requires a centralized artifact-path resolver. Direct hardcoded artifact paths inside model modules must be replaced by shared config-backed resolution.

Config compatibility rule:

- preserve the current public model names: `tyre_model`, `laptime_model`, `pit_rl`, `pit_policy_q`, and `winner_ensemble`
- add explicit config-backed path resolution for `pit_policy_q`, which is public today but not yet represented centrally

## Testing Strategy

### Resolver Tests

Add tests that verify:

- FastF1 season roster resolution
- OpenF1 enrichment merge behavior
- 2026-only fallback to `teams.py`
- no historical fallback to static 2026 data
- `resolved_roster` propagates through agent, simulation, and subagent contexts

### Tool Contract Tests

Add tests that verify:

- `race_data` returns season-correct rosters
- `race_sim` consumes resolved entrants instead of defaults
- `pit_strategy` uses resolved current-race context
- `predict_winner` accepts dynamic entry lists

### Dataset Builder Tests

Add tests that verify:

- results-era feature tables
- telemetry-era feature tables
- stable schemas across missing optional inputs

### Baseline Training Tests

Add tests that verify:

- baseline trainers produce real artifacts
- loaders can read both shipped and user artifacts
- evaluation commands stay green

## Rollout Order

1. Season-aware roster resolver
2. Tool rewiring to resolved roster/session objects
3. Historical dataset builder
4. Real winner baseline
5. Real tyre and lap-time baselines
6. Real pit-policy baseline
7. Configurable shipped-vs-user artifact selection

There is one explicit migration item inside step 6:

- remove or replace the current `train_pit_q.load_artifact()` auto-train behavior that can trigger from rival-analysis turns

## Non-Goals

This slice does not include:

- silent online learning during chat
- replacing the main agent loop again
- full RL-heavy production training infrastructure in the first pass
- broad refactors unrelated to roster or model correctness

## Expected Outcome

After this work:

- Hannah will stop assuming a single static grid across seasons
- the tools will operate on correct historical entrants
- the shipped models will be materially stronger than deterministic smoke priors
- users will be able to keep the shipped defaults or train their own artifacts without changing the runtime contract
