# Hannah Smith V1 Release Notes

Updated: 2026-03-18

## Release Status

**Hannah Smith v1 is GREEN.**

Release validation at this point:
- Full test suite: `81 passed, 1 warning`
- Representative v1 CLI smokes (including direct trace command): `6/6 passed`
- V1.5 S1+S2 coverage is green across agent seams, scenario contracts, simulation trace contracts, and hidden acceptance
- Provider seam preserved
- Deterministic simulator boundary preserved
- Runtime recursion still excluded from the v1 main loop

## What V1 Ships

Hannah v1 now ships a working CLI-first F1 strategy agent with:
- a deterministic strategy and simulation core
- public scenario contracts plus hidden acceptance coverage
- smoke-real model training artifacts
- a local fallback provider path so the CLI still works when hosted LLM access is missing
- honest tool boundaries for race data, simulation, strategy, prediction, and training

This is not the final product vision from `GOAL.md`. It is the stable v1 base that preserves the architecture and makes v1.5 work safe.

## Validation Snapshot

### Automated

- `python3 -m pytest -q /Users/deepedge/Desktop/projects/files/tests`
- Result: `81 passed, 1 warning`

### V1.5 Coverage (S1 + S2)

- `python3 -m pytest -q /Users/deepedge/Desktop/projects/files/tests/agent`
- Result: `9 passed`
- Added:
  - [tests/agent/test_agent_loop_seams.py](/Users/deepedge/Desktop/projects/files/tests/agent/test_agent_loop_seams.py)
  - [tests/agent/test_subagents_provider_seams.py](/Users/deepedge/Desktop/projects/files/tests/agent/test_subagents_provider_seams.py)
- `python3 -m pytest -q /Users/deepedge/Desktop/projects/files/tests/acceptance`
- Result: `38 passed`
- Added:
  - [tests/acceptance/test_hidden_acceptance_agent_loop.py](/Users/deepedge/Desktop/projects/files/tests/acceptance/test_hidden_acceptance_agent_loop.py)
  - [tests/acceptance/test_hidden_acceptance_trace_replay.py](/Users/deepedge/Desktop/projects/files/tests/acceptance/test_hidden_acceptance_trace_replay.py)
- `python3 -m pytest -q /Users/deepedge/Desktop/projects/files/tests/simulation/test_debug_trace.py /Users/deepedge/Desktop/projects/files/tests/scenarios/test_strategy_scenarios.py /Users/deepedge/Desktop/projects/files/tests/test_imports.py`
- Result: `12 passed, 1 warning`
- Added:
  - [tests/simulation/test_debug_trace.py](/Users/deepedge/Desktop/projects/files/tests/simulation/test_debug_trace.py)
- Updated:
  - [tests/scenarios/contracts.py](/Users/deepedge/Desktop/projects/files/tests/scenarios/contracts.py)
  - [tests/scenarios/test_strategy_scenarios.py](/Users/deepedge/Desktop/projects/files/tests/scenarios/test_strategy_scenarios.py)
  - [tests/test_imports.py](/Users/deepedge/Desktop/projects/files/tests/test_imports.py)

### Sequential V1 CLI Smokes

All run with `HANNAH_FORCE_LOCAL_PROVIDER=1`.

1. `python3 /Users/deepedge/Desktop/projects/files/hannah.py train all --years 2022,2023,2024 --races bahrain,monaco,singapore`
2. `python3 /Users/deepedge/Desktop/projects/files/hannah.py strategy --race bahrain --lap 18 --driver VER --type optimal`
3. `python3 /Users/deepedge/Desktop/projects/files/hannah.py strategy --race monaco --lap 30 --driver LEC --type overcut`
4. `python3 /Users/deepedge/Desktop/projects/files/hannah.py simulate --race silverstone --driver VER --laps 52 --weather mixed`
5. `python3 /Users/deepedge/Desktop/projects/files/hannah.py predict --race singapore --year 2025`
6. `python3 /Users/deepedge/Desktop/projects/files/hannah.py trace --race silverstone --year 2025 --drivers VER,NOR,LEC --laps 52 --weather mixed --checkpoints 12,26,52`

Result: **6/6 passed**

Known runtime caveats:
- OpenF1 can still return `404` or `429`.
- The local Python shell is still 3.9 even though `pyproject.toml` targets 3.11.
- Parallel training plus inference can race on artifact reads; sequential validation is green.

## V1.5 Slice Status

Slices completed:
- **S1 — Agent Loop + Provider Seam Hardening (tests-first)**
- **S2 — Deterministic Trace/Replay Slice (tests-first)**

**V1.5 is complete.**

Checklist:
- [x] Add focused `tests/agent` coverage for loop roundtrip, argument coercion, error serialization, sub-agent failure containment, and provider selection metadata.
- [x] Add isolated hidden acceptance seam coverage for loop toolflow and sub-agent failure containment.
- [x] Add deterministic trace/replay contracts and keep baseline `race_sim` contract stable unless trace mode is explicitly requested.
- [x] Add a thin direct CLI trace command that uses simulator-owned outputs instead of LLM synthesis.
- [x] Run validation ladder: targeted tests, scenario tests, hidden acceptance tests, full suite, sequential CLI smokes.
- [x] Keep architecture boundaries intact (provider seam, deterministic simulator boundary, no runtime recursion in main loop).
- [x] Close v1.5 with release docs and next-session handoff prompt.

## Why V1 Was Written This Way

These choices were deliberate:

- **Deterministic-first**  
  The simulator and tools own race-state transitions and outputs. The LLM layer orchestrates.

- **Scenario-first**  
  Public contract tests and hidden acceptance scenarios were written before deeper implementation so behavior got locked before the donor ports drifted.

- **Provider seam preserved**  
  Hosted LLM access and local fallback share the same loop entry so v2 can swap model routing without rewriting tools or orchestration.

- **Smoke-real training, not fake placeholders**  
  The model layer writes real artifacts and runs real lightweight deterministic training paths, but avoids pretending v1 has a fully trained production ML stack.

- **Sub-agent-assisted implementation**  
  Donor repo analysis, scenario generation, simulation work, model work, and provider work were split across isolated workers to keep ownership clear and context stable.

## Files Touched For V1 And Why

This section lists the implementation files touched during the v1 build-out and why each exists in its current form.

### Public Scenario Contracts

- [tests/scenarios/README.md](/Users/deepedge/Desktop/projects/files/tests/scenarios/README.md)  
  Documents the purpose of the public scenario layer and why it stays broad rather than fully prescriptive.

- [tests/scenarios/__init__.py](/Users/deepedge/Desktop/projects/files/tests/scenarios/__init__.py)  
  Makes the scenario package importable for harness-based testing.

- [tests/scenarios/contracts.py](/Users/deepedge/Desktop/projects/files/tests/scenarios/contracts.py)  
  Defines the 20 public scenario contracts that lock tool paths, payload shapes, and pass/fail criteria.

- [tests/scenarios/harness.py](/Users/deepedge/Desktop/projects/files/tests/scenarios/harness.py)  
  Provides deterministic patching of external data and simulation dependencies so scenario tests stay stable.

- [tests/scenarios/test_scenario_matrix.py](/Users/deepedge/Desktop/projects/files/tests/scenarios/test_scenario_matrix.py)  
  Verifies matrix size, uniqueness, category coverage, and tool declarations.

- [tests/scenarios/test_strategy_scenarios.py](/Users/deepedge/Desktop/projects/files/tests/scenarios/test_strategy_scenarios.py)  
  Locks the public strategy toolflow and output contract.

- [tests/scenarios/test_prediction_scenarios.py](/Users/deepedge/Desktop/projects/files/tests/scenarios/test_prediction_scenarios.py)  
  Locks the public prediction toolflow and normalized probability output shape.

- [tests/scenarios/test_training_smoke.py](/Users/deepedge/Desktop/projects/files/tests/scenarios/test_training_smoke.py)  
  Locks the training tool output contract without pretending to validate model quality.

### Hidden Acceptance Coverage

- [tests/acceptance/test_hidden_acceptance_toolflows.py](/Users/deepedge/Desktop/projects/files/tests/acceptance/test_hidden_acceptance_toolflows.py)  
  Masks stricter end-to-end toolflow assertions from the main implementation loop.

- [tests/acceptance/test_hidden_acceptance_prediction_domain.py](/Users/deepedge/Desktop/projects/files/tests/acceptance/test_hidden_acceptance_prediction_domain.py)  
  Catches command-parser and prompt-surface regressions without exposing all acceptance expectations.

- [tests/acceptance/test_hidden_acceptance_training_seams.py](/Users/deepedge/Desktop/projects/files/tests/acceptance/test_hidden_acceptance_training_seams.py)  
  Guards training dispatch and provider/config seam behavior.

### Domain Layer

- [hannah/domain/commands.py](/Users/deepedge/Desktop/projects/files/hannah/domain/commands.py)  
  Expanded into a more FormulaGPT-like command parser with explicit pit, pace, cancel, and stay-out handling while staying acceptance-compatible.

- [hannah/domain/teams.py](/Users/deepedge/Desktop/projects/files/hannah/domain/teams.py)  
  Centralizes driver/team metadata, aliases, rivals, and strategy-style hints so strategy and simulation code do not duplicate them.

- [hannah/domain/race_state.py](/Users/deepedge/Desktop/projects/files/hannah/domain/race_state.py)  
  Provides typed snapshots, event windows, and pit-rejoin helpers so prompts and simulation have a common state vocabulary.

- [hannah/domain/prompts.py](/Users/deepedge/Desktop/projects/files/hannah/domain/prompts.py)  
  Keeps strategist-facing prompt fragments centralized and aligned to the deterministic domain model.

### Simulation Core

- [hannah/simulation/sandbox.py](/Users/deepedge/Desktop/projects/files/hannah/simulation/sandbox.py)  
  Builds a richer deterministic `RaceState` from context or race-data payloads, with stable seeds and event windows.

- [hannah/simulation/monte_carlo.py](/Users/deepedge/Desktop/projects/files/hannah/simulation/monte_carlo.py)  
  Replaced toy simulation with a deterministic Monte Carlo path that respects pit windows, compounds, events, and monkeypatched RNG compatibility.

- [hannah/simulation/tyre_model.py](/Users/deepedge/Desktop/projects/files/hannah/simulation/tyre_model.py)  
  Added compound profiles and deterministic degradation logic with trained-artifact fallback.

- [hannah/simulation/gap_engine.py](/Users/deepedge/Desktop/projects/files/hannah/simulation/gap_engine.py)  
  Keeps undercut/overcut feasibility logic separate from the strategy synthesizer.

- [hannah/simulation/strategy_engine.py](/Users/deepedge/Desktop/projects/files/hannah/simulation/strategy_engine.py)  
  Converts simulation outputs into one decisive pit-wall call while preserving the stable tool-facing shape.

- [hannah/simulation/environment.py](/Users/deepedge/Desktop/projects/files/hannah/simulation/environment.py)  
  Upgraded from placeholder state to a deterministic RL-smoke environment rather than a fake shell object.

- [hannah/simulation/competitor_agents.py](/Users/deepedge/Desktop/projects/files/hannah/simulation/competitor_agents.py)  
  Produces structured deterministic rival opinions instead of free-form placeholder noise.

### Data And Training Inputs

- [hannah/data/openpitwall_loader.py](/Users/deepedge/Desktop/projects/files/hannah/data/openpitwall_loader.py)  
  Normalizes lightweight corpus inputs from `json`, `jsonl`, `csv`, and text-like files for smoke training paths.

- [hannah/data/preprocess.py](/Users/deepedge/Desktop/projects/files/hannah/data/preprocess.py)  
  Builds feature tables and normalization paths that still work in lightweight environments without mandatory pandas availability.

### Model And Evaluation Layer

- [hannah/models/train_tyre_deg.py](/Users/deepedge/Desktop/projects/files/hannah/models/train_tyre_deg.py)  
  Writes a real deterministic tyre degradation artifact instead of returning a fake path.

- [hannah/models/train_laptime.py](/Users/deepedge/Desktop/projects/files/hannah/models/train_laptime.py)  
  Produces a lightweight deterministic lap-time artifact using a stable regression fit.

- [hannah/models/train_pit_rl.py](/Users/deepedge/Desktop/projects/files/hannah/models/train_pit_rl.py)  
  Produces a deterministic policy artifact inside the expected `.zip` path so the training seam is real in v1.

- [hannah/models/train_winner.py](/Users/deepedge/Desktop/projects/files/hannah/models/train_winner.py)  
  Persists a deterministic winner prior while preserving the expected normalized prediction behavior.

- [hannah/models/evaluate.py](/Users/deepedge/Desktop/projects/files/hannah/models/evaluate.py)  
  Replaced a placeholder score with deterministic artifact-aware evaluation output.

### Provider And Orchestration Layer

- [hannah/providers/local_fallback.py](/Users/deepedge/Desktop/projects/files/hannah/providers/local_fallback.py)  
  Adds the deterministic local planner/synthesizer so the CLI stays usable when hosted model access is unavailable.

- [hannah/providers/litellm_provider.py](/Users/deepedge/Desktop/projects/files/hannah/providers/litellm_provider.py)  
  Preserves the hosted provider path but falls back cleanly on import or completion failure.

- [hannah/providers/registry.py](/Users/deepedge/Desktop/projects/files/hannah/providers/registry.py)  
  Introduces a clearer provider contract and selection metadata without breaking the loop interface.

- [hannah/providers/__init__.py](/Users/deepedge/Desktop/projects/files/hannah/providers/__init__.py)  
  Makes the provider seam explicit at the package boundary.

- [hannah/agent/tool_registry.py](/Users/deepedge/Desktop/projects/files/hannah/agent/tool_registry.py)  
  Hardened the call path for sync or async tool functions and added return-type checks.

- [hannah/agent/loop.py](/Users/deepedge/Desktop/projects/files/hannah/agent/loop.py)  
  Keeps the main control loop small, readable, and robust to tool/provider response shape differences.

- [hannah/agent/subagents.py](/Users/deepedge/Desktop/projects/files/hannah/agent/subagents.py)  
  Keeps rival/sim/strategy/predict assistants behind a narrow async boundary and compatible with the local fallback path.

### Simulation Unit Tests

- [tests/simulation/test_tyre_model.py](/Users/deepedge/Desktop/projects/files/tests/simulation/test_tyre_model.py)  
  Verifies degradation behavior and keeps the tyre model from regressing back into arbitrary outputs.

- [tests/simulation/test_pit_windows.py](/Users/deepedge/Desktop/projects/files/tests/simulation/test_pit_windows.py)  
  Verifies pit-window behavior and RNG compatibility expectations.

- [tests/simulation/test_environment.py](/Users/deepedge/Desktop/projects/files/tests/simulation/test_environment.py)  
  Verifies deterministic environment stepping for the RL-smoke layer.

### Model And Data Tests

- [tests/models/test_training_artifacts.py](/Users/deepedge/Desktop/projects/files/tests/models/test_training_artifacts.py)  
  Verifies the smoke trainers actually write the expected artifacts.

- [tests/models/test_winner_predictor.py](/Users/deepedge/Desktop/projects/files/tests/models/test_winner_predictor.py)  
  Verifies deterministic normalized winner probability output.

- [tests/models/test_evaluate_models.py](/Users/deepedge/Desktop/projects/files/tests/models/test_evaluate_models.py)  
  Verifies the evaluation payload is sensible and artifact-aware.

- [tests/models/test_data_loader_and_preprocess.py](/Users/deepedge/Desktop/projects/files/tests/models/test_data_loader_and_preprocess.py)  
  Verifies corpus loading and preprocessing behavior in lightweight environments.

### Provider Regression Test

- [tests/agent/test_local_fallback.py](/Users/deepedge/Desktop/projects/files/tests/agent/test_local_fallback.py)  
  Guards the specific bug where English prompt words like `RUN` leaked into driver extraction.

### V1.5 Agent/Provider Seam Tests

- [tests/agent/test_agent_loop_seams.py](/Users/deepedge/Desktop/projects/files/tests/agent/test_agent_loop_seams.py)  
  Verifies provider tool-call -> tool execution -> second-pass final response flow, plus tool-argument coercion and tool-error serialization.

- [tests/agent/test_subagents_provider_seams.py](/Users/deepedge/Desktop/projects/files/tests/agent/test_subagents_provider_seams.py)  
  Verifies `spawn_all` failure containment and provider registry metadata/selection seams.

- [tests/acceptance/test_hidden_acceptance_agent_loop.py](/Users/deepedge/Desktop/projects/files/tests/acceptance/test_hidden_acceptance_agent_loop.py)  
  Adds masked end-to-end seam assertions for loop toolflow and rival failure isolation.

### V1.5 Deterministic Trace/Replay Slice

- [hannah/simulation/replay_trace.py](/Users/deepedge/Desktop/projects/files/hannah/simulation/replay_trace.py)  
  Adds deterministic replay/debug trace builders inspired by FormulaGPT scoreboard/pit-projection shape, but owned by Hannah simulation semantics.

- [hannah/simulation/monte_carlo.py](/Users/deepedge/Desktop/projects/files/hannah/simulation/monte_carlo.py)  
  Exposes `build_replay_trace(...)` at the simulation boundary so trace generation remains deterministic and tool-call friendly.

- [hannah/tools/race_sim/tool.py](/Users/deepedge/Desktop/projects/files/hannah/tools/race_sim/tool.py)  
  Adds explicit trace mode (`trace`, `trace_checkpoints`, `replay`) and deterministic trace IDs while preserving baseline output when trace mode is not requested.

- [hannah/cli/app.py](/Users/deepedge/Desktop/projects/files/hannah/cli/app.py)  
  Adds a thin direct `trace` command so operators can inspect deterministic replay outputs without routing through the LLM loop.

- [hannah/cli/format.py](/Users/deepedge/Desktop/projects/files/hannah/cli/format.py)  
  Adds compact deterministic trace summary rendering for CLI output.

- [tests/simulation/test_debug_trace.py](/Users/deepedge/Desktop/projects/files/tests/simulation/test_debug_trace.py)  
  Locks deterministic trace structure and repeatability.

- [tests/acceptance/test_hidden_acceptance_trace_replay.py](/Users/deepedge/Desktop/projects/files/tests/acceptance/test_hidden_acceptance_trace_replay.py)  
  Locks hidden acceptance contracts for trace mode and replay stability.

## What Was Intentionally Left Out Of V1

Still intentionally not part of v1:
- local RLM runtime
- recursive Python execution in the main loop
- richer self-play or strategy learning
- replay/debug viewer layer
- web UI or TUI product layer

The legacy root duplicates have been removed. Keep primary logic in the packaged modules instead:
- [loop.py](/Users/deepedge/Desktop/projects/files/hannah/agent/loop.py)
- [persona.py](/Users/deepedge/Desktop/projects/files/hannah/agent/persona.py)
- [monte_carlo.py](/Users/deepedge/Desktop/projects/files/hannah/simulation/monte_carlo.py)
- [server.py](/Users/deepedge/Desktop/projects/files/hannah/rlm/server.py)
- [subagents.py](/Users/deepedge/Desktop/projects/files/hannah/agent/subagents.py)

## How To Proceed After V1.5

V1.5 is complete. The next phase should be a bounded post-v1.5 step (v2 prep or one targeted reliability upgrade), not a rewrite.

Read these next:
- [V1_5_RELEASE.md](/Users/deepedge/Desktop/projects/files/hannah/docs/V1_5_RELEASE.md)
- [V2_FINAL_CALL.md](/Users/deepedge/Desktop/projects/files/hannah/docs/V2_FINAL_CALL.md)

Primary next targets:
- v2 provider seam readiness (hosted to local RLM swap remains config-only)
- safer artifact writes if parallel training plus inference becomes supported
- deeper rival behavior only if deterministic simulator ownership remains untouched
- evaluation depth and scenario breadth expansion without loosening contracts

Do not break:
- provider boundary
- deterministic simulator boundary
- tool registry boundary
- public scenario matrix
- hidden acceptance separation

## Post-V1.5 Coding Loop

Use this loop every time:

1. Read the source-of-truth docs first.
   Read `GOAL.md`, `PRD.md`, `ARCHITECTURE.md`, `roadmap.md`, `V1_RELEASE.md`, `V1_5_RELEASE.md`, `V2_FINAL_CALL.md`, and `AGENTS.md`.

2. Pick one bounded next-phase slice.
   Good slices:
   - one sub-agent behavior upgrade
   - one scenario family expansion
   - one evaluation improvement
   - one data/preprocess enhancement

3. Lock tests before writing code.
   - Add or extend public scenario contracts if the slice changes visible behavior.
   - Add or extend focused unit tests in the module’s own test area.
   - Keep hidden acceptance authored separately from the main implementation loop.

4. Use donor repos through sub-agents, not by copying whole repos.
   - `nanobot` sub-agent for loop/provider/memory/session patterns
   - `FormulaGPT` sub-agent for race-state semantics, prompts, command grammar
   - `pit-stop-simulator` sub-agent for env/training/feature ideas
   - `rlm-cli` only when explicitly doing later v2 seam work

5. Use worker sub-agents as a team with disjoint ownership.
   Recommended split:
   - worker A owns `hannah/domain/` or `hannah/agent/`
   - worker B owns `hannah/simulation/`
   - worker C owns `hannah/models/` and related data prep
   - worker D owns docs/tests if needed

6. Preserve context in the main thread.
   The main agent should stay the integrator:
   - track the architecture
   - keep the shared branch green
   - resolve conflicts
   - decide when a slice is actually done

7. Keep hidden scenario creation isolated.
   A dedicated sub-agent should curate acceptance scenarios independently from the main implementation context. The main implementation loop should know the public contract, not every hidden answer.

8. Run validation in this order.
   - targeted unit tests
   - public scenario tests
   - hidden acceptance tests
   - full suite
   - sequential CLI smokes for the affected commands

9. Append memory before ending.
   Add one short AGENTS entry with:
   - what worked
   - what failed or stayed risky
   - touched paths
   - next follow-ups

## New Session Prompt

Use this to start the next session:

```text
Read these files first and treat them as source of truth:

- /Users/deepedge/Desktop/projects/files/hannah/docs/GOAL.md
- /Users/deepedge/Desktop/projects/files/hannah/docs/PRD.md
- /Users/deepedge/Desktop/projects/files/hannah/docs/ARCHITECTURE.md
- /Users/deepedge/Desktop/projects/files/hannah/docs/roadmap.md
- /Users/deepedge/Desktop/projects/files/hannah/docs/V1_RELEASE.md
- /Users/deepedge/Desktop/projects/files/hannah/docs/V1_5_RELEASE.md
- /Users/deepedge/Desktop/projects/files/hannah/docs/V2_FINAL_CALL.md
- /Users/deepedge/Desktop/projects/files/AGENTS.md

Repo root:
- /Users/deepedge/Desktop/projects/files

Current status:
- Hannah v1 is green
- v1.5 is complete (S1 agent/provider seam hardening + S2 deterministic trace/replay)
- tests/agent: 9 passed
- hidden acceptance suite: 38 passed
- full suite: 81 passed, 1 warning
- representative v1 CLI smokes (including direct trace command) pass sequentially
- provider seam and deterministic simulator boundary must stay intact
- a prior hosted-LLM smoke exposed a tool-contract bug: `race_data` received an unexpected `lap` argument during a natural-language ask turn

Rules:
- do not rewrite architecture casually
- do not merge donor repos wholesale
- do not let the LLM fake the simulator
- do not put recursive Python execution into the main loop
- do not add new primary logic to the root legacy duplicates

Implementation method for v2:
1. choose one bounded v2 slice
2. write or extend tests first
3. use donor repo sub-agents for exact mapping
4. use worker sub-agents with disjoint write scopes
5. keep hidden acceptance scenario authoring isolated from the main implementation loop
6. implement the slice and keep the main thread as the integrator
7. run targeted tests, scenario tests, hidden acceptance, full suite, and sequential CLI smokes
8. defer legitimate hosted/local model smokes until after the slice is implemented and automated validation is green
9. append a short memory entry before ending the session

Start by confirming the pinned V2 order in `V2_FINAL_CALL.md`, then propose the smallest high-value V2-S1 slice without rewriting the v1.5 architecture, pin the checklist for that slice, and implement it.
```
