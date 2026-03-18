# Hannah Smith Roadmap

Updated: 2026-03-18

## V1 Status

**V1 is GREEN.**

Current validation signal:
- Full automated suite: `117 passed`
- Managed `uv` Python 3.11 env validated for repo runs
- Representative v1 CLI smoke runs (including direct trace command): `6/6 passed`
- SCENARIOS local CLI lane: `12/12 passed`
- SCENARIOS local freeform ask lane: `9/9 passed`
- Hosted fallback-blocked OpenAI lane: `3/3 passed`
- OpenAI-compatible base-override lane via `HANNAH_RLM_API_BASE`: `3/3 passed`
- Optional `rlm-probe` graceful-failure smoke: `1/1 passed`
- Provider seam preserved: hosted model path and deterministic local fallback both work through the same loop
- Deterministic simulator boundary preserved: the LLM orchestrates, tools/simulation produce state transitions and outputs
- Runtime recursion remains out of the v1 main loop

## V1.5 Status

**V1.5 is COMPLETE and GREEN.**

Completed slices:
- Agent loop/provider seam hardening through tests-first coverage
- Added [tests/agent/test_agent_loop_seams.py](/Users/deepedge/Desktop/projects/files/tests/agent/test_agent_loop_seams.py)
- Added [tests/agent/test_subagents_provider_seams.py](/Users/deepedge/Desktop/projects/files/tests/agent/test_subagents_provider_seams.py)
- Added isolated hidden acceptance coverage: [tests/acceptance/test_hidden_acceptance_agent_loop.py](/Users/deepedge/Desktop/projects/files/tests/acceptance/test_hidden_acceptance_agent_loop.py)
- Deterministic replay/debug trace slice through tests-first coverage
- Added [hannah/simulation/replay_trace.py](/Users/deepedge/Desktop/projects/files/hannah/simulation/replay_trace.py)
- Added trace boundary export in [hannah/simulation/monte_carlo.py](/Users/deepedge/Desktop/projects/files/hannah/simulation/monte_carlo.py)
- Added explicit trace mode in [hannah/tools/race_sim/tool.py](/Users/deepedge/Desktop/projects/files/hannah/tools/race_sim/tool.py)
- Added thin direct trace command in [hannah/cli/app.py](/Users/deepedge/Desktop/projects/files/hannah/cli/app.py)
- Added hidden acceptance trace/replay coverage: [tests/acceptance/test_hidden_acceptance_trace_replay.py](/Users/deepedge/Desktop/projects/files/tests/acceptance/test_hidden_acceptance_trace_replay.py)

Slice checklist:
- [x] bounded v1.5 slice selected
- [x] tests written before implementation changes
- [x] donor mapping used for loop/provider semantics
- [x] hidden acceptance authored in isolation
- [x] validation ladder rerun end-to-end
- [x] v1.5 closure docs and handoff prompt updated

## V2 Status

**V2 is COMPLETE and GREEN.**

Completed slices:
- V2-S1 provider/tool boundary hardening: hosted tool-call normalization, provider payload sanitization, OpenF1 session lookup repair, and fallback-blocked hosted smoke validation
- V2-S2 evaluation + scenario depth: explicit evaluation thresholds, scenario-backed evaluation depth, and stronger trace/event-window coherence contracts
- V2-S3 rival strategist depth: donor-guided `pit_policy_q` trainer plus Q-policy metadata wired underneath `RivalAgent`
- V2-S4 operational hardening: atomic artifact writes across every trainer with masked acceptance coverage
- V2-S5 optional runtime helper: `hannah.rlm.helper` plus direct `hannah.py rlm-probe` support outside the main loop

Validation snapshot:
- `tests/agent`: `16 passed`
- `tests/scenarios`: `18 passed`
- `tests/acceptance`: `50 passed`
- full suite: `117 passed`
- core sequential CLI smokes: `6/6 passed`
- optional `rlm-probe` graceful-failure smoke: `1/1 passed`
- hosted fallback-blocked OpenAI smoke lane: `3/3 passed`
- OpenAI-compatible base-override smoke lane: `3/3 passed`

## Purpose

Build Hannah as:
- a nanobot-style CLI agent kernel
- a FormulaGPT-inspired deterministic race strategy system
- a pit-stop-simulator-inspired training and prediction stack
- an rlm-cli-inspired optional Python runtime helper later

Core rule:
- The LLM is the strategist/orchestrator.
- The simulator and ML stack produce the state transitions and predictions.
- Recursive Python execution is optional and must not contaminate the main loop.

## Sources Of Truth

Local Hannah repo:
- `/Users/deepedge/Desktop/projects/files`

Primary docs:
- [GOAL.md](/Users/deepedge/Desktop/projects/files/hannah/docs/GOAL.md)
- [PRD.md](/Users/deepedge/Desktop/projects/files/hannah/docs/PRD.md)
- [ARCHITECTURE.md](/Users/deepedge/Desktop/projects/files/hannah/docs/ARCHITECTURE.md)
- [AGENTS.md](/Users/deepedge/Desktop/projects/files/AGENTS.md)
- [roadmap.md](/Users/deepedge/Desktop/projects/files/hannah/docs/roadmap.md)
- [V1_5_RELEASE.md](/Users/deepedge/Desktop/projects/files/hannah/docs/V1_5_RELEASE.md)
- [V2_FINAL_CALL.md](/Users/deepedge/Desktop/projects/files/hannah/docs/V2_FINAL_CALL.md)

Donor repos:
- `nanobot`: `/Users/deepedge/Desktop/projects/nanobot`
- `FormulaGPT`: `/Users/deepedge/Desktop/projects/FormulaGPT`
- `pit-stop-simulator`: `/Users/deepedge/Desktop/projects/pit-stop-simulator`
- `rlm-cli`: `/Users/deepedge/Desktop/projects/rlm-cli`

## Donor Porting Rules

### `nanobot`

Use:
- agent loop shape
- provider seam
- tool registry ideas
- memory/session separation
- thin CLI orchestration

Do not port:
- channels
- bus
- cron
- heartbeat
- extra product transport glue

### `FormulaGPT`

Use:
- deterministic race-state semantics
- command grammar
- strategist prompt structure
- scoreboard/pit projection ideas

Do not port:
- React/frontend structure
- browser animation loop
- frontend-only state plumbing

### `pit-stop-simulator`

Use:
- environment design ideas
- tyre/lap-time feature ideas
- pit reward shaping
- PPO/Q-learning smoke concepts

Do not port:
- Streamlit layer
- old `env/race_env.py`
- README claims that contradict implementation

### `rlm-cli`

Use later only:
- optional runtime bridge ideas

Do not port into v1:
- recursive runtime as the default control path
- REPL/viewer/TUI product logic

## Current Hannah Target Map

Primary runtime code:
- [hannah/agent](/Users/deepedge/Desktop/projects/files/hannah/agent)
- [hannah/domain](/Users/deepedge/Desktop/projects/files/hannah/domain)
- [hannah/data](/Users/deepedge/Desktop/projects/files/hannah/data)
- [hannah/simulation](/Users/deepedge/Desktop/projects/files/hannah/simulation)
- [hannah/tools](/Users/deepedge/Desktop/projects/files/hannah/tools)
- [hannah/models](/Users/deepedge/Desktop/projects/files/hannah/models)
- [hannah/providers](/Users/deepedge/Desktop/projects/files/hannah/providers)
- [hannah/rlm](/Users/deepedge/Desktop/projects/files/hannah/rlm)

Thin entrypoints and config:
- [hannah.py](/Users/deepedge/Desktop/projects/files/hannah.py)
- [config.yaml](/Users/deepedge/Desktop/projects/files/config.yaml)
- [pyproject.toml](/Users/deepedge/Desktop/projects/files/pyproject.toml)

Tests:
- [tests/scenarios](/Users/deepedge/Desktop/projects/files/tests/scenarios)
- [tests/acceptance](/Users/deepedge/Desktop/projects/files/tests/acceptance)
- [tests/simulation](/Users/deepedge/Desktop/projects/files/tests/simulation)
- [tests/models](/Users/deepedge/Desktop/projects/files/tests/models)
- [tests/agent](/Users/deepedge/Desktop/projects/files/tests/agent)

## Phase Status

| Phase | Status | Notes |
|---|---|---|
| Phase A: scenario-first design | GREEN | Public scenario matrix exists with 20 contracts under `tests/scenarios/`; hidden acceptance coverage exists under `tests/acceptance/`. |
| Phase B: deterministic race domain | GREEN | Domain command parsing, team metadata, track metadata, race snapshot helpers, and strategist prompt helpers are in place under `hannah/domain/`. |
| Phase C: sim engine real | GREEN | Deterministic simulation stack exists under `hannah/simulation/` with dedicated unit coverage. |
| Phase D: honest tools | GREEN | `race_data`, `race_sim`, `pit_strategy`, `predict_winner`, and `train_model` all route through real code paths and stable output shapes. |
| Phase E: training real | GREEN | Deterministic smoke trainers write artifacts for tyre, lap-time, pit policy, Q-learning pit policy, and winner models. |
| Phase F: rival/team sub-agents | GREEN | Sub-agents now carry structured trainable Q-policy metadata underneath `RivalAgent`; provider text remains optional narration, not simulator ownership. |
| Phase G: optional Python runtime helper | GREEN (optional) | `hannah.rlm.helper` and `hannah.py rlm-probe` are in place and stay outside the default loop. |

## Test Strategy

The intended order is now implemented:

1. Import tests
- package imports
- tool discovery
- config loading

2. Deterministic unit tests
- command parsing
- tyre degradation math
- pit-window heuristics
- environment stepping

3. Scenario tests
- public contract validation under `tests/scenarios/`

4. Hidden acceptance tests
- stronger masked end-to-end assertions under `tests/acceptance/`

5. Training smoke tests
- artifact creation
- one short deterministic training pass
- one prediction/evaluation path

6. CLI smoke runs
- direct CLI validation through the provider seam

## Current Validation

### Automated

- `python3 -m pytest -q /Users/deepedge/Desktop/projects/files/tests`
- Result on 2026-03-18: `117 passed`

Targeted slice runs from the completed V2 pass:
- `python3 -m pytest -q /Users/deepedge/Desktop/projects/files/tests/agent`
- Result: `16 passed`
- `python3 -m pytest -q /Users/deepedge/Desktop/projects/files/tests/scenarios`
- Result: `18 passed`
- `python3 -m pytest -q /Users/deepedge/Desktop/projects/files/tests/acceptance`
- Result: `50 passed`
- `python3 -m pytest -q /Users/deepedge/Desktop/projects/files/tests/rlm/test_runtime_helper.py /Users/deepedge/Desktop/projects/files/tests/models/test_atomic_artifact_writes.py /Users/deepedge/Desktop/projects/files/tests/agent/test_v2_s3_rival_q_policy_backend.py`
- Result: `11 passed, 1 warning`
- `python3 -m pytest -q /Users/deepedge/Desktop/projects/files/tests/agent/test_agent_loop_seams.py /Users/deepedge/Desktop/projects/files/tests/acceptance/test_hidden_acceptance_v2_s1_boundaries.py`
- Result: `10 passed`

Environment note:
- prefer `.venv/bin/python` from the managed `uv` Python 3.11 env for repo validation
- the old `/usr/bin/python3` 3.9 path can still emit host-shell warning noise, but the repo validation ladder above is clean in `.venv`

### Representative V1 CLI Smokes

All of these were run sequentially with `HANNAH_FORCE_LOCAL_PROVIDER=1` to verify the deterministic v1 path.

1. `python3 /Users/deepedge/Desktop/projects/files/hannah.py train all --years 2022,2023,2024 --races bahrain,monaco,singapore`
Status: PASS

2. `python3 /Users/deepedge/Desktop/projects/files/hannah.py strategy --race bahrain --lap 18 --driver VER --type optimal`
Status: PASS

3. `python3 /Users/deepedge/Desktop/projects/files/hannah.py strategy --race monaco --lap 30 --driver LEC --type overcut`
Status: PASS

4. `python3 /Users/deepedge/Desktop/projects/files/hannah.py simulate --race silverstone --driver VER --laps 52 --weather mixed`
Status: PASS

5. `python3 /Users/deepedge/Desktop/projects/files/hannah.py predict --race singapore --year 2025`
Status: PASS

6. `python3 /Users/deepedge/Desktop/projects/files/hannah.py trace --race silverstone --year 2025 --drivers VER,NOR,LEC --laps 52 --weather mixed --checkpoints 12,26,52`
Status: PASS

Notes from smoke runs:
- The OpenF1 lookup shape bug is fixed; FastF1 remains the reliable historical fetch path and the hosted/local seams stay tolerant when live data is imperfect.
- The old parallel artifact-read race is now mitigated by atomic artifact writes, but the release ladder still uses sequential CLI validation.
- Optional runtime probing is available through `hannah.py rlm-probe` and remains outside the default agent loop.
- Hosted OpenAI prompts now stay inside context limits because oversized `race_data` tool payloads are compacted before the second provider pass.

## V1 To V2 Ladder

### V1

Achieved:
- CLI agent works
- race data tool works
- deterministic race scenarios run
- Monte Carlo returns useful structure
- basic pit strategy works
- winner prediction works
- training scripts save artifacts

### V1.5

Completed:
- S1 agent/provider seam hardening
- S2 deterministic replay/debug traces

Post-v1.5 likely upgrades:
- stronger model evaluation
- broader scenario library
- richer rival strategist behavior (without collapsing simulator boundaries)
- safer artifact writes if parallel training + inference becomes a supported workflow

### V2

Achieved:
- provider/tool boundary hardening stays config-only across hosted and local-compatible routing
- evaluation is scenario-backed and thresholded instead of score-only
- rival strategy depth now includes a donor-guided trainable Q-policy backend under the sub-agent layer
- artifact writes are atomic across all trainers
- optional runtime probing exists without entering the main loop

Never break:
- provider boundary
- deterministic simulator boundary
- tool registry boundary
- scenario test suite

## Post-V2 Checklist

1. Read:
- [GOAL.md](/Users/deepedge/Desktop/projects/files/hannah/docs/GOAL.md)
- [PRD.md](/Users/deepedge/Desktop/projects/files/hannah/docs/PRD.md)
- [ARCHITECTURE.md](/Users/deepedge/Desktop/projects/files/hannah/docs/ARCHITECTURE.md)
- [V2_FINAL_CALL.md](/Users/deepedge/Desktop/projects/files/hannah/docs/V2_FINAL_CALL.md)
- [AGENTS.md](/Users/deepedge/Desktop/projects/files/AGENTS.md)
- [roadmap.md](/Users/deepedge/Desktop/projects/files/hannah/docs/roadmap.md)

2. Confirm:
- donor repo paths still exist
- Python env still has the required packages
- full test suite is still green
- hosted fallback-blocked smoke is still green if provider-boundary changes are made

3. Treat V2 as stable.

4. Only start post-V2 work if it stays bounded:
- real local-RLM training or stronger local endpoint behavior behind the existing seam
- richer rival policy families beyond the current Q-learning baseline
- deeper evaluation/reporting surfaces if they stay deterministic and tool-owned

5. Keep runtime recursion out of the main Hannah loop.

## Post-V2 Prompt

Use this prompt if you start post-V2 work:

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
- Hannah v2 is complete
- tests/agent is 16 passed
- tests/scenarios is 18 passed
- hidden acceptance suite is 50 passed
- full suite is 117 passed
- representative core CLI smokes (including direct trace command) pass sequentially
- optional rlm-probe smoke passes as a graceful failure/reporting path
- hosted fallback-blocked smoke passes
- managed uv Python 3.11 env is available and preferred for repo runs
- provider seam and deterministic simulator boundary must stay intact

Rules:
- do not rewrite architecture casually
- do not merge donor repos wholesale
- do not let the LLM fake the simulator
- do not put recursive Python execution into the main loop
- do not add new primary logic to the root legacy duplicates

Implementation method for post-v2 work:
1. choose one bounded post-v2 slice
2. write or extend tests first
3. use donor repo sub-agents for exact mapping
4. use worker sub-agents with disjoint write scopes
5. keep hidden acceptance scenario authoring isolated from the main implementation loop
6. implement the slice and keep the main thread as the integrator
7. run targeted tests, scenario tests, hidden acceptance, full suite, and sequential CLI smokes
8. defer legitimate hosted/local model smokes until after the slice is implemented and automated validation is green
9. append a short memory entry before ending the session

Start by confirming V2 is complete, then propose the smallest bounded post-V2 slice without rewriting the architecture, pin the checklist for that slice, and implement it.
```
