# Hannah Smith V1.5 Release Notes

Updated: 2026-03-18

## Release Call

**Ship Hannah v1.5.**

This is the clean v1.5 ship point:
- v1 remained green while the highest-value near-term seams were hardened
- deterministic trace/replay now exists as a simulator-owned inspection path
- provider seam and deterministic simulator ownership remained intact

## What V1.5 Adds

### S1 — Agent / Provider Seam Hardening

What shipped:
- focused `tests/agent` coverage for loop roundtrip behavior
- tool argument coercion coverage
- tool error serialization coverage
- sub-agent failure containment coverage
- provider selection metadata coverage

Key files:
- `tests/agent/test_agent_loop_seams.py`
- `tests/agent/test_subagents_provider_seams.py`
- `tests/acceptance/test_hidden_acceptance_agent_loop.py`

### S2 — Deterministic Trace / Replay

What shipped:
- simulator-owned replay payloads with deterministic structure
- direct CLI trace command
- replay stability contracts
- public scenario coverage for trace-aware strategy paths

Key files:
- `hannah/simulation/replay_trace.py`
- `hannah/simulation/monte_carlo.py`
- `hannah/tools/race_sim/tool.py`
- `hannah/cli/app.py`
- `hannah/cli/format.py`
- `tests/simulation/test_debug_trace.py`
- `tests/acceptance/test_hidden_acceptance_trace_replay.py`

## Validation Snapshot

### Automated

- `python3 -m pytest -q /Users/deepedge/Desktop/projects/files/tests`
- Result: `81 passed, 1 warning`

### Targeted V1.5 Runs

- `python3 -m pytest -q /Users/deepedge/Desktop/projects/files/tests/agent`
- Result: `9 passed`

- `python3 -m pytest -q /Users/deepedge/Desktop/projects/files/tests/acceptance`
- Result: `38 passed`

- `python3 -m pytest -q /Users/deepedge/Desktop/projects/files/tests/simulation/test_debug_trace.py /Users/deepedge/Desktop/projects/files/tests/scenarios/test_strategy_scenarios.py /Users/deepedge/Desktop/projects/files/tests/test_imports.py`
- Result: `12 passed, 1 warning`

### Sequential CLI Smokes

All run with `HANNAH_FORCE_LOCAL_PROVIDER=1`.

1. `python3 /Users/deepedge/Desktop/projects/files/hannah.py train all --years 2022,2023,2024 --races bahrain,monaco,singapore`
2. `python3 /Users/deepedge/Desktop/projects/files/hannah.py strategy --race bahrain --lap 18 --driver VER --type optimal`
3. `python3 /Users/deepedge/Desktop/projects/files/hannah.py strategy --race monaco --lap 30 --driver LEC --type overcut`
4. `python3 /Users/deepedge/Desktop/projects/files/hannah.py simulate --race silverstone --driver VER --laps 52 --weather mixed`
5. `python3 /Users/deepedge/Desktop/projects/files/hannah.py predict --race singapore --year 2025`
6. `python3 /Users/deepedge/Desktop/projects/files/hannah.py trace --race silverstone --year 2025 --drivers VER,NOR,LEC --laps 52 --weather mixed --checkpoints 12,26,52`

Result: **6/6 passed**

## Boundaries Preserved

These stayed intact through v1.5:
- provider seam remains the runtime swap boundary
- simulator and tools still own state transitions and outputs
- trace/replay is deterministic and simulator-owned, not LLM-generated
- runtime recursion is still excluded from the main loop
- donor repos were used as mapping references, not merged wholesale

## Known Caveats

Still true after v1.5:
- OpenF1 can still return `404` or `429`
- the local shell is still Python 3.9 while `pyproject.toml` targets 3.11
- the recurring `urllib3` / LibreSSL warning is still present
- parallel training plus inference can still race on artifact reads; sequential validation is the supported path

## Shipping Guidance

Treat v1.5 as:
- the stable handoff point after v1 hardening
- the last v1-era release before post-v1.5 / v2-directed work
- the baseline that post-v1.5 work must preserve

Do not use v1.5 as a reason to rewrite architecture.

## Next Artifact

Before any next-phase implementation work, read:
- [V2_FINAL_CALL.md](/Users/deepedge/Desktop/projects/files/hannah/docs/V2_FINAL_CALL.md)

That document is the execution brief for the final push to complete Hannah.
