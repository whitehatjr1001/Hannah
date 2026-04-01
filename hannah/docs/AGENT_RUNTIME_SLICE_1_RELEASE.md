# Hannah Agent Runtime Slice 1 Release Notes

Updated: 2026-03-25

## Release Call

**Ship the runtime migration slice.**

This release changes Hannah from a command-led loop into an agent-first CLI runtime with shared orchestration, bounded runtime workers, and persisted runtime events.

The important release call is:

- `hannah agent` is now the primary runtime surface
- `RuntimeCore` now owns the turn loop
- generic bounded subagents are live through a runtime `spawn` tool
- runtime events now stream and persist
- legacy user-facing commands still work through compatibility paths

---

## What Shipped

### 1. Shared runtime core

What changed:

- extracted a shared `RuntimeCore`
- moved provider/tool roundtrip ownership into that core
- reduced `AgentLoop` to a Hannah-specific compatibility adapter

Key files:

- `hannah/runtime/core.py`
- `hannah/runtime/context.py`
- `hannah/runtime/turn_state.py`
- `hannah/agent/loop.py`

### 2. Agent-first CLI surface

What changed:

- `hannah agent` is now the primary runtime command
- wrapper commands route into the same runtime path
- ephemeral wrapper behavior was preserved where required

Key files:

- `hannah/cli/app.py`
- `hannah/cli/agent_command.py`
- `hannah/cli/chat.py`
- `hannah/cli/format.py`

### 3. Generic bounded worker runtime

What changed:

- added a runtime `spawn` tool
- replaced fixed subagent workflow assumptions with generic worker specs
- enforced `allowed_tools` and `result_contract`
- blocked nested spawn for this slice

Key files:

- `hannah/agent/worker_runtime.py`
- `hannah/agent/tool_registry.py`
- `hannah/agent/subagents.py`

### 4. Event streaming and persistence

What changed:

- runtime event bus is now first-class
- worker progress is visible in the CLI
- runtime events persist as JSONL-safe event records
- session-backed runtime flows now capture both messages and runtime events

Key files:

- `hannah/runtime/events.py`
- `hannah/runtime/bus.py`
- `hannah/session/manager.py`
- `hannah/session/event_records.py`

### 5. Acceptance and compatibility locks

What changed:

- acceptance coverage now locks the wrapper/runtime contract
- runtime event ordering is explicitly covered
- tool-boundary hardening remains covered alongside the new loop

Key files:

- `tests/acceptance/test_hidden_acceptance_agent_loop.py`
- `tests/acceptance/test_hidden_acceptance_runtime_events.py`
- `tests/agent/test_v2_s1_tool_boundary_hardening.py`
- `tests/agent/test_worker_spawn_policy.py`
- `tests/agent/test_worker_result_reinjection.py`
- `tests/runtime/test_runtime_core.py`

---

## Validation Snapshot

### Automated

- `.venv/bin/pytest -q`
- Result: `178 passed in 24.48s`

### Targeted runtime acceptance

- `.venv/bin/pytest -q tests/acceptance/test_hidden_acceptance_agent_loop.py tests/acceptance/test_hidden_acceptance_runtime_events.py tests/agent/test_v2_s1_tool_boundary_hardening.py`
- Result: `10 passed in 0.61s`

### Interactive entrypoint smokes

Commands run:

1. `printf 'agent\nexit\n' | .venv/bin/python -m hannah.cli.app`
2. `printf 'sessions\nexit\n' | .venv/bin/python -m hannah.cli.app`

Result: both exited cleanly.

### Final verification details

- `git diff --check c716d7c..8350727` is clean
- managed Python 3.11 environment remains the preferred validation path

---

## Architecture Outcome

Before this release, the runtime still felt heavily command-led.

After this release:

- the CLI mostly acts as ingress
- the runtime owns orchestration
- workers are runtime-created, not hardwired workflow steps
- event streaming and event persistence are built into the runtime

This is closer to the intended nanobot-style product shape without discarding Hannah’s F1-specific tool and simulator boundaries.

---

## Boundaries Preserved

These constraints still hold after the migration:

- tools and simulation layers still own deterministic F1 work
- the provider seam remains the model boundary
- the LLM is still the orchestrator, not the simulator
- worker autonomy is bounded by restricted tools and explicit result contracts
- nested worker recursion is still blocked in this slice

---

## Known Caveats

Still true after this release:

- one independent extra review agent was still pending when the release note was written, but the code itself was directly re-verified and green
- the repo still has unrelated existing changes that were intentionally left untouched
- external live-data volatility remains a separate concern from this runtime migration

Untouched unrelated paths:

- `hannah/tools/race_data/tool.py`
- `docs/superpowers/plans/`
- `hannah/_data_/`

---

## Next Step

Treat this as the new runtime baseline.

Post-release work should now be bounded follow-on slices, for example:

- richer worker policies beyond depth-1 spawn
- better worker prompt/skill loading
- stronger session replay and runtime inspection tooling
- deeper autonomous research/analysis flows on top of the same runtime core
