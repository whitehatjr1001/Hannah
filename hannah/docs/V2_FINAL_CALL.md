# Hannah Smith V2 Final Call

Updated: 2026-03-18

## Mission

Complete Hannah from the stable v1.5 base without rewriting the architecture.

The core rule stays the same:
- the LLM is the strategist and orchestrator
- the simulator, tools, and model stack own deterministic state transitions, predictions, and replayable outputs

V2 is not a greenfield rebuild. It is a bounded completion phase.

## Closure Status

Current repo state:
- v1 is green
- v1.5 is complete
- v2 is complete
- `tests/agent`: `16 passed`
- `tests/scenarios`: `18 passed`
- hidden acceptance suite: `50 passed`
- full suite: `117 passed`
- sequential core CLI smokes: `6/6 passed`
- optional `rlm-probe` smoke: `1/1 passed`
- hosted fallback-blocked OpenAI smoke lane: `3/3 passed`
- OpenAI-compatible base-override smoke lane: `3/3 passed`
- managed `uv` Python 3.11 env is validated for repo runs
- provider seam is intact
- deterministic simulator boundary is intact

Resolved hosted-path signal:
- the prior hosted smoke that surfaced `race_data(..., lap=...)` contract noise is now covered by the hardened tool/provider boundary
- oversized `race_data` payloads are compacted before the second provider pass, so hosted OpenAI calls no longer blow past the context window
- the hosted path remains real and usable without changing simulator ownership

Read first:
- [GOAL.md](/Users/deepedge/Desktop/projects/files/hannah/docs/GOAL.md)
- [PRD.md](/Users/deepedge/Desktop/projects/files/hannah/docs/PRD.md)
- [ARCHITECTURE.md](/Users/deepedge/Desktop/projects/files/hannah/docs/ARCHITECTURE.md)
- [roadmap.md](/Users/deepedge/Desktop/projects/files/hannah/docs/roadmap.md)
- [V1_RELEASE.md](/Users/deepedge/Desktop/projects/files/hannah/docs/V1_RELEASE.md)
- [V1_5_RELEASE.md](/Users/deepedge/Desktop/projects/files/hannah/docs/V1_5_RELEASE.md)
- [AGENTS.md](/Users/deepedge/Desktop/projects/files/AGENTS.md)

## Non-Negotiables

Never break these:
- provider boundary
- deterministic simulator boundary
- tool registry boundary
- public scenario matrix
- hidden acceptance separation

Never do these:
- do not rewrite architecture casually
- do not merge donor repos wholesale
- do not let the LLM fake the simulator
- do not put recursive Python execution into the main loop
- do not add new primary logic to the root legacy duplicates

## V2 Definition Of Done

V2 is complete when these are true:

1. Local RLM swap is real through the same provider seam.
   - hosted and local model routing remains a config change, not an agent rewrite
   - OpenAI-compatible local endpoint behavior is covered by tests and smokes

2. Evaluation is strong enough to compare strategy/model/provider changes honestly.
   - better evaluation depth
   - stable comparison artifacts
   - no fake quality claims

3. Scenario coverage is broader where Hannah is still weak.
   - rival pressure
   - event windows
   - crossover / mixed conditions
   - trace / replay and acceptance stability

4. Rival strategist behavior is deeper without collapsing simulator ownership.
   - richer domain state
   - better deterministic pressures surfaced to rivals
   - no prompt-only illusion of intelligence

5. Supported workflows are operationally safer.
   - artifact writes are atomic if parallel workflows are meant to be supported
   - failure modes are explicit instead of accidental

6. Optional runtime helper, if revisited, stays optional.
   - no recursive runtime as the default Hannah loop

## Validation Policy

For v2, do not start the session by burning time on live hosted/local smokes.

Use this order:
- lock the slice with tests first
- implement against donor-guided mappings
- keep hidden acceptance isolated
- run automated validation first
- run legitimate hosted/local user-style smokes only after the bounded slice is implemented and green

The prior hosted smoke still matters because it exposed a real tool-contract bug. It should shape the implementation plan, but it should not hijack the kickoff into an early manual smoke session.

## Pinned V2 Order

- [x] **V2-S1 — Provider + Tool Boundary Hardening**
  Hardened tool-call normalization, provider payload sanitization, and live hosted-path behavior without changing loop ownership.
- [x] **V2-S2 — Evaluation + Scenario Depth**
  Added scenario-backed evaluation depth, thresholds, stronger event-window contracts, and deeper masked acceptance coverage.
- [x] **V2-S3 — Rival Strategist Depth**
  Added a donor-guided trainable Q-policy backend under `RivalAgent` instead of relying on prompt-only rival behavior.
- [x] **V2-S4 — Operational Hardening**
  Switched all training artifacts to atomic writes and covered the workflow with unit and hidden acceptance tests.
- [x] **V2-S5 — Optional Runtime Helper**
  Added an optional runtime-helper seam and `rlm-probe` without introducing recursive runtime execution into the main loop.

## Completed Slices

### V2-S1

Provider + tool boundary hardening.

Delivered:
- signature-aware tool-call normalization under the agent tool registry
- provider payload coercion/sanitization across hosted and local-compatible paths
- OpenF1 session lookup repair without moving data ownership away from the tool layer
- fallback-blocked hosted smoke proving the real provider path still works after hardening

### V2-S2

Evaluation and scenario depth.

Delivered:
- `evaluate_model` thresholding and explicit `meets_threshold`
- scenario-backed `evaluation_depth` with scorecard, coverage, and stability signals
- stronger public and hidden scenario pressure around mixed conditions, event windows, and evaluation contract shape
- replay trace enrichment so event windows are visible in deterministic output

### V2-S3

Rival strategist depth.

Delivered:
- donor-guided lightweight Q-learning trainer at [hannah/models/train_pit_q.py](/Users/deepedge/Desktop/projects/files/hannah/models/train_pit_q.py)
- rival metadata and action selection wired through [hannah/agent/subagents.py](/Users/deepedge/Desktop/projects/files/hannah/agent/subagents.py)
- `train all` and evaluation flows extended to include `pit_policy_q`
- public agent/scenario coverage proving the rival layer stays deterministic and tool-owned

### V2-S4

Operational hardening.

Delivered:
- atomic artifact write helpers under [hannah/models/artifacts.py](/Users/deepedge/Desktop/projects/files/hannah/models/artifacts.py)
- all supported trainers switched off direct in-place artifact writes
- masked acceptance coverage for artifact integrity and training workflow safety

### V2-S5

Optional runtime helper.

Delivered:
- optional helper config/probe utilities under [hannah/rlm/helper.py](/Users/deepedge/Desktop/projects/files/hannah/rlm/helper.py)
- direct [hannah/cli/app.py](/Users/deepedge/Desktop/projects/files/hannah/cli/app.py) `rlm-probe` command
- no recursion added to the default Hannah control loop

## Implementation Method

The completed V2 pass followed the intended loop:

1. Choose one bounded v2 slice.
2. Write or extend tests first.
3. Use donor repo sub-agents for exact mapping.
4. Use worker sub-agents with disjoint write scopes.
5. Keep hidden acceptance scenario authoring isolated from the main implementation loop.
6. Implement the slice and keep the main thread as the integrator.
7. Run targeted tests, scenario tests, hidden acceptance, full suite, and sequential CLI smokes.
8. Run legitimate hosted/local model smokes only after the bounded slice is implemented and automated validation is green.
9. Append a short memory entry before ending the session.

## Post-V2 Prompt

Use this prompt if post-V2 work is intentionally started:

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
- Hannah v1.5 is complete
- Hannah v2 is complete
- tests/agent is 16 passed
- tests/scenarios is 18 passed
- hidden acceptance suite is 50 passed
- full suite is 117 passed
- representative core CLI smokes (including direct trace command) pass sequentially
- optional rlm-probe smoke passes
- hosted fallback-blocked smoke passes
- OpenAI-compatible base-override smokes pass through `HANNAH_RLM_API_BASE`
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
