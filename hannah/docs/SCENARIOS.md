# Hannah Production Smoke Scenarios

Updated: 2026-03-18

## Purpose

This file is the production smoke runbook for Hannah.

Use it when a human or agent needs to exercise:
- every supported CLI surface,
- the freeform agent path,
- the provider seam,
- the main failure cases likely to appear in production.

This file complements:
- [tests/scenarios/README.md](/Users/deepedge/Desktop/projects/files/tests/scenarios/README.md)
- [V1_RELEASE.md](/Users/deepedge/Desktop/projects/files/hannah/docs/V1_RELEASE.md)
- [V1_5_RELEASE.md](/Users/deepedge/Desktop/projects/files/hannah/docs/V1_5_RELEASE.md)
- [V2_FINAL_CALL.md](/Users/deepedge/Desktop/projects/files/hannah/docs/V2_FINAL_CALL.md)

This file is not:
- a hidden acceptance answer key,
- a claim that every future user prompt is enumerable,
- permission to loosen Hannah's architecture to satisfy a smoke.

## Core Rules

- Keep Hannah as a strategy orchestrator first.
- Do not let the LLM fake the simulator.
- Keep provider swap behavior config-only.
- Do not count silent fallback as a hosted or local-endpoint pass.
- Keep hidden acceptance authoring separate from this smoke runbook.
- Run automated validation before live provider smokes.

## Status Tags

- `NOW`: should be runnable on the current green v1.5 base.
- `V2-S1`: required once provider + tool boundary hardening lands.
- `V2+`: later lane, useful after the first v2 slice is stable.

## Common Setup

```bash
export HANNAH_ROOT=/Users/deepedge/Desktop/projects/files
cd "$HANNAH_ROOT"
```

Provider modes used in this file:

- Local deterministic lane

```bash
export HANNAH_FORCE_LOCAL_PROVIDER=1
unset HANNAH_RLM_API_BASE
unset HANNAH_RLM_API_KEY
```

- Hosted live lane

```bash
unset HANNAH_FORCE_LOCAL_PROVIDER
export HANNAH_MODEL=<hosted-model-name>
```

- Local OpenAI-compatible lane

```bash
unset HANNAH_FORCE_LOCAL_PROVIDER
export HANNAH_MODEL=<local-openai-compatible-model-name>
export HANNAH_RLM_API_BASE=<local-openai-compatible-base>
export HANNAH_RLM_API_KEY=<local-openai-compatible-key-if-needed>
```

## Execution Order

1. Run the automated gate.
2. Run the local deterministic CLI lane.
3. Run the local freeform ask lane.
4. Run provider seam smokes only after the slice is implemented and automated validation is green.
5. Run failure drills relevant to the current slice.
6. Close with the release signoff checklist.

## Automated Gate

Run these first:

```bash
python3 -m pytest -q "$HANNAH_ROOT/tests/agent"
python3 -m pytest -q "$HANNAH_ROOT/tests/scenarios"
python3 -m pytest -q "$HANNAH_ROOT/tests/acceptance"
python3 -m pytest -q "$HANNAH_ROOT/tests"
```

Pass criteria:
- no unexpected failures,
- no new traceback loops,
- the current slice remains green before any live-provider smoke begins.

## Lane A: CLI Sanity

| ID | Status | Command | Pass criteria |
|---|---|---|---|
| C00 | NOW | `python3 "$HANNAH_ROOT/hannah.py" --help` | CLI prints command list and exits cleanly. |
| C01 | NOW | `python3 "$HANNAH_ROOT/hannah.py" tools` | Registered tools render without import warnings becoming fatal. |
| C02 | NOW | `python3 "$HANNAH_ROOT/hannah.py" model` | Active model metadata prints cleanly. |

## Lane B: Deterministic CLI Smokes

Run these sequentially with `HANNAH_FORCE_LOCAL_PROVIDER=1`.

| ID | Status | Command | Broad expected path | Pass criteria |
|---|---|---|---|---|
| C10 | NOW | `python3 "$HANNAH_ROOT/hannah.py" strategy --race bahrain --lap 18 --driver VER --type optimal` | `race_data -> race_sim -> pit_strategy` | One decisive pit call, no traceback, recommendation shape is intact. |
| C11 | NOW | `python3 "$HANNAH_ROOT/hannah.py" strategy --race monaco --lap 30 --driver LEC --type overcut` | `race_data -> race_sim -> pit_strategy` | Overcut-style reasoning completes without breaking the tool loop. |
| C12 | NOW | `python3 "$HANNAH_ROOT/hannah.py" simulate --race silverstone --driver VER --laps 52 --weather mixed` | `race_sim` via agent orchestration | Simulation-oriented answer completes without LLM fabricating the sim payload. |
| C13 | NOW | `python3 "$HANNAH_ROOT/hannah.py" predict --race singapore --year 2025` | `race_data -> predict_winner` | Winner probability output is surfaced cleanly. |
| C14 | NOW | `python3 "$HANNAH_ROOT/hannah.py" fetch --race bahrain --year 2025 --session R --driver VER` | `race_data` | Fetch/cache confirmation completes even if external data is partial. |
| C15 | NOW | `python3 "$HANNAH_ROOT/hannah.py" trace --race silverstone --year 2025 --drivers VER,NOR,LEC --laps 52 --weather mixed --checkpoints 12,26,52` | direct `race_sim` with `trace=True` | Trace output includes replay/timeline structure and exits cleanly. |
| C16 | NOW | `python3 "$HANNAH_ROOT/hannah.py" sandbox --agents VER,NOR,LEC --race bahrain --laps 12 --weather dry` | agent loop over sandbox prompt | Multi-driver sandbox turn completes without loop failure. |
| C17 | NOW | `python3 "$HANNAH_ROOT/hannah.py" train tyre_model --years 2023,2024 --races bahrain,jeddah` | `train_model` | Saved artifact path returns cleanly. |
| C18 | NOW | `python3 "$HANNAH_ROOT/hannah.py" train all --years 2022,2023,2024 --races bahrain,monaco,singapore` | `train_model` | All-model training returns a dict of saved artifacts and does not corrupt later reads. |

## Lane C: Freeform User-Style Ask Prompts

Run these first in local deterministic mode.
Promote the same prompts into hosted or local-endpoint lanes only after the current slice is green.

| ID | Status | Command | Broad expected path | Pass criteria |
|---|---|---|---|---|
| A10 | NOW | `python3 "$HANNAH_ROOT/hannah.py" ask "Should VER box on lap 18 in Bahrain 2025? Use race data and simulation tools if needed, then give one decisive pit-wall call."` | `race_data -> race_sim -> pit_strategy` | One decisive call, not two contradictory strategies. |
| A11 | NOW | `python3 "$HANNAH_ROOT/hannah.py" ask "Compare undercut versus overcut for LEC at Monaco on lap 30. Give one final pit-wall call and the main rival threat."` | `race_data -> race_sim -> pit_strategy` | Final recommendation chooses one path and names the main rival pressure. |
| A12 | NOW | `python3 "$HANNAH_ROOT/hannah.py" ask "If a safety car lands around lap 24 in Singapore, what is the best move for VER? Use race data and simulation tools before answering."` | `race_data -> race_sim -> pit_strategy` | Safety-car framing does not break tool selection or the answer shape. |
| A13 | NOW | `python3 "$HANNAH_ROOT/hannah.py" ask "Run a mixed-weather Silverstone simulation for VER, NOR, and LEC over 52 laps and summarize the most important pit windows."` | `race_sim` | Answer is grounded in a simulation path, not pure prose. |
| A14 | NOW | `python3 "$HANNAH_ROOT/hannah.py" ask "Project winner odds for Singapore 2025 between VER, NOR, and LEC, then explain the top two outcomes."` | `race_data -> predict_winner` | Probability answer is surfaced clearly and stays within the winner-prediction tool boundary. |
| A15 | NOW | `python3 "$HANNAH_ROOT/hannah.py" ask "Fetch Bahrain 2025 race data for VER and tell me what telemetry families were retrieved."` | `race_data` | Fetch answer completes even if upstream data is partial. |
| A16 | NOW | `python3 "$HANNAH_ROOT/hannah.py" ask "Train the winner ensemble on 2023 and 2024 data and tell me where the artifacts were saved."` | `train_model` | Training answer confirms saved artifacts without hanging in the loop. |
| A17 | V2+ | `python3 "$HANNAH_ROOT/hannah.py" ask "Should we box now?"` | clarifying question or explicit assumptions | Hannah should ask for missing context or state explicit assumptions instead of pretending it knows the race state. |
| A18 | V2+ | `python3 "$HANNAH_ROOT/hannah.py" ask "Write me a poem about F1 strategy."` | scope control | Hannah should stay inside product scope and avoid fake tool use. |

## Lane D: Provider Seam Smokes

These are not kickoff tasks.
Run them only after:
- the bounded slice is implemented,
- the automated gate is green,
- the local deterministic lane is green.

### Fallback-Blocked Provider Harness

Use this pattern whenever the goal is to prove a legitimate hosted or local-endpoint model call.
Do not count a silent local fallback as a pass.

```bash
export PROMPT='Should VER box on lap 18 in Bahrain 2025? Use race data and simulation tools if needed, then give one decisive pit-wall call.'

python3 - <<'PY'
import asyncio
import os
from unittest.mock import patch

from hannah.agent.loop import AgentLoop


async def main() -> None:
    with patch(
        "hannah.providers.litellm_provider.LiteLLMProvider._local_complete",
        side_effect=RuntimeError("fallback blocked"),
    ):
        await AgentLoop().run_command(os.environ["PROMPT"])


asyncio.run(main())
PY
```

### Hosted And Local-Endpoint Matrix

| ID | Status | Mode | Prompt | Pass criteria |
|---|---|---|---|---|
| P10 | V2-S1 | Hosted model | Bahrain lap-18 pit-call prompt from `A10` | Hosted call succeeds without falling back locally; tool loop stays intact. |
| P11 | V2-S1 | Hosted model | Singapore winner-odds prompt from `A14` | Hosted path completes and keeps prediction logic inside `predict_winner`. |
| P12 | V2-S1 | Hosted model | Bahrain fetch prompt from `A15` | Hosted path tolerates partial external data without crashing the turn. |
| P13 | V2-S1 | Local OpenAI-compatible endpoint | Bahrain lap-18 pit-call prompt from `A10` | Same loop contract works through `HANNAH_RLM_API_BASE` without an agent rewrite. |
| P14 | V2-S1 | Local OpenAI-compatible endpoint | Singapore winner-odds prompt from `A14` | Config-only swap works for the prediction path. |
| P15 | V2-S1 | Local OpenAI-compatible endpoint | Mixed-weather simulation prompt from `A13` | Local endpoint path preserves simulator/tool ownership. |

## Lane E: Failure Drills

These are the production cases most likely to expose dishonest seams.

| ID | Status | Execution method | Scenario | Pass criteria |
|---|---|---|---|---|
| F10 | V2-S1 | provider stub or integration test | Model emits extra tool args, such as `race_data` receiving `lap` unexpectedly. | Tool boundary ignores, strips, or safely coerces noisy args and the turn still completes. |
| F11 | NOW | provider stub or integration test | Provider emits an unknown tool name. | Agent serializes the tool error without crashing the loop. |
| F12 | NOW | provider stub or integration test | One tool fails while another tool in the same turn succeeds. | The surviving result still reaches the second pass and the final answer remains bounded. |
| F13 | NOW | monkeypatch network client | OpenF1 or FastF1 returns `404` or `429`. | Turn degrades gracefully and does not fake missing telemetry. |
| F14 | NOW | env switch | Hosted credentials missing. | Hannah either falls back locally or fails explicitly according to the test goal; it does not hang. |
| F15 | V2-S1 | env switch + fallback-blocked harness | `HANNAH_RLM_API_BASE` is set but the endpoint is unreachable. | Failure is explicit and bounded; no silent fallback is counted as a seam pass. |
| F16 | V2+ | parallel shell run | Train and inference overlap on artifacts. | Either atomic writes work or the workflow is explicitly marked unsupported. |
| F17 | V2+ | freeform ask | User asks outside Hannah's domain. | Hannah stays scoped and avoids fake tool use. |

## Release Signoff Checklist

- [ ] Automated gate is green.
- [ ] CLI sanity lane is green.
- [ ] Deterministic CLI smokes are green, run sequentially.
- [ ] Freeform ask lane is green in local deterministic mode.
- [ ] Provider seam lane is green for the prompts relevant to the current slice.
- [ ] Failure drills relevant to the current slice are green.
- [ ] No pass was counted where fallback masked a hosted or local-endpoint failure.
- [ ] Provider seam remains config-only.
- [ ] Deterministic simulator boundary remains intact.
- [ ] Hidden acceptance stayed isolated from the implementation loop.
- [ ] A short session memory entry was appended to `AGENTS.md`.

## Agent Run Log Template

Copy this block into the working session when an agent runs the matrix:

```markdown
### Smoke Session

- Date:
- Slice:
- Provider mode:
- Environment:

| ID | Result | Notes |
|---|---|---|
| C00 |  |  |
| C01 |  |  |
| C02 |  |  |
| C10 |  |  |
| C11 |  |  |
| C12 |  |  |
| C13 |  |  |
| C14 |  |  |
| C15 |  |  |
| C16 |  |  |
| C17 |  |  |
| C18 |  |  |
| A10 |  |  |
| A11 |  |  |
| A12 |  |  |
| A13 |  |  |
| A14 |  |  |
| A15 |  |  |
| A16 |  |  |
| A17 |  |  |
| A18 |  |  |
| P10 |  |  |
| P11 |  |  |
| P12 |  |  |
| P13 |  |  |
| P14 |  |  |
| P15 |  |  |
| F10 |  |  |
| F11 |  |  |
| F12 |  |  |
| F13 |  |  |
| F14 |  |  |
| F15 |  |  |
| F16 |  |  |
| F17 |  |  |
```
