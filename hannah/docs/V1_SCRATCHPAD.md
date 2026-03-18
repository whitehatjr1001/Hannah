# CLAUDE.md — Hannah Smith: Coding Agent Kickstart Instructions
> Feed this file to Claude Code at project start.
> These are your complete build instructions for the Hannah Smith F1 Strategy CLI Agent.

---

## 🎯 What You Are Building

**Hannah Smith** is a CLI-first agentic AI system that acts as a virtual Red Bull Race Director.
It runs fast async F1 race simulations, makes data-backed strategy calls, and predicts race outcomes.

Read `GOAL.md` and `PRD.md` first — they are your north star and architecture spec.
Every decision you make must align with those documents.

---

## 📁 Files Already Provided (your starting scaffold)

```
hannah.py                   ← CLI entrypoint (complete)
agent/core.py               ← LiteLLM agent loop (complete)
agent/persona.py            ← Hannah system prompt (complete)
agent/sub_agents.py         ← async sub-agent spawner (complete)
simulation/monte_carlo.py   ← fast vectorised sim engine (complete)
rlm/server.py               ← v2 RLM FastAPI scaffold (complete)
requirements.txt            ← all dependencies (complete)
.env.example                ← env var template (complete)
config.yaml                 ← runtime config (complete)
GOAL.md                     ← project goals + success metrics
PRD.md                      ← full architecture + spec
```

**Do not rewrite these files unless a bug is found.**
Your job is to build everything that is missing around them.

---

## 🔨 Step-by-Step Build Instructions

Work through these phases in order. Complete each phase fully before moving to the next.
After each phase, run the smoke test listed and confirm it passes.

---

### PHASE 1 — Project Wiring + Init Files

**Task:** Create all `__init__.py` files, folder structure, and the memory + tool registry modules.

#### 1.1 Create folder structure

Create these directories if they do not exist:
```
agent/
tools/
tools/race_data/
tools/pit_strategy/
tools/race_sim/
tools/predict_winner/
tools/train_model/
simulation/
models/
models/saved/
rlm/
data/
data/fastf1_cache/
```

#### 1.2 Create `__init__.py` files

Create empty `__init__.py` in: `agent/`, `tools/`, `simulation/`, `models/`, `rlm/`, `data/`

#### 1.3 Create `agent/memory.py`

SQLite-backed session memory. Requirements:
- Class `Memory` with methods: `add(role, content)`, `get_recent(n=10) -> list[dict]`
- Stores messages in `data/hannah_memory.db` (auto-create if not exists)
- Table: `messages(id, role, content, timestamp)`
- `get_recent(n)` returns last N messages as `[{"role": ..., "content": ...}]`
- `clear()` method to wipe session
- No external deps beyond `sqlite3` (stdlib)

#### 1.4 Create `agent/tool_registry.py`

SKILL.md-based tool registry. Requirements:
- Class `ToolRegistry` with methods:
  - `get_tool_specs() -> list[dict]` — returns OpenAI-format tool specs for LiteLLM
  - `call(name, args) -> dict` — async dispatcher to the correct tool
  - `list_tools()` — prints all registered tools to console with Rich
- Auto-discovers tools by scanning `tools/*/tool.py`
- Each tool must have a `SKILL` dict at module level (see SKILL spec below)
- Gracefully handles import errors (prints warning, skips tool)

**SKILL dict format** (every `tools/*/tool.py` must define this at top level):
```python
SKILL = {
    "name": "race_data",
    "description": "Fetches F1 race data from FastF1 and OpenF1.",
    "parameters": {
        "type": "object",
        "properties": {
            "race": {"type": "string", "description": "Race name e.g. bahrain"},
            "year": {"type": "integer", "description": "Season year"},
            "session": {"type": "string", "enum": ["R", "Q", "FP1", "FP2", "FP3"]},
            "driver": {"type": "string", "description": "Driver code e.g. VER (optional)"},
        },
        "required": ["race"],
    }
}

async def run(**kwargs) -> dict:
    ...
```

**Smoke test Phase 1:**
```bash
python -c "from agent.memory import Memory; m = Memory(); m.add('user', 'test'); print(m.get_recent(1))"
python -c "from agent.tool_registry import ToolRegistry; ToolRegistry().list_tools()"
```

---

### PHASE 2 — Data Layer

**Task:** Build the FastF1 and OpenF1 data fetchers.

#### 2.1 Create `data/fastf1_loader.py`

Requirements:
- Function `fetch_session(race, year, session_type) -> dict` — returns dict with keys:
  `laps` (DataFrame), `weather` (DataFrame), `car_data` (DataFrame), `results` (DataFrame)
- Uses `fastf1` library with cache at `data/fastf1_cache/`
- Enables FastF1 cache on import: `fastf1.Cache.enable_cache("data/fastf1_cache")`
- Converts DataFrames to JSON-serialisable dicts before returning
- Handles `fastf1.core.DataNotLoadedError` gracefully — returns empty dict with error key
- Progress shown via Rich console: `[dim]fetching {race} {year} {session}...[/dim]`

#### 2.2 Create `data/openf1_client.py`

Requirements:
- Class `OpenF1Client` with methods:
  - `get_sessions(year, race_name) -> list[dict]`
  - `get_laps(session_key, driver_number=None) -> list[dict]`
  - `get_stints(session_key) -> list[dict]`
  - `get_weather(session_key) -> list[dict]`
  - `get_drivers(session_key) -> list[dict]`
- Base URL: `https://api.openf1.org/v1`
- Uses `requests` with 10s timeout
- Caches responses to `data/fastf1_cache/openf1_{endpoint}_{params_hash}.json`
- Returns empty list on any HTTP error, logs warning via Rich

#### 2.3 Create `data/preprocess.py`

Requirements:
- Function `build_features(laps_df, stints_df, weather_df) -> pd.DataFrame`
- Output columns: `driver`, `lap_number`, `lap_time_s`, `compound`, `tyre_age`,
  `stint_number`, `air_temp`, `track_temp`, `rainfall`, `position`, `gap_to_leader_s`
- Handles missing columns gracefully with sensible defaults
- Function `normalise(df) -> pd.DataFrame` — min-max normalise numeric columns

#### 2.4 Create `tools/race_data/tool.py`

Wire the data layer into a SKILL tool:
- Calls `fastf1_loader.fetch_session` + `OpenF1Client` in parallel using `asyncio.gather`
- Returns merged dict: `{laps, stints, weather, drivers, session_info}`
- Define `SKILL` dict (see format above)
- Define `async def run(**kwargs) -> dict`

**Smoke test Phase 2:**
```bash
python -c "
import asyncio
from tools.race_data.tool import run
result = asyncio.run(run(race='bahrain', year=2024, session='R'))
print('Keys:', list(result.keys()))
"
```

---

### PHASE 3 — Simulation Layer

**Task:** Complete the simulation engine around the provided `monte_carlo.py`.

#### 3.1 Create `simulation/sandbox.py`

Requirements:
- Class `RaceState` — full race state container (extend the stub in `monte_carlo.py`)
  Fields: `race, year, laps, n_drivers, drivers, compounds, base_lap_times, weather,
  safety_car_prob, current_lap, positions, gaps, tyre_ages`
- `from_context(ctx) -> RaceState` classmethod — builds from `RaceContext`
- `from_race_data(race_data_dict) -> RaceState` classmethod — builds from tool output
- `to_dict() -> dict` — JSON-serialisable snapshot
- `update(lap_result) -> None` — advances state by one lap

#### 3.2 Create `simulation/tyre_model.py`

Requirements:
- Class `TyreModel` with method `predict(compound, age, track_temp=30) -> float`
  Returns lap time penalty in seconds for given compound + age
- Uses `COMPOUND_DEG` constants (copy from `monte_carlo.py`)
- `predict_batch(compounds, ages, n_worlds) -> np.ndarray` — vectorised, shape `(n_worlds, n_laps)`
- If `models/saved/tyre_deg_v1.pkl` exists, loads and uses trained sklearn model
- Falls back to physics formula if no saved model found (log a `[dim]using physics fallback[/dim]`)

#### 3.3 Create `simulation/gap_engine.py`

Requirements:
- Class `GapEngine` with methods:
  - `undercut_feasibility(gap_to_ahead, pit_delta=22.5, lap_delta=0.8) -> dict`
    Returns `{feasible: bool, required_gap: float, recommendation: str}`
  - `overcut_feasibility(gap_to_behind, tyre_age_delta, deg_rate) -> dict`
  - `compute_deltas(all_times, pit_laps) -> object` with `.final_positions` and `.total_time`

#### 3.4 Create `simulation/strategy_engine.py`

Requirements:
- Class `StrategyEngine` with method `analyse(race_state, sim_result) -> dict`
- Returns: `{recommended_pit_lap, recommended_compound, strategy_type, confidence,
  undercut_window, rival_threats, reasoning}`
- `strategy_type` is one of: `"undercut"`, `"overcut"`, `"alternate"`, `"stay_out"`
- `confidence` is a float 0.0–1.0

#### 3.5 Create `tools/race_sim/tool.py`

SKILL tool wrapping the simulation:
- Calls `monte_carlo.run_fast(state, n_worlds=1000)`
- Returns sim result as dict
- Also calls `StrategyEngine.analyse()` on the result
- `SKILL` dict + `async def run(**kwargs) -> dict`

**Smoke test Phase 3:**
```bash
python -c "
import asyncio, numpy as np
from simulation.monte_carlo import run_fast, RaceState
state = RaceState('bahrain', 57, 3, ['SOFT','SOFT','MEDIUM'],
                  np.array([90.2,90.8,91.1]), 'dry', 0.18)
result = asyncio.run(run_fast(state, n_worlds=500))
print('Winner probs:', result.winner_probs)
print('Sim time <2s: check')
"
```

---

### PHASE 4 — ML Models + Training CLI

**Task:** Build the four training scripts and the predict_winner + train_model tools.

#### 4.1 Create `models/train_tyre_deg.py`

Requirements:
- Function `train(years: list[int], races: list[str] = None) -> str`
  Returns path to saved model
- Features: `tyre_age, compound_encoded, track_temp, air_temp, rainfall`
- Target: `lap_time_s` (predicting deg component)
- Model: `sklearn.ensemble.GradientBoostingRegressor`
- Saves to `models/saved/tyre_deg_v1.pkl`
- Streams epoch/fold progress to console via Rich progress bar
- Prints final RMSE on held-out test set

#### 4.2 Create `models/train_laptime.py`

Requirements:
- Function `train(years, races=None) -> str`
- PyTorch LSTM: input = `[lap_number, tyre_age, compound_encoded, fuel_load, gap_to_leader]`
- Target: `lap_time_s`
- Hidden: 128, Layers: 2, Dropout: 0.2
- Saves to `models/saved/laptime_v1.pt`
- Prints train/val loss per epoch via Rich

#### 4.3 Create `models/train_winner.py`

Requirements:
- Function `train(years, races=None) -> str` — trains winner ensemble
- Function `load_and_predict(ctx) -> dict` — loads saved model and returns `{driver: prob}`
- Features: `grid_position, q3_time, team_encoded, track_type, tyre_strategy_encoded,
  avg_pace_delta, safety_car_prob`
- Model: `XGBClassifier` + `RandomForestClassifier` soft-vote ensemble
- Saves to `models/saved/winner_ensemble_v1.pkl`
- Accuracy metric: top-3 accuracy on held-out 2024 races

#### 4.4 Create `models/train_pit_rl.py`

Requirements:
- Uses `stable_baselines3.PPO` with a custom `gymnasium.Env`
- Environment `PitStopEnv`:
  - Observation: `[lap, tyre_age, compound, gap_ahead, gap_behind, deg_rate, fuel_remaining]`
  - Actions: `[0: stay_out, 1: pit_soft, 2: pit_medium, 3: pit_hard]`
  - Reward: negative total race time (minimise time = maximise reward)
  - Episode = one full race
- Saves to `models/saved/pit_rl_v1.zip`
- Streams training timesteps via Rich

#### 4.5 Create `models/evaluate.py`

Requirements:
- Function `evaluate_all()` — runs eval on all saved models, prints a Rich table:
  `Model | Metric | Score | Status`
- `evaluate_tyre_deg()` → RMSE
- `evaluate_winner()` → top-3 accuracy
- `evaluate_pit_rl()` → mean reward over 50 episodes

#### 4.6 Create `tools/predict_winner/tool.py` and `tools/train_model/tool.py`

`predict_winner/tool.py`:
- Calls `train_winner.load_and_predict(ctx)`
- Falls back to "model not trained yet" message if no saved model

`train_model/tool.py`:
- Dispatches to the correct `train_*.py` based on `model_name` kwarg
- Streams progress back to agent as text chunks

**Smoke test Phase 4:**
```bash
python -c "from models.evaluate import evaluate_all; evaluate_all()"
```

---

### PHASE 5 — Competitor Agents + Full Sandbox

**Task:** Wire competitor agents and complete the sandbox race loop.

#### 5.1 Create `simulation/competitor_agents.py`

Requirements:
- Class `CompetitorAgent` — wraps `RivalAgent` from `agent/sub_agents.py`
- Method `decide_pit(race_state) -> dict` — returns `{pit_this_lap: bool, compound: str, reasoning: str}`
- Uses LiteLLM with team-specific persona (reuse `RivalAgent.TEAM_PERSONAS`)
- Method `get_all_decisions(race_state, drivers) -> dict[str, dict]` — concurrent via `asyncio.gather`

#### 5.2 Update `simulation/sandbox.py` — add `run_sandbox_race`

```python
async def run_sandbox_race(ctx: RaceContext) -> dict:
    """
    Full multi-agent sandbox race.
    Each lap: competitor agents decide strategy, state advances, gaps update.
    Returns lap-by-lap summary + final positions.
    """
```
- Lap loop is fast (no sleep/delay) — this is prediction mode
- Each lap calls `CompetitorAgent.get_all_decisions()` concurrently
- Accumulates `lap_summaries: list[dict]` with key events (pits, position changes)
- Returns `{final_positions, lap_summaries, key_moments, winner}`

**Smoke test Phase 5:**
```bash
python -c "
import asyncio
from agent.sub_agents import RaceContext, spawn_all
ctx = RaceContext('bahrain', 2025, 57, 'dry', ['VER','NOR','LEC'])
results = asyncio.run(spawn_all(ctx))
print('Sub-agents completed:', list(results.keys()))
"
```

---

### PHASE 6 — End-to-End Integration + Full CLI Test

**Task:** Wire everything together, test all CLI commands end-to-end.

#### 6.1 Verify `agent/core.py` tool dispatch works

The provided `core.py` calls `self.registry.call(fn_name, args)`.
Confirm `ToolRegistry.call()` correctly routes to each `tools/*/tool.py::run()`.

#### 6.2 Run all CLI commands

```bash
# These must all run without import errors (data/model errors are OK if API key missing)
python hannah.py --help
python hannah.py tools
python hannah.py model
python hannah.py predict --race bahrain --year 2025
python hannah.py simulate --race bahrain --driver VER --laps 57
python hannah.py strategy --race bahrain --lap 22 --driver VER
python hannah.py sandbox --agents VER,NOR,LEC --race bahrain
python hannah.py ask "Should I pit under VSC on lap 34?"
```

#### 6.3 Create `data/openpiwall_loader.py`

Requirements:
- Function `load(years: list[int]) -> pd.DataFrame`
- Downloads OpenPitWall dataset from GitHub:
  `https://github.com/theOehrly/Fast-F1/releases` (or use FastF1 ergast fallback)
- Returns DataFrame with columns matching `preprocess.build_features()` output
- Caches to `data/fastf1_cache/openpiwall_{year}.parquet`

#### 6.4 Create `rlm/model.py` stub

```python
class RLMModel:
    def generate(self, messages, temperature=0.2, max_tokens=2048) -> str:
        raise NotImplementedError(
            "RLM model not yet trained. "
            "Run: hannah train all --years 2022,2023,2024 first."
        )
```

#### 6.5 Final integration smoke test

```bash
python -c "
import asyncio
from agent.core import AgentCore
# This should complete without crashing (may need API key for LLM call)
print('AgentCore imports OK')
from simulation.monte_carlo import run_fast, RaceState
import numpy as np
state = RaceState('monza', 53, 2, ['SOFT','MEDIUM'],
                  np.array([79.5, 80.1]), 'dry', 0.12)
r = asyncio.run(run_fast(state, 200))
print(f'Monte Carlo OK — winner probs: {r.winner_probs}')
"
```

---

## 🛑 Hard Rules — Do Not Violate These

1. **Never rewrite `hannah.py`, `agent/core.py`, `agent/persona.py`, `agent/sub_agents.py`,
   `simulation/monte_carlo.py`, `rlm/server.py`** unless fixing a concrete bug.
   These files are the architecture. Build around them.

2. **All LLM calls must go through LiteLLM** — never import `anthropic` directly in new code.
   The only exception is `requirements.txt` where it is a dependency of LiteLLM.

3. **The model is always read from env**: `os.getenv("HANNAH_MODEL", "claude-sonnet-4-6")`
   Never hardcode a model name in tool or simulation code.

4. **All simulation work must be async** — use `asyncio.to_thread()` for CPU-bound numpy work.
   Never block the event loop with a synchronous numpy loop.

5. **Every tool must define a `SKILL` dict** at module top level.
   The `ToolRegistry` auto-discovers tools via this dict.

6. **Models fall back gracefully** — if no `.pkl`/`.pt` file exists in `models/saved/`,
   the tool must return a helpful message, not crash.

7. **Rich console only** — all terminal output goes through `rich.console.Console()`.
   No raw `print()` except in `__main__` guards.

8. **Config from `config.yaml`** — load runtime config via:
   ```python
   import yaml
   with open("config.yaml") as f:
       cfg = yaml.safe_load(f)
   ```
   Never hardcode paths, URLs, or numeric constants that are already in `config.yaml`.

---

## 📦 Dependency Notes

- `fastf1` requires internet on first fetch per race/year — subsequent calls use local cache
- `stable-baselines3` requires `torch` — install order matters: `torch` first, then `stable-baselines3`
- `litellm` auto-installs `anthropic` as a dependency — do not pin `anthropic` version separately
- `gymnasium` must be `>=0.29` for the `PitStopEnv` `step()` return signature `(obs, reward, terminated, truncated, info)`

---

## 🗂️ What the Finished Project Looks Like

When all phases are complete, `python hannah.py --help` should show:

```
Usage: hannah.py [OPTIONS] COMMAND [ARGS]...

  Hannah Smith — F1 Strategy Simulation CLI Agent

Options:
  --help  Show this message and exit.

Commands:
  ask       Ask Hannah a freeform strategy question.
  fetch     Fetch and cache F1 data from FastF1 and OpenF1.
  model     Show the current LiteLLM model in use.
  predict   Predict the winner for an upcoming race.
  sandbox   Run a multi-agent LLM sandbox race.
  simulate  Run a fast async race simulation (prediction mode).
  strategy  Get a pit strategy call for the current race situation.
  tools     List all registered SKILL tools.
  train     Train ML models from the CLI.
```

And `python hannah.py tools` should list:
```
◆ race_data        FastF1 + OpenF1 telemetry fetcher
◆ pit_strategy     RL pit agent caller
◆ race_sim         Monte Carlo sim caller
◆ predict_winner   ensemble model caller
◆ train_model      CLI training launcher
```

---

## 🚀 Quick Start for Coding Agent

```bash
# 1. Set up environment
cp .env.example .env
# Add your ANTHROPIC_API_KEY to .env

# 2. Install dependencies
pip install -r requirements.txt

# 3. Build Phase 1 first
# Create: agent/memory.py, agent/tool_registry.py, all __init__.py files
# Smoke test Phase 1

# 4. Continue through phases in order
# Each phase has a smoke test — run it before moving on

# 5. When all phases pass:
python hannah.py predict --race bahrain --year 2025
```

---

*Hannah Smith v0.1.0-alpha — Red Bull Race Director*
*Built on NanoChat + LiteLLM + FastF1 + OpenPitWall*
*See GOAL.md for success metrics. See PRD.md for full architecture.*