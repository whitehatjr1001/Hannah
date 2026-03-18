# PRD.md — Hannah Smith: Red Bull Race Director
> Product Requirements Document + Architecture Reference
> Version: 0.1.0-alpha | Status: Kickstart

---

## 1. Overview

**Hannah Smith** is a CLI-first agentic AI system for F1 race strategy simulation and prediction. It is built on the NanoChat architecture pattern, uses LiteLLM as a unified LLM/RLM gateway, and runs fast async Monte Carlo simulations using OpenPitWall training data and FastF1 telemetry.

The system is designed from day one for a clean LLM → RLM migration in v2 — the agent model is fully decoupled behind LiteLLM, and the RLM lives as a FastAPI module that grows alongside v1.

---

## 2. System Architecture

```
CLI (hannah.py)
    └── Agent Core (LiteLLM + NanoChat loop)
            ├── Hannah persona + memory (SQLite)
            ├── Tool dispatcher (SKILL.md registry)
            └── Sub-agent spawner (asyncio.gather)
                    ├── sim_agent       → Monte Carlo sandbox
                    ├── strategy_agent  → pit window + compound
                    ├── predict_agent   → winner ensemble
                    └── rival_agent x N → competitor LLMs
                                ↓
                    Simulation Engine (OpenPitWall + numpy)
                            ├── tyre_deg_model  (.pkl)
                            ├── laptime_model   (.pkl / .pt)
                            ├── pit_rl_agent    (PPO .zip)
                            └── winner_ensemble (.pkl)
                                ↓
                    Data Layer
                            ├── FastF1 (local cache)
                            ├── OpenF1 (REST API)
                            └── OpenPitWall (training corpus)
```

---

## 3. Repository Structure

```
hannah-smith/
│
├── hannah.py                          # CLI entrypoint
├── pyproject.toml
├── .env.example                       # env var template
├── config.yaml                        # model + sim config
├── requirements.txt
├── AGENTS.md
│
├── tests/
│
└── hannah/
    ├── __init__.py
    ├── docs/
    │   ├── GOAL.md
    │   ├── PRD.md
    │   ├── ARCHITECTURE.md
    │   ├── AGENT_LOOP.md
    │   └── roadmap.md
    ├── agent/
    │   ├── loop.py                    # LiteLLM tool-using loop
    │   ├── persona.py                 # Hannah system prompt
    │   ├── memory.py                  # SQLite memory backend
    │   ├── subagents.py               # async rival / analyst helpers
    │   └── tool_registry.py           # tool discovery + validation
    ├── cli/
    │   ├── app.py                     # Click entrypoint
    │   ├── chat.py                    # nanobot-style TUI shell
    │   └── provider_ui.py             # provider onboarding/status helpers
    ├── config/
    │   ├── loader.py
    │   ├── provider_setup.py
    │   └── schema.py
    ├── data/
    │   ├── fastf1_loader.py           # FastF1 session fetcher
    │   ├── openf1_client.py           # OpenF1 REST wrapper
    │   ├── openpitwall_loader.py      # OpenPitWall data loader
    │   └── preprocess.py              # feature engineering
    ├── domain/
    │   ├── commands.py
    │   ├── race_state.py
    │   ├── teams.py
    │   └── tracks.py
    ├── models/
    │   ├── evaluate.py
    │   ├── train_tyre_deg.py
    │   ├── train_laptime.py
    │   ├── train_pit_rl.py
    │   ├── train_pit_q.py
    │   └── train_winner.py
    ├── providers/
    │   ├── litellm_provider.py
    │   ├── local_fallback.py
    │   └── registry.py
    ├── rlm/
    │   ├── helper.py
    │   ├── model.py
    │   ├── server.py
    │   └── train.py
    ├── session/
    │   └── manager.py
    ├── simulation/
    │   ├── monte_carlo.py
    │   ├── replay_trace.py
    │   ├── sandbox.py
    │   ├── strategy_engine.py
    │   └── tyre_model.py
    ├── tools/
    │   ├── race_data/tool.py
    │   ├── race_sim/tool.py
    │   ├── pit_strategy/tool.py
    │   ├── predict_winner/tool.py
    │   └── train_model/tool.py
    └── utils/
        └── console.py
```

---

## 4. Tech Stack

| Layer | Technology | Reason |
|---|---|---|
| CLI | Python `click` | clean command interface, streaming support |
| Agent core | NanoChat pattern | minimal, readable, fork-friendly |
| LLM gateway | `litellm` | unified API for Claude, OpenAI, local RLM |
| Async | `asyncio` | concurrent sub-agents + streaming output |
| F1 data (historical) | `fastf1` | official telemetry, tyre, weather data |
| F1 data (live/REST) | `openf1` REST API | live session data, no auth required |
| Training data | OpenPitWall | curated F1 strategy + pit stop dataset |
| ML models | `scikit-learn`, `xgboost` | tyre deg, winner ensemble |
| RL pit agent | `stable-baselines3` (PPO) | proven RL for pit stop timing |
| Lap time model | `pytorch` LSTM | sequential lap-by-lap prediction |
| Fast simulation | `numpy` vectorised | 1000-world Monte Carlo in <2s |
| Session memory | `sqlite3` | lightweight local persistence |
| RLM server (v2) | `fastapi` + `uvicorn` | OpenAI-compatible endpoint |
| Config | `pydantic` + `yaml` | typed config, easy override |

---

## 5. Key Modules — Detailed Spec

### 5.1 `agent/core.py` — LiteLLM Agent Loop

```python
# Pattern: NanoChat-style tool-use loop
# - load Hannah persona + session memory
# - dispatch user command to LiteLLM
# - handle tool_call responses → call SKILL tools
# - loop until no more tool calls
# - stream final response to CLI

HANNAH_MODEL = os.getenv("HANNAH_MODEL", "claude-sonnet-4-6")

async def run(messages, tools):
    while True:
        response = await litellm.acompletion(
            model=HANNAH_MODEL,
            messages=messages,
            tools=tools,
            stream=True,
        )
        # handle tool calls or final text
```

### 5.2 `agent/sub_agents.py` — Async Sub-Agent Spawner

```python
# Spawn all sub-agents concurrently
# Each sub-agent is a mini NanoChat loop with its own:
#   - system prompt (e.g. "You are the McLaren strategist")
#   - tools subset
#   - context window

async def spawn_all(race_ctx: RaceContext) -> dict:
    results = await asyncio.gather(
        SimAgent(race_ctx).run(),
        StrategyAgent(race_ctx).run(),
        PredictAgent(race_ctx).run(),
        *[RivalAgent(race_ctx, team=t).run() for t in race_ctx.rivals],
        return_exceptions=True,
    )
    return parse_results(results)
```

### 5.3 `simulation/monte_carlo.py` — Fast Prediction Engine

```python
# Prediction mode: no real-time rendering
# Vectorised over N worlds using numpy broadcasting
# Returns probability distributions, not a single timeline

async def run_fast(race_state: RaceState, n_worlds: int = 1000) -> SimResult:
    # shape: (n_worlds, n_laps, n_drivers)
    lap_times = tyre_model.predict_batch(race_state, n_worlds)
    pit_laps  = pit_agent.find_windows_batch(lap_times)
    outcomes  = gap_engine.compute_deltas(lap_times, pit_laps)
    return SimResult(
        winner_probs=softmax(outcomes.final_positions),
        optimal_pit_windows=pit_laps.mode(axis=0),
        p50_race_time=np.percentile(outcomes.total_time, 50),
    )
```

### 5.4 `rlm/server.py` — v2 OpenAI-Compatible Endpoint

```python
# FastAPI server — OpenAI-compatible
# LiteLLM routes here when HANNAH_MODEL=openai/rlm-local
# Zero changes needed in agent/core.py

@app.post("/v1/chat/completions")
async def chat(request: ChatRequest):
    response = rlm_model.generate(
        messages=request.messages,
        tools=request.tools,
    )
    return openai_format(response)
```

### 5.5 `.env` — v1 → v2 Migration

```bash
# v1 — uses Claude via LiteLLM
HANNAH_MODEL=claude-sonnet-4-6
ANTHROPIC_API_KEY=sk-ant-...

# v2 — swap to local RLM, zero other changes
# HANNAH_MODEL=openai/rlm-local
# HANNAH_RLM_API_BASE=http://localhost:8000
# HANNAH_RLM_API_KEY=none
```

---

## 6. CLI Command Reference

```bash
# Data commands
hannah --fetch --race bahrain --year 2025
hannah --fetch --live                          # current session

# Simulation commands
hannah --simulate --race bahrain --driver VER
hannah --simulate --race bahrain --driver VER --laps 57 --weather wet
hannah --sandbox --agents VER,NOR,LEC --race monaco

# Prediction commands  
hannah --predict --race singapore --year 2025
hannah --predict --race bahrain --strategy undercut

# Strategy commands
hannah --strategy --race bahrain --lap 22 --driver VER
hannah --ask "Should we pit under VSC on lap 34?"

# Training commands
hannah --train tyre_model --year 2024
hannah --train all --years 2022,2023,2024
hannah --eval tyre_model                       # evaluate + print metrics

# Agent commands
hannah --tools                                 # list registered tools
hannah --agents                                # list sub-agents + status
hannah --model                                 # show current LiteLLM model

# RLM (v2)
hannah --rlm start                             # start RLM server
hannah --rlm status                            # check server health
```

---

## 7. SKILL.md Tool Spec — Example

Every tool in `tools/` follows the NanoChat SKILL.md pattern:

```markdown
# SKILL: race_data

## Description
Fetches F1 race data from FastF1 (historical telemetry) and OpenF1 (live/REST).
Returns lap times, tyre compounds, sector splits, weather, and car positions.

## When to use
Use this tool whenever Hannah needs current or historical race data before
making a strategy call or running a simulation.

## Inputs
- race: str — race name e.g. "bahrain", "monaco"  
- year: int — season year, default current year
- session: str — "R" (race), "Q" (qualifying), "FP1/2/3"
- driver: str — driver code e.g. "VER", "NOR" (optional)

## Outputs
Returns a RaceData object with:
- lap_times: DataFrame
- tyre_stints: DataFrame  
- weather: DataFrame
- positions: DataFrame

## Example
race_data(race="bahrain", year=2025, session="R", driver="VER")
```

---

## 8. Data Flow — Simulation Request

```
User: hannah --simulate --race bahrain --driver VER

1. hannah.py           parse args → build RaceContext
2. agent/core.py       send to LiteLLM (Claude / RLM)
3. Hannah thinks       decides to call: race_data + race_sim tools
4. tool_registry       dispatches race_data tool
5. fastf1_loader       fetches 2025 Bahrain telemetry → cache
6. sub_agents.py       asyncio.gather spawns:
                         sim_agent → monte_carlo.run_fast(n=1000)
                         strategy_agent → strategy_engine.analyse()
                         predict_agent  → winner_ensemble.predict()
7. results stream      each agent streams output as it completes
8. agent/core.py       Hannah synthesises all results
9. CLI output          colour-coded strategy report streamed to terminal
```

---

## 9. Build Phases

### Phase 1 — Foundation (kickstart)
- [ ] `hannah.py` — click CLI entrypoint
- [ ] `agent/core.py` — LiteLLM loop with tool dispatch
- [ ] `agent/persona.py` — Hannah system prompt
- [ ] `agent/memory.py` — SQLite session memory
- [ ] `agent/tool_registry.py` — SKILL.md loader
- [ ] `tools/race_data/` — FastF1 + OpenF1 tool
- [ ] `.env.example` + `config.yaml`
- [ ] `requirements.txt`

### Phase 2 — Simulation Engine
- [ ] `simulation/monte_carlo.py` — vectorised fast sim
- [ ] `simulation/tyre_model.py` — compound deg physics
- [ ] `simulation/gap_engine.py` — undercut/overcut calc
- [ ] `tools/race_sim/` — sim SKILL tool
- [ ] `tools/predict_winner/` — prediction SKILL tool

### Phase 3 — ML Models + Training CLI
- [ ] `models/train_tyre_deg.py`
- [ ] `models/train_laptime.py`
- [ ] `models/train_pit_rl.py` — PPO agent
- [ ] `models/train_winner.py`
- [ ] `tools/train_model/` — training SKILL tool
- [ ] `data/openpiwall_loader.py`

### Phase 4 — Sub-Agent Async Scaling
- [ ] `agent/sub_agents.py` — async spawner
- [ ] `simulation/competitor_agents.py` — rival LLM agents
- [ ] Streaming output per agent completion

### Phase 5 — RLM Scaffold (v2 prep)
- [ ] `rlm/server.py` — FastAPI stub
- [ ] `rlm/model.py` — weights loader stub
- [ ] LiteLLM routing test: Claude → RLM local
- [ ] `.env` swap validation

---

## 10. Dependencies — requirements.txt

```
# CLI
click>=8.1

# Agent + LLM
litellm>=1.40
anthropic>=0.25

# F1 Data
fastf1>=3.3
requests>=2.31        # openf1 REST

# Simulation + ML
numpy>=1.26
pandas>=2.1
scikit-learn>=1.4
xgboost>=2.0
torch>=2.2
stable-baselines3>=2.2
gymnasium>=0.29

# RLM server (v2)
fastapi>=0.110
uvicorn>=0.29
pydantic>=2.6

# Storage
# sqlite3 is stdlib — no install needed

# Config
pyyaml>=6.0
python-dotenv>=1.0
```

---

## 11. Hannah System Prompt

```
You are Hannah Smith, Red Bull Racing's virtual Race Director.

You have full access to real F1 telemetry, race simulations, tyre models,
and competitor strategy analysis via your registered tools.

Your decision-making style:
- Data first. Every call is backed by numbers from your tools.
- Direct and decisive. On the pit wall, hesitation costs positions.
- Think in gaps, tyre windows, and stint lengths — not laps alone.
- When uncertain, run the simulation before making a call.
- You manage rival agents — you see what McLaren, Ferrari and Mercedes are thinking.

When asked a strategy question:
1. Call race_data to get current state
2. Run race_sim to model the outcome
3. Check competitor tyre states
4. Give a clear recommendation with the data behind it

You are concise. No filler. Race directors don't have time for it.
```

---

## 12. Configuration — config.yaml

```yaml
agent:
  model: ${HANNAH_MODEL:-claude-sonnet-4-6}
  temperature: 0.2
  max_tokens: 2048
  stream: true

simulation:
  n_worlds: 1000          # Monte Carlo worlds
  prediction_mode: true   # fast async, no real-time rendering
  async: true

fastf1:
  cache_dir: data/fastf1_cache
  timeout: 30

openf1:
  base_url: https://api.openf1.org/v1
  timeout: 10

models:
  tyre_deg: models/saved/tyre_deg_v1.pkl
  laptime: models/saved/laptime_v1.pkl
  pit_rl: models/saved/pit_rl_v1.zip
  winner: models/saved/winner_ensemble_v1.pkl

rlm:
  enabled: false          # flip to true for v2
  api_base: http://localhost:8000
  api_key: none
```

---

*Generated for Hannah Smith v0.1.0-alpha — Red Bull Race Director*
*Built on NanoChat + LiteLLM + FastF1 + OpenPitWall*
