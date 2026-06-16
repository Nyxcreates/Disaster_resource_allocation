---
title: Disaster Management
emoji: 🚨
colorFrom: red
colorTo: red
sdk: docker
app_port: 7860
pinned: false
---
👥 Team & Contribution

This project was developed as part of a **3-member team collaboration**.

### 💡 My Contribution
- Contributed to the design and development of the disaster response simulation system.
- Assisted in implementing and testing resource allocation strategies within simulated emergency scenarios.
- Participated in analyzing system outputs and evaluating response efficiency under different disaster conditions.
- Supported project documentation, deployment, and collaborative development activities.
- Worked closely with team members to improve system functionality and overall project performance.

# Disaster Resource Allocation — OpenEnv Submission v2

> **Real-world task:** AI-driven emergency resource allocation across multiple disaster zones under time pressure, escalating crises, and random events.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![OpenEnv Compliant](https://img.shields.io/badge/OpenEnv-2.0-green)](https://openenv.ai)
[![HuggingFace Space](https://img.shields.io/badge/🤗-Space-yellow)](https://huggingface.co/spaces)

---

## What this environment simulates

A multi-zone disaster response coordination system. An AI agent acts as an incident commander deciding how to distribute limited emergency supplies (food, medical kits, rescue teams) across several disaster-hit zones each round. Every decision has real consequences:

- Send food to Zone C → disease risk drops, morale recovers
- Ignore a blocked zone → rescue teams cannot enter, casualties mount
- Waste medical kits on a cleared zone → penalty and wasted stock
- Wait too long on an earthquake zone → 10% more injured each round (escalation)

The simulation includes **random mid-episode events** (aftershocks, supply airdrops, road clearances) that force genuine adaptive decision-making.

---

## Environment description

### Observation space
Per zone (8 values, normalised 0–1):

| Field | Description |
|-------|-------------|
| `injured` | People needing medical aid |
| `food_need` | Food units still required |
| `severity` | Overall zone severity (0=minor, 1=catastrophic) |
| `rescue_blocked` | Whether rescue teams can enter |
| `disaster_type` | Encoded: earthquake/flood/hurricane/wildfire |
| `morale` | Community resilience (drops when unhelped) |
| `disease_risk` | Outbreak probability (grows if food unmet) |
| `infra_damage` | Infrastructure damage (reduces resource efficiency) |

Plus 3 global values: `food_stock`, `med_stock`, `resc_stock`

**Total obs size:** `num_zones × 8 + 3` → Easy=27, Medium=43, Hard=67

### Action space
`Discrete(3 × num_zones × 3)` — decoded as:
- **resource_type**: 0=food, 1=medical, 2=rescue
- **zone_id**: 0 to num_zones−1
- **quantity_index**: 0=10 units, 1=20 units, 2=30 units

### Reward function (5 components)

| Component | Weight | Description |
|-----------|--------|-------------|
| Coverage score | 40% | Fraction of total need already met |
| Triage score | 25% | Did we help the highest-severity zone? |
| Rescue priority | 15% | Bonus for unblocking blocked zones |
| Efficiency score | 10% | Right resource type for the actual need |
| Morale bonus | 5% | Maintaining community resilience |
| Waste penalty | −0–15% | Wrong resource sent to wrong zone |

### Difficulty levels

| Level | Zones | Escalation | Events | Max steps |
|-------|-------|-----------|--------|-----------|
| Easy | 3 | 0% | None | 20 |
| Medium | 5 | 5%/step | Moderate | 25 |
| Hard | 8 | 10%/step | Frequent | 30 |

---

## Quick start

```bash
# 1. Clone / download
git clone https://github.com/YOUR_USERNAME/disaster-resource-allocation
cd disaster-resource-allocation

# 2. Install
pip install -r requirements.txt

# 3. Verify everything works
python scripts/quick_demo.py

# 4. Start the API server
python main.py
# Open http://localhost:7860/docs  ← interactive API docs
```

---

## All commands

```bash
# API server (default — what OpenEnv evaluator calls)
python main.py

# Run LLM agent inference on all 3 tasks
python main.py --mode inference

# Train PPO with curriculum learning (easy → medium → hard)
python main.py --mode train --curriculum --steps 500000

# Train single difficulty
python main.py --mode train --difficulty medium --steps 100000

# Benchmark all agents (50 episodes each, with confidence intervals)
python main.py --mode benchmark --episodes 50

# Generate comparison chart
python main.py --mode compare --episodes 20

# Run test suite
python main.py --mode test
```

---

## Project structure

```
disaster-resource-allocation/
├── env/                     # Core simulation (OpenEnv API)
│   ├── disaster_env.py      # Main env: reset() / step() / state()
│   ├── zone.py              # Zone with morale, disease_risk, infra_damage
│   ├── config.py            # Difficulty configs (easy/medium/hard)
│   ├── rewards.py           # 5-component reward module
│   ├── events.py            # Random event generator
│   ├── gym_wrapper.py       # Gymnasium wrapper for PPO training
│   └── models.py            # Typed data models (Pydantic + dataclass)
├── agents/                  # All agent implementations
│   ├── base.py              # Abstract BaseAgent interface
│   ├── random_agent.py      # Random baseline
│   ├── rule_based.py        # Triage heuristic agent
│   ├── llm_agent.py         # Chain-of-thought LLM agent
│   └── ppo_agent.py         # Trained PPO model wrapper
├── training/                # RL training pipeline
│   ├── train.py             # Single-difficulty training with CLI
│   ├── curriculum.py        # Curriculum: easy → medium → hard
│   └── callbacks.py         # Progress + best-model-save callbacks
├── eval/                    # Evaluation suite
│   ├── graders.py           # OpenEnv graders (easy/medium/hard)
│   ├── benchmark.py         # Statistical benchmark (mean ± 95% CI)
│   └── compare.py           # Multi-agent comparison chart
├── api/                     # FastAPI REST server
│   ├── server.py            # All endpoints + startup inference
│   └── inference.py         # LLM agent inference runner
├── tests/
│   └── test_env.py          # 10 automated tests
├── scripts/
│   └── quick_demo.py        # First-run verification script
├── main.py                  # Single entry point (--mode flag)
├── openenv.yaml             # OpenEnv specification
├── Dockerfile               # HuggingFace Spaces deployment
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## API reference

| Method | Path | Body | Description |
|--------|------|------|-------------|
| `GET` | `/` | — | Health check |
| `POST` | `/reset` | `{"difficulty": "medium"}` | Start new episode |
| `POST` | `/step` | `{"resource_type":1,"zone_id":2,"quantity_index":2}` | Take one action |
| `GET` | `/state` | — | Current world state |
| `POST` | `/grade` | `{"task_id":"medium","n_episodes":5,"agent":"rule_based"}` | Run grader |
| `GET` | `/benchmark` | — | Quick 10-episode benchmark |

Full interactive docs at `/docs` (Swagger UI).

---

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `API_KEY` | — | OpenAI / Scaler API key for LLM agent |
| `API_BASE_URL` | OpenAI | Override API base URL |
| `MODEL_NAME` | `gpt-4o-mini` | LLM model name |
| `PORT` | `7860` | Server port |

---

## Agents

### LLM Agent (primary)
Uses chain-of-thought reasoning with a structured observation table. The system prompt enforces triage priorities: unblock first → disease risk → most injured → most hungry. Falls back to the rule-based agent if no API key is set.

### Rule-based Agent (baseline)
Deterministic triage logic: rescue blocked zones → treat disease-risk zones with food → help most injured with medical → help most hungry with food. Quantity adapts based on remaining stock.

### PPO Agent (trained)
Proximal Policy Optimization trained via curriculum learning. Starts on easy, auto-promotes to medium then hard when average reward exceeds thresholds.

---

## Reproducible baseline scores

Rule-based agent, 50 episodes, seed=42:

| Difficulty | Avg reward | Score | Status |
|------------|-----------|-------|--------|
| Easy | 7.13 ± 0.62 | 0.594 | PASS (≥0.5) |
| Medium | 6.92 ± 0.30 | 0.461 | PASS (≥0.3) |
| Hard | 9.73 ± 0.18 | 0.541 | PASS (≥0.15) |

---

## Setup for development

```bash
pip install -r requirements.txt

# With optional RL training support
pip install stable-baselines3 gymnasium

# Run tests
python tests/test_env.py

# Run quick demo (no API key needed)
python scripts/quick_demo.py
```

---

## Deploying to Hugging Face Spaces

1. Create a new Space (Docker SDK)
2. Push this repo
3. Add secrets: `API_KEY`, `MODEL_NAME`
4. The Space runs `python main.py` via the Dockerfile

The `/` health endpoint is used as the health check.
