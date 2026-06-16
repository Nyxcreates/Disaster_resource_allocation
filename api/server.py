"""
api/server.py — FastAPI server exposing the DisasterEnv as a REST API.

Endpoints:
    GET  /            — health check + version
    POST /reset       — start a new episode
    POST /step        — take one action
    GET  /state       — current world state
    POST /grade       — run a task grader
    GET  /benchmark   — quick 10-episode benchmark
    GET  /docs        — auto-generated OpenAPI docs (FastAPI built-in)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import uvicorn

from env.disaster_env    import DisasterEnv
from eval.graders        import EasyGrader, MediumGrader, HardGrader
from agents.rule_based   import RuleBasedAgent
from agents.random_agent import RandomAgent

app = FastAPI(
    title       = "Disaster Resource Allocation — OpenEnv API",
    description = (
        "Multi-zone disaster relief simulation. "
        "Allocate food, medical kits and rescue teams across disaster zones. "
        "OpenEnv 2.0 compliant."
    ),
    version     = "2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

_env: Optional[DisasterEnv] = None


# ── Pydantic request schemas ───────────────────────────────────────────

class ResetRequest(BaseModel):
    difficulty: str = "medium"

class StepRequest(BaseModel):
    resource_type:  Optional[int] = None
    zone_id:        Optional[int] = None
    quantity_index: Optional[int] = None
    action:         Optional[int] = None   # flat integer (Gym Discrete) alternative

class GradeRequest(BaseModel):
    task_id:    str = "medium"
    n_episodes: int = 5
    agent:      str = "rule_based"   # "rule_based" | "random"


# ── Helpers ────────────────────────────────────────────────────────────

def _decode_action(req: StepRequest, env: DisasterEnv) -> tuple:
    if req.action is not None:
        nz  = len(env.zones)
        res = req.action // (nz * 3)
        rem = req.action  % (nz * 3)
        return (res, rem // 3, rem % 3)
    return (req.resource_type or 0, req.zone_id or 0, req.quantity_index or 0)


# ── Startup: trigger inference in background thread ───────────────────

@app.on_event("startup")
def _startup():
    import threading
    def _run():
        try:
            from api.inference import run_all_tasks
            run_all_tasks()
        except Exception as e:
            print(f"[server] inference error: {e}", flush=True)
    threading.Thread(target=_run, daemon=True).start()


# ── Endpoints ──────────────────────────────────────────────────────────

@app.get("/")
def health():
    return {
        "status":  "ok",
        "version": "2.0.0",
        "env":     "disaster-resource-allocation",
        "docs":    "/docs",
    }


@app.post("/reset")
def reset(req: ResetRequest = ResetRequest()):
    global _env
    _env = DisasterEnv(req.difficulty)
    _env.reset()
    return _env.state()


@app.post("/step")
def step(req: StepRequest = StepRequest()):
    global _env
    if _env is None:
        _env = DisasterEnv("medium")
        _env.reset()
    action = _decode_action(req, _env)
    obs, reward, done, info = _env.step(action)
    return {"observation": obs, "reward": reward, "done": done, "info": info}


@app.get("/state")
def state():
    if _env is None:
        raise HTTPException(status_code=400, detail="Call /reset first")
    return _env.state()


@app.post("/grade")
def grade(req: GradeRequest = GradeRequest()):
    graders = {"easy": EasyGrader(), "medium": MediumGrader(), "hard": HardGrader()}
    grader  = graders.get(req.task_id, MediumGrader())
    agents  = {"rule_based": RuleBasedAgent(), "random": RandomAgent()}
    agent   = agents.get(req.agent, RuleBasedAgent())
    result  = grader.run(agent, req.n_episodes)
    return result.dict()


@app.get("/benchmark")
def benchmark():
    """Quick reproducible 10-episode benchmark, rule-based vs random."""
    import random as _r
    results = {}
    for diff in ["easy", "medium", "hard"]:
        results[diff] = {}
        for name, agent in [("random", RandomAgent()), ("rule_based", RuleBasedAgent())]:
            _r.seed(42)
            ep_rewards = []
            for _ in range(10):
                env = DisasterEnv(diff)
                env.reset()
                done, ep_r = False, 0.0
                while not done:
                    _, r, done, _ = env.step(agent.act(env))
                    ep_r += r
                ep_rewards.append(ep_r)
            results[diff][name] = round(sum(ep_rewards) / len(ep_rewards), 4)
    return results


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("api.server:app", host="0.0.0.0", port=port, reload=False)
