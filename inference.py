import os
import sys
import json

# Make all project modules importable from repo root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.disaster_env  import DisasterEnv, FOOD, MEDICAL, RESCUE
from agents.rule_based import RuleBasedAgent

TASKS = [
    {"id": "easy",   "difficulty": "easy",   "max_steps": 20, "threshold": 0.3},
    {"id": "medium", "difficulty": "medium", "max_steps": 25, "threshold": 0.3},
    {"id": "hard",   "difficulty": "hard",   "max_steps": 30, "threshold": 0.15},
]

RESOURCE_NAMES = {0: "food", 1: "medical", 2: "rescue"}
QTY_MAP        = {0: 10,    1: 20,        2: 30}


# ── Agent selection ────────────────────────────────────────────────────
# Uses LLM agent if API_KEY is set, falls back to rule-based automatically.

def _get_agent():
    """Return the best available agent."""
    try:
        from agents.llm_agent import LLMAgent
        agent = LLMAgent()
        if agent.using_llm:
            print(f"[INFO] LLM agent active — model={agent._model}", flush=True)
            return agent
    except Exception:
        pass

    print("[INFO] No API key found — using rule-based agent (fallback)", flush=True)
    return RuleBasedAgent()


# ── Single task runner ─────────────────────────────────────────────────

def run_task(task: dict, agent):
    """Run one full episode and print evaluator-compatible log lines."""
    env = DisasterEnv(task["difficulty"])
    env.reset()

    # Determine model name for the START line
    model_name = getattr(agent, "_model", None) or agent.name

    print(f"[START] task={task['id']} env=disaster model={model_name}", flush=True)

    rewards    = []
    step_count = 0
    done       = False

    # Reset per-episode state if agent supports it
    if hasattr(agent, "reset"):
        agent.reset()

    while not done:
        # Get action — works for both BaseAgent (.act) and plain callables
        if hasattr(agent, "act"):
            action = agent.act(env)
        else:
            action = agent(env)

        _, reward, done, _ = env.step(action)
        rewards.append(reward)
        step_count += 1

        res_name   = RESOURCE_NAMES[action[0]]
        action_str = f"{res_name}-{action[1]}-{QTY_MAP[action[2]]}"

        print(
            f"[STEP] step={step_count} action={action_str} "
            f"reward={reward:.2f} done={str(done).lower()} error=null",
            flush=True,
        )

    total = sum(rewards)
    score = round(min(1.0, max(0.0, total / (task["max_steps"] * 0.6))), 3)
    ok    = score >= task["threshold"]

    print(
        f"[END] success={str(ok).lower()} steps={step_count} "
        f"score={score:.3f} rewards={','.join(f'{r:.2f}' for r in rewards)}",
        flush=True,
    )

    return {"task_id": task["id"], "score": score, "success": ok, "steps": step_count}


# ── Run all 3 tasks ────────────────────────────────────────────────────

def run_all_tasks():
    import random
    random.seed(42)   # fixed seed — reproducible scores every run

    agent   = _get_agent()
    results = []

    for task in TASKS:
        try:
            result = run_task(task, agent)
            results.append(result)
        except Exception as e:
            print(f"[ERROR] task={task['id']} error={e}", flush=True)

    return results


if __name__ == "__main__":
    run_all_tasks()
