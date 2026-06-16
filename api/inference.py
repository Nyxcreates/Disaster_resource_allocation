"""
api/inference.py — LLM agent inference for OpenEnv evaluation.

This is what the Scaler evaluator calls.
It runs all three tasks (easy / medium / hard) using the LLMAgent,
printing structured log lines that the evaluator parses.

Log format (do not change — evaluator depends on it):
    [START] task=<id> env=disaster model=<name>
    [STEP]  step=N action=<res>-<zone>-<qty> reward=R.RR done=true/false error=null
    [END]   success=true/false steps=N score=S.SSS rewards=R1,R2,...
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.disaster_env import DisasterEnv
from agents.llm_agent  import LLMAgent

TASKS = [
    {"id": "easy",   "difficulty": "easy",   "max_steps": 20, "threshold": 0.5},
    {"id": "medium", "difficulty": "medium", "max_steps": 25, "threshold": 0.3},
    {"id": "hard",   "difficulty": "hard",   "max_steps": 30, "threshold": 0.15},
]

RESOURCE_NAMES = {0: "food", 1: "medical", 2: "rescue"}
QTY_MAP        = {0: 10, 1: 20, 2: 30}


def run_task(task: dict, agent: LLMAgent):
    env = DisasterEnv(task["difficulty"])
    env.reset()

    model_name = agent._model or "fallback"
    print(f"[START] task={task['id']} env=disaster model={model_name}", flush=True)

    rewards, step_count, done = [], 0, False

    while not done:
        action              = agent.act(env)
        _, reward, done, _  = env.step(action)
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
    score = min(1.0, max(0.0, total / (task["max_steps"] * 0.6)))
    ok    = score >= task["threshold"]

    print(
        f"[END] success={str(ok).lower()} steps={step_count} "
        f"score={score:.3f} rewards={','.join(f'{r:.2f}' for r in rewards)}",
        flush=True,
    )


def run_all_tasks():
    agent = LLMAgent()
    for task in TASKS:
        try:
            agent.reset()
            run_task(task, agent)
        except Exception as e:
            print(f"[ERROR] task={task['id']} error={e}", flush=True)


if __name__ == "__main__":
    run_all_tasks()
