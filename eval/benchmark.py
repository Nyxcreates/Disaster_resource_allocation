"""
eval/benchmark.py — Rigorous statistical benchmark across all agents and difficulties.

Runs N episodes per agent per difficulty, computes mean ± std and
95% confidence intervals, and prints/saves a professional results table.

Usage:
    python eval/benchmark.py                        # 50 episodes, all agents
    python eval/benchmark.py --episodes 100        # more episodes
    python eval/benchmark.py --ppo checkpoints/disaster_agent_medium
"""

import argparse
import json
import math
import os
import random
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.disaster_env  import DisasterEnv
from agents.random_agent import RandomAgent
from agents.rule_based   import RuleBasedAgent
from agents.llm_agent    import LLMAgent


DIFFICULTIES = ["easy", "medium", "hard"]
SEED         = 2024


# ── Statistics helpers ─────────────────────────────────────────────────

def mean(vals):
    return sum(vals) / len(vals)

def std(vals):
    if len(vals) < 2:
        return 0.0
    m = mean(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))

def ci95(vals):
    """95% confidence interval half-width (t ≈ 2 for n ≥ 30)."""
    if len(vals) < 2:
        return 0.0
    return 2.0 * std(vals) / math.sqrt(len(vals))


# ── Episode runner ─────────────────────────────────────────────────────

def run_episodes(agent, difficulty: str, n: int, seed: int) -> dict:
    """Run n episodes and return summary statistics."""
    random.seed(seed)
    rewards, steps, cleared = [], [], []

    if hasattr(agent, "reset"):
        agent.reset()

    for _ in range(n):
        env  = DisasterEnv(difficulty)
        env.reset()
        done, ep_r = False, 0.0

        while not done:
            action = agent.act(env) if hasattr(agent, "act") else agent(env)
            _, r, done, _ = env.step(action)
            ep_r += r

        rewards.append(ep_r)
        steps.append(env.step_count)
        cleared.append(sum(1 for z in env.zones if z.injured == 0 and z.food_need == 0))

    return {
        "mean":    round(mean(rewards), 4),
        "std":     round(std(rewards),  4),
        "ci95":    round(ci95(rewards), 4),
        "min":     round(min(rewards),  4),
        "max":     round(max(rewards),  4),
        "avg_steps":   round(mean(steps),   1),
        "avg_cleared": round(mean(cleared), 2),
        "n":       n,
    }


# ── Main benchmark ─────────────────────────────────────────────────────

def run_benchmark(n_episodes: int = 50, ppo_path: str = None) -> dict:
    # Build agent list
    agents = [
        ("Random",     RandomAgent()),
        ("Rule-based", RuleBasedAgent()),
        ("LLM",        LLMAgent()),
    ]

    if ppo_path and os.path.exists(ppo_path + ".zip"):
        try:
            from agents.ppo_agent import PPOAgent
            ppo_diff = _detect_ppo_difficulty(ppo_path)
            agents.append(("PPO", PPOAgent(ppo_path, ppo_diff)))
            print(f"  Loaded PPO model: {ppo_path} (difficulty={ppo_diff})")
        except Exception as e:
            print(f"  PPO load failed: {e}")

    results = {}

    print(f"\n{'='*72}")
    print(f"  BENCHMARK  —  {n_episodes} episodes per cell  |  seed={SEED}")
    print(f"{'='*72}")

    # Header
    col_w = 22
    header = f"  {'Agent':<16}"
    for diff in DIFFICULTIES:
        header += f"  {diff.upper():^{col_w}}"
    print(header)
    print(f"  {'─'*16}" + (f"  {'─'*col_w}" * 3))

    for agent_name, agent in agents:
        results[agent_name] = {}
        row = f"  {agent_name:<16}"

        for diff in DIFFICULTIES:
            print(f"    Running {agent_name} on {diff}...", end="\r", flush=True)
            stats = run_episodes(agent, diff, n_episodes, SEED)
            results[agent_name][diff] = stats

            cell = f"{stats['mean']:.3f} ± {stats['ci95']:.3f}"
            row += f"  {cell:^{col_w}}"

        print(row + "  " * 10)   # clear \r remnant

    print(f"\n  Format: mean ± 95% CI  (n={n_episodes})")
    print(f"{'='*72}\n")

    # Detailed breakdown
    print(_detailed_table(results, n_episodes))

    return results


def _detailed_table(results: dict, n: int) -> str:
    lines = [f"  DETAILED RESULTS (n={n} episodes each)", "  " + "─" * 68]
    for agent_name, by_diff in results.items():
        lines.append(f"\n  {agent_name}")
        for diff, s in by_diff.items():
            lines.append(
                f"    {diff:<8} mean={s['mean']:6.3f}  std={s['std']:5.3f}  "
                f"ci95=±{s['ci95']:5.3f}  min={s['min']:5.3f}  max={s['max']:5.3f}  "
                f"steps={s['avg_steps']:4.1f}  cleared={s['avg_cleared']:.1f}"
            )
    return "\n".join(lines)


def _detect_ppo_difficulty(model_path: str) -> str:
    try:
        from stable_baselines3 import PPO
        model = PPO.load(model_path)
        size  = model.observation_space.shape[0]
        # v2 obs: 8 values/zone + 3 global
        mapping = {27: "easy", 43: "medium", 67: "hard"}
        return mapping.get(size, "medium")
    except Exception:
        return "medium"


def save_results(results: dict, path: str = "benchmark_results.json"):
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved → {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--ppo",      type=str, default=None,
                        help="Path to PPO model (without .zip)")
    parser.add_argument("--save",     type=str, default="benchmark_results.json")
    args = parser.parse_args()

    results = run_benchmark(n_episodes=args.episodes, ppo_path=args.ppo)
    save_results(results, args.save)
