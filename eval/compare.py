"""
eval/compare.py — Compare agents side-by-side and save a publication-quality chart.

Usage:
    python eval/compare.py                         # all agents, 20 episodes
    python eval/compare.py --episodes 50 --ppo checkpoints/disaster_agent_medium
"""

import argparse
import os
import random
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    print("pip install matplotlib numpy")
    sys.exit(1)

from env.disaster_env    import DisasterEnv
from agents.random_agent  import RandomAgent
from agents.rule_based    import RuleBasedAgent
from agents.llm_agent     import LLMAgent

DIFFICULTIES = ["easy", "medium", "hard"]
SEED         = 42


def _run(agent, difficulty: str, n: int) -> list:
    random.seed(SEED)
    rewards = []
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
    return rewards


def collect(n_episodes: int, ppo_path: str = None) -> dict:
    agents = [
        ("Random",     RandomAgent()),
        ("Rule-based", RuleBasedAgent()),
        ("LLM",        LLMAgent()),
    ]

    if ppo_path and os.path.exists(ppo_path + ".zip"):
        try:
            from agents.ppo_agent import PPOAgent
            diff = _detect_diff(ppo_path)
            agents.append(("PPO (trained)", PPOAgent(ppo_path, diff)))
        except Exception as e:
            print(f"  PPO skip: {e}")

    results = {name: {} for name, _ in agents}

    for name, agent in agents:
        for diff in DIFFICULTIES:
            print(f"  {name:<16} {diff}... ", end="", flush=True)
            r = _run(agent, diff, n_episodes)
            results[name][diff] = r
            print(f"avg={sum(r)/len(r):.3f}")

    return results


def plot(results: dict, save_path: str = "comparison_chart.png"):
    agent_names = list(results.keys())
    colors      = ["#B4B2A9", "#5DCAA5", "#7F77DD", "#EF9F27"][:len(agent_names)]
    bar_w       = 0.8 / len(agent_names)
    x           = np.arange(len(DIFFICULTIES))

    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor("#FAFAF8")
    ax.set_facecolor("#FAFAF8")

    for i, (name, color) in enumerate(zip(agent_names, colors)):
        means, errs, positions = [], [], []
        for j, diff in enumerate(DIFFICULTIES):
            vals = results[name].get(diff, [])
            if vals:
                m = sum(vals) / len(vals)
                s = (sum((v - m) ** 2 for v in vals) / max(1, len(vals) - 1)) ** 0.5
                means.append(m)
                errs.append(s)
                positions.append(x[j] + (i - len(agent_names) / 2 + 0.5) * bar_w)

        if not means:
            continue

        bars = ax.bar(
            positions, means, bar_w, label=name, color=color,
            yerr=errs, capsize=4, error_kw=dict(elinewidth=1, ecolor="#888780"),
            zorder=3,
        )
        for bar, m in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.008,
                f"{m:.2f}", ha="center", va="bottom",
                fontsize=8.5, color="#444441",
            )

    ax.set_xlabel("Difficulty", fontsize=12, color="#444441", labelpad=8)
    ax.set_ylabel("Avg total reward / episode", fontsize=12, color="#444441", labelpad=8)
    ax.set_title(
        "Agent comparison — Random vs Rule-based vs LLM vs PPO",
        fontsize=13, color="#2C2C2A", pad=14, fontweight="normal",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d in DIFFICULTIES], fontsize=11, color="#444441")
    ax.tick_params(axis="y", labelcolor="#888780", labelsize=10)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#D3D1C7")
    ax.spines["bottom"].set_color("#D3D1C7")
    ax.yaxis.grid(True, color="#E8E6DF", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=10, framealpha=0, labelcolor="#444441")

    plt.tight_layout()
    full_path = os.path.join(os.path.dirname(__file__), "..", save_path)
    plt.savefig(full_path, dpi=150, bbox_inches="tight")
    print(f"\n  Chart saved → {full_path}")
    return full_path


def _detect_diff(model_path: str) -> str:
    try:
        from stable_baselines3 import PPO
        model = PPO.load(model_path)
        size  = model.observation_space.shape[0]
        return {27: "easy", 43: "medium", 67: "hard"}.get(size, "medium")
    except Exception:
        return "medium"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--ppo",      type=str, default=None)
    parser.add_argument("--out",      type=str, default="comparison_chart.png")
    args = parser.parse_args()

    print(f"\n{'='*58}")
    print(f"  Agent comparison  ({args.episodes} episodes each)")
    print(f"{'='*58}\n")

    results = collect(args.episodes, args.ppo)
    plot(results, args.out)
    print("\n  Done.")
