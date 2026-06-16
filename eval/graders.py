"""
eval/graders.py — Deterministic task graders for OpenEnv compliance.

Each grader runs N episodes with a fixed seed and returns a GraderResult.
Results are reproducible: same agent + same seed = same score every time.

Grader hierarchy:
    BaseGrader
    ├── EasyGrader   (task_id="easy",   threshold=0.5)
    ├── MediumGrader (task_id="medium", threshold=0.3)
    └── HardGrader   (task_id="hard",   threshold=0.15)
"""

import random
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.disaster_env import DisasterEnv, FOOD, MEDICAL, RESCUE
from env.models       import GraderResult


class BaseGrader:
    task_id           = "base"
    difficulty        = "easy"
    success_threshold = 0.5
    seed              = 42

    def run(self, agent, n_episodes: int = 10) -> GraderResult:
        """
        Run n_episodes with agent and return a GraderResult.

        agent: object with .act(env) -> (resource, zone, qty) method
               OR a plain callable  fn(env) -> tuple
        """
        random.seed(self.seed)

        all_rewards, all_steps, all_cleared = [], [], []
        last_env = None

        for _ in range(n_episodes):
            env  = DisasterEnv(self.difficulty)
            obs  = env.reset()
            done = False
            ep_r = 0.0

            if hasattr(agent, "reset"):
                agent.reset()

            while not done:
                if hasattr(agent, "act"):
                    action = agent.act(env)
                else:
                    action = agent(env)           # plain callable
                obs, reward, done, info = env.step(action)
                ep_r += reward

            all_rewards.append(ep_r)
            all_steps.append(env.step_count)
            cleared = sum(1 for z in env.zones if z.injured == 0 and z.food_need == 0)
            all_cleared.append(cleared)
            last_env = env

        avg_r       = sum(all_rewards) / len(all_rewards)
        avg_steps   = int(sum(all_steps)   / len(all_steps))
        avg_cleared = int(sum(all_cleared) / len(all_cleared))
        score       = round(min(1.0, max(0.0, avg_r / self._max_reward())), 4)

        details = {
            "avg_total_reward":  round(avg_r, 4),
            "min_reward":        round(min(all_rewards), 4),
            "max_reward":        round(max(all_rewards), 4),
            "std_reward":        round(_std(all_rewards), 4),
            "avg_steps":         avg_steps,
            "n_episodes":        n_episodes,
            "final_zones": [
                {
                    "name":         z.name,
                    "injured_left": z.injured,
                    "food_left":    z.food_need,
                    "cleared":      z.injured == 0 and z.food_need == 0,
                    "morale":       round(z.morale, 3),
                    "disease_risk": round(z.disease_risk, 3),
                }
                for z in (last_env.zones if last_env else [])
            ],
        }

        return GraderResult(
            task_id      = self.task_id,
            score        = score,
            passed       = score >= self.success_threshold,
            steps_taken  = avg_steps,
            total_reward = round(avg_r, 4),
            zones_cleared= avg_cleared,
            details      = details,
        )

    def _max_reward(self) -> float:
        from env.config import get_config
        return get_config(self.difficulty)["max_steps"] * 0.6


class EasyGrader(BaseGrader):
    task_id           = "easy"
    difficulty        = "easy"
    success_threshold = 0.5


class MediumGrader(BaseGrader):
    task_id           = "medium"
    difficulty        = "medium"
    success_threshold = 0.3


class HardGrader(BaseGrader):
    task_id           = "hard"
    difficulty        = "hard"
    success_threshold = 0.15


# ── Convenience functions (for server.py / graders.py compat) ─────────

def run_all_graders(agent, n_episodes: int = 10) -> list:
    """Run all 3 graders and print a summary table."""
    graders = [EasyGrader(), MediumGrader(), HardGrader()]
    results = []

    print(f"\n{'='*58}")
    print(f"  Running all graders  ({n_episodes} episodes each)")
    print(f"{'='*58}")

    for g in graders:
        print(f"  Grading {g.task_id.upper()}...", end="", flush=True)
        r = g.run(agent, n_episodes)
        results.append(r)
        status = "PASS ✓" if r.passed else "FAIL ✗"
        print(f"  score={r.score:.4f}  [{status}]  cleared={r.zones_cleared}")

    print(f"\n  Overall: {'ALL PASSED' if all(r.passed for r in results) else 'SOME FAILED'}")
    print(f"{'='*58}\n")
    return results


def _std(vals: list) -> float:
    if len(vals) < 2:
        return 0.0
    mean = sum(vals) / len(vals)
    return (sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5
