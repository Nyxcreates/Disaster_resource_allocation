"""
training/callbacks.py — Custom callbacks for PPO training.

ProgressCallback  — live console progress every N steps
CheckpointCallback — saves best model whenever a new high score is hit
"""

import os
from stable_baselines3.common.callbacks import BaseCallback


class ProgressCallback(BaseCallback):
    """
    Prints training progress to console every `print_every` steps.
    Shows rolling average, best-ever, and current difficulty.
    """

    def __init__(self, print_every: int = 2000, difficulty: str = "?", verbose: int = 0):
        super().__init__(verbose)
        self.print_every        = print_every
        self.difficulty         = difficulty
        self.episode_rewards    = []
        self.current_ep_reward  = 0.0
        self._best              = float("-inf")

    def _on_step(self) -> bool:
        self.current_ep_reward += self.locals["rewards"][0]

        if self.locals["dones"][0]:
            self.episode_rewards.append(self.current_ep_reward)
            self.current_ep_reward = 0.0

        if self.num_timesteps % self.print_every == 0 and self.episode_rewards:
            recent = self.episode_rewards[-20:]
            avg    = sum(recent) / len(recent)
            best   = max(self.episode_rewards)
            trend  = "▲" if len(recent) > 1 and recent[-1] > recent[0] else "▼"
            print(
                f"  [{self.difficulty.upper()}] step {self.num_timesteps:>8,} | "
                f"avg(20)={avg:6.3f} | best={best:6.3f} {trend}",
                flush=True,
            )
        return True


class BestModelCallback(BaseCallback):
    """
    Saves the model whenever it achieves a new best average reward.
    Keeps you from losing a good checkpoint if later training degrades.
    """

    def __init__(self, save_path: str, check_every: int = 5000, verbose: int = 0):
        super().__init__(verbose)
        self.save_path      = save_path
        self.check_every    = check_every
        self.episode_rewards= []
        self.current_ep     = 0.0
        self._best_avg      = float("-inf")

    def _on_step(self) -> bool:
        self.current_ep += self.locals["rewards"][0]
        if self.locals["dones"][0]:
            self.episode_rewards.append(self.current_ep)
            self.current_ep = 0.0

        if self.num_timesteps % self.check_every == 0 and len(self.episode_rewards) >= 10:
            avg = sum(self.episode_rewards[-20:]) / min(20, len(self.episode_rewards))
            if avg > self._best_avg:
                self._best_avg = avg
                path = f"{self.save_path}_best"
                self.model.save(path)
                print(f"  ✓ New best avg={avg:.4f} — saved to {path}.zip", flush=True)
        return True
