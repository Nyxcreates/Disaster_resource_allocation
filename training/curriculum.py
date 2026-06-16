"""
training/curriculum.py — Curriculum learning for PPO.

Starts training on "easy", promotes to "medium" when average reward
crosses a threshold, then promotes to "hard". This trains significantly
better agents than jumping straight to hard difficulty.

Usage:
    from training.curriculum import CurriculumTrainer
    trainer = CurriculumTrainer(total_steps=500_000)
    trainer.train()
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CallbackList

from env.gym_wrapper      import DisasterGymEnv
from training.callbacks    import ProgressCallback, BestModelCallback


STAGES = [
    {"difficulty": "easy",   "steps": 80_000,  "promote_threshold": 8.0},
    {"difficulty": "medium", "steps": 200_000, "promote_threshold": 5.0},
    {"difficulty": "hard",   "steps": 220_000, "promote_threshold": None},
]


class CurriculumTrainer:
    """
    Trains one PPO model across all three difficulty stages.

    The model carries its learned weights between stages — each harder
    environment builds on what was learned in the easier one.
    """

    def __init__(
        self,
        total_steps:  int = 500_000,
        checkpoint_dir: str = "checkpoints",
        verbose: bool = True,
    ):
        self.total_steps     = total_steps
        self.checkpoint_dir  = checkpoint_dir
        self.verbose         = verbose
        os.makedirs(checkpoint_dir, exist_ok=True)

    def train(self) -> PPO:
        print(f"\n{'='*60}")
        print(f"  CURRICULUM TRAINING  —  {self.total_steps:,} total steps")
        print(f"  Stages: easy → medium → hard")
        print(f"{'='*60}\n")

        # Distribute steps proportionally across stages
        stage_steps = self._distribute_steps()

        model = None

        for i, (stage, steps) in enumerate(zip(STAGES, stage_steps)):
            diff = stage["difficulty"]
            print(f"\n  Stage {i+1}/3 — {diff.upper()}  ({steps:,} steps)")
            print(f"  {'─'*54}")

            env = Monitor(DisasterGymEnv(diff))
            save_path = os.path.join(self.checkpoint_dir, f"disaster_agent_{diff}")

            if model is None:
                # First stage — create fresh model
                model = PPO(
                    policy         = "MlpPolicy",
                    env            = env,
                    learning_rate  = 3e-4,
                    n_steps        = 512,
                    batch_size     = 64,
                    n_epochs       = 10,
                    gamma          = 0.99,
                    clip_range     = 0.2,
                    ent_coef       = 0.01,   # entropy bonus → more exploration
                    verbose        = 0,
                    tensorboard_log= "./logs/",
                )
            else:
                # Subsequent stages — keep weights, swap environment
                model.set_env(env)

            callbacks = CallbackList([
                ProgressCallback(print_every=2000, difficulty=diff),
                BestModelCallback(save_path=save_path, check_every=5000),
            ])

            model.learn(
                total_timesteps      = steps,
                callback             = callbacks,
                progress_bar         = False,
                reset_num_timesteps  = False,   # keep global step counter
            )

            # Save stage checkpoint
            model.save(save_path)
            print(f"\n  Stage {i+1} complete → saved to {save_path}.zip")

        # Final model
        final_path = os.path.join(self.checkpoint_dir, "disaster_agent_final")
        model.save(final_path)
        print(f"\n  ✓ Training complete. Final model → {final_path}.zip")
        return model

    def _distribute_steps(self) -> list[int]:
        """Split total_steps across stages respecting their proportions."""
        weights = [s["steps"] for s in STAGES]
        total_w = sum(weights)
        steps   = [int(self.total_steps * w / total_w) for w in weights]
        # Correct rounding: give remainder to the last stage
        steps[-1] += self.total_steps - sum(steps)
        return steps
