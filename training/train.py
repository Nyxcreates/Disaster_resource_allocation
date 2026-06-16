"""
training/train.py — Train a PPO agent on the disaster environment.

Usage:
    # Quick test:
    python training/train.py --difficulty medium --steps 100000

    # Full curriculum (recommended):
    python training/train.py --curriculum --steps 500000

    # Single difficulty, long run:
    python training/train.py --difficulty hard --steps 300000

After training, checkpoints are saved to ./checkpoints/
"""

import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CallbackList

from env.gym_wrapper       import DisasterGymEnv
from training.callbacks    import ProgressCallback, BestModelCallback
from training.curriculum   import CurriculumTrainer


def train_single(difficulty: str, total_steps: int, checkpoint_dir: str = "checkpoints") -> PPO:
    os.makedirs(checkpoint_dir, exist_ok=True)
    env       = Monitor(DisasterGymEnv(difficulty))
    save_path = os.path.join(checkpoint_dir, f"disaster_agent_{difficulty}")

    print(f"\n{'='*60}")
    print(f"  Training PPO  |  {difficulty.upper()}  |  {total_steps:,} steps")
    print(f"{'='*60}\n")
    print(f"  {'Step':>10} | {'Avg(20 eps)':^18} | {'Best':>10}")
    print(f"  {'─'*10}─{'─'*20}─{'─'*10}")

    model = PPO(
        policy         = "MlpPolicy",
        env            = env,
        learning_rate  = 3e-4,
        n_steps        = 512,
        batch_size     = 64,
        n_epochs       = 10,
        gamma          = 0.99,
        clip_range     = 0.2,
        ent_coef       = 0.01,
        verbose        = 0,
        tensorboard_log= "./logs/",
    )

    callbacks = CallbackList([
        ProgressCallback(print_every=2000, difficulty=difficulty),
        BestModelCallback(save_path=save_path, check_every=5000),
    ])

    model.learn(total_timesteps=total_steps, callback=callbacks, progress_bar=False)
    model.save(save_path)
    print(f"\n  Final model → {save_path}.zip\n")
    return model


def evaluate(model_path: str, difficulty: str, n_episodes: int = 10):
    from stable_baselines3 import PPO
    model = PPO.load(model_path)
    env   = DisasterGymEnv(difficulty)

    BASELINES = {"easy": 0.59, "medium": 0.16, "hard": 0.09}
    all_rewards = []

    print(f"\n  Evaluating {n_episodes} episodes on {difficulty.upper()}...")
    for ep in range(1, n_episodes + 1):
        obs, _ = env.reset()
        done, ep_reward = False, 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, _ = env.step(int(action))
            ep_reward += r
            done = term or trunc
        all_rewards.append(ep_reward)

    avg      = sum(all_rewards) / len(all_rewards)
    baseline = BASELINES.get(difficulty, 0.0)
    print(f"  Avg reward: {avg:.4f}  |  Rule-based baseline: {baseline:.4f}")
    if avg > baseline:
        pct = (avg - baseline) / baseline * 100
        print(f"  ✓ Beats baseline by {pct:.1f}%")
    else:
        print(f"  ✗ Below baseline — try more training steps")


def main():
    parser = argparse.ArgumentParser(description="Train disaster-relief AI agent")
    parser.add_argument("--difficulty",  choices=["easy","medium","hard"], default="medium")
    parser.add_argument("--steps",       type=int, default=100_000)
    parser.add_argument("--curriculum",  action="store_true", help="Use curriculum training")
    parser.add_argument("--eval",        action="store_true", help="Evaluate after training")
    parser.add_argument("--checkpoint",  default="checkpoints", help="Checkpoint directory")
    args = parser.parse_args()

    if args.curriculum:
        trainer = CurriculumTrainer(total_steps=args.steps, checkpoint_dir=args.checkpoint)
        model   = trainer.train()
        model_path = os.path.join(args.checkpoint, "disaster_agent_final")
    else:
        model = train_single(args.difficulty, args.steps, args.checkpoint)
        model_path = os.path.join(args.checkpoint, f"disaster_agent_{args.difficulty}")

    if args.eval:
        diff = "medium" if args.curriculum else args.difficulty
        evaluate(model_path, diff)


if __name__ == "__main__":
    main()
