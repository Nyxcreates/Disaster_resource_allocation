"""
main.py — Project entry point.

This is the ONE file you run. It delegates to the right module
based on the --mode flag.

Usage:
    # Start the API server (default — what OpenEnv calls):
    python main.py

    # Run inference only (LLM agent on all 3 tasks):
    python main.py --mode inference

    # Train with curriculum learning:
    python main.py --mode train --steps 500000

    # Train single difficulty:
    python main.py --mode train --difficulty medium --steps 100000

    # Benchmark all agents:
    python main.py --mode benchmark --episodes 50

    # Run comparison chart:
    python main.py --mode compare --episodes 20

    # Run test suite:
    python main.py --mode test
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def mode_server():
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    print(f"[main] Starting API server on port {port}", flush=True)
    uvicorn.run("api.server:app", host="0.0.0.0", port=port, reload=False)


def mode_inference():
    from api.inference import run_all_tasks
    run_all_tasks()


def mode_train(args):
    if args.curriculum:
        from training.curriculum import CurriculumTrainer
        trainer = CurriculumTrainer(
            total_steps    = args.steps,
            checkpoint_dir = args.checkpoint,
        )
        trainer.train()
    else:
        from training.train import train_single, evaluate
        model_path = os.path.join(args.checkpoint, f"disaster_agent_{args.difficulty}")
        train_single(args.difficulty, args.steps, args.checkpoint)
        if args.eval:
            evaluate(model_path, args.difficulty)


def mode_benchmark(args):
    from eval.benchmark import run_benchmark, save_results
    results = run_benchmark(n_episodes=args.episodes, ppo_path=args.ppo)
    save_results(results)


def mode_compare(args):
    from eval.compare import collect, plot
    results = collect(n_episodes=args.episodes, ppo_path=args.ppo)
    plot(results)


def mode_test():
    import subprocess
    result = subprocess.run(
        [sys.executable, "tests/test_env.py"],
        capture_output=False,
    )
    sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(
        description="Disaster Resource Allocation — OpenEnv submission"
    )
    parser.add_argument(
        "--mode",
        choices=["server", "inference", "train", "benchmark", "compare", "test"],
        default="server",
    )
    # Train options
    parser.add_argument("--difficulty",  choices=["easy","medium","hard"], default="medium")
    parser.add_argument("--steps",       type=int, default=100_000)
    parser.add_argument("--curriculum",  action="store_true")
    parser.add_argument("--eval",        action="store_true")
    parser.add_argument("--checkpoint",  default="checkpoints")
    # Eval options
    parser.add_argument("--episodes",    type=int, default=50)
    parser.add_argument("--ppo",         type=str, default=None,
                        help="Path to PPO checkpoint (no .zip extension)")

    args = parser.parse_args()

    if args.mode == "server":
        mode_server()
    elif args.mode == "inference":
        mode_inference()
    elif args.mode == "train":
        mode_train(args)
    elif args.mode == "benchmark":
        mode_benchmark(args)
    elif args.mode == "compare":
        mode_compare(args)
    elif args.mode == "test":
        mode_test()


if __name__ == "__main__":
    main()
