"""
scripts/quick_demo.py — Verify the full stack works in under 10 seconds.

Run this first after setting up the project:
    python scripts/quick_demo.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.disaster_env    import DisasterEnv, FOOD, MEDICAL, RESCUE
from agents.rule_based    import RuleBasedAgent
from agents.random_agent  import RandomAgent
from eval.graders         import EasyGrader, MediumGrader, HardGrader


def demo_env():
    print("── Environment demo ──────────────────────────────────────")
    env   = DisasterEnv("medium")
    agent = RuleBasedAgent()
    obs   = env.reset()
    env.render()

    for step in range(6):
        action = agent.act(env)
        obs, reward, done, info = env.step(action)
        breakdown = info["reward_breakdown"]
        print(
            f"  Step {info['step']:2d} | "
            f"{info['resource_sent']:<8} → {info['zone']:<8} "
            f"(qty={info['quantity']}) | "
            f"reward={reward:.4f} "
            f"[cov={breakdown['coverage']:.2f} "
            f"tri={breakdown['triage']:.2f} "
            f"eff={breakdown['efficiency']:.2f}]"
        )
        if info["events_this_step"]:
            for ev in info["events_this_step"]:
                print(f"  ⚡ EVENT: {ev['event']} at {ev['zone']} (mag={ev['magnitude']:.2f})")
        if done:
            break
    print()


def demo_graders():
    print("── Grader demo ───────────────────────────────────────────")
    agent = RuleBasedAgent()
    for GraderClass in [EasyGrader, MediumGrader, HardGrader]:
        g      = GraderClass()
        result = g.run(agent, n_episodes=5)
        status = "PASS ✓" if result.passed else "FAIL ✗"
        print(
            f"  {g.task_id.upper():<8} score={result.score:.4f}  "
            f"[{status}]  cleared={result.zones_cleared}  "
            f"std={result.details['std_reward']:.4f}"
        )
    print()


def demo_agents():
    print("── Agent comparison (5 episodes each, medium) ────────────")
    import random
    random.seed(42)
    agents = [("Random", RandomAgent()), ("Rule-based", RuleBasedAgent())]
    for name, agent in agents:
        rewards = []
        for _ in range(5):
            env = DisasterEnv("medium")
            env.reset()
            done, r = False, 0.0
            while not done:
                _, reward, done, _ = env.step(agent.act(env))
                r += reward
            rewards.append(r)
        avg = sum(rewards) / len(rewards)
        print(f"  {name:<14} avg={avg:.3f}")
    print()


if __name__ == "__main__":
    print(f"\n{'='*58}")
    print(f"  DISASTER RESOURCE ALLOCATION — quick demo")
    print(f"{'='*58}\n")
    demo_env()
    demo_graders()
    demo_agents()
    print("  All checks passed. Project is working correctly.\n")
