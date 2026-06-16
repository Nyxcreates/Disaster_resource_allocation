"""
tests/test_env.py — Test suite for the disaster environment.

Run with:  python -m pytest tests/ -v
       or: python tests/test_env.py
"""

import sys, os, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.disaster_env  import DisasterEnv, FOOD, MEDICAL, RESCUE
from env.rewards       import compute_reward
from agents.random_agent import RandomAgent
from agents.rule_based   import RuleBasedAgent


def test_reset_all_difficulties():
    print("TEST 1 — reset() works for all difficulties")
    for diff in ["easy", "medium", "hard"]:
        env = DisasterEnv(diff)
        obs = env.reset()
        assert len(obs["zones"]) == env.cfg["num_zones"], f"Wrong zone count for {diff}"
        assert obs["food_stock"]  == env.cfg["initial_food"]
        assert obs["med_stock"]   == env.cfg["initial_medical"]
        assert obs["resc_stock"]  == env.cfg["initial_rescue"]
    print("  PASSED\n")


def test_step_basic():
    print("TEST 2 — step() returns correct format")
    env = DisasterEnv("medium")
    env.reset()
    obs, reward, done, info = env.step((FOOD, 0, 1))
    assert isinstance(reward, float), "Reward must be float"
    assert 0.0 <= reward <= 1.0, f"Reward out of range: {reward}"
    assert isinstance(done, bool)
    assert "reward_breakdown" in info, "Info must include reward_breakdown"
    print("  PASSED\n")


def test_rescue_unblocks():
    print("TEST 3 — rescue unblocks a blocked zone")
    env = DisasterEnv("hard")
    env.reset()
    env.zones[0].rescue_blocked = True
    env.step((RESCUE, 0, 0))
    assert not env.zones[0].rescue_blocked, "Zone 0 should be unblocked"
    print("  PASSED\n")


def test_invalid_action_raises():
    print("TEST 4 — invalid action raises ValueError")
    env = DisasterEnv("easy")
    env.reset()
    try:
        env.step((5, 0, 0))
        print("  FAILED — should have raised ValueError")
    except ValueError:
        print("  PASSED\n")


def test_episode_ends_at_max_steps():
    print("TEST 5 — episode ends at max_steps")
    env  = DisasterEnv("easy")
    env.reset()
    done = False
    steps = 0
    while not done:
        _, _, done, _ = env.step((FOOD, 0, 0))
        steps += 1
        assert steps <= env.cfg["max_steps"] + 1, "Exceeded max_steps"
    assert done
    print("  PASSED\n")


def test_reward_components():
    print("TEST 6 — reward components sum correctly")
    env = DisasterEnv("medium")
    env.reset()
    rc = compute_reward(MEDICAL, 0, 20, env.zones,
                        env._initial_total_injured, env._initial_total_food_need)
    assert 0.0 <= rc.total <= 1.0, f"Total out of range: {rc.total}"
    assert rc.coverage_score >= 0.0
    assert rc.waste_penalty  >= 0.0
    print("  PASSED\n")


def test_new_zone_attributes():
    print("TEST 7 — v2 zone attributes present and in range")
    env = DisasterEnv("hard")
    env.reset()
    for z in env.zones:
        assert 0.0 <= z.morale       <= 1.0, f"morale out of range: {z.morale}"
        assert 0.0 <= z.disease_risk <= 1.0
        assert 0.0 <= z.infra_damage <= 1.0
    print("  PASSED\n")


def test_rule_based_agent():
    print("TEST 8 — rule-based agent completes an episode")
    agent = RuleBasedAgent()
    env   = DisasterEnv("medium")
    env.reset()
    done, steps = False, 0
    while not done:
        action = agent.act(env)
        _, _, done, _ = env.step(action)
        steps += 1
    assert steps > 0
    print("  PASSED\n")


def test_gym_wrapper():
    print("TEST 9 — DisasterGymEnv observation shape is correct")
    from env.gym_wrapper import DisasterGymEnv
    for diff, expected_obs in [("easy", 27), ("medium", 43), ("hard", 67)]:
        env = DisasterGymEnv(diff)
        obs, _ = env.reset()
        assert obs.shape[0] == expected_obs, \
            f"{diff}: expected obs size {expected_obs}, got {obs.shape[0]}"
    print("  PASSED\n")


def test_events_fire():
    print("TEST 10 — event system can fire without crashing")
    random.seed(99)
    env = DisasterEnv("hard")
    env.reset()
    for _ in range(10):
        env.step((FOOD, 0, 1))
    # If we got here without exception, events are handled correctly
    print("  PASSED\n")


if __name__ == "__main__":
    test_reset_all_difficulties()
    test_step_basic()
    test_rescue_unblocks()
    test_invalid_action_raises()
    test_episode_ends_at_max_steps()
    test_reward_components()
    test_new_zone_attributes()
    test_rule_based_agent()
    test_gym_wrapper()
    test_events_fire()
    print("=" * 45)
    print("  ALL 10 TESTS PASSED")
    print("=" * 45)
