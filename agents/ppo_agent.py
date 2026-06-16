"""
agents/ppo_agent.py — Wraps a saved PPO model into the BaseAgent interface.

Usage:
    agent = PPOAgent("checkpoints/disaster_agent_medium")
    agent.act(env)   # returns (resource_type, zone_id, qty_index)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.base import BaseAgent
from env.gym_wrapper import DisasterGymEnv


class PPOAgent(BaseAgent):
    name        = "ppo_agent"
    description = "Trained PPO (Proximal Policy Optimization) neural network agent."

    def __init__(self, model_path: str, difficulty: str = "medium"):
        try:
            from stable_baselines3 import PPO
        except ImportError:
            raise ImportError("Run: pip install stable-baselines3")

        self._model      = PPO.load(model_path)
        self._difficulty = difficulty

        # We need a gym env to do obs encoding
        self._gym_env    = DisasterGymEnv(difficulty)
        self._gym_env.reset()

    def act(self, env) -> tuple:
        obs = self._gym_env._encode_obs(env.state())
        action_int, _ = self._model.predict(obs, deterministic=True)
        return self._gym_env._action_lookup[int(action_int)]
