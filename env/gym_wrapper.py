"""
env/gym_wrapper.py — Gymnasium wrapper for DisasterEnv.

Observation vector v2 (per zone, 8 values now instead of 5):
    injured, food_need, severity, rescue_blocked,
    disaster_type_enc, morale, disease_risk, infra_damage

Plus 3 global stock values at the end.

Medium (5 zones): 5×8 + 3 = 43 values
Easy   (3 zones): 3×8 + 3 = 27 values
Hard   (8 zones): 8×8 + 3 = 67 values
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.disaster_env import DisasterEnv, FOOD, MEDICAL, RESCUE


class DisasterGymEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    DISASTER_ENCODING = {
        "earthquake": 0.25,
        "flood":      0.50,
        "hurricane":  0.75,
        "wildfire":   1.00,
    }

    def __init__(self, difficulty: str = "medium"):
        super().__init__()
        self.difficulty = difficulty
        self._env = DisasterEnv(difficulty)
        self._env.reset()

        cfg       = self._env.cfg
        num_zones = cfg["num_zones"]

        self.total_actions = 3 * num_zones * len(cfg["quantity_options"])
        self.action_space  = spaces.Discrete(self.total_actions)

        obs_size = num_zones * 8 + 3   # 8 values per zone in v2
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )

        self._action_lookup = self._build_action_lookup()
        self._max_need  = 500
        self._max_stock = max(
            cfg["initial_food"], cfg["initial_medical"], cfg["initial_rescue"]
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs_dict = self._env.reset()
        return self._encode_obs(obs_dict), {}

    def step(self, action: int):
        decoded = self._action_lookup[int(action)]
        obs_dict, reward, done, info = self._env.step(decoded)
        terminated = done and self._all_zones_clear()
        truncated  = done and not terminated
        return self._encode_obs(obs_dict), reward, terminated, truncated, info

    def render(self):
        self._env.render()

    def _encode_obs(self, obs_dict: dict) -> np.ndarray:
        values = []
        for zone in obs_dict["zones"]:
            values.append(min(1.0, zone["injured"]    / self._max_need))
            values.append(min(1.0, zone["food_need"]  / self._max_need))
            values.append(zone["severity"])
            values.append(1.0 if zone["rescue_blocked"] else 0.0)
            values.append(self.DISASTER_ENCODING.get(zone["disaster_type"], 0.0))
            values.append(zone.get("morale",       1.0))
            values.append(zone.get("disease_risk", 0.0))
            values.append(zone.get("infra_damage", 0.0))

        values.append(min(1.0, obs_dict["food_stock"] / max(1, self._max_stock)))
        values.append(min(1.0, obs_dict["med_stock"]  / max(1, self._max_stock)))
        values.append(min(1.0, obs_dict["resc_stock"] / max(1, self._max_stock)))

        return np.clip(np.array(values, dtype=np.float32), 0.0, 1.0)

    def _build_action_lookup(self) -> dict:
        lookup    = {}
        num_zones = self._env.cfg["num_zones"]
        num_qtys  = len(self._env.cfg["quantity_options"])
        idx = 0
        for resource in [FOOD, MEDICAL, RESCUE]:
            for zone_id in range(num_zones):
                for qty_idx in range(num_qtys):
                    lookup[idx] = (resource, zone_id, qty_idx)
                    idx += 1
        return lookup

    def _all_zones_clear(self) -> bool:
        return all(z.injured == 0 and z.food_need == 0 for z in self._env.zones)
