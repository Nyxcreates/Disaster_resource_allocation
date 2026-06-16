from env.disaster_env import DisasterEnv, FOOD, MEDICAL, RESCUE
from env.config       import get_config, CONFIGS
from env.rewards      import compute_reward, RewardComponents

def __getattr__(name):
    if name == "DisasterGymEnv":
        from env.gym_wrapper import DisasterGymEnv
        return DisasterGymEnv
    raise AttributeError(f"module 'env' has no attribute {name!r}")

__all__ = [
    "DisasterEnv", "DisasterGymEnv",
    "FOOD", "MEDICAL", "RESCUE",
    "get_config", "CONFIGS",
    "compute_reward", "RewardComponents",
]
