"""
env/config.py — Difficulty level settings for the disaster environment.

v2 additions:
  - event_prob : probability of a random mid-episode event each step
  - supply_drop_prob : chance of a helpful supply drop event
  - aftershock_prob  : chance of a damaging aftershock event
"""

CONFIGS = {

    "easy": {
        "num_zones":        3,
        "initial_food":     200,
        "initial_medical":  100,
        "initial_rescue":   15,
        "quantity_options": [10, 20, 30],
        "escalation_rate":  0.0,
        "max_steps":        20,
        "restock_rate":     0.0,
        # Events (new in v2)
        "aftershock_prob":  0.0,
        "supply_drop_prob": 0.05,
    },

    "medium": {
        "num_zones":        5,
        "initial_food":     300,
        "initial_medical":  150,
        "initial_rescue":   20,
        "quantity_options": [10, 20, 30],
        "escalation_rate":  0.05,
        "max_steps":        25,
        "restock_rate":     0.05,
        "aftershock_prob":  0.08,
        "supply_drop_prob": 0.04,
    },

    "hard": {
        "num_zones":        8,
        "initial_food":     250,
        "initial_medical":  100,
        "initial_rescue":   12,
        "quantity_options": [10, 20, 30],
        "escalation_rate":  0.10,
        "max_steps":        30,
        "restock_rate":     0.02,
        "aftershock_prob":  0.15,
        "supply_drop_prob": 0.02,
    },
}


def get_config(difficulty: str) -> dict:
    if difficulty not in CONFIGS:
        raise ValueError(
            f"Unknown difficulty '{difficulty}'. "
            f"Choose from: {list(CONFIGS.keys())}"
        )
    return CONFIGS[difficulty]
