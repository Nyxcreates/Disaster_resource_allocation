"""
agents/rule_based.py — Improved triage agent (v2).

Decision priority (in order):
  1. Unblock any rescue-blocked zone
  2. Help zones with imminent disease outbreak (disease_risk > 0.6) with food
  3. Help the zone with highest injured count with medical aid
  4. Help the zone with highest food_need with food
  5. Tie-break by severity

v2 improvements over v1:
  - Aware of disease_risk (new zone attribute)
  - Considers morale when severity is tied
  - Uses largest quantity option when resources are plentiful
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.base import BaseAgent
from env.disaster_env import FOOD, MEDICAL, RESCUE


class RuleBasedAgent(BaseAgent):
    name        = "rule_based"
    description = (
        "Hand-coded triage logic: unblock → disease risk → "
        "most injured → most hungry, prioritised by severity."
    )

    def act(self, env) -> tuple:
        zones = env.zones
        cfg   = env.cfg

        # --- 1. Unblock the most severe blocked zone ---
        blocked = [z for z in zones if z.rescue_blocked]
        if blocked:
            target = max(blocked, key=lambda z: z.severity)
            return (RESCUE, target.zone_id, 1)   # send 20 teams

        # --- 2. Disease outbreak imminent ---
        high_disease = [z for z in zones if z.disease_risk > 0.6]
        if high_disease:
            target = max(high_disease, key=lambda z: z.disease_risk)
            return (FOOD, target.zone_id, 2)   # food reduces disease risk

        # --- 3. Medical emergency ---
        most_injured = max(zones, key=lambda z: (z.injured, z.severity))

        # --- 4. Hunger crisis ---
        most_hungry = max(zones, key=lambda z: (z.food_need, z.severity))

        # --- Decide: which need is more urgent? ---
        if most_injured.injured >= most_hungry.food_need:
            qty_idx = self._quantity_index(env.med_stock, cfg)
            return (MEDICAL, most_injured.zone_id, qty_idx)
        else:
            qty_idx = self._quantity_index(env.food_stock, cfg)
            return (FOOD, most_hungry.zone_id, qty_idx)

    @staticmethod
    def _quantity_index(stock: int, cfg: dict) -> int:
        """Send more when we have plenty, less when scarce."""
        options = cfg["quantity_options"]
        if stock >= options[-1] * 3:
            return 2   # 30 units
        elif stock >= options[1] * 2:
            return 1   # 20 units
        return 0       # 10 units — conserve
