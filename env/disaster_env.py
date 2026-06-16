"""
env/disaster_env.py — Multi-Zone Disaster Resource Allocation Environment v2.

OpenEnv API:
    env = DisasterEnv("medium")
    obs = env.reset()
    while not done:
        obs, reward, done, info = env.step((resource, zone_id, qty_idx))

Changes from v1:
  - reward computed via env/rewards.py (richer, named components)
  - mid-episode events via env/events.py (aftershocks, supply drops)
  - zone state includes morale, disease_risk, infra_damage
  - info dict includes full reward breakdown for LLM agent context
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.zone    import Zone
from env.config  import get_config
from env.rewards import compute_reward, FOOD, MEDICAL, RESCUE
from env.events  import EventGenerator

RESOURCE_NAMES = {FOOD: "food", MEDICAL: "medical", RESCUE: "rescue"}


class DisasterEnv:
    """
    Multi-Zone Disaster Resource Allocation Environment.

    Action: tuple (resource_type, zone_id, quantity_index)
      resource_type  : 0=food  1=medical  2=rescue
      zone_id        : 0 … num_zones-1
      quantity_index : 0=10  1=20  2=30  units
    """

    def __init__(self, difficulty: str = "medium"):
        self.difficulty = difficulty
        self.cfg        = get_config(difficulty)
        self._events    = EventGenerator(self.cfg)

        self.zones      = []
        self.food_stock = 0
        self.med_stock  = 0
        self.resc_stock = 0
        self.step_count = 0
        self.done       = False

        self._initial_total_injured   = 0
        self._initial_total_food_need = 0
        self._episode_events: list    = []   # log of all events this episode

    # ──────────────────────────────────────────────────────────────
    # OpenEnv API
    # ──────────────────────────────────────────────────────────────

    def reset(self) -> dict:
        num_zones   = self.cfg["num_zones"]
        self.zones  = [Zone(i) for i in range(num_zones)]
        for z in self.zones:
            z.randomize(self.difficulty)

        self.food_stock = self.cfg["initial_food"]
        self.med_stock  = self.cfg["initial_medical"]
        self.resc_stock = self.cfg["initial_rescue"]
        self.step_count = 0
        self.done       = False
        self._episode_events = []

        self._initial_total_injured   = sum(z.injured   for z in self.zones)
        self._initial_total_food_need = sum(z.food_need for z in self.zones)

        return self.state()

    def step(self, action: tuple) -> tuple:
        if self.done:
            raise RuntimeError("Episode over — call reset().")

        resource_type, zone_id, quantity_index = action
        self._validate_action(resource_type, zone_id, quantity_index)

        quantity = self.cfg["quantity_options"][quantity_index]

        # Apply action, get reward components
        actual_sent, reward_components = self._apply_action(
            resource_type, zone_id, quantity
        )

        # Mid-episode events
        triggered = self._events.roll(self.zones, self.step_count)
        for tid, event in triggered:
            self.zones[tid].apply_event(event)
            self._episode_events.append({
                "step": self.step_count,
                "zone": self.zones[tid].name,
                "event": event.event_type,
                "magnitude": event.magnitude,
            })

        # Escalation
        rate = self.cfg["escalation_rate"]
        if rate > 0:
            for z in self.zones:
                z.escalate(rate)

        # Restock
        self._restock()
        self.step_count += 1
        self.done = self._check_done()

        info = {
            "step":           self.step_count,
            "resource_sent":  RESOURCE_NAMES[resource_type],
            "zone":           self.zones[zone_id].name,
            "quantity":       actual_sent,
            "food_stock":     self.food_stock,
            "med_stock":      self.med_stock,
            "resc_stock":     self.resc_stock,
            "reward_breakdown": reward_components.to_dict(),
            "events_this_step": [e for e in self._episode_events
                                  if e["step"] == self.step_count - 1],
        }

        return self.state(), reward_components.total, self.done, info

    def state(self) -> dict:
        return {
            "zones":      [z.to_dict() for z in self.zones],
            "food_stock": self.food_stock,
            "med_stock":  self.med_stock,
            "resc_stock": self.resc_stock,
            "step":       self.step_count,
            "max_steps":  self.cfg["max_steps"],
            "difficulty": self.difficulty,
        }

    # ──────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────

    def _apply_action(self, resource_type: int, zone_id: int, quantity: int):
        zone = self.zones[zone_id]

        if resource_type == FOOD:
            actual = min(quantity, self.food_stock)
            self.food_stock -= actual
            zone.apply_food(actual)

        elif resource_type == MEDICAL:
            actual = min(quantity, self.med_stock)
            self.med_stock -= actual
            zone.apply_medical(actual)

        elif resource_type == RESCUE:
            actual = min(quantity, self.resc_stock)
            self.resc_stock -= actual
            zone.apply_rescue(actual)

        rc = compute_reward(
            resource_type, zone_id, actual, self.zones,
            self._initial_total_injured, self._initial_total_food_need,
        )
        return actual, rc

    def _restock(self):
        rate = self.cfg["restock_rate"]
        if rate > 0:
            self.food_stock  += int(self.cfg["initial_food"]    * rate)
            self.med_stock   += int(self.cfg["initial_medical"] * rate)
            self.resc_stock  += int(self.cfg["initial_rescue"]  * rate)

    def _check_done(self) -> bool:
        if self.step_count >= self.cfg["max_steps"]:
            return True
        return all(z.injured == 0 and z.food_need == 0 for z in self.zones)

    def _validate_action(self, resource_type, zone_id, quantity_index):
        if resource_type not in (FOOD, MEDICAL, RESCUE):
            raise ValueError(f"resource_type must be 0-2. Got: {resource_type}")
        if not (0 <= zone_id < len(self.zones)):
            raise ValueError(f"zone_id must be 0-{len(self.zones)-1}. Got: {zone_id}")
        if not (0 <= quantity_index < len(self.cfg["quantity_options"])):
            raise ValueError(f"quantity_index must be 0-2. Got: {quantity_index}")

    def render(self):
        print(f"\n{'='*60}")
        print(f"  Step {self.step_count}/{self.cfg['max_steps']}  |  {self.difficulty.upper()}")
        print(f"  Stock — Food:{self.food_stock}  Medical:{self.med_stock}  Rescue:{self.resc_stock}")
        print(f"{'='*60}")
        for z in self.zones:
            print(f"  {z}")
        if self._episode_events:
            last = [e for e in self._episode_events if e["step"] == self.step_count - 1]
            for ev in last:
                print(f"  ⚡ EVENT [{ev['zone']}]: {ev['event']} (mag={ev['magnitude']:.2f})")
        print()

    def action_space_size(self) -> int:
        return 3 * len(self.zones) * len(self.cfg["quantity_options"])
