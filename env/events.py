"""
env/events.py — Random mid-episode event system.

Events add unpredictability to the simulation, forcing agents to
adapt rather than memorise a fixed strategy.

Event types:
  aftershock      — damages a zone, increases injured + infra_damage
  supply_drop     — external aid reduces food_need in one zone
  road_cleared    — unblocks a random blocked zone for free
  disease_surge   — food deprivation causes sudden injury spike
  media_attention — boosts morale in the hardest-hit zone

Usage (called inside DisasterEnv.step):
    events = EventGenerator(cfg)
    triggered = events.roll(zones, step)
    for zone_id, event in triggered:
        zones[zone_id].apply_event(event)
"""

import random
from env.zone import ZoneEvent


class EventGenerator:
    """
    Rolls random events each step based on difficulty config probabilities.
    Each event type has its own probability and is independently tested.
    """

    def __init__(self, cfg: dict):
        self.aftershock_prob  = cfg.get("aftershock_prob", 0.0)
        self.supply_drop_prob = cfg.get("supply_drop_prob", 0.0)

    def roll(self, zones: list, step: int) -> list[tuple[int, ZoneEvent]]:
        """
        Test each possible event and return a list of (zone_id, ZoneEvent).
        Multiple events can fire in the same step.
        """
        triggered = []

        # ── Aftershock ────────────────────────────────────────────
        if self.aftershock_prob > 0 and random.random() < self.aftershock_prob:
            # Hits the zone with highest severity (already the worst off)
            target_id = max(range(len(zones)), key=lambda i: zones[i].severity)
            mag = round(random.uniform(0.3, 0.8), 2)
            event = ZoneEvent("aftershock", mag, step)
            triggered.append((target_id, event))

        # ── Supply drop ───────────────────────────────────────────
        if self.supply_drop_prob > 0 and random.random() < self.supply_drop_prob:
            # Drops on the zone with highest food_need
            target_id = max(range(len(zones)), key=lambda i: zones[i].food_need)
            mag = round(random.uniform(0.4, 1.0), 2)
            event = ZoneEvent("supply_drop", mag, step)
            triggered.append((target_id, event))

        # ── Road cleared (occasional free unblock) ────────────────
        blocked = [z for z in zones if z.rescue_blocked]
        if blocked and random.random() < 0.04:
            target = random.choice(blocked)
            event = ZoneEvent("road_cleared", 1.0, step)
            triggered.append((target.zone_id, event))

        return triggered
