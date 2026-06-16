"""
env/zone.py — Single disaster zone with rich state and event support.

Improvements over v1:
  - morale attribute: drops when help is delayed, boosts reward when high
  - aftershock / surge events can be applied externally by the env
  - disease_risk grows if food_need stays unmet too long
  - infrastructure_dmg affects how efficiently resources are used
"""

import random
from dataclasses import dataclass, field
from typing import Optional

DISASTER_TYPES = ["earthquake", "flood", "hurricane", "wildfire"]


@dataclass
class ZoneEvent:
    """A mid-episode event that modifies zone state."""
    event_type: str   # "aftershock" | "disease_outbreak" | "supply_drop" | "road_cleared"
    magnitude:  float # 0.0–1.0 severity / benefit
    step:       int   # which step it occurred


class Zone:
    """
    A single disaster-affected zone.

    New attributes vs v1:
        morale          float  0–1   community resilience; drops each unhelped step
        disease_risk    float  0–1   grows if food unmet; triggers injury surge
        infra_damage    float  0–1   reduces resource effectiveness (0=intact, 1=destroyed)
        events          list   mid-episode events that happened here
        steps_unhelped  int    how many consecutive steps with zero aid received
    """

    def __init__(self, zone_id: int):
        self.zone_id        = zone_id
        self.name           = f"Zone {chr(65 + zone_id)}"

        self.disaster_type  = ""
        self.population     = 0
        self.injured        = 0
        self.food_need      = 0
        self.rescue_blocked = False
        self.severity       = 0.0

        # New attributes
        self.morale         = 1.0   # starts high; degrades over time
        self.disease_risk   = 0.0   # 0=safe, 1=outbreak imminent
        self.infra_damage   = 0.0   # infrastructure damage level

        self.initial_injured   = 0
        self.initial_food_need = 0
        self.steps_unhelped    = 0
        self.events: list[ZoneEvent] = []

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def randomize(self, difficulty: str):
        self.disaster_type = random.choice(DISASTER_TYPES)

        if difficulty == "easy":
            self.population     = random.randint(100, 400)
            self.injured        = random.randint(5, 50)
            self.food_need      = random.randint(20, 80)
            self.severity       = round(random.uniform(0.1, 0.4), 2)
            self.rescue_blocked = False
            self.infra_damage   = round(random.uniform(0.0, 0.2), 2)

        elif difficulty == "medium":
            self.population     = random.randint(200, 800)
            self.injured        = random.randint(20, 150)
            self.food_need      = random.randint(50, 200)
            self.severity       = round(random.uniform(0.3, 0.7), 2)
            self.rescue_blocked = random.random() < 0.2
            self.infra_damage   = round(random.uniform(0.1, 0.5), 2)

        elif difficulty == "hard":
            self.population     = random.randint(500, 2000)
            self.injured        = random.randint(100, 500)
            self.food_need      = random.randint(150, 400)
            self.severity       = round(random.uniform(0.6, 1.0), 2)
            self.rescue_blocked = random.random() < 0.4
            self.infra_damage   = round(random.uniform(0.3, 0.8), 2)

        self.morale            = round(random.uniform(0.6, 1.0), 2)
        self.disease_risk      = 0.0
        self.initial_injured   = self.injured
        self.initial_food_need = self.food_need
        self.steps_unhelped    = 0
        self.events            = []

    # ------------------------------------------------------------------
    # Resource application  (infra_damage reduces effectiveness)
    # ------------------------------------------------------------------

    def apply_food(self, units: int) -> int:
        """Return actual units applied (reduced by infra damage)."""
        effective = max(1, int(units * (1.0 - self.infra_damage * 0.4)))
        applied   = min(effective, self.food_need)
        self.food_need     = max(0, self.food_need - applied)
        self.steps_unhelped = 0
        # food aid boosts morale
        self.morale = min(1.0, self.morale + 0.05)
        return applied

    def apply_medical(self, kits: int) -> int:
        effective = max(1, int(kits * (1.0 - self.infra_damage * 0.3)))
        applied   = min(effective, self.injured)
        self.injured        = max(0, self.injured - applied)
        self.steps_unhelped = 0
        self.disease_risk   = max(0.0, self.disease_risk - 0.15)
        self.morale = min(1.0, self.morale + 0.08)
        return applied

    def apply_rescue(self, teams: int) -> int:
        actual_teams = teams
        if self.rescue_blocked:
            self.rescue_blocked = False
            actual_teams -= 1
            self.infra_damage = max(0.0, self.infra_damage - 0.1)

        injured_saved = 0
        if actual_teams > 0:
            injured_saved = actual_teams * 5
            self.injured  = max(0, self.injured - injured_saved)
            self.steps_unhelped = 0
            self.morale   = min(1.0, self.morale + 0.10)

        return injured_saved

    # ------------------------------------------------------------------
    # Per-step updates
    # ------------------------------------------------------------------

    def escalate(self, rate: float):
        """Situation worsens each round; morale and disease risk shift too."""
        self.food_need  = int(self.food_need * (1 + rate))
        self.injured    = int(self.injured   * (1 + rate * 0.5))
        self.steps_unhelped += 1

        # Morale degrades when unhelped
        morale_loss = 0.03 * (1.0 + self.severity)
        self.morale = max(0.0, self.morale - morale_loss)

        # Disease risk grows if food unmet
        if self.food_need > self.initial_food_need * 0.5:
            self.disease_risk = min(1.0, self.disease_risk + 0.04)

        # Disease outbreak: sudden injury surge
        if self.disease_risk > 0.7 and random.random() < 0.15:
            surge = int(self.population * 0.05)
            self.injured += surge
            self.events.append(ZoneEvent("disease_outbreak", self.disease_risk, -1))

    def apply_event(self, event: ZoneEvent):
        """Apply a mid-episode event to this zone."""
        self.events.append(event)

        if event.event_type == "aftershock":
            extra_injured = int(self.population * 0.04 * event.magnitude)
            self.injured += extra_injured
            self.infra_damage = min(1.0, self.infra_damage + 0.1 * event.magnitude)
            self.severity = min(1.0, self.severity + 0.05 * event.magnitude)

        elif event.event_type == "supply_drop":
            self.food_need  = max(0, self.food_need - int(30 * event.magnitude))
            self.morale     = min(1.0, self.morale + 0.1)

        elif event.event_type == "road_cleared":
            self.rescue_blocked = False
            self.infra_damage   = max(0.0, self.infra_damage - 0.2)

    # ------------------------------------------------------------------
    # State snapshot
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "zone_id":        self.zone_id,
            "name":           self.name,
            "disaster_type":  self.disaster_type,
            "population":     self.population,
            "injured":        self.injured,
            "food_need":      self.food_need,
            "rescue_blocked": self.rescue_blocked,
            "severity":       self.severity,
            "morale":         round(self.morale, 3),
            "disease_risk":   round(self.disease_risk, 3),
            "infra_damage":   round(self.infra_damage, 3),
        }

    def __repr__(self):
        blocked = " [BLOCKED]" if self.rescue_blocked else ""
        return (
            f"{self.name}{blocked} | {self.disaster_type} | "
            f"sev={self.severity:.2f} | inj={self.injured} | "
            f"food={self.food_need} | morale={self.morale:.2f} | "
            f"disease={self.disease_risk:.2f}"
        )
