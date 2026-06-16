"""
env/rewards.py — Reward computation extracted into its own module.

Having a separate reward module makes it:
  - Easier to tune weights without touching the environment
  - Testable in isolation
  - Inspectable (the agent can see WHY it got a certain score)

RewardComponents is a dataclass holding every sub-score.
compute_reward() assembles them into a final float.

Reward components (v2):
  coverage_score    (40%) — fraction of total need already met
  triage_score      (25%) — did we help the highest-severity zone?
  rescue_priority   (15%) — bonus for unblocking a blocked zone
  efficiency_score  (10%) — did we send the right resource type?
  morale_bonus      (05%) — zones with high morale give small bonus
  waste_penalty     (up to -0.15) — wrong resource to wrong zone
"""

from dataclasses import dataclass
from typing import List


FOOD    = 0
MEDICAL = 1
RESCUE  = 2


@dataclass
class RewardComponents:
    coverage_score:   float = 0.0
    triage_score:     float = 0.0
    rescue_priority:  float = 0.0
    efficiency_score: float = 0.0
    morale_bonus:     float = 0.0
    waste_penalty:    float = 0.0
    total:            float = 0.0

    def to_dict(self) -> dict:
        return {
            "coverage":        round(self.coverage_score, 4),
            "triage":          round(self.triage_score, 4),
            "rescue_priority": round(self.rescue_priority, 4),
            "efficiency":      round(self.efficiency_score, 4),
            "morale_bonus":    round(self.morale_bonus, 4),
            "waste_penalty":   round(self.waste_penalty, 4),
            "total":           round(self.total, 4),
        }


def compute_reward(
    resource_type:  int,
    zone_id:        int,
    quantity_sent:  int,
    zones:          list,
    initial_total_injured:   int,
    initial_total_food_need: int,
) -> RewardComponents:
    """
    Compute a structured reward for one action.

    Args:
        resource_type  : 0=food, 1=medical, 2=rescue
        zone_id        : which zone was targeted
        quantity_sent  : actual units delivered (after stock cap)
        zones          : current list of Zone objects
        initial_total_injured   : episode-start sum of injured
        initial_total_food_need : episode-start sum of food_need

    Returns:
        RewardComponents with all sub-scores and a .total field
    """
    zone = zones[zone_id]
    rc   = RewardComponents()

    # ── 1. Coverage score (40%) ────────────────────────────────────
    # How much of the original total need has been satisfied?
    total_injured   = sum(z.injured   for z in zones)
    total_food_need = sum(z.food_need for z in zones)

    inj_covered  = 1.0 - (total_injured   / max(1, initial_total_injured))
    food_covered = 1.0 - (total_food_need / max(1, initial_total_food_need))
    rc.coverage_score = max(0.0, min(1.0, (inj_covered + food_covered) / 2.0))

    # ── 2. Triage score (25%) ──────────────────────────────────────
    # Did we help the most critical zone (highest severity)?
    most_critical_id = max(range(len(zones)), key=lambda i: zones[i].severity)
    if zone_id == most_critical_id:
        rc.triage_score = 1.0
    elif zones[zone_id].severity > 0.5:
        rc.triage_score = 0.6   # helped a high-severity zone, just not the worst
    else:
        rc.triage_score = 0.2

    # ── 3. Rescue priority (15%) ───────────────────────────────────
    # Extra credit for unblocking a blocked zone (or helping a high-need zone)
    blocked_zones = [z for z in zones if z.rescue_blocked]
    if resource_type == RESCUE and not zone.rescue_blocked and blocked_zones:
        # Sent rescue but there was a blocked zone we should have hit
        rc.rescue_priority = 0.1
    elif resource_type == RESCUE and zone_id in [z.zone_id for z in blocked_zones]:
        rc.rescue_priority = 1.0   # perfect: unblocked a blocked zone
    else:
        rc.rescue_priority = 0.4   # neutral action

    # ── 4. Efficiency score (10%) ──────────────────────────────────
    # Right resource for the zone's actual need?
    if resource_type == FOOD and zone.food_need > 0:
        # Extra efficiency if food was the most urgent need
        rc.efficiency_score = 1.0 if zone.food_need >= zone.injured else 0.7
    elif resource_type == MEDICAL and zone.injured > 0:
        rc.efficiency_score = 1.0 if zone.injured >= zone.food_need else 0.7
    elif resource_type == RESCUE and (zone.rescue_blocked or zone.injured > 50):
        rc.efficiency_score = 1.0
    else:
        rc.efficiency_score = 0.2  # resource wasn't needed here

    # ── 5. Morale bonus (5%) ──────────────────────────────────────
    # Zones with high morale respond better; reward maintaining it
    avg_morale = sum(z.morale for z in zones) / max(1, len(zones))
    rc.morale_bonus = avg_morale * 0.5  # scaled: max contribution is 0.5

    # ── 6. Waste penalty (up to -0.15) ────────────────────────────
    if resource_type == FOOD and zone.food_need == 0:
        rc.waste_penalty = 0.15
    elif resource_type == MEDICAL and zone.injured == 0:
        rc.waste_penalty = 0.15
    elif resource_type == RESCUE and not zone.rescue_blocked and zone.injured == 0:
        rc.waste_penalty = 0.10

    # ── Assemble total ─────────────────────────────────────────────
    rc.total = (
        rc.coverage_score   * 0.40
        + rc.triage_score   * 0.25
        + rc.rescue_priority * 0.15
        + rc.efficiency_score * 0.10
        + rc.morale_bonus   * 0.05
        - rc.waste_penalty
    )
    rc.total = round(max(0.0, min(1.0, rc.total)), 4)

    return rc
