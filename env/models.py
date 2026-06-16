"""
env/models.py — Data models for the OpenEnv spec.

Uses dataclasses (stdlib) so the env works without pydantic installed.
Pydantic validation is added when available (FastAPI server uses it).
"""

from dataclasses import dataclass, field as dc_field

# ── Try pydantic for the API server; fall back to dataclasses ─────────
try:
    from pydantic import BaseModel, Field

    class ZoneState(BaseModel):
        zone_id:        int
        name:           str
        disaster_type:  str
        population:     int
        injured:        int
        food_need:      int
        rescue_blocked: bool
        severity:       float
        morale:         float = 1.0
        disease_risk:   float = 0.0
        infra_damage:   float = 0.0

    class Observation(BaseModel):
        zones:      list
        food_stock: int
        med_stock:  int
        resc_stock: int
        step:       int
        max_steps:  int
        difficulty: str

    class Action(BaseModel):
        resource_type:  int
        zone_id:        int
        quantity_index: int

        def to_tuple(self):
            return (self.resource_type, self.zone_id, self.quantity_index)

    class RewardBreakdown(BaseModel):
        coverage:        float
        triage:          float
        rescue_priority: float
        efficiency:      float
        morale_bonus:    float
        waste_penalty:   float
        total:           float

except ImportError:

    @dataclass
    class ZoneState:
        zone_id:        int
        name:           str
        disaster_type:  str
        population:     int
        injured:        int
        food_need:      int
        rescue_blocked: bool
        severity:       float
        morale:         float = 1.0
        disease_risk:   float = 0.0
        infra_damage:   float = 0.0

    @dataclass
    class Observation:
        zones:      list
        food_stock: int
        med_stock:  int
        resc_stock: int
        step:       int
        max_steps:  int
        difficulty: str

    @dataclass
    class Action:
        resource_type:  int
        zone_id:        int
        quantity_index: int

        def to_tuple(self):
            return (self.resource_type, self.zone_id, self.quantity_index)

    @dataclass
    class RewardBreakdown:
        coverage:        float
        triage:          float
        rescue_priority: float
        efficiency:      float
        morale_bonus:    float
        waste_penalty:   float
        total:           float


# GraderResult is always a plain dataclass — no HTTP serialisation needed
@dataclass
class GraderResult:
    task_id:       str
    score:         float
    passed:        bool
    steps_taken:   int
    total_reward:  float
    zones_cleared: int
    details:       dict = dc_field(default_factory=dict)

    def dict(self):
        return {
            "task_id":       self.task_id,
            "score":         self.score,
            "passed":        self.passed,
            "steps_taken":   self.steps_taken,
            "total_reward":  self.total_reward,
            "zones_cleared": self.zones_cleared,
            "details":       self.details,
        }
