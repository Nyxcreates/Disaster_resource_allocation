"""agents/random_agent.py — Picks completely random actions."""

import random
from agents.base import BaseAgent


class RandomAgent(BaseAgent):
    name        = "random"
    description = "Picks resource, zone, and quantity uniformly at random."

    def act(self, env) -> tuple:
        resource = random.randint(0, 2)
        zone     = random.randint(0, len(env.zones) - 1)
        qty      = random.randint(0, 2)
        return (resource, zone, qty)
