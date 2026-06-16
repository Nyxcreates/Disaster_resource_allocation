"""
agents/base.py — Abstract base class for all agents.

Every agent (random, rule-based, LLM, PPO) implements this interface.
This makes it trivial to swap agents in graders, comparisons, and benchmarks.
"""

from abc import ABC, abstractmethod


class BaseAgent(ABC):
    """
    All agents must implement act().
    Optionally override reset() to clear per-episode state.
    """

    name: str = "base"
    description: str = ""

    def reset(self):
        """Called at the start of each episode. Override if stateful."""
        pass

    @abstractmethod
    def act(self, env) -> tuple:
        """
        Choose an action given the current environment state.

        Args:
            env: DisasterEnv instance (call env.state() for the observation)

        Returns:
            tuple: (resource_type, zone_id, quantity_index)
        """
        ...

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name!r})"
