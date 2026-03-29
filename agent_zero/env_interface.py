"""
Generic environment interface.

Every environment — ARC game, text puzzle, code task, search space —
implements this interface. The agent never sees environment internals.
"""

from abc import ABC, abstractmethod
from typing import Any


class Env(ABC):
    """
    Minimal environment interface.

    State:   any hashable (string, tuple, int)
    Action:  any hashable (int, string, tuple)
    Reward:  float (0 = no progress, 1 = state change, 100 = level solved)
    """

    @abstractmethod
    def reset(self) -> tuple[Any, set]:
        """Reset. Returns (state, available_actions)."""

    @abstractmethod
    def step(self, action) -> tuple[Any, set, float, bool, bool]:
        """
        Take action. Returns (state, available_actions, reward, level_up, done).
        reward: 0 = no change, >0 = progress, 100 = level complete
        level_up: True if a level was just solved
        done: True if game over (episode end or fully solved)
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Environment name."""

    @property
    def total_levels(self) -> int:
        return 1

    @property
    def current_level(self) -> int:
        return 0
