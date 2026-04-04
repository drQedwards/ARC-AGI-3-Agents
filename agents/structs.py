"""Backwards-compatible re-exports for the ``agents.structs`` namespace.

When the codebase migrated to the ``arc_agi`` package, the local struct
definitions were removed.  This module re-exports the equivalents so that
existing imports continue to work.
"""

from arc_agi.scorecard import EnvironmentScorecard as Scorecard
from arcengine import ActionInput, FrameData, GameAction, GameState

__all__ = [
    "ActionInput",
    "FrameData",
    "GameAction",
    "GameState",
    "Scorecard",
    "Card",
]


# ---------------------------------------------------------------------------
# Card — lightweight per-game statistics helper (removed from arcengine in the
# arc_agi migration; kept here for backwards compatibility with existing tests).
# ---------------------------------------------------------------------------

from typing import List, Optional  # noqa: E402


class Card:
    """Per-game play statistics container."""

    def __init__(
        self,
        game_id: str = "",
        total_plays: int = 0,
        scores: Optional[List[int]] = None,
        states: Optional[List[GameState]] = None,
        actions: Optional[List[int]] = None,
        resets: Optional[List[int]] = None,
    ) -> None:
        self.game_id = game_id
        self.total_plays = total_plays
        self.scores: List[int] = scores or []
        self.states: List[GameState] = states or []
        self.actions: List[int] = actions or []
        self.resets: List[int] = resets or []

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def started(self) -> bool:
        return self.total_plays > 0

    @property
    def idx(self) -> int:
        return self.total_plays - 1

    @property
    def score(self) -> Optional[int]:
        return self.scores[-1] if self.scores else None

    @property
    def high_score(self) -> int:
        return max(self.scores, default=0)

    @property
    def state(self) -> Optional[GameState]:
        return self.states[-1] if self.states else None

    @property
    def action_count(self) -> int:
        return self.actions[-1] if self.actions else 0

    @property
    def total_actions(self) -> int:
        return sum(self.actions)

    def model_dump(self) -> dict:
        return {
            "game_id": self.game_id,
            "total_plays": self.total_plays,
            "scores": self.scores,
            "states": [s.value for s in self.states],
            "actions": self.actions,
            "resets": self.resets,
        }
