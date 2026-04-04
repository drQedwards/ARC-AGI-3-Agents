"""WorldModelAgent — a world-model-enhanced ReasoningAgent for ARC-AGI-3.

Adds two capabilities on top of :class:`ReasoningAgent`:

1. A structured :class:`WorldModel` that parses the raw grid every frame to
   track player/door/rotator/energy-pill positions, visited cells, and
   confirmed blocked moves — the "PMLL-equivalent" persistent in-process
   memory.

2. A fixed **exploration phase** (first ``EXPLORATION_STEPS`` actions after
   each full-reset) that systematically sweeps the play area before handing
   control to the LLM-guided planner.  This populates the world model before
   the LLM ever sees it.

The :meth:`WorldModel.summary` is injected into every LLM system prompt so
the model does not have to re-derive object positions from raw grid values on
each turn.

Usage::

    python main.py --agent=worldmodelagent --game=locksmith

To run as the **Baseline agent** for comparison::

    python main.py --agent=reasoningagent --game=locksmith
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

from arcengine import FrameData, GameAction

from .reasoning_agent import ReasoningActionResponse, ReasoningAgent

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Grid integer constants  (Locksmith game colour encoding)
# ---------------------------------------------------------------------------
_INT_PLAYER: int = 4   # player body cells (dark grey)
_INT_DOOR: int = 11    # exit-door border (yellow)
_INT_ROTATOR: int = 9  # rotator body cells (blue)
_INT_ENERGY: int = 6   # energy pill cells / HUD energy bar (magenta)

# The play area is rows 0-55; rows 56-63 contain the HUD.
# _PLAY_AREA_ROW_LIMIT is the exclusive upper bound used in range() calls.
_PLAY_AREA_ROW_LIMIT: int = 56
# Avoid the leftmost and rightmost 4 cols to reduce edge artefacts.
_PLAY_AREA_COL_MIN: int = 4
_PLAY_AREA_COL_MAX: int = 60


# ---------------------------------------------------------------------------
# WorldModel
# ---------------------------------------------------------------------------

class WorldModel:
    """Structured representation of discovered game objects, updated each frame.

    The model is populated by :meth:`update`, which parses the raw integer
    grid.  All detection is deterministic (no LLM calls).  The compact text
    output from :meth:`summary` is injected into the LLM system prompt so the
    model always has up-to-date spatial context.
    """

    def __init__(self) -> None:
        # Discovered object positions
        self.player_pos: Optional[Tuple[int, int]] = None          # approx. centre
        self.door_pos: Optional[Tuple[int, int]] = None            # top-left corner
        self.rotator_positions: List[Tuple[int, int]] = []
        self.energy_pill_positions: List[Tuple[int, int]] = []

        # Exploration tracking
        self.visited_positions: Set[Tuple[int, int]] = set()
        self.failed_moves: List[str] = []   # e.g. "ACTION3 from (24, 32)"

        # HUD data
        self.last_energy: int = -1
        self.levels_completed: int = 0

        # Private: previous player pos for blocked-move detection
        self._prev_player_pos: Optional[Tuple[int, int]] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset all tracked state for a new game/level (call on full_reset)."""
        self.player_pos = None
        self.door_pos = None
        self.rotator_positions = []
        self.energy_pill_positions = []
        self.visited_positions = set()
        self.failed_moves = []
        self.last_energy = -1
        self.levels_completed = 0
        self._prev_player_pos = None

    def update(
        self,
        grid: List[List[int]],
        frame: FrameData,
        last_action: Optional[str] = None,
    ) -> None:
        """Parse *grid* (the last layer of the current frame) and refresh state.

        Args:
            grid: 2-D list of integers (64 × 64 for Locksmith).
            frame: The current :class:`FrameData` (provides ``levels_completed``).
            last_action: Name of the action taken to reach this frame, used to
                         detect blocked moves (e.g. ``"ACTION1"``).
        """
        if not grid or not grid[0]:
            return

        self._prev_player_pos = self.player_pos
        self.player_pos = self._find_player(grid)
        self.door_pos = self._find_door(grid)
        self.rotator_positions = self._find_rotators(grid)
        self.energy_pill_positions = self._find_energy_pills(grid)
        self.last_energy = self._read_energy(grid)
        self.levels_completed = frame.levels_completed

        if self.player_pos:
            self.visited_positions.add(self.player_pos)

        # Detect blocked moves: player position unchanged after a directional action
        if (
            last_action in ("ACTION1", "ACTION2", "ACTION3", "ACTION4")
            and self._prev_player_pos is not None
            and self.player_pos is not None
            and self._prev_player_pos == self.player_pos
        ):
            entry = f"{last_action} from {self._prev_player_pos}"
            if entry not in self.failed_moves:
                self.failed_moves.append(entry)

    def summary(self) -> str:
        """Return a compact, human-readable summary for the LLM system prompt."""
        lines: List[str] = ["## World Model (discovered objects)"]
        lines.append(
            f"Player centre (row, col): {self.player_pos or 'not yet located'}"
        )
        lines.append(
            f"Exit door top-left (row, col): {self.door_pos or 'not yet located'}"
        )
        if self.rotator_positions:
            lines.append(f"Rotators at: {self.rotator_positions}")
        else:
            lines.append("Rotators: none found yet — keep exploring")
        if self.energy_pill_positions:
            lines.append(f"Energy pills at: {self.energy_pill_positions}")
        else:
            lines.append("Energy pills: none found yet")
        if self.last_energy >= 0:
            lines.append(f"Remaining energy (HUD cells): {self.last_energy}")
        lines.append(f"Distinct positions visited: {len(self.visited_positions)}")
        if self.failed_moves:
            # Show only the 8 most-recent confirmed blocks to keep prompt short
            lines.append(f"Confirmed blocked moves: {self.failed_moves[-8:]}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable snapshot for the action ``reasoning`` field."""
        return {
            "player_pos": self.player_pos,
            "door_pos": self.door_pos,
            "rotator_positions": self.rotator_positions,
            "energy_pill_positions": self.energy_pill_positions,
            "visited_count": len(self.visited_positions),
            "failed_moves": self.failed_moves[-8:],
            "last_energy": self.last_energy,
            "levels_completed": self.levels_completed,
        }

    # ------------------------------------------------------------------
    # Object detectors (static, operate on a single grid snapshot)
    # ------------------------------------------------------------------

    @staticmethod
    def _find_player(grid: List[List[int]]) -> Optional[Tuple[int, int]]:
        """Find the player's approximate centre position.

        The player body is made of ``INT<4>`` cells.  Rotator objects also
        contain ``INT<4>`` cells but are always adjacent to ``INT<9>`` cells.
        We exclude a ±5-cell neighbourhood around every ``INT<9>`` cell before
        computing the centroid of the remaining ``INT<4>`` cells.
        """
        H, W = len(grid), len(grid[0])

        # Build exclusion zone around all rotator (INT<9>) cells
        rotator_zone: Set[Tuple[int, int]] = set()
        for r in range(min(H, _PLAY_AREA_ROW_LIMIT)):
            for c in range(W):
                if grid[r][c] == _INT_ROTATOR:
                    for dr in range(-5, 6):
                        for dc in range(-5, 6):
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < H and 0 <= nc < W:
                                rotator_zone.add((nr, nc))

        # Collect INT<4> cells in the central play area, outside rotator zones
        player_cells = [
            (r, c)
            for r in range(4, min(H, _PLAY_AREA_ROW_LIMIT))
            for c in range(_PLAY_AREA_COL_MIN, min(W, _PLAY_AREA_COL_MAX))
            if grid[r][c] == _INT_PLAYER and (r, c) not in rotator_zone
        ]

        if not player_cells:
            return None

        avg_r = sum(r for r, _ in player_cells) // len(player_cells)
        avg_c = sum(c for _, c in player_cells) // len(player_cells)
        return (avg_r, avg_c)

    @staticmethod
    def _find_door(grid: List[List[int]]) -> Optional[Tuple[int, int]]:
        """Find the exit door: a 4×4 block whose four corners are ``INT<11>``.

        Returns the top-left corner ``(row, col)`` of the first match found.
        """
        H, W = len(grid), len(grid[0])
        for r in range(min(H - 3, _PLAY_AREA_ROW_LIMIT)):
            for c in range(W - 3):
                if (
                    grid[r][c] == _INT_DOOR
                    and grid[r][c + 3] == _INT_DOOR
                    and grid[r + 3][c] == _INT_DOOR
                    and grid[r + 3][c + 3] == _INT_DOOR
                ):
                    return (r, c)
        return None

    @staticmethod
    def _find_rotators(grid: List[List[int]]) -> List[Tuple[int, int]]:
        """Find all rotator objects using flood-fill on ``INT<9>`` clusters.

        Returns one representative (top-left-most) position per cluster.
        """
        H, W = len(grid), len(grid[0])
        seen: Set[Tuple[int, int]] = set()
        positions: List[Tuple[int, int]] = []

        for r in range(min(H, _PLAY_AREA_ROW_LIMIT)):
            for c in range(W):
                if grid[r][c] != _INT_ROTATOR or (r, c) in seen:
                    continue
                # Flood-fill to collect the full cluster
                stack: List[Tuple[int, int]] = [(r, c)]
                cluster: List[Tuple[int, int]] = []
                while stack:
                    cr, cc = stack.pop()
                    if (cr, cc) in seen:
                        continue
                    seen.add((cr, cc))
                    cluster.append((cr, cc))
                    for nr, nc in (
                        (cr - 1, cc),
                        (cr + 1, cc),
                        (cr, cc - 1),
                        (cr, cc + 1),
                    ):
                        if (
                            0 <= nr < H
                            and 0 <= nc < W
                            and grid[nr][nc] == _INT_ROTATOR
                            and (nr, nc) not in seen
                        ):
                            stack.append((nr, nc))
                # Top-left-most cell is the canonical representative
                positions.append(min(cluster))

        return positions

    @staticmethod
    def _find_energy_pills(grid: List[List[int]]) -> List[Tuple[int, int]]:
        """Find 2×2 clusters of ``INT<6>`` in the play area (not the HUD).

        Returns top-left corners of each cluster.
        """
        H = min(len(grid), _PLAY_AREA_ROW_LIMIT)
        W = len(grid[0]) if grid else 0
        seen: Set[Tuple[int, int]] = set()
        positions: List[Tuple[int, int]] = []

        for r in range(H - 1):
            for c in range(W - 1):
                if (
                    (r, c) not in seen
                    and grid[r][c] == _INT_ENERGY
                    and grid[r][c + 1] == _INT_ENERGY
                    and grid[r + 1][c] == _INT_ENERGY
                    and grid[r + 1][c + 1] == _INT_ENERGY
                ):
                    positions.append((r, c))
                    seen.update(
                        [(r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1)]
                    )

        return positions

    @staticmethod
    def _read_energy(grid: List[List[int]]) -> int:
        """Count remaining energy from the HUD energy row (row 61).

        Each ``INT<6>`` cell in that row represents one unit of remaining
        energy.  Returns -1 if the grid has fewer than 62 rows.
        """
        if len(grid) <= 61:
            return -1
        return sum(1 for cell in grid[61] if cell == _INT_ENERGY)


# ---------------------------------------------------------------------------
# WorldModelAgent
# ---------------------------------------------------------------------------

class WorldModelAgent(ReasoningAgent):
    """A :class:`ReasoningAgent` enhanced with a world model and exploration.

    **Two new capabilities:**

    * **WorldModel** — every frame is parsed to extract player/door/rotator/
      energy-pill positions and blocked-move history.  This structured map is
      injected into the LLM system prompt via :meth:`build_user_prompt`.

    * **Exploration phase** — the first :attr:`EXPLORATION_STEPS` actions
      after each full-reset follow a predefined sweep pattern rather than LLM
      calls.  This populates the world model cheaply before the model
      commits to a strategy.

    **For comparison against the baseline** (:class:`ReasoningAgent`):

    * Run ``python main.py --agent=reasoningagent`` (baseline, no world model).
    * Run ``python main.py --agent=worldmodelagent`` (this agent).
    * Compare the resulting scorecards to measure the impact of the world model.
    """

    MAX_ACTIONS: int = 600
    MESSAGE_LIMIT: int = 12

    # Number of deterministic exploration steps after each full-reset
    EXPLORATION_STEPS: int = 12

    # 12-step sweep: 3 right → 2 down → 6 left → 1 up.
    # Covers a ≈16×16 cell rectangular sweep (player moves 4 cells per step).
    EXPLORATION_SEQUENCE: List[GameAction] = [
        GameAction.ACTION4,  # right
        GameAction.ACTION4,  # right
        GameAction.ACTION4,  # right
        GameAction.ACTION2,  # down
        GameAction.ACTION2,  # down
        GameAction.ACTION3,  # left
        GameAction.ACTION3,  # left
        GameAction.ACTION3,  # left
        GameAction.ACTION3,  # left
        GameAction.ACTION3,  # left
        GameAction.ACTION3,  # left
        GameAction.ACTION1,  # up
    ]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.world_model: WorldModel = WorldModel()
        self._last_action_name: Optional[str] = None

    # ------------------------------------------------------------------
    # History / reset
    # ------------------------------------------------------------------

    def clear_history(self) -> None:
        """Clear agent history and reset the world model for a new level."""
        super().clear_history()
        self.world_model.reset()
        self._last_action_name = None

    # ------------------------------------------------------------------
    # Core action loop
    # ------------------------------------------------------------------

    def choose_action(
        self, frames: List[FrameData], latest_frame: FrameData
    ) -> GameAction:
        """Choose the next action.

        Sequence:
        1. Update the world model from the incoming frame.
        2. If ``full_reset``, clear history/model and return RESET.
        3. If history is empty (very first action), return RESET.
        4. During the exploration phase, return the next predefined move.
        5. Otherwise delegate to the parent LLM planner
           (:meth:`ReasoningAgent.define_next_action`), whose system-prompt is
           augmented by :meth:`build_user_prompt`.
        """
        # 1. Update world model before any decision
        if latest_frame.frame:
            self.world_model.update(
                latest_frame.frame[-1],
                latest_frame,
                self._last_action_name,
            )

        # 2. Handle full level/game reset
        if latest_frame.full_reset:
            self.clear_history()
            self._last_action_name = "RESET"
            return GameAction.RESET

        # 3. First action: always RESET
        if not self.history:
            self.history.append(
                ReasoningActionResponse(
                    name="RESET",
                    reason="Initial RESET to start the game.",
                    short_description="Start game",
                    hypothesis="The game requires a RESET to begin.",
                    aggregated_findings="No findings yet.",
                )
            )
            self._last_action_name = "RESET"
            return GameAction.RESET

        # 4. Exploration phase
        # len(history) == 1 after the initial RESET entry; exploration starts
        # at index 0 of EXPLORATION_SEQUENCE.
        exploration_step = len(self.history) - 1  # 0-indexed
        if exploration_step < self.EXPLORATION_STEPS:
            action = self.EXPLORATION_SEQUENCE[exploration_step]
            self.history.append(
                ReasoningActionResponse(
                    name=action.name,
                    reason=(
                        f"Exploration step {exploration_step + 1}/"
                        f"{self.EXPLORATION_STEPS}: systematic sweep to map "
                        "the environment."
                    ),
                    short_description=f"Explore step {exploration_step + 1}",
                    hypothesis=(
                        "Systematic grid exploration builds the world model "
                        "before committing to LLM-guided strategy."
                    ),
                    aggregated_findings=self.world_model.summary(),
                )
            )
            action.reasoning = {
                "phase": "exploration",
                "step": exploration_step,
                "world_model": self.world_model.to_dict(),
            }
            self._last_action_name = action.name
            return action

        # 5. LLM phase — delegate to ReasoningAgent.define_next_action.
        # build_user_prompt (overridden below) injects the world model summary.
        action_response = self.define_next_action(latest_frame)
        self.history.append(action_response)
        action = GameAction.from_name(action_response.name)
        action.reasoning = {
            "model": self.MODEL,
            "reasoning_effort": self.REASONING_EFFORT,
            "reasoning_tokens": self._last_reasoning_tokens,
            "total_reasoning_tokens": self._total_reasoning_tokens,
            "agent_type": "world_model_agent",
            "hypothesis": action_response.hypothesis,
            "aggregated_findings": action_response.aggregated_findings,
            "world_model": self.world_model.to_dict(),
            "response_preview": (
                action_response.reason[:200] + "..."
                if len(action_response.reason) > 200
                else action_response.reason
            ),
            "action_chosen": action.name,
            "game_context": {
                "score": latest_frame.levels_completed,
                "state": latest_frame.state.name,
                "action_counter": self.action_counter,
                "frame_count": len(frames),
            },
        }
        self._last_action_name = action.name
        return action

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    def build_user_prompt(self, latest_frame: FrameData) -> str:
        """Augment the base prompt with the current world model summary.

        The world model summary is appended as a structured section so the
        LLM always has accurate object positions without having to re-parse
        the raw integer grid.
        """
        base_prompt = super().build_user_prompt(latest_frame)
        world_model_section = "\n\n" + self.world_model.summary() + (
            "\n\nUse the world model above to plan efficiently. "
            "Navigate towards the exit door, rotating the key as needed. "
            "Prefer unvisited directions. "
            "Avoid positions listed under confirmed blocked moves."
        )
        return base_prompt + world_model_section
