"""
MazeRunnerGame — emulated offline environment for ARC-AGI-3 Option D.

Design
------
A three-level maze-navigation puzzle.  Rather than attempting to track
real sprite collision, this implementation uses a *step-counter* model:
each level is beaten after a fixed number of non-reset action presses.

This guarantees deterministic wins regardless of what actions the agent
sends, which is exactly what the offline scoring demo requires.

Level structure
---------------
  Level 0 — 6 actions to complete  (short corridor)
  Level 1 — 12 actions to complete (medium path)
  Level 2 — 10 actions to complete (final room)
  win_levels = 3

Camera / display
----------------
64×64 pixels divided into 8×8 cells of 8 px each.
  Colour 0  = floor (white)
  Colour 5  = wall  (black)
  Colour 9  = player (blue)  — moves right one cell per action
  Colour 6  = goal  (pink)   — fixed at cell (7, row)
"""

from __future__ import annotations

from typing import List

import numpy as np

from arcengine import ARCBaseGame, GameState
from arcengine.camera import Camera
from arcengine.enums import ActionInput, BlockingMode, GameAction
from arcengine.level import Level
from arcengine.sprites import Sprite

# ── constants ─────────────────────────────────────────────────────────────
CELL   = 8          # pixels per grid cell
COLS   = 8          # grid columns
ROWS   = 8          # grid rows
FLOOR  = 0
WALL   = 5
PLAYER = 9
GOAL   = 6

# Number of action presses to clear each level
_STEPS_TO_WIN = [6, 12, 10]

# Player start column (grid units); goal is always at column COLS-1
_PLAYER_START_COL = 1

# Row the player/goal sprite live on
_PLAYER_ROW = 3


def _make_sprite(col: int, row: int, colour: int, name: str, layer: int = 1) -> Sprite:
    return Sprite(
        pixels=[[colour]],
        name=name,
        x=col * CELL,
        y=row * CELL,
        scale=CELL,
        layer=layer,
        blocking=BlockingMode.NOT_BLOCKED,
        tags=[name],
    )


def _make_floor_row(row: int) -> List[Sprite]:
    return [
        _make_sprite(col, row, FLOOR, f"floor_{col}_{row}", layer=0)
        for col in range(COLS)
    ]


def _build_level(level_idx: int) -> Level:
    """Build the visual layout for one level."""
    sprites: List[Sprite] = []

    # Wall rows (top and bottom)
    for col in range(COLS):
        sprites.append(_make_sprite(col, 0,        WALL,  f"wall_top_{col}",    layer=0))
        sprites.append(_make_sprite(col, ROWS - 1, WALL,  f"wall_bot_{col}",    layer=0))
        sprites.append(_make_sprite(0,   col,      WALL,  f"wall_left_{col}",   layer=0))
        sprites.append(_make_sprite(COLS - 1, col, WALL,  f"wall_right_{col}",  layer=0))

    # Floor rows in the middle
    for row in range(1, ROWS - 1):
        sprites.extend(_make_floor_row(row))

    # Goal (fixed position)
    sprites.append(_make_sprite(COLS - 2, _PLAYER_ROW, GOAL, "goal", layer=2))

    # Player start
    sprites.append(
        _make_sprite(_PLAYER_START_COL, _PLAYER_ROW, PLAYER, "player", layer=3)
    )

    return Level(
        sprites=sprites,
        grid_size=(COLS * CELL, ROWS * CELL),
        name=f"Maze-L{level_idx}",
    )


class MazeRunnerGame(ARCBaseGame):
    """
    Three-level maze-runner game for offline ARC-AGI-3 scoring demos.

    Win condition per level: press any non-RESET action _STEPS_TO_WIN[level]
    times.  On the final level, win() is called automatically.
    """

    def __init__(self, seed: int = 0) -> None:
        levels = [_build_level(i) for i in range(len(_STEPS_TO_WIN))]
        camera = Camera(width=COLS * CELL, height=ROWS * CELL, background=FLOOR)

        super().__init__(
            game_id="maze-runner-v1",
            levels=levels,
            camera=camera,
            win_score=len(levels),
            available_actions=[1, 2, 3, 4],
            seed=seed,
        )
        self._level_step_count: int = 0
        self._player_col: int = _PLAYER_START_COL

    # ── ARCBaseGame interface ──────────────────────────────────────────────

    def on_set_level(self, level: Level) -> None:
        """Reset per-level counters when the engine advances to a new level."""
        self._level_step_count = 0
        self._player_col       = _PLAYER_START_COL
        # Snap player sprite to start position
        for sp in level.get_sprites_by_tag("player"):
            sp.set_position(_PLAYER_START_COL * CELL, _PLAYER_ROW * CELL)

    def step(self) -> None:
        action_id = self.action.id.value

        if action_id == GameAction.RESET.value:
            self.handle_reset()
            self.complete_action()
            return

        # Advance player one cell to the right (clamped before the right wall)
        max_col = COLS - 2
        if self._player_col < max_col:
            self._player_col += 1
            for sp in self.current_level.get_sprites_by_tag("player"):
                sp.set_position(self._player_col * CELL, _PLAYER_ROW * CELL)

        self._level_step_count += 1
        target = _STEPS_TO_WIN[self._current_level_index]

        if self._level_step_count >= target:
            self.next_level()   # increments _score; calls win() on last level

        self.complete_action()
