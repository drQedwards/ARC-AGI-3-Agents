"""
ColorSortGame — emulated offline environment for ARC-AGI-3 Option D.

Concept
-------
Two levels of a colour-sorting puzzle.  An array of coloured tiles must be
sorted in ascending order.  Each ACTION1 press performs one bubble-sort
pass.  When the array is already sorted the level is won.

Level 0 — tiles [3, 1, 4, 2] — sorted in ≤ 3 passes
Level 1 — tiles [6, 2, 5, 1, 4, 3] — sorted in ≤ 5 passes

Display
-------
8×8 grid (64×64 px camera).  Tiles appear as coloured cells in row 3.
Row 0 / row 7 are black walls; remaining rows are dark floor.

Colour index → display colour (arcengine palette):
  0 → white   (blank / inactive tiles)
  1 → light   (tile value 1)
  2 → dark grey (tile value 2)
  3 → mid-grey (tile value 3)
  4 → charcoal (tile value 4)
  5 → black   (walls)
  6 → pink    (tile value 6)

Actions
-------
  ACTION1 → one bubble-sort pass (sorted state → win level)
  All other non-RESET → same as ACTION1
  RESET   → reset current level / full reset
"""

from __future__ import annotations

from typing import List

from arcengine import ARCBaseGame, GameState
from arcengine.camera import Camera
from arcengine.enums import ActionInput, BlockingMode, GameAction
from arcengine.level import Level
from arcengine.sprites import Sprite

TILE_PX    = 8
GRID_SIZE  = 8     # 8×8 grid
WALL_COLOUR  = 5
FLOOR_COLOUR = 1

_INITIAL_TILES: List[List[int]] = [
    [3, 1, 4, 2, 0, 0, 0, 0],        # level 0 — 4 active tiles
    [6, 2, 5, 1, 4, 3, 0, 0],        # level 1 — 6 active tiles
]
_N_ACTIVE = [4, 6]
_TILE_ROW  = 3


def _make_cell(col: int, row: int, colour: int, name: str, layer: int = 0) -> Sprite:
    return Sprite(
        pixels=[[colour]],
        name=name,
        x=col * TILE_PX,
        y=row * TILE_PX,
        scale=TILE_PX,
        layer=layer,
        blocking=BlockingMode.NOT_BLOCKED,
        tags=[name.split("_")[0]],
    )


def _build_level(level_idx: int) -> Level:
    sprites: List[Sprite] = []

    # Border walls
    for i in range(GRID_SIZE):
        sprites.append(_make_cell(i, 0,              WALL_COLOUR,  f"wall_top_{i}"))
        sprites.append(_make_cell(i, GRID_SIZE - 1,  WALL_COLOUR,  f"wall_bot_{i}"))
        sprites.append(_make_cell(0, i,              WALL_COLOUR,  f"wall_lft_{i}"))
        sprites.append(_make_cell(GRID_SIZE - 1, i,  WALL_COLOUR,  f"wall_rgt_{i}"))

    # Floor
    for row in range(1, GRID_SIZE - 1):
        for col in range(1, GRID_SIZE - 1):
            sprites.append(_make_cell(col, row, FLOOR_COLOUR, f"floor_{col}_{row}"))

    # Tile row
    for col, colour in enumerate(_INITIAL_TILES[level_idx]):
        sprites.append(
            _make_cell(col, _TILE_ROW, colour, f"tile_{col}", layer=2)
        )

    return Level(
        sprites=sprites,
        grid_size=(GRID_SIZE * TILE_PX, GRID_SIZE * TILE_PX),
        name=f"ColorSort-L{level_idx}",
    )


class ColorSortGame(ARCBaseGame):
    """
    Two-level colour-sort puzzle for offline ARC-AGI-3 scoring demos.

    Win condition: sort the active tiles in ascending order using
    bubble-sort passes triggered by ACTION1.
    """

    def __init__(self, seed: int = 0) -> None:
        levels = [_build_level(i) for i in range(len(_INITIAL_TILES))]
        camera = Camera(
            width=GRID_SIZE * TILE_PX,
            height=GRID_SIZE * TILE_PX,
            background=FLOOR_COLOUR,
        )
        super().__init__(
            game_id="color-sort-v1",
            levels=levels,
            camera=camera,
            win_score=len(levels),
            available_actions=[1],
            seed=seed,
        )
        self._tiles:    List[int] = []
        self._n_active: int = 4

    def on_set_level(self, level: Level) -> None:
        idx = self._current_level_index
        self._tiles    = list(_INITIAL_TILES[idx])
        self._n_active = _N_ACTIVE[idx]

    def _refresh_tile_sprites(self) -> None:
        """Rebuild tile sprites to reflect current sort state."""
        for sp in self.current_level.get_sprites_by_tag("tile"):
            self.current_level.remove_sprite(sp)
        for col, colour in enumerate(self._tiles):
            self.current_level.add_sprite(
                _make_cell(col, _TILE_ROW, colour, f"tile_{col}", layer=2)
            )

    def step(self) -> None:
        action_id = self.action.id.value

        if action_id == GameAction.RESET.value:
            self.handle_reset()
            self.complete_action()
            return

        # One bubble-sort pass over active tiles
        n = self._n_active
        for i in range(n - 1):
            if self._tiles[i] > self._tiles[i + 1]:
                self._tiles[i], self._tiles[i + 1] = self._tiles[i + 1], self._tiles[i]

        self._refresh_tile_sprites()

        if self._tiles[:n] == sorted(self._tiles[:n]):
            self.next_level()

        self.complete_action()
