"""the persistence in memory — hashed-frame silo agent for ARC-AGI-3."""
from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from typing import Any

from arcengine import FrameData, GameAction, GameState

from ..agent import Agent


class PersistenceInMemory(Agent):
    """PMLL-style short-term silo: hash frames, prefer novel moves, click components.

    Competition-mode public cards:
    - https://arcprize.org/scorecards/fa62e88d-607e-402d-91d4-ca61ad597cab
    - https://arcprize.org/scorecards/6424f517-8080-4c22-8039-accb5bf5877e
    Community: https://github.com/arcprize/ARC-AGI-Community-Leaderboard/pull/51
    """

    MAX_ACTIONS = 180

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.tried: set[tuple[str, str, str]] = set()
        self.click_hits: list[tuple[int, int]] = []

    @property
    def name(self) -> str:
        return f"{super().name}.silo"

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state is GameState.WIN

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            action = GameAction.RESET
            action.reasoning = "level reset after GAME_OVER / not played"
            return action

        grid = _last_grid(latest_frame.frame)
        fh = _frame_hash(grid)
        avail = list(latest_frame.available_actions or [])
        action_ids = [int(a) for a in avail if int(a) != 0]

        if 6 in action_ids:
            for x, y in _click_targets(grid, self.click_hits):
                extra = f"{x},{y}"
                if (fh, "ACTION6", extra) not in self.tried:
                    self.tried.add((fh, "ACTION6", extra))
                    action = GameAction.ACTION6
                    action.set_data({"x": x, "y": y})
                    action.reasoning = {"policy": "component-click", "x": x, "y": y}
                    self.click_hits.append((x, y))
                    if len(self.click_hits) > 40:
                        self.click_hits = self.click_hits[-40:]
                    return action

        keys = [a for a in action_ids if a in (1, 2, 3, 4, 5, 7)]
        random.shuffle(keys)
        for a in keys:
            name = f"ACTION{a}"
            if (fh, name, "") not in self.tried:
                self.tried.add((fh, name, ""))
                action = GameAction.from_id(a)
                action.reasoning = {"policy": "novel-simple", "action": name}
                return action

        action = GameAction.from_id(random.choice(action_ids) if action_ids else 1)
        if action.is_complex():
            action.set_data({"x": random.randint(0, 63), "y": random.randint(0, 63)})
        action.reasoning = {"policy": "fallback"}
        return action


def _last_grid(frame: Any) -> list[list[int]]:
    if not frame:
        return []
    g = frame
    if isinstance(frame, list) and frame and isinstance(frame[0], list) and frame[0] and isinstance(frame[0][0], list):
        g = frame[-1]
    return g or []


def _frame_hash(grid: list[list[int]]) -> str:
    raw = json.dumps(grid, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def _click_targets(grid: list[list[int]], hits: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not grid or not grid[0]:
        return [(random.randint(8, 55), random.randint(8, 55))]
    counts: dict[int, int] = Counter(int(v) for row in grid for v in row)
    bg = max(counts, key=counts.get) if counts else 0
    h, w = len(grid), len(grid[0])
    seen = [[False] * w for _ in range(h)]
    pts: list[tuple[int, int]] = []
    for y in range(h):
        for x in range(w):
            if seen[y][x] or int(grid[y][x]) == bg:
                continue
            color = int(grid[y][x])
            stack = [(x, y)]
            seen[y][x] = True
            cells = []
            while stack:
                cx, cy = stack.pop()
                cells.append((cx, cy))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < w and 0 <= ny < h and not seen[ny][nx] and int(grid[ny][nx]) == color:
                        seen[ny][nx] = True
                        stack.append((nx, ny))
            pts.append((sum(c[0] for c in cells) // len(cells), sum(c[1] for c in cells) // len(cells)))
    pts.extend(hits[-8:])
    random.shuffle(pts)
    return pts or [(32, 32)]
