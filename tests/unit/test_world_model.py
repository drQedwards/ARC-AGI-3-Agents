"""Unit tests for WorldModel and WorldModelAgent.

These tests exercise the deterministic grid-parsing logic in WorldModel and
the exploration/action-selection logic in WorldModelAgent without requiring
a live API connection or any LLM calls.
"""

import pytest
from arcengine import FrameData, GameAction, GameState

from agents.templates.world_model_agent import (
    WorldModel,
    WorldModelAgent,
    _INT_DOOR,
    _INT_ENERGY,
    _INT_PLAYER,
    _INT_ROTATOR,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _empty_grid(rows: int = 64, cols: int = 64, fill: int = 8) -> list[list[int]]:
    """Return a solid-fill grid (default: all floor INT<8>)."""
    return [[fill] * cols for _ in range(rows)]


def _make_frame(
    grid: list[list[int]] | None = None,
    state: GameState = GameState.NOT_FINISHED,
    levels_completed: int = 0,
    full_reset: bool = False,
) -> FrameData:
    """Convenience factory for FrameData."""
    g = grid if grid is not None else _empty_grid()
    return FrameData(
        game_id="test-game",
        frame=[g],
        state=state,
        levels_completed=levels_completed,
        full_reset=full_reset,
    )


# ---------------------------------------------------------------------------
# WorldModel — unit tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestWorldModelInit:
    def test_initial_state(self) -> None:
        wm = WorldModel()
        assert wm.player_pos is None
        assert wm.door_pos is None
        assert wm.rotator_positions == []
        assert wm.energy_pill_positions == []
        assert wm.visited_positions == set()
        assert wm.failed_moves == []
        assert wm.last_energy == -1
        assert wm.levels_completed == 0

    def test_reset_clears_all_state(self) -> None:
        wm = WorldModel()
        wm.player_pos = (10, 20)
        wm.door_pos = (5, 40)
        wm.rotator_positions = [(8, 8)]
        wm.energy_pill_positions = [(20, 20)]
        wm.visited_positions = {(10, 20), (14, 20)}
        wm.failed_moves = ["ACTION1 from (10, 20)"]
        wm.last_energy = 15
        wm.levels_completed = 2
        wm._prev_player_pos = (10, 20)

        wm.reset()

        assert wm.player_pos is None
        assert wm.door_pos is None
        assert wm.rotator_positions == []
        assert wm.energy_pill_positions == []
        assert wm.visited_positions == set()
        assert wm.failed_moves == []
        assert wm.last_energy == -1
        assert wm.levels_completed == 0
        assert wm._prev_player_pos is None


@pytest.mark.unit
class TestWorldModelDetection:
    def test_find_player_single_cluster(self) -> None:
        """Player-body INT<4> cells at a known location should be found."""
        grid = _empty_grid()
        # Place a 3×3 player body at row 20, col 30
        for dr in range(3):
            for dc in range(3):
                grid[20 + dr][30 + dc] = _INT_PLAYER

        pos = WorldModel._find_player(grid)
        assert pos is not None
        # Centre of the 3×3 block should be (21, 31)
        assert pos == (21, 31)

    def test_find_player_ignores_rotator_proximity(self) -> None:
        """INT<4> cells adjacent to INT<9> (rotator) should not be the player."""
        grid = _empty_grid()
        # Rotator at (10, 10) — also place INT<4> nearby (within exclusion zone)
        grid[10][10] = _INT_ROTATOR
        grid[10][11] = _INT_PLAYER  # within ±5 of rotator → excluded

        pos = WorldModel._find_player(grid)
        # No INT<4> outside the rotator exclusion zone → None
        assert pos is None

    def test_find_player_prefers_cells_outside_rotator_zone(self) -> None:
        """Player cells far from a rotator should still be detected."""
        grid = _empty_grid()
        grid[10][10] = _INT_ROTATOR  # rotator
        # Place player far from rotator
        for dr in range(3):
            for dc in range(3):
                grid[40 + dr][40 + dc] = _INT_PLAYER

        pos = WorldModel._find_player(grid)
        assert pos is not None
        assert pos == (41, 41)

    def test_find_player_returns_none_on_empty_grid(self) -> None:
        grid = _empty_grid(fill=8)  # no INT<4> cells
        assert WorldModel._find_player(grid) is None

    def test_find_door_detects_four_corners(self) -> None:
        """Four INT<11> corners in a 4×4 pattern should be found."""
        grid = _empty_grid()
        r, c = 15, 25
        grid[r][c] = _INT_DOOR
        grid[r][c + 3] = _INT_DOOR
        grid[r + 3][c] = _INT_DOOR
        grid[r + 3][c + 3] = _INT_DOOR

        door = WorldModel._find_door(grid)
        assert door == (r, c)

    def test_find_door_returns_none_when_absent(self) -> None:
        grid = _empty_grid()
        assert WorldModel._find_door(grid) is None

    def test_find_rotators_single_cluster(self) -> None:
        """A connected cluster of INT<9> cells is reported as one rotator."""
        grid = _empty_grid()
        for dr in range(3):
            for dc in range(3):
                grid[8 + dr][8 + dc] = _INT_ROTATOR

        rotators = WorldModel._find_rotators(grid)
        assert len(rotators) == 1
        assert rotators[0] == (8, 8)

    def test_find_rotators_two_separate_clusters(self) -> None:
        """Two disconnected INT<9> clusters produce two entries."""
        grid = _empty_grid()
        grid[8][8] = _INT_ROTATOR
        grid[30][40] = _INT_ROTATOR

        rotators = WorldModel._find_rotators(grid)
        assert len(rotators) == 2

    def test_find_energy_pills_2x2_cluster(self) -> None:
        """A 2×2 block of INT<6> in the play area is one energy pill."""
        grid = _empty_grid()
        r, c = 20, 20
        grid[r][c] = _INT_ENERGY
        grid[r][c + 1] = _INT_ENERGY
        grid[r + 1][c] = _INT_ENERGY
        grid[r + 1][c + 1] = _INT_ENERGY

        pills = WorldModel._find_energy_pills(grid)
        assert len(pills) == 1
        assert pills[0] == (r, c)

    def test_find_energy_pills_ignores_hud_row(self) -> None:
        """INT<6> cells in the HUD rows (56-63) should NOT be returned as a pill.

        A 2×2 block that would straddle rows 55-56 is not fully within the play
        area (rows 0-55) so it should not appear in the result.
        """
        grid = _empty_grid()
        # Place 2×2 straddling the play-area / HUD boundary (rows 55-56)
        grid[55][10] = _INT_ENERGY
        grid[55][11] = _INT_ENERGY
        grid[56][10] = _INT_ENERGY  # row 56 is the first HUD row — excluded
        grid[56][11] = _INT_ENERGY

        pills = WorldModel._find_energy_pills(grid)
        assert (55, 10) not in pills

    def test_read_energy_counts_int6_in_row_61(self) -> None:
        """Energy reading should count INT<6> cells in row 61."""
        grid = _empty_grid()
        for i in range(18):
            grid[61][i] = _INT_ENERGY  # 18 units of energy

        energy = WorldModel._read_energy(grid)
        assert energy == 18

    def test_read_energy_returns_minus1_on_short_grid(self) -> None:
        short_grid = _empty_grid(rows=50)
        assert WorldModel._read_energy(short_grid) == -1


@pytest.mark.unit
class TestWorldModelUpdate:
    def test_update_sets_levels_completed(self) -> None:
        wm = WorldModel()
        grid = _empty_grid()
        frame = _make_frame(grid=grid, levels_completed=3)
        wm.update(grid, frame)
        assert wm.levels_completed == 3

    def test_update_tracks_visited_positions(self) -> None:
        wm = WorldModel()
        grid = _empty_grid()
        # Plant a player cluster so player_pos is found
        for dr in range(3):
            for dc in range(3):
                grid[20 + dr][20 + dc] = _INT_PLAYER

        frame = _make_frame(grid=grid)
        wm.update(grid, frame)
        assert len(wm.visited_positions) == 1

    def test_update_detects_blocked_move(self) -> None:
        """If the player position does not change after a directional action,
        the move is recorded as blocked."""
        wm = WorldModel()
        grid = _empty_grid()
        for dr in range(3):
            for dc in range(3):
                grid[20 + dr][20 + dc] = _INT_PLAYER

        frame = _make_frame(grid=grid)

        # First update to set player pos
        wm.update(grid, frame, last_action=None)
        first_pos = wm.player_pos

        # Second update with same grid (player didn't move) after ACTION1
        wm.update(grid, frame, last_action="ACTION1")

        assert first_pos == wm.player_pos
        assert len(wm.failed_moves) == 1
        assert "ACTION1" in wm.failed_moves[0]

    def test_update_no_false_blocked_on_reset(self) -> None:
        """RESET action should not trigger a blocked-move record."""
        wm = WorldModel()
        grid = _empty_grid()
        for dr in range(3):
            for dc in range(3):
                grid[20 + dr][20 + dc] = _INT_PLAYER

        frame = _make_frame(grid=grid)
        wm.update(grid, frame, last_action=None)
        wm.update(grid, frame, last_action="RESET")

        assert wm.failed_moves == []

    def test_update_no_op_on_empty_grid(self) -> None:
        """An empty grid should not crash and should not change state."""
        wm = WorldModel()
        frame = _make_frame(grid=[[]])
        wm.update([[]], frame)
        assert wm.player_pos is None

    def test_summary_contains_key_fields(self) -> None:
        wm = WorldModel()
        grid = _empty_grid()
        # Plant detectable objects
        for dr in range(3):
            for dc in range(3):
                grid[20 + dr][20 + dc] = _INT_PLAYER
        r, c = 10, 40
        grid[r][c] = _INT_DOOR
        grid[r][c + 3] = _INT_DOOR
        grid[r + 3][c] = _INT_DOOR
        grid[r + 3][c + 3] = _INT_DOOR

        frame = _make_frame(grid=grid, levels_completed=1)
        wm.update(grid, frame)

        text = wm.summary()
        assert "Player centre" in text
        assert "Exit door" in text
        assert "Distinct positions visited" in text


@pytest.mark.unit
class TestWorldModelAgent:
    def _make_agent(self) -> WorldModelAgent:
        return WorldModelAgent(
            card_id="test-card",
            game_id="test-game",
            agent_name="test-agent",
            ROOT_URL="https://example.com",
            record=False,
            arc_env=None,  # type: ignore[arg-type]
        )

    def test_agent_init(self) -> None:
        agent = self._make_agent()

        assert agent.MAX_ACTIONS == 600
        assert agent.MESSAGE_LIMIT == 12
        assert agent.EXPLORATION_STEPS == 12
        assert isinstance(agent.world_model, WorldModel)
        assert agent._last_action_name is None

    def test_exploration_sequence_length(self) -> None:
        agent = self._make_agent()
        assert len(agent.EXPLORATION_SEQUENCE) == agent.EXPLORATION_STEPS

    def test_first_action_is_reset(self) -> None:
        agent = self._make_agent()
        frame = _make_frame(state=GameState.NOT_PLAYED)
        action = agent.choose_action([frame], frame)
        assert action == GameAction.RESET

    def test_full_reset_returns_reset_and_clears_history(self) -> None:
        agent = self._make_agent()
        # Seed some history using valid string lengths
        from agents.templates.reasoning_agent import ReasoningActionResponse

        agent.history.append(
            ReasoningActionResponse(
                name="RESET",
                reason="Resetting to start a new game.",
                short_description="Start game",
                hypothesis="The game requires a RESET to begin.",
                aggregated_findings="No findings yet available.",
            )
        )
        agent.world_model.player_pos = (10, 10)

        frame = _make_frame(full_reset=True)
        action = agent.choose_action([frame], frame)

        assert action == GameAction.RESET
        assert agent.history == []
        assert agent.world_model.player_pos is None

    def test_exploration_phase_uses_sequence(self) -> None:
        """After the initial RESET, the agent should emit exploration moves."""
        agent = self._make_agent()
        # Trigger initial RESET
        frame = _make_frame(state=GameState.NOT_PLAYED)
        agent.choose_action([frame], frame)  # action 0 → RESET

        # Next EXPLORATION_STEPS actions must match the sequence
        active_frame = _make_frame(state=GameState.NOT_FINISHED)
        for idx, expected_action in enumerate(agent.EXPLORATION_SEQUENCE):
            action = agent.choose_action([active_frame], active_frame)
            assert action == expected_action, (
                f"Exploration step {idx}: expected {expected_action.name}, "
                f"got {action.name}"
            )

    def test_exploration_reasoning_metadata(self) -> None:
        """Exploration actions should carry a 'phase' key in their reasoning."""
        agent = self._make_agent()
        frame = _make_frame(state=GameState.NOT_PLAYED)
        agent.choose_action([frame], frame)  # initial RESET

        active = _make_frame(state=GameState.NOT_FINISHED)
        action = agent.choose_action([active], active)
        assert action.reasoning is not None
        assert action.reasoning.get("phase") == "exploration"

    def test_exploration_step_recorded_in_history(self) -> None:
        agent = self._make_agent()
        frame_np = _make_frame(state=GameState.NOT_PLAYED)
        agent.choose_action([frame_np], frame_np)  # RESET

        active = _make_frame(state=GameState.NOT_FINISHED)
        agent.choose_action([active], active)  # exploration step 1

        # history: [RESET entry, exploration_step_1 entry]
        assert len(agent.history) == 2
        assert agent.history[1].name == agent.EXPLORATION_SEQUENCE[0].name

    def test_clear_history_resets_world_model(self) -> None:
        agent = self._make_agent()
        agent.world_model.player_pos = (5, 5)
        agent.world_model.failed_moves = ["ACTION3 from (5, 5)"]
        agent.clear_history()
        assert agent.world_model.player_pos is None
        assert agent.world_model.failed_moves == []

    def test_world_model_in_available_agents(self) -> None:
        """WorldModelAgent must be accessible via the AVAILABLE_AGENTS registry."""
        from agents import AVAILABLE_AGENTS, WorldModelAgent as WMA

        assert "worldmodelagent" in AVAILABLE_AGENTS
        assert AVAILABLE_AGENTS["worldmodelagent"] is WMA

    def test_build_user_prompt_includes_world_model(self) -> None:
        """The augmented system prompt must contain the world model summary."""
        agent = self._make_agent()
        frame = _make_frame()
        prompt = agent.build_user_prompt(frame)
        assert "World Model" in prompt
