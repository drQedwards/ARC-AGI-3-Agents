"""Language of Thought (LoT) Agent for ARC-AGI-3.

Two-phase reasoning approach:
  Phase 1 (Thought Synthesis): Generate a structured symbolic thought program
    from the current frame using gpt-4o-mini (fast, cheap).
  Phase 2 (Action Selection): Use the synthesised program + accumulated theory
    to select an action via o4-mini tool calls.

The agent maintains a growing `theory` string across turns — a formal program
body that gets refined each turn as new observations accumulate.
"""

import logging
import os
import textwrap
from typing import Any, Optional

from arcengine import FrameData, GameAction
from openai import OpenAI as OpenAIClient

from .llm_agents import ReasoningLLM

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt constants
# ---------------------------------------------------------------------------

_SYNTHESIS_SYSTEM = (
    "You are a logical thought synthesizer for ARC-AGI-3 games. "
    "Output ONLY a structured thought program — no prose, no markdown fences."
)

_SYNTHESIS_USER_TMPL = textwrap.dedent(
    """\
    # Current Frame (state={state}, score={score})
    {frame}

    # Accumulated Theory (last 1500 chars)
    {theory}

    Produce a thought program in this exact format:

    OBSERVE:
      grid_size = (H, W)
      player_pos = (approx_x, approx_y) or unknown
      dominant_color = N
      nonzero_cells = K
      notable = [list of observations]

    RULES:
      <if-then rules about the game mechanics learned so far; update/refine from prior theory>

    PLAN:
      IF <condition> THEN <ACTION1|ACTION2|ACTION3|ACTION4|ACTION5|RESET>
      ELIF <condition> THEN <action>
      ELSE <default_action>
    """
)

_ACTION_USER_TMPL = textwrap.dedent(
    """\
    # CONTEXT
    You are an agent playing an ARC-AGI-3 game. Your objective is to WIN and
    avoid GAME_OVER while minimising actions.

    One action produces one Frame. One Frame is made of one or more sequential
    Grids. Each Grid is a matrix of INT<0,63> x INT<0,63> filled with
    INT<0,15> values.

    # CURRENT THOUGHT PROGRAM
    {thought_program}

    # ACCUMULATED THEORY (last 1500 chars)
    {theory}

    # TURN
    Call exactly one action.
    """
)


class LanguageOfThought(ReasoningLLM):
    """Language of Thought agent — two-phase: synthesise thought program, then act.

    Phase 1 uses gpt-4o-mini to produce a structured OBSERVE / RULES / PLAN
    program each turn. Phase 2 passes that program plus the rolling theory
    context to o4-mini for tool-call action selection.
    """

    MAX_ACTIONS: int = 400
    DO_OBSERVATION: bool = False   # we handle our own two-phase observation
    MODEL: str = "o4-mini"
    MODEL_REQUIRES_TOOLS: bool = True
    MESSAGE_LIMIT: int = 8
    REASONING_EFFORT: Optional[str] = "medium"

    # Model used only for fast/cheap thought synthesis (Phase 1)
    SYNTHESIS_MODEL: str = "gpt-4o-mini"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.theory: str = ""               # accumulated program theory
        self.lot_history: list[dict] = []   # per-turn thought programs + metadata

    # ------------------------------------------------------------------
    # Phase 1: Thought Synthesis
    # ------------------------------------------------------------------

    def _synthesize_thought(self, latest_frame: FrameData) -> str:
        """Call gpt-4o-mini to produce a structured thought program for this frame.

        Returns the raw thought-program string (plain text, no tool calls).
        """
        client = OpenAIClient(api_key=os.environ.get("OPENAI_API_KEY", ""))

        frame_text = self.pretty_print_3d(latest_frame.frame)
        theory_snippet = self.theory[-1500:] if self.theory else "(none yet)"

        user_content = _SYNTHESIS_USER_TMPL.format(
            state=latest_frame.state.name,
            score=latest_frame.levels_completed,
            frame=frame_text,
            theory=theory_snippet,
        )

        try:
            response = client.chat.completions.create(
                model=self.SYNTHESIS_MODEL,
                messages=[
                    {"role": "system", "content": _SYNTHESIS_SYSTEM},
                    {"role": "user", "content": user_content},
                ],
            )
            thought_program = response.choices[0].message.content or ""
            tokens_used = response.usage.total_tokens if response.usage else 0
            logger.info(
                f"[LoT] Thought synthesis complete ({tokens_used} tokens, "
                f"{self.SYNTHESIS_MODEL}). Preview: "
                f"{thought_program[:120].replace(chr(10), ' ')}"
            )
        except Exception as exc:
            logger.warning(f"[LoT] Thought synthesis failed: {exc}")
            thought_program = (
                "OBSERVE:\n  notable = [synthesis failed]\n"
                "RULES:\n  (unknown)\n"
                "PLAN:\n  ELSE ACTION1\n"
            )

        return thought_program.strip()

    # ------------------------------------------------------------------
    # Phase 2: Override user prompt to inject LoT context
    # ------------------------------------------------------------------

    def build_user_prompt(self, latest_frame: FrameData) -> str:
        """Inject the current thought program and accumulated theory."""
        thought_program = getattr(
            self, "_current_thought_program", "(not yet synthesised — first turn)"
        )
        theory_snippet = self.theory[-1500:] if self.theory else "(none yet)"

        return _ACTION_USER_TMPL.format(
            thought_program=thought_program,
            theory=theory_snippet,
        )

    # ------------------------------------------------------------------
    # Main action-selection override
    # ------------------------------------------------------------------

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        """Two-phase LoT action selection.

        Turn 0 (no messages yet) → delegate to parent which returns RESET.
        Turn 1+                  → synthesise thought, update theory, then
                                   delegate to parent for tool-call action.
        """
        # First call: no messages yet — parent bootstraps with RESET
        if len(self.messages) == 0:
            logger.info("[LoT] First turn — delegating RESET to parent")
            action = super().choose_action(frames, latest_frame)
            return action

        # --- Phase 1: synthesise thought program ---
        thought_program = self._synthesize_thought(latest_frame)
        self._current_thought_program = thought_program

        # Store per-turn metadata
        self.lot_history.append(
            {
                "turn": self.action_counter,
                "thought_program": thought_program,
                "state": latest_frame.state.name,
                "score": latest_frame.levels_completed,
            }
        )

        # Update accumulated theory (rolling window, max 3000 chars)
        separator = f"\n\n--- Turn {self.action_counter} ---\n"
        self.theory += separator + thought_program
        if len(self.theory) > 3000:
            self.theory = self.theory[-3000:]

        # --- Phase 2: parent selects action via tool-call ---
        action = super().choose_action(frames, latest_frame)

        # Attach LoT reasoning metadata
        lot_meta = {
            "agent": "LanguageOfThought",
            "approach": "language_of_thought",
            "synthesis_model": self.SYNTHESIS_MODEL,
            "action_model": self.MODEL,
            "turn": self.action_counter,
            "action_chosen": action.name,
            "thought_program": thought_program,
            "theory_length_chars": len(self.theory),
            "game_context": {
                "score": latest_frame.levels_completed,
                "state": latest_frame.state.name,
                "action_counter": self.action_counter,
                "frame_count": len(frames),
            },
        }

        if isinstance(action.reasoning, dict):
            action.reasoning["lot"] = lot_meta
        else:
            action.reasoning = lot_meta

        logger.info(
            f"[LoT] Turn {self.action_counter} → {action.name} "
            f"(theory={len(self.theory)} chars)"
        )

        return action


# Convenience alias used in AVAILABLE_AGENTS
LotAgent = LanguageOfThought
