"""
Language of Thought (LoT) Agent for ARC-AGI-3

Two-phase reasoning:
  Phase 1 (Thought Synthesis): Generate a structured pseudo-code thought program
    summarising observations, accumulated rules, and a decision plan.
  Phase 2 (Action Selection): Use the synthesised program + theory context to
    select the next GameAction via tool-call.

The agent maintains a growing `theory` string across turns that encodes
everything it has learned about the current game.
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
# Thought program template (shown to the synthesis model)
# ---------------------------------------------------------------------------
_THOUGHT_PROGRAM_TEMPLATE = """\
OBSERVE:
  grid_size = (H, W)
  player_pos = (approx_x, approx_y) or unknown
  dominant_color = N
  nonzero_cells = K
  notable = [list of concise observations about the frame]

RULES:
  <if-then rules about game mechanics learned so far; refine from prior theory>

PLAN:
  IF <condition> THEN <ACTION1|ACTION2|ACTION3|ACTION4|ACTION5|RESET>
  ELIF <condition> THEN <action>
  ELSE <default_action>
"""

_SYNTHESIS_SYSTEM_PROMPT = (
    "You are a logical thought synthesizer for ARC-AGI-3 games. "
    "Output ONLY a structured thought program using the exact headings "
    "OBSERVE, RULES, and PLAN. Do not include any prose outside the program."
)


class LanguageOfThought(ReasoningLLM):
    """Language-of-Thought agent.

    Uses a cheap fast model (gpt-4o-mini) to synthesise a symbolic thought
    program each turn, then feeds that program plus the accumulated theory into
    o4-mini for final action selection via tool-call.
    """

    MAX_ACTIONS: int = 400
    DO_OBSERVATION: bool = False   # we handle our own two-phase observation
    MODEL: str = "o4-mini"
    MODEL_REQUIRES_TOOLS: bool = True
    MESSAGE_LIMIT: int = 8
    REASONING_EFFORT: Optional[str] = "medium"

    # Model used only for fast/cheap thought synthesis (Phase 1)
    SYNTHESIS_MODEL: str = "gpt-4o-mini"
    # Maximum chars to retain in the accumulated theory
    THEORY_MAX_CHARS: int = 3000
    # Maximum chars from theory shown in the user prompt
    THEORY_PROMPT_CHARS: int = 1500

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.theory: str = ""          # accumulated program theory about the game
        self.lot_history: list[dict] = []  # per-turn thought programs + metadata

    # ------------------------------------------------------------------
    # Phase 1: Thought Synthesis
    # ------------------------------------------------------------------

    def _synthesize_thought(self, latest_frame: FrameData) -> str:
        """Call gpt-4o-mini to produce a structured thought program for this frame.

        Returns the raw thought-program string (no tool calls, plain text).
        """
        client = OpenAIClient(api_key=os.environ.get("OPENAI_API_KEY", ""))

        frame_text = self.pretty_print_3d(latest_frame.frame)

        user_content = textwrap.dedent(
            f"""\
            ## Current Frame (state={latest_frame.state.name}, score={latest_frame.levels_completed})
            {frame_text}

            ## Accumulated Theory (last {self.THEORY_PROMPT_CHARS} chars)
            {self.theory[-self.THEORY_PROMPT_CHARS:] if self.theory else "(none yet)"}

            ## Template
            {_THOUGHT_PROGRAM_TEMPLATE}

            Produce a thought program for this frame. Fill in concrete values
            where possible. Update RULES based on what you have observed so far.
            """
        )

        try:
            response = client.chat.completions.create(
                model=self.SYNTHESIS_MODEL,
                messages=[
                    {"role": "system", "content": _SYNTHESIS_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
            )
            thought_program = response.choices[0].message.content or ""
            tokens_used = response.usage.total_tokens if response.usage else 0
            logger.info(
                f"[LoT] Thought synthesis used {tokens_used} tokens "
                f"({self.SYNTHESIS_MODEL})"
            )
        except Exception as exc:
            logger.warning(f"[LoT] Thought synthesis failed: {exc}")
            thought_program = (
                "OBSERVE:\n  notable = [synthesis failed]\n"
                "RULES:\n  (unknown)\n"
                "PLAN:\n  ELSE ACTION1\n"
            )

        return thought_program

    # ------------------------------------------------------------------
    # Phase 2: Override user prompt to inject LoT context
    # ------------------------------------------------------------------

    def build_user_prompt(self, latest_frame: FrameData) -> str:
        """Inject the current thought program and accumulated theory."""
        # The current thought program is stored as the last entry in lot_history
        if self.lot_history:
            current_thought = self.lot_history[-1].get("thought_program", "")
        else:
            current_thought = "(no thought program yet — first turn)"

        theory_excerpt = (
            self.theory[-self.THEORY_PROMPT_CHARS:]
            if self.theory
            else "(no accumulated theory yet)"
        )

        return textwrap.dedent(
            f"""\
            # CONTEXT
            You are an agent playing an ARC-AGI-3 game.  Your objective is to
            WIN and avoid GAME_OVER while minimising actions.

            One action produces one Frame. One Frame is made of one or more
            sequential Grids. Each Grid is a matrix of INT<0,63> x INT<0,63>
            filled with INT<0,15> values.

            # CURRENT THOUGHT PROGRAM
            {current_thought}

            # ACCUMULATED THEORY (last {self.THEORY_PROMPT_CHARS} chars)
            {theory_excerpt}

            # TURN
            Call exactly one action.
            """
        )

    # ------------------------------------------------------------------
    # Main action-selection override
    # ------------------------------------------------------------------

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        """Two-phase LoT action selection.

        Turn 0  → return RESET immediately (required by parent protocol).
        Turn 1+ → synthesise thought, update theory, then delegate to parent
                  for tool-call action selection.
        """
        # First call: send RESET to initialise the game
        if self.action_counter == 0:
            logger.info("[LoT] First turn — returning RESET")
            # Let the parent handle the initial RESET bookkeeping
            action = super().choose_action(frames, latest_frame)
            return action

        # --- Phase 1: synthesise thought program ---
        thought_program = self._synthesize_thought(latest_frame)
        logger.info(f"[LoT] Thought program:\n{thought_program}")

        # Store per-turn metadata
        turn_entry = {
            "turn": self.action_counter,
            "thought_program": thought_program,
            "state": latest_frame.state.name,
            "score": latest_frame.levels_completed,
        }
        self.lot_history.append(turn_entry)

        # Update accumulated theory (rolling window)
        separator = f"\n\n--- Turn {self.action_counter} ---\n"
        self.theory += separator + thought_program
        if len(self.theory) > self.THEORY_MAX_CHARS:
            self.theory = self.theory[-self.THEORY_MAX_CHARS:]

        # --- Phase 2: parent chooses action via tool-call ---
        action = super().choose_action(frames, latest_frame)

        # Attach LoT reasoning metadata
        action.reasoning = {
            "agent": "LanguageOfThought",
            "model_action": self.MODEL,
            "model_synthesis": self.SYNTHESIS_MODEL,
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

        return action


# Convenience alias used in AVAILABLE_AGENTS
LotAgent = LanguageOfThought
