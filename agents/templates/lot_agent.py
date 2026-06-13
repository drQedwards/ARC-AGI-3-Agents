"""Language of Thought (LoT) Agent for ARC-AGI-3.

Two-phase reasoning approach:
  Phase 1 (Thought Synthesis): Generate a structured symbolic thought program
    from the current frame using a fast model.
  Phase 2 (Action Selection): Use the synthesised program + accumulated theory
    to select an action via tool calls.

Supports OpenAI and Venice (OpenAI-compatible) backends:
  - Set OPENAI_API_KEY for OpenAI  (models: gpt-4o-mini / o4-mini)
  - Set VENICE_API_KEY for Venice  (models: llama-3.3-70b / llama-3.3-70b)
    VENICE_BASE_URL defaults to https://api.venice.ai/api/v1
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
# Backend detection
# ---------------------------------------------------------------------------

_VENICE_BASE_URL = "https://api.venice.ai/api/v1"

def _make_client(base_url: Optional[str] = None, api_key_env: str = "OPENAI_API_KEY") -> OpenAIClient:
    """Build an OpenAI-compatible client, optionally pointed at Venice."""
    venice_key = os.environ.get("VENICE_API_KEY", "")
    openai_key = os.environ.get("OPENAI_API_KEY", "")

    if venice_key:
        url = os.environ.get("VENICE_BASE_URL", _VENICE_BASE_URL)
        return OpenAIClient(api_key=venice_key, base_url=url)
    return OpenAIClient(api_key=openai_key, base_url=base_url)


def _default_models() -> tuple[str, str]:
    """Return (synthesis_model, action_model) based on available keys."""
    if os.environ.get("VENICE_API_KEY"):
        return "llama-3.3-70b", "llama-3.3-70b"
    return "gpt-4o-mini", "o4-mini"


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

    Phase 1 produces a structured OBSERVE / RULES / PLAN program each turn.
    Phase 2 passes that program plus the rolling theory to the action model
    for tool-call action selection.

    Backend: Venice (VENICE_API_KEY) or OpenAI (OPENAI_API_KEY).
    """

    MAX_ACTIONS: int = 400
    DO_OBSERVATION: bool = False
    MODEL_REQUIRES_TOOLS: bool = True
    MESSAGE_LIMIT: int = 8
    REASONING_EFFORT: Optional[str] = None  # Venice models don't support reasoning_effort

    # Defaults; overridden at __init__ based on env
    MODEL: str = "llama-3.3-70b"
    SYNTHESIS_MODEL: str = "llama-3.3-70b"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        synthesis_model, action_model = _default_models()
        self.SYNTHESIS_MODEL = synthesis_model
        self.MODEL = action_model
        # Venice doesn't support reasoning_effort param
        if os.environ.get("VENICE_API_KEY"):
            self.REASONING_EFFORT = None
        self.theory: str = ""
        self.lot_history: list[dict] = []
        backend = "Venice" if os.environ.get("VENICE_API_KEY") else "OpenAI"
        logger.info(
            f"[LoT] Backend={backend} synthesis={self.SYNTHESIS_MODEL} action={self.MODEL}"
        )

    # ------------------------------------------------------------------
    # Phase 1: Thought Synthesis
    # ------------------------------------------------------------------

    def _synthesize_thought(self, latest_frame: FrameData) -> str:
        """Produce a structured thought program for this frame."""
        client = _make_client()
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
    # Phase 2: Inject LoT context into the action prompt
    # ------------------------------------------------------------------

    def build_user_prompt(self, latest_frame: FrameData) -> str:
        thought_program = getattr(
            self, "_current_thought_program", "(not yet synthesised — first turn)"
        )
        theory_snippet = self.theory[-1500:] if self.theory else "(none yet)"
        return _ACTION_USER_TMPL.format(
            thought_program=thought_program,
            theory=theory_snippet,
        )

    # ------------------------------------------------------------------
    # Build tools — Venice requires strict=False
    # ------------------------------------------------------------------

    def build_tools(self) -> list[dict[str, Any]]:
        functions = self.build_functions()
        tools: list[dict[str, Any]] = []
        for f in functions:
            tool: dict[str, Any] = {
                "type": "function",
                "function": {
                    "name": f["name"],
                    "description": f["description"],
                    "parameters": f.get("parameters", {}),
                },
            }
            # strict mode is OpenAI-only; skip for Venice
            if not os.environ.get("VENICE_API_KEY"):
                tool["function"]["strict"] = True
            tools.append(tool)
        return tools

    # ------------------------------------------------------------------
    # Override choose_action to use Venice-aware client
    # ------------------------------------------------------------------

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        if len(self.messages) == 0:
            logger.info("[LoT] First turn — delegating RESET to parent")
            action = super().choose_action(frames, latest_frame)
            return action

        # Phase 1: synthesise
        thought_program = self._synthesize_thought(latest_frame)
        self._current_thought_program = thought_program

        self.lot_history.append(
            {
                "turn": self.action_counter,
                "thought_program": thought_program,
                "state": latest_frame.state.name,
                "score": latest_frame.levels_completed,
            }
        )

        separator = f"\n\n--- Turn {self.action_counter} ---\n"
        self.theory += separator + thought_program
        if len(self.theory) > 3000:
            self.theory = self.theory[-3000:]

        # Phase 2: parent tool-call selection (uses self.MODEL via llm_agents.py)
        action = super().choose_action(frames, latest_frame)

        lot_meta = {
            "agent": "LanguageOfThought",
            "approach": "language_of_thought",
            "backend": "venice" if os.environ.get("VENICE_API_KEY") else "openai",
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


LotAgent = LanguageOfThought
