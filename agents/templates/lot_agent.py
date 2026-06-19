"""Language of Thought (LoT) Agent for ARC-AGI-3.

Two-phase reasoning approach:
  Phase 1 (Thought Synthesis): Generate a structured symbolic thought program
    from the current frame using a fast model.
  Phase 2 (Action Selection): Use the synthesised program + accumulated theory
    to select an action via tool calls.

Supports OpenAI and Venice (OpenAI-compatible) backends:
  - Set OPENAI_API_KEY for OpenAI  (models: gpt-4o-mini / o4-mini)
  - Set VENICE_API_KEY for Venice  (models: llama-3.3-70b for both phases)
    VENICE_BASE_URL defaults to https://api.venice.ai/api/v1
"""

import base64
import io
import logging
import os
import textwrap
from collections import Counter
from typing import Any, Optional

import numpy as np
from arcengine import FrameData, GameAction
from openai import OpenAI as OpenAIClient
from PIL import Image, ImageDraw

from .llm_agents import ReasoningLLM

logger = logging.getLogger(__name__)

_VENICE_BASE_URL = "https://api.venice.ai/api/v1"

# ARC-AGI-3 official 16-colour palette (index -> hex)
_KEY_COLORS = {
    0: "#FFFFFF", 1: "#CCCCCC", 2: "#999999", 3: "#666666",
    4: "#333333", 5: "#000000", 6: "#E53AA3", 7: "#FF7BCC",
    8: "#F93C31", 9: "#1E93FF", 10: "#88D8F1", 11: "#FFDC00",
    12: "#FF851B", 13: "#921231", 14: "#4FCC30", 15: "#A356D6",
}


def _venice_active() -> bool:
    return bool(os.environ.get("VENICE_API_KEY"))


def _make_client() -> OpenAIClient:
    if _venice_active():
        return OpenAIClient(
            api_key=os.environ["VENICE_API_KEY"],
            base_url=os.environ.get("VENICE_BASE_URL", _VENICE_BASE_URL),
        )
    return OpenAIClient(api_key=os.environ.get("OPENAI_API_KEY", ""))


def _default_models() -> tuple[str, str]:
    if _venice_active():
        return "llama-3.3-70b", "llama-3.3-70b"
    return "gpt-4o-mini", "o4-mini"


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

    Backend auto-selects: VENICE_API_KEY -> Venice (llama-3.3-70b),
    else OPENAI_API_KEY -> OpenAI (gpt-4o-mini / o4-mini).
    """

    MAX_ACTIONS: int = 400
    DO_OBSERVATION: bool = False
    MODEL_REQUIRES_TOOLS: bool = True
    MESSAGE_LIMIT: int = 8
    REASONING_EFFORT: Optional[str] = None

    MODEL: str = "o4-mini"
    SYNTHESIS_MODEL: str = "gpt-4o-mini"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        synthesis_model, action_model = _default_models()
        self.SYNTHESIS_MODEL = synthesis_model
        self.MODEL = action_model
        if _venice_active():
            self.REASONING_EFFORT = None
        else:
            self.REASONING_EFFORT = "medium"
        self.theory: str = ""
        self.lot_history: list[dict] = []
        backend = "Venice" if _venice_active() else "OpenAI"
        logger.info(
            f"[LoT] Backend={backend} synthesis={self.SYNTHESIS_MODEL} action={self.MODEL}"
        )

    # ------------------------------------------------------------------
    # Vision + token-diet helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _last_grid(frame: FrameData) -> list[list[int]]:
        return frame.frame[-1] if frame.frame else []

    def _grid_summary(self, frame: FrameData) -> str:
        """Compact textual stats instead of dumping the full grid (token diet)."""
        grid = self._last_grid(frame)
        if not grid or not grid[0]:
            return "empty grid"
        arr = np.array(grid)
        h, w = arr.shape
        counts = Counter(arr.flatten().tolist())
        top = ", ".join(f"{c}x{n}" for c, n in counts.most_common(6))
        return f"grid={h}x{w}, nonzero={int(np.sum(arr != 0))}, color_counts=[{top}]"

    def _render_png(self, frame: FrameData, cell: int = 8) -> Optional[bytes]:
        """Render the latest grid to a PNG so a vision model can locate the player."""
        grid = self._last_grid(frame)
        if not grid or not grid[0]:
            return None
        h, w = len(grid), len(grid[0])
        img = Image.new("RGB", (w * cell, h * cell), color="white")
        draw = ImageDraw.Draw(img)
        for y in range(h):
            for x in range(w):
                draw.rectangle(
                    [x * cell, y * cell, (x + 1) * cell, (y + 1) * cell],
                    fill=_KEY_COLORS.get(grid[y][x], "#888888"),
                )
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def _synthesize_thought(self, latest_frame: FrameData) -> str:
        client = _make_client()
        frame_text = self._grid_summary(latest_frame)
        theory_snippet = self.theory[-1500:] if self.theory else "(none yet)"

        user_text = _SYNTHESIS_USER_TMPL.format(
            state=latest_frame.state.name,
            score=latest_frame.levels_completed,
            frame=frame_text,
            theory=theory_snippet,
        )

        # Attach a rendered image so the model can actually locate the player.
        # Vision is OpenAI-only here (gpt-4o-mini); Venice llama is text-only.
        user_content: Any = user_text
        if not _venice_active():
            png = self._render_png(latest_frame)
            if png:
                b64 = base64.b64encode(png).decode()
                user_content = [
                    {"type": "text", "text": user_text},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{b64}",
                            "detail": "high",
                        },
                    },
                ]

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

    def build_func_resp_prompt(self, latest_frame: FrameData) -> str:
        """Compact tool-response (token diet): summary stats, NOT the full grid.

        The parent LLM dumps the entire grid here each turn (30-90k tokens).
        The LoT agent already digested the grid (with vision) during synthesis,
        so the action phase only needs a compact summary.
        """
        return textwrap.dedent(
            f"""\
            # State: {latest_frame.state.name}
            # Score (levels completed): {latest_frame.levels_completed}
            # Frame summary: {self._grid_summary(latest_frame)}
            """
        )

    def build_user_prompt(self, latest_frame: FrameData) -> str:
        thought_program = getattr(
            self, "_current_thought_program", "(not yet synthesised - first turn)"
        )
        theory_snippet = self.theory[-1500:] if self.theory else "(none yet)"
        return _ACTION_USER_TMPL.format(
            thought_program=thought_program,
            theory=theory_snippet,
        )

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
            if not _venice_active():
                tool["function"]["strict"] = True
            tools.append(tool)
        return tools

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        if len(self.messages) == 0:
            logger.info("[LoT] First turn - delegating RESET to parent")
            return super().choose_action(frames, latest_frame)

        thought_program = self._synthesize_thought(latest_frame)
        self._current_thought_program = thought_program

        self.lot_history.append({
            "turn": self.action_counter,
            "thought_program": thought_program,
            "state": latest_frame.state.name,
            "score": latest_frame.levels_completed,
        })

        separator = f"\n\n--- Turn {self.action_counter} ---\n"
        self.theory += separator + thought_program
        if len(self.theory) > 3000:
            self.theory = self.theory[-3000:]

        # Phase 2: parent tool-call selection. Route through Venice if active.
        if _venice_active():
            from openai import OpenAI
            import agents.templates.llm_agents as llm_mod
            original = llm_mod.OpenAIClient
            llm_mod.OpenAIClient = lambda api_key="", **kw: OpenAI(
                api_key=os.environ["VENICE_API_KEY"],
                base_url=os.environ.get("VENICE_BASE_URL", _VENICE_BASE_URL),
            )
            try:
                action = super().choose_action(frames, latest_frame)
            except IndexError as exc:
                logger.warning(f"[LoT] Venice returned no tool_call ({exc}); falling back to ACTION1")
                action = GameAction.ACTION1
            finally:
                llm_mod.OpenAIClient = original
        else:
            try:
                action = super().choose_action(frames, latest_frame)
            except IndexError as exc:
                logger.warning(f"[LoT] Empty tool_calls ({exc}); falling back to ACTION1")
                action = GameAction.ACTION1
            except Exception as exc:
                # e.g. pydantic ValidationError when model emits out-of-range coords
                logger.warning(f"[LoT] Action validation failed ({type(exc).__name__}: {exc}); falling back to ACTION1")
                action = GameAction.ACTION1

        lot_meta = {
            "agent": "LanguageOfThought",
            "approach": "language_of_thought",
            "backend": "venice" if _venice_active() else "openai",
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
            f"[LoT] Turn {self.action_counter} -> {action.name} "
            f"(theory={len(self.theory)} chars)"
        )

        return action


LotAgent = LanguageOfThought
