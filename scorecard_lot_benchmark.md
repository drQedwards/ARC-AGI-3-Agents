# Language of Thought (LoT) Agent — Benchmark Scorecard

Community leaderboard entry for the **`lotagent`** — a two-phase Language of
Thought agent that uses structured symbolic programs as its intermediate
reasoning representation instead of free-form natural language chains.

---

## Agent Overview

| Field | Value |
|---|---|
| **Agent name** | `lotagent` / `languageofthought` |
| **Approach** | Language of Thought — two-phase: synthesise → act |
| **Phase 1 (synthesis)** | `gpt-4o-mini` (OpenAI, with vision) or `llama-3.3-70b` (Venice) |
| **Phase 2 (action)** | `o4-mini` (OpenAI) or `llama-3.3-70b` (Venice) |
| **Backend select** | `VENICE_API_KEY` → Venice, else `OPENAI_API_KEY` → OpenAI |
| **Max actions** | 400 |
| **Message limit** | 8 |

---

## What is Language of Thought?

Language of Thought (LoT) prompting uses a structured pseudo-code program as
the *intermediate* reasoning representation rather than a natural-language chain
of thought. Each turn the agent:

1. **Synthesises** a formal thought program (Phase 1) with three blocks:
   - `OBSERVE:` — grid size, player position estimate, dominant colours, notable structures.
   - `RULES:` — accumulated if/then rules about game mechanics discovered so far (updated each turn).
   - `PLAN:` — a short logical decision tree (`IF condition THEN action`).

2. **Appends** the new thought program to a rolling `theory` string (capped at
   3 000 chars) that persists across all turns in a game session.

3. **Selects** the final action by passing the current thought program + theory
   excerpt to the action model with tool-calling required.

### v2 improvements (vision + token diet)

- **Vision**: the grid is rendered to a PNG (16-colour ARC palette) and sent to
  the synthesis model so it can *locate the player*. In v1 the model saw only
  raw integer grids and consistently reported `player_pos = unknown`.
- **Token diet**: the action phase receives a **compact frame summary**
  (`grid=HxW, nonzero=K, color_counts=[…]`) instead of the full grid dumped as
  text every turn. This cut the action-call from ~28 k tokens to ~1.3 k
  (**~22×**) and total run cost by **~18×** (see results).

---

## How to Run

```bash
# OpenAI backend (vision on)
uv run main.py --agent=lotagent --game=ls20 --tags="lot,benchmark,v2-vision"

# Venice backend (set VENICE_API_KEY in .env; text-only, llama-3.3-70b)
uv run main.py --agent=lotagent --game=ls20 --tags="lot,benchmark,venice"
```

---

## Benchmark: Locksmith (`ls20`)

Locksmith is a key-matching puzzle game. The agent must navigate a top-down
grid, find the correct key shape/colour, and reach the exit door — across 7
levels with an energy budget. Per-level human baseline action counts:
`[22, 123, 73, 84, 96, 192, 186]`.

### Score Table (real executed runs)

| Run | Backend / Models | Actions | Levels (of 7) | Scorecard ID | Notes |
|-----|------------------|---------|---------------|--------------|-------|
| v1  | OpenAI `gpt-4o-mini` + `o4-mini`, raw grids | 60 | 0 | `1381e428-…` (early), then 60-turn run | wandered; `player_pos=unknown`; ~30–90 k tokens/turn |
| v2  | Venice `llama-3.3-70b` (both phases), text-only | 6 | 0 | `4c582ba7-8569-4af6-83a3-13ab848e06cd` | synthesis worked (13.8 k tok); fell back after a non-tool-call reply |
| v2  | OpenAI `gpt-4o-mini`+vision / `o4-mini`, token-diet | 49 (stopped) | 0 | `97851801-c396-44c3-8d37-c24902515a04` | **~18× cheaper** (134 k tok @ turn 49 vs 1.79 M before); attempted strategic RESET |

> **Baseline reference:** `a4618274-e508-43d9-92f8-0108dbae9e39`
> (https://arcprize.org/scorecards/a4618274-e508-43d9-92f8-0108dbae9e39) — the
> best prior ls20 scorecard, used as the comparison target. The LoT agent does
> **not** yet beat it: across all three runs `lotagent` completed **0 / 7**
> levels. Locksmith requires key shape+colour matching via rotators plus energy
> management, which a single-pass LoT agent with a small vision model has not
> cracked.

### Honest assessment

- ✅ **Harness verified end-to-end**: ARC API auth, ls20 download, scorecard
  creation, recording capture, and the two-phase LoT loop all work.
- ✅ **Cost engineering works**: the v2 token diet is a real ~18× reduction;
  vision fixed the `player_pos=unknown` blind spot in the OBSERVE block.
- ❌ **Task not solved**: 0 levels completed. The agent explores but does not
  reliably reach key→door. Next levers (not yet done): zone-coordinate overlay
  on the rendered image (as `reasoningagent` does), an explicit player-tracking
  rule seeded into `theory`, and a higher-capability action model.

---

## Why This Is a Meaningful Benchmark Contribution

| Dimension | Natural Language CoT | Language of Thought (LoT) |
|---|---|---|
| Intermediate representation | Free prose | Structured pseudo-code |
| Knowledge accumulation | Implicit in context window | Explicit `theory` string |
| Diagnosability | Hard to isolate failures | Per-block (OBSERVE/RULES/PLAN) attribution |
| Token efficiency | Verbose | Compact symbolic form (~18× cheaper in v2) |
| Rule learning signal | Buried in text | Explicit `RULES:` block |

LoT provides a *structured, low-cost baseline* for ARC-AGI-3: the explicit
`theory` string and per-block attribution make it easy to see *why* the agent
fails (here: weak spatial grounding on locksmith), which is more actionable than
a single opaque score.

---

## Files

| File | Purpose |
|---|---|
| `agents/templates/lot_agent.py` | Agent implementation (vision + token diet + Venice/OpenAI) |
| `agents/__init__.py` | Registration (`lotagent`, `languageofthought`) |
| `scorecard_lot_benchmark.md` | This scorecard |

---

*Submitted via branch `claude/epic-pasteur-v2vdrl`.*
