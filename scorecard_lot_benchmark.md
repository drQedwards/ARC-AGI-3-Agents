# Language of Thought (LoT) Benchmark Scorecard

Community leaderboard entry for the **`lotagent`** — a two-phase Language of
Thought agent that uses structured symbolic programs as its intermediate
reasoning representation instead of free-form natural language chains.

---

## Agent Overview

| Field | Value |
|---|---|
| **Agent name** | `lotagent` |
| **Approach** | Language of Thought — two-phase: synthesise → act |
| **Phase 1 model** | `gpt-4o-mini` (thought synthesis) |
| **Phase 2 model** | `o4-mini` (action selection via tool-call) |
| **Reasoning effort** | `medium` |
| **Max actions** | 400 |
| **Message limit** | 8 |

---

## What is Language of Thought?

Language of Thought (LoT) prompting uses a structured pseudo-code program as
the *intermediate* reasoning representation rather than a natural-language chain
of thought.  Each turn the agent:

1. **Synthesises** a formal thought program with three blocks:
   - `OBSERVE:` — grid size, player position estimate, dominant colours,
     notable structures.
   - `RULES:` — accumulated if/then rules about game mechanics discovered so
     far (updated each turn).
   - `PLAN:` — a short logical decision tree (`IF condition THEN action`).

2. **Appends** the new thought program to a rolling `theory` string (capped at
   3 000 chars) that persists across all turns in a game session.

3. **Selects** the final action by passing the current thought program + theory
   excerpt to `o4-mini` with tool-calling required.

This approach enforces *explicit symbolic accounting* of what the agent believes
about the world, making it easier to diagnose where reasoning breaks down and
providing a natural mechanism for knowledge accumulation across turns.

---

## How to Run

```bash
# Single run
uv run main.py --agent=lotagent --game=ls20 --tags="lot,benchmark,v1"

# Record replay for later inspection
uv run main.py --agent=lotagent --game=ls20 --tags="lot,benchmark,v1" --record

# Run with verbose logging to inspect thought programs
LOG_LEVEL=INFO uv run main.py --agent=lotagent --game=ls20 --tags="lot,benchmark,v1"
```

---

## Benchmark: Locksmith (`ls20`)

Locksmith is a key-matching puzzle game.  The agent must navigate a top-down
grid, find the correct key shape/colour, and reach the exit door — across 6
levels with an energy budget.

### Score Table

| Run | Date | Levels Completed | Actions Taken | Scorecard URL |
|-----|------|-----------------|---------------|---------------|
| v1  | TBD  | —               | —             | —             |
| v2  | TBD  | —               | —             | —             |

> Fill in results after running:
> `uv run main.py --agent=lotagent --game=ls20 --tags="lot,benchmark,v1"`

---

## Why This Is a Meaningful Benchmark Contribution

| Dimension | Natural Language CoT | Language of Thought (LoT) |
|---|---|---|
| Intermediate representation | Free prose | Structured pseudo-code |
| Knowledge accumulation | Implicit in context window | Explicit `theory` string |
| Diagnosability | Hard to isolate failures | Per-block attribution |
| Token efficiency | Verbose | Compact symbolic form |
| Rule learning signal | Buried in text | Explicit `RULES:` block |

LoT agents provide a *structured baseline* for understanding how much
performance comes from symbolic world-model accumulation vs. raw language
reasoning — a meaningful question for ARC-AGI-3 which requires genuine
generalisation.

---

## Files

- `agents/templates/lot_agent.py` — agent implementation
- `agents/__init__.py` — registration (`lotagent`, `languageofthought`)
- `scorecard_lot_benchmark.md` — this file

---

*Submitted via `claude/epic-pasteur-v2vdrl`.*
