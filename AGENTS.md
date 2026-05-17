# AGENTS.md

## Cursor Cloud specific instructions

### Overview

This is the **ARC-AGI-3-Agents** codebase — a Python framework for building and benchmarking AI agents that play ARC-AGI-3 puzzle games. It uses Python 3.12, `uv` for dependency management, and has no Docker/database dependencies.

### Running the application

- Copy `.env.example` to `.env` before first run: `cp .env.example .env`
- For offline mode (no API keys required): set `OPERATION_MODE=offline` in `.env`
- Run agents with: `OPERATION_MODE=offline python3 -m uv run main.py --agent=random --game=maze-runner`
- Two local games are bundled: `maze-runner-v1` and `color-sort-v1`
- Online/normal modes require `ARC_API_KEY` and optionally `OPENAI_API_KEY` (for LLM agents)

### Lint, type-check, test

- **Lint**: `python3 -m uv run ruff check .` and `python3 -m uv run ruff format --check .`
- **Type check**: `python3 -m uv run mypy . --exclude tests`
- **Test**: `python3 -m uv run pytest`
- Note: `uv` is installed as a pip package (not a standalone binary), so invoke it as `python3 -m uv` rather than bare `uv`.

### Known pre-existing issues

- 18 of 82 tests fail due to `arc_agi` library API changes (field renames like `score` → `levels_completed`, constructor signature changes). These are not regressions from code changes.
- mypy reports ~116 errors (mostly in agent templates and due to missing type stubs for `arc_agi`/`requests`). Pre-existing.
- ruff reports pre-existing lint/format issues across several files including notebooks.
