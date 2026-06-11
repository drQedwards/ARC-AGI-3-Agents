## Cursor Cloud specific instructions

### Overview

ARC-AGI-3-Agents is a Python framework for building and benchmarking AI agents that play turn-based ARC-AGI-3 game environments. It uses `uv` as the package manager with `pyproject.toml`.

### Running the application

- **Offline mode (no API keys needed):** `OPERATION_MODE=offline uv run main.py --agent=random --game=maze-runner`
- Two bundled local environments are available: `maze-runner-v1` and `color-sort-v1`
- For online/normal mode, `ARC_API_KEY` and `OPENAI_API_KEY` must be set in `.env`
- See `README.md` for full quickstart and agent options

### Key commands

| Task | Command |
|------|---------|
| Install deps | `uv sync` |
| Lint (ruff) | `uv run ruff check .` |
| Format check | `uv run ruff format --check .` |
| Type check | `uv run mypy .` |
| Tests | `uv run pytest tests/` |
| Run agent | `OPERATION_MODE=offline uv run main.py --agent=random --game=maze-runner` |

### Gotchas

- The `.env` file must exist (copy from `.env.example`). The update script handles this automatically.
- If `uv` is not on PATH (installed via `pip install uv`), invoke it as `python3 -m uv` instead of bare `uv`.
- Some unit tests in `tests/unit/test_core.py` and `tests/unit/test_swarm.py` fail due to pre-existing mismatches with the current `arc-agi` library API (old field names like `score` vs `levels_completed`). These are not regressions from your changes.
- The `ERROR | API request failed with status 401` log in offline mode is expected — the engine tries the online API first, then falls back to local environments.
- Pre-commit hooks use `ruff` (lint + format) and `mypy`. There are pre-existing lint/format issues in the repo.
