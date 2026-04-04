#!/usr/bin/env bash
# run_comparison.sh — Run the ReasoningAgent vs WorldModelAgent comparison across
# Option A (local/normal mode), Option B (online API), Option C (Kaggle-style
# full online sweep), and Option D (fully offline, behind-firewall mode).
#
# Usage:
#   cp .env.example .env          # then edit .env with your keys
#   bash run_comparison.sh [GAME]
#
# GAME defaults to "locksmith" if not provided.
#
# Prerequisites:
#   • OPENAI_API_KEY set in .env  (or exported in your shell)
#   • ARC_API_KEY   set in .env  (or exported in your shell)
#   • uv installed  (pip install uv)
#   • Run: uv sync
#
# --------------------------------------------------------------------------

set -euo pipefail

GAME="${1:-locksmith}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/${TIMESTAMP}"
mkdir -p "${LOG_DIR}"

echo "========================================================"
echo "  ARC-AGI-3 Agent Comparison Run"
echo "  Game   : ${GAME}"
echo "  Started: $(date)"
echo "========================================================"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
run_agent() {
    local LABEL="$1"
    local AGENT="$2"
    local TAGS="$3"
    local ENV_OVERRIDES="$4"   # extra env vars, e.g. "OPERATION_MODE=online"
    local LOG_FILE="${LOG_DIR}/${LABEL}.log"

    echo ""
    echo "--------------------------------------------------------"
    echo "  Running: ${LABEL}  (agent=${AGENT}, tags=${TAGS})"
    echo "  Log   : ${LOG_FILE}"
    echo "--------------------------------------------------------"

    env ${ENV_OVERRIDES} uv run main.py \
        --agent="${AGENT}" \
        --game="${GAME}" \
        --tags="${TAGS}" \
        2>&1 | tee "${LOG_FILE}"

    echo "  ✓ ${LABEL} finished."
}

# ---------------------------------------------------------------------------
# Option A — Normal mode (local environments + online API fallback)
#   OPERATION_MODE=normal scans environment_files/ first; if no local copy is
#   found it falls back to the online API.  This mirrors running a local game
#   server alongside the API.
# ---------------------------------------------------------------------------
echo ""
echo "========================================================"
echo "  OPTION A — Normal mode (local + API fallback)"
echo "========================================================"

run_agent \
    "optionA_baseline_reasoningagent" \
    "reasoningagent" \
    "baseline,optionA,normal" \
    "OPERATION_MODE=normal"

run_agent \
    "optionA_worldmodel_worldmodelagent" \
    "worldmodelagent" \
    "worldmodel,optionA,normal" \
    "OPERATION_MODE=normal"

# ---------------------------------------------------------------------------
# Option B — Online-only mode (three.arcprize.org API)
#   Set OPERATION_MODE=online so the Arcade class only uses the remote API and
#   never looks for local environment files.
# ---------------------------------------------------------------------------
echo ""
echo "========================================================"
echo "  OPTION B — Online-only mode (three.arcprize.org)"
echo "========================================================"

run_agent \
    "optionB_baseline_reasoningagent" \
    "reasoningagent" \
    "baseline,optionB,online" \
    "OPERATION_MODE=online SCHEME=https HOST=three.arcprize.org PORT=443"

run_agent \
    "optionB_worldmodel_worldmodelagent" \
    "worldmodelagent" \
    "worldmodel,optionB,online" \
    "OPERATION_MODE=online SCHEME=https HOST=three.arcprize.org PORT=443"

# ---------------------------------------------------------------------------
# Option C — Full online sweep (all available games)
#   Same as Option B but without a --game filter, so the Swarm plays every
#   game returned by the API.  This is the "Kaggle-style" full evaluation.
#
#   Prefer the Kaggle notebook for Option C when running on Kaggle:
#     arc_agi3_comparison_kaggle.ipynb
# ---------------------------------------------------------------------------
echo ""
echo "========================================================"
echo "  OPTION C — Full online sweep (all available games)"
echo "========================================================"

echo ""
echo "  Running: optionC_baseline_reasoningagent (all games)"
LOG_FILE="${LOG_DIR}/optionC_baseline_reasoningagent.log"
env OPERATION_MODE=online SCHEME=https HOST=three.arcprize.org PORT=443 \
    uv run main.py \
        --agent=reasoningagent \
        --tags="baseline,optionC,full" \
        2>&1 | tee "${LOG_FILE}"
echo "  ✓ optionC_baseline_reasoningagent finished."

echo ""
echo "  Running: optionC_worldmodel_worldmodelagent (all games)"
LOG_FILE="${LOG_DIR}/optionC_worldmodel_worldmodelagent.log"
env OPERATION_MODE=online SCHEME=https HOST=three.arcprize.org PORT=443 \
    uv run main.py \
        --agent=worldmodelagent \
        --tags="worldmodel,optionC,full" \
        2>&1 | tee "${LOG_FILE}"
echo "  ✓ optionC_worldmodel_worldmodelagent finished."

# ---------------------------------------------------------------------------
# Option D — Fully offline mode (local environment_files/ only, no API calls)
#   OPERATION_MODE=offline tells Arcade to skip all HTTP requests.
#
#   Pre-condition: environment_files/ must contain at least one game folder,
#   each with a metadata.json and a Python game implementation file.
#
#   Populate environment_files/ from any Kaggle dataset input that ships
#   metadata.json + game .py files (e.g. arc-agi-3-environment-files):
#
#     python - <<'EOF'
#     import json, shutil
#     from pathlib import Path
#     for mf in Path("/kaggle/input").rglob("metadata.json"):
#         m = json.loads(mf.read_text())
#         gid = m.get("game_id")
#         if gid:
#             dst = Path("environment_files") / gid
#             dst.mkdir(parents=True, exist_ok=True)
#             shutil.copy2(mf, dst / "metadata.json")
#             for py in mf.parent.glob("*.py"):
#                 shutil.copy2(py, dst / py.name)
#     EOF
#
#   For Kaggle, prefer the dedicated notebook:
#     arc_agi3_option_d_offline_kaggle.ipynb
# ---------------------------------------------------------------------------
echo ""
echo "========================================================"
echo "  OPTION D — Fully offline mode (local environment_files/)"
echo "========================================================"

run_agent \
    "optionD_baseline_reasoningagent" \
    "reasoningagent" \
    "baseline,optionD,offline" \
    "OPERATION_MODE=offline"

run_agent \
    "optionD_worldmodel_worldmodelagent" \
    "worldmodelagent" \
    "worldmodel,optionD,offline" \
    "OPERATION_MODE=offline"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "========================================================"
echo "  All runs complete.  Logs saved to: ${LOG_DIR}/"
echo ""
echo "  Scorecard URLs (Options A/B/C) printed at end of each log."
echo "  View them at: https://three.arcprize.org/scorecards/<card_id>"
echo ""
echo "  Option D runs offline; no scorecard URL is generated."
echo ""
echo "  To compare results:"
echo "    grep -h 'levels_completed\|scorecard' ${LOG_DIR}/*.log"
echo "========================================================"
