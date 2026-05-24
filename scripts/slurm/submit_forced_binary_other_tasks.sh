#!/bin/bash
# Submit forced-binary reruns for "other tasks" across three target models.
#
# Default tasks exclude the already-run vocab/trog/matrix set:
#   egma-math,mental-rotation,theory-of-mind
#
# Usage:
#   bash scripts/slurm/submit_forced_binary_other_tasks.sh
#   DRY_RUN=1 bash scripts/slurm/submit_forced_binary_other_tasks.sh
#   TASKS_CSV="egma-math,theory-of-mind" bash scripts/slurm/submit_forced_binary_other_tasks.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_SUBMIT_SCRIPT="$SCRIPT_DIR/submit_historic_vlms_local.sh"

if [[ ! -f "$BASE_SUBMIT_SCRIPT" ]]; then
  echo "ERROR: missing submit script: $BASE_SUBMIT_SCRIPT" >&2
  exit 1
fi

PROJECT_ROOT="${PROJECT_ROOT:-/projects/m000102}"
CODE_DIR="${CODE_DIR:-$PROJECT_ROOT/code/levante-bench}"
VERSION="${VERSION:-v1}"
TASKS_CSV="${TASKS_CSV:-egma-math,mental-rotation,theory-of-mind}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$CODE_DIR/results/forced_binary_other_tasks}"

# Preserve previously working env routing defaults.
CONDA_ENV_PATH="${CONDA_ENV_PATH:-$PROJECT_ROOT/envs/levante-bench-py311}"
SMOLVLM_CONDA_ENV_PATH="${SMOLVLM_CONDA_ENV_PATH:-}"
COGVLM_CONDA_ENV_PATH="${COGVLM_CONDA_ENV_PATH:-$PROJECT_ROOT/envs/levante-cogvlm}"

# Force binary scoring for generic label models and CogVLM adapter.
FORCE_BINARY_LABEL_SCORING="${FORCE_BINARY_LABEL_SCORING:-1}"
COGVLM_LABEL_SCORING_MODE="${COGVLM_LABEL_SCORING_MODE:-binary}"
DRY_RUN="${DRY_RUN:-0}"

if [[ -z "$SMOLVLM_CONDA_ENV_PATH" ]]; then
  CANDIDATES=(
    "$PROJECT_ROOT/envs/${USER:-unknown}-levante-py311"
    "$PROJECT_ROOT/envs/david81-levante-py311"
    "$CONDA_ENV_PATH"
  )
  for candidate in "${CANDIDATES[@]}"; do
    if [[ -x "$candidate/bin/python" ]] && "$candidate/bin/python" -c "import num2words" >/dev/null 2>&1; then
      SMOLVLM_CONDA_ENV_PATH="$candidate"
      break
    fi
  done
fi
SMOLVLM_CONDA_ENV_PATH="${SMOLVLM_CONDA_ENV_PATH:-$CONDA_ENV_PATH}"

MODELS=(
  "cogvlm"
  "smolvlm2:256M"
  "smolvlm2:500M"
)

echo "Submitting forced-binary 'other tasks' jobs:"
echo "  VERSION=$VERSION"
echo "  TASKS_CSV=$TASKS_CSV"
echo "  OUTPUT_ROOT=$OUTPUT_ROOT"
echo "  CONDA_ENV_PATH=$CONDA_ENV_PATH"
echo "  SMOLVLM_CONDA_ENV_PATH=$SMOLVLM_CONDA_ENV_PATH"
echo "  COGVLM_CONDA_ENV_PATH=$COGVLM_CONDA_ENV_PATH"
echo "  FORCE_BINARY_LABEL_SCORING=$FORCE_BINARY_LABEL_SCORING"
echo "  COGVLM_LABEL_SCORING_MODE=$COGVLM_LABEL_SCORING_MODE"
echo "  DRY_RUN=$DRY_RUN"
echo "  MODELS=${MODELS[*]}"
echo ""

DRY_RUN="$DRY_RUN" \
VERSION="$VERSION" \
TASKS_CSV="$TASKS_CSV" \
OUTPUT_ROOT="$OUTPUT_ROOT" \
CONDA_ENV_PATH="$CONDA_ENV_PATH" \
SMOLVLM_CONDA_ENV_PATH="$SMOLVLM_CONDA_ENV_PATH" \
COGVLM_CONDA_ENV_PATH="$COGVLM_CONDA_ENV_PATH" \
FORCE_BINARY_LABEL_SCORING="$FORCE_BINARY_LABEL_SCORING" \
COGVLM_LABEL_SCORING_MODE="$COGVLM_LABEL_SCORING_MODE" \
bash "$BASE_SUBMIT_SCRIPT" "${MODELS[@]}"
