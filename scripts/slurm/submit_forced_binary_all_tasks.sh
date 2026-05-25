#!/bin/bash
# Submit forced-binary reruns for selected local VLMs on Marlowe.
#
# Defaults:
#   - all benchmark tasks
#   - cogvlm, smolvlm2:256M, smolvlm2:500M
#   - output root: results/forced_binary_all_tasks
#
# Usage:
#   bash scripts/slurm/submit_forced_binary_all_tasks.sh
#   DRY_RUN=1 bash scripts/slurm/submit_forced_binary_all_tasks.sh
#   TASKS_CSV="trog,vocab" bash scripts/slurm/submit_forced_binary_all_tasks.sh
#   MODELS_CSV="cogvlm,smolvlm2:256M,qwen35:0.8B" bash scripts/slurm/submit_forced_binary_all_tasks.sh
#   bash scripts/slurm/submit_forced_binary_all_tasks.sh cogvlm smolvlm2:500M

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
TASKS_CSV="${TASKS_CSV:-egma-math,matrix-reasoning,mental-rotation,theory-of-mind,trog,vocab}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$CODE_DIR/results/forced_binary_all_tasks}"

CONDA_ENV_PATH="${CONDA_ENV_PATH:-$PROJECT_ROOT/envs/levante-bench-py311}"
SMOLVLM_CONDA_ENV_PATH="${SMOLVLM_CONDA_ENV_PATH:-}"
COGVLM_CONDA_ENV_PATH="${COGVLM_CONDA_ENV_PATH:-$PROJECT_ROOT/envs/levante-cogvlm}"

FORCE_BINARY_LABEL_SCORING="${FORCE_BINARY_LABEL_SCORING:-1}"
COGVLM_LABEL_SCORING_MODE="${COGVLM_LABEL_SCORING_MODE:-binary}"
DRY_RUN="${DRY_RUN:-0}"
MODELS_CSV="${MODELS_CSV:-}"

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

if [[ $# -gt 0 ]]; then
  MODELS=("$@")
elif [[ -n "$MODELS_CSV" ]]; then
  IFS=',' read -r -a MODELS <<< "$MODELS_CSV"
else
  MODELS=(
    "cogvlm"
    "smolvlm2:256M"
    "smolvlm2:500M"
  )
fi

echo "Submitting forced-binary jobs:"
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
