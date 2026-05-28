#!/bin/bash
# Submit forced-binary paper-model runs for the current missing 9 models.
#
# Usage:
#   bash scripts/slurm/submit_forced_binary_missing_9.sh
#   DRY_RUN=1 bash scripts/slurm/submit_forced_binary_missing_9.sh
#   FORCE=1 bash scripts/slurm/submit_forced_binary_missing_9.sh
#   bash scripts/slurm/submit_forced_binary_missing_9.sh gemma4:E2B-it internvl35:38B

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_SCRIPT="$SCRIPT_DIR/submit_paper_model_list_forced_binary.sh"

if [[ ! -f "$BASE_SCRIPT" ]]; then
  echo "ERROR: missing submit script: $BASE_SCRIPT" >&2
  exit 1
fi

PROJECT_ROOT="${PROJECT_ROOT:-/projects/m000102}"
CODE_DIR="${CODE_DIR:-$PROJECT_ROOT/code/levante-bench}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$CODE_DIR/results/forced_binary_paper_models}"
TASKS_CSV="${TASKS_CSV:-egma-math,matrix-reasoning,mental-rotation,theory-of-mind,trog,vocab}"

DRY_RUN="${DRY_RUN:-0}"
FORCE="${FORCE:-0}"
MAX_TIME_LIMIT="${MAX_TIME_LIMIT:-04:00:00}"
STRICT_QWEN_PREFLIGHT="${STRICT_QWEN_PREFLIGHT:-0}"
PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-0}"

DEFAULT_MODELS=(
  "gemma4:E2B-it"
  "gemma4:E4B-it"
  "gemma4:26B-A4B-it"
  "gemma4:31B-it"
  "internvl35:38B"
  "molmo2:8B"
  "smolvlm2:2.2B"
  "tinyllava:2.4B"
  "tinyllava:3.1B"
)

if [[ $# -gt 0 ]]; then
  MODELS=("$@")
else
  MODELS=("${DEFAULT_MODELS[@]}")
fi

ONLY_MODELS="$(printf '%s ' "${MODELS[@]}")"
ONLY_MODELS="${ONLY_MODELS% }"

echo "Submitting forced-binary missing-model set:"
echo "  MODELS=$ONLY_MODELS"
echo "  OUTPUT_ROOT=$OUTPUT_ROOT"
echo "  TASKS_CSV=$TASKS_CSV"
echo "  MAX_TIME_LIMIT=$MAX_TIME_LIMIT"
echo "  FORCE=$FORCE"
echo "  DRY_RUN=$DRY_RUN"
echo ""

ONLY="$ONLY_MODELS" \
DRY_RUN="$DRY_RUN" \
FORCE="$FORCE" \
OUTPUT_ROOT="$OUTPUT_ROOT" \
TASKS_CSV="$TASKS_CSV" \
MAX_TIME_LIMIT="$MAX_TIME_LIMIT" \
STRICT_QWEN_PREFLIGHT="$STRICT_QWEN_PREFLIGHT" \
PYTHONNOUSERSITE="$PYTHONNOUSERSITE" \
QWEN_CONDA_ENV_PATH="${QWEN_CONDA_ENV_PATH:-}" \
GEMMA_CONDA_ENV_PATH="${GEMMA_CONDA_ENV_PATH:-}" \
SMOLVLM_CONDA_ENV_PATH="${SMOLVLM_CONDA_ENV_PATH:-}" \
COGVLM_CONDA_ENV_PATH="${COGVLM_CONDA_ENV_PATH:-}" \
bash "$BASE_SCRIPT"
