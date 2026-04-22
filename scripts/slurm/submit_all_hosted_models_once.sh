#!/bin/bash
# Submit one deterministic single-run Slurm job per hosted model.
#
# Behavior:
# - Runs all 6 tasks
# - num_runs=1
# - true_random_option_order=false (deterministic option order)
# - Writes under repo-local results/<model_name>/...
#
# Usage:
#   bash scripts/slurm/submit_all_hosted_models_once.sh
#   VERSION=v1_new_parser bash scripts/slurm/submit_all_hosted_models_once.sh
#   bash scripts/slurm/submit_all_hosted_models_once.sh gpt53 qwen3vl_30b_hf

set -euo pipefail

if ! command -v sbatch >/dev/null 2>&1; then
  echo "ERROR: sbatch not found in PATH." >&2
  echo "Run this script on a Marlowe login node (or any Slurm head node)." >&2
  exit 127
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_SCRIPT="$SCRIPT_DIR/run_local_model_experiment.sbatch"

if [[ ! -f "$SBATCH_SCRIPT" ]]; then
  echo "ERROR: missing sbatch template: $SBATCH_SCRIPT" >&2
  exit 1
fi

PROJECT_ROOT="${PROJECT_ROOT:-/projects/m000102}"
CODE_DIR="${PROJECT_ROOT}/code/levante-bench"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-$PROJECT_ROOT/envs/levante-bench-py311}"
VERSION="${VERSION:-v1}"
DEVICE="${DEVICE:-cuda}"

declare -A MODEL_SIZE_MAP=(
  ["qwen25vl_32b_hf"]=""
  ["qwen25vl_72b_hf"]=""
  ["qwen3vl_30b_hf"]=""
  ["qwen3vl_235b_hf"]=""
  ["aya_vision_32b_hf"]=""
  ["gemini_pro"]=""
  ["gpt52"]=""
  ["gpt53"]=""
)

DEFAULT_MODELS=(
  qwen25vl_32b_hf
  qwen25vl_72b_hf
  qwen3vl_30b_hf
  qwen3vl_235b_hf
  aya_vision_32b_hf
  gemini_pro
  gpt52
  gpt53
)

if [[ $# -gt 0 ]]; then
  MODELS=("$@")
else
  MODELS=("${DEFAULT_MODELS[@]}")
fi

echo "Submitting deterministic single-run hosted-model jobs:"
echo "  VERSION=$VERSION"
echo "  DEVICE=$DEVICE"
echo "  PROJECT_ROOT=$PROJECT_ROOT"
echo "  CONDA_ENV_PATH=$CONDA_ENV_PATH"
echo ""

for model in "${MODELS[@]}"; do
  if [[ -z "${MODEL_SIZE_MAP[$model]+x}" ]]; then
    echo "Skipping unknown hosted model key: $model" >&2
    continue
  fi

  size="${MODEL_SIZE_MAP[$model]}"
  results_root="$CODE_DIR/results/$model"

  echo "Submitting hosted model=$model size=${size:-<default>} results_root=$results_root"
  sbatch \
    --export=ALL,PROJECT_ROOT="$PROJECT_ROOT",CONDA_ENV_PATH="$CONDA_ENV_PATH",MODEL_NAME="$model",MODEL_SIZE="$size",VERSION="$VERSION",DEVICE="$DEVICE",BATCH_SIZE=1,NUM_RUNS=1,TRUE_RANDOM_OPTION_ORDER=false,USE_JOB_OUTPUT_ROOT=0,RESULTS_ROOT="$results_root" \
    "$SBATCH_SCRIPT"
done
