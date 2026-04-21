#!/bin/bash
# Submit one deterministic single-run Slurm job per local model.
#
# Behavior:
# - Runs all 6 tasks
# - num_runs=1
# - true_random_option_order=false (deterministic option order)
# - Writes under repo-local results/<model_name>/...
#
# Usage:
#   bash scripts/slurm/submit_all_local_models_once.sh
#   VERSION=v1_new_parser bash scripts/slurm/submit_all_local_models_once.sh
#   bash scripts/slurm/submit_all_local_models_once.sh internvl35 qwen3vl_30b

set -euo pipefail

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
  ["smolvlm2"]="2.2B"
  ["internvl35"]="8B"
  ["qwen35"]="4B"
  ["qwen3vl_30b"]=""
  ["qwen25vl_32b"]=""
  ["tinyllava"]=""
  ["aquila_vl"]=""
  ["gemma3"]="4b-it"
  ["gemma4"]="E4B-it"
)

declare -A BATCH_SIZE_MAP=(
  ["smolvlm2"]="1"
  ["internvl35"]="1"
  ["qwen35"]="1"
  ["qwen3vl_30b"]="1"
  ["qwen25vl_32b"]="1"
  ["tinyllava"]="1"
  ["aquila_vl"]="1"
  ["gemma3"]="1"
  ["gemma4"]="1"
)

DEFAULT_MODELS=(
  smolvlm2
  internvl35
  qwen35
  qwen3vl_30b
  qwen25vl_32b
  tinyllava
  aquila_vl
  gemma3
  gemma4
)

if [[ $# -gt 0 ]]; then
  MODELS=("$@")
else
  MODELS=("${DEFAULT_MODELS[@]}")
fi

echo "Submitting deterministic single-run local-model jobs:"
echo "  VERSION=$VERSION"
echo "  DEVICE=$DEVICE"
echo "  PROJECT_ROOT=$PROJECT_ROOT"
echo "  CONDA_ENV_PATH=$CONDA_ENV_PATH"
echo ""

for model in "${MODELS[@]}"; do
  if [[ -z "${BATCH_SIZE_MAP[$model]+x}" ]]; then
    echo "Skipping unknown model key: $model" >&2
    continue
  fi

  size="${MODEL_SIZE_MAP[$model]}"
  batch_size="${BATCH_SIZE_MAP[$model]}"
  results_root="$CODE_DIR/results/$model"

  echo "Submitting model=$model size=${size:-<default>} batch_size=$batch_size results_root=$results_root"
  sbatch \
    --export=ALL,PROJECT_ROOT="$PROJECT_ROOT",CONDA_ENV_PATH="$CONDA_ENV_PATH",MODEL_NAME="$model",MODEL_SIZE="$size",VERSION="$VERSION",DEVICE="$DEVICE",BATCH_SIZE="$batch_size",NUM_RUNS=1,TRUE_RANDOM_OPTION_ORDER=false,USE_JOB_OUTPUT_ROOT=0,RESULTS_ROOT="$results_root" \
    "$SBATCH_SCRIPT"
done
