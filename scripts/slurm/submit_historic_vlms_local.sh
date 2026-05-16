#!/bin/bash
# Submit one Marlowe job per historic VLM local model.
#
# Usage:
#   bash scripts/slurm/submit_historic_vlms_local.sh
#   DRY_RUN=1 bash scripts/slurm/submit_historic_vlms_local.sh
#   VERSION=v1_additional_images bash scripts/slurm/submit_historic_vlms_local.sh
#   bash scripts/slurm/submit_historic_vlms_local.sh llava15_13b cogvlm openflamingo9b

set -euo pipefail

if ! command -v sbatch >/dev/null 2>&1; then
  echo "ERROR: sbatch not found in PATH." >&2
  echo "Run this script on a Marlowe login node." >&2
  exit 127
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_SCRIPT="$SCRIPT_DIR/run_historic_vlm_local.sbatch"

if [[ ! -f "$SBATCH_SCRIPT" ]]; then
  echo "ERROR: missing sbatch template: $SBATCH_SCRIPT" >&2
  exit 1
fi

PROJECT_ROOT="${PROJECT_ROOT:-/projects/m000102}"
CODE_DIR="${CODE_DIR:-$PROJECT_ROOT/code/levante-bench}"
VERSION="${VERSION:-v1}"
DEVICE="${DEVICE:-cuda}"
NUM_RUNS="${NUM_RUNS:-1}"
TRUE_RANDOM_OPTION_ORDER="${TRUE_RANDOM_OPTION_ORDER:-false}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$CODE_DIR/results/v1_additional_models}"
DRY_RUN="${DRY_RUN:-0}"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-$PROJECT_ROOT/envs/levante-bench-py311}"
IMAGE_SIZE="${IMAGE_SIZE:-}"

# Per-model defaults. Override globally via env vars if needed.
declare -A TIME_MAP=(
  ["llava15_13b"]="08:00:00"
  ["cogvlm"]="10:00:00"
  ["openflamingo9b"]="10:00:00"
)
declare -A MEM_MAP=(
  ["llava15_13b"]="96G"
  ["cogvlm"]="120G"
  ["openflamingo9b"]="120G"
)
declare -A BATCH_SIZE_MAP=(
  ["llava15_13b"]="1"
  ["cogvlm"]="1"
  ["openflamingo9b"]="1"
)
DEFAULT_MODELS=(
  "llava15_13b"
  "cogvlm"
  "openflamingo9b"
)

if [[ $# -gt 0 ]]; then
  MODELS=("$@")
else
  MODELS=("${DEFAULT_MODELS[@]}")
fi

echo "Submitting historic local VLM jobs:"
echo "  PROJECT_ROOT=$PROJECT_ROOT"
echo "  CODE_DIR=$CODE_DIR"
echo "  VERSION=$VERSION"
echo "  DEVICE=$DEVICE"
echo "  OUTPUT_ROOT=$OUTPUT_ROOT"
echo "  NUM_RUNS=$NUM_RUNS"
echo "  TRUE_RANDOM_OPTION_ORDER=$TRUE_RANDOM_OPTION_ORDER"
echo "  DRY_RUN=$DRY_RUN"
echo "  CONDA_ENV_PATH=$CONDA_ENV_PATH"
echo "  IMAGE_SIZE=${IMAGE_SIZE:-<default>}"
echo ""

for model in "${MODELS[@]}"; do
  if [[ -z "${TIME_MAP[$model]+x}" ]]; then
    echo "Skipping unknown model target: $model" >&2
    continue
  fi

  time_limit="${TIME_MAP[$model]}"
  mem_limit="${MEM_MAP[$model]}"
  batch_size="${BATCH_SIZE_MAP[$model]}"
  job_name="historic-${model}"

  cmd=(
    sbatch
    --job-name "$job_name"
    --time "$time_limit"
    --mem "$mem_limit"
    --export "ALL,PROJECT_ROOT=$PROJECT_ROOT,CODE_DIR=$CODE_DIR,CONDA_ENV_PATH=$CONDA_ENV_PATH,MODEL_ID=$model,VERSION=$VERSION,DEVICE=$DEVICE,BATCH_SIZE=$batch_size,NUM_RUNS=$NUM_RUNS,TRUE_RANDOM_OPTION_ORDER=$TRUE_RANDOM_OPTION_ORDER,IMAGE_SIZE=$IMAGE_SIZE,OUTPUT_ROOT=$OUTPUT_ROOT,USE_JOB_OUTPUT_ROOT=0,HF_TOKEN,HUGGINGFACEHUB_API_TOKEN,HF_HOME,HF_HUB_CACHE,TRANSFORMERS_CACHE"
    "$SBATCH_SCRIPT"
  )

  echo "Model=$model time=$time_limit mem=$mem_limit env=$CONDA_ENV_PATH batch_size=$batch_size"
  if [[ "$DRY_RUN" == "1" ]]; then
    printf 'DRY_RUN:'
    printf ' %q' "${cmd[@]}"
    printf '\n'
  else
    "${cmd[@]}"
  fi
done

