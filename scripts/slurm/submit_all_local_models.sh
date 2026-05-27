#!/bin/bash
# Submit one Slurm job per local model using run_local_model_experiment.sbatch.
#
# Usage:
#   bash scripts/slurm/submit_all_local_models.sh
#   VERSION=v1_new_parser NUM_RUNS=5 bash scripts/slurm/submit_all_local_models.sh
#   bash scripts/slurm/submit_all_local_models.sh internvl35:8B qwen35:4B

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENSURE_SBATCH_SCRIPT="$SCRIPT_DIR/_ensure_sbatch.sh"
if [[ ! -f "$ENSURE_SBATCH_SCRIPT" ]]; then
  echo "ERROR: missing sbatch helper: $ENSURE_SBATCH_SCRIPT" >&2
  exit 1
fi
# shellcheck disable=SC1090
source "$ENSURE_SBATCH_SCRIPT"
ensure_sbatch_available
SBATCH_SCRIPT="$SCRIPT_DIR/run_local_model_experiment.sbatch"

if [[ ! -f "$SBATCH_SCRIPT" ]]; then
  echo "ERROR: missing sbatch template: $SBATCH_SCRIPT" >&2
  exit 1
fi

PROJECT_ROOT="${PROJECT_ROOT:-/projects/m000102}"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-$PROJECT_ROOT/envs/levante-bench-py311}"
VERSION="${VERSION:-v1}"
DEVICE="${DEVICE:-cuda}"
NUM_RUNS="${NUM_RUNS:-10}"
TRUE_RANDOM_OPTION_ORDER="${TRUE_RANDOM_OPTION_ORDER:-true}"
# Always run all six benchmark tasks for this bulk launcher.
TASKS_CSV="egma-math,matrix-reasoning,mental-rotation,theory-of-mind,trog,vocab"

# Safe defaults for Marlowe single-GPU runs.
declare -A BATCH_SIZE_MAP=(
  ["gemma4:E2B-it"]="1"
  ["gemma4:E4B-it"]="1"
  ["gemma4:26B-A4B-it"]="1"
  ["gemma4:31B-it"]="1"
  ["internvl35:1B"]="1"
  ["internvl35:2B"]="1"
  ["internvl35:4B"]="1"
  ["internvl35:8B"]="1"
  ["internvl35:14B"]="1"
  ["internvl35:38B"]="1"
  ["molmo2:4B"]="1"
  ["molmo2:O-7B"]="1"
  ["molmo2:8B"]="1"
  ["qwen35:0.8B"]="1"
  ["qwen35:2B"]="1"
  ["qwen35:4B"]="1"
  ["qwen35:9B"]="1"
  ["qwen35:27B"]="1"
  ["smolvlm2:256M"]="1"
  ["smolvlm2:500M"]="1"
  ["smolvlm2:2.2B"]="1"
  ["tinyllava:2.4B"]="1"
  ["tinyllava:3.1B"]="1"
)

DEFAULT_TARGETS=(
  "gemma4:E2B-it"
  "gemma4:E4B-it"
  "gemma4:26B-A4B-it"
  "gemma4:31B-it"
  "internvl35:1B"
  "internvl35:2B"
  "internvl35:4B"
  "internvl35:8B"
  "internvl35:14B"
  "internvl35:38B"
  "molmo2:4B"
  "molmo2:O-7B"
  "molmo2:8B"
  "qwen35:0.8B"
  "qwen35:2B"
  "qwen35:4B"
  "qwen35:9B"
  "qwen35:27B"
  "smolvlm2:256M"
  "smolvlm2:500M"
  "smolvlm2:2.2B"
  "tinyllava:2.4B"
  "tinyllava:3.1B"
)

if [[ $# -gt 0 ]]; then
  TARGETS=("$@")
else
  TARGETS=("${DEFAULT_TARGETS[@]}")
fi

echo "Submitting local-model Marlowe jobs:"
echo "  VERSION=$VERSION"
echo "  DEVICE=$DEVICE"
echo "  NUM_RUNS=$NUM_RUNS"
echo "  TRUE_RANDOM_OPTION_ORDER=$TRUE_RANDOM_OPTION_ORDER"
echo "  TASKS_CSV=$TASKS_CSV (fixed full task set)"
echo "  PROJECT_ROOT=$PROJECT_ROOT"
echo "  CONDA_ENV_PATH=$CONDA_ENV_PATH"
echo ""

for target in "${TARGETS[@]}"; do
  if [[ "$target" != *:* ]]; then
    echo "Skipping malformed target, expected model:size: $target" >&2
    continue
  fi
  if [[ -z "${BATCH_SIZE_MAP[$target]+x}" ]]; then
    echo "Skipping unknown paper target: $target" >&2
    continue
  fi

  model="${target%%:*}"
  size="${target#*:}"
  batch_size="${BATCH_SIZE_MAP[$target]}"

  echo "Submitting model=$model size=${size:-<default>} batch_size=$batch_size"
  sbatch \
    --export=PROJECT_ROOT="$PROJECT_ROOT",CONDA_ENV_PATH="$CONDA_ENV_PATH",MODEL_NAME="$model",MODEL_SIZE="$size",VERSION="$VERSION",DEVICE="$DEVICE",BATCH_SIZE="$batch_size",NUM_RUNS="$NUM_RUNS",TRUE_RANDOM_OPTION_ORDER="$TRUE_RANDOM_OPTION_ORDER",TASKS_CSV="$TASKS_CSV",HF_TOKEN,HUGGINGFACEHUB_API_TOKEN,HF_HOME,HF_HUB_CACHE,TRANSFORMERS_CACHE \
    "$SBATCH_SCRIPT"
done
