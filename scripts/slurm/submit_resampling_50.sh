#!/bin/bash
# Submit 10 Slurm launches of 5 true-random runs each (total 100).
#
# Default target model is qwen35-4B: typically much stronger than smolvlm2-256M
# while still practical for repeated hosted runs.
#
# Usage:
#   bash scripts/slurm/submit_resampling_50.sh
#   MODEL_NAME=internvl35 MODEL_SIZE=8B VERSION=v1_new_parser bash scripts/slurm/submit_resampling_50.sh

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
CODE_DIR="${PROJECT_ROOT}/code/levante-bench"

MODEL_NAME="${MODEL_NAME:-qwen35}"
MODEL_SIZE="${MODEL_SIZE:-4B}"
VERSION="${VERSION:-v1}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-2}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
USE_JSON_FORMAT="${USE_JSON_FORMAT:-true}"

TOTAL_LAUNCHES=10
RUNS_PER_LAUNCH=5 #down from 10 for better timing

RESULTS_ROOT="${RESULTS_ROOT:-$CODE_DIR/results/resampling/${MODEL_NAME}-${MODEL_SIZE}}"

echo "Submitting resampling workload:"
echo "  model=${MODEL_NAME} size=${MODEL_SIZE}"
echo "  launches=${TOTAL_LAUNCHES}, runs_per_launch=${RUNS_PER_LAUNCH} (target total=100)"
echo "  version=${VERSION}, device=${DEVICE}, batch_size=${BATCH_SIZE}"
echo "  max_new_tokens=${MAX_NEW_TOKENS}, use_json_format=${USE_JSON_FORMAT}"
echo "  results_root=${RESULTS_ROOT}"
echo ""

for chunk in $(seq -w 1 "$TOTAL_LAUNCHES"); do
  run_label="chunk_${chunk}"
  echo "Submitting ${run_label}..."
  sbatch \
    --export=PROJECT_ROOT="$PROJECT_ROOT",CONDA_ENV_PATH="$CONDA_ENV_PATH",MODEL_NAME="$MODEL_NAME",MODEL_SIZE="$MODEL_SIZE",MAX_NEW_TOKENS="$MAX_NEW_TOKENS",USE_JSON_FORMAT="$USE_JSON_FORMAT",VERSION="$VERSION",DEVICE="$DEVICE",BATCH_SIZE="$BATCH_SIZE",NUM_RUNS="$RUNS_PER_LAUNCH",TRUE_RANDOM_OPTION_ORDER=true,RUN_LABEL="$run_label",SLURM_RUN_LABEL=false,USE_JOB_OUTPUT_ROOT=0,RESULTS_ROOT="$RESULTS_ROOT",HF_TOKEN,HUGGINGFACEHUB_API_TOKEN,HF_HOME,HF_HUB_CACHE,TRANSFORMERS_CACHE \
    "$SBATCH_SCRIPT"
done

echo ""
echo "Submitted all chunks."
echo "After all jobs finish, run:"
echo "  python scripts/analysis/stitch_resampling_runs.py \\"
echo "    --source-root \"$RESULTS_ROOT/v1/${MODEL_NAME}-${MODEL_SIZE}\" \\"
echo "    --output-root \"$RESULTS_ROOT/v1/${MODEL_NAME}-${MODEL_SIZE}_r0100\""
