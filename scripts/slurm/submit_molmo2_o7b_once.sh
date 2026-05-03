#!/bin/bash
set -euo pipefail

if ! command -v sbatch >/dev/null 2>&1; then
  echo "ERROR: sbatch not found in PATH." >&2
  echo "Run this on a Marlowe login node." >&2
  exit 127
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_SCRIPT="$SCRIPT_DIR/run_local_model_experiment.sbatch"

if [[ ! -f "$SBATCH_SCRIPT" ]]; then
  echo "ERROR: missing sbatch template: $SBATCH_SCRIPT" >&2
  exit 1
fi

PROJECT_ROOT="${PROJECT_ROOT:-/projects/m000102}"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-$PROJECT_ROOT/envs/levante-bench-py311}"
CODE_DIR="${PROJECT_ROOT}/code/levante-bench"

MODEL_NAME="${MODEL_NAME:-molmo2}"
MODEL_SIZE="${MODEL_SIZE:-O-7B}"
VERSION="${VERSION:-v1}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
USE_JSON_FORMAT="${USE_JSON_FORMAT:-true}"
WALLTIME="${WALLTIME:-02:00:00}"

RUN_LABEL="${RUN_LABEL:-o7b_single}"
RESULTS_ROOT="${RESULTS_ROOT:-$CODE_DIR/results/resampling/${MODEL_NAME}-${MODEL_SIZE}}"

echo "Submitting single Molmo2 O-7B run:"
echo "  model=${MODEL_NAME} size=${MODEL_SIZE}"
echo "  version=${VERSION}, device=${DEVICE}, batch_size=${BATCH_SIZE}"
echo "  max_new_tokens=${MAX_NEW_TOKENS}, use_json_format=${USE_JSON_FORMAT}"
echo "  walltime=${WALLTIME}"
echo "  run_label=${RUN_LABEL}"
echo "  results_root=${RESULTS_ROOT}"
echo ""

sbatch \
  --time="$WALLTIME" \
  --export=PROJECT_ROOT="$PROJECT_ROOT",CONDA_ENV_PATH="$CONDA_ENV_PATH",MODEL_NAME="$MODEL_NAME",MODEL_SIZE="$MODEL_SIZE",MAX_NEW_TOKENS="$MAX_NEW_TOKENS",USE_JSON_FORMAT="$USE_JSON_FORMAT",VERSION="$VERSION",DEVICE="$DEVICE",BATCH_SIZE="$BATCH_SIZE",NUM_RUNS=1,TRUE_RANDOM_OPTION_ORDER=true,RUN_LABEL="$RUN_LABEL",SLURM_RUN_LABEL=false,USE_JOB_OUTPUT_ROOT=0,RESULTS_ROOT="$RESULTS_ROOT",HF_TOKEN,HUGGINGFACEHUB_API_TOKEN,HF_HOME,HF_HUB_CACHE,TRANSFORMERS_CACHE \
  "$SBATCH_SCRIPT"

echo ""
echo "Submitted 1 Slurm job with NUM_RUNS=1."