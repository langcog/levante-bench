#!/bin/bash
#SBATCH --job-name=levante-local-model
#SBATCH --partition=preempt
#SBATCH --account=marlowe-m000102
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/projects/m000102/outputs/slurm/%x-%j.out
#SBATCH --error=/projects/m000102/outputs/slurm/%x-%j.err
#SBATCH --requeue

set -euo pipefail

# Required
MODEL_NAME="${MODEL_NAME:?MODEL_NAME is required (e.g., internvl35)}"

# Optional runtime knobs (override via sbatch --export=ALL,...)
MODEL_SIZE="${MODEL_SIZE:-}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-}"
VERSION="${VERSION:-v1}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_RUNS="${NUM_RUNS:-10}"
TRUE_RANDOM_OPTION_ORDER="${TRUE_RANDOM_OPTION_ORDER:-true}"
# Optional true-random grouping controls.
RUN_LABEL="${RUN_LABEL:-}"
SLURM_RUN_LABEL="${SLURM_RUN_LABEL:-true}"
USE_JOB_OUTPUT_ROOT="${USE_JOB_OUTPUT_ROOT:-1}"
PROJECT_ROOT="${PROJECT_ROOT:-/projects/m000102}"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-$PROJECT_ROOT/envs/levante-bench-py311}"
# Always run the full benchmark task set.
TASKS_CSV="egma-math,matrix-reasoning,mental-rotation,theory-of-mind,trog,vocab"

CODE_DIR="${PROJECT_ROOT}/code/levante-bench"
if [[ ! -d "$CODE_DIR" ]]; then
  echo "ERROR: CODE_DIR not found: $CODE_DIR" >&2
  exit 1
fi

mkdir -p "$PROJECT_ROOT/outputs/slurm"

EXPERIMENT_CONFIG="$(mktemp "/tmp/levante-${MODEL_NAME}-${SLURM_JOB_ID:-manual}-XXXX.yaml")"
trap 'rm -f "$EXPERIMENT_CONFIG"' EXIT

{
  echo "models:"
  echo "  - name: ${MODEL_NAME}"
  if [[ -n "$MODEL_SIZE" ]]; then
    echo "    size: ${MODEL_SIZE}"
  fi
  if [[ -n "$MAX_NEW_TOKENS" ]]; then
    echo "    max_new_tokens: ${MAX_NEW_TOKENS}"
  fi
  echo ""
  echo "tasks:"
  IFS=',' read -r -a TASK_LIST <<< "$TASKS_CSV"
  for task in "${TASK_LIST[@]}"; do
    task_trimmed="$(echo "$task" | xargs)"
    if [[ -n "$task_trimmed" ]]; then
      echo "  - ${task_trimmed}"
    fi
  done
  echo ""
  echo "data_root: data"
  echo "output_dir: results"
  echo "version: ${VERSION}"
  echo "device: ${DEVICE}"
  echo "batch_size: ${BATCH_SIZE}"
  echo "true_random_option_order: ${TRUE_RANDOM_OPTION_ORDER}"
  echo "num_runs: ${NUM_RUNS}"
  if [[ -n "$RUN_LABEL" ]]; then
    echo "run_label: ${RUN_LABEL}"
  fi
  echo "slurm_run_label: ${SLURM_RUN_LABEL}"
} > "$EXPERIMENT_CONFIG"

echo "Generated experiment config: $EXPERIMENT_CONFIG"
sed -n '1,120p' "$EXPERIMENT_CONFIG"

export PROJECT_ROOT
export CONDA_ENV_PATH
export USE_JOB_OUTPUT_ROOT
export EXPERIMENT_CONFIG
# Include model name in output root when launching multiple model jobs.
export RESULTS_ROOT="${RESULTS_ROOT:-$PROJECT_ROOT/outputs/results/$MODEL_NAME}"

source "$CODE_DIR/scripts/slurm/_levante_common.sh"
