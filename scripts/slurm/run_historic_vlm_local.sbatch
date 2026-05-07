#!/bin/bash
#SBATCH --job-name=historic-vlm
#SBATCH --partition=preempt
#SBATCH --account=marlowe-m000102
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/projects/m000102/outputs/slurm/%x-%j.out
#SBATCH --error=/projects/m000102/outputs/slurm/%x-%j.err
#SBATCH --export=NONE
#SBATCH --requeue

set -euo pipefail

# Required
MODEL_ID="${MODEL_ID:?MODEL_ID is required (e.g., llava15_13b)}"

# Optional runtime knobs (override via sbatch --export)
VERSION="${VERSION:-v1}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_RUNS="${NUM_RUNS:-1}"
TRUE_RANDOM_OPTION_ORDER="${TRUE_RANDOM_OPTION_ORDER:-false}"
TASKS_CSV="${TASKS_CSV:-egma-math,matrix-reasoning,mental-rotation,theory-of-mind,trog,vocab}"

PROJECT_ROOT="${PROJECT_ROOT:-/projects/m000102}"
CODE_DIR="${CODE_DIR:-${PROJECT_ROOT}/code/levante-bench}"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-$PROJECT_ROOT/envs/levante-bench-py311}"
CUDA_MODULE="${CUDA_MODULE:-stockcuda/12.6.2}"
SLURM_LOG_DIR="${SLURM_LOG_DIR:-$PROJECT_ROOT/outputs/slurm}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$CODE_DIR/results/v1_additional_models}"
USE_JOB_OUTPUT_ROOT="${USE_JOB_OUTPUT_ROOT:-0}"

if [[ ! -d "$CODE_DIR" ]]; then
  echo "ERROR: CODE_DIR not found: $CODE_DIR" >&2
  exit 1
fi

mkdir -p "$SLURM_LOG_DIR" "$OUTPUT_ROOT"

EXPERIMENT_CONFIG="$(mktemp "/tmp/levante-historic-${MODEL_ID}-${SLURM_JOB_ID:-manual}-XXXX.yaml")"
trap 'rm -f "$EXPERIMENT_CONFIG"' EXIT

{
  echo "models:"
  echo "  - name: ${MODEL_ID}"
  echo ""
  echo "tasks:"
  tasks_normalized="${TASKS_CSV//;/,}"
  IFS=',' read -r -a TASK_LIST <<< "$tasks_normalized"
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
} > "$EXPERIMENT_CONFIG"

echo "Generated experiment config: $EXPERIMENT_CONFIG"
sed -n '1,120p' "$EXPERIMENT_CONFIG"

export PROJECT_ROOT
export CONDA_ENV_PATH
export USE_JOB_OUTPUT_ROOT
export EXPERIMENT_CONFIG
export RESULTS_ROOT="$OUTPUT_ROOT"
export CUDA_MODULE

source "$CODE_DIR/scripts/slurm/_levante_common.sh"

