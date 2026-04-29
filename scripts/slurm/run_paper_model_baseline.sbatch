#!/bin/bash
#SBATCH --job-name=levante-paper-model
#SBATCH --partition=preempt
#SBATCH --account=marlowe-m000102
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/projects/m000102/outputs/slurm/%x-%j.out
#SBATCH --error=/projects/m000102/outputs/slurm/%x-%j.err
#SBATCH --requeue

set -euo pipefail

MODEL_NAME="${MODEL_NAME:?MODEL_NAME is required (e.g., internvl35)}"
MODEL_SIZE="${MODEL_SIZE:?MODEL_SIZE is required (e.g., 8B)}"

MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
USE_JSON_FORMAT="${USE_JSON_FORMAT:-}"
VERSION="${VERSION:-v1}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-1}"
PROJECT_ROOT="${PROJECT_ROOT:-/projects/m000102}"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-$PROJECT_ROOT/envs/levante-bench-py311}"
TASKS_CSV="${TASKS_CSV:-egma-math,matrix-reasoning,mental-rotation,theory-of-mind,trog,vocab}"
BASELINE_ALIAS="${BASELINE_ALIAS:-1}"
USE_JOB_OUTPUT_ROOT="${USE_JOB_OUTPUT_ROOT:-0}"

CODE_DIR="${PROJECT_ROOT}/code/levante-bench"
if [[ ! -d "$CODE_DIR" ]]; then
  echo "ERROR: CODE_DIR not found: $CODE_DIR" >&2
  exit 1
fi

mkdir -p "$PROJECT_ROOT/outputs/slurm"

EXPERIMENT_CONFIG="$(mktemp "/tmp/levante-paper-${MODEL_NAME}-${SLURM_JOB_ID:-manual}-XXXX.yaml")"
trap 'rm -f "$EXPERIMENT_CONFIG"' EXIT

{
  echo "models:"
  echo "  - name: ${MODEL_NAME}"
  echo "    size: ${MODEL_SIZE}"
  echo "    max_new_tokens: ${MAX_NEW_TOKENS}"
  if [[ -n "$USE_JSON_FORMAT" ]]; then
    echo "    use_json_format: ${USE_JSON_FORMAT}"
  fi
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
  echo "true_random_option_order: false"
  echo "num_runs: 1"
} > "$EXPERIMENT_CONFIG"

echo "Generated experiment config: $EXPERIMENT_CONFIG"
sed -n '1,120p' "$EXPERIMENT_CONFIG"

export PROJECT_ROOT
export CONDA_ENV_PATH
export USE_JOB_OUTPUT_ROOT
export EXPERIMENT_CONFIG
export RESULTS_ROOT="${RESULTS_ROOT:-$PROJECT_ROOT/outputs/results/$MODEL_NAME}"

source "$CODE_DIR/scripts/slurm/_levante_common.sh"

if [[ "$BASELINE_ALIAS" == "1" ]]; then
  MODEL_LABEL="${MODEL_NAME}-${MODEL_SIZE}"
  MODEL_OUTPUT_DIR="$CODE_DIR/results/$VERSION/$MODEL_LABEL"
  BASELINE_DIR="$MODEL_OUTPUT_DIR/baseline"

  if [[ ! -f "$MODEL_OUTPUT_DIR/summary.csv" ]]; then
    echo "WARNING: cannot create baseline alias; missing $MODEL_OUTPUT_DIR/summary.csv" >&2
    exit 0
  fi

  mkdir -p "$BASELINE_DIR"
  shopt -s nullglob
  for output_file in "$MODEL_OUTPUT_DIR"/*.csv "$MODEL_OUTPUT_DIR"/*.json "$MODEL_OUTPUT_DIR"/*.npy; do
    cp -f "$output_file" "$BASELINE_DIR/"
  done
  if [[ -f "$MODEL_OUTPUT_DIR/cache/responses.json" ]]; then
    mkdir -p "$BASELINE_DIR/cache"
    cp -f "$MODEL_OUTPUT_DIR/cache/responses.json" "$BASELINE_DIR/cache/responses.json"
  fi
  echo "Created dashboard baseline alias: $BASELINE_DIR"
fi
