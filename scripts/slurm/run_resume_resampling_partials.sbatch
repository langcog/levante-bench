#!/bin/bash
#SBATCH --job-name=levante-resume-partials
#SBATCH --partition=preempt
#SBATCH --account=marlowe-m000102
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/projects/m000102/outputs/slurm/%x-%j.out
#SBATCH --error=/projects/m000102/outputs/slurm/%x-%j.err
#SBATCH --export=NONE
#SBATCH --requeue

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/projects/m000102}"
CODE_DIR="${CODE_DIR:-$PROJECT_ROOT/code/levante-bench}"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-$PROJECT_ROOT/envs/levante-bench-py311}"
CUDA_MODULE="${CUDA_MODULE:-stockcuda/12.6.2}"
RUN_ROOT="${RUN_ROOT:?RUN_ROOT is required}"
DATA_ROOT="${DATA_ROOT:-$CODE_DIR/data}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-}"
USE_JSON_FORMAT="${USE_JSON_FORMAT:-}"
TASKS_CSV="${TASKS_CSV:-}"
MAX_RUNS="${MAX_RUNS:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

module purge
module load conda
module load "$CUDA_MODULE"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_PATH"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"

cd "$CODE_DIR"

CMD=(
  python scripts/slurm/resume_resampling_partials.py
  --run-root "$RUN_ROOT"
  --data-root "$DATA_ROOT"
  --device "$DEVICE"
)

if [[ -n "$BATCH_SIZE" ]]; then
  CMD+=(--batch-size "$BATCH_SIZE")
fi
if [[ -n "$MAX_NEW_TOKENS" ]]; then
  CMD+=(--max-new-tokens "$MAX_NEW_TOKENS")
fi
if [[ -n "$USE_JSON_FORMAT" ]]; then
  CMD+=(--use-json-format "$USE_JSON_FORMAT")
fi
if [[ -n "$TASKS_CSV" ]]; then
  tasks_normalized="${TASKS_CSV//;/,}"
  IFS=',' read -r -a TASK_LIST <<< "$tasks_normalized"
  for task in "${TASK_LIST[@]}"; do
    task_trimmed="$(echo "$task" | xargs)"
    if [[ -n "$task_trimmed" ]]; then
      CMD+=(--task "$task_trimmed")
    fi
  done
fi
if [[ -n "$MAX_RUNS" ]]; then
  CMD+=(--max-runs "$MAX_RUNS")
fi
if [[ -n "$EXTRA_ARGS" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARG_ARRAY=( $EXTRA_ARGS )
  CMD+=("${EXTRA_ARG_ARRAY[@]}")
fi

echo "Starting partial-run resume on $(hostname) at $(date)"
echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Conda env: $CONDA_ENV_PATH"
echo "Python: $(command -v python)"
echo "Command: ${CMD[*]}"

"${CMD[@]}"
