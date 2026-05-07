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

if [[ ! -d "$CODE_DIR" ]]; then
  echo "ERROR: CODE_DIR not found: $CODE_DIR" >&2
  exit 1
fi

mkdir -p "$SLURM_LOG_DIR" "$OUTPUT_ROOT"

module purge
module load conda
module load "$CUDA_MODULE"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_PATH"

export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export PYTHONPATH="${PYTHONPATH:-src}"

cd "$CODE_DIR"

# Build --task flags from TASKS_CSV
tasks_normalized="${TASKS_CSV//;/,}"
IFS=',' read -r -a TASK_LIST <<< "$tasks_normalized"
TASK_ARGS=()
for task in "${TASK_LIST[@]}"; do
  t="$(echo "$task" | xargs)"
  [[ -n "$t" ]] && TASK_ARGS+=(--task "$t")
done

CMD=(
  python -m levante_bench.cli run-eval
  --model "$MODEL_ID"
  --version "$VERSION"
  --device "$DEVICE"
  --output-dir "$OUTPUT_ROOT"
  --batch-size "$BATCH_SIZE"
  --num-runs "$NUM_RUNS"
)

if [[ "$TRUE_RANDOM_OPTION_ORDER" == "true" ]]; then
  CMD+=(--true-random-option-order)
fi

CMD+=("${TASK_ARGS[@]}")

echo "Starting historic local VLM job on $(hostname) at $(date)"
echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Model: $MODEL_ID"
echo "Conda env: $CONDA_ENV_PATH"
echo "Python: $(command -v python)"
echo "Command: ${CMD[*]}"

"${CMD[@]}"

