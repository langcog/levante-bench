#!/bin/bash
# Shared runner for LEVANTE Slurm wrappers.

set -euo pipefail

# Required from caller:
# - PROJECT_ROOT
# - EXPERIMENT_CONFIG
#
# Optional from caller:
# - CODE_DIR (default: "$PROJECT_ROOT/code/levante-bench")
# - SLURM_LOG_DIR (default: "$PROJECT_ROOT/outputs/slurm")
# - RESULTS_ROOT (default: "$PROJECT_ROOT/outputs/results")
# - CONDA_ENV_PATH (default: "$PROJECT_ROOT/envs/levante-bench-py311")
# - CUDA_MODULE (default: "stockcuda/12.6.2")
# - DEVICE (default: "cuda")
# - USE_JOB_OUTPUT_ROOT (default: "1")
# - EXTRA_OVERRIDES (space-separated dotlist args)

PROJECT_ROOT="${PROJECT_ROOT:?PROJECT_ROOT is required}"
EXPERIMENT_CONFIG="${EXPERIMENT_CONFIG:?EXPERIMENT_CONFIG is required}"

CODE_DIR="${CODE_DIR:-$PROJECT_ROOT/code/levante-bench}"
SLURM_LOG_DIR="${SLURM_LOG_DIR:-$PROJECT_ROOT/outputs/slurm}"
RESULTS_ROOT="${RESULTS_ROOT:-$PROJECT_ROOT/outputs/results}"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-$PROJECT_ROOT/envs/levante-bench-py311}"
CUDA_MODULE="${CUDA_MODULE:-stockcuda/12.6.2}"
DEVICE="${DEVICE:-cuda}"
USE_JOB_OUTPUT_ROOT="${USE_JOB_OUTPUT_ROOT:-1}"
EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"

module purge
module load conda
module load "$CUDA_MODULE"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_PATH"

mkdir -p "$SLURM_LOG_DIR"
mkdir -p "$RESULTS_ROOT"

cd "$CODE_DIR"

CMD=(./run_experiment.sh "$EXPERIMENT_CONFIG" "device=$DEVICE")

if [[ "$USE_JOB_OUTPUT_ROOT" == "1" ]]; then
  JOB_OUTPUT_ROOT="$RESULTS_ROOT/job_${SLURM_JOB_ID:-manual}"
  mkdir -p "$JOB_OUTPUT_ROOT"
  CMD+=("output_dir=$JOB_OUTPUT_ROOT")
fi

if [[ -n "$EXTRA_OVERRIDES" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARGS=( $EXTRA_OVERRIDES )
  CMD+=("${EXTRA_ARGS[@]}")
fi

echo "Starting levante-bench on $(hostname) at $(date)"
echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Experiment: $EXPERIMENT_CONFIG"
echo "Conda env: $CONDA_ENV_PATH"
echo "Python after activate: $(command -v python)"
echo "Python version: $(python -V 2>&1)"
export LEVANTE_PYTHON_BIN="python3"
echo "LEVANTE_PYTHON_BIN: $LEVANTE_PYTHON_BIN"
if [[ "$USE_JOB_OUTPUT_ROOT" == "1" ]]; then
  echo "Job output root: $JOB_OUTPUT_ROOT"
fi
echo "Command: ${CMD[*]}"

# Use conda run to guarantee execution in the requested env even in
# non-interactive Slurm shells where activation can be brittle.
conda run -p "$CONDA_ENV_PATH" "${CMD[@]}"
