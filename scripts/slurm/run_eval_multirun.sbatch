#!/bin/bash
#
# Example Slurm launcher for runner-based multi-run evaluation.
#
# This is designed for clusters like Marlowe where each job should write under:
#   results/<version>/<model>/<job-label>/<0001...>
#
# The runner automatically derives <job-label> from SLURM_* variables when
# --true-random-option-order is enabled and --no-slurm-run-label is NOT set.
#
# Usage:
#   sbatch scripts/slurm/run_eval_multirun.sbatch
#
# Override key settings at submission time:
#   sbatch --export=ALL,MODEL=smolvlm2,VERSION=v1,NUM_RUNS=7,BATCH_SIZE=4 \
#     scripts/slurm/run_eval_multirun.sbatch
#
#SBATCH --job-name=levante-multirun
#SBATCH --output=results/slurm/%x-%j.out
#SBATCH --error=results/slurm/%x-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
if [[ ! -d "$PROJECT_ROOT/src/levante_bench" ]]; then
  echo "Error: PROJECT_ROOT does not look like levante-bench: $PROJECT_ROOT" >&2
  exit 1
fi
cd "$PROJECT_ROOT"

mkdir -p results/slurm

# Parameters (override with --export=ALL,... on sbatch command)
MODEL="${MODEL:-smolvlm2}"
VERSION="${VERSION:-v1}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_RUNS="${NUM_RUNS:-7}"
PROMPT_LANGUAGE="${PROMPT_LANGUAGE:-en}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-}"
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"

# Optional Conda activation (recommended on Marlowe).
if [[ -n "$CONDA_ENV_NAME" ]]; then
  if [[ ! -f "$CONDA_SH" ]]; then
    echo "Error: CONDA_SH not found at $CONDA_SH" >&2
    exit 1
  fi
  # shellcheck disable=SC1090
  source "$CONDA_SH"
  conda activate "$CONDA_ENV_NAME"
fi

if [[ -n "${PYTHON_BIN:-}" ]]; then
  :
elif [[ -n "$CONDA_ENV_NAME" ]]; then
  PYTHON_BIN="python"
else
  PYTHON_BIN=".venv/bin/python"
  if [[ ! -x "$PYTHON_BIN" ]]; then
    PYTHON_BIN="python3"
  fi
fi

echo "Starting multi-run eval on Slurm"
echo "  MODEL=${MODEL}"
echo "  VERSION=${VERSION}"
echo "  DEVICE=${DEVICE}"
echo "  BATCH_SIZE=${BATCH_SIZE}"
echo "  NUM_RUNS=${NUM_RUNS}"
echo "  SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "  SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-}"
echo "  CONDA_ENV_NAME=${CONDA_ENV_NAME:-}"
echo "  PYTHON_BIN=${PYTHON_BIN}"

PYTHONPATH=src "$PYTHON_BIN" -m levante_bench.cli run-eval \
  --model "$MODEL" \
  --version "$VERSION" \
  --device "$DEVICE" \
  --batch-size "$BATCH_SIZE" \
  --prompt-language "$PROMPT_LANGUAGE" \
  --true-random-option-order \
  --num-runs "$NUM_RUNS"

echo "Done."
