#!/bin/bash
# Submit CogVLM benchmark job using predownloaded /scratch snapshots.
#
# Usage:
#   bash scripts/slurm/submit_cogvlm_from_scratch.sh
#   SMOKE=1 bash scripts/slurm/submit_cogvlm_from_scratch.sh
#   PREFETCH=1 bash scripts/slurm/submit_cogvlm_from_scratch.sh

set -euo pipefail

if ! command -v sbatch >/dev/null 2>&1; then
  echo "ERROR: sbatch not found in PATH." >&2
  exit 127
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_SCRIPT="$SCRIPT_DIR/run_cogvlm_from_scratch.sbatch"
PREFETCH_SCRIPT="$SCRIPT_DIR/prefetch_cogvlm_to_scratch.sh"

if [[ ! -f "$SBATCH_SCRIPT" ]]; then
  echo "ERROR: missing sbatch script: $SBATCH_SCRIPT" >&2
  exit 1
fi

PROJECT_ROOT="${PROJECT_ROOT:-/projects/m000102}"
CODE_DIR="${CODE_DIR:-$PROJECT_ROOT/code/levante-bench}"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-$PROJECT_ROOT/envs/levante-cogvlm}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/m000102/$USER/levante-hf}"
VERSION="${VERSION:-v1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$CODE_DIR/results/v1_additional_models}"
SMOKE="${SMOKE:-0}"
PREFETCH="${PREFETCH:-0}"
DRY_RUN="${DRY_RUN:-0}"

TASKS_CSV="${TASKS_CSV:-egma-math,matrix-reasoning,mental-rotation,theory-of-mind,trog,vocab}"
TIME_LIMIT="${TIME_LIMIT:-12:00:00}"
MEM_LIMIT="${MEM_LIMIT:-220G}"

if [[ "$SMOKE" == "1" ]]; then
  TASKS_CSV="vocab"
  TIME_LIMIT="${TIME_LIMIT:-01:30:00}"
  MEM_LIMIT="${MEM_LIMIT:-160G}"
fi

if [[ "$PREFETCH" == "1" ]]; then
  echo "Running prefetch script first..."
  SCRATCH_ROOT="$SCRATCH_ROOT" ENV_PATH="$CONDA_ENV_PATH" bash "$PREFETCH_SCRIPT"
fi

cmd=(
  sbatch
  --job-name historic-cogvlm
  --time "$TIME_LIMIT"
  --mem "$MEM_LIMIT"
  --export "PROJECT_ROOT=$PROJECT_ROOT,CODE_DIR=$CODE_DIR,CONDA_ENV_PATH=$CONDA_ENV_PATH,SCRATCH_ROOT=$SCRATCH_ROOT,VERSION=$VERSION,DEVICE=cuda,BATCH_SIZE=1,NUM_RUNS=1,TRUE_RANDOM_OPTION_ORDER=false,TASKS_CSV=$TASKS_CSV,OUTPUT_ROOT=$OUTPUT_ROOT,USE_JOB_OUTPUT_ROOT=0,PYTHONNOUSERSITE=1,PYTHONPATH=$CODE_DIR/src,HF_HOME=$SCRATCH_ROOT/hf-home,HF_HUB_CACHE=$SCRATCH_ROOT/hub,TRANSFORMERS_CACHE=$SCRATCH_ROOT/transformers,HF_HUB_OFFLINE=1,HF_HUB_DISABLE_TELEMETRY=1,HF_TOKEN,HUGGINGFACEHUB_API_TOKEN"
  "$SBATCH_SCRIPT"
)

echo "Submitting CogVLM from scratch cache:"
echo "  CONDA_ENV_PATH=$CONDA_ENV_PATH"
echo "  SCRATCH_ROOT=$SCRATCH_ROOT"
echo "  TASKS_CSV=$TASKS_CSV"
echo "  TIME_LIMIT=$TIME_LIMIT"
echo "  MEM_LIMIT=$MEM_LIMIT"

if [[ "$DRY_RUN" == "1" ]]; then
  printf 'DRY_RUN:'
  printf ' %q' "${cmd[@]}"
  printf '\n'
else
  "${cmd[@]}"
fi
