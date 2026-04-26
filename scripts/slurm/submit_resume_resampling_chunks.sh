#!/bin/bash
# Submit one partial-resume Slurm job per resampling chunk.
#
# Each submitted job receives RUN_ROOT=<model_root>/chunk_XX, so jobs work on
# disjoint run folders and can safely run in parallel.
#
# Usage:
#   bash scripts/slurm/submit_resume_resampling_chunks.sh
#   MODEL_NAME=internvl35 MODEL_SIZE=8B VERSION=v1 bash scripts/slurm/submit_resume_resampling_chunks.sh

set -euo pipefail

if ! command -v sbatch >/dev/null 2>&1; then
  echo "ERROR: sbatch not found in PATH." >&2
  echo "Run this script on a Marlowe login/head node." >&2
  exit 127
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_SCRIPT="$SCRIPT_DIR/run_resume_resampling_partials.sbatch"

if [[ ! -f "$SBATCH_SCRIPT" ]]; then
  echo "ERROR: missing sbatch template: $SBATCH_SCRIPT" >&2
  exit 1
fi

PROJECT_ROOT="${PROJECT_ROOT:-/projects/m000102}"
CODE_DIR="${CODE_DIR:-$PROJECT_ROOT/code/levante-bench}"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-$PROJECT_ROOT/envs/levante-bench-py311}"

MODEL_NAME="${MODEL_NAME:-qwen35}"
MODEL_SIZE="${MODEL_SIZE:-4B}"
VERSION="${VERSION:-v1}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-2}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
USE_JSON_FORMAT="${USE_JSON_FORMAT:-true}"
TOTAL_CHUNKS="${TOTAL_CHUNKS:-10}"
WALLTIME="${WALLTIME:-04:00:00}"

RESULTS_ROOT="${RESULTS_ROOT:-$CODE_DIR/results/resampling/${MODEL_NAME}-${MODEL_SIZE}}"
MODEL_ROOT="${MODEL_ROOT:-$RESULTS_ROOT/$VERSION/${MODEL_NAME}-${MODEL_SIZE}}"

echo "Submitting one partial-resume job per chunk:"
echo "  model=${MODEL_NAME} size=${MODEL_SIZE}"
echo "  version=${VERSION}, device=${DEVICE}, batch_size=${BATCH_SIZE}"
echo "  max_new_tokens=${MAX_NEW_TOKENS}, use_json_format=${USE_JSON_FORMAT}"
echo "  model_root=${MODEL_ROOT}"
echo "  total_chunks=${TOTAL_CHUNKS}, walltime=${WALLTIME}"
echo ""

submitted=0
for chunk in $(seq -w 1 "$TOTAL_CHUNKS"); do
  run_label="chunk_${chunk}"
  run_root="$MODEL_ROOT/$run_label"

  if [[ ! -d "$run_root" ]]; then
    echo "${run_label}: missing directory ${run_root}, skipping."
    continue
  fi

  echo "${run_label}: submitting resume job for ${run_root}"
  sbatch \
    --job-name="levante-resume-${MODEL_NAME}-${MODEL_SIZE}-${run_label}" \
    --time="$WALLTIME" \
    --export=ALL,PROJECT_ROOT="$PROJECT_ROOT",CODE_DIR="$CODE_DIR",CONDA_ENV_PATH="$CONDA_ENV_PATH",RUN_ROOT="$run_root",DEVICE="$DEVICE",BATCH_SIZE="$BATCH_SIZE",MAX_NEW_TOKENS="$MAX_NEW_TOKENS",USE_JSON_FORMAT="$USE_JSON_FORMAT" \
    "$SBATCH_SCRIPT"
  submitted=$((submitted + 1))
done

echo ""
echo "Submitted ${submitted} chunk resume job(s)."
