#!/bin/bash
# Submit top-up Slurm jobs for resampling chunks that are not complete yet.
#
# Default behavior:
# - target: 10 chunks (chunk_01..chunk_10)
# - target runs per chunk: 10 completed runs (summary.csv present)
# - wallclock: 4 hours
# - reruns only chunks with missing completed runs
#
# This is safe to run repeatedly. It uses existing chunk labels so runner output
# continues under the same parent folders.
#
# Usage:
#   bash scripts/slurm/resubmit_resampling_remaining.sh
#   WALLTIME=06:00:00 bash scripts/slurm/resubmit_resampling_remaining.sh
#   MODEL_NAME=qwen35 MODEL_SIZE=4B VERSION=v1 bash scripts/slurm/resubmit_resampling_remaining.sh

set -euo pipefail

if ! command -v sbatch >/dev/null 2>&1; then
  echo "ERROR: sbatch not found in PATH." >&2
  echo "Run this script on a Marlowe login/head node." >&2
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

MODEL_NAME="${MODEL_NAME:-qwen35}"
MODEL_SIZE="${MODEL_SIZE:-4B}"
VERSION="${VERSION:-v1}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
USE_JSON_FORMAT="${USE_JSON_FORMAT:-false}"

TOTAL_CHUNKS="${TOTAL_CHUNKS:-10}"
TARGET_RUNS_PER_CHUNK="${TARGET_RUNS_PER_CHUNK:-10}"
WALLTIME="${WALLTIME:-04:00:00}"

RESULTS_ROOT="${RESULTS_ROOT:-$CODE_DIR/results/resampling/${MODEL_NAME}-${MODEL_SIZE}}"
MODEL_ROOT="$RESULTS_ROOT/$VERSION/${MODEL_NAME}-${MODEL_SIZE}"

count_completed_runs() {
  local chunk_label="$1"
  local chunk_dir="$MODEL_ROOT/$chunk_label"
  if [[ ! -d "$chunk_dir" ]]; then
    echo 0
    return
  fi
  # Completed run = run folder with summary.csv.
  # shellcheck disable=SC2012
  ls -1d "$chunk_dir"/[0-9][0-9][0-9][0-9] 2>/dev/null | while read -r run_dir; do
    [[ -f "$run_dir/summary.csv" ]] && echo 1 || true
  done | wc -l | tr -d ' '
}

echo "Resubmitting incomplete resampling chunks:"
echo "  model=${MODEL_NAME} size=${MODEL_SIZE}"
echo "  version=${VERSION}, device=${DEVICE}, batch_size=${BATCH_SIZE}"
echo "  max_new_tokens=${MAX_NEW_TOKENS}, use_json_format=${USE_JSON_FORMAT}"
echo "  results_root=${RESULTS_ROOT}"
echo "  model_root=${MODEL_ROOT}"
echo "  target_runs_per_chunk=${TARGET_RUNS_PER_CHUNK}, total_chunks=${TOTAL_CHUNKS}"
echo "  walltime=${WALLTIME}"
echo ""

submitted=0
already_complete=0
for chunk in $(seq -w 1 "$TOTAL_CHUNKS"); do
  run_label="chunk_${chunk}"
  completed="$(count_completed_runs "$run_label")"
  remaining=$(( TARGET_RUNS_PER_CHUNK - completed ))
  if (( remaining <= 0 )); then
    echo "${run_label}: complete (${completed}/${TARGET_RUNS_PER_CHUNK}), skipping."
    already_complete=$((already_complete + 1))
    continue
  fi

  echo "${run_label}: completed=${completed}, remaining=${remaining} -> submitting top-up job."
  sbatch \
    --time="$WALLTIME" \
    --export=ALL,PROJECT_ROOT="$PROJECT_ROOT",CONDA_ENV_PATH="$CONDA_ENV_PATH",MODEL_NAME="$MODEL_NAME",MODEL_SIZE="$MODEL_SIZE",MAX_NEW_TOKENS="$MAX_NEW_TOKENS",USE_JSON_FORMAT="$USE_JSON_FORMAT",VERSION="$VERSION",DEVICE="$DEVICE",BATCH_SIZE="$BATCH_SIZE",NUM_RUNS="$remaining",TRUE_RANDOM_OPTION_ORDER=true,RUN_LABEL="$run_label",SLURM_RUN_LABEL=false,USE_JOB_OUTPUT_ROOT=0,RESULTS_ROOT="$RESULTS_ROOT" \
    "$SBATCH_SCRIPT"
  submitted=$((submitted + 1))
done

echo ""
echo "Done. submitted=${submitted}, already_complete=${already_complete}."
echo "After jobs finish, you can stitch completed runs with:"
echo "  python scripts/analysis/stitch_resampling_runs.py \\"
echo "    --source-root \"$MODEL_ROOT\" \\"
echo "    --output-root \"$RESULTS_ROOT/$VERSION/${MODEL_NAME}-${MODEL_SIZE}_r0100\""
