#!/bin/bash
# Resume partial large-model true-random resampling chunks.
#
# This launcher scans:
#   results/resampling/<model-size>/<version>/<model-size>/chunk_XX
# and submits run_resume_resampling_partials.sbatch for chunks that contain
# run folders with metadata.json but no summary.csv.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENSURE_SBATCH_SCRIPT="$SCRIPT_DIR/_ensure_sbatch.sh"
if [[ ! -f "$ENSURE_SBATCH_SCRIPT" ]]; then
  echo "ERROR: missing sbatch helper: $ENSURE_SBATCH_SCRIPT" >&2
  exit 1
fi
# shellcheck disable=SC1090
source "$ENSURE_SBATCH_SCRIPT"
SBATCH_SCRIPT="$SCRIPT_DIR/run_resume_resampling_partials.sbatch"

if [[ ! -f "$SBATCH_SCRIPT" ]]; then
  echo "ERROR: missing sbatch template: $SBATCH_SCRIPT" >&2
  exit 1
fi

PROJECT_ROOT="${PROJECT_ROOT:-/projects/m000102}"
CODE_DIR="${CODE_DIR:-$PROJECT_ROOT/code/levante-bench}"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-$PROJECT_ROOT/envs/levante-bench-py311}"
VERSION="${VERSION:-v1}"
TOTAL_RUNS="${TOTAL_RUNS:-40}"
DRY_RUN="${DRY_RUN:-0}"

if [[ "$DRY_RUN" != "1" ]]; then
  ensure_sbatch_available
fi

DEFAULT_TARGETS=(
  "gemma4:26B-A4B-it"
  "gemma4:31B-it"
  "internvl35:14B"
  "internvl35:38B"
  "molmo2:4B"
  "molmo2:O-7B"
  "molmo2:8B"
  "qwen35:9B"
  "qwen35:27B"
  "smolvlm2:2.2B"
  "tinyllava:2.4B"
  "tinyllava:3.1B"
)

if [[ -n "${ONLY:-}" ]]; then
  # shellcheck disable=SC2206
  TARGETS=( $ONLY )
else
  TARGETS=("${DEFAULT_TARGETS[@]}")
fi

declare -A BATCH_SIZE_MAP=(
  ["gemma4:26B-A4B-it"]="1"
  ["gemma4:31B-it"]="1"
  ["internvl35:14B"]="1"
  ["internvl35:38B"]="1"
  ["molmo2:4B"]="1"
  ["molmo2:O-7B"]="1"
  ["molmo2:8B"]="1"
  ["qwen35:9B"]="1"
  ["qwen35:27B"]="1"
  ["smolvlm2:2.2B"]="1"
  ["tinyllava:2.4B"]="1"
  ["tinyllava:3.1B"]="1"
)

declare -A GPU_MAP=(
  ["gemma4:26B-A4B-it"]="1"
  ["gemma4:31B-it"]="1"
  ["internvl35:14B"]="1"
  ["internvl35:38B"]="1"
  ["qwen35:27B"]="1"
)

declare -A MEM_MAP=(
  ["gemma4:26B-A4B-it"]="144G"
  ["gemma4:31B-it"]="160G"
  ["internvl35:14B"]="128G"
  ["internvl35:38B"]="160G"
  ["qwen35:27B"]="144G"
)

declare -A TIME_MAP=(
  ["gemma4:26B-A4B-it"]="02:00:00"
  ["gemma4:31B-it"]="02:00:00"
  ["internvl35:14B"]="02:00:00"
  ["internvl35:38B"]="02:00:00"
  ["molmo2:4B"]="01:30:00"
  ["molmo2:O-7B"]="01:30:00"
  ["molmo2:8B"]="01:30:00"
  ["qwen35:9B"]="02:00:00"
  ["qwen35:27B"]="02:00:00"
  ["smolvlm2:2.2B"]="01:00:00"
  ["tinyllava:2.4B"]="01:00:00"
  ["tinyllava:3.1B"]="01:00:00"
)

chunk_has_partial_runs() {
  local chunk_dir="$1"
  [[ -d "$chunk_dir" ]] || return 1
  local run_dir
  for run_dir in "$chunk_dir"/[0-9][0-9][0-9][0-9]; do
    [[ -d "$run_dir" ]] || continue
    if [[ -f "$run_dir/metadata.json" && ! -f "$run_dir/summary.csv" ]]; then
      return 0
    fi
  done
  return 1
}

echo "Submitting resume jobs for partial large-model resampling chunks:"
echo "  VERSION=$VERSION"
echo "  PROJECT_ROOT=$PROJECT_ROOT"
echo "  CODE_DIR=$CODE_DIR"
echo "  CONDA_ENV_PATH=$CONDA_ENV_PATH"
echo "  TOTAL_RUNS=$TOTAL_RUNS"
echo "  DRY_RUN=$DRY_RUN"
echo ""

submitted=0
skipped=0

for target in "${TARGETS[@]}"; do
  if [[ "$target" != *:* ]]; then
    echo "Skipping malformed target: $target" >&2
    continue
  fi

  model="${target%%:*}"
  size="${target#*:}"
  label="${model}-${size}"
  safe_label="${label//[^A-Za-z0-9]/-}"
  results_root="${RESULTS_ROOT:-$CODE_DIR/results/resampling/$label}"
  model_root="$results_root/$VERSION/$label"
  batch_size="${BATCH_SIZE:-${BATCH_SIZE_MAP[$target]:-1}}"
  gpus="${GPUS:-${GPU_MAP[$target]:-1}}"
  mem="${MEM:-${MEM_MAP[$target]:-96G}}"
  time_limit="${TIME:-${TIME_MAP[$target]:-02:00:00}}"

  echo "Target $label:"
  echo "  model_root=$model_root"

  for ((chunk_index = 1; chunk_index <= TOTAL_RUNS; chunk_index++)); do
    chunk="$(printf "%02d" "$chunk_index")"
    run_label="chunk_${chunk}"
    chunk_dir="$model_root/$run_label"

    if ! chunk_has_partial_runs "$chunk_dir"; then
      echo "  ${run_label}: no partial runs to resume, skipping."
      skipped=$((skipped + 1))
      continue
    fi

    export_parts=(
      PROJECT_ROOT="$PROJECT_ROOT"
      CODE_DIR="$CODE_DIR"
      CONDA_ENV_PATH="$CONDA_ENV_PATH"
      RUN_ROOT="$chunk_dir"
      BATCH_SIZE="$batch_size"
      MODEL_SIZE="$size"
      HF_TOKEN
      HUGGINGFACEHUB_API_TOKEN
      HF_HOME
      HF_HUB_CACHE
      TRANSFORMERS_CACHE
    )
    if [[ -n "${MAX_NEW_TOKENS:-}" ]]; then
      export_parts+=(MAX_NEW_TOKENS="$MAX_NEW_TOKENS")
    fi
    if [[ -n "${USE_JSON_FORMAT:-}" ]]; then
      export_parts+=(USE_JSON_FORMAT="$USE_JSON_FORMAT")
    fi
    if [[ -n "${TASKS_CSV:-}" ]]; then
      tasks_export="${TASKS_CSV//,/;}"
      export_parts+=(TASKS_CSV="$tasks_export")
    fi
    if [[ -n "${EXTRA_ARGS:-}" ]]; then
      export_parts+=(EXTRA_ARGS="$EXTRA_ARGS")
    fi

    export_arg="$(IFS=,; echo "${export_parts[*]}")"
    cmd=(
      sbatch
      --job-name="resume-${safe_label}-${run_label}"
      --gres="gpu:${gpus}"
      --mem="$mem"
      --time="$time_limit"
      --export="$export_arg"
      "$SBATCH_SCRIPT"
    )

    echo "  ${run_label}: submitting resume job."
    if [[ "$DRY_RUN" == "1" ]]; then
      printf '    %q' "${cmd[@]}"
      printf '\n'
    else
      "${cmd[@]}"
    fi
    submitted=$((submitted + 1))
  done
  echo ""
done

echo "Done. submitted=$submitted skipped=$skipped"
