#!/bin/bash
# Submit 40 true-random runs for larger local models, one run per Slurm job.
#
# Output layout:
#   results/resampling/<model-size>/<version>/<model-size>/chunk_XX/0001/...
#
# Usage:
#   DRY_RUN=1 ONLY="gemma4:31B-it qwen35:27B" bash scripts/slurm/submit_large_model_resampling_40.sh
#   ONLY="gemma4:31B-it" bash scripts/slurm/submit_large_model_resampling_40.sh
#   MAX_SUBMISSIONS=10 ONLY="gemma4:31B-it" bash scripts/slurm/submit_large_model_resampling_40.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_SCRIPT="$SCRIPT_DIR/run_local_model_experiment.sbatch"

if [[ ! -f "$SBATCH_SCRIPT" ]]; then
  echo "ERROR: missing sbatch template: $SBATCH_SCRIPT" >&2
  exit 1
fi

PROJECT_ROOT="${PROJECT_ROOT:-/projects/m000102}"
CODE_DIR="${CODE_DIR:-$PROJECT_ROOT/code/levante-bench}"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-$PROJECT_ROOT/envs/levante-bench-py311}"
VERSION="${VERSION:-v1}"
DEVICE="${DEVICE:-cuda}"
TOTAL_RUNS="${TOTAL_RUNS:-40}"
MAX_SUBMISSIONS="${MAX_SUBMISSIONS:-10}"
DRY_RUN="${DRY_RUN:-0}"
FORCE="${FORCE:-0}"

if [[ ! "$MAX_SUBMISSIONS" =~ ^[0-9]+$ ]]; then
  echo "ERROR: MAX_SUBMISSIONS must be a non-negative integer (0 means unlimited)." >&2
  exit 2
fi

if [[ "$DRY_RUN" != "1" ]] && ! command -v sbatch >/dev/null 2>&1; then
  echo "ERROR: sbatch not found in PATH." >&2
  echo "Run this script on a Marlowe login/head node, or set DRY_RUN=1." >&2
  exit 127
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
  ["gemma4:26B-A4B-it"]="06:00:00"
  ["gemma4:31B-it"]="06:00:00"
  ["internvl35:14B"]="04:00:00"
  ["internvl35:38B"]="06:00:00"
  ["molmo2:4B"]="03:00:00"
  ["molmo2:O-7B"]="03:00:00"
  ["molmo2:8B"]="03:00:00"
  ["qwen35:9B"]="04:00:00"
  ["qwen35:27B"]="06:00:00"
  ["smolvlm2:2.2B"]="02:00:00"
  ["tinyllava:2.4B"]="02:00:00"
  ["tinyllava:3.1B"]="02:00:00"
)

chunk_is_complete() {
  local chunk_dir="$1"
  [[ -d "$chunk_dir" ]] || return 1
  compgen -G "$chunk_dir/[0-9][0-9][0-9][0-9]/summary.csv" >/dev/null
}

chunk_is_partial() {
  local chunk_dir="$1"
  [[ -d "$chunk_dir" ]] || return 1
  compgen -G "$chunk_dir/[0-9][0-9][0-9][0-9]/metadata.json" >/dev/null
}

echo "Submitting large-model resampling chunks:"
echo "  VERSION=$VERSION"
echo "  PROJECT_ROOT=$PROJECT_ROOT"
echo "  CODE_DIR=$CODE_DIR"
echo "  CONDA_ENV_PATH=$CONDA_ENV_PATH"
echo "  TOTAL_RUNS=$TOTAL_RUNS"
echo "  RUNS_PER_JOB=1"
echo "  MAX_SUBMISSIONS=$MAX_SUBMISSIONS"
echo "  FORCE=$FORCE"
echo "  DRY_RUN=$DRY_RUN"
echo ""

submitted=0
skipped_complete=0
skipped_partial=0
cap_reached=0

for target in "${TARGETS[@]}"; do
  if [[ "$MAX_SUBMISSIONS" != "0" && "$submitted" -ge "$MAX_SUBMISSIONS" ]]; then
    cap_reached=1
    break
  fi

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
  stitched_label="$(printf "r%04d" "$TOTAL_RUNS")"
  batch_size="${BATCH_SIZE:-${BATCH_SIZE_MAP[$target]:-1}}"
  gpus="${GPUS:-${GPU_MAP[$target]:-1}}"
  mem="${MEM:-${MEM_MAP[$target]:-96G}}"
  time_limit="${TIME:-${TIME_MAP[$target]:-04:00:00}}"
  max_new_tokens="${MAX_NEW_TOKENS:-}"
  use_json_format="${USE_JSON_FORMAT:-}"

  echo "Target $label:"
  echo "  results_root=$results_root"
  echo "  batch_size=$batch_size gpu=$gpus mem=$mem time=$time_limit"

  for ((chunk_index = 1; chunk_index <= TOTAL_RUNS; chunk_index++)); do
    if [[ "$MAX_SUBMISSIONS" != "0" && "$submitted" -ge "$MAX_SUBMISSIONS" ]]; then
      echo "  submission cap reached; stop here and rerun later to continue."
      cap_reached=1
      break
    fi

    chunk="$(printf "%02d" "$chunk_index")"
    run_label="chunk_${chunk}"
    chunk_dir="$model_root/$run_label"

    if [[ "$FORCE" != "1" ]] && chunk_is_complete "$chunk_dir"; then
      echo "  ${run_label}: complete, skipping."
      skipped_complete=$((skipped_complete + 1))
      continue
    fi
    if [[ "$FORCE" != "1" ]] && chunk_is_partial "$chunk_dir"; then
      echo "  ${run_label}: partial, skipping. Use submit_resume_large_model_resampling_40.sh."
      skipped_partial=$((skipped_partial + 1))
      continue
    fi

    export_parts=(
      PROJECT_ROOT="$PROJECT_ROOT"
      CODE_DIR="$CODE_DIR"
      CONDA_ENV_PATH="$CONDA_ENV_PATH"
      MODEL_NAME="$model"
      MODEL_SIZE="$size"
      VERSION="$VERSION"
      DEVICE="$DEVICE"
      BATCH_SIZE="$batch_size"
      NUM_RUNS="1"
      TRUE_RANDOM_OPTION_ORDER="true"
      RUN_LABEL="$run_label"
      SLURM_RUN_LABEL="false"
      USE_JOB_OUTPUT_ROOT="0"
      RESULTS_ROOT="$results_root"
      EXTRA_OVERRIDES="output_dir=$results_root"
      HF_TOKEN
      HUGGINGFACEHUB_API_TOKEN
      HF_HOME
      HF_HUB_CACHE
      TRANSFORMERS_CACHE
    )
    if [[ -n "$max_new_tokens" ]]; then
      export_parts+=(MAX_NEW_TOKENS="$max_new_tokens")
    fi
    if [[ -n "$use_json_format" ]]; then
      export_parts+=(USE_JSON_FORMAT="$use_json_format")
    fi
    if [[ -n "${TASKS_CSV:-}" ]]; then
      tasks_export="${TASKS_CSV//,/;}"
      export_parts+=(TASKS_CSV="$tasks_export")
    fi

    export_arg="$(IFS=,; echo "${export_parts[*]}")"
    cmd=(
      sbatch
      --job-name="${safe_label}-${run_label}"
      --gres="gpu:${gpus}"
      --mem="$mem"
      --time="$time_limit"
      --export="$export_arg"
      "$SBATCH_SCRIPT"
    )

    echo "  ${run_label}: submitting one run."
    if [[ "$DRY_RUN" == "1" ]]; then
      printf '    %q' "${cmd[@]}"
      printf '\n'
    else
      "${cmd[@]}"
    fi
    submitted=$((submitted + 1))
  done

  echo "  Stitch completed chunks with:"
  echo "    python scripts/analysis/stitch_resampling_runs.py \\"
  echo "      --source-root \"$model_root\" \\"
  echo "      --output-root \"$results_root/$VERSION/${label}_${stitched_label}\""
  echo ""
done

if [[ "$cap_reached" == "1" ]]; then
  echo "Submission cap reached at MAX_SUBMISSIONS=$MAX_SUBMISSIONS."
  echo "Rerun this launcher after the current batch finishes; completed chunks will be skipped."
fi
echo "Done. submitted=$submitted skipped_complete=$skipped_complete skipped_partial=$skipped_partial"
