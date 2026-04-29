#!/bin/bash
# Submit the updated LEVANTE-bench paper model list on Marlowe.
#
# Default behavior:
# - submits deterministic single-run baselines for models in the paper list
# - skips model-size labels that already have a baseline summary in GCS or locally
# - writes local results to results/v1/<model-size>/ and mirrors files into
#   results/v1/<model-size>/baseline/ for dashboard ingestion
#
# Usage:
#   bash scripts/slurm/submit_paper_model_list.sh
#   DRY_RUN=1 bash scripts/slurm/submit_paper_model_list.sh
#   FORCE=1 bash scripts/slurm/submit_paper_model_list.sh
#   ONLY="internvl35:14B qwen35:27B" bash scripts/slurm/submit_paper_model_list.sh

set -euo pipefail

if ! command -v sbatch >/dev/null 2>&1; then
  echo "ERROR: sbatch not found in PATH." >&2
  echo "Run this script on a Marlowe login node." >&2
  exit 127
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_SCRIPT="$SCRIPT_DIR/run_paper_model_baseline.sbatch"

if [[ ! -f "$SBATCH_SCRIPT" ]]; then
  echo "ERROR: missing sbatch template: $SBATCH_SCRIPT" >&2
  exit 1
fi

PROJECT_ROOT="${PROJECT_ROOT:-/projects/m000102}"
CODE_DIR="${PROJECT_ROOT}/code/levante-bench"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-$PROJECT_ROOT/envs/levante-bench-py311}"
VERSION="${VERSION:-v1}"
DEVICE="${DEVICE:-cuda}"
GCS_RESULTS_ROOT="${GCS_RESULTS_ROOT:-gs://levante-bench/results}"
FORCE="${FORCE:-0}"
DRY_RUN="${DRY_RUN:-0}"

DEFAULT_TARGETS=(
  "gemma4:E2B-it"
  "gemma4:E4B-it"
  "gemma4:26B-A4B-it"
  "gemma4:31B-it"
  "internvl35:1B"
  "internvl35:2B"
  "internvl35:4B"
  "internvl35:8B"
  "internvl35:14B"
  "internvl35:38B"
  "molmo2:4B"
  "molmo2:O-7B"
  "molmo2:8B"
  "qwen35:0.8B"
  "qwen35:2B"
  "qwen35:4B"
  "qwen35:9B"
  "qwen35:27B"
  "smolvlm2:256M"
  "smolvlm2:500M"
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
  ["gemma4:E2B-it"]="2"
  ["gemma4:E4B-it"]="2"
  ["gemma4:26B-A4B-it"]="1"
  ["gemma4:31B-it"]="1"
  ["internvl35:1B"]="4"
  ["internvl35:2B"]="2"
  ["internvl35:4B"]="2"
  ["internvl35:8B"]="1"
  ["internvl35:14B"]="1"
  ["internvl35:38B"]="1"
  ["molmo2:4B"]="2"
  ["molmo2:O-7B"]="1"
  ["molmo2:8B"]="1"
  ["qwen35:0.8B"]="4"
  ["qwen35:2B"]="2"
  ["qwen35:4B"]="2"
  ["qwen35:9B"]="1"
  ["qwen35:27B"]="1"
  ["smolvlm2:256M"]="4"
  ["smolvlm2:500M"]="2"
  ["smolvlm2:2.2B"]="1"
  ["tinyllava:2.4B"]="2"
  ["tinyllava:3.1B"]="2"
)

declare -A GPU_MAP=(
  ["internvl35:38B"]="1"
  ["qwen35:27B"]="1"
  ["gemma4:26B-A4B-it"]="1"
  ["gemma4:31B-it"]="1"
)

declare -A MEM_MAP=(
  ["internvl35:38B"]="160G"
  ["qwen35:27B"]="144G"
  ["gemma4:26B-A4B-it"]="144G"
  ["gemma4:31B-it"]="160G"
)

declare -A TIME_MAP=(
  ["internvl35:38B"]="12:00:00"
  ["qwen35:27B"]="12:00:00"
  ["gemma4:26B-A4B-it"]="12:00:00"
  ["gemma4:31B-it"]="12:00:00"
)

has_existing_summary() {
  local label="$1"

  if [[ -f "$CODE_DIR/results/$VERSION/$label/baseline/summary.csv" ]]; then
    return 0
  fi
  if [[ -f "$CODE_DIR/results/$VERSION/$label/summary.csv" ]]; then
    return 0
  fi

  if command -v gcloud >/dev/null 2>&1; then
    if gcloud storage ls "$GCS_RESULTS_ROOT/$VERSION/$label/baseline/summary.csv" >/dev/null 2>&1; then
      return 0
    fi
    if gcloud storage ls "$GCS_RESULTS_ROOT/$VERSION/$label/summary.csv" >/dev/null 2>&1; then
      return 0
    fi
  fi

  return 1
}

echo "Submitting updated paper model list:"
echo "  VERSION=$VERSION"
echo "  PROJECT_ROOT=$PROJECT_ROOT"
echo "  CODE_DIR=$CODE_DIR"
echo "  CONDA_ENV_PATH=$CONDA_ENV_PATH"
echo "  GCS_RESULTS_ROOT=$GCS_RESULTS_ROOT"
echo "  FORCE=$FORCE"
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

  if [[ "$FORCE" != "1" ]] && has_existing_summary "$label"; then
    echo "Skipping existing baseline: $label"
    skipped=$((skipped + 1))
    continue
  fi

  batch_size="${BATCH_SIZE_MAP[$target]:-1}"
  gpus="${GPU_MAP[$target]:-1}"
  mem="${MEM_MAP[$target]:-96G}"
  time_limit="${TIME_MAP[$target]:-06:00:00}"

  cmd=(
    sbatch
    --job-name="levante-${safe_label}"
    --gres="gpu:${gpus}"
    --mem="$mem"
    --time="$time_limit"
    --export=ALL,PROJECT_ROOT="$PROJECT_ROOT",CONDA_ENV_PATH="$CONDA_ENV_PATH",MODEL_NAME="$model",MODEL_SIZE="$size",VERSION="$VERSION",DEVICE="$DEVICE",BATCH_SIZE="$batch_size",BASELINE_ALIAS=1,USE_JOB_OUTPUT_ROOT=0
    "$SBATCH_SCRIPT"
  )

  echo "Submitting $label (batch_size=$batch_size gpu=$gpus mem=$mem time=$time_limit)"
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '  %q' "${cmd[@]}"
    printf '\n'
  else
    "${cmd[@]}"
  fi
  submitted=$((submitted + 1))
done

echo ""
echo "Done. submitted=$submitted skipped=$skipped"
