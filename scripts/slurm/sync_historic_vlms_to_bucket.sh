#!/bin/bash
# Sync completed historic local VLM results to GCS.
#
# Usage:
#   bash scripts/slurm/sync_historic_vlms_to_bucket.sh
#   DRY_RUN=0 bash scripts/slurm/sync_historic_vlms_to_bucket.sh
#   MODELS="llava15_13b cogvlm" DRY_RUN=0 bash scripts/slurm/sync_historic_vlms_to_bucket.sh

set -euo pipefail

if ! command -v gcloud >/dev/null 2>&1; then
  echo "ERROR: gcloud not found in PATH." >&2
  exit 127
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOCAL_ROOT="${LOCAL_ROOT:-$ROOT_DIR/results/v1_additional_models}"
BUCKET_PREFIX="${BUCKET_PREFIX:-gs://levante-bench/results/v1_additional_models}"
DRY_RUN="${DRY_RUN:-1}"
STRICT="${STRICT:-0}"
MODELS_CSV="${MODELS:-llava15_13b,cogvlm,openflamingo9b}"

models_normalized="${MODELS_CSV// /,}"
IFS=',' read -r -a MODELS_LIST <<< "$models_normalized"

if [[ ! -d "$LOCAL_ROOT" ]]; then
  echo "ERROR: LOCAL_ROOT not found: $LOCAL_ROOT" >&2
  exit 1
fi

echo "Historic VLM sync"
echo "  LOCAL_ROOT=$LOCAL_ROOT"
echo "  BUCKET_PREFIX=$BUCKET_PREFIX"
echo "  DRY_RUN=$DRY_RUN"
echo "  STRICT=$STRICT"
echo ""

missing_any=0

for model in "${MODELS_LIST[@]}"; do
  model="$(echo "$model" | xargs)"
  [[ -z "$model" ]] && continue

  local_model_dir="$LOCAL_ROOT/$model"
  remote_model_dir="$BUCKET_PREFIX/$model"

  if [[ ! -d "$local_model_dir" ]]; then
    echo "SKIP $model: local directory missing ($local_model_dir)"
    missing_any=1
    continue
  fi

  # Consider a model "complete enough to publish" when at least one summary.csv
  # exists somewhere under the model directory.
  summary_count="$(rg --files -g 'summary.csv' "$local_model_dir" | wc -l | xargs)"
  if [[ "${summary_count:-0}" == "0" ]]; then
    echo "SKIP $model: no summary.csv found under $local_model_dir"
    missing_any=1
    continue
  fi

  echo "SYNC $model"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "  gcloud storage rsync --recursive --dry-run \"$local_model_dir\" \"$remote_model_dir\""
  else
    gcloud storage rsync --recursive "$local_model_dir" "$remote_model_dir"
  fi
done

if [[ "$STRICT" == "1" && "$missing_any" == "1" ]]; then
  echo ""
  echo "ERROR: one or more models were missing or incomplete (STRICT=1)." >&2
  exit 2
fi

