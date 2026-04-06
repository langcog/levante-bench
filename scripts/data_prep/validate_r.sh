#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run R/Redivis validation checks.

Usage:
  scripts/validate_r.sh [options]

Options:
  --check-packages-only        Verify R and required packages (default behavior).
  --run-comparison-smoke       Also run a comparison smoke test after package checks.
  --task <task_id>             Task id for smoke comparison (default: trog).
  --model <model_id>           Model id for smoke comparison (default: clip_base).
  --version <version>          Data/results version for smoke comparison (default: current).
  --results-dir <dir>          Results dir root used by compare script (default: results).
  --output-dir <dir>           Output dir for comparison CSVs (default: results/comparison).
  -h, --help                   Show this help.

Notes:
  - Smoke comparison requires existing prerequisites:
      1) data/responses/<version>/responses_by_ability/<task>_proportions_by_ability.csv
      2) results/<version>/<model>/<task>.npy
EOF
}

RUN_COMPARISON_SMOKE=0
TASK_ID="trog"
MODEL_ID="clip_base"
VERSION="current"
RESULTS_DIR="results"
OUTPUT_DIR="results/comparison"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --check-packages-only)
      RUN_COMPARISON_SMOKE=0
      shift
      ;;
    --run-comparison-smoke)
      RUN_COMPARISON_SMOKE=1
      shift
      ;;
    --task)
      TASK_ID="${2:-}"
      shift 2
      ;;
    --model)
      MODEL_ID="${2:-}"
      shift 2
      ;;
    --version)
      VERSION="${2:-}"
      shift 2
      ;;
    --results-dir)
      RESULTS_DIR="${2:-}"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 2
      ;;
  esac
done

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "==> Checking Rscript"
if ! command -v Rscript >/dev/null 2>&1; then
  echo "Rscript not found. Install R before running R validations." >&2
  exit 1
fi

echo "==> Checking required R packages"
Rscript -e "pkgs <- c('tidyverse','philentropy','nloptr','reticulate','jsonlite','redivis'); missing <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly=TRUE)]; if (length(missing) > 0) { stop(paste('Missing R packages:', paste(missing, collapse=', '))) }"

if [[ "$RUN_COMPARISON_SMOKE" -eq 0 ]]; then
  echo "R package checks passed."
  exit 0
fi

safe_task="$(printf '%s' "$TASK_ID" | sed -E 's/[^a-zA-Z0-9_-]/_/g')"
safe_model="$(printf '%s' "$MODEL_ID" | sed -E 's/[^a-zA-Z0-9_-]/_/g')"

resp_file="data/responses/${VERSION}/responses_by_ability/${safe_task}_proportions_by_ability.csv"
model_npy="${RESULTS_DIR}/${VERSION}/${safe_model}/${safe_task}.npy"

echo "==> Checking smoke prerequisites"
if [[ ! -f "$resp_file" ]]; then
  echo "Missing required responses-by-ability file: $resp_file" >&2
  echo "Run: Rscript scripts/download_levante_data.R --version ${VERSION}" >&2
  exit 1
fi
if [[ ! -f "$model_npy" ]]; then
  echo "Missing required model output file: $model_npy" >&2
  echo "Run: levante-bench run-eval --task ${TASK_ID} --model ${MODEL_ID} --version ${VERSION}" >&2
  exit 1
fi

echo "==> Running comparison smoke test"
levante-bench run-comparison \
  --task "$TASK_ID" \
  --model "$MODEL_ID" \
  --version "$VERSION" \
  --results-dir "$RESULTS_DIR" \
  --output-dir "$OUTPUT_DIR"

dkl_csv="${OUTPUT_DIR}/${safe_task}_${safe_model}_d_kl.csv"
acc_csv="${OUTPUT_DIR}/${safe_task}_${safe_model}_accuracy.csv"
if [[ ! -f "$dkl_csv" || ! -f "$acc_csv" ]]; then
  echo "Expected comparison outputs not found: $dkl_csv and/or $acc_csv" >&2
  exit 1
fi

echo "R smoke comparison passed."
