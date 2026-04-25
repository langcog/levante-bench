#!/bin/bash
# Submit one Slurm job per local model for vocab + synthetic-vocab,
# then build a consolidated comparison table.
#
# Usage:
#   bash scripts/slurm/submit_vocab_synthetic_all_models.sh
#   WAIT_FOR_COMPLETION=0 bash scripts/slurm/submit_vocab_synthetic_all_models.sh
#   MODELS="smolvlm2 internvl35 qwen35" bash scripts/slurm/submit_vocab_synthetic_all_models.sh

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
CODE_DIR="${PROJECT_ROOT}/code/levante-bench"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-$PROJECT_ROOT/envs/levante-bench-py311}"
VERSION="${VERSION:-v1}"
DEVICE="${DEVICE:-cuda}"
TASKS_CSV="${TASKS_CSV:-vocab,synthetic-vocab}"
WAIT_FOR_COMPLETION="${WAIT_FOR_COMPLETION:-1}"
POLL_SECONDS="${POLL_SECONDS:-20}"

RESULTS_ROOT="${RESULTS_ROOT:-$PROJECT_ROOT/outputs/results/vocab_synthetic_all_models}"
OUTPUT_CSV="${OUTPUT_CSV:-$RESULTS_ROOT/vocab_synthetic_table.csv}"
OUTPUT_MD="${OUTPUT_MD:-$RESULTS_ROOT/vocab_synthetic_table.md}"

declare -A MODEL_SIZE_MAP=(
  ["smolvlm2"]="2.2B"
  ["internvl35"]="8B"
  ["qwen35"]="4B"
  ["qwen3vl_30b"]=""
  ["qwen25vl_32b"]=""
  ["tinyllava"]=""
  ["aquila_vl"]=""
  ["gemma3"]="4b-it"
  ["gemma4"]="E4B-it"
)

declare -A BATCH_SIZE_MAP=(
  ["smolvlm2"]="1"
  ["internvl35"]="1"
  ["qwen35"]="1"
  ["qwen3vl_30b"]="1"
  ["qwen25vl_32b"]="1"
  ["tinyllava"]="1"
  ["aquila_vl"]="1"
  ["gemma3"]="1"
  ["gemma4"]="1"
)

DEFAULT_MODELS=(
  smolvlm2
  internvl35
  qwen35
  qwen3vl_30b
  qwen25vl_32b
  tinyllava
  aquila_vl
  gemma3
  gemma4
)

if [[ -n "${MODELS:-}" ]]; then
  # shellcheck disable=SC2206
  MODELS_ARR=( $MODELS )
else
  MODELS_ARR=("${DEFAULT_MODELS[@]}")
fi

mkdir -p "$RESULTS_ROOT"

echo "Submitting vocab + synthetic-vocab jobs:"
echo "  VERSION=$VERSION"
echo "  DEVICE=$DEVICE"
echo "  TASKS_CSV=$TASKS_CSV"
echo "  RESULTS_ROOT=$RESULTS_ROOT"
echo "  WAIT_FOR_COMPLETION=$WAIT_FOR_COMPLETION"
echo ""

JOB_IDS=()
EXPECTED_MODELS=()

for model in "${MODELS_ARR[@]}"; do
  if [[ -z "${BATCH_SIZE_MAP[$model]+x}" ]]; then
    echo "Skipping unknown model key: $model" >&2
    continue
  fi

  size="${MODEL_SIZE_MAP[$model]}"
  batch_size="${BATCH_SIZE_MAP[$model]}"
  EXPECTED_MODELS+=("${model}${size:+-$size}")

  submit_output="$(
    sbatch \
      --export=ALL,PROJECT_ROOT="$PROJECT_ROOT",CONDA_ENV_PATH="$CONDA_ENV_PATH",MODEL_NAME="$model",MODEL_SIZE="$size",VERSION="$VERSION",DEVICE="$DEVICE",TASKS_CSV="$TASKS_CSV",BATCH_SIZE="$batch_size",NUM_RUNS=1,TRUE_RANDOM_OPTION_ORDER=false,USE_JOB_OUTPUT_ROOT=1,RESULTS_ROOT="$RESULTS_ROOT" \
      "$SBATCH_SCRIPT"
  )"
  echo "$submit_output"
  job_id="$(echo "$submit_output" | awk '{print $NF}')"
  if [[ "$job_id" =~ ^[0-9]+$ ]]; then
    JOB_IDS+=("$job_id")
  fi
done

if [[ ${#JOB_IDS[@]} -eq 0 ]]; then
  echo "No jobs were submitted; skipping table build." >&2
  exit 1
fi

if [[ "$WAIT_FOR_COMPLETION" == "1" ]]; then
  ids_csv="$(IFS=,; echo "${JOB_IDS[*]}")"
  echo ""
  echo "Waiting for jobs to finish: $ids_csv"
  while true; do
    remaining="$(squeue -h -j "$ids_csv" | wc -l | tr -d ' ')"
    if [[ "$remaining" == "0" ]]; then
      break
    fi
    echo "  remaining_jobs=$remaining (poll in ${POLL_SECONDS}s)"
    sleep "$POLL_SECONDS"
  done
  echo "All jobs finished."
fi

expected_csv="$(IFS=,; echo "${EXPECTED_MODELS[*]}")"
echo ""
echo "Building comparison table..."
python "$CODE_DIR/scripts/analysis/build_vocab_synthetic_table.py" \
  --results-root "$RESULTS_ROOT" \
  --version "$VERSION" \
  --expected-models "$expected_csv" \
  --output-csv "$OUTPUT_CSV" \
  --output-md "$OUTPUT_MD"

echo ""
echo "Done."
echo "CSV: $OUTPUT_CSV"
echo "MD:  $OUTPUT_MD"
