#!/bin/bash
# Run one deterministic single-run evaluation per hosted model locally.
#
# Behavior:
# - Runs all 6 tasks
# - num_runs=1
# - true_random_option_order=false
# - Writes under results/<model_name>/...
#
# Usage:
#   bash scripts/submit_all_hosted_models_once_local.sh
#   VERSION=v1_new_parser DEVICE=auto bash scripts/submit_all_hosted_models_once_local.sh
#   bash scripts/submit_all_hosted_models_once_local.sh gpt53 qwen3vl_30b_hf

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RUN_EXPERIMENT="$REPO_ROOT/run_experiment.sh"

if [[ ! -x "$RUN_EXPERIMENT" ]]; then
  echo "ERROR: run_experiment.sh not found or not executable: $RUN_EXPERIMENT" >&2
  exit 1
fi

VERSION="${VERSION:-v1}"
DEVICE="${DEVICE:-auto}"
BATCH_SIZE="${BATCH_SIZE:-1}"
TASKS=(
  egma-math
  matrix-reasoning
  mental-rotation
  theory-of-mind
  trog
  vocab
)

DEFAULT_MODELS=(
  qwen25vl_32b_hf
  qwen25vl_72b_hf
  qwen3vl_30b_hf
  qwen3vl_235b_hf
  aya_vision_32b_hf
  gemini_pro
  gpt52
  gpt53
)

if [[ $# -gt 0 ]]; then
  MODELS=("$@")
else
  MODELS=("${DEFAULT_MODELS[@]}")
fi

echo "Running deterministic single-run hosted-model evals locally:"
echo "  VERSION=$VERSION"
echo "  DEVICE=$DEVICE"
echo "  BATCH_SIZE=$BATCH_SIZE"
echo "  REPO_ROOT=$REPO_ROOT"
echo ""

cd "$REPO_ROOT"

ok_models=()
skipped_models=()
failed_models=()

for model in "${MODELS[@]}"; do
  echo "=== Running hosted model: $model ==="
  TASK_ARGS=()
  for task in "${TASKS[@]}"; do
    TASK_ARGS+=(--task "$task")
  done
  tmp_log="$(mktemp "/tmp/levante-hosted-${model}-XXXX.log")"
  if PYTHONPATH=src python -m levante_bench.cli run-eval \
      "${TASK_ARGS[@]}" \
      --model "$model" \
      --version "$VERSION" \
      --device "$DEVICE" \
      --batch-size "$BATCH_SIZE" \
      --num-runs 1 \
      --output-dir "results/${model}" \
      >"$tmp_log" 2>&1; then
    cat "$tmp_log"
    ok_models+=("$model")
  else
    cat "$tmp_log"
    if grep -q "model_not_supported" "$tmp_log"; then
      echo "Skipping unsupported hosted model/provider combination: $model" >&2
      skipped_models+=("$model")
    else
      echo "Model run failed (non-support error): $model" >&2
      failed_models+=("$model")
    fi
  fi
  rm -f "$tmp_log"
done

echo ""
echo "Hosted run summary:"
echo "  succeeded (${#ok_models[@]}): ${ok_models[*]:-none}"
echo "  skipped   (${#skipped_models[@]}): ${skipped_models[*]:-none}"
echo "  failed    (${#failed_models[@]}): ${failed_models[*]:-none}"

# Do not fail the whole batch for provider support misses.
if [[ ${#failed_models[@]} -gt 0 ]]; then
  exit 1
fi

echo "Done."
