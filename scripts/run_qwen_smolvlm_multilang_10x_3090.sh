#!/usr/bin/env bash
# Requested multilingual sweep on a single RTX 3090:
# - qwen35 0.8B (es)
# - qwen35 2B, 4B, 9B (es/de)
# - smolvlm2 2.2B (es/de)
# Each combo runs 10x with true-random option ordering.
#
# Outputs:
#   results/v1/<model>-<size>-<lang>/<run_label>/0001..0010/

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VERSION="${VERSION:-v1}"
RUNS="${RUNS:-10}"
BATCH_SIZE="${BATCH_SIZE:-1}"
DATE_TAG="${DATE_TAG:-$(date +%F)}"

echo "Launching requested qwen/smolvlm multilingual 10x sweep on RTX 3090"
echo "  VERSION=$VERSION"
echo "  RUNS=$RUNS"
echo "  BATCH_SIZE=$BATCH_SIZE"
echo "  DATE_TAG=$DATE_TAG"
echo ""

run_combo() {
  local model="$1"
  local size="$2"
  local lang="$3"
  local run_label="rtx3090-${DATE_TAG}-${model}-${size}-${lang}-10x"

  echo "=== model=$model size=$size lang=$lang run_label=$run_label ==="
  bash run_experiment.sh configs/experiments/smolvlm2_500m_v1.yaml \
    "models=[{name: ${model}, size: ${size}, use_json_format: false}]" \
    "task_overrides={__all__: {prompt_language: ${lang}}}" \
    "version=${VERSION}" \
    "device=cuda" \
    "batch_size=${BATCH_SIZE}" \
    "num_runs=${RUNS}" \
    "true_random_option_order=true" \
    "run_label=${run_label}" \
    "slurm_run_label=false"
  echo ""
}

# qwen 0.8B es
run_combo qwen35 0.8B es

# qwen 2B, 4B, 9B es/de
for size in 2B 4B 9B; do
  for lang in es de; do
    run_combo qwen35 "$size" "$lang"
  done
done

# smolvlm 2.2B es/de
for lang in es de; do
  run_combo smolvlm2 2.2B "$lang"
done

echo "All requested qwen/smolvlm multilingual 10x runs complete."
