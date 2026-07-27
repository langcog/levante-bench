#!/usr/bin/env bash
# Run feasible InternVL3.5 sizes on a single RTX 3090:
#   sizes: 1B, 2B, 4B
#   languages: de, es
#   runs per combo: 10 (true-random option ordering)
#
# Outputs land under:
#   results/v1/internvl35-<size>-<lang>/<run_label>/<0001..0010>/
#
# Usage:
#   bash scripts/run_internvl35_feasible_de_es_10x_3090.sh
#   VERSION=v1 BATCH_SIZE=1 bash scripts/run_internvl35_feasible_de_es_10x_3090.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VERSION="${VERSION:-v1}"
RUNS="${RUNS:-10}"
BATCH_SIZE="${BATCH_SIZE:-1}"
DATE_TAG="${DATE_TAG:-$(date +%F)}"
SIZES=(1B 2B 4B)
LANGS=(de es)

echo "Launching feasible InternVL3.5 DE/ES 10x sweep on RTX 3090"
echo "  VERSION=$VERSION"
echo "  RUNS=$RUNS"
echo "  BATCH_SIZE=$BATCH_SIZE"
echo "  DATE_TAG=$DATE_TAG"
echo ""

for size in "${SIZES[@]}"; do
  for lang in "${LANGS[@]}"; do
    run_label="rtx3090-${DATE_TAG}-internvl35-${size}-${lang}-10x"
    echo "=== internvl35 size=$size lang=$lang run_label=$run_label ==="

    # Use experiment-style entrypoint so we can pass model dicts (name+size).
    bash run_experiment.sh configs/experiments/smolvlm2_2b_10x_random.yaml \
      "models=[{name: internvl35, size: ${size}, use_json_format: false}]" \
      "task_overrides={__all__: {prompt_language: ${lang}}}" \
      "version=${VERSION}" \
      "device=cuda" \
      "batch_size=${BATCH_SIZE}" \
      "num_runs=${RUNS}" \
      "true_random_option_order=true" \
      "run_label=${run_label}" \
      "slurm_run_label=false"

    echo ""
  done
done

echo "All feasible InternVL3.5 DE/ES runs complete."
