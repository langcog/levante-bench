#!/usr/bin/env bash
# Run remaining requested multilingual combos as deterministic single-run baselines:
# - qwen35 2B, 4B, 9B (es/de)
# - smolvlm2 2.2B (es/de)
#
# This intentionally excludes qwen35 0.8B es (already running in current sweep).
# Baseline alias files are copied to:
#   results/v1/<model>-<size>-<lang>/baseline/

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VERSION="${VERSION:-v1}"
BATCH_SIZE="${BATCH_SIZE:-1}"

echo "Launching remaining deterministic baseline runs on RTX 3090"
echo "  VERSION=$VERSION"
echo "  BATCH_SIZE=$BATCH_SIZE"
echo ""

run_baseline_combo() {
  local model="$1"
  local size="$2"
  local lang="$3"
  local model_label="${model}-${size}-${lang}"
  local out_dir="results/${VERSION}/${model_label}"
  local baseline_dir="${out_dir}/baseline"

  echo "=== baseline model=$model size=$size lang=$lang ==="
  bash run_experiment.sh configs/experiments/smolvlm2_500m_v1.yaml \
    "models=[{name: ${model}, size: ${size}, use_json_format: false}]" \
    "task_overrides={__all__: {prompt_language: ${lang}}}" \
    "version=${VERSION}" \
    "device=cuda" \
    "batch_size=${BATCH_SIZE}" \
    "num_runs=1" \
    "true_random_option_order=false" \
    "slurm_run_label=false"

  if [[ ! -f "${out_dir}/summary.csv" ]]; then
    echo "WARNING: missing summary for ${model_label}; skipping baseline alias copy" >&2
    echo ""
    return 0
  fi

  mkdir -p "${baseline_dir}"
  shopt -s nullglob
  for output_file in "${out_dir}"/*.csv "${out_dir}"/*.json "${out_dir}"/*.npy; do
    cp -f "${output_file}" "${baseline_dir}/"
  done
  if [[ -f "${out_dir}/cache/responses.json" ]]; then
    mkdir -p "${baseline_dir}/cache"
    cp -f "${out_dir}/cache/responses.json" "${baseline_dir}/cache/responses.json"
  fi
  echo "Created baseline alias: ${baseline_dir}"
  echo ""
}

for size in 2B 4B 9B; do
  for lang in es de; do
    run_baseline_combo qwen35 "$size" "$lang"
  done
done

for lang in es de; do
  run_baseline_combo smolvlm2 2.2B "$lang"
done

echo "Remaining baseline runs complete."
