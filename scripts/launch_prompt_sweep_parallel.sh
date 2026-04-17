#!/usr/bin/env bash
# Launch prompt robustness sweep on 3 GPUs in parallel.
#
# Usage (on the remote node):
#   cd /path/to/levante-bench
#   bash scripts/launch_prompt_sweep_parallel.sh
#
# Prerequisites:
#   pip install omegaconf pyyaml requests torch transformers accelerate
#
# Adjust FREE_GPUS below based on `nvidia-smi` output.

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────
FREE_GPUS=(3 4 6)          # GPUs with 0% utilization
SUBSET=0.3                  # Fraction of trials (0.3 = pilot, 1.0 = full)
OUTPUT_BASE="results/prompt_robustness"
TASKS="mental-rotation matrix-reasoning vocab trog theory-of-mind egma-math"

# Models to evaluate — one per GPU slot, grouped by size.
# Wave 1: Small models (3 in parallel, ~20 min)
WAVE1_MODELS=(
    "qwen35:0.8B"
    "smolvlm2:2.2B"
    "internvl35:1B"
)

# Wave 2: Medium models (3 in parallel, ~30 min)
WAVE2_MODELS=(
    "qwen35:4B"
    "internvl35:4B"
    "gemma3:4b-it"
)

# Wave 3: Large models (run sequentially or 1-2 at a time, ~40 min each)
# Uncomment to include:
# WAVE3_MODELS=(
#     "internvl35:8B"
# )

# ── Helper ────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SWEEP_SCRIPT="$SCRIPT_DIR/prompt_robustness_sweep.py"
PYTHON="${PYTHON:-python}"

mkdir -p "$OUTPUT_BASE/logs"

run_model_on_gpu() {
    local gpu_id=$1
    local model_spec=$2    # format: "model_name:size"
    local model_name="${model_spec%%:*}"
    local model_size="${model_spec##*:}"
    local log_file="$OUTPUT_BASE/logs/${model_name}_${model_size}.log"
    local out_dir="$OUTPUT_BASE/${model_name}_${model_size}"

    echo "[GPU $gpu_id] Starting $model_name (size=$model_size) → $log_file"

    CUDA_VISIBLE_DEVICES=$gpu_id $PYTHON "$SWEEP_SCRIPT" \
        --models "$model_name" \
        --model-sizes "$model_size" \
        --tasks $TASKS \
        --subset-fraction "$SUBSET" \
        --output-dir "$out_dir" \
        --device cuda \
        > "$log_file" 2>&1 &

    echo $!  # return PID
}

wait_for_pids() {
    local pids=("$@")
    local failed=0
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            echo "  [WARN] PID $pid exited with error"
            ((failed++))
        fi
    done
    return $failed
}

run_wave() {
    local wave_name=$1
    shift
    local models=("$@")
    local n_models=${#models[@]}
    local n_gpus=${#FREE_GPUS[@]}
    local pids=()

    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "  $wave_name: ${models[*]}"
    echo "════════════════════════════════════════════════════════════"
    local start_time=$(date +%s)

    for i in "${!models[@]}"; do
        local gpu_idx=$((i % n_gpus))
        local gpu_id=${FREE_GPUS[$gpu_idx]}
        local pid
        pid=$(run_model_on_gpu "$gpu_id" "${models[$i]}")
        pids+=("$pid")
    done

    echo "  Waiting for ${#pids[@]} processes: ${pids[*]}"
    wait_for_pids "${pids[@]}" || true

    local end_time=$(date +%s)
    local elapsed=$((end_time - start_time))
    echo "  $wave_name completed in ${elapsed}s ($(( elapsed / 60 ))m$(( elapsed % 60 ))s)"
}

# ── Main ──────────────────────────────────────────────────────────────
echo "Prompt Robustness Sweep — Parallel Launcher"
echo "GPUs: ${FREE_GPUS[*]}"
echo "Subset: $SUBSET"
echo "Output: $OUTPUT_BASE"
echo "Tasks: $TASKS"
echo ""

TOTAL_START=$(date +%s)

# Wave 1: Small models
run_wave "Wave 1 (small)" "${WAVE1_MODELS[@]}"

# Wave 2: Medium models (uncomment to run)
# run_wave "Wave 2 (medium)" "${WAVE2_MODELS[@]}"

# Wave 3: Large models (uncomment to run)
# if [ -n "${WAVE3_MODELS+x}" ]; then
#     run_wave "Wave 3 (large)" "${WAVE3_MODELS[@]}"
# fi

TOTAL_END=$(date +%s)
TOTAL_ELAPSED=$((TOTAL_END - TOTAL_START))

# ── Merge results ─────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Merging results"
echo "════════════════════════════════════════════════════════════"

$PYTHON -c "
import csv, glob, os
all_rows = []
header = None
for f in sorted(glob.glob('$OUTPUT_BASE/*/sweep_results.csv')):
    with open(f) as fh:
        reader = csv.DictReader(fh)
        if header is None:
            header = reader.fieldnames
        for row in reader:
            all_rows.append(row)
    print(f'  Loaded {f}')

if all_rows and header:
    merged = '$OUTPUT_BASE/sweep_results_merged.csv'
    with open(merged, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=header)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f'  Merged {len(all_rows)} rows → {merged}')
else:
    print('  No results to merge')
"

echo ""
echo "Done! Total time: ${TOTAL_ELAPSED}s ($(( TOTAL_ELAPSED / 60 ))m$(( TOTAL_ELAPSED % 60 ))s)"
echo ""
echo "Next steps:"
echo "  1. Check logs:   ls $OUTPUT_BASE/logs/"
echo "  2. Run analysis: $PYTHON scripts/analyze_prompt_robustness.py \\"
echo "       --results $OUTPUT_BASE/sweep_results_merged.csv"
