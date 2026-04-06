#!/usr/bin/env bash
# Run all TROG phase experiments sequentially.
# Usage: bash scripts/run_all_trog_phases.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PY="${PY:-${ROOT}/.venv/bin/python}"
LOGDIR="${ROOT}/results/prompt_optimization/trog/qwen-0.8b"
mkdir -p "$LOGDIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SUITE_LOG="${LOGDIR}/suite_${TIMESTAMP}.log"

run_one() {
    local tag="$1"
    shift
    local logfile="${LOGDIR}/run_${tag}.log"
    echo "──── ${tag} ────" | tee -a "$SUITE_LOG"
    "$PY" "${ROOT}/scripts/prompt_optimization/trog/experiment_qwen0.8b.py" "$@" 2>&1 | tee "$logfile" | tee -a "$SUITE_LOG"
    echo "" | tee -a "$SUITE_LOG"
}

echo "TROG Phase Experiment Suite — $(date)" | tee "$SUITE_LOG"
echo "Model: Qwen/Qwen3.5-0.8B (images resized to 512px)" | tee -a "$SUITE_LOG"
echo "Log: $SUITE_LOG" | tee -a "$SUITE_LOG"
echo "" | tee -a "$SUITE_LOG"

# Individual phases
run_one "phase_0"       --phase 0
run_one "phase_1"       --phase 1
run_one "phase_2"       --phase 2
run_one "phase_3"       --phase 3
run_one "phase_4"       --phase 4
run_one "phase_5"       --phase 5

# Combined runs
run_one "phase_1_2_3"   --phase 1 2 3
run_one "phase_1_5"     --phase 1 5
run_one "phase_1_2_3_4_5" --phase 1 2 3 4 5

# Summary
echo "──── SUMMARY ────" | tee -a "$SUITE_LOG"
"$PY" "${ROOT}/scripts/prompt_optimization/trog/summarize_qwen0.8b.py" --dir "$LOGDIR" 2>&1 | tee -a "$SUITE_LOG"

echo ""
echo "Suite complete. Full log: $SUITE_LOG"
