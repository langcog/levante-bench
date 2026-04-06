#!/usr/bin/env bash
# Run all Mental Rotation InternVL3.5-2B phase experiments sequentially.
# Usage: bash scripts/run_all_mrot_internvl_phases.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PY="${PY:-${ROOT}/.venv/bin/python}"
LOGDIR="${ROOT}/results/prompt_optimization/mental-rotation/internvl-3.5-2b"
mkdir -p "$LOGDIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SUITE_LOG="${LOGDIR}/suite_${TIMESTAMP}.log"

run_one() {
    local tag="$1"
    shift
    local logfile="${LOGDIR}/run_${tag}.log"
    echo "──── ${tag} ────" | tee -a "$SUITE_LOG"
    "$PY" "${ROOT}/scripts/prompt_optimization/mental-rotation/experiment_internvl2b.py" "$@" 2>&1 | tee "$logfile" | tee -a "$SUITE_LOG"
    echo "" | tee -a "$SUITE_LOG"
}

echo "Mental Rotation InternVL3.5-2B Phase Suite — $(date)" | tee "$SUITE_LOG"
echo "Model: OpenGVLab/InternVL3_5-2B-HF (images resized to 512px)" | tee -a "$SUITE_LOG"
echo "Log: $SUITE_LOG" | tee -a "$SUITE_LOG"
echo "" | tee -a "$SUITE_LOG"

run_one "phase_baseline"    --phase 0
run_one "phase_1"           --phase 1
run_one "phase_2"           --phase 2
run_one "phase_3"           --phase 3
run_one "phase_4"           --phase 4
run_one "phase_5"           --phase 5
run_one "phase_1_2_3"       --phase 1 2 3
run_one "phase_1_4"         --phase 1 4
run_one "phase_1_2_3_4_5"   --phase 1 2 3 4 5

echo "──── SUMMARY ────" | tee -a "$SUITE_LOG"
"$PY" "${ROOT}/scripts/prompt_optimization/mental-rotation/summarize_internvl2b.py" --dir "$LOGDIR" 2>&1 | tee -a "$SUITE_LOG"

echo ""
echo "Suite complete. Full log: $SUITE_LOG"
