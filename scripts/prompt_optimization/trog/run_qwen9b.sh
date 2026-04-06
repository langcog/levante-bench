#!/usr/bin/env bash
# TROG experiments on Qwen3.5-9B — baseline, best structural, and describe-first.
#
# Goal: test whether describe-first gains scale with model size.
# At 2B: +2 pp from describe-first (64.6% vs 62.6%)
# Hypothesis: 9B shows +5-10 pp from describe-first due to higher description quality.
#
# Estimated runtime: ~18-22 min/run (99 trials × ~12s baseline, ~20s describe-first)
# Total: ~55-65 min

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PY="$PROJECT_ROOT/.venv/bin/python"
DEVICE="${DEVICE:-cuda}"
EXP="$SCRIPT_DIR/experiment_qwen9b.py"
RESULTS="$PROJECT_ROOT/results/trog-9b-phases"
LOG_DIR="$RESULTS"

mkdir -p "$LOG_DIR"

run_phase() {
    local label="$1"
    shift
    local phases=("$@")
    local log="$LOG_DIR/run_phase_${label}.log"
    echo "════════════════════════════════════════════════════════"
    echo "  Launching: phases=${phases[*]}  →  $log"
    echo "════════════════════════════════════════════════════════"
    HF_HUB_OFFLINE=1 "$PY" "$EXP" --phase "${phases[@]}" --device "$DEVICE" 2>&1 | tee "$log"
    echo ""
}

# Baseline — lets us measure 9B starting point (~32 min)
run_phase "baseline" 0

# Describe-first — the main hypothesis: does 9B benefit more than 2B's +2 pp? (~42 min)
run_phase "1_2_3_9" 1 2 3 9

echo "All 9B runs complete."
