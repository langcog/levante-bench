#!/usr/bin/env bash
# Run Phase 9 (describe-first) TROG experiments on Qwen3.5-2B.
# Tests 3 configurations: Phase 9 alone, 1+2+3+9, and 1+2+3+4+9 (best-so-far + describe-first)
#
# Estimated runtime: ~10-12 min per run (99 trials × ~6s/trial with 400 token output)
# Total: ~35-40 min

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PY="$PROJECT_ROOT/.venv/bin/python"
DEVICE="${DEVICE:-cuda}"
EXP="$SCRIPT_DIR/experiment_qwen2b.py"
RESULTS="$PROJECT_ROOT/results/prompt_optimization/trog/qwen-3.5-2b"
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
    "$PY" "$EXP" --phase "${phases[@]}" --device "$DEVICE" 2>&1 | tee "$log"
    echo ""
}

# Phase 9 alone (describe-first, no structural prompt)
run_phase "9" 9

# Phase 1+2+3+9 (structured + enhanced-parse + format + describe-first)
run_phase "1_2_3_9" 1 2 3 9

# Phase 1+2+3+4+9 (current best 62.6% + describe-first)
run_phase "1_2_3_4_9" 1 2 3 4 9

echo "All Phase 9 runs complete."
echo "Run summarize_trog_2b_phases.py to see updated results."
