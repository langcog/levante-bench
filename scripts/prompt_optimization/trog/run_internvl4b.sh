#!/usr/bin/env bash
# TROG experiments on InternVL3.5-4B-HF — tests describe-first hypothesis.
#
# Why InternVL: doesn't reason spontaneously (clean baseline), stronger vision
# encoder than Qwen3.5-2B → better visual descriptions → larger describe-first gain.
#
# Estimated runtime: ~5-6s/trial × 99 trials = ~9-10 min/run
# Total: ~30 min for 3 configs

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PY="$PROJECT_ROOT/.venv/bin/python"
EXP="$SCRIPT_DIR/experiment_internvl4b.py"
RESULTS="$PROJECT_ROOT/results/trog-internvl-phases"

mkdir -p "$RESULTS"

run_phase() {
    local label="$1"; shift
    local phases=("$@")
    local log="$RESULTS/run_phase_${label}.log"
    echo "════════════════════════════════════════════════════════"
    echo "  Launching: phases=${phases[*]}  →  $log"
    echo "════════════════════════════════════════════════════════"
    HF_HUB_OFFLINE=1 "$PY" "$EXP" --phase "${phases[@]}" --device mps 2>&1 | tee "$log"
    echo ""
}

# Baseline
run_phase "baseline" 0

# Best structural from Qwen3.5-2B (62.6%) — how does InternVL compare?
run_phase "1_2_3_4" 1 2 3 4

# Describe-first — the main hypothesis
run_phase "1_2_3_9" 1 2 3 9

echo "All InternVL TROG runs complete."
