#!/usr/bin/env bash
# run_all_trog_2b_phases.sh — Qwen3.5-2B TROG phase suite
# 8 configurations: baseline, existing best phases from 0.8B run,
# new grammar CoT (phase 6) and language expert system prompt (phase 7).
# Expected runtime: ~25-35 min on M5 Pro 48GB.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PY="${ROOT}/.venv/bin/python"
SCRIPT="$(dirname "$0")/experiment_qwen2b.py"
LOGDIR="${ROOT}/results/prompt_optimization/trog/qwen-3.5-2b"
MODEL_ID="Qwen/Qwen3.5-2B"
SUITE_LOG="${LOGDIR}/suite_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$LOGDIR"

run_one() {
    local tag="$1"; shift
    echo "──── ${tag} ────" | tee -a "$SUITE_LOG"
    "$PY" "$SCRIPT" --model-id "$MODEL_ID" "$@" \
        --output-dir "$LOGDIR" 2>&1 | tee "${LOGDIR}/run_${tag}.log" | tee -a "$SUITE_LOG"
    echo "" | tee -a "$SUITE_LOG"
}

echo "TROG — Qwen3.5-2B — $(date)" | tee "$SUITE_LOG"
echo "Suite log: $SUITE_LOG"
echo ""

# ── Baseline ────────────────────────────────────────────────────────────────
run_one "baseline"          --phase 0

# ── Single phases ────────────────────────────────────────────────────────────
run_one "phase_3"           --phase 3
run_one "phase_7"           --phase 7

# ── Replicate best 0.8B config (1+2+3) for direct comparison ─────────────────
run_one "phase_1_2_3"       --phase 1 2 3

# ── Best 0.8B + grounding (1+2+3+4) ──────────────────────────────────────────
run_one "phase_1_2_3_4"     --phase 1 2 3 4

# ── New: language expert system prompt combos ─────────────────────────────────
run_one "phase_1_2_7"       --phase 1 2 7

# ── New: grammar CoT targeting 0% failure types ───────────────────────────────
run_one "phase_1_2_3_6"     --phase 1 2 3 6

# ── Best combo candidate: expert system + grammar CoT ────────────────────────
run_one "phase_1_2_7_6"     --phase 1 2 7 6

# ── Summary ──────────────────────────────────────────────────────────────────
echo "──── SUMMARY ────" | tee -a "$SUITE_LOG"
"$PY" "${ROOT}/scripts/summarize_trog_2b_phases.py" --dir "$LOGDIR" 2>&1 | tee -a "$SUITE_LOG"

echo ""
echo "Suite complete. Log: $SUITE_LOG"
