#!/usr/bin/env bash
# run_all_matrix_8b_phases.sh — InternVL3.5-8B Matrix Reasoning phase suite
# Tests: baseline at 512px and 1024px, structured+expert at both resolutions,
#        rule hint at 1024px, describe-first approach at 1024px.
# Expected runtime: ~35-45 min on M5 Pro 48GB.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PY="${ROOT}/.venv/bin/python"
SCRIPT="$(dirname "$0")/experiment_internvl8b.py"
LOGDIR="${ROOT}/results/prompt_optimization/matrix-reasoning/internvl-3.5-8b"
MODEL_ID="OpenGVLab/InternVL3_5-8B-HF"
SUITE_LOG="${LOGDIR}/suite_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$LOGDIR"

run_one() {
    local tag="$1"; shift
    echo "──── ${tag} ────" | tee -a "$SUITE_LOG"
    "$PY" "$SCRIPT" --model-id "$MODEL_ID" "$@" \
        --output-dir "$LOGDIR" 2>&1 | tee "${LOGDIR}/run_${tag}.log" | tee -a "$SUITE_LOG"
    echo "" | tee -a "$SUITE_LOG"
}

echo "Matrix Reasoning — InternVL3.5-8B — $(date)" | tee "$SUITE_LOG"
echo "Suite log: $SUITE_LOG"
echo ""

# ── Baselines (resolution comparison) ──────────────────────────────────────
run_one "baseline_512"       --phase 0 --max-image-size 512
run_one "baseline_1024"      --phase 0 --max-image-size 1024

# ── Structured prompt (replicates 2B phase 1) ──────────────────────────────
run_one "phase_1_1024"       --phase 1 --max-image-size 1024

# ── Expert system prompt combos ─────────────────────────────────────────────
run_one "phase_1_3_512"      --phase 1 3 --max-image-size 512
run_one "phase_1_3_1024"     --phase 1 3 --max-image-size 1024

# ── Rule hint at high-res ───────────────────────────────────────────────────
run_one "phase_1_4_1024"     --phase 1 4 --max-image-size 1024

# ── Best 2B config (1+3+4) replicated on 8B at high-res ────────────────────
run_one "phase_1_3_4_1024"   --phase 1 3 4 --max-image-size 1024

# ── Describe-first approach ─────────────────────────────────────────────────
run_one "phase_5_1024"       --phase 5 --max-image-size 1024
run_one "phase_5_3_1024"     --phase 5 3 --max-image-size 1024

# ── Summary ──────────────────────────────────────────────────────────────────
echo "──── SUMMARY ────" | tee -a "$SUITE_LOG"
"$PY" "${ROOT}/scripts/summarize_matrix_8b_phases.py" --dir "$LOGDIR" 2>&1 | tee -a "$SUITE_LOG"

echo ""
echo "Suite complete. Log: $SUITE_LOG"
