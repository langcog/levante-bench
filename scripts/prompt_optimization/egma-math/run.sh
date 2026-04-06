#!/usr/bin/env bash
# Run egma-math phase experiments sequentially; logs under results/prompt_optimization/egma-math/qwen-0.8b/
set -u
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PY="${ROOT}/.venv/bin/python"
SCRIPT="$(dirname "$0")/experiment.py"
LOGDIR="${ROOT}/results/prompt_optimization/egma-math/qwen-0.8b"
mkdir -p "$LOGDIR"

if [[ ! -x "$PY" ]]; then
  echo "Missing venv python: $PY" >&2
  exit 1
fi

run_one() {
  local tag="$1"
  shift
  echo ""
  echo "========== $(date -Iseconds) START ${tag} =========="
  echo "========== $(date -Iseconds) START ${tag} ==========" >>"${LOGDIR}/suite.log"
  "$PY" "$SCRIPT" "$@" 2>&1 | tee -a "${LOGDIR}/run_${tag}.log"
  echo "========== END ${tag} ==========" >>"${LOGDIR}/suite.log"
  echo "========== END ${tag} =========="
}

cd "$ROOT"

run_one "phase_0" --phase 0
run_one "phase_1" --phase 1
run_one "phase_2" --phase 2
run_one "phase_3" --phase 3
run_one "phase_4" --phase 4
run_one "phase_5" --phase 5
run_one "phase_1_5" --phase 1 5
run_one "phase_1_2_3_4_5" --phase 1 2 3 4 5

echo ""
echo "All runs finished. Logs: ${LOGDIR}/run_*.log"

echo ""
echo "========== $(date -Iseconds) SUMMARY TABLE =========="
"$PY" "${ROOT}/scripts/prompt_optimization/egma-math/summarize.py" --dir "$LOGDIR" | tee -a "${LOGDIR}/suite.log"
