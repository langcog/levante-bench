#!/usr/bin/env bash
# Wait for qwen35 0.8B es 10x to finish, then stop old 10x sweep
# and launch remaining combos as deterministic baselines.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATE_TAG="${DATE_TAG:-2026-05-04}"
VERSION="${VERSION:-v1}"
TARGET_DIR="results/${VERSION}/qwen35-0.8B-es/rtx3090-${DATE_TAG}-qwen35-0.8B-es-10x"
NEEDED_RUNS="${NEEDED_RUNS:-10}"

echo "Watching for completion: ${TARGET_DIR} (${NEEDED_RUNS} summaries needed)"

while true; do
  count="$(python3 - <<'PY'
import os
from pathlib import Path
version=os.environ.get("VERSION","v1")
date_tag=os.environ.get("DATE_TAG","2026-05-04")
p=Path(f"results/{version}/qwen35-0.8B-es/rtx3090-{date_tag}-qwen35-0.8B-es-10x")
print(len(list(p.glob("*/summary.csv"))) if p.exists() else 0)
PY
)"
  echo "qwen35 0.8B es summaries: ${count}/${NEEDED_RUNS}"
  if [[ "${count}" -ge "${NEEDED_RUNS}" ]]; then
    break
  fi
  sleep 60
done

echo "Current qwen35 0.8B es block is complete. Stopping old sweep."
pkill -f "bash scripts/run_qwen_smolvlm_multilang_10x_3090.sh" || true
sleep 2

echo "Launching remaining deterministic baseline combos."
bash scripts/run_remaining_qwen_smolvlm_baselines_3090.sh
echo "Handoff complete."
