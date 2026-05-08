#!/bin/bash
# Prefetch CogVLM + tokenizer repos into shared /scratch cache.
#
# Usage:
#   bash scripts/slurm/prefetch_cogvlm_to_scratch.sh
#   SCRATCH_ROOT=/scratch/m000102/$USER/levante-hf bash scripts/slurm/prefetch_cogvlm_to_scratch.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

ENV_PATH="${ENV_PATH:-/projects/m000102/envs/levante-cogvlm}"
PY_BIN="${PY_BIN:-$ENV_PATH/bin/python}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/m000102/$USER/levante-hf}"

export HF_HOME="${HF_HOME:-$SCRATCH_ROOT/hf-home}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$SCRATCH_ROOT/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$SCRATCH_ROOT/transformers}"
export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"

mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE"

if [[ ! -x "$PY_BIN" ]]; then
  echo "ERROR: Python binary not executable: $PY_BIN" >&2
  exit 1
fi

echo "Prefetching to shared scratch cache:"
echo "  HF_HOME=$HF_HOME"
echo "  HF_HUB_CACHE=$HF_HUB_CACHE"
echo "  TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE"
echo "  PY_BIN=$PY_BIN"
echo ""

cd "$REPO_DIR"

PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=0 "$PY_BIN" - <<'PY'
from huggingface_hub import snapshot_download

repos = [
    "THUDM/cogvlm-chat-hf",
    "lmsys/vicuna-7b-v1.5",
    "EleutherAI/gpt-neox-20b",
]
for repo in repos:
    path = snapshot_download(repo_id=repo, resume_download=True, max_workers=1)
    print(f"{repo} -> {path}")
PY

echo ""
echo "Prefetch complete."
echo "Cache size:"
du -sh "$HF_HUB_CACHE" || true
