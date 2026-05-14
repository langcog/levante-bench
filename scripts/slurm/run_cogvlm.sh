#!/usr/bin/env bash
# Save as: /projects/m000102/code/levante-bench/scripts/slurm/run_cogvlm_from_scratch.sh
# Usage:
#   bash scripts/slurm/run_cogvlm_from_scratch.sh            # predownload + smoke
#   FULL=1 bash scripts/slurm/run_cogvlm_from_scratch.sh     # predownload + full run
#   FULL=1 BATCH_SIZE=1 VERSION=v1 bash scripts/slurm/run_cogvlm_from_scratch.sh

set -euo pipefail

REPO="/projects/m000102/code/levante-bench"
ENV="/projects/m000102/envs/levante-cogvlm"

VERSION="${VERSION:-v1}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-1}"
FULL="${FULL:-0}"                # 0 = vocab smoke, 1 = full benchmark

SCR_ROOT="/scratch/m000102/${USER}/levante-hf"
export HF_HOME="${SCR_ROOT}/hf-home"
export HF_HUB_CACHE="${SCR_ROOT}/hub"
export TRANSFORMERS_CACHE="${SCR_ROOT}/transformers"
export HF_HUB_DISABLE_TELEMETRY=1

mkdir -p "${HF_HOME}" "${HF_HUB_CACHE}" "${TRANSFORMERS_CACHE}"

echo "==> Repo: ${REPO}"
echo "==> Env:  ${ENV}"
echo "==> HF cache root: ${SCR_ROOT}"

cd "${REPO}"

# 1) Predownload required repos to shared /scratch
"${ENV}/bin/python" - <<'PY'
from huggingface_hub import snapshot_download

repos = [
    "THUDM/cogvlm-chat-hf",
    "lmsys/vicuna-7b-v1.5",
]
for r in repos:
    p = snapshot_download(repo_id=r, resume_download=True)
    print(f"Downloaded: {r} -> {p}")
PY

# 2) Run eval using local cache (offline)
export PYTHONNOUSERSITE=1
export PYTHONPATH="${REPO}/src"
export HF_HUB_OFFLINE=1

if [[ "${FULL}" == "1" ]]; then
  echo "==> Running FULL cogvlm benchmark"
  "${ENV}/bin/python" -m levante_bench.cli run-eval \
    --model cogvlm \
    --version "${VERSION}" \
    --device "${DEVICE}" \
    --batch-size "${BATCH_SIZE}"
else
  echo "==> Running VOCAB smoke for cogvlm"
  "${ENV}/bin/python" -m levante_bench.cli run-eval \
    --model cogvlm \
    --task vocab \
    --version "${VERSION}" \
    --device "${DEVICE}" \
    --batch-size "${BATCH_SIZE}"
fi

echo "==> Done."
