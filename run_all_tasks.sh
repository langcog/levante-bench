#!/bin/bash
# Run all tasks with SmolVLM2 on GPU.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

uv run python -m levante_bench.cli experiment=configs/experiment.yaml device=cuda
