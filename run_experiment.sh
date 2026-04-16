#!/bin/bash
# Run an evaluation experiment.
# Usage:
#   ./run_experiment.sh configs/experiments/experiment.yaml
#   ./run_experiment.sh configs/experiments/experiment.yaml device=cuda
#   ./run_experiment.sh configs/my_custom_exp.yaml models=[smolvlm2] device=cuda

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ -x ".venv/bin/python" ]]; then
    PYTHON_BIN=".venv/bin/python"
else
    PYTHON_BIN="python3"
fi

if [ -z "$1" ]; then
    echo "Usage: $0 <experiment_config.yaml> [overrides...]"
    exit 1
fi

EXPERIMENT_CONFIG="$1"
shift

"$PYTHON_BIN" -m levante_bench.cli experiment="$EXPERIMENT_CONFIG" "$@"
