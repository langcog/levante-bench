#!/bin/bash
# Run an evaluation experiment.
# Usage:
#   ./run_experiment.sh configs/experiment.yaml
#   ./run_experiment.sh configs/experiment.yaml device=cuda
#   ./run_experiment.sh configs/my_custom_exp.yaml models=[smolvlm2] device=cuda

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <experiment_config.yaml> [overrides...]"
    exit 1
fi

EXPERIMENT_CONFIG="$1"
shift

python -m levante_bench.cli experiment="$EXPERIMENT_CONFIG" "$@"
