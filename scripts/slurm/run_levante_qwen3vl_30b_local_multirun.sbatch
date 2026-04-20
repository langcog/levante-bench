#!/bin/bash
#SBATCH --job-name=qwen3vl_30b_local_multirun
#SBATCH --partition=preempt
#SBATCH --account=marlowe-m000102
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/projects/m000102/outputs/slurm/%x-%j.out
#SBATCH --error=/projects/m000102/outputs/slurm/%x-%j.err
#SBATCH --requeue

set -euo pipefail

# ----------------------------
# User-tunable parameters
# ----------------------------
PROJECT_ROOT="/projects/m000102"
EXPERIMENT_CONFIG="configs/experiments/qwen3vl_30b_local_multirun.yaml"
USE_JOB_OUTPUT_ROOT="1"

# Optional overrides (usually leave as-is)
CONDA_ENV_PATH="$PROJECT_ROOT/envs/levante-bench-py311"
DEVICE="cuda"
# EXTRA_OVERRIDES="num_runs=3"

source "$PROJECT_ROOT/code/levante-bench/scripts/slurm/_levante_common.sh"
