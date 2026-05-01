#!/bin/bash
#SBATCH --job-name=gemma4_e2b_multirun
#SBATCH --partition=preempt
#SBATCH --account=marlowe-m000102
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/projects/m000102/outputs/slurm/%x-%j.out
#SBATCH --error=/projects/m000102/outputs/slurm/%x-%j.err
#SBATCH --export=NONE
#SBATCH --requeue

set -euo pipefail

PROJECT_ROOT="/projects/m000102"
EXPERIMENT_CONFIG="configs/experiments/gemma4_e2b_multirun.yaml"
USE_JOB_OUTPUT_ROOT="1"

CONDA_ENV_PATH="$PROJECT_ROOT/envs/levante-bench-py311"
DEVICE="cuda"

source "$PROJECT_ROOT/code/levante-bench/scripts/slurm/_levante_common.sh"
