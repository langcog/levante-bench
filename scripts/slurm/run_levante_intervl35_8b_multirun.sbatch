#!/bin/bash
#SBATCH --job-name=intervl35_8b_multirun
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

module purge
module load conda
module load stockcuda/12.6.2

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /projects/m000102/envs/levante-bench-py311

cd /projects/m000102/code/levante-bench
mkdir -p /projects/m000102/outputs/slurm

EXPERIMENT_CONFIG="configs/experiments/internvl35_8b_multirun.yaml"

echo "Starting levante-bench on $(hostname) at $(date)"
echo "Command: ./run_experiment.sh \"$EXPERIMENT_CONFIG\" device=cuda"

./run_experiment.sh "$EXPERIMENT_CONFIG" device=cuda
#!/bin/bash
#SBATCH --job-name=levante-smolvlm2-multirun
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

module purge
module load conda
module load stockcuda/12.6.2

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /projects/m000102/envs/levante-bench-py311

# Base paths
PROJECT_ROOT=/projects/m000102
CODE_DIR=$PROJECT_ROOT/code/levante-bench
SLURM_LOG_DIR=$PROJECT_ROOT/outputs/slurm
RUN_ROOT=$PROJECT_ROOT/outputs/runs

mkdir -p "$SLURM_LOG_DIR"
mkdir -p "$RUN_ROOT"

# Unique directory for this job
RUN_DIR="$RUN_ROOT/job_${SLURM_JOB_ID}"
mkdir -p "$RUN_DIR"

cd "$CODE_DIR"

EXPERIMENT_CONFIG="configs/experiments/internvl35_8b_multirun.yaml"

echo "Starting levante-bench on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Run directory: $RUN_DIR"
echo "Command: ./run_experiment.sh \"$EXPERIMENT_CONFIG\" device=cuda output_dir=\"$RUN_DIR\""

./run_experiment.sh "$EXPERIMENT_CONFIG" device=cuda output_dir="$RUN_DIR"
#!/bin/bash
#SBATCH --job-name=levante-intervl8b-multirun
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

module purge
module load conda
module load stockcuda/12.6.2

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /projects/m000102/envs/levante-bench-py311

PROJECT_ROOT=/projects/m000102
CODE_DIR=$PROJECT_ROOT/code/levante-bench
SLURM_LOG_DIR=$PROJECT_ROOT/outputs/slurm
RESULTS_ROOT=$PROJECT_ROOT/outputs/results

mkdir -p "$SLURM_LOG_DIR"
mkdir -p "$RESULTS_ROOT"

EXPERIMENT_CONFIG="configs/experiments/internvl35_multirun.yaml"

# Unique root tree for this job
JOB_OUTPUT_ROOT="$RESULTS_ROOT/job_${SLURM_JOB_ID}"
mkdir -p "$JOB_OUTPUT_ROOT"

cd "$CODE_DIR"

echo "Starting levante-bench on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Job output root: $JOB_OUTPUT_ROOT"
echo "Command: ./run_experiment.sh \"$EXPERIMENT_CONFIG\" device=cuda output_dir=\"$JOB_OUTPUT_ROOT\""

./run_experiment.sh "$EXPERIMENT_CONFIG" device=cuda output_dir="$JOB_OUTPUT_ROOT"
