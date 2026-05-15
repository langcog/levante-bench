#!/bin/bash
#SBATCH --job-name=cogvlm-scratch
#SBATCH --partition=preempt
#SBATCH --account=marlowe-m000102
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=220G
#SBATCH --time=12:00:00
#SBATCH --output=/projects/m000102/outputs/slurm/%x-%j.out
#SBATCH --error=/projects/m000102/outputs/slurm/%x-%j.err
#SBATCH --export=NONE
#SBATCH --requeue

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/projects/m000102}"
CODE_DIR="${CODE_DIR:-$PROJECT_ROOT/code/levante-bench}"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-$PROJECT_ROOT/envs/levante-cogvlm}"
CUDA_MODULE="${CUDA_MODULE:-stockcuda/12.6.2}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/m000102/$USER/levante-hf}"
VERSION="${VERSION:-v1}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_RUNS="${NUM_RUNS:-1}"
TRUE_RANDOM_OPTION_ORDER="${TRUE_RANDOM_OPTION_ORDER:-false}"
TASKS_CSV="${TASKS_CSV:-egma-math,matrix-reasoning,mental-rotation,theory-of-mind,trog,vocab}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$CODE_DIR/results/v1_additional_models}"
USE_JOB_OUTPUT_ROOT="${USE_JOB_OUTPUT_ROOT:-0}"

export HF_HOME="${HF_HOME:-$SCRATCH_ROOT/hf-home}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$SCRATCH_ROOT/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$SCRATCH_ROOT/transformers}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export PYTHONPATH="${PYTHONPATH:-$CODE_DIR/src}"

if [[ ! -d "$CODE_DIR" ]]; then
  echo "ERROR: CODE_DIR not found: $CODE_DIR" >&2
  exit 1
fi

if [[ ! -d "$HF_HUB_CACHE" ]]; then
  echo "ERROR: HF_HUB_CACHE not found: $HF_HUB_CACHE" >&2
  echo "Run scripts/slurm/prefetch_cogvlm_to_scratch.sh first." >&2
  exit 1
fi

COG_SNAPSHOT="$(ls -d "$HF_HUB_CACHE"/models--THUDM--cogvlm-chat-hf/snapshots/* 2>/dev/null | head -n1 || true)"
if [[ -z "$COG_SNAPSHOT" || ! -d "$COG_SNAPSHOT" ]]; then
  echo "ERROR: CogVLM snapshot not found under $HF_HUB_CACHE" >&2
  exit 1
fi

if [[ -n "${TOKENIZER_SNAPSHOT:-}" ]]; then
  TOK_SNAPSHOT="$TOKENIZER_SNAPSHOT"
else
  TOK_SNAPSHOT="$(ls -d "$HF_HUB_CACHE"/models--lmsys--vicuna-7b-v1.5/snapshots/* 2>/dev/null | head -n1 || true)"
  if [[ -z "$TOK_SNAPSHOT" || ! -d "$TOK_SNAPSHOT" ]]; then
    TOK_SNAPSHOT="$(ls -d "$HF_HUB_CACHE"/models--EleutherAI--gpt-neox-20b/snapshots/* 2>/dev/null | head -n1 || true)"
  fi
fi

if [[ -z "$TOK_SNAPSHOT" || ! -d "$TOK_SNAPSHOT" ]]; then
  echo "ERROR: tokenizer snapshot not found (vicuna or gpt-neox)." >&2
  exit 1
fi

mkdir -p "$PROJECT_ROOT/outputs/slurm" "$OUTPUT_ROOT"

EXPERIMENT_CONFIG="$(mktemp "/tmp/levante-cogvlm-${SLURM_JOB_ID:-manual}-XXXX.yaml")"
trap 'rm -f "$EXPERIMENT_CONFIG"' EXIT

{
  echo "models:"
  echo "  - name: cogvlm"
  echo "    hf_name: ${COG_SNAPSHOT}"
  echo "    tokenizer_hf_name: ${TOK_SNAPSHOT}"
  echo "    dtype: float16"
  echo "    attn_implementation: sdpa"
  echo "    trust_remote_code: true"
  echo "    use_json_format: false"
  echo "    max_new_tokens: 16"
  echo ""
  echo "tasks:"
  tasks_normalized="${TASKS_CSV//;/,}"
  IFS=',' read -r -a TASK_LIST <<< "$tasks_normalized"
  for task in "${TASK_LIST[@]}"; do
    task_trimmed="$(echo "$task" | xargs)"
    if [[ -n "$task_trimmed" ]]; then
      echo "  - ${task_trimmed}"
    fi
  done
  echo ""
  echo "data_root: data"
  echo "output_dir: ${OUTPUT_ROOT}"
  echo "version: ${VERSION}"
  echo "device: ${DEVICE}"
  echo "batch_size: ${BATCH_SIZE}"
  echo "true_random_option_order: ${TRUE_RANDOM_OPTION_ORDER}"
  echo "num_runs: ${NUM_RUNS}"
} > "$EXPERIMENT_CONFIG"

echo "Using CogVLM snapshot: $COG_SNAPSHOT"
echo "Using tokenizer snapshot: $TOK_SNAPSHOT"
echo "HF_HUB_OFFLINE=$HF_HUB_OFFLINE"
echo "Generated experiment config: $EXPERIMENT_CONFIG"
sed -n '1,140p' "$EXPERIMENT_CONFIG"

export PROJECT_ROOT
export CONDA_ENV_PATH
export CUDA_MODULE
export USE_JOB_OUTPUT_ROOT
export EXPERIMENT_CONFIG
export RESULTS_ROOT="$OUTPUT_ROOT"

source "$CODE_DIR/scripts/slurm/_levante_common.sh"
