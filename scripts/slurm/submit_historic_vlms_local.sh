#!/bin/bash
# Submit one Marlowe job per historic VLM local model.
#
# Usage:
#   bash scripts/slurm/submit_historic_vlms_local.sh
#   DRY_RUN=1 bash scripts/slurm/submit_historic_vlms_local.sh
#   VERSION=v1_additional_images bash scripts/slurm/submit_historic_vlms_local.sh
#   bash scripts/slurm/submit_historic_vlms_local.sh llava15_13b cogvlm openflamingo9b

set -euo pipefail

DRY_RUN="${DRY_RUN:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENSURE_SBATCH_SCRIPT="$SCRIPT_DIR/_ensure_sbatch.sh"

if [[ ! -f "$ENSURE_SBATCH_SCRIPT" ]]; then
  echo "ERROR: missing sbatch helper: $ENSURE_SBATCH_SCRIPT" >&2
  exit 1
fi
# shellcheck disable=SC1090
source "$ENSURE_SBATCH_SCRIPT"
if [[ "$DRY_RUN" != "1" ]]; then
  ensure_sbatch_available
fi

SBATCH_SCRIPT="$SCRIPT_DIR/run_historic_vlm_local.sbatch"

if [[ ! -f "$SBATCH_SCRIPT" ]]; then
  echo "ERROR: missing sbatch template: $SBATCH_SCRIPT" >&2
  exit 1
fi

PROJECT_ROOT="${PROJECT_ROOT:-/projects/m000102}"
CODE_DIR="${CODE_DIR:-$PROJECT_ROOT/code/levante-bench}"
VERSION="${VERSION:-v1}"
DEVICE="${DEVICE:-cuda}"
NUM_RUNS="${NUM_RUNS:-1}"
TRUE_RANDOM_OPTION_ORDER="${TRUE_RANDOM_OPTION_ORDER:-false}"
TASKS_CSV="${TASKS_CSV:-egma-math,matrix-reasoning,mental-rotation,theory-of-mind,trog,vocab}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$CODE_DIR/results/v1_additional_models}"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-$PROJECT_ROOT/envs/levante-bench-py311}"
COGVLM_CONDA_ENV_PATH="${COGVLM_CONDA_ENV_PATH:-$PROJECT_ROOT/envs/levante-cogvlm}"
SMOLVLM_CONDA_ENV_PATH="${SMOLVLM_CONDA_ENV_PATH:-$CONDA_ENV_PATH}"
QWEN_CONDA_ENV_PATH="${QWEN_CONDA_ENV_PATH:-$CONDA_ENV_PATH}"
IMAGE_SIZE="${IMAGE_SIZE:-}"
COGVLM_LABEL_SCORING_MODE="${COGVLM_LABEL_SCORING_MODE:-}"
FORCE_BINARY_LABEL_SCORING="${FORCE_BINARY_LABEL_SCORING:-}"
STRICT_QWEN_PREFLIGHT="${STRICT_QWEN_PREFLIGHT:-0}"

# Per-model defaults. Override globally via env vars if needed.
declare -A TIME_MAP=(
  ["llava15_13b"]="08:00:00"
  ["cogvlm"]="10:00:00"
  ["openflamingo9b"]="10:00:00"
  ["smolvlm2-256M"]="04:00:00"
  ["smolvlm2-500M"]="04:00:00"
)
declare -A MEM_MAP=(
  ["llava15_13b"]="96G"
  ["cogvlm"]="120G"
  ["openflamingo9b"]="120G"
  ["smolvlm2-256M"]="96G"
  ["smolvlm2-500M"]="96G"
)
declare -A BATCH_SIZE_MAP=(
  ["llava15_13b"]="1"
  ["cogvlm"]="1"
  ["openflamingo9b"]="1"
  ["smolvlm2-256M"]="1"
  ["smolvlm2-500M"]="1"
)
DEFAULT_MODELS=(
  "llava15_13b"
  "cogvlm"
  "openflamingo9b"
)

normalize_model_id() {
  local raw="$1"
  case "$raw" in
    smolvlm2:256M) echo "smolvlm2-256M" ;;
    smolvlm2:500M) echo "smolvlm2-500M" ;;
    *) echo "$raw" ;;
  esac
}

resolve_model_fields() {
  local normalized="$1"
  MODEL_NAME_RESOLVED="$normalized"
  MODEL_SIZE_RESOLVED=""
  if [[ "$normalized" == *:* ]]; then
    MODEL_NAME_RESOLVED="${normalized%%:*}"
    MODEL_SIZE_RESOLVED="${normalized#*:}"
    return
  fi
  case "$normalized" in
    smolvlm2-256M)
      MODEL_NAME_RESOLVED="smolvlm2"
      MODEL_SIZE_RESOLVED="256M"
      ;;
    smolvlm2-500M)
      MODEL_NAME_RESOLVED="smolvlm2"
      MODEL_SIZE_RESOLVED="500M"
      ;;
    *-*)
      local maybe_name="${normalized%-*}"
      local maybe_size="${normalized##*-}"
      if [[ "$maybe_size" =~ ^[0-9]+(\.[0-9]+)?[A-Za-z]+$ ]]; then
        MODEL_NAME_RESOLVED="$maybe_name"
        MODEL_SIZE_RESOLVED="$maybe_size"
      fi
      ;;
  esac
}

if [[ $# -gt 0 ]]; then
  MODELS=("$@")
else
  MODELS=("${DEFAULT_MODELS[@]}")
fi

echo "Submitting historic local VLM jobs:"
echo "  PROJECT_ROOT=$PROJECT_ROOT"
echo "  CODE_DIR=$CODE_DIR"
echo "  VERSION=$VERSION"
echo "  DEVICE=$DEVICE"
echo "  OUTPUT_ROOT=$OUTPUT_ROOT"
echo "  NUM_RUNS=$NUM_RUNS"
echo "  TRUE_RANDOM_OPTION_ORDER=$TRUE_RANDOM_OPTION_ORDER"
echo "  TASKS_CSV=$TASKS_CSV"
echo "  DRY_RUN=$DRY_RUN"
echo "  CONDA_ENV_PATH=$CONDA_ENV_PATH"
echo "  COGVLM_CONDA_ENV_PATH=$COGVLM_CONDA_ENV_PATH"
echo "  SMOLVLM_CONDA_ENV_PATH=$SMOLVLM_CONDA_ENV_PATH"
echo "  QWEN_CONDA_ENV_PATH=$QWEN_CONDA_ENV_PATH"
echo "  STRICT_QWEN_PREFLIGHT=$STRICT_QWEN_PREFLIGHT"
echo "  IMAGE_SIZE=${IMAGE_SIZE:-<default>}"
echo "  COGVLM_LABEL_SCORING_MODE=${COGVLM_LABEL_SCORING_MODE:-<default>}"
echo "  FORCE_BINARY_LABEL_SCORING=${FORCE_BINARY_LABEL_SCORING:-<default>}"
echo ""

# sbatch --export uses commas as separators; encode task list.
TASKS_CSV_EXPORT="${TASKS_CSV//,/;}"

for model in "${MODELS[@]}"; do
  model="$(normalize_model_id "$model")"
  resolve_model_fields "$model"

  time_limit="${TIME_LIMIT:-${TIME_MAP[$model]:-04:00:00}}"
  mem_limit="${MEM_LIMIT:-${MEM_MAP[$model]:-96G}}"
  batch_size="${BATCH_SIZE:-${BATCH_SIZE_MAP[$model]:-1}}"
  job_name="historic-${model}"
  job_conda_env="$CONDA_ENV_PATH"
  if [[ "$MODEL_NAME_RESOLVED" == "cogvlm" ]]; then
    job_conda_env="$COGVLM_CONDA_ENV_PATH"
  elif [[ "$MODEL_NAME_RESOLVED" == "qwen35" ]]; then
    job_conda_env="$QWEN_CONDA_ENV_PATH"
    if [[ "$DRY_RUN" != "1" ]]; then
      if [[ ! -x "$job_conda_env/bin/python" ]]; then
        echo "ERROR: Qwen env python missing at $job_conda_env/bin/python" >&2
        echo "Set QWEN_CONDA_ENV_PATH to an env with a newer Transformers build." >&2
        exit 1
      fi
      if ! "$job_conda_env/bin/python" - <<'PY' >/dev/null 2>&1
import omegaconf  # noqa: F401
PY
      then
        echo "ERROR: omegaconf missing in Qwen env: $job_conda_env" >&2
        echo "Install core deps in that env (e.g. pip install omegaconf)." >&2
        exit 1
      fi
      if ! "$job_conda_env/bin/python" - <<'PY' >/dev/null 2>&1
from PIL import Image  # noqa: F401
PY
      then
        echo "ERROR: Pillow (PIL) missing in Qwen env: $job_conda_env" >&2
        echo "Install core deps in that env (e.g. pip install Pillow)." >&2
        exit 1
      fi
      if ! "$job_conda_env/bin/python" - <<'PY' >/dev/null 2>&1
import requests  # noqa: F401
PY
      then
        echo "ERROR: requests missing in Qwen env: $job_conda_env" >&2
        echo "Install core deps in that env (e.g. pip install requests)." >&2
        exit 1
      fi
      if ! "$job_conda_env/bin/python" - <<'PY' >/dev/null 2>&1
import pandas as pd  # noqa: F401
PY
      then
        echo "ERROR: pandas missing in Qwen env: $job_conda_env" >&2
        echo "Install core deps in that env (e.g. pip install pandas)." >&2
        exit 1
      fi
      if ! "$job_conda_env/bin/python" - <<'PY' >/dev/null 2>&1
import torchvision  # noqa: F401
PY
      then
        echo "ERROR: torchvision missing in Qwen env: $job_conda_env" >&2
        echo "Install torchvision in that env (must match torch build)." >&2
        exit 1
      fi
      if ! "$job_conda_env/bin/python" - <<'PY' >/dev/null 2>&1
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
if "qwen3_5" not in CONFIG_MAPPING_NAMES:
    raise SystemExit(1)
PY
      then
        if [[ "$STRICT_QWEN_PREFLIGHT" == "1" ]]; then
          echo "ERROR: Qwen env lacks qwen3_5 architecture support: $job_conda_env" >&2
          echo "Use an env with updated Transformers, then set QWEN_CONDA_ENV_PATH." >&2
          exit 1
        else
          echo "WARNING: Qwen preflight did not detect qwen3_5 support in $job_conda_env; proceeding anyway (STRICT_QWEN_PREFLIGHT=0)." >&2
        fi
      fi
    fi
  elif [[ "$MODEL_NAME_RESOLVED" == "smolvlm2" ]]; then
    job_conda_env="$SMOLVLM_CONDA_ENV_PATH"
    if [[ "$DRY_RUN" != "1" ]]; then
      if [[ ! -x "$job_conda_env/bin/python" ]]; then
        echo "ERROR: SmolVLM env python missing at $job_conda_env/bin/python" >&2
        echo "Set SMOLVLM_CONDA_ENV_PATH to an env containing num2words." >&2
        exit 1
      fi
      if ! "$job_conda_env/bin/python" -c "import num2words" >/dev/null 2>&1; then
        echo "ERROR: num2words missing in SmolVLM env: $job_conda_env" >&2
        echo "Install num2words there, or set SMOLVLM_CONDA_ENV_PATH accordingly." >&2
        exit 1
      fi
    fi
  fi

  cmd=(
    sbatch
    --job-name "$job_name"
    --time "$time_limit"
    --mem "$mem_limit"
    --export "PROJECT_ROOT=$PROJECT_ROOT,CODE_DIR=$CODE_DIR,CONDA_ENV_PATH=$job_conda_env,LEVANTE_PYTHON_BIN=$job_conda_env/bin/python,MODEL_ID=$model,MODEL_NAME=$MODEL_NAME_RESOLVED,MODEL_SIZE=$MODEL_SIZE_RESOLVED,VERSION=$VERSION,DEVICE=$DEVICE,BATCH_SIZE=$batch_size,NUM_RUNS=$NUM_RUNS,TRUE_RANDOM_OPTION_ORDER=$TRUE_RANDOM_OPTION_ORDER,TASKS_CSV=$TASKS_CSV_EXPORT,IMAGE_SIZE=$IMAGE_SIZE,COGVLM_LABEL_SCORING_MODE=$COGVLM_LABEL_SCORING_MODE,FORCE_BINARY_LABEL_SCORING=$FORCE_BINARY_LABEL_SCORING,OUTPUT_ROOT=$OUTPUT_ROOT,USE_JOB_OUTPUT_ROOT=0,HF_TOKEN,HUGGINGFACEHUB_API_TOKEN,HF_HOME,HF_HUB_CACHE,TRANSFORMERS_CACHE"
    "$SBATCH_SCRIPT"
  )

  echo "Model=$model resolved=${MODEL_NAME_RESOLVED}${MODEL_SIZE_RESOLVED:+:$MODEL_SIZE_RESOLVED} time=$time_limit mem=$mem_limit env=$job_conda_env batch_size=$batch_size"
  if [[ "$DRY_RUN" == "1" ]]; then
    printf 'DRY_RUN:'
    printf ' %q' "${cmd[@]}"
    printf '\n'
  else
    "${cmd[@]}"
  fi
done

