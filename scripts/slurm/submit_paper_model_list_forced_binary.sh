#!/bin/bash
# Submit forced-binary scoring runs for the paper-model list on Marlowe.
#
# Default behavior:
# - runs all six benchmark tasks
# - enables forced-binary label scoring
# - skips already-completed local summaries unless FORCE=1
# - skips frontier/hosted models by default (ALLOW_FRONTIER=1 to override)
#
# Usage:
#   bash scripts/slurm/submit_paper_model_list_forced_binary.sh
#   DRY_RUN=1 bash scripts/slurm/submit_paper_model_list_forced_binary.sh
#   FORCE=1 bash scripts/slurm/submit_paper_model_list_forced_binary.sh
#   ONLY="qwen35:0.8B smolvlm2:500M gemini_pro" bash scripts/slurm/submit_paper_model_list_forced_binary.sh
#   TASKS_CSV="trog,vocab" bash scripts/slurm/submit_paper_model_list_forced_binary.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_SUBMIT_SCRIPT="$SCRIPT_DIR/submit_historic_vlms_local.sh"

if [[ ! -f "$BASE_SUBMIT_SCRIPT" ]]; then
  echo "ERROR: missing submit script: $BASE_SUBMIT_SCRIPT" >&2
  exit 1
fi

PROJECT_ROOT="${PROJECT_ROOT:-/projects/m000102}"
CODE_DIR="${CODE_DIR:-$PROJECT_ROOT/code/levante-bench}"
VERSION="${VERSION:-v1}"
TASKS_CSV="${TASKS_CSV:-egma-math,matrix-reasoning,mental-rotation,theory-of-mind,trog,vocab}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$CODE_DIR/results/forced_binary_paper_models}"
DRY_RUN="${DRY_RUN:-0}"
FORCE="${FORCE:-0}"
ALLOW_FRONTIER="${ALLOW_FRONTIER:-0}"

CONDA_ENV_PATH="${CONDA_ENV_PATH:-$PROJECT_ROOT/envs/levante-bench-py311}"
SMOLVLM_CONDA_ENV_PATH="${SMOLVLM_CONDA_ENV_PATH:-$CONDA_ENV_PATH}"
COGVLM_CONDA_ENV_PATH="${COGVLM_CONDA_ENV_PATH:-$PROJECT_ROOT/envs/levante-cogvlm}"
QWEN_CONDA_ENV_PATH="${QWEN_CONDA_ENV_PATH:-}"

FORCE_BINARY_LABEL_SCORING="${FORCE_BINARY_LABEL_SCORING:-1}"
COGVLM_LABEL_SCORING_MODE="${COGVLM_LABEL_SCORING_MODE:-binary}"
STRICT_QWEN_PREFLIGHT="${STRICT_QWEN_PREFLIGHT:-0}"

qwen_env_supports_qwen35() {
  local env_path="$1"
  [[ -x "$env_path/bin/python" ]] || return 1
  "$env_path/bin/python" - <<'PY' >/dev/null 2>&1
import transformers  # noqa: F401
from transformers import AutoModelForImageTextToText  # noqa: F401
PY
}

if [[ -z "$QWEN_CONDA_ENV_PATH" ]]; then
  PREFERRED_CANDIDATES=(
    "$PROJECT_ROOT/envs/${USER:-unknown}-qwen-py311"
    "$PROJECT_ROOT/envs/david81-qwen-py311"
  )
  COMPAT_CANDIDATES=(
    "$PROJECT_ROOT/envs/${USER:-unknown}-levante-py311"
    "$PROJECT_ROOT/envs/david81-levante-py311"
    "$CONDA_ENV_PATH"
  )
  # Prefer explicit qwen env naming when present, even if online probe is flaky.
  for candidate in "${PREFERRED_CANDIDATES[@]}"; do
    if [[ -x "$candidate/bin/python" ]]; then
      QWEN_CONDA_ENV_PATH="$candidate"
      break
    fi
  done
  # If no dedicated qwen env exists, pick first env that passes compatibility probe.
  if [[ -z "$QWEN_CONDA_ENV_PATH" ]]; then
    for candidate in "${COMPAT_CANDIDATES[@]}"; do
      if qwen_env_supports_qwen35 "$candidate"; then
        QWEN_CONDA_ENV_PATH="$candidate"
        break
      fi
    done
  fi
fi
QWEN_CONDA_ENV_PATH="${QWEN_CONDA_ENV_PATH:-$CONDA_ENV_PATH}"

DEFAULT_TARGETS=(
  "gemma4:E2B-it"
  "gemma4:E4B-it"
  "gemma4:26B-A4B-it"
  "gemma4:31B-it"
  "internvl35:1B"
  "internvl35:2B"
  "internvl35:4B"
  "internvl35:8B"
  "internvl35:14B"
  "internvl35:38B"
  "molmo2:4B"
  "molmo2:O-7B"
  "molmo2:8B"
  "qwen35:0.8B"
  "qwen35:2B"
  "qwen35:4B"
  "qwen35:9B"
  "qwen35:27B"
  "smolvlm2:256M"
  "smolvlm2:500M"
  "smolvlm2:2.2B"
  "tinyllava:2.4B"
  "tinyllava:3.1B"
)

declare -A BATCH_SIZE_MAP=(
  ["gemma4:E2B-it"]="2"
  ["gemma4:E4B-it"]="2"
  ["gemma4:26B-A4B-it"]="1"
  ["gemma4:31B-it"]="1"
  ["internvl35:1B"]="4"
  ["internvl35:2B"]="2"
  ["internvl35:4B"]="2"
  ["internvl35:8B"]="1"
  ["internvl35:14B"]="1"
  ["internvl35:38B"]="1"
  ["molmo2:4B"]="2"
  ["molmo2:O-7B"]="1"
  ["molmo2:8B"]="1"
  ["qwen35:0.8B"]="4"
  ["qwen35:2B"]="2"
  ["qwen35:4B"]="2"
  ["qwen35:9B"]="1"
  ["qwen35:27B"]="1"
  ["smolvlm2:256M"]="4"
  ["smolvlm2:500M"]="2"
  ["smolvlm2:2.2B"]="1"
  ["tinyllava:2.4B"]="2"
  ["tinyllava:3.1B"]="2"
)

declare -A MEM_MAP=(
  ["internvl35:38B"]="160G"
  ["qwen35:27B"]="144G"
  ["gemma4:26B-A4B-it"]="144G"
  ["gemma4:31B-it"]="160G"
)

declare -A TIME_MAP=(
  ["gemma4:E2B-it"]="01:30:00"
  ["gemma4:E4B-it"]="02:00:00"
  ["gemma4:26B-A4B-it"]="04:00:00"
  ["gemma4:31B-it"]="04:00:00"
  ["internvl35:1B"]="01:30:00"
  ["internvl35:2B"]="01:30:00"
  ["internvl35:4B"]="02:00:00"
  ["internvl35:8B"]="02:00:00"
  ["internvl35:14B"]="04:00:00"
  ["internvl35:38B"]="06:00:00"
  ["molmo2:4B"]="02:00:00"
  ["molmo2:O-7B"]="02:00:00"
  ["molmo2:8B"]="02:00:00"
  ["qwen35:0.8B"]="01:30:00"
  ["qwen35:2B"]="01:30:00"
  ["qwen35:4B"]="02:00:00"
  ["qwen35:9B"]="04:00:00"
  ["qwen35:27B"]="04:00:00"
  ["smolvlm2:256M"]="01:30:00"
  ["smolvlm2:500M"]="01:30:00"
  ["smolvlm2:2.2B"]="01:30:00"
  ["tinyllava:2.4B"]="01:30:00"
  ["tinyllava:3.1B"]="01:30:00"
)

if [[ -n "${ONLY:-}" ]]; then
  # shellcheck disable=SC2206
  TARGETS=( $ONLY )
else
  TARGETS=("${DEFAULT_TARGETS[@]}")
fi

is_frontier_or_hosted() {
  local model_name="$1"
  case "$model_name" in
    gpt*|gemini*|claude*|*_hf|aya_vision_32b_hf) return 0 ;;
    *) return 1 ;;
  esac
}

has_existing_summary() {
  local label="$1"
  [[ -f "$OUTPUT_ROOT/$label/summary.csv" ]] && return 0
  [[ -f "$OUTPUT_ROOT/$VERSION/$label/summary.csv" ]] && return 0
  [[ -f "$OUTPUT_ROOT/$label/baseline/summary.csv" ]] && return 0
  [[ -f "$OUTPUT_ROOT/$VERSION/$label/baseline/summary.csv" ]] && return 0
  return 1
}

echo "Submitting forced-binary paper-model jobs:"
echo "  VERSION=$VERSION"
echo "  TASKS_CSV=$TASKS_CSV"
echo "  OUTPUT_ROOT=$OUTPUT_ROOT"
echo "  DRY_RUN=$DRY_RUN"
echo "  FORCE=$FORCE"
echo "  ALLOW_FRONTIER=$ALLOW_FRONTIER"
echo "  FORCE_BINARY_LABEL_SCORING=$FORCE_BINARY_LABEL_SCORING"
echo "  COGVLM_LABEL_SCORING_MODE=$COGVLM_LABEL_SCORING_MODE"
echo "  QWEN_CONDA_ENV_PATH=$QWEN_CONDA_ENV_PATH"
echo "  STRICT_QWEN_PREFLIGHT=$STRICT_QWEN_PREFLIGHT"
echo ""

submitted=0
skipped_existing=0
skipped_frontier=0
skipped_malformed=0
skipped_unsupported=0

for target in "${TARGETS[@]}"; do
  if [[ "$target" != *:* ]]; then
    echo "Skipping malformed target (expected model:size): $target" >&2
    skipped_malformed=$((skipped_malformed + 1))
    continue
  fi
  model="${target%%:*}"
  size="${target#*:}"
  label="${model}-${size}"

  if [[ "$ALLOW_FRONTIER" != "1" ]] && is_frontier_or_hosted "$model"; then
    echo "Skipping frontier/hosted model for forced-binary: $target"
    skipped_frontier=$((skipped_frontier + 1))
    continue
  fi

  if [[ "$model" == "qwen35" ]] && ! qwen_env_supports_qwen35 "$QWEN_CONDA_ENV_PATH"; then
    if [[ "$STRICT_QWEN_PREFLIGHT" == "1" ]]; then
      echo "Skipping unsupported Qwen target (Qwen preflight failed in $QWEN_CONDA_ENV_PATH): $target"
      skipped_unsupported=$((skipped_unsupported + 1))
      continue
    else
      echo "WARNING: Qwen preflight failed in $QWEN_CONDA_ENV_PATH; submitting anyway: $target" >&2
    fi
  fi

  if [[ "$FORCE" != "1" ]] && has_existing_summary "$label"; then
    echo "Skipping existing result: $label"
    skipped_existing=$((skipped_existing + 1))
    continue
  fi

  batch_size="${BATCH_SIZE_MAP[$target]:-1}"
  mem_limit="${MEM_MAP[$target]:-96G}"
  time_limit="${TIME_MAP[$target]:-04:00:00}"

  echo "Submitting $target (batch_size=$batch_size mem=$mem_limit time=$time_limit)"
  DRY_RUN="$DRY_RUN" \
  VERSION="$VERSION" \
  TASKS_CSV="$TASKS_CSV" \
  OUTPUT_ROOT="$OUTPUT_ROOT" \
  CONDA_ENV_PATH="$CONDA_ENV_PATH" \
  SMOLVLM_CONDA_ENV_PATH="$SMOLVLM_CONDA_ENV_PATH" \
  COGVLM_CONDA_ENV_PATH="$COGVLM_CONDA_ENV_PATH" \
  QWEN_CONDA_ENV_PATH="$QWEN_CONDA_ENV_PATH" \
  STRICT_QWEN_PREFLIGHT="$STRICT_QWEN_PREFLIGHT" \
  FORCE_BINARY_LABEL_SCORING="$FORCE_BINARY_LABEL_SCORING" \
  COGVLM_LABEL_SCORING_MODE="$COGVLM_LABEL_SCORING_MODE" \
  BATCH_SIZE="$batch_size" \
  MEM_LIMIT="$mem_limit" \
  TIME_LIMIT="$time_limit" \
  bash "$BASE_SUBMIT_SCRIPT" "$target"

  submitted=$((submitted + 1))
done

echo ""
echo "Done. submitted=$submitted skipped_existing=$skipped_existing skipped_frontier=$skipped_frontier skipped_unsupported=$skipped_unsupported skipped_malformed=$skipped_malformed"
