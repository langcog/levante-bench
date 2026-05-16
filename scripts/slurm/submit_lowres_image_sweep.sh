#!/bin/bash
# Submit low-resolution image-size sweeps for the 4 lowest-performing small models.
#
# For each selected model, submits two jobs:
#   - IMAGE_SIZE=256
#   - IMAGE_SIZE=128
# on tasks: egma-math,vocab
#
# Usage:
#   bash scripts/slurm/submit_lowres_image_sweep.sh
#   DRY_RUN=1 bash scripts/slurm/submit_lowres_image_sweep.sh
#   VERSION=v1 OUTPUT_BASE=/projects/m000102/code/levante-bench/results/lowres_sweep bash scripts/slurm/submit_lowres_image_sweep.sh
#   MODELS_CSV="qwen35:0.8B,smolvlm2:256M,tinyllava:2.4B,internvl35:1B" bash scripts/slurm/submit_lowres_image_sweep.sh

set -euo pipefail

if ! command -v sbatch >/dev/null 2>&1; then
  echo "ERROR: sbatch not found in PATH." >&2
  echo "Run this script on a Marlowe login node." >&2
  exit 127
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_SCRIPT="$SCRIPT_DIR/run_lowres_image_sweep.sbatch"
if [[ ! -f "$SBATCH_SCRIPT" ]]; then
  echo "ERROR: missing sbatch template: $SBATCH_SCRIPT" >&2
  exit 1
fi

PROJECT_ROOT="${PROJECT_ROOT:-/projects/m000102}"
CODE_DIR="${CODE_DIR:-$PROJECT_ROOT/code/levante-bench}"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-$PROJECT_ROOT/envs/levante-bench-py311}"
VERSION="${VERSION:-v1}"
DEVICE="${DEVICE:-cuda}"
TASKS_CSV="${TASKS_CSV:-egma-math,vocab}"
IMAGE_SIZES_CSV="${IMAGE_SIZES_CSV:-256,128}"
OUTPUT_BASE="${OUTPUT_BASE:-$CODE_DIR/results/lowres_sweep}"
TIME_LIMIT="${TIME_LIMIT:-04:00:00}"
MEM_LIMIT="${MEM_LIMIT:-96G}"
DRY_RUN="${DRY_RUN:-0}"

# Optional explicit model list, comma-separated "model:size" tokens.
MODELS_CSV="${MODELS_CSV:-}"

# Candidate small/low-capacity models for auto-selection.
DEFAULT_CANDIDATES=(
  "internvl35:1B"
  "internvl35:2B"
  "qwen35:0.8B"
  "qwen35:2B"
  "smolvlm2:256M"
  "smolvlm2:500M"
  "tinyllava:2.4B"
  "tinyllava:3.1B"
)

echo "Submitting low-resolution sweep jobs:"
echo "  CODE_DIR=$CODE_DIR"
echo "  VERSION=$VERSION"
echo "  TASKS_CSV=$TASKS_CSV"
echo "  IMAGE_SIZES_CSV=$IMAGE_SIZES_CSV"
echo "  OUTPUT_BASE=$OUTPUT_BASE"
echo "  TIME_LIMIT=$TIME_LIMIT"
echo "  MEM_LIMIT=$MEM_LIMIT"
echo "  DRY_RUN=$DRY_RUN"
echo ""

if [[ ! -d "$CODE_DIR" ]]; then
  echo "ERROR: CODE_DIR not found: $CODE_DIR" >&2
  exit 1
fi

if [[ -n "$MODELS_CSV" ]]; then
  IFS=',' read -r -a SELECTED_MODELS <<< "$MODELS_CSV"
else
  # Auto-pick bottom 4 by mean accuracy on egma-math + vocab from existing v1 summaries.
  mapfile -t SELECTED_MODELS < <(
    CODE_DIR="$CODE_DIR" VERSION="$VERSION" python - <<'PY'
import csv
import os
from pathlib import Path

code_dir = Path(os.environ.get("CODE_DIR", "/projects/m000102/code/levante-bench"))
version = os.environ.get("VERSION", "v1")
candidates = [
  "internvl35:1B",
  "internvl35:2B",
  "qwen35:0.8B",
  "qwen35:2B",
  "smolvlm2:256M",
  "smolvlm2:500M",
  "tinyllava:2.4B",
  "tinyllava:3.1B",
]

def summary_path(model_size: str) -> Path | None:
  model, size = model_size.split(":", 1)
  slug = f"{model}-{size}"
  options = [
    code_dir / "results" / version / slug / "summary.csv",
    code_dir / "results" / "v1_additional_models" / slug / "summary.csv",
    code_dir / "results" / version / slug / "baseline" / "summary.csv",
  ]
  for p in options:
    if p.exists():
      return p
  return None

rows = []
for token in candidates:
  p = summary_path(token)
  if p is None:
    continue
  acc = {}
  with p.open() as f:
    for r in csv.DictReader(f):
      try:
        acc[r["task_id"]] = float(r["accuracy"])
      except Exception:
        pass
  if "egma-math" in acc and "vocab" in acc:
    mean_acc = (acc["egma-math"] + acc["vocab"]) / 2.0
    rows.append((mean_acc, token))

rows.sort(key=lambda x: x[0])
for _, token in rows[:4]:
  print(token)
PY
  )
fi

if [[ ${#SELECTED_MODELS[@]} -eq 0 ]]; then
  echo "ERROR: No models selected. Provide MODELS_CSV or ensure candidate summaries exist." >&2
  exit 1
fi

IFS=',' read -r -a IMAGE_SIZES <<< "$IMAGE_SIZES_CSV"
# sbatch --export uses commas to separate entries, so encode task lists.
TASKS_CSV_EXPORT="${TASKS_CSV//,/;}"

for target in "${SELECTED_MODELS[@]}"; do
  target_trimmed="$(echo "$target" | xargs)"
  if [[ "$target_trimmed" != *:* ]]; then
    echo "Skipping malformed target (expected model:size): $target_trimmed" >&2
    continue
  fi
  model="${target_trimmed%%:*}"
  size="${target_trimmed#*:}"
  for image_size in "${IMAGE_SIZES[@]}"; do
    image_size_trimmed="$(echo "$image_size" | xargs)"
    output_root="${OUTPUT_BASE}/img${image_size_trimmed}"
    job_name="lowres-${model}-${size}-${image_size_trimmed}"
    cmd=(
      sbatch
      --job-name "$job_name"
      --time "$TIME_LIMIT"
      --mem "$MEM_LIMIT"
      --export "ALL,PROJECT_ROOT=$PROJECT_ROOT,CODE_DIR=$CODE_DIR,CONDA_ENV_PATH=$CONDA_ENV_PATH,LEVANTE_PYTHON_BIN=$CONDA_ENV_PATH/bin/python,MODEL_NAME=$model,MODEL_SIZE=$size,IMAGE_SIZE=$image_size_trimmed,VERSION=$VERSION,DEVICE=$DEVICE,BATCH_SIZE=1,NUM_RUNS=1,TRUE_RANDOM_OPTION_ORDER=false,TASKS_CSV=$TASKS_CSV_EXPORT,OUTPUT_ROOT=$output_root,USE_JOB_OUTPUT_ROOT=0,HF_TOKEN,HUGGINGFACEHUB_API_TOKEN,HF_HOME,HF_HUB_CACHE,TRANSFORMERS_CACHE"
      "$SBATCH_SCRIPT"
    )
    echo "Model=$model size=$size image_size=$image_size_trimmed output_root=$output_root"
    if [[ "$DRY_RUN" == "1" ]]; then
      printf 'DRY_RUN:'
      printf ' %q' "${cmd[@]}"
      printf '\n'
    else
      "${cmd[@]}"
    fi
  done
done
