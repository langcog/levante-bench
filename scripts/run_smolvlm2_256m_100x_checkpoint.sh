#!/usr/bin/env bash
set -euo pipefail

# Runs 100 true-random runs in 10-run chunks so progress is checkpointed.
# Safe to re-run: completed chunks are skipped using marker files.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"

RUN_LABEL="${RUN_LABEL:-smolvlm2-256M-100x-checkpoint}"
TOTAL_CHUNKS="${TOTAL_CHUNKS:-10}"
RUNS_PER_CHUNK="${RUNS_PER_CHUNK:-10}"
BATCH_SIZE="${BATCH_SIZE:-8}"
DEVICE="${DEVICE:-cuda}"
EXPERIMENT="${EXPERIMENT:-configs/experiments/smolvlm2_256m_v1.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python}"

MARKER_DIR="results/v1/smolvlm2-256M/${RUN_LABEL}/.checkpoints"
mkdir -p "$MARKER_DIR"

check_for_conflicting_runs() {
  # Set ALLOW_PARALLEL_RUNS=1 to bypass this safety check.
  if [[ "${ALLOW_PARALLEL_RUNS:-0}" == "1" ]]; then
    return 0
  fi

  local conflict_pids=()
  local pid
  while IFS= read -r pid; do
    [[ -z "$pid" ]] && continue
    [[ "$pid" == "$$" || "$pid" == "$BASHPID" || "$pid" == "$PPID" ]] && continue
    conflict_pids+=("$pid")
  done < <(pgrep -f "run_smolvlm2_256m_100x_checkpoint.sh|levante_bench\\.cli" || true)

  if (( ${#conflict_pids[@]} > 0 )); then
    echo "Refusing to start: another benchmark/checkpoint process is active."
    for pid in "${conflict_pids[@]}"; do
      local cmd
      cmd="$(ps -p "$pid" -o args= 2>/dev/null || true)"
      if [[ -n "$cmd" ]]; then
        echo "  pid=${pid} cmd=${cmd}"
      else
        echo "  pid=${pid}"
      fi
    done
    echo "Stop competing jobs first, or set ALLOW_PARALLEL_RUNS=1 to override."
    exit 2
  fi
}

check_for_conflicting_runs

echo "Starting checkpointed run:"
echo "  run_label=$RUN_LABEL"
echo "  total_chunks=$TOTAL_CHUNKS"
echo "  runs_per_chunk=$RUNS_PER_CHUNK"
echo "  batch_size=$BATCH_SIZE"
echo "  marker_dir=$MARKER_DIR"

for chunk in $(seq 1 "$TOTAL_CHUNKS"); do
  marker_file="$MARKER_DIR/chunk_$(printf "%02d" "$chunk").done"
  if [[ -f "$marker_file" ]]; then
    echo "Skipping chunk $chunk/$TOTAL_CHUNKS (already completed)"
    continue
  fi

  check_for_conflicting_runs
  echo "Running chunk $chunk/$TOTAL_CHUNKS ..."
  "${PYTHON_BIN}" -m levante_bench.cli \
    "experiment=$EXPERIMENT" \
    "device=$DEVICE" \
    "batch_size=$BATCH_SIZE" \
    "num_runs=$RUNS_PER_CHUNK" \
    "true_random_option_order=true" \
    "run_label=$RUN_LABEL" \
    "slurm_run_label=false"

  date -Iseconds > "$marker_file"
  echo "Completed chunk $chunk/$TOTAL_CHUNKS"
done

echo "All requested chunks complete."
