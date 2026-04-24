#!/usr/bin/env bash
set -euo pipefail

RUN_LABEL="${RUN_LABEL:-smolvlm2-256M-100x-checkpoint}"
BASE_DIR="results/v1/smolvlm2-256M/${RUN_LABEL}"
CHECKPOINT_DIR="${BASE_DIR}/.checkpoints"

TOTAL_CHUNKS="${TOTAL_CHUNKS:-10}"
RUNS_PER_CHUNK="${RUNS_PER_CHUNK:-10}"
TOTAL_RUNS=$((TOTAL_CHUNKS * RUNS_PER_CHUNK))

# Historical estimate from existing smolvlm2-256M v1 runs on this setup.
EST_SECONDS_PER_RUN="${EST_SECONDS_PER_RUN:-924}"

if [[ ! -d "${BASE_DIR}" ]]; then
  echo "No run directory yet: ${BASE_DIR}"
  exit 0
fi

if [[ -d "${CHECKPOINT_DIR}" ]]; then
  shopt -s nullglob
  chunk_markers=("${CHECKPOINT_DIR}"/chunk_*.done)
  shopt -u nullglob
  CHUNKS_DONE="${#chunk_markers[@]}"
else
  CHUNKS_DONE=0
fi

RUNS_DONE=$(find "${BASE_DIR}" -maxdepth 1 -type d -regex ".*/[0-9][0-9][0-9][0-9]" | wc -l | tr -d ' ')
if [[ "${RUNS_DONE}" -gt "${TOTAL_RUNS}" ]]; then
  RUNS_DONE="${TOTAL_RUNS}"
fi

COMPLETED_RUNS=$(find "${BASE_DIR}" -maxdepth 2 -type f -name "summary.csv" | wc -l | tr -d ' ')
if [[ "${COMPLETED_RUNS}" -gt "${TOTAL_RUNS}" ]]; then
  COMPLETED_RUNS="${TOTAL_RUNS}"
fi

RUNS_REMAINING=$((TOTAL_RUNS - RUNS_DONE))
COMPLETED_RUNS_REMAINING=$((TOTAL_RUNS - COMPLETED_RUNS))
EST_REMAINING_SECONDS=$((COMPLETED_RUNS_REMAINING * EST_SECONDS_PER_RUN))

if [[ "${COMPLETED_RUNS}" -gt 0 ]]; then
  START_EPOCH=$(find "${BASE_DIR}" -maxdepth 2 -type f -name "summary.csv" -printf '%T@\n' 2>/dev/null | sort -n | head -n 1 | cut -d. -f1)
  NOW_EPOCH=$(date +%s)
  ELAPSED_SECONDS=$((NOW_EPOCH - START_EPOCH))
  if [[ "${ELAPSED_SECONDS}" -gt 0 ]]; then
    ACTUAL_SECONDS_PER_RUN=$((ELAPSED_SECONDS / COMPLETED_RUNS))
  else
    ACTUAL_SECONDS_PER_RUN=0
  fi
else
  ACTUAL_SECONDS_PER_RUN=0
fi

if [[ "${ACTUAL_SECONDS_PER_RUN}" -gt 0 ]]; then
  ACTUAL_EST_REMAINING_SECONDS=$((COMPLETED_RUNS_REMAINING * ACTUAL_SECONDS_PER_RUN))
else
  ACTUAL_EST_REMAINING_SECONDS="${EST_REMAINING_SECONDS}"
fi

format_hms() {
  local total="$1"
  local h=$((total / 3600))
  local m=$(((total % 3600) / 60))
  local s=$((total % 60))
  printf "%02dh %02dm %02ds" "$h" "$m" "$s"
}

echo "run_label: ${RUN_LABEL}"
echo "base_dir: ${BASE_DIR}"
echo "chunks_done: ${CHUNKS_DONE}/${TOTAL_CHUNKS}"
echo "runs_started: ${RUNS_DONE}/${TOTAL_RUNS}"
echo "runs_completed: ${COMPLETED_RUNS}/${TOTAL_RUNS}"
echo "runs_remaining_to_start: ${RUNS_REMAINING}"
echo "runs_remaining_to_complete: ${COMPLETED_RUNS_REMAINING}"
echo "est_sec_per_run (historical): ${EST_SECONDS_PER_RUN}"
if [[ "${ACTUAL_SECONDS_PER_RUN}" -gt 0 ]]; then
  echo "est_sec_per_run (observed): ${ACTUAL_SECONDS_PER_RUN}"
else
  echo "est_sec_per_run (observed): n/a (no completed runs yet)"
fi
echo "eta_remaining (historical): $(format_hms "${EST_REMAINING_SECONDS}")"
echo "eta_remaining (observed):   $(format_hms "${ACTUAL_EST_REMAINING_SECONDS}")"

if pgrep -f "run_smolvlm2_256m_100x_checkpoint.sh" >/dev/null 2>&1; then
  echo "runner_process: active"
else
  echo "runner_process: not active"
fi
