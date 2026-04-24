#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATUS_SCRIPT="${ROOT_DIR}/scripts/check_smolvlm2_100x_status.sh"

INTERVAL_SECONDS="${INTERVAL_SECONDS:-30}"

if [[ ! -x "${STATUS_SCRIPT}" ]]; then
  echo "Status script not found or not executable: ${STATUS_SCRIPT}"
  exit 1
fi

run_once() {
  "${STATUS_SCRIPT}"
}

if [[ "${1:-}" == "--once" ]]; then
  run_once
  exit 0
fi

while true; do
  clear
  echo "SmolVLM2 100x status (refresh every ${INTERVAL_SECONDS}s)"
  echo "Timestamp: $(date -Iseconds)"
  echo
  run_once
  sleep "${INTERVAL_SECONDS}"
done
