#!/usr/bin/env bash
set -euo pipefail

# Wrapper for publishing benchmark results:
# Google Drive -> local staging -> comparison report -> GCS bucket
#
# Usage:
#   scripts/publish_results.sh --remaining-ok
#   scripts/publish_results.sh --skip-download --keep-staging

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python"
fi

"${PYTHON_BIN}" "${ROOT_DIR}/scripts/analysis/publish_results_from_drive_to_bucket.py" "$@"
