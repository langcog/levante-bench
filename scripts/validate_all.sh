#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run project validations (lint, tests, benchmark smoke/full runs).

Usage:
  scripts/validate_all.sh [options]

Options:
  --full-benchmarks         Run full v1 and vocab benchmarks.
  --skip-benchmarks         Only run lint/tests/gpu check.
  --with-r-validation       Run R package validation via scripts/validate_r.sh.
  --with-r-smoke            Run R package validation + comparison smoke test.
  --device <auto|cpu|cuda>  Device for benchmark commands (default: auto).
  --data-version <version>  Data/assets version (default: 2026-03-24).
  --model-id <hf-model-id>  Model id override for benchmark runs.
  -h, --help                Show this help.

Default behavior:
  - Runs ruff + pytest + levante-bench check-gpu
  - Runs benchmark smoke tests:
      v1:    --max-items-math 2 --max-items-tom 2
      vocab: --max-items-vocab 8
  - R validation is opt-in:
      scripts/validate_all.sh --with-r-validation
      scripts/validate_all.sh --with-r-smoke
EOF
}

FULL_BENCHMARKS=0
SKIP_BENCHMARKS=0
DEVICE="auto"
DATA_VERSION="2026-03-24"
MODEL_ID=""
WITH_R_VALIDATION=0
WITH_R_SMOKE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --full-benchmarks)
      FULL_BENCHMARKS=1
      shift
      ;;
    --skip-benchmarks)
      SKIP_BENCHMARKS=1
      shift
      ;;
    --device)
      DEVICE="${2:-}"
      shift 2
      ;;
    --with-r-validation)
      WITH_R_VALIDATION=1
      shift
      ;;
    --with-r-smoke)
      WITH_R_VALIDATION=1
      WITH_R_SMOKE=1
      shift
      ;;
    --data-version)
      DATA_VERSION="${2:-}"
      shift 2
      ;;
    --model-id)
      MODEL_ID="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ "$FULL_BENCHMARKS" -eq 1 && "$SKIP_BENCHMARKS" -eq 1 ]]; then
  echo "Cannot combine --full-benchmarks with --skip-benchmarks." >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

echo "==> Ruff"
ruff check .

echo "==> Pytest"
set +e
pytest
PYTEST_STATUS=$?
set -e
if [[ "$PYTEST_STATUS" -eq 5 ]]; then
  echo "Pytest collected no tests; continuing."
elif [[ "$PYTEST_STATUS" -ne 0 ]]; then
  echo "Pytest failed with exit code $PYTEST_STATUS." >&2
  exit "$PYTEST_STATUS"
fi

echo "==> GPU check"
levante-bench check-gpu

if [[ "$WITH_R_VALIDATION" -eq 1 ]]; then
  if [[ "$WITH_R_SMOKE" -eq 1 ]]; then
    echo "==> R validation (packages + comparison smoke)"
    scripts/validate_r.sh --run-comparison-smoke --version "$DATA_VERSION"
  else
    echo "==> R validation (package checks)"
    scripts/validate_r.sh --check-packages-only
  fi
fi

if [[ "$SKIP_BENCHMARKS" -eq 1 ]]; then
  echo "==> Skipping benchmarks (--skip-benchmarks)"
  exit 0
fi

COMMON_BENCH_ARGS=(
  --device "$DEVICE"
  --data-version "$DATA_VERSION"
)
if [[ -n "$MODEL_ID" ]]; then
  COMMON_BENCH_ARGS+=(--model-id "$MODEL_ID")
fi

if [[ "$FULL_BENCHMARKS" -eq 1 ]]; then
  echo "==> Full benchmark v1"
  levante-bench run-benchmark --benchmark v1 "${COMMON_BENCH_ARGS[@]}"

  echo "==> Full benchmark vocab"
  levante-bench run-benchmark --benchmark vocab "${COMMON_BENCH_ARGS[@]}"
else
  echo "==> Smoke benchmark v1 (2 math, 2 tom)"
  levante-bench run-benchmark \
    --benchmark v1 \
    "${COMMON_BENCH_ARGS[@]}" \
    --max-items-math 2 \
    --max-items-tom 2

  echo "==> Smoke benchmark vocab (8 items)"
  levante-bench run-benchmark \
    --benchmark vocab \
    "${COMMON_BENCH_ARGS[@]}" \
    --max-items-vocab 8
fi

echo "==> All validations completed."
