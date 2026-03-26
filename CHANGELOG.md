# Changelog

All notable updates to this repository should be documented in this file.

## 2026-03-26

### Added
- R/Redivis validation helper: `scripts/validate_r.sh`
  - Supports package-only validation and optional comparison smoke test
  - Checks expected prerequisites and comparison output artifacts for smoke runs
- Script index: `scripts/README.md`
  - Provides purpose/inputs/outputs/example usage for pipeline and utility scripts

### Changed
- Unified validation runner (`scripts/validate_all.sh`) now supports:
  - `--with-r-validation` for R package checks
  - `--with-r-smoke` for R package checks plus comparison smoke test
- Documentation updates in `README.md`, `docs/README.md`, and `comparison/README.md`
  - Added R validation commands and clearer operational guidance

## 2026-03-25

### Added
- Unified validation runner: `scripts/validate_all.sh`
  - Runs `ruff`, `pytest`, `levante-bench check-gpu`
  - Supports smoke benchmark validation (default) and full benchmark mode
- Benchmark/result history reporter: `scripts/list_benchmark_results.py`
  - Summarizes `results/benchmark` (`v1`, `vocab`) and `results/prompts` runs
  - Shows metric deltas versus prior runs
- Numberline multimodal support in math pipeline:
  - `scripts/build_math_prompts.py` can attach numberline images (`--numberline-graphics-dir`)
  - `scripts/run_smolvlmv2_math_eval.py` now consumes image-backed prompt records (`image_paths`)
- Numberline prompt instruction modes:
  - `--numberline-instruction-style {minimal,stepwise}`
  - Endpoint-reading guidance improved for non-0..10 scales

### Changed
- Math prompt option shuffling now defaults to enabled.
- `README.md` expanded with:
  - Recent updates summary
  - Result inspection commands
  - Validation pipeline usage
- Lint cleanup and script hygiene fixes:
  - Resolved Ruff `E741` ambiguous variable names in `scripts/run_benchmark_v1.py`
  - Removed unused imports/variables across benchmark/runtime modules
- Validation script behavior:
  - Treats `pytest` exit code `5` (no tests collected) as non-fatal

### Fixed
- Environment compatibility issue during validation:
  - Aligned `torch`/`torchvision` versions to avoid runtime operator import errors.
