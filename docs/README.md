# LEVANTE VLM Benchmark – Documentation

This directory contains user-facing and developer documentation for the LEVANTE VLM Benchmark.

- **[data_schema.md](data_schema.md)** – Canonical schema for trials, human responses, and item_uid → corpus → assets mapping.
- **[releases.md](releases.md)** – How to obtain LEVANTE trials data (Redivis) and run the R download script; versioning.
- **[adding_tasks.md](adding_tasks.md)** – How to add a LEVANTE task to the benchmark.
- **[adding_models.md](adding_models.md)** – How to add a VLM to the benchmark.

## Quick start

1. Install R and the `redivis` package; configure auth per [releases.md](releases.md).
2. Run `scripts/download_levante_assets.py` (optional `--version YYYY-MM-DD`) to download corpus and images from the public LEVANTE assets bucket.
3. Run `scripts/download_levante_data.R` to fetch trials from Redivis into `data/responses/<version>/`.
4. Validate environment and GPU:
   - `levante-bench list-tasks`
   - `levante-bench list-models`
   - `levante-bench check-gpu`
5. Run evaluation and benchmarks:
   - `levante-bench run-eval --task <task> --model <model> [--version <version>]`
   - `levante-bench run-benchmark --benchmark v1 --device auto`
   - `levante-bench run-benchmark --benchmark vocab --device auto`
6. Run comparison (R):
   - `levante-bench run-comparison --task <task> --model <model> --version <version>`
   - or run `comparison/compare_levante.R` directly
7. Use validation and result-history helpers:
   - `scripts/validate_all.sh` (smoke validations)
   - `scripts/validate_all.sh --full-benchmarks`
   - `scripts/validate_all.sh --with-r-validation` (adds R package checks)
   - `scripts/validate_r.sh --run-comparison-smoke --version <version>` (R comparison smoke test)
   - `python3 scripts/list_benchmark_results.py --limit 20`

## Citing

When using this benchmark, cite the LEVANTE manuscript and the DevBench paper (see main README).
