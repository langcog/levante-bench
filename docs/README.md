# LEVANTE VLM Benchmark – Documentation

This directory contains user-facing and developer documentation for the LEVANTE VLM Benchmark.

- **[data_schema.md](data_schema.md)** – Canonical schema for trials, human responses, and item_uid → corpus → assets mapping.
- **[releases.md](releases.md)** – How to obtain LEVANTE trials data (Redivis) and run the R download script; versioning.
- **[adding_tasks.md](adding_tasks.md)** – How to add a LEVANTE task to the benchmark.
- **[adding_models.md](adding_models.md)** – How to add a VLM to the benchmark.
- **[runtime_exports.md](runtime_exports.md)** – Public runtime API for external repos (`load_model`, `run_trials`, `run-trials-jsonl`).

## Quick start

1. Install R and the `redivis` package; configure auth per [releases.md](releases.md).
2. Run `scripts/data_prep/download_levante_assets.py` (optional `--version YYYY-MM-DD`) to download corpus and images from the public LEVANTE assets bucket.
3. Run `scripts/download_levante_data.R` to fetch trials from Redivis into `data/responses/<version>/`.
4. Validate environment and GPU:
   - `levante-bench list-tasks`
   - `levante-bench list-models`
   - `levante-bench check-gpu`
5. Run evaluation and benchmarks:
   - `levante-bench run-eval --task <task> --model <model> [--version <version>]`
   - `levante-bench run-eval --task <task> --model <model> --true-random-option-order --num-runs 3`
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

## Experiment YAML mode (eval-style)

You can run YAML-defined experiments directly through the CLI:

- `python -m levante_bench.cli experiment=configs/experiments/experiment.yaml`
- `bash run_experiment.sh configs/experiments/experiment.yaml`

You can also use OmegaConf dotlist overrides for task subsets and smoke caps:

- `python -m levante_bench.cli experiment=configs/experiments/experiment.yaml tasks=[vocab] max_items_vocab=8 device=cpu`
- `python -m levante_bench.cli experiment=configs/experiments/experiment.yaml tasks=[egma-math] max_items_math=2 device=cpu`
- `python -m levante_bench.cli experiment=configs/experiments/experiment.yaml tasks=[theory-of-mind] max_items_tom=2 device=cpu`

When `true_random_option_order` is enabled (CLI flag or experiment YAML), run outputs are written under numbered subfolders (`0001`, `0002`, ...) and per-item option ordering seeds are recorded in `cache/responses.json`. On Slurm/`sbatch`, run folders default to job-based labels (for example `job12345-task7`) to prevent cross-job collisions.

## Runner migration checklist

Use this checklist when moving legacy benchmark scripts onto the registry-based
`levante_bench.evaluation.runner` path.

1. **Lock parity target artifacts**
   - Math: predictions + summary + by-type outputs
   - ToM: predictions + summary and trial-type breakdown
   - Vocab: predictions + summary and quadrant stats
2. **Implement task adapter hooks**
   - Add per-task prepare/postprocess hooks in runner for script-only logic
3. **Run parity gates**
   - Row counts, parse rates, and metrics must match agreed tolerances
   - Required output files must exist with compatible schemas
4. **Switch command routing incrementally**
   - Move one task at a time from legacy scripts to runner-backed flow
5. **Deprecate legacy paths only after stable overlap**
   - Keep `--legacy` or equivalent during transition window

## Citing

When using this benchmark, cite the LEVANTE manuscript and the DevBench paper (see main README).
