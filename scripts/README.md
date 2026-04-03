# Scripts Index

This directory contains data acquisition, benchmark pipelines, analysis utilities, and validation helpers.

## Core data and benchmark pipelines

- `download_levante_data.R`
  - **Purpose:** download Redivis trials + IRT assets; build response summaries
  - **Inputs:** `--version`, `--irt-dataset`, `--irt-table`
  - **Outputs:** `data/responses/<version>/...` (trials, IRT files, responses_by_ability)
  - **Example:** `Rscript scripts/download_levante_data.R --version 2026-03-24`
- `download_levante_assets.py`
  - **Purpose:** download corpus + visual assets from the LEVANTE bucket
  - **Inputs:** `--version` (or auto-detect latest bucket version prefix), `--workers`
  - **Outputs:** `data/assets/<version>/...`
  - **Example:** `python scripts/download_levante_assets.py --version hackathon --workers 24`
- `migrate_assets_to_versioned_bucket.py`
  - **Purpose:** copy corpus/visual/manifest from source bucket into versioned destination prefix
  - **Inputs:** `--version`, `--dest-bucket`, optional `--dest-root-prefix` (default `corpus_data`), optional `--task`, `--dry-run`
  - **Outputs:** `gs://<dest-bucket>/<dest-root-prefix>/<version>/...`
  - **Example:** `python scripts/migrate_assets_to_versioned_bucket.py --version 2026-03-24 --dest-bucket gs://levante-bench --dest-root-prefix corpus_data --dry-run`
- `download_results_from_drive.py`
  - **Purpose:** download shared benchmark result folders from Google Drive
  - **Inputs:** `--folder-url`, `--output-dir`
  - **Outputs:** synced folders under `results/`
  - **Example:** `python scripts/download_results_from_drive.py --output-dir results`
- `run_benchmark_v1.py`
  - **Purpose:** run the v1 benchmark bundle (math + ToM robustness)
  - **Inputs:** `--data-version`, `--model-id`, `--device`, `--max-items-*`
  - **Outputs:** `results/benchmark/v1/<timestamp>/...`
  - **Example:** `python scripts/run_benchmark_v1.py --device auto --max-items-math 2 --max-items-tom 2`
- `run_experiment.sh`
  - **Purpose:** eval-style wrapper for experiment YAML runs
  - **Inputs:** experiment YAML path + OmegaConf dotlist overrides
  - **Outputs:** task-dependent experiment artifacts under configured output dirs
  - **Example:** `bash run_experiment.sh configs/experiment.yaml tasks=[vocab] max_items_vocab=8`
- `run_smolvlmv2_vocab_eval.py`
  - **Purpose:** run vocab image-grid evaluation
  - **Inputs:** `--corpus-csv`, `--visual-dir`, `--model-id`
  - **Outputs:** predictions JSONL + summary JSON
  - **Example:** `python scripts/run_smolvlmv2_vocab_eval.py --help`

## Math pipeline

- `build_math_prompts.py`
  - **Purpose:** build EGMA math prompt JSONL for SmolVLM
  - **Inputs:** `--corpus-csv`, `--numberline-graphics-dir`, `--numberline-instruction-style`
  - **Outputs:** prompt JSONL
  - **Example:** `python scripts/build_math_prompts.py --help`
- `run_smolvlmv2_math_eval.py`
  - **Purpose:** run SmolVLM on math prompts (supports `image_paths`)
  - **Inputs:** `--input-jsonl`, `--model-id`, `--device`
  - **Outputs:** predictions JSONL + summary JSON
  - **Example:** `python scripts/run_smolvlmv2_math_eval.py --help`
- `analyze_math_type_accuracy.py`
  - **Purpose:** compute by-type accuracy/chance and generate chart
  - **Inputs:** prompts JSONL + predictions JSONL
  - **Outputs:** by-type CSV + PNG
  - **Example:** `python scripts/analyze_math_type_accuracy.py --help`

## ToM pipeline and utilities

- `run_smolvlmv2_tom_eval.py`
  - **Purpose:** stateful ToM evaluator (memory modes, prompt styles)
  - **Inputs:** corpus CSV + eval flags
  - **Outputs:** predictions JSONL + summary JSON
  - **Example:** `python scripts/run_smolvlmv2_tom_eval.py --help`
- `run_tom_robustness.py`
  - **Purpose:** orchestrate multi-seed ToM variant runs
  - **Inputs:** `--variants`, `--seeds`
  - **Outputs:** per-run and summary CSV/JSONL
  - **Example:** `python scripts/run_tom_robustness.py --help`
- `run_tom_symbolic_engine.py`
  - **Purpose:** symbolic ToM baseline
  - **Inputs:** corpus CSV
  - **Outputs:** predictions JSONL + summary JSON
  - **Example:** `python scripts/run_tom_symbolic_engine.py --help`
- `run_tom_modal_eval.py`
  - **Purpose:** evaluate ToM across visual input modes
  - **Inputs:** corpus + mode flags
  - **Outputs:** predictions and summary artifacts
  - **Example:** `python scripts/run_tom_modal_eval.py --help`
- `generate_tom_visual_descriptions.py`
  - **Purpose:** generate visual descriptions for ToM screenshots
  - **Inputs:** screenshot directory + model/flags
  - **Outputs:** description CSV/JSONL
  - **Example:** `python scripts/generate_tom_visual_descriptions.py --help`
- `postprocess_tom_visual_context.py`
  - **Purpose:** clean and normalize generated visual context
  - **Inputs:** generated description artifacts
  - **Outputs:** cleaned artifacts
  - **Example:** `python scripts/postprocess_tom_visual_context.py --help`
- `clean_tom_screenshot_csv.py`
  - **Purpose:** clean screenshot metadata CSVs
  - **Inputs:** screenshot CSV
  - **Outputs:** cleaned CSV
  - **Example:** `python scripts/clean_tom_screenshot_csv.py --help`
- `correlate_tom_screenshots.py`
  - **Purpose:** match screenshot files with ToM item prompts
  - **Inputs:** corpus + screenshot metadata
  - **Outputs:** correlation/report artifacts
  - **Example:** `python scripts/correlate_tom_screenshots.py --help`

## Vocab graphics utility

- `build_vocab_quadrant_graphics.py`
  - **Purpose:** build tracked 2x2 vocab graphics bundle
  - **Inputs:** vocab corpus + visual directory
  - **Outputs:** `local_data/vocab_graphics/images` + manifest + summary
  - **Example:** `python scripts/build_vocab_quadrant_graphics.py --help`

## Validation and reporting helpers

- `validate_all.sh`
  - **Purpose:** one-command validation for lint/tests/GPU/benchmarks
  - **Inputs:** optional flags `--full-benchmarks`, `--with-r-*`
  - **Outputs:** pass/fail plus benchmark artifacts
  - **Example:** `scripts/validate_all.sh --with-r-validation`
- `validate_r.sh`
  - **Purpose:** R/Redivis validation (package checks + optional comparison smoke)
  - **Inputs:** `--check-packages-only` or `--run-comparison-smoke`
  - **Outputs:** pass/fail + optional comparison CSV outputs
  - **Example:** `scripts/validate_r.sh --run-comparison-smoke --version 2026-03-24`
- `list_benchmark_results.py`
  - **Purpose:** show benchmark/prompt experiment history with deltas
  - **Inputs:** `--results-root`, `--prompts-root`, `--limit`
  - **Outputs:** console summary tables
  - **Example:** `python scripts/list_benchmark_results.py --limit 20`
- `build_model_comparison_report.py`
  - **Purpose:** collect all `summary.csv` files and export detailed model comparison JSON
  - **Inputs:** `--results-root`, `--output-json`
  - **Outputs:** JSON with per-run metrics and per-model aggregated stats
  - **Example:** `python scripts/build_model_comparison_report.py --results-root results --output-json results/model-comparison-report.json`
- `plot_model_comparison_lines.py`
  - **Purpose:** plot line chart of task accuracies by model from the comparison JSON
  - **Inputs:** `--report-json`, `--output`, `--min-tasks`
  - **Outputs:** PNG line chart (tasks on x-axis, accuracy on y-axis)
  - **Example:** `python scripts/plot_model_comparison_lines.py --report-json results/model-comparison-report.json --output results/model-comparison-line-chart.png`
- `plot_aquila_stages.py`
  - **Purpose:** compare Aquila intermediate stages (`stage2a/b/c`, `stage3`) and final production performance by task
  - **Inputs:** `--results-root`, `--output`
  - **Outputs:** PNG with task-wise stage lines + mean-accuracy bars
  - **Example:** `python scripts/analysis/plot_aquila_stages.py --results-root scripts/results/aquila-checkpoints/2026-03-29`

## Notes

- Most scripts support `--help`; prefer checking each script's CLI for the latest flags.
- For end-to-end user workflows, start in the top-level `README.md` and `comparison/README.md`.
