# LEVANTE VLM Benchmark

Extensible Python-first benchmark comparing VLMs (CLIP-style and LLaVA-style) to children's behavioral data from LEVANTE. R is used for downloading trials (Redivis), fetching IRT models, and for statistical comparison; Python is used for config, data loaders, model adapters, and the evaluation runner.

## Install (pinned)

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .            # install this package
# Optional: pip install -r requirements-transformers.txt   # for CLIP
```

Use **Python 3.10–3.13**. On **3.13**, `requirements.txt` pins `torch>=2.6` and newer numpy/pandas so pip can install wheels (older torch/pandas often have no `cp313` builds).

Pinned deps: [requirements.txt](requirements.txt). Dev: [requirements-dev.txt](requirements-dev.txt).

## Quick start

1. **IRT model mapping:** Edit `src/levante_bench/config/irt_model_mapping.csv` to map each task to its IRT model `.rds` file in the Redivis model registry (e.g. `trog,trog/multigroup_site/overlap_items/trog_rasch_f1_scalar.rds`).
2. **Data (R):** Install R and the `redivis` package; run `Rscript scripts/download_levante_data.R` to fetch trials and IRT models into `data/responses/<version>/`.
3. **Assets (Python):** Run `python scripts/download_levante_assets.py [--version VERSION]` to download corpus and images from a versioned bucket prefix into `data/assets/<version>/`. If `--version` is omitted, the script uses `LEVANTE_DATA_VERSION` or auto-detects a bucket default (latest date-style prefix; otherwise prefers `v1` when present; otherwise the sole non-date prefix). Visual asset downloads are parallelized (`--workers`, default `8`).
4. **Evaluate:** Then:
   - `levante-bench list-tasks`
   - `levante-bench list-models`
   - `levante-bench check-gpu`  # verify local CUDA availability
   - `levante-bench run-eval --task trog --model clip_base [--version VERSION] [--prompt-language de]`
   - `levante-bench run-eval --task trog --model gemma3 --version v1 --device cuda`  # Gemma 3 smoke run
   - `levante-bench run-benchmark --benchmark v1 --device auto`
   - `levante-bench run-benchmark --benchmark vocab --device auto`
   - `levante-bench run-workflow --workflow smol-vocab -- --help`
   - `levante-bench run-workflow --workflow benchmark-v1 -- --help`
   - `scripts/validate_all.sh`  # ruff + pytest + GPU check + benchmark smoke runs
   - `scripts/validate_all.sh --full-benchmarks`  # same checks + full v1 and vocab benchmarks
   - `scripts/validate_all.sh --with-r-validation`  # include R/Redivis package checks
   - `scripts/validate_r.sh --run-comparison-smoke --version 2026-03-24`  # optional R comparison smoke test
5. **Compare (R):** Run `levante-bench run-comparison --task trog --model clip_base` or run `Rscript comparison/compare_levante.R --task TASK --model MODEL` directly. Outputs accuracy (with IRT item difficulty) and D_KL (by ability bin) to `results/comparison/`.

For multilingual runs (`--prompt-language` not English), per-model result folders include a 2-letter language suffix. Result layout is deterministic: `results/<version>/<model>-<size>[-<lang>]/` and each model folder includes a `metadata.json`. Example: `results/<version>/qwen35-2B-de/`.

## Web results dashboard (Vercel)

- A lightweight dashboard is available at `/` when deployed with Vercel.
- API endpoint `/api/results-report` supports:
  1. live bucket aggregation (`RESULTS_SOURCE_MODE=bucket_compute`, default),
  2. remote prebuilt JSON (`RESULTS_SOURCE_MODE=remote` + `RESULTS_REPORT_URL`), or
  3. local JSON fallback (`RESULTS_SOURCE_MODE=local`).
- API endpoint `/api/human-age-accuracy` serves aggregated child accuracy lines from
  `results/human-accuracy-by-age-lines.csv` (bucket or local mode).
- The dashboard supports model-vs-children comparison with:
  - tabbed series selection (`Models` / `Children`),
  - shared task and language filters across both tabs,
  - child series grouped by age bin and filtered by the same language selector used for models.
- Generate or refresh child comparison data from Redivis trials with:

```bash
python scripts/analysis/plot_human_accuracy_by_age_lines.py \
  --trials-csv data/responses/v1/trials.csv \
  --output-csv results/human-accuracy-by-age-lines.csv \
  --output results/human-accuracy-by-age-lines.png
```

- Build report data before local preview:

```bash
python scripts/analysis/build_model_comparison_report.py --results-root results --output-json results/model-comparison-report.json
```

## Experiment YAML mode (eval-style)

You can run experiment configs directly using the eval-style command structure:

```bash
# Direct
python -m levante_bench.cli experiment=configs/experiment.yaml

# Wrapper (same behavior)
bash run_experiment.sh configs/experiment.yaml
```

Use dotlist-style overrides to change task subsets and smoke caps:

```bash
# Vocab smoke
python -m levante_bench.cli experiment=configs/experiment.yaml tasks=[vocab] max_items_vocab=8 device=cpu

# Math smoke
python -m levante_bench.cli experiment=configs/experiment.yaml tasks=[egma-math] max_items_math=2 device=cpu

# ToM smoke
python -m levante_bench.cli experiment=configs/experiment.yaml tasks=[theory-of-mind] max_items_tom=2 device=cpu
```

## Recent updates (March 2026)

- **Framework integration:** SmolVLM benchmark scripts are now integrated under the `levante-bench` CLI (`run-workflow` and `run-benchmark`), including first-class `v1` and `vocab` benchmark presets.
- **GPU-aware execution:** Added `levante-bench check-gpu` and automatic device resolution (`--device auto`) with safe CUDA->CPU fallback.
- **Math prompt improvements:** `scripts/build_math_prompts.py` now defaults to shuffled options, supports numberline image attachment via `--numberline-graphics-dir`, and has configurable numberline instruction styles (`minimal`, `stepwise`).
- **Numberline multimodal evaluation:** `scripts/run_smolvlmv2_math_eval.py` now accepts image-backed prompt records (`image_paths`) so numberline items can be evaluated with actual graphics.
- **Vocab benchmark support:** Added image-grid vocab evaluation flow and integrated it into `levante-bench run-benchmark --benchmark vocab`.
- **Validation runner:** Added `scripts/validate_all.sh` to run lint/tests/GPU check plus smoke or full benchmark validations in one command.
- **Result history reporting:** Added `scripts/list_benchmark_results.py` to list benchmark and prompt-experiment outputs with metric deltas vs prior runs.

## Result visualization

```bash
# Heatmap of models x tasks accuracy
python scripts/plot_results.py

# Specific version
python scripts/plot_results.py --version 2026-03-24

# Text table only
python scripts/plot_results.py --no-plot
```

## Result inspection

Use these commands to verify what ran and compare with prior runs:

```bash
# Show benchmark + prompt experiment history with deltas.
python3 scripts/list_benchmark_results.py --limit 20

# Run full validation pipeline (lint/tests/gpu + smoke benchmarks).
scripts/validate_all.sh

# Run full benchmarks instead of smoke.
scripts/validate_all.sh --full-benchmarks

# Include R package validation in the full validation pass.
scripts/validate_all.sh --with-r-validation
```

## Testing strategy

The test suite is split into fast unit/property tests (default) and opt-in
integration tests (model loading / dataset end-to-end checks).

- **Default pytest run:** `python -m pytest`
  - Runs unit tests for parsing, scoring, aggregation, runner utils, cache
    behavior, and API retry logic.
  - Includes property-based fuzz tests (Hypothesis) for parser robustness.
- **Integration tests (opt-in):**
  - `tests/test_model_inference.py`
  - `tests/test_task_datasets.py`
  - These are intentionally gated behind `LEVANTE_RUN_INTEGRATION=1` so
    default CI/local runs stay deterministic and fast.
  - Run with:
    - `LEVANTE_RUN_INTEGRATION=1 python -m pytest`

Current parser-focused coverage includes:

- `parse_answer` / `parse_answer_v2` branch coverage (JSON, embedded JSON,
  phrase patterns, exact/prefix forms, ambiguous-prose rejection).
- `parse_numeric_answer` / `parse_numeric_v2` branch coverage (strict JSON,
  embedded JSON, slider mode constraints, fallback behavior).
- `<imageN>` interleaving behavior across model adapters.
- `evaluate_trial` correctness for label, numeric, and slider formats.
- postprocessing accuracy aggregation and ordering checks.
- cache round-trip and cache-hit behavior in `run_eval`.
- GPT-5.3 retry logic (`5xx` retry and token-cap doubling path).

## Parser v2 model

Evaluation now uses a canonical parse layer with provenance so correctness is
decided in normalized answer space, not output surface format.

### Canonicalization

- **Label tasks:** normalize to `predicted_label` in `option_labels`.
- **Numeric/slider tasks:** normalize to `predicted_value` (float), then compare
  to `target_value` using `slider_tolerance`.
- **Slider tasks:** normalize slider position, clamp to `[0,1]`, then map back to
  task scale via `slider_min`/`slider_max`.

### Parser v2 outputs

`ParseResult` (in `src/levante_bench/models/base.py`) returns:

- `value` (canonical parsed value/label or `None`)
- `reason` (extracted reason or source text)
- `parse_method` (which rule matched)
- `parse_confidence` (`high` / `medium` / `low` / `none`)
- `raw_candidate` (raw extracted token)

`evaluate_trial()` now uses:

- `parse_answer_result(...)` for label tasks
- `parse_numeric_result(...)` for numeric/slider tasks

Backward-compatible APIs (`parse_answer`, `parse_numeric_answer`) are kept for
existing callers, but benchmark scoring uses parser-v2 paths.

### CSV provenance fields

Per-task CSV outputs now include parser provenance columns:

- `parse_method`
- `parse_confidence`
- `parse_raw_candidate`

This supports score audits (for example, reviewing accuracy by parse method or
identifying low-confidence parses).

## Comparison approach

The benchmark compares model outputs to human behavioral data on two dimensions:

- **Accuracy vs item difficulty:** Model accuracy (correct/incorrect per item) is paired with IRT item difficulty parameters extracted from fitted Rasch models. A negative correlation indicates the model finds harder items harder, as children do.
- **Response distribution D_KL by ability bin:** Human response distributions are computed within subgroups of children binned by IRT ability (1-logit width bins on the logit scale). KL divergence between these human distributions and the model's softmax distribution quantifies alignment at each ability level.

See [comparison/README.md](comparison/README.md) for details.

## Docs

See [docs/README.md](docs/README.md) for data schema, releases, adding tasks/models, and secrets setup.
See [docs/aquila-intermediate-runbook.md](docs/aquila-intermediate-runbook.md) for Aquila intermediate checkpoint integration and dual-environment setup.
See [docs/environment-split.md](docs/environment-split.md) for benchmark vs Aquila virtualenv activation and usage.
See [scripts/README.md](scripts/README.md) for a script-by-script command index.
See [CHANGELOG.md](CHANGELOG.md) for ongoing project update history.

### Bucket migration workflow

To migrate from the legacy `-prod` flat layout into a versioned bucket layout:

```bash
# Preview copy operations
python scripts/migrate_assets_to_versioned_bucket.py \
  --version 2026-03-24 \
  --dest-bucket gs://levante-bench \
  --dry-run

# Execute migration
python scripts/migrate_assets_to_versioned_bucket.py \
  --version 2026-03-24 \
  --dest-bucket gs://levante-bench
```

Then point downloads at the destination bucket:

```bash
export LEVANTE_ASSETS_BUCKET_URL=https://storage.googleapis.com/levante-bench/corpus_data
python scripts/download_levante_assets.py --version hackathon --workers 8
```

`corpus_data` is the default destination prefix in the migration script, and can
be changed with `--dest-root-prefix`.

Versioned snapshots also include:

- `manifest.csv`
- `translations/item-bank-translations.csv`

When running benchmark/eval commands with `--version current`, local version
resolution now picks the most recently modified folder under `data/assets/`
(not only `YYYY-MM-DD` names), so labels like `hackathon` are supported.

## Citing

Cite the LEVANTE manuscript and the DevBench (NeurIPS 2024) paper when using this benchmark.
