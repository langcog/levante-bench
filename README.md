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
3. **Assets (Python):** Run `python scripts/download_levante_assets.py [--version YYYY-MM-DD]` to download corpus and images from the public LEVANTE bucket into `data/assets/<version>/`.
4. **Evaluate:** Then:
   - `levante-bench list-tasks`
   - `levante-bench list-models`
   - `levante-bench check-gpu`  # verify local CUDA availability
   - `levante-bench run-eval --task trog --model clip_base [--version VERSION]`
   - `levante-bench run-benchmark --benchmark v1 --device auto`
   - `levante-bench run-benchmark --benchmark vocab --device auto`
   - `levante-bench run-workflow --workflow smol-vocab -- --help`
   - `levante-bench run-workflow --workflow benchmark-v1 -- --help`
   - `scripts/validate_all.sh`  # ruff + pytest + GPU check + benchmark smoke runs
   - `scripts/validate_all.sh --full-benchmarks`  # same checks + full v1 and vocab benchmarks
   - `scripts/validate_all.sh --with-r-validation`  # include R/Redivis package checks
   - `scripts/validate_r.sh --run-comparison-smoke --version 2026-03-24`  # optional R comparison smoke test
5. **Compare (R):** Run `levante-bench run-comparison --task trog --model clip_base` or run `Rscript comparison/compare_levante.R --task TASK --model MODEL` directly. Outputs accuracy (with IRT item difficulty) and D_KL (by ability bin) to `results/comparison/`.

## Recent updates (March 2026)

- **Framework integration:** SmolVLM benchmark scripts are now integrated under the `levante-bench` CLI (`run-workflow` and `run-benchmark`), including first-class `v1` and `vocab` benchmark presets.
- **GPU-aware execution:** Added `levante-bench check-gpu` and automatic device resolution (`--device auto`) with safe CUDA->CPU fallback.
- **Math prompt improvements:** `scripts/build_math_prompts.py` now defaults to shuffled options, supports numberline image attachment via `--numberline-graphics-dir`, and has configurable numberline instruction styles (`minimal`, `stepwise`).
- **Numberline multimodal evaluation:** `scripts/run_smolvlmv2_math_eval.py` now accepts image-backed prompt records (`image_paths`) so numberline items can be evaluated with actual graphics.
- **Vocab benchmark support:** Added image-grid vocab evaluation flow and integrated it into `levante-bench run-benchmark --benchmark vocab`.
- **Validation runner:** Added `scripts/validate_all.sh` to run lint/tests/GPU check plus smoke or full benchmark validations in one command.
- **Result history reporting:** Added `scripts/list_benchmark_results.py` to list benchmark and prompt-experiment outputs with metric deltas vs prior runs.

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

## Comparison approach

The benchmark compares model outputs to human behavioral data on two dimensions:

- **Accuracy vs item difficulty:** Model accuracy (correct/incorrect per item) is paired with IRT item difficulty parameters extracted from fitted Rasch models. A negative correlation indicates the model finds harder items harder, as children do.
- **Response distribution D_KL by ability bin:** Human response distributions are computed within subgroups of children binned by IRT ability (1-logit width bins on the logit scale). KL divergence between these human distributions and the model's softmax distribution quantifies alignment at each ability level.

See [comparison/README.md](comparison/README.md) for details.

## Docs

See [docs/README.md](docs/README.md) for data schema, releases, adding tasks/models, and secrets setup.
See [CHANGELOG.md](CHANGELOG.md) for ongoing project update history.

## Citing

Cite the LEVANTE manuscript and the DevBench (NeurIPS 2024) paper when using this benchmark.
