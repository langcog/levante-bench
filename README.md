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
   - `levante-bench run-workflow --workflow smol-vocab -- --help`
   - `levante-bench run-workflow --workflow benchmark-v1 -- --help`
5. **Compare (R):** Run `levante-bench run-comparison --task trog --model clip_base` or run `Rscript comparison/compare_levante.R --task TASK --model MODEL` directly. Outputs accuracy (with IRT item difficulty) and D_KL (by ability bin) to `results/comparison/`.

## Comparison approach

The benchmark compares model outputs to human behavioral data on two dimensions:

- **Accuracy vs item difficulty:** Model accuracy (correct/incorrect per item) is paired with IRT item difficulty parameters extracted from fitted Rasch models. A negative correlation indicates the model finds harder items harder, as children do.
- **Response distribution D_KL by ability bin:** Human response distributions are computed within subgroups of children binned by IRT ability (1-logit width bins on the logit scale). KL divergence between these human distributions and the model's softmax distribution quantifies alignment at each ability level.

See [comparison/README.md](comparison/README.md) for details.

## Docs

See [docs/README.md](docs/README.md) for data schema, releases, adding tasks/models, and secrets setup.

## Citing

Cite the LEVANTE manuscript and the DevBench (NeurIPS 2024) paper when using this benchmark.
