# Historic VLM Runbook (2022-2023 Candidates)

This note captures how to run older "frontier-era" VLMs in this repo.

## TL;DR

- **What works today with minimal code changes:** API-hosted models via `hf_hosted`.
- **What is not wired today:** dedicated local adapters for LLaVA-1.5 / CogVLM / OpenFlamingo.
- **If you want local runs on a larger GPU machine:** we need adapter/config work first, then standard `run-eval`.

## Current support in this repo

- **API-hosted path (already implemented):**
  - Adapter: `src/levante_bench/models/hf_hosted.py`
  - Existing hosted configs: `configs/models/*_hf.yaml`
  - Auth: `HF_TOKEN` (or `HUGGINGFACEHUB_API_TOKEN`)

- **Local model adapters already implemented (non-historic targets):**
  - `smolvlm2`, `qwen35`, `internvl35`, `tinyllava`, `aquila_vl`, `clip_base`, etc.

## Recommendation for 2022-2023 additions

1. Try **HF-hosted inference first** (fastest path to benchmark inclusion).
2. Only build local adapters if hosted providers do not expose stable endpoints.

## Hosted (API) workflow

1. Add a model config under `configs/models/<model_id>.yaml` using the `*_hf` pattern:
   - `name: <model_id>`
   - `hf_name: <provider/model-repo-id>`
   - `api_key_env: HF_TOKEN`
   - `api_base: https://router.huggingface.co/v1`
   - token/retry limits as needed
2. Register alias if needed in `src/levante_bench/models/hf_hosted.py` via `@register("<model_id>")`.
3. Run:

```bash
export HF_TOKEN=...
PYTHONPATH=src .venv/bin/python -m levante_bench.cli run-eval \
  --model <model_id> \
  --version v1 \
  --device cpu
```

## Local (big GPU) workflow

Only do this if hosted path is unavailable.

1. Implement adapter in `src/levante_bench/models/<model>.py` as a `VLMModel` subclass.
2. Register in `src/levante_bench/models/__init__.py`.
3. Add config in `configs/models/<model>.yaml`.
4. Validate single-task smoke run, then full `run-eval`.

Example run command (local GPU):

```bash
PYTHONPATH=src .venv/bin/python -m levante_bench.cli run-eval \
  --model <local_model_id> \
  --version v1 \
  --device cuda
```

## Practical GPU note

For 2023-era ~7B-13B VLMs, local inference often requires:
- ~24-48GB VRAM (more without quantization),
- model-specific runtime dependencies and prompt/image formatting.

Given current repo state, **hosted first** is the shortest path to add 2022-2023 points to the historic plot.

## Local checklist: LLaVA-1.5-13B on a larger GPU machine

Use this when hosted APIs are unavailable.

### A) Machine prerequisites

- NVIDIA GPU with ~24GB+ VRAM (48GB preferred for easier headroom).
- CUDA driver/runtime aligned with your PyTorch build.
- Enough local disk for model weights (20GB+ free recommended).

### B) Create dedicated env (LLaVA-compatible)

Use a separate env to avoid breaking the core `.venv` stack.

```bash
cd ~/levante/levante-bench
python3 -m venv .venv-llava15
source .venv-llava15/bin/activate
pip install -U pip wheel

# Baseline deps from repo
pip install -r requirements.txt

# LLaVA-family compatibility (same pattern used by Aquila env docs)
pip install "transformers==4.40.0.dev0" "tokenizers~=0.15"
pip install llava
```

Quick sanity:

```bash
python - <<'PY'
import transformers
print("transformers", transformers.__version__)
import llava
print("llava import OK")
PY
```

### C) Add model adapter/config in repo

1. Implement `src/levante_bench/models/llava15.py` as a `VLMModel` subclass
   (copy structure from `aquila_vl.py` / `tinyllava.py` for prompt + image handling).
2. Register alias in `src/levante_bench/models/__init__.py`.
3. Add `configs/models/llava15_13b.yaml` with:
   - `name: llava15_13b`
   - `hf_name: llava-hf/llava-1.5-13b-hf`
   - `use_json_format` choice and basic generation knobs
   - `capabilities` including `single_image` and `multi_image`

### D) Smoke test before full run

```bash
source .venv-llava15/bin/activate
PYTHONPATH=src python -m levante_bench.cli run-eval \
  --model llava15_13b \
  --task vocab \
  --version v1 \
  --device cuda \
  --batch-size 1
```

Confirm output exists and has `summary.csv` plus per-task CSV/NPY in `results/v1/...`.

### E) Full benchmark run

```bash
source .venv-llava15/bin/activate
PYTHONPATH=src python -m levante_bench.cli run-eval \
  --model llava15_13b \
  --version v1 \
  --device cuda
```

### F) Move to historic additional models + plot

```bash
mv results/v1/llava15_13b results/v1_additional_models/
python scripts/analysis/plot_historic_frontier_models.py \
  --results-root results/v1_additional_models
```

If needed, upload:

```bash
gcloud storage rsync --recursive \
  results/v1_additional_models/llava15_13b \
  gs://levante-bench/results/v1_additional_models/llava15_13b
```

## Local checklist: CogVLM (2023)

Use this when hosted APIs are unavailable.

### A) Machine prerequisites

- NVIDIA GPU with ~24GB+ VRAM (48GB recommended for smoother inference).
- CUDA + PyTorch compatibility validated.
- 20GB+ free disk for model weights/artifacts.

### B) Create dedicated env

CogVLM has had frequent dependency drift; isolate it from the core benchmark env.

```bash
cd ~/levante/levante-bench
python3 -m venv .venv-cogvlm
source .venv-cogvlm/bin/activate
pip install -U pip wheel
pip install -r requirements.txt
```

Then install model-specific stack based on the CogVLM implementation you pick
(HF-native vs upstream repo). Pin versions in a local notes file once stable.

### C) Add model adapter/config in repo

1. Implement `src/levante_bench/models/cogvlm.py` as `VLMModel`.
2. Register alias in `src/levante_bench/models/__init__.py`.
3. Add `configs/models/cogvlm.yaml` with `hf_name`, generation params, and
   `capabilities` (`single_image`, `multi_image`).

### D) Smoke test

```bash
source .venv-cogvlm/bin/activate
PYTHONPATH=src python -m levante_bench.cli run-eval \
  --model cogvlm \
  --task vocab \
  --version v1 \
  --device cuda \
  --batch-size 1
```

### E) Full benchmark + publish

```bash
source .venv-cogvlm/bin/activate
PYTHONPATH=src python -m levante_bench.cli run-eval \
  --model cogvlm \
  --version v1 \
  --device cuda

mv results/v1/cogvlm results/v1_additional_models/
gcloud storage rsync --recursive \
  results/v1_additional_models/cogvlm \
  gs://levante-bench/results/v1_additional_models/cogvlm
```

## Local checklist: OpenFlamingo (2022-style open alternative)

Original DeepMind Flamingo is not publicly released; use OpenFlamingo as the
closest open historical proxy.

### A) Machine prerequisites

- NVIDIA GPU with ~24GB+ VRAM (larger preferred).
- CUDA/PyTorch compatibility validated.
- Enough disk for checkpoint + tokenizer + vision encoder assets.

### B) Create dedicated env

```bash
cd ~/levante/levante-bench
python3 -m venv .venv-openflamingo
source .venv-openflamingo/bin/activate
pip install -U pip wheel
pip install -r requirements.txt
```

Install OpenFlamingo dependencies/checkpoint tooling according to the specific
checkpoint variant you choose.

### C) Add model adapter/config in repo

1. Implement `src/levante_bench/models/openflamingo.py` as `VLMModel`.
2. Register in `src/levante_bench/models/__init__.py`.
3. Add `configs/models/openflamingo9b.yaml` with model identifiers and caps.

### D) Smoke test

```bash
source .venv-openflamingo/bin/activate
PYTHONPATH=src python -m levante_bench.cli run-eval \
  --model openflamingo9b \
  --task vocab \
  --version v1 \
  --device cuda \
  --batch-size 1
```

### E) Full benchmark + publish

```bash
source .venv-openflamingo/bin/activate
PYTHONPATH=src python -m levante_bench.cli run-eval \
  --model openflamingo9b \
  --version v1 \
  --device cuda

mv results/v1/openflamingo9b results/v1_additional_models/
gcloud storage rsync --recursive \
  results/v1_additional_models/openflamingo9b \
  gs://levante-bench/results/v1_additional_models/openflamingo9b
```
