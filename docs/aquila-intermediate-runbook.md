# Aquila Intermediate Integration Runbook

This document describes the full setup needed to run Aquila intermediate checkpoints in this repo on a separate machine.

It assumes you already have the code changes checked in (model adapter + config + experiment files).

## What was added

The Aquila integration lives in these tracked files:

- `src/levante_bench/models/aquila_vl.py`
- `configs/models/aquila_vl.yaml`
- `configs/experiment_aquila_vl.yaml`
- `src/levante_bench/models/__init__.py` (register import)
- `.gitignore` (adds `.venv-aquila/`)
- `docs/environment-split.md`

## Why two environments

Aquila intermediate checkpoints currently require the LLaVA stack, which is not dependency-compatible with the default benchmark stack.

Use:

- `.venv` for standard benchmark models (`smolvlm2`, `qwen35`, `internvl35`, `tinyllava`, API models)
- `.venv-aquila` for `aquila_vl`

## 1) Base project setup

From repo root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-transformers.txt
```

Quick sanity check:

```bash
PYTHONPATH=src python -m levante_bench.cli list-models
```

## 2) Aquila environment setup

Create separate env:

```bash
python -m venv .venv-aquila
source .venv-aquila/bin/activate
pip install -r requirements.txt
```

Install LLaVA + required runtime deps:

```bash
pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git
pip install einops av accelerate
```

Pin LLaVA-compatible transformers:

```bash
pip install --upgrade \
  "transformers @ git+https://github.com/huggingface/transformers.git@1c39974a4c4036fd641bc1191cc32799f85715a4" \
  "tokenizers~=0.15.2"
```

Notes:

- Do not run this install in `.venv` (it can break standard model adapters).
- `open_clip_torch` is not required for our current Aquila path.

## 3) Aquila checkpoint format notes

The intermediate repo is folder-structured:

- `BAAI/Aquila-VL-2B-Intermediate/stage2-a-llava-qwen`
- `BAAI/Aquila-VL-2B-Intermediate/stage2-b-llava-qwen`
- `BAAI/Aquila-VL-2B-Intermediate/stage2-c-llava-qwen`
- `BAAI/Aquila-VL-2B-Intermediate/stage3-llava-qwen`

Our config uses:

- `hf_name: BAAI/Aquila-VL-2B-Intermediate`
- `checkpoint_subdir: stage3-llava-qwen`

The adapter downloads that subdir and normalizes non-portable vision tower paths automatically.

## 4) Run Aquila benchmark

Activate Aquila env:

```bash
source .venv-aquila/bin/activate
```

Run full Aquila experiment:

```bash
PYTHONPATH=src python -m levante_bench.cli \
  experiment=configs/experiment_aquila_vl.yaml \
  device=cuda
```

Run quick smoke (single task):

```bash
PYTHONPATH=src python -m levante_bench.cli \
  run-eval --task vocab --model aquila_vl --version current --device cuda
```

## 5) Known issues and fixes

### A) `config.json` not found at repo root

Symptom:

- `BAAI/Aquila-VL-2B-Intermediate does not appear to have config.json`

Cause:

- Must load a stage subfolder, not repo root.

Fix:

- Keep `checkpoint_subdir` configured.

### B) FlashAttention2 import errors

Symptom:

- FlashAttention2 requested but `flash_attn` not installed.

Fix:

- Use `attn_implementation: sdpa` (already set in adapter/config path).

### C) CUDA runtime mismatch (`libnvrtc-builtins.so.13.0`)

Symptom:

- NVRTC builtins error at runtime.

Cause:

- CUDA 13 wheel stack on machine with CUDA 12 runtime.

Fix:

- Ensure `.venv-aquila` uses a torch/CUDA build matching the target machine.
- On a larger GPU machine, install a stack aligned with that host's CUDA runtime.

### D) Standard models break after Aquila install

Symptom:

- Missing `AutoModelForImageTextToText` in `transformers`.

Cause:

- LLaVA-compatible transformers replaced standard benchmark transformers.

Fix:

- Keep strict env split (`.venv` vs `.venv-aquila`).

## 6) Pre-run checklist on new machine

- NVIDIA driver + CUDA runtime are healthy (`nvidia-smi` works).
- Enough VRAM for Aquila stage checkpoint + vision tower.
- Activated correct venv:
  - `.venv` for standard models
  - `.venv-aquila` for Aquila
- `PYTHONPATH=src` set in run commands.

