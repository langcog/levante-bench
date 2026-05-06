# Adding a VLM

To add a new vision–language model to the benchmark:

1. **Model adapter:** Create a new module under `src/levante_bench/models/` that subclasses `VLMModel` from `models/base.py`. Most adapters implement `load()` plus `generate(prompt_text, image_paths, max_new_tokens) -> str`; the base class then handles answer parsing via `evaluate_trial()` / `parse_answer_result()`.
   - For **similarity models** (CLIP-style) that have no text decoder, override `evaluate_trial(trial)` directly and return the canonical result dict (`predicted_label`, `is_correct`, etc.). See `models/clip.py` for the reference pattern: image-image scoring when the trial has a context image, text-image scoring otherwise, and skipped results for numeric / slider tasks.
   - For **generative models** (LLaVA-style), implementing `generate()` is sufficient; optionally override `score_choices()` for logit-forced two-alternative scoring.

2. **Registration:** Add `@register("<model_id>")` in your adapter and import the module in `src/levante_bench/models/__init__.py` so the decorator runs at package import time. The runner and CLI then accept `--model <model_id>`.

3. **Outputs:** No changes needed to `evaluation/runner.py` or the CSV/NPY writers; they read the canonical result fields populated by `evaluate_trial()`.

4. **Dependencies:** Add any new Python dependencies (e.g. `transformers`, model-specific packages) to `pyproject.toml`, `requirements.txt`, or `requirements-transformers.txt` and document in the README.

## Hosted API-backed models

For hosted models (for example large HF Inference Providers VLMs), you can
reuse an existing adapter and add only a model config.

- Adapter: `src/levante_bench/models/hf_hosted.py`
- Registration: add a new `@register("<model_id>")` alias
- Config: add `configs/models/<model_id>.yaml` with fields like:
  - `name`
  - `hf_name`
  - `api_key_env` (for example `HF_TOKEN`)
  - `api_base` (`https://router.huggingface.co/v1`)
  - `max_new_tokens`, retry/timeout knobs, and `capabilities`

Then run through standard runner/CLI flow:

- `levante-bench run-eval --model <model_id> --version current --device cpu`
