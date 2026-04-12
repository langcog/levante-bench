# Adding a VLM

To add a new vision–language model to the benchmark:

1. **Model adapter:** Create a new module under `src/levante_bench/models/` that implements the same interface as the base classes in `models/base.py`:
   - For **similarity models** (CLIP-style): subclass `EvalModel` and provide the model and processor; the base class exposes `get_all_image_feats`, `get_all_text_feats`, `get_all_sim_scores`.
   - For **generative models** (LLaVA-style): subclass `GenEvalModel` and implement `get_ntp_logits` (and optionally `get_ll_logits`) and ensure `get_all_sim_scores` returns scores compatible with the R comparison scripts (e.g. per-option logits or probabilities).

2. **Registration:** Register the model in the package’s model registry (e.g. in `models/__init__.py`) so the runner and CLI can select it by name (e.g. `--model clip_base`). No change to the evaluation runner beyond the registry.

3. **R comparison:** No change to R comparison scripts is required; they read model outputs (e.g. .npy per task/model) and expect a consistent shape. Ensure your adapter writes outputs in the same format (e.g. trial × options) as existing models.

4. **Dependencies:** Add any new Python dependencies (e.g. `transformers`, model-specific packages) to `pyproject.toml` or `requirements.txt` and document in the README.

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
