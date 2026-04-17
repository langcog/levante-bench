# Runtime exports for sibling repos

This document describes the reusable runtime surface exported by `levante-bench` and how another repository can use it without adopting LEVANTE task loaders.

## Exported API

From `levante_bench.runtime`:

- `load_model(...)`
- `run_trials(...)`
- `evaluate_trials(...)`
- `resolve_model_config(...)`
- `build_model(...)`
- `score_choices(...)`
- `run_logit_forced_12(...)`

These are designed for external callers that already have their own dataset/task pipeline.

## Trial contract (input rows)

Each trial is a dictionary. Required fields:

- `trial_id`
- `item_uid`
- `prompt`
- `option_labels` (non-empty list)

And at least one of:

- `correct_label` (label tasks), or
- `target_value` (numeric/slider tasks), or
- `answer_format` (explicit task format)

Optional fields:

- `context_image_paths` (list of file paths)
- `option_image_paths` (list of file paths)
- `max_new_tokens`
- `task_id`

## Python integration example

```python
from levante_bench.runtime import load_model, run_trials

model = load_model(model_name="qwen35", device="auto")

trials = [
    {
        "trial_id": "sample-1",
        "item_uid": "sample-1",
        "prompt": "Pick the best option.",
        "option_labels": ["A", "B", "C", "D"],
        "correct_label": "A",
        "context_image_paths": [],
        "option_image_paths": [],
        "answer_format": "label",
    }
]

results = run_trials(model, trials, max_new_tokens=64, task_id="custom")
```

## CLI integration for non-Python pipelines

`run-trials-jsonl` allows external repos to call the runtime through JSONL files.

```bash
levante-bench run-trials-jsonl \
  --input-jsonl ./trials.jsonl \
  --output-jsonl ./results.jsonl \
  --model qwen35 \
  --device auto \
  --max-new-tokens 64 \
  --task-id custom
```

`trials.jsonl` must contain one trial object per line following the trial contract above.

## Logit-forced 1-vs-2 scoring

For repos that need 2AFC logit-forced behavior (e.g. ShapeBias `logit_forced_12`), use:

```python
from levante_bench.runtime import load_model, run_logit_forced_12

model = load_model(model_name="qwen35", device="auto")
out = run_logit_forced_12(
    model=model,
    prompt_text="Which candidate matches the reference? Reply 1 or 2.",
    image_paths=["/path/reference.png", "/path/image1.png", "/path/image2.png"],
    swap_correct=True,
)
```

`run_logit_forced_12` returns:
- `predicted_choice` (`"1"`, `"2"`, or `None`)
- `choice_probs` for choices 1/2
- `choice_logits` for choices 1/2
- optional `swap_corrected` details when `swap_correct=True`

Note: this requires a model adapter that implements `score_choices` (currently local HF-backed adapters like `qwen35`, `smolvlm2`, `internvl35`).

Smoke-test script:

```bash
python scripts/smoke_tests/smoke_test_runtime_logit_forced.py --model qwen35 --device auto
```

## Config resolution options

External callers can load model configuration in multiple ways:

- registry model id:
  - `load_model(model_name="qwen35")`
- explicit config file:
  - `load_model(model_config_path="/abs/path/model.yaml")`
- inline config mapping:
  - `load_model(model_config={...})`
- custom config tree:
  - `load_model(model_name="...", configs_root="/abs/path/configs")`

## Caching and model weights

Runtime model adapters use `transformers.from_pretrained(...)`.
If weights were previously downloaded in Hugging Face cache, external runs reuse them automatically.

## Remote/provider-backed models from sibling repos

External repos can also use remote models configured in `configs/models`, including:

- `gemini_pro`
- `gpt53` (and `gpt52` alias)

Example:

```python
from levante_bench.runtime import load_model, run_trials

model = load_model(model_name="gemini_pro", device="auto")
results = run_trials(model, trials, max_new_tokens=256, task_id="custom")
```

Required environment variables:

- `gemini_pro`: `GEMINI_API_KEY`
- `gpt53` / `gpt52`: `OPENAI_API_KEY`

These models are loaded through the same exported runtime API. They are selected by model id, and credentials are read from environment variables at `load_model(...)` time.

## Example sibling workflow (ShapeBias)

For a concrete cross-repo setup, see:

- `shapebias-bench2/LEVANTE_RUNTIME_INTEGRATION.md`

That guide documents editable install, environment setup, and local execution commands.
