"""Helpers for stochastic decoding and label aggregation (prototypes, notebooks)."""

from __future__ import annotations

from collections import Counter

from levante_bench.models.base import (
    ANSWER_FORMAT_INSTRUCTION,
    NUMERIC_ANSWER_FORMAT_INSTRUCTION,
    SLIDER_POSITION_FORMAT_INSTRUCTION,
    VLMModel,
)


def image_paths_for_trial(trial: dict) -> list[str] | None:
    ctx = trial.get("context_image_paths", []) or []
    opt = trial.get("option_image_paths", []) or []
    paths = list(ctx) + list(opt)
    return paths if paths else None


def eval_prompt_for_trial(model: VLMModel, trial: dict) -> str:
    """Mirror ``evaluate_trial`` prompt augmentation (e.g. JSON answer format)."""
    prompt = trial["prompt"]
    answer_format = str(trial.get("answer_format", "label")).strip().lower()
    if model.use_json_format:
        if answer_format == "slider_position":
            prompt += SLIDER_POSITION_FORMAT_INSTRUCTION
        elif answer_format == "numeric":
            prompt += NUMERIC_ANSWER_FORMAT_INSTRUCTION
        else:
            prompt += ANSWER_FORMAT_INSTRUCTION
    return prompt


def collect_parsed_labels(
    model: VLMModel,
    trial: dict,
    *,
    n_samples: int,
    do_sample: bool = True,
    temperature: float = 0.8,
    top_p: float = 0.95,
    base_seed: int = 0,
    max_new_tokens: int | None = None,
) -> list[str | None]:
    """Run ``n_samples`` generations on the same trial; return parsed option labels."""
    mnt = max_new_tokens if max_new_tokens is not None else trial.get("max_new_tokens", 64)
    paths = image_paths_for_trial(trial)
    prompt_text = eval_prompt_for_trial(model, trial)
    out: list[str | None] = []
    for i in range(n_samples):
        raw = model.generate(
            prompt_text=prompt_text,
            image_paths=paths,
            max_new_tokens=int(mnt),
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            sample_seed=None if not do_sample else base_seed + i,
        )
        clean = model.parse_response(raw)
        label, _reason = model.parse_answer(clean, trial["option_labels"])
        out.append(label)
    return out


def greedy_predicted_label(
    model: VLMModel,
    trial: dict,
    *,
    max_new_tokens: int | None = None,
) -> str | None:
    """Single greedy decode (``do_sample=False``), same prompt wiring as the runner."""
    mnt = max_new_tokens if max_new_tokens is not None else trial.get("max_new_tokens", 64)
    paths = image_paths_for_trial(trial)
    prompt_text = eval_prompt_for_trial(model, trial)
    raw = model.generate(
        prompt_text=prompt_text,
        image_paths=paths,
        max_new_tokens=int(mnt),
        do_sample=False,
    )
    clean = model.parse_response(raw)
    label, _reason = model.parse_answer(clean, trial["option_labels"])
    return label


def label_histogram(labels: list[str | None]) -> Counter[str]:
    """Count labels; unparseable responses are bucketed as ``\"_unparsed\"``."""
    c: Counter[str] = Counter()
    for lab in labels:
        c[str(lab) if lab is not None else "_unparsed"] += 1
    return c
