"""Client-facing runtime API for external trial evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

from omegaconf import DictConfig, OmegaConf

from levante_bench.models import VLMModel
from levante_bench.runtime.modeling import build_model, resolve_model_config


def _validate_choice_texts(model: VLMModel, choice_texts: tuple[str, str]) -> None:
    if len(choice_texts) != 2:
        raise ValueError(
            f"choice_texts must contain exactly 2 entries, got {len(choice_texts)}."
        )
    for choice in choice_texts:
        if not str(choice).strip():
            raise ValueError("choice_texts entries must be non-empty strings.")

    tokenizer = getattr(getattr(model, "processor", None), "tokenizer", None)
    if tokenizer is None:
        return
    for choice in choice_texts:
        toks = tokenizer.encode(str(choice), add_special_tokens=False)
        if len(toks) != 1:
            raise ValueError(
                "Logit-forced scoring requires one-token choices. "
                f"Choice {choice!r} maps to token ids {toks}."
            )


def resolve_device(device: str = "auto") -> str:
    """Resolve auto device selection with safe CUDA -> CPU fallback."""
    choice = (device or "auto").strip().lower()
    if choice != "auto":
        return choice
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _load_model_cfg_from_path(path: str | Path) -> dict[str, Any]:
    cfg = OmegaConf.load(path)
    if not isinstance(cfg, DictConfig):
        raise ValueError(f"Model config at '{path}' did not load as a mapping.")
    return OmegaConf.to_container(cfg, resolve=False)


def load_model(
    model_name: str | None = None,
    *,
    model_config: Mapping[str, Any] | DictConfig | None = None,
    model_config_path: str | Path | None = None,
    model_overrides: Mapping[str, Any] | None = None,
    configs_root: str | Path | None = None,
    device: str = "auto",
    auto_load: bool = True,
) -> VLMModel:
    """Load one registered model for ad-hoc external evaluation."""
    base_cfg: Mapping[str, Any] | DictConfig | None = model_config
    if model_config_path is not None:
        base_cfg = _load_model_cfg_from_path(model_config_path)

    if model_name is None:
        candidate_name = ""
        if base_cfg is not None:
            if isinstance(base_cfg, DictConfig):
                candidate_name = str(base_cfg.get("name") or "").strip()
            else:
                candidate_name = str(base_cfg.get("name") or "").strip()
        if not candidate_name:
            raise ValueError(
                "model_name is required unless model_config/model_config_path contains 'name'."
            )
        model_name = candidate_name

    resolved_cfg = resolve_model_config(
        model_name=model_name,
        model_overrides=model_overrides,
        model_config=base_cfg,
        configs_root=configs_root,
    )
    resolved_device = resolve_device(device)
    return build_model(
        model_name=model_name,
        model_cfg=resolved_cfg,
        device=resolved_device,
        auto_load=auto_load,
    )


def run_trials(
    model: VLMModel,
    trials: Iterable[Mapping[str, Any]],
    *,
    max_new_tokens: int | None = None,
    task_id: str | None = None,
) -> list[dict[str, Any]]:
    """Evaluate a sequence of standardized trial dictionaries."""
    results: list[dict[str, Any]] = []
    for raw_trial in trials:
        trial = dict(raw_trial)
        if task_id is not None and "task_id" not in trial:
            trial["task_id"] = task_id
        if max_new_tokens is not None and "max_new_tokens" not in trial:
            trial["max_new_tokens"] = max_new_tokens
        results.append(model.evaluate_trial(trial))
    return results


def score_choices(
    model: VLMModel,
    *,
    prompt_text: str,
    image_paths: list[str],
    choice_texts: tuple[str, str] = ("1", "2"),
) -> dict[str, Any]:
    """Score fixed next-token choices for logit-forced decisions."""
    _validate_choice_texts(model, choice_texts)
    try:
        return model.score_choices(
            prompt_text=prompt_text,
            image_paths=image_paths,
            choice_texts=choice_texts,
        )
    except NotImplementedError as exc:
        raise ValueError(
            f"Model {model.__class__.__name__} does not support score_choices/logit-forced scoring."
        ) from exc


def run_logit_forced_12(
    model: VLMModel,
    *,
    prompt_text: str,
    image_paths: list[str],
    swap_correct: bool = False,
) -> dict[str, Any]:
    """Run a two-choice logit-forced decision with optional swap correction."""
    base = score_choices(
        model=model,
        prompt_text=prompt_text,
        image_paths=image_paths,
        choice_texts=("1", "2"),
    )
    p1, p2 = base["choice_probs"]
    l1, l2 = base["choice_logits"]

    corrected = None
    if swap_correct and len(image_paths) >= 3:
        swapped = [image_paths[0], image_paths[2], image_paths[1], *image_paths[3:]]
        sw = score_choices(
            model=model,
            prompt_text=prompt_text,
            image_paths=swapped,
            choice_texts=("1", "2"),
        )
        sp1, sp2 = sw["choice_probs"]
        p_a = 0.5 * (p1 + sp2)
        p_b = 0.5 * (p2 + sp1)
        corrected = {
            "choice_probs": [float(p_a), float(p_b)],
            "base_choice_probs": [float(p1), float(p2)],
            "swap_choice_probs": [float(sp1), float(sp2)],
        }
        probs = [float(p_a), float(p_b)]
        total_time = float(base.get("generation_time_s", 0.0)) + float(
            sw.get("generation_time_s", 0.0)
        )
    else:
        probs = [float(p1), float(p2)]
        total_time = float(base.get("generation_time_s", 0.0))

    if probs[0] > probs[1]:
        predicted_choice = "1"
    elif probs[1] > probs[0]:
        predicted_choice = "2"
    else:
        predicted_choice = None

    out: dict[str, Any] = {
        "predicted_choice": predicted_choice,
        "choice_probs": probs,
        "choice_logits": [float(l1), float(l2)],
        "model_name": base.get("model_name"),
        "generation_time_s": total_time,
    }
    if corrected is not None:
        out["swap_corrected"] = corrected
    return out


def evaluate_trials(
    trials: Iterable[Mapping[str, Any]],
    *,
    model_name: str,
    model_config: Mapping[str, Any] | DictConfig | None = None,
    model_config_path: str | Path | None = None,
    model_overrides: Mapping[str, Any] | None = None,
    configs_root: str | Path | None = None,
    device: str = "auto",
    max_new_tokens: int | None = None,
    task_id: str | None = None,
) -> list[dict[str, Any]]:
    """Convenience one-shot API: load a model and evaluate trials."""
    model = load_model(
        model_name=model_name,
        model_config=model_config,
        model_config_path=model_config_path,
        model_overrides=model_overrides,
        configs_root=configs_root,
        device=device,
        auto_load=True,
    )
    return run_trials(
        model=model,
        trials=trials,
        max_new_tokens=max_new_tokens,
        task_id=task_id,
    )
