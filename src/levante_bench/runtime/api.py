"""Client-facing runtime API for external trial evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

from omegaconf import DictConfig, OmegaConf

from levante_bench.models import VLMModel
from levante_bench.runtime.modeling import build_model, resolve_model_config


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
