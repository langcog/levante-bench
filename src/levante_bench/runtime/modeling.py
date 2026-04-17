"""Reusable helpers for loading and constructing registered models."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Mapping

from omegaconf import DictConfig, OmegaConf

from levante_bench.config.loader import load_model_config
from levante_bench.models import VLMModel, get_model_class

_MODEL_CFG_EXCLUDE = {
    "name",
    "hf_name",
    "size",
    "max_new_tokens",
    "use_json_format",
    "capabilities",
}


def _to_dict(cfg: Mapping[str, Any] | DictConfig) -> dict[str, Any]:
    if isinstance(cfg, DictConfig):
        return OmegaConf.to_container(cfg, resolve=False)
    return dict(cfg)


def resolve_model_config(
    model_name: str,
    model_overrides: Mapping[str, Any] | None = None,
    model_config: Mapping[str, Any] | DictConfig | None = None,
    configs_root: str | Path | None = None,
) -> dict[str, Any]:
    """Load and merge model config into a plain dict with OmegaConf resolution."""
    if model_config is None:
        loaded_cfg = load_model_config(model_name, configs_root=configs_root)
        if loaded_cfg is None:
            raise ValueError(f"No model config found for '{model_name}'.")
        base_cfg = OmegaConf.to_container(loaded_cfg, resolve=False)
    else:
        base_cfg = _to_dict(model_config)

    merged = dict(base_cfg)
    if model_overrides:
        merged.update(dict(model_overrides))

    return OmegaConf.to_container(OmegaConf.create(merged), resolve=True)


def build_model(
    model_name: str,
    model_cfg: Mapping[str, Any],
    device: str,
    auto_load: bool = True,
) -> VLMModel:
    """Instantiate a registered model using benchmark model config fields."""
    model_cls = get_model_class(model_name)
    if model_cls is None:
        raise ValueError(f"Model '{model_name}' is not registered.")

    hf_name = model_cfg.get("hf_name")
    if not hf_name:
        raise ValueError(f"Missing hf_name in model config for '{model_name}'.")

    ctor_cfg = {
        key: value
        for key, value in model_cfg.items()
        if key not in _MODEL_CFG_EXCLUDE
    }
    sig = inspect.signature(model_cls.__init__)
    has_var_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    if not has_var_kwargs:
        accepted = {
            name for name in sig.parameters if name not in {"self", "model_name", "device"}
        }
        ctor_cfg = {k: v for k, v in ctor_cfg.items() if k in accepted}

    model = model_cls(model_name=str(hf_name), device=device, **ctor_cfg)
    model.use_json_format = bool(model_cfg.get("use_json_format", True))
    if auto_load:
        model.load()
    return model
