"""VLM model adapters. Registry: name -> class."""

import importlib

from levante_bench.models.base import VLMModel
from levante_bench.models.registry import (
    get_model_class,
    list_models,
    register,
)

# Import model modules so @register() decorators execute.
for _module_name in (
    "smolvlm2",
    "qwen35",
    "internvl35",
    "tinyllava",
    "aquila_vl",
    "gemma3",
    "gemma4",
    "molmo2",
    "hf_hosted",
    "gemini",
    "gpt",
    "clip",
    "historic_local",
):
    try:
        importlib.import_module(f"levante_bench.models.{_module_name}")
    except ModuleNotFoundError:
        # Keep core CLI usable when optional local adapters are not present
        # on a specific machine/branch checkout.
        if _module_name != "historic_local":
            raise

__all__ = [
    "VLMModel",
    "get_model_class",
    "list_models",
    "register",
]
