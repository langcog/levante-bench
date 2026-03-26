"""VLM model adapters. Registry: name -> class."""

from levante_bench.models.base import VLMModel
from levante_bench.models.registry import (
    get_model_class,
    list_models,
    register,
)

# Import model modules so @register() runs
from levante_bench.models import vlm  # noqa: F401

__all__ = [
    "VLMModel",
    "get_model_class",
    "list_models",
    "register",
]
