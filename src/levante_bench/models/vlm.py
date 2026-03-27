"""Backward-compatibility shim.

All model classes have been moved to their own modules:
    levante_bench.models.smolvlm2   → SmolVLM2Model
    levante_bench.models.qwen35     → Qwen35Model
    levante_bench.models.internvl35 → InternVL35Model

This file re-exports them so that existing code importing from
``levante_bench.models.vlm`` continues to work unchanged.
"""

from levante_bench.models.smolvlm2 import SmolVLM2Model  # noqa: F401
from levante_bench.models.qwen35 import Qwen35Model  # noqa: F401
from levante_bench.models.internvl35 import InternVL35Model  # noqa: F401
from levante_bench.models.tinyllava import TinyLLaVAModel  # noqa: F401
from levante_bench.models._common import DTYPE_MAP  # noqa: F401

__all__ = ["SmolVLM2Model", "Qwen35Model", "InternVL35Model", "TinyLLaVAModel", "DTYPE_MAP"]
