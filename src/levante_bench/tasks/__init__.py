"""Per-task datasets. Each subclass of VLMDataset handles loading
and formatting raw task data into standard trial dicts."""

from levante_bench.tasks.registry import get_task_dataset, list_task_datasets

# Import task modules so @register_task() runs
from levante_bench.tasks import egma_math  # noqa: F401
from levante_bench.tasks import matrix_reasoning  # noqa: F401
from levante_bench.tasks import theory_of_mind  # noqa: F401
from levante_bench.tasks import egma_math_manifest  # noqa: F401
from levante_bench.tasks import matrix_reasoning  # noqa: F401
from levante_bench.tasks import mental_rotation  # noqa: F401
from levante_bench.tasks import theory_of_mind_manifest  # noqa: F401
from levante_bench.tasks import trog  # noqa: F401
from levante_bench.tasks import vocab  # noqa: F401

__all__ = [
    "get_task_dataset",
    "list_task_datasets",
]
