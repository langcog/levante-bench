"""VLMDataset base and task-specific subclasses."""

from pathlib import Path
from typing import Optional

from PIL import Image
from torch.utils.data import Dataset

from levante_bench.data.schema import TaskDef


class VLMDataset(Dataset):
    """Base dataset that serves standardized trial dicts.

    Subclasses set up raw data references in __init__ and implement
    __getitem__ to return a single formatted trial dict with:
        trial_id, item_uid, prompt, options, option_labels,
        correct_label, context_images, option_images,
        context_type, option_type
    """

    def __init__(self, task_def: TaskDef, version: str, data_root: Optional[Path] = None) -> None:
        self.task_def = task_def
        self.version = version
        self.data_root = data_root

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> dict:
        """Return one formatted trial dict with images loaded."""
        raise NotImplementedError
