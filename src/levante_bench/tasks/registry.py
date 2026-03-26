"""Task dataset registry. Maps task_id to its VLMDataset subclass."""

from typing import Type

_TASK_REGISTRY: dict[str, Type] = {}


def register_task(task_id: str):
    """Decorator to register a VLMDataset subclass for a task_id."""
    def decorator(cls: Type) -> Type:
        _TASK_REGISTRY[task_id] = cls
        return cls
    return decorator


def get_task_dataset(task_id: str) -> Type | None:
    """Return the dataset class for a task_id, or None."""
    return _TASK_REGISTRY.get(task_id)


def list_task_datasets() -> list[str]:
    """Return all registered task_ids."""
    return list(_TASK_REGISTRY.keys())
