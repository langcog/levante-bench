"""Shared helpers for task image path lookup."""

from pathlib import Path


def build_image_index(directory: Path) -> dict[str, Path]:
    """Map both image stems and filenames to their paths."""
    index = {}
    for path in directory.iterdir():
        if path.is_file():
            index[path.stem] = path
            index[path.name] = path
    return index
