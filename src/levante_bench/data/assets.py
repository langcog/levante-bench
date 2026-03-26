"""Asset index: item_uid -> local paths lookup. Used by download script."""

import json
from pathlib import Path
from typing import Any


def load_asset_index(data_root: Path, version: str) -> dict[str, dict[str, Any]]:
    """Load item_uid index from data/assets/<version>/item_uid_index.json."""
    path = data_root / "assets" / version / "item_uid_index.json"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_paths(item_uid: str, task_id: str, data_root: Path, version: str) -> dict[str, Any] | None:
    """Look up item_uid in asset index. Returns corpus_row + resolved image_paths, or None."""
    index = load_asset_index(data_root, version)
    if not index:
        return None
    entry = index.get(item_uid)
    if not entry:
        return None
    if entry.get("task") != task_id and entry.get("internal_name") != task_id:
        return None
    image_paths = entry.get("image_paths") or []
    base = data_root / "assets" / version
    resolved = [base / Path(p) if not Path(p).is_absolute() else Path(p) for p in image_paths]
    return {
        "task": entry.get("task"),
        "internal_name": entry.get("internal_name"),
        "corpus_row": entry.get("corpus_row") or {},
        "image_paths": resolved,
    }
