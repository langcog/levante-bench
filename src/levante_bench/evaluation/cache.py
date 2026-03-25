"""Cache for model responses. Stored at results/<model_id>/cache/responses.json."""

import hashlib
import json
from pathlib import Path


def trial_hash(trial: dict) -> str:
    """Deterministic hash of trial inputs for cache lookup."""
    # Hash the fields that define a unique trial input
    key_parts = [
        trial.get("trial_id", ""),
        trial.get("item_uid", ""),
        trial.get("prompt", ""),
        str(trial.get("options", [])),
        str(trial.get("option_labels", [])),
    ]
    key = "|".join(key_parts)
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def load_cache(cache_path: Path) -> dict:
    """Load cache dict from disk, or return empty dict if missing."""
    if cache_path.exists():
        with open(cache_path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(cache_path: Path, cache: dict) -> None:
    """Write cache dict to disk."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)
