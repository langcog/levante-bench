"""Default paths and LEVANTE assets bucket URL."""

import os
from pathlib import Path

LEVANTE_ASSETS_BUCKET_URL = "https://storage.googleapis.com/levante-assets-prod"


def get_task_mapping_path() -> Path:
    """Path to task_name_mapping.csv (used by download script)."""
    return Path(__file__).resolve().parent / "task_name_mapping.csv"


def get_assets_base_url() -> str:
    """Base URL for LEVANTE assets bucket."""
    return os.environ.get("LEVANTE_ASSETS_BUCKET_URL", LEVANTE_ASSETS_BUCKET_URL)


def detect_data_version(data_root: Path | None = None) -> str:
    """Return the asset version to use, resolved in this order:

    1. ``LEVANTE_DATA_VERSION`` environment variable (explicit override).
    2. The most recently modified subfolder in ``<data_root>/assets/``.

    Raises ``RuntimeError`` if no asset folder is found and the env var
    is not set.  Pass ``data_root`` as the project ``data/`` directory;
    defaults to the ``data/`` sibling of the repo root inferred from this
    file's location.
    """
    env = os.environ.get("LEVANTE_DATA_VERSION", "").strip()
    if env:
        return env

    if data_root is None:
        # src/levante_bench/config/defaults.py → up 4 levels → repo root / data
        data_root = Path(__file__).resolve().parents[4] / "data"

    assets_dir = Path(data_root) / "assets"
    if not assets_dir.is_dir():
        raise RuntimeError(
            f"Assets directory not found: {assets_dir}. "
            "Run scripts/download_levante_assets.py first, or set "
            "LEVANTE_DATA_VERSION."
        )

    candidates = [d for d in assets_dir.iterdir() if d.is_dir()]
    if not candidates:
        raise RuntimeError(
            f"No asset folders found in {assets_dir}. "
            "Run scripts/download_levante_assets.py first, or set "
            "LEVANTE_DATA_VERSION."
        )

    # Most recently modified wins; tie-break by name for deterministic output.
    newest = max(candidates, key=lambda d: (d.stat().st_mtime_ns, d.name))
    return newest.name
