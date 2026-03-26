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
