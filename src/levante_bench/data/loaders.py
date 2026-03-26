"""Raw data loading utilities used by task dataset subclasses."""

from pathlib import Path

import pandas as pd



def load_trials_csv(path: Path) -> pd.DataFrame:
    """Load trials from a CSV file. Returns empty DataFrame if missing."""
    if not path or not Path(path).exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _safe_task_id(task_id: str) -> str:
    import re
    return re.sub(r"[^a-zA-Z0-9_-]", "_", task_id)
