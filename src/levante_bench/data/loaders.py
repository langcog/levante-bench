"""Raw data loading utilities used by task dataset subclasses."""

import re
from pathlib import Path
from typing import Optional

import pandas as pd


def load_trials_csv(path: Path) -> pd.DataFrame:
    """Load trials from a CSV file. Returns empty DataFrame if missing."""
    if not path or not Path(path).exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _safe_task_id(task_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", task_id)


def _normalize_option_str(val) -> str:
    """Normalize a canonical option value to a clean, matchable string.

    Handles the common case where pandas promotes numeric columns to float
    (e.g., ``12.0``) when NaN sentinels are present in the same column.
    Integer-valued floats are reduced to plain integer strings (``"12"``).
    True NaN/None values become empty string (treated as absent option slot).
    """
    if pd.isna(val):
        return ""
    s = str(val).strip()
    # "12.0" → "12"  (only strips trailing .0 when the prefix is a plain integer)
    try:
        f = float(s)
        if f == int(f):
            s = str(int(f))
    except (ValueError, OverflowError):
        pass
    return s


def load_human_proportions(
    proportions_path: Path,
    option_key_path: Optional[Path] = None,
) -> dict[str, dict]:
    """Load human response proportions for a task from the R-generated CSVs.

    Returns a dict keyed by item_uid.  Each value has:
        canonical_options : list[str]   — the option labels from the option_key
                                          (image1=first option, often the correct
                                          answer for tasks like math; for vocab
                                          these are distractor words because of how
                                          the Redivis answer column is structured)
        proportions       : list[float] — response proportions aligned with
                                          canonical_options, summing to ~1

    If proportions_path does not exist an empty dict is returned (caller
    treats human data as unavailable).

    The option_key CSV is derived from proportions_path by replacing
    ``_proportions.csv`` → ``_option_key.csv`` when option_key_path is None.
    """
    proportions_path = Path(proportions_path)
    if not proportions_path.exists():
        return {}

    props_df = pd.read_csv(proportions_path)
    image_cols = [c for c in props_df.columns if re.match(r"^image\d+$", c)]
    if not image_cols:
        return {}

    if option_key_path is None:
        option_key_path = proportions_path.parent / proportions_path.name.replace(
            "_proportions.csv", "_option_key.csv"
        )

    key_index: dict[str, list[str]] = {}
    if Path(option_key_path).exists():
        key_df = pd.read_csv(option_key_path)
        for _, row in key_df.iterrows():
            uid = str(row["item_uid"]).strip()
            opts = [_normalize_option_str(row.get(col)) for col in image_cols]
            key_index[uid] = opts

    result: dict[str, dict] = {}
    for _, row in props_df.iterrows():
        uid = str(row["item_uid"]).strip()
        proportions = [
            float(row[col]) if pd.notna(row.get(col)) else 0.0
            for col in image_cols
        ]
        canonical_options = key_index.get(uid, image_cols)
        result[uid] = {
            "canonical_options": canonical_options,
            "proportions": proportions,
        }

    return result
