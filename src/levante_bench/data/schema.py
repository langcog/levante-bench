"""Canonical types and schema for LEVANTE trials and human response data.

Aligns with the Redivis trials table and derived layout (manifest + human
response CSVs). item_uid is the key for mapping trials to assets via the
LEVANTE corpus.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# --- Task definition (config) ---


@dataclass
class TaskDef:
    """Definition of a LEVANTE task for the benchmark."""

    task_id: str
    benchmark_name: str
    internal_name: str  # bucket path segment, e.g. egma-math
    manifest_path: Optional[Path] = None  # path to manifest CSV under data/raw/<version>/tasks/
    human_response_path: Optional[Path] = None  # path to human aggregates under data/raw/<version>/human/
    task_type: str = "forced-choice"  # e.g. forced-choice, similarity
    n_options: int = 4
    has_correct: bool = True  # whether there is a correct-answer key
    corpus_file: Optional[str] = None  # corpus CSV filename in bucket
    context_type: str = "none"  # "none" | "single_image" | "multi_image"
    option_type: str = "text"  # "text" | "image"
    include_numberline: bool = False  # egma-math manifest: include Number Line rows
    prompt_language: str = "en"  # language key in translations/item-bank-translations.csv
    true_random_option_order: bool = False  # randomize option order per run
    option_order_run_seed: Optional[int] = None  # run seed used to derive item seeds


# --- Trial (one row from trials table / manifest) ---


@dataclass
class Trial:
    """One trial: aligns with Redivis trials table and manifest rows."""

    trial_id: str
    item_uid: str  # key for joining to corpus / asset paths
    task_id: str
    user_id: Optional[str] = None
    run_id: Optional[str] = None
    trial_number: Optional[int] = None
    response: Optional[str] = None
    correct: Optional[bool] = None
    answer: Optional[str] = None  # correct answer label
    # Resolved asset paths (filled by loaders using item_uid index)
    image_paths: list[Path] = field(default_factory=list)  # e.g. [image1, image2, ...]
    text_paths: list[Path] = field(default_factory=list)
    # Optional: raw columns from Redivis
    rt_numeric: Optional[float] = None
    difficulty: Optional[float] = None
    chance: Optional[float] = None


# --- Human response aggregates (for R comparison) ---


@dataclass
class HumanResponseSummary:
    """Per-trial (or per item_uid) human response distribution over options."""

    trial_id: str
    item_uid: str
    task_id: str
    option_labels: list[str]  # e.g. ["A", "B", "C", "D"] or image1..image4
    proportions: list[float]  # one per option, sum to 1
    age_bin: Optional[str] = None  # e.g. "5-6", "7-8"
    n_respondents: Optional[int] = None


# --- Asset index (item_uid → local paths) ---

# The asset index is built by the download script and consumed by loaders.
# Typing: (item_uid, task_id) -> dict with keys like "image1", "image2", "text1", "corpus_row"
# We do not define a dedicated dataclass for the index entry; loaders use
# dict or a small TypedDict if needed (see assets.py).
