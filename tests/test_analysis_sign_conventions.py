"""Tests for analysis sign-convention validation helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "analysis"
    / "validate_sign_conventions.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("validate_sign_conventions", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_validate_task_treats_higher_irt_d_as_easier(tmp_path: Path) -> None:
    module = _load_module()
    irt_dir = tmp_path / "irt_models"
    irt_dir.mkdir()

    pd.DataFrame(
        {
            "run_id": ["c1", "c2", "c3", "c4"],
            "ability": [-2.0, -0.5, 0.5, 2.0],
        }
    ).to_csv(irt_dir / "vocab_ability_scores.csv", index=False)
    pd.DataFrame(
        {
            "item_uid": ["hard", "medium", "easy"],
            "difficulty": [-1.5, 0.0, 1.5],
        }
    ).to_csv(irt_dir / "vocab_item_params.csv", index=False)

    trials = pd.DataFrame(
        {
            "task_id": ["vocab"] * 12,
            "run_id": ["c1", "c2", "c3", "c4"] * 3,
            "age": [5.0, 7.0, 9.0, 11.0] * 3,
            "item_uid": ["hard"] * 4 + ["medium"] * 4 + ["easy"] * 4,
            "correct": [0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1],
        }
    )

    result = module.validate_task(
        task_id="vocab",
        trials=trials,
        irt_dir=irt_dir,
        min_correlation=0.05,
    )

    assert result["age_ability_corr"] > 0
    assert result["age_ability_sign_ok"] is True
    assert result["irt_d_human_accuracy_corr"] > 0
    assert result["irt_d_sign_ok"] is True
    assert result["irt_d_interpretation"] == "higher_d_is_easier"
