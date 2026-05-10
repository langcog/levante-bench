#!/usr/bin/env python3
"""Validate sign conventions used by analysis and comparison scripts.

This script checks empirical invariants that should hold for the current
LEVANTE response export:

1. Higher child IRT ability should correlate positively with child age.
2. The exported IRT ``d`` item parameter is easiness-oriented in practice:
   higher ``d`` should correlate positively with human item accuracy.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]

TASK_ORDER: tuple[str, ...] = (
    "egma-math",
    "matrix-reasoning",
    "mental-rotation",
    "theory-of-mind",
    "trog",
    "vocab",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default="v1", help="Data version under data/responses.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=REPO_ROOT,
        help="Project root path.",
    )
    parser.add_argument(
        "--min-correlation",
        type=float,
        default=0.05,
        help="Minimum positive correlation required for each sign check.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=REPO_ROOT / "results" / "analysis" / "sign_convention_validation.csv",
        help="Where to write per-task validation results.",
    )
    return parser.parse_args()


def to_bool(series: pd.Series) -> pd.Series:
    normalized = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .replace({"true": "1", "false": "0"})
    )
    numeric = pd.to_numeric(normalized, errors="coerce")
    return numeric == 1


def validate_task(
    task_id: str,
    trials: pd.DataFrame,
    irt_dir: Path,
    min_correlation: float,
) -> dict[str, object]:
    task_trials = trials[trials["task_id"] == task_id].copy()
    ability_path = irt_dir / f"{task_id}_ability_scores.csv"
    params_path = irt_dir / f"{task_id}_item_params.csv"

    age_ability_corr = None
    n_ability_runs = 0
    if ability_path.is_file():
        ability = pd.read_csv(ability_path, usecols=["run_id", "ability"])
        child_age = (
            task_trials[["run_id", "age"]]
            .dropna()
            .drop_duplicates(subset=["run_id"])
            .copy()
        )
        merged = child_age.merge(ability, on="run_id", how="inner")
        n_ability_runs = len(merged)
        if len(merged) >= 3:
            age_ability_corr = float(merged["age"].corr(merged["ability"]))

    d_accuracy_corr = None
    n_items = 0
    if params_path.is_file():
        params = pd.read_csv(params_path, usecols=["item_uid", "difficulty"])
        item_accuracy = (
            task_trials.assign(is_correct=to_bool(task_trials["correct"]))
            .groupby("item_uid", as_index=False)
            .agg(human_accuracy=("is_correct", "mean"), n_trials=("is_correct", "size"))
        )
        merged_items = item_accuracy.merge(params, on="item_uid", how="inner")
        n_items = len(merged_items)
        if len(merged_items) >= 3:
            d_accuracy_corr = float(merged_items["difficulty"].corr(merged_items["human_accuracy"]))

    ability_ok = age_ability_corr is not None and age_ability_corr > min_correlation
    d_ok = d_accuracy_corr is not None and d_accuracy_corr > min_correlation
    return {
        "task_id": task_id,
        "n_ability_runs": n_ability_runs,
        "age_ability_corr": age_ability_corr,
        "age_ability_sign_ok": ability_ok,
        "n_items_with_d": n_items,
        "irt_d_human_accuracy_corr": d_accuracy_corr,
        "irt_d_sign_ok": d_ok,
        "irt_d_interpretation": "higher_d_is_easier",
    }


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    response_dir = project_root / "data" / "responses" / args.version
    trials_path = response_dir / "trials.csv"
    irt_dir = response_dir / "irt_models"

    trials = pd.read_csv(trials_path, low_memory=False)
    needed = {"task_id", "run_id", "age", "item_uid", "correct"}
    missing = needed - set(trials.columns)
    if missing:
        raise RuntimeError(f"Missing required columns in {trials_path}: {sorted(missing)}")
    trials = trials[trials["task_id"].isin(TASK_ORDER)].copy()
    trials["age"] = pd.to_numeric(trials["age"], errors="coerce")

    rows = [
        validate_task(
            task_id=task_id,
            trials=trials,
            irt_dir=irt_dir,
            min_correlation=args.min_correlation,
        )
        for task_id in TASK_ORDER
    ]
    out = pd.DataFrame(rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)

    print(f"Wrote {args.output_csv}")
    print(out.to_string(index=False))

    failed = out[
        (~out["age_ability_sign_ok"].astype(bool)) | (~out["irt_d_sign_ok"].astype(bool))
    ]
    if not failed.empty:
        print("\nFAILED sign convention checks:")
        print(failed[["task_id", "age_ability_corr", "irt_d_human_accuracy_corr"]].to_string(index=False))
        return 1

    print("\nAll sign convention checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
