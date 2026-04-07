#!/usr/bin/env python3
"""Build closest IRT ability-bin report from KL comparison CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize closest ability bin per task/model using mean D_KL."
    )
    parser.add_argument(
        "--comparison-dir",
        default="results/comparison",
        help="Directory containing *_d_kl.csv files.",
    )
    parser.add_argument(
        "--output-csv",
        default="results/comparison/closest_ability_bin_by_model_task.csv",
        help="Output CSV path.",
    )
    return parser.parse_args()


def parse_bin_start(value: str) -> int:
    try:
        return int(str(value).split("_", 1)[0])
    except Exception:
        return 10**9


def main() -> int:
    args = parse_args()
    comparison_dir = Path(args.comparison_dir)
    output_csv = Path(args.output_csv)
    if not comparison_dir.exists():
        raise FileNotFoundError(f"Comparison directory not found: {comparison_dir}")

    rows: list[pd.DataFrame] = []
    for csv_path in sorted(comparison_dir.glob("*_d_kl.csv")):
        df = pd.read_csv(csv_path)
        expected = {"task", "model", "ability_bin", "D_KL"}
        if not expected.issubset(df.columns):
            continue
        rows.append(df[["task", "model", "ability_bin", "D_KL"]].copy())

    if not rows:
        raise RuntimeError("No KL files found with required columns.")

    all_rows = pd.concat(rows, ignore_index=True)
    grouped = (
        all_rows.groupby(["task", "model", "ability_bin"], as_index=False)["D_KL"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "mean_d_kl", "count": "n_item_rows"})
    )
    grouped["ability_bin_start"] = grouped["ability_bin"].map(parse_bin_start)

    winners = (
        grouped.sort_values(
            by=["task", "model", "mean_d_kl", "ability_bin_start", "ability_bin"],
            ascending=[True, True, True, True, True],
        )
        .groupby(["task", "model"], as_index=False)
        .first()
        .rename(columns={"ability_bin": "closest_ability_bin"})
    )

    out = winners[
        ["task", "model", "closest_ability_bin", "mean_d_kl", "n_item_rows"]
    ].copy()
    out = out.sort_values(by=["task", "model"], kind="stable")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    print(f"Wrote {output_csv} ({len(out)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
