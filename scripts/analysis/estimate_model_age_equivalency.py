#!/usr/bin/env python3
"""Estimate task-level model age-equivalency from KL-by-ability outputs."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate model age-equivalency per task by mapping KL-preferred "
            "ability bins to child age distributions."
        )
    )
    parser.add_argument("--version", default="v1", help="Data/results version.")
    parser.add_argument(
        "--project-root",
        default=".",
        help="Project root path (default: current directory).",
    )
    parser.add_argument(
        "--comparison-dir",
        default="results/comparison",
        help="Directory containing *_d_kl.csv files.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.25,
        help=(
            "Soft-match temperature for KL-to-weight conversion "
            "(weights proportional to exp(-(kl-min_kl)/temperature))."
        ),
    )
    parser.add_argument(
        "--output-csv",
        default="results/comparison/model_age_equivalency.csv",
        help="Output CSV path for model age-equivalency summary.",
    )
    parser.add_argument(
        "--output-bin-map-csv",
        default="results/comparison/task_ability_bin_age_map.csv",
        help="Output CSV path for per-task ability-bin age map.",
    )
    return parser.parse_args()


def ability_bin_start(bin_label: str) -> int:
    try:
        return int(str(bin_label).split("_", 1)[0])
    except Exception:
        return 10**9


def soft_weights(kl_values: np.ndarray, temperature: float) -> np.ndarray:
    if kl_values.size == 0:
        return kl_values
    t = max(float(temperature), 1e-6)
    shifted = kl_values - np.nanmin(kl_values)
    scores = np.exp(-shifted / t)
    denom = scores.sum()
    if denom <= 0:
        return np.ones_like(scores) / max(len(scores), 1)
    return scores / denom


def normalized_entropy(weights: np.ndarray) -> float:
    if weights.size <= 1:
        return 0.0
    w = np.clip(weights, 1e-12, 1.0)
    ent = -float(np.sum(w * np.log(w)))
    return ent / math.log(len(weights))


def build_task_bin_age_map(project_root: Path, version: str) -> pd.DataFrame:
    trials_path = project_root / "data" / "responses" / version / "trials.csv"
    irt_dir = project_root / "data" / "responses" / version / "irt_models"

    trials = pd.read_csv(trials_path, usecols=["task_id", "run_id", "age"])
    trials = trials.dropna(subset=["task_id", "run_id", "age"]).copy()

    out_rows: list[pd.DataFrame] = []
    for ability_file in sorted(irt_dir.glob("*_ability_scores.csv")):
        task_id = ability_file.name.replace("_ability_scores.csv", "")
        ability = pd.read_csv(ability_file, usecols=["run_id", "ability"])
        ability = ability.dropna(subset=["run_id", "ability"]).copy()
        if ability.empty:
            continue

        task_trials = trials[trials["task_id"] == task_id].copy()
        if task_trials.empty:
            continue

        merged = task_trials.merge(ability, on="run_id", how="inner")
        if merged.empty:
            continue

        merged["bin_start"] = np.floor(merged["ability"]).astype(int)
        merged["ability_bin"] = merged["bin_start"].astype(str) + "_" + (
            merged["bin_start"] + 1
        ).astype(str)

        grouped = (
            merged.groupby("ability_bin", as_index=False)
            .agg(
                n_children=("run_id", "nunique"),
                age_mean=("age", "mean"),
                age_median=("age", "median"),
                age_min=("age", "min"),
                age_max=("age", "max"),
            )
            .copy()
        )
        grouped["task"] = task_id
        grouped["bin_start"] = grouped["ability_bin"].map(ability_bin_start)
        out_rows.append(grouped)

    if not out_rows:
        raise RuntimeError("Could not build any task ability-bin age mappings.")

    out = pd.concat(out_rows, ignore_index=True)
    out = out[
        [
            "task",
            "ability_bin",
            "bin_start",
            "n_children",
            "age_mean",
            "age_median",
            "age_min",
            "age_max",
        ]
    ].sort_values(["task", "bin_start", "ability_bin"])
    return out


def read_kl_rows(comparison_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for p in sorted(comparison_dir.glob("*_d_kl.csv")):
        df = pd.read_csv(p)
        required = {"task", "model", "ability_bin", "D_KL"}
        if not required.issubset(df.columns):
            continue
        frames.append(df[["task", "model", "ability_bin", "D_KL"]].copy())
    if not frames:
        raise RuntimeError("No KL CSV files found with required columns.")
    all_kl = pd.concat(frames, ignore_index=True)
    return all_kl


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    comparison_dir = project_root / args.comparison_dir
    out_csv = project_root / args.output_csv
    out_bin_map_csv = project_root / args.output_bin_map_csv

    bin_map = build_task_bin_age_map(project_root=project_root, version=args.version)
    out_bin_map_csv.parent.mkdir(parents=True, exist_ok=True)
    bin_map.to_csv(out_bin_map_csv, index=False)

    kl_rows = read_kl_rows(comparison_dir)
    kl_bin = (
        kl_rows.groupby(["task", "model", "ability_bin"], as_index=False)["D_KL"]
        .mean()
        .rename(columns={"D_KL": "mean_d_kl"})
    )

    results: list[dict] = []
    for (task, model), grp in kl_bin.groupby(["task", "model"], sort=True):
        merged = grp.merge(
            bin_map[bin_map["task"] == task][
                ["ability_bin", "age_mean", "age_median", "n_children", "bin_start"]
            ],
            on="ability_bin",
            how="inner",
        )
        if merged.empty:
            continue

        merged = merged.sort_values(["mean_d_kl", "bin_start", "ability_bin"])
        hard = merged.iloc[0]

        kls = merged["mean_d_kl"].to_numpy(dtype=float)
        ws = soft_weights(kls, temperature=args.temperature)
        age_means = merged["age_mean"].to_numpy(dtype=float)
        age_medians = merged["age_median"].to_numpy(dtype=float)

        soft_age_mean = float(np.sum(ws * age_means))
        soft_age_median = float(np.sum(ws * age_medians))
        ent = normalized_entropy(ws)
        confidence = 1.0 - ent

        results.append(
            {
                "task": task,
                "model": model,
                "n_bins_used": int(len(merged)),
                "closest_ability_bin": str(hard["ability_bin"]),
                "closest_bin_mean_d_kl": float(hard["mean_d_kl"]),
                "closest_bin_age_mean": float(hard["age_mean"]),
                "closest_bin_age_median": float(hard["age_median"]),
                "soft_age_eq_mean": soft_age_mean,
                "soft_age_eq_median": soft_age_median,
                "soft_match_confidence": confidence,
                "soft_match_entropy_norm": ent,
                "temperature": float(args.temperature),
            }
        )

    out = pd.DataFrame(results).sort_values(["task", "model"], kind="stable")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} ({len(out)} rows)")
    print(f"Wrote {out_bin_map_csv} ({len(bin_map)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
