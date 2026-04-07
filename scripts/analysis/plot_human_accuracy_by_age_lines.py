#!/usr/bin/env python3
"""Plot human benchmark accuracy by age bin (task on x-axis, accuracy on y-axis)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


TASK_ORDER = [
    "egma-math",
    "matrix-reasoning",
    "mental-rotation",
    "theory-of-mind",
    "trog",
    "vocab",
]


def _infer_language(df: pd.DataFrame) -> pd.Series:
    # Prefer explicit language-like columns if present.
    for col in ("language", "prompt_language", "lang", "locale"):
        if col in df.columns:
            out = df[col].astype(str).str.strip().str.lower()
            return out.replace({"": "unknown", "nan": "unknown"})

    # Fallback: infer from dataset/site naming conventions in pilot exports.
    empty = pd.Series([""] * len(df), index=df.index, dtype="object")
    dataset = df["dataset"].astype(str).str.lower() if "dataset" in df.columns else empty
    site = df["site"].astype(str).str.lower() if "site" in df.columns else empty
    merged = (dataset + " " + site).str.strip()

    lang = pd.Series(["unknown"] * len(df), index=df.index, dtype="object")
    lang = lang.mask(merged.str.contains(r"\bde\b|german|mpieva", regex=True), "de")
    lang = lang.mask(merged.str.contains(r"\bco\b|spanish|uniandes", regex=True), "es")
    lang = lang.mask(merged.str.contains(r"\bca\b|western|english", regex=True), "en")
    return lang


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build and plot human task accuracy curves by age bin.",
    )
    p.add_argument(
        "--trials-csv",
        type=Path,
        default=Path("data/responses/v1/trials.csv"),
        help="Path to trials.csv (default: data/responses/v1/trials.csv).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("results/human-accuracy-by-age-lines.png"),
        help="Output chart PNG path (default: results/human-accuracy-by-age-lines.png).",
    )
    p.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/human-accuracy-by-age-lines.csv"),
        help="Output aggregated CSV path (default: results/human-accuracy-by-age-lines.csv).",
    )
    p.add_argument(
        "--min-age",
        type=float,
        default=3.0,
        help="Minimum age to include (default: 3).",
    )
    p.add_argument(
        "--max-age",
        type=float,
        default=12.0,
        help="Maximum age to include (default: 12).",
    )
    p.add_argument(
        "--bin-width",
        type=float,
        default=2.0,
        help="Age bin width in years (default: 2).",
    )
    p.add_argument(
        "--min-samples",
        type=int,
        default=20,
        help="Minimum rows required per age_bin/task to include a point (default: 20).",
    )
    return p.parse_args()


def _to_bool(series: pd.Series) -> pd.Series:
    normalized = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .replace({"true": "1", "false": "0"})
    )
    numeric = pd.to_numeric(normalized, errors="coerce")
    return numeric == 1


def _build_age_bins(df: pd.DataFrame, min_age: float, max_age: float, bin_width: float) -> pd.DataFrame:
    if bin_width <= 0:
        raise ValueError("--bin-width must be > 0")
    edges = []
    x = float(min_age)
    max_edge = float(max_age) + float(bin_width)
    while x <= max_edge + 1e-9:
        edges.append(x)
        x += bin_width
    if len(edges) < 2:
        edges = [float(min_age), float(max_age) + float(bin_width)]
    labels = []
    for i in range(len(edges) - 1):
        lo = int(edges[i])
        hi = int(edges[i + 1] - 1e-6)
        if lo == hi:
            labels.append(f"{lo}")
        else:
            labels.append(f"{lo}-{hi}")
    out = df.copy()
    out["age_bin"] = pd.cut(
        out["age"],
        bins=edges,
        labels=labels,
        include_lowest=True,
        right=False,
    )
    return out


def _age_bin_sort_key(label: str) -> tuple[int, int]:
    lo, _, hi = label.partition("-")
    try:
        return (int(lo), int(hi or lo))
    except ValueError:
        return (10**9, 10**9)


def main() -> int:
    args = parse_args()
    trials_csv = args.trials_csv.resolve()
    if not trials_csv.exists():
        raise FileNotFoundError(f"Trials CSV not found: {trials_csv}")

    df = pd.read_csv(trials_csv, low_memory=False)
    required_cols = {"task_id", "age", "correct"}
    missing = required_cols - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing required columns in trials.csv: {sorted(missing)}")

    df = df[df["task_id"].isin(TASK_ORDER)].copy()
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df = df[df["age"].notna()].copy()
    df = df[(df["age"] >= args.min_age) & (df["age"] <= args.max_age)].copy()
    if df.empty:
        raise RuntimeError("No rows left after filtering by task + age range.")

    df["is_correct"] = _to_bool(df["correct"])
    df = _build_age_bins(df, args.min_age, args.max_age, args.bin_width)
    df = df[df["age_bin"].notna()].copy()
    if df.empty:
        raise RuntimeError("No rows mapped into age bins.")

    df["language"] = _infer_language(df)
    agg = (
        df.groupby(["age_bin", "task_id", "language"], observed=True)
        .agg(
            n=("is_correct", "size"),
            accuracy=("is_correct", "mean"),
        )
        .reset_index()
    )
    agg = agg[agg["n"] >= args.min_samples].copy()
    if agg.empty:
        raise RuntimeError(
            "No age_bin/task groups met --min-samples. Try lowering --min-samples."
        )

    age_bins = sorted([str(x) for x in agg["age_bin"].unique()], key=_age_bin_sort_key)
    x = list(range(len(TASK_ORDER)))

    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    languages = sorted([str(x) for x in agg["language"].unique()])
    for language in languages:
        for age_bin in age_bins:
            row = agg[
                (agg["age_bin"].astype(str) == age_bin)
                & (agg["language"].astype(str) == language)
            ].set_index("task_id")
            if row.empty:
                continue
            y = [row.loc[t, "accuracy"] if t in row.index else float("nan") for t in TASK_ORDER]
            ax.plot(
                x,
                y,
                marker="o",
                linewidth=2.0,
                markersize=5.5,
                label=f"{age_bin}-{language}",
            )

    ax.set_title("Human Accuracy by Task and Age Bin")
    ax.set_xlabel("Task")
    ax.set_ylabel("Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(TASK_ORDER, rotation=25, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(title="Age (years)", loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout()

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    output_csv = args.output_csv.resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    agg = agg.sort_values(["age_bin", "task_id"]).copy()
    agg["age_bin"] = agg["age_bin"].astype(str)
    agg.to_csv(output_csv, index=False)

    print(f"Saved chart: {output_path}")
    print(f"Saved data:  {output_csv}")
    print(f"Age bins:    {', '.join(age_bins)}")
    print(f"Languages:   {', '.join(languages)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
