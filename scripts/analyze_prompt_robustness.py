#!/usr/bin/env python3
"""Analyze prompt robustness sweep results.

Reads sweep_results.csv and produces:
1. TF × OF heatmaps per task (mean accuracy across models)
2. Best prompt selection per task using maximin criterion
3. Family-specific recommendations where interactions exist
4. Summary table for paper inclusion

Usage:
    python scripts/analyze_prompt_robustness.py \\
        --results results/prompt_robustness/sweep_results.csv \\
        --output-dir results/prompt_robustness/analysis
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def load_results(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows from {path}")
    print(f"Models: {sorted(df['model'].unique())}")
    print(f"Tasks: {sorted(df['task'].unique())}")
    print(f"Task framings: {sorted(df['task_framing'].unique())}")
    print(f"Output formats: {sorted(df['output_format'].unique())}")
    return df


def analyze_task(df_task: pd.DataFrame, task_id: str, output_dir: Path):
    """Analyze one task: heatmaps, rankings, recommendations."""
    print(f"\n{'='*60}")
    print(f"Task: {task_id}")
    print(f"{'='*60}")

    models = sorted(df_task["model"].unique())
    tf_names = sorted(df_task["task_framing"].unique())
    of_names = sorted(df_task["output_format"].unique())

    # ── 1. TF × OF accuracy matrix (averaged across models) ──────────
    print("\n--- Mean accuracy across models (TF × OF) ---")
    pivot_mean = df_task.pivot_table(
        index="task_framing",
        columns="output_format",
        values="accuracy",
        aggfunc="mean",
    )
    print(pivot_mean.round(3).to_string())

    # ── 2. Min accuracy across models (worst-case robustness) ─────────
    print("\n--- Min accuracy across models (TF × OF) ---")
    pivot_min = df_task.pivot_table(
        index="task_framing",
        columns="output_format",
        values="accuracy",
        aggfunc="min",
    )
    print(pivot_min.round(3).to_string())

    # ── 3. Std across models (sensitivity) ────────────────────────────
    print("\n--- Std accuracy across models (TF × OF) ---")
    pivot_std = df_task.pivot_table(
        index="task_framing",
        columns="output_format",
        values="accuracy",
        aggfunc="std",
    )
    print(pivot_std.round(3).to_string())

    # ── 4. Parse rate matrix ──────────────────────────────────────────
    print("\n--- Mean parse rate (TF × OF) ---")
    pivot_parse = df_task.pivot_table(
        index="task_framing",
        columns="output_format",
        values="parse_rate",
        aggfunc="mean",
    )
    print(pivot_parse.round(3).to_string())

    # ── 5. Bias analysis ──────────────────────────────────────────────
    print("\n--- Mean bias ratio (TF × OF, closer to 0.5 = less biased) ---")
    # Compute distance from ideal (1/n_options)
    n_options = len(df_task.iloc[0].get("correct_label", "AB"))  # rough
    pivot_bias = df_task.pivot_table(
        index="task_framing",
        columns="output_format",
        values="bias_ratio",
        aggfunc="mean",
    )
    print(pivot_bias.round(3).to_string())

    # ── 6. Maximin selection ──────────────────────────────────────────
    print("\n--- Maximin criterion (maximize worst-case accuracy) ---")
    cells = []
    for tf in tf_names:
        for of in of_names:
            mask = (df_task["task_framing"] == tf) & (df_task["output_format"] == of)
            sub = df_task[mask]
            if sub.empty:
                continue
            min_acc = sub["accuracy"].min()
            mean_acc = sub["accuracy"].mean()
            mean_parse = sub["parse_rate"].mean()
            mean_bias = sub["bias_ratio"].mean()
            cells.append({
                "task_framing": tf,
                "output_format": of,
                "min_acc": min_acc,
                "mean_acc": mean_acc,
                "mean_parse": mean_parse,
                "mean_bias": mean_bias,
            })

    cells_df = pd.DataFrame(cells).sort_values("min_acc", ascending=False)
    print(cells_df.head(10).to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    best = cells_df.iloc[0]
    print(f"\n>>> Best (maximin): {best['task_framing']} × {best['output_format']}")
    print(f"    min_acc={best['min_acc']:.3f}, mean_acc={best['mean_acc']:.3f}, "
          f"parse={best['mean_parse']:.3f}, bias={best['mean_bias']:.3f}")

    # ── 7. Family-specific analysis ───────────────────────────────────
    families = sorted(df_task["model_family"].unique())
    if len(families) > 1:
        print(f"\n--- Family-specific best (per family maximin) ---")
        for family in families:
            sub = df_task[df_task["model_family"] == family]
            fam_cells = []
            for tf in tf_names:
                for of in of_names:
                    mask = (sub["task_framing"] == tf) & (sub["output_format"] == of)
                    ss = sub[mask]
                    if ss.empty:
                        continue
                    fam_cells.append({
                        "family": family,
                        "task_framing": tf,
                        "output_format": of,
                        "min_acc": ss["accuracy"].min(),
                        "mean_acc": ss["accuracy"].mean(),
                        "mean_parse": ss["parse_rate"].mean(),
                    })
            if fam_cells:
                fam_df = pd.DataFrame(fam_cells).sort_values("mean_acc", ascending=False)
                top = fam_df.iloc[0]
                print(f"  {family}: {top['task_framing']} × {top['output_format']} "
                      f"(mean_acc={top['mean_acc']:.3f}, parse={top['mean_parse']:.3f})")

    # ── 8. TF effect (marginalized over OF) ───────────────────────────
    print("\n--- Task Framing effect (marginalized over OF) ---")
    tf_summary = df_task.groupby("task_framing").agg(
        mean_acc=("accuracy", "mean"),
        min_acc=("accuracy", "min"),
        std_acc=("accuracy", "std"),
        mean_parse=("parse_rate", "mean"),
    ).sort_values("mean_acc", ascending=False)
    print(tf_summary.round(3).to_string())

    # ── 9. OF effect (marginalized over TF) ───────────────────────────
    print("\n--- Output Format effect (marginalized over TF) ---")
    of_summary = df_task.groupby("output_format").agg(
        mean_acc=("accuracy", "mean"),
        min_acc=("accuracy", "min"),
        std_acc=("accuracy", "std"),
        mean_parse=("parse_rate", "mean"),
    ).sort_values("mean_acc", ascending=False)
    print(of_summary.round(3).to_string())

    # ── 10. ANOVA: TF × OF × model_family interaction ────────────────
    if len(families) > 1 and len(tf_names) > 1 and len(of_names) > 1:
        print("\n--- Interaction test: does TF or OF depend on model family? ---")
        # Simple approach: compare TF ranking across families
        for family in families:
            sub = df_task[df_task["model_family"] == family]
            tf_rank = sub.groupby("task_framing")["accuracy"].mean().sort_values(ascending=False)
            print(f"  {family} TF ranking: {' > '.join(tf_rank.index)}")

    # Save task analysis
    task_dir = output_dir / task_id
    task_dir.mkdir(parents=True, exist_ok=True)

    pivot_mean.to_csv(task_dir / "tf_of_mean_accuracy.csv")
    pivot_min.to_csv(task_dir / "tf_of_min_accuracy.csv")
    cells_df.to_csv(task_dir / "maximin_ranking.csv", index=False)

    return {
        "task": task_id,
        "best_tf": best["task_framing"],
        "best_of": best["output_format"],
        "best_min_acc": best["min_acc"],
        "best_mean_acc": best["mean_acc"],
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze Prompt Robustness Sweep")
    parser.add_argument(
        "--results", required=True,
        help="Path to sweep_results.csv",
    )
    parser.add_argument(
        "--output-dir", default="results/prompt_robustness/analysis",
        help="Output directory for analysis artifacts",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_results(args.results)

    recommendations = []
    for task_id in sorted(df["task"].unique()):
        df_task = df[df["task"] == task_id]
        rec = analyze_task(df_task, task_id, output_dir)
        recommendations.append(rec)

    # ── Global summary ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("GLOBAL RECOMMENDATIONS")
    print("=" * 60)
    rec_df = pd.DataFrame(recommendations)
    print(rec_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    rec_df.to_csv(output_dir / "recommendations.csv", index=False)
    print(f"\nSaved to {output_dir}")

    # Check if a shared OF works across tasks
    print("\n--- Shared Output Format analysis ---")
    of_by_task = df.groupby(["output_format", "task"])["accuracy"].mean().unstack()
    of_means = of_by_task.mean(axis=1).sort_values(ascending=False)
    of_mins = of_by_task.min(axis=1).sort_values(ascending=False)
    print("  Mean accuracy per OF (across all tasks + models):")
    for of_name in of_means.index:
        print(f"    {of_name}: mean={of_means[of_name]:.3f}, min={of_mins[of_name]:.3f}")

    best_shared_of = of_mins.idxmax()
    print(f"\n  >>> Best shared OF (maximin across tasks): {best_shared_of}")


if __name__ == "__main__":
    main()
