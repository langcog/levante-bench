"""Visualize model performance across tasks as a heatmap table.

Reads summary.csv files from results/<model>/<version>/ and produces
a models x tasks heatmap with accuracy values annotated.

Usage:
    python scripts/plot_results.py
    python scripts/plot_results.py --results-dir results --version 2026-03-24
    python scripts/plot_results.py --output results/heatmap.png
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def load_summaries(results_dir: Path, version: str | None = None) -> pd.DataFrame:
    """Load all summary.csv files into a single DataFrame.

    Supports both:
        results/<version>/<model_label>/summary.csv  (new)
        results/<model>/<version>/summary.csv         (old)

    Returns DataFrame with columns: model, task_id, accuracy
    """
    rows = []

    for child in sorted(results_dir.iterdir()):
        if not child.is_dir():
            continue
        for sub in sorted(child.iterdir()):
            if not sub.is_dir():
                continue
            summary = sub / "summary.csv"
            if not summary.exists():
                continue
            # Determine which layout: check if parent or child is the version
            if version and child.name == version:
                model_name = sub.name
            elif version and sub.name == version:
                model_name = child.name
            else:
                # Guess: if child looks like a date, it's version/model
                model_name = sub.name if "-" in child.name and len(child.name) == 10 else child.name

            df = pd.read_csv(summary)
            df["model"] = model_name
            rows.append(df)

    if not rows:
        return pd.DataFrame(columns=["model", "task_id", "accuracy"])
    return pd.concat(rows, ignore_index=True)


def plot_heatmap(df: pd.DataFrame, output_path: Path | None = None) -> None:
    """Plot a models x tasks heatmap with accuracy values."""
    pivot = df.pivot_table(index="model", columns="task_id", values="accuracy")

    # Sort tasks alphabetically, models by mean accuracy (best on top)
    pivot = pivot[sorted(pivot.columns)]
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 1.5), max(3, len(pivot) * 0.8)))

    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Accuracy"},
    )

    ax.set_title("Model Performance Across Tasks")
    ax.set_xlabel("")
    ax.set_ylabel("")

    # Add chance level line info
    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    else:
        plt.show()

    plt.close(fig)


def print_table(df: pd.DataFrame) -> None:
    """Print a simple text table of results."""
    pivot = df.pivot_table(index="model", columns="task_id", values="accuracy")
    pivot = pivot[sorted(pivot.columns)]
    pivot["mean"] = pivot.mean(axis=1)
    pivot = pivot.sort_values("mean", ascending=False)

    # Header
    tasks = [c for c in pivot.columns if c != "mean"]
    header = f"{'model':<20}" + "".join(f"{t:>15}" for t in tasks) + f"{'mean':>10}"
    print(header)
    print("-" * len(header))

    for model, row in pivot.iterrows():
        line = f"{model:<20}"
        for t in tasks:
            v = row.get(t, float("nan"))
            line += f"{v:>15.4f}" if not np.isnan(v) else f"{'—':>15}"
        line += f"{row['mean']:>10.4f}"
        print(line)


def main():
    parser = argparse.ArgumentParser(description="Plot model x task accuracy heatmap")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--version", type=str, default=None)
    parser.add_argument("--output", type=str, default=None, help="Save plot to file (e.g. results/heatmap.png)")
    parser.add_argument("--no-plot", action="store_true", help="Only print text table, no plot")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    df = load_summaries(results_dir, args.version)

    if df.empty:
        print("No summary.csv files found.", file=sys.stderr)
        return 1

    print_table(df)

    if not args.no_plot:
        if args.output:
            output_path = Path(args.output)
        elif args.version:
            output_path = results_dir / args.version / "heatmap.png"
        else:
            # Use first version dir found
            versions = [d.name for d in sorted(results_dir.iterdir()) if d.is_dir()]
            output_path = results_dir / versions[0] / "heatmap.png" if versions else results_dir / "heatmap.png"
        plot_heatmap(df, output_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
