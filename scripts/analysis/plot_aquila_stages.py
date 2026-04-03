#!/usr/bin/env python3
"""Plot Aquila checkpoint stage performance across benchmark tasks."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


DEFAULT_STAGE_FOLDERS: list[tuple[str, str]] = [
    ("aquila_vl_checkpoint_stage2a", "stage2a"),
    ("aquila_vl_checkpoint_stage2b", "stage2b"),
    ("aquila_vl_checkpoint_stage2c", "stage2c"),
    ("aquila_vl_checkpoint_stage3", "stage3"),
    ("aquila_vl_production", "final"),
]

PREFERRED_TASK_ORDER = [
    "egma-math",
    "matrix-reasoning",
    "mental-rotation",
    "theory-of-mind",
    "trog",
    "vocab",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot Aquila checkpoint stage performance from downloaded summary.csv files."
        )
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("scripts/results/aquila-checkpoints/2026-03-29"),
        help=(
            "Directory containing Aquila stage result folders (default: "
            "scripts/results/aquila-checkpoints/2026-03-29)."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path (default: <results-root>/aquila-stage-comparison.png).",
    )
    return parser.parse_args()


def _load_stage_metrics(summary_csv: Path) -> dict[str, float]:
    metrics: dict[str, float] = {}
    with open(summary_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            task_id = (row.get("task_id") or "").strip()
            acc_raw = (row.get("accuracy") or "").strip()
            if not task_id or not acc_raw:
                continue
            try:
                metrics[task_id] = float(acc_raw)
            except ValueError:
                continue
    return metrics


def main() -> int:
    args = parse_args()
    results_root = args.results_root.resolve()
    if not results_root.exists():
        raise FileNotFoundError(f"Results root not found: {results_root}")

    output_path = (
        args.output.resolve()
        if args.output
        else (results_root / "aquila-stage-comparison.png").resolve()
    )

    series: list[tuple[str, dict[str, float]]] = []
    for folder_name, label in DEFAULT_STAGE_FOLDERS:
        summary_csv = results_root / folder_name / "summary.csv"
        if not summary_csv.exists():
            continue
        metrics = _load_stage_metrics(summary_csv)
        if metrics:
            series.append((label, metrics))

    if not series:
        raise RuntimeError(
            "No Aquila stage summaries found. Expected summary.csv files under "
            f"{results_root}."
        )

    tasks = [task for task in PREFERRED_TASK_ORDER if any(task in m for _, m in series)]
    if not tasks:
        raise RuntimeError("No matching benchmark task metrics found in Aquila summaries.")

    fig, (ax_line, ax_bar) = plt.subplots(
        1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [3, 1]}
    )

    x = list(range(len(tasks)))
    for label, metrics in series:
        y = [metrics.get(task, float("nan")) for task in tasks]
        ax_line.plot(x, y, marker="o", linewidth=2, label=label)

    ax_line.set_title("Aquila stage accuracy by task")
    ax_line.set_xlabel("Task")
    ax_line.set_ylabel("Accuracy")
    ax_line.set_xticks(x)
    ax_line.set_xticklabels(tasks, rotation=25, ha="right")
    ax_line.set_ylim(0.0, 1.0)
    ax_line.grid(True, axis="y", alpha=0.3)
    ax_line.legend(loc="lower left", frameon=False)

    labels = [label for label, _ in series]
    means = [
        sum(metrics[task] for task in tasks if task in metrics)
        / max(1, sum(1 for task in tasks if task in metrics))
        for _, metrics in series
    ]
    colors = ["#8ecae6", "#219ebc", "#126782", "#023047", "#fb8500"][: len(labels)]
    ax_bar.bar(labels, means, color=colors)
    ax_bar.set_title("Mean accuracy")
    ax_bar.set_ylabel("Accuracy")
    ax_bar.set_ylim(0.0, 1.0)
    ax_bar.grid(True, axis="y", alpha=0.3)
    for i, mean in enumerate(means):
        ax_bar.text(i, mean + 0.01, f"{mean:.3f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
