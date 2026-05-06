#!/usr/bin/env python3
"""Plot macro benchmark accuracy for a curated \"historic frontier\" model lineup.

Reads ``results/v1/<slug>/summary.csv`` for each configured model, computes the
mean accuracy across the six v1 tasks, and saves a simple line + markers chart
versus a proxy calendar position (release / availability window).

Examples::

    python scripts/analysis/plot_historic_frontier_models.py
    python scripts/analysis/plot_historic_frontier_models.py \\
        --results-root results/v1 --output results/analysis/historic_frontier.png
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ModelSpec:
    """Folder name under ``results/<version>/`` plus plotting metadata."""

    slug: str
    """Directory name, e.g. ``gpt4o``."""

    year: float
    """X-axis position (fractional year OK for ordering)."""

    label: str
    """Legend / annotation text."""


# Proxy timeline for provider-facing checkpoints relevant to the LEVANTE v1
# benchmark. Years are approximate availability milestones for readability on
# the x-axis, not legal release-date claims.
DEFAULT_MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec("clip_base", 2021.0, "CLIP ViT-B/32"),
    ModelSpec("gpt4o", 2024.42, "GPT-4o"),
    ModelSpec("gemini25_pro", 2025.18, "Gemini 2.5 Pro"),
    ModelSpec("gpt41", 2025.33, "GPT-4.1"),
    ModelSpec("gpt53", 2025.58, "GPT-5.3"),
    ModelSpec("gpt55", 2025.72, "GPT-5.5"),
    ModelSpec("gemini3_flash", 2026.02, "Gemini 3 Flash"),
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--results-root",
        type=Path,
        default=REPO_ROOT / "results" / "v1",
        help="Directory containing per-model folders with summary.csv.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "results" / "analysis" / "historic_frontier_models.png",
        help="Output PNG path.",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Figure DPI.",
    )
    return p.parse_args()


def mean_accuracy_from_summary(path: Path) -> tuple[float, list[tuple[str, float]]]:
    """Return (macro_mean, [(task_id, acc), ...])."""
    rows: list[tuple[str, float]] = []
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tid = row["task_id"].strip()
            acc = float(row["accuracy"])
            rows.append((tid, acc))
    if not rows:
        raise ValueError(f"No rows in {path}")
    macro = sum(a for _, a in rows) / len(rows)
    return macro, rows


def main() -> int:
    args = parse_args()
    root: Path = args.results_root.resolve()
    out: Path = args.output.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    points: list[tuple[float, float, str, str]] = []
    missing: list[str] = []

    for spec in sorted(DEFAULT_MODEL_SPECS, key=lambda s: (s.year, s.slug)):
        summary = root / spec.slug / "summary.csv"
        if not summary.is_file():
            missing.append(spec.slug)
            continue
        macro, _ = mean_accuracy_from_summary(summary)
        points.append((spec.year, macro, spec.label, spec.slug))

    if len(points) < 2:
        print(
            "Need at least two models with summary.csv. "
            f"Found {len(points)} under {root}. Missing: {missing}",
            flush=True,
        )
        return 1

    import matplotlib.lines as mlines
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 5.8), layout="constrained")
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    labels = [p[2] for p in points]

    ax.plot(xs, ys, color="#334155", linewidth=1.5, alpha=0.85, zorder=1)
    ax.scatter(xs, ys, s=120, color="#2563eb", edgecolors="#1e293b", linewidths=1.2, zorder=2)

    legend_handles = [
        mlines.Line2D(
            [],
            [],
            color="#2563eb",
            marker="o",
            linestyle="",
            markersize=8,
            label=f"{lab} — {y:.1%}",
        )
        for x, y, lab in sorted(zip(xs, ys, labels), key=lambda t: (t[0], t[1]))
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0,
        fontsize=9,
        framealpha=0.92,
    )

    ax.set_xlabel("Proxy milestone (fractional year)")
    ax.set_ylabel("Macro accuracy (mean over 6 v1 tasks)")
    ax.set_title("LEVANTE v1 — historic frontier checkpoints")
    ax.set_ylim(0.0, 1.02)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.grid(True, alpha=0.35, linestyle="--")
    ax.set_xticks(sorted({round(x, 2) for x in xs}))

    note = (
        "Macro accuracy = unweighted mean of task accuracies in each summary.csv. "
        "CLIP scores ~0 on egma-math (no numeric head), which lowers its macro mean."
    )
    if missing:
        note += f" Skipped (no summary): {', '.join(missing)}."
        print(f"Skipped (no summary.csv): {', '.join(missing)}", flush=True)
    print(
        "Plotting: "
        + ", ".join(f"{p[3]} ({p[1]:.1%})" for p in sorted(points, key=lambda t: t[0])),
        flush=True,
    )
    fig.text(0.02, 0.02, note, fontsize=8, color="#475569", wrap=True)

    fig.savefig(out, dpi=args.dpi)
    plt.close(fig)
    print(f"Wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
