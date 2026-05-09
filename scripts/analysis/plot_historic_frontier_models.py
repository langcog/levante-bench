#!/usr/bin/env python3
"""Plot benchmark accuracy for a curated \"historic frontier\" model lineup.

Reads ``results/v1/<slug>/summary.csv`` for each configured model versus a proxy
calendar position. Writes:

1. Macro mean accuracy (single series).
2. One line per v1 task (six series).

Examples::

    python scripts/analysis/plot_historic_frontier_models.py
    python scripts/analysis/plot_historic_frontier_models.py \\
        --results-root results/v1 \\
        --output results/analysis/historic_frontier_models.png \\
        --output-by-task results/analysis/historic_frontier_by_task.png
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

TASK_ORDER: tuple[str, ...] = (
    "egma-math",
    "matrix-reasoning",
    "mental-rotation",
    "theory-of-mind",
    "trog",
    "vocab",
)

TASK_LEGEND_LABELS: dict[str, str] = {
    "egma-math": "Egma math",
    "matrix-reasoning": "Matrix reasoning",
    "mental-rotation": "Mental rotation",
    "theory-of-mind": "Theory of mind",
    "trog": "TROG",
    "vocab": "Vocab",
}

TASK_COLORS: tuple[str, ...] = (
    "#dc2626",
    "#ea580c",
    "#ca8a04",
    "#16a34a",
    "#2563eb",
    "#7c3aed",
)


@dataclass(frozen=True)
class ModelSpec:
    """Folder name under ``results/<version>/`` plus plotting metadata."""

    slug: str
    year: float
    label: str


# Proxy timeline for provider-facing checkpoints. Fractional years are for
# ordering only.
DEFAULT_MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec("clip_base", 2021.0, "CLIP ViT-B/32"),
    ModelSpec("llava15_13b", 2023.92, "LLaVA 1.5 13B"),
    ModelSpec("gpt4o", 2024.42, "GPT-4o"),
    ModelSpec("gemini25_pro", 2025.18, "Gemini 2.5 Pro"),
    ModelSpec("gpt41", 2025.33, "GPT-4.1"),
    ModelSpec("gpt53", 2025.58, "GPT-5.3"),
    ModelSpec("gpt55", 2026.15, "GPT-5.5"),
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
        help="Output PNG for macro-mean chart.",
    )
    p.add_argument(
        "--output-by-task",
        type=Path,
        default=REPO_ROOT / "results" / "analysis" / "historic_frontier_by_task.png",
        help="Output PNG for per-task lines chart.",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Figure DPI.",
    )
    return p.parse_args()


def load_summary_task_accuracies(path: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            tid = row["task_id"].strip()
            out[tid] = float(row["accuracy"])
    missing = set(TASK_ORDER) - set(out)
    extra = set(out) - set(TASK_ORDER)
    if missing:
        raise ValueError(f"{path}: missing tasks {sorted(missing)}")
    if extra:
        raise ValueError(f"{path}: unexpected tasks {sorted(extra)}")
    return {tid: out[tid] for tid in TASK_ORDER}


def collect_series(
    root: Path,
    specs: tuple[ModelSpec, ...],
) -> tuple[list[tuple[float, str, str, dict[str, float]]], list[str]]:
    """Return (series rows ordered by year), missing_slugs."""
    rows: list[tuple[float, str, str, dict[str, float]]] = []
    missing: list[str] = []
    for spec in sorted(specs, key=lambda s: (s.year, s.slug)):
        summary = root / spec.slug / "summary.csv"
        if not summary.is_file():
            # Some hosted/API runs are stored under a baseline subfolder.
            summary = root / spec.slug / "baseline" / "summary.csv"
        if not summary.is_file():
            missing.append(spec.slug)
            continue
        acc = load_summary_task_accuracies(summary)
        rows.append((spec.year, spec.slug, spec.label, acc))
    return rows, missing


def plot_macro(
    rows: list[tuple[float, str, str, dict[str, float]]],
    *,
    missing: list[str],
    out: Path,
    dpi: int,
) -> None:
    import matplotlib.lines as mlines
    import matplotlib.pyplot as plt

    points = [(r[0], sum(r[3].values()) / len(TASK_ORDER), r[2], r[1]) for r in rows]
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    labels = [p[2] for p in points]

    fig, ax = plt.subplots(figsize=(11, 5.8), layout="constrained")
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
    ax.set_title("LEVANTE v1 — historic frontier checkpoints (macro)")
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
    fig.text(0.02, 0.02, note, fontsize=8, color="#475569", wrap=True)

    fig.savefig(out, dpi=dpi)
    plt.close(fig)


def plot_by_task(
    rows: list[tuple[float, str, str, dict[str, float]]],
    *,
    missing: list[str],
    out: Path,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt

    xs = [r[0] for r in rows]
    fig, ax = plt.subplots(figsize=(12, 6.2), layout="constrained")

    for i, tid in enumerate(TASK_ORDER):
        ys = [r[3][tid] for r in rows]
        color = TASK_COLORS[i % len(TASK_COLORS)]
        ax.plot(
            xs,
            ys,
            color=color,
            linewidth=2.0,
            marker="o",
            markersize=6,
            label=TASK_LEGEND_LABELS[tid],
            alpha=0.92,
        )

    ax.set_xlabel("Proxy milestone (fractional year)")
    ax.set_ylabel("Task accuracy")
    ax.set_title("LEVANTE v1 — historic frontier checkpoints (by task)")
    ax.set_ylim(0.0, 1.02)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.grid(True, alpha=0.35, linestyle="--")
    ax.set_xticks(sorted({round(x, 2) for x in xs}))
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=9, framealpha=0.92)

    note = (
        "Each line tracks one task's accuracy from summary.csv across checkpoints "
        "(same x-axis proxies as macro chart)."
    )
    if missing:
        note += f" Skipped (no summary): {', '.join(missing)}."
    fig.text(0.02, 0.02, note, fontsize=8, color="#475569", wrap=True)

    fig.savefig(out, dpi=dpi)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    root: Path = args.results_root.resolve()
    out_macro: Path = args.output.resolve()
    out_tasks: Path = args.output_by_task.resolve()
    out_macro.parent.mkdir(parents=True, exist_ok=True)
    out_tasks.parent.mkdir(parents=True, exist_ok=True)

    rows, missing = collect_series(root, DEFAULT_MODEL_SPECS)

    if len(rows) < 2:
        print(
            "Need at least two models with summary.csv. "
            f"Found {len(rows)} under {root}. Missing: {missing}",
            flush=True,
        )
        return 1

    if missing:
        print(f"Skipped (no summary.csv): {', '.join(missing)}", flush=True)

    macros = [sum(r[3].values()) / len(TASK_ORDER) for r in rows]
    print(
        "Plotting: "
        + ", ".join(f"{r[1]} ({m:.1%})" for r, m in zip(rows, macros)),
        flush=True,
    )

    plot_macro(rows, missing=missing, out=out_macro, dpi=args.dpi)
    print(f"Wrote {out_macro}", flush=True)

    plot_by_task(rows, missing=missing, out=out_tasks, dpi=args.dpi)
    print(f"Wrote {out_tasks}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
