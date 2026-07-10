#!/usr/bin/env python3
"""Plot forced-binary improvements for paper models across tasks.

Reads consolidated forced-binary summaries from `results/paper_models_forced_binary`,
aligns each model/task with baseline `results/v1/<model>/summary.csv`, and writes:
1) long-form comparison CSV
2) summary-by-model CSV
3) summary-by-task CSV
4) three-panel PNG figure
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TASK_ORDER = [
    "egma-math",
    "matrix-reasoning",
    "mental-rotation",
    "theory-of-mind",
    "trog",
    "vocab",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--forced-roots",
        type=str,
        default=",".join(
            [
                "results/paper_models_forced_binary",
                "results/forced_binary_other_tasks",
                "results/vocab_binary_ablation",
                "results/trog_matrix_binary_ablation",
                "results/cogvlm_vocab_ablation/binary_default_r2/v1",
            ]
        ),
        help=(
            "Comma-separated list of directories containing per-model forced-binary "
            "summary.csv files. Earlier roots take precedence for overlapping model/task pairs."
        ),
    )
    p.add_argument(
        "--baseline-root",
        type=Path,
        default=REPO_ROOT / "results" / "v1",
        help="Directory containing baseline per-model summary.csv files.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "analysis",
        help="Directory for CSV and PNG outputs.",
    )
    p.add_argument(
        "--prefix",
        type=str,
        default="paper_forced_binary",
        help="Filename prefix for outputs.",
    )
    p.add_argument(
        "--tasks",
        type=str,
        default=",".join(DEFAULT_TASK_ORDER),
        help="Comma-separated task order to display.",
    )
    p.add_argument("--dpi", type=int, default=160, help="Figure DPI.")
    p.add_argument(
        "--complete-suffix",
        type=str,
        default="complete_only",
        help="Suffix for complete-task-only outputs.",
    )
    return p.parse_args()


def read_summary(path: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    if not path.exists():
        return out
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            task = (row.get("task_id") or "").strip()
            acc = row.get("accuracy")
            if not task or acc is None:
                continue
            try:
                out[task] = float(acc)
            except ValueError:
                continue
    return out


def resolve_baseline_summary_path(baseline_root: Path, model_slug: str) -> Path | None:
    candidates = [
        baseline_root / model_slug / "summary.csv",
        baseline_root / model_slug / "baseline" / "summary.csv",
        REPO_ROOT / "results" / "v1_additional_models" / model_slug / "summary.csv",
        REPO_ROOT / "results" / "v1_additional_models" / model_slug / "baseline" / "summary.csv",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def collect_rows(forced_roots: list[Path], baseline_root: Path) -> list[dict[str, str | float]]:
    rows: list[dict[str, str | float]] = []
    # Merge model/task forced accuracies across roots with first-root precedence.
    forced_by_model: dict[str, dict[str, float]] = {}
    for forced_root in forced_roots:
        if not forced_root.exists():
            continue
        for model_dir in sorted(p for p in forced_root.iterdir() if p.is_dir()):
            forced_summary_path = model_dir / "summary.csv"
            forced = read_summary(forced_summary_path)
            if not forced:
                continue
            bucket = forced_by_model.setdefault(model_dir.name, {})
            for task, acc in forced.items():
                if task not in bucket:
                    bucket[task] = acc

    for model_name, forced in sorted(forced_by_model.items()):
        baseline_path = resolve_baseline_summary_path(baseline_root, model_name)
        if baseline_path is None:
            continue
        baseline = read_summary(baseline_path)
        for task, forced_acc in forced.items():
            if task not in baseline:
                continue
            base_acc = baseline[task]
            rows.append(
                {
                    "model": model_name,
                    "task": task,
                    "baseline_accuracy": base_acc,
                    "forced_binary_accuracy": forced_acc,
                    "delta_pp": (forced_acc - base_acc) * 100.0,
                }
            )
    rows.sort(key=lambda r: (str(r["model"]), str(r["task"])))
    return rows


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str | float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def build_outputs(
    rows: list[dict[str, str | float]],
    tasks_requested: list[str],
    output_dir: Path,
    prefix: str,
    dpi: int,
) -> bool:
    if not rows:
        return False

    task_present = sorted({str(r["task"]) for r in rows})
    tasks = [t for t in tasks_requested if t in task_present] + [t for t in task_present if t not in tasks_requested]
    models = sorted({str(r["model"]) for r in rows})
    if not tasks or not models:
        return False

    long_csv = output_dir / f"{prefix}_long.csv"
    write_csv(
        long_csv,
        ["model", "task", "baseline_accuracy", "forced_binary_accuracy", "delta_pp"],
        rows,
    )

    model_summary_rows: list[dict[str, str | float]] = []
    for m in models:
        vals = [float(r["delta_pp"]) for r in rows if str(r["model"]) == m]
        model_summary_rows.append(
            {
                "model": m,
                "mean_delta_pp": float(np.mean(vals)),
                "num_tasks": len(vals),
            }
        )
    model_csv = output_dir / f"{prefix}_by_model.csv"
    write_csv(model_csv, ["model", "mean_delta_pp", "num_tasks"], model_summary_rows)

    task_summary_rows: list[dict[str, str | float]] = []
    for t in tasks:
        vals = [float(r["delta_pp"]) for r in rows if str(r["task"]) == t]
        task_summary_rows.append(
            {
                "task": t,
                "mean_delta_pp": float(np.mean(vals)),
                "num_models": len(vals),
            }
        )
    task_csv = output_dir / f"{prefix}_by_task.csv"
    write_csv(task_csv, ["task", "mean_delta_pp", "num_models"], task_summary_rows)

    # Build heatmap matrix
    model_index = {m: i for i, m in enumerate(models)}
    task_index = {t: i for i, t in enumerate(tasks)}
    mat = np.full((len(models), len(tasks)), np.nan, dtype=float)
    for r in rows:
        mat[model_index[str(r["model"])], task_index[str(r["task"])]] = float(r["delta_pp"])

    vmax = float(np.nanmax(np.abs(mat))) if np.isfinite(mat).any() else 1.0
    vmax = max(vmax, 1.0)
    vmin = -vmax

    # Order models in summary bar by mean improvement
    model_mean = np.array([float(np.nanmean(mat[i, :])) for i in range(len(models))], dtype=float)
    order = np.argsort(model_mean)
    models_ordered = [models[i] for i in order]
    model_mean_ordered = model_mean[order]

    task_mean = np.array([float(np.nanmean(mat[:, j])) for j in range(len(tasks))], dtype=float)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, max(6.5, 0.35 * len(models) + 3)),
        gridspec_kw={"width_ratios": [2.2, 1.1, 1.0]},
        layout="constrained",
    )

    # Panel A: heatmap
    ax = axes[0]
    im = ax.imshow(mat[order, :], cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(tasks)))
    ax.set_xticklabels(tasks, rotation=20, ha="right")
    ax.set_yticks(np.arange(len(models_ordered)))
    ax.set_yticklabels(models_ordered)
    ax.set_title("A) Delta vs baseline (pp)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("Forced binary - baseline (pp)")

    # Panel B: per-model average
    ax = axes[1]
    colors = ["#16a34a" if v >= 0 else "#dc2626" for v in model_mean_ordered]
    ax.barh(np.arange(len(models_ordered)), model_mean_ordered, color=colors)
    ax.axvline(0.0, color="black", linewidth=1)
    ax.set_yticks(np.arange(len(models_ordered)))
    ax.set_yticklabels(models_ordered)
    ax.set_title("B) Mean delta by model")
    ax.set_xlabel("pp")
    ax.grid(True, axis="x", linestyle="--", alpha=0.35)

    # Panel C: per-task average
    ax = axes[2]
    colors = ["#16a34a" if v >= 0 else "#dc2626" for v in task_mean]
    ax.bar(np.arange(len(tasks)), task_mean, color=colors)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xticks(np.arange(len(tasks)))
    ax.set_xticklabels(tasks, rotation=25, ha="right")
    ax.set_title("C) Mean delta by task")
    ax.set_ylabel("pp")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)

    png = output_dir / f"{prefix}_figure.png"
    fig.savefig(png, dpi=dpi)
    plt.close(fig)

    print(f"Wrote {long_csv}")
    print(f"Wrote {model_csv}")
    print(f"Wrote {task_csv}")
    print(f"Wrote {png}")
    return True


def main() -> int:
    args = parse_args()
    forced_roots = []
    for item in args.forced_roots.split(","):
        rel = item.strip()
        if not rel:
            continue
        p = Path(rel)
        forced_roots.append((p if p.is_absolute() else REPO_ROOT / p).resolve())
    baseline_root = args.baseline_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_rows(forced_roots, baseline_root)
    if not rows:
        print("No comparable forced-vs-baseline rows found.")
        return 1

    tasks_requested = [t.strip() for t in args.tasks.split(",") if t.strip()]

    ok = build_outputs(
        rows=rows,
        tasks_requested=tasks_requested,
        output_dir=output_dir,
        prefix=args.prefix,
        dpi=args.dpi,
    )
    if not ok:
        print("No outputs produced for primary view.")
        return 1

    # Second view: keep only models with all requested tasks present.
    model_to_tasks: dict[str, set[str]] = {}
    for r in rows:
        model_to_tasks.setdefault(str(r["model"]), set()).add(str(r["task"]))
    complete_models = {
        model
        for model, seen_tasks in model_to_tasks.items()
        if all(task in seen_tasks for task in tasks_requested)
    }
    complete_rows = [r for r in rows if str(r["model"]) in complete_models]
    complete_prefix = f"{args.prefix}_{args.complete_suffix}"
    if complete_rows:
        build_outputs(
            rows=complete_rows,
            tasks_requested=tasks_requested,
            output_dir=output_dir,
            prefix=complete_prefix,
            dpi=args.dpi,
        )
    else:
        print(
            "No complete-task-only outputs produced "
            f"(no model has all requested tasks: {','.join(tasks_requested)})."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

