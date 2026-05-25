#!/usr/bin/env python3
"""Plot forced-binary accuracy improvements for selected tasks."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--forced-root",
        type=Path,
        default=REPO_ROOT / "results" / "vocab_binary_ablation",
        help="Directory containing per-model forced-binary vocab results.",
    )
    p.add_argument(
        "--forced-task-root",
        type=Path,
        default=REPO_ROOT / "results" / "trog_matrix_binary_ablation",
        help="Directory containing per-model forced-binary trog/matrix results.",
    )
    p.add_argument(
        "--forced-other-root",
        type=Path,
        default=REPO_ROOT / "results" / "forced_binary_other_tasks",
        help="Directory containing per-model forced-binary results for other tasks.",
    )
    p.add_argument(
        "--baseline-root",
        type=Path,
        default=REPO_ROOT / "results" / "v1",
        help="Directory containing baseline per-model results.",
    )
    p.add_argument(
        "--output-png",
        type=Path,
        default=REPO_ROOT / "results" / "analysis" / "vocab_forced_binary_improvement.png",
        help="Output plot path.",
    )
    p.add_argument(
        "--output-csv",
        type=Path,
        default=REPO_ROOT / "results" / "analysis" / "vocab_forced_binary_improvement.csv",
        help="Output tabular summary path for vocab-only compatibility.",
    )
    p.add_argument(
        "--output-task-csv",
        type=Path,
        default=REPO_ROOT / "results" / "analysis" / "forced_binary_task_overrides.csv",
        help="Output tabular summary path for all selected tasks.",
    )
    p.add_argument(
        "--tasks",
        type=str,
        default="egma-math,matrix-reasoning,mental-rotation,theory-of-mind,trog,vocab",
        help="Comma-separated tasks to include in the comparison plot.",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Plot DPI.",
    )
    return p.parse_args()


def read_task_accuracy(summary_csv: Path, task_id: str) -> float | None:
    if not summary_csv.exists():
        return None
    with summary_csv.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("task_id", "").strip() == task_id:
                return float(row["accuracy"])
    return None


def resolve_baseline_summary_path(baseline_root: Path, model_slug: str) -> Path:
    if model_slug == "cogvlm":
        preferred = (
            REPO_ROOT / "results" / "v1_additional_models" / "cogvlm_backup_pre_fix" / "summary.csv"
        )
        if preferred.exists():
            return preferred
    direct = baseline_root / model_slug / "summary.csv"
    if direct.exists():
        return direct
    additional = REPO_ROOT / "results" / "v1_additional_models" / model_slug / "summary.csv"
    if additional.exists():
        return additional
    return baseline_root / model_slug / "baseline" / "summary.csv"


def maybe_append_cogvlm_vocab_row(rows: list[dict[str, str | float]], repo_root: Path) -> None:
    """Include CogVLM from validated ablation paths when available."""
    if any(r["model"] == "cogvlm" and r["task"] == "vocab" for r in rows):
        return
    baseline_summary = (
        repo_root / "results" / "v1_additional_models" / "cogvlm_backup_pre_fix" / "summary.csv"
    )
    forced_summary = (
        repo_root
        / "results"
        / "cogvlm_vocab_ablation"
        / "binary_default_r2"
        / "v1"
        / "cogvlm"
        / "summary.csv"
    )
    baseline_vocab = read_task_accuracy(baseline_summary, "vocab")
    forced_vocab = read_task_accuracy(forced_summary, "vocab")
    if baseline_vocab is None or forced_vocab is None:
        return
    rows.append(
        {
            "model": "cogvlm",
            "task": "vocab",
            "baseline": baseline_vocab,
            "forced": forced_vocab,
            "delta": forced_vocab - baseline_vocab,
        }
    )


def collect_rows_for_task(task_id: str, forced_root: Path, baseline_root: Path) -> list[dict[str, str | float]]:
    rows: list[dict[str, str | float]] = []
    if not forced_root.exists():
        return rows
    forced_models = sorted([p for p in forced_root.iterdir() if p.is_dir()])
    for model_dir in forced_models:
        model_slug = model_dir.name
        forced_score = read_task_accuracy(model_dir / "summary.csv", task_id)
        baseline_score = read_task_accuracy(
            resolve_baseline_summary_path(baseline_root, model_slug),
            task_id,
        )
        if forced_score is None or baseline_score is None:
            continue
        rows.append(
            {
                "model": model_slug,
                "task": task_id,
                "baseline": baseline_score,
                "forced": forced_score,
                "delta": forced_score - baseline_score,
            }
        )
    return rows


def main() -> int:
    args = parse_args()
    forced_root = args.forced_root.resolve()
    forced_task_root = args.forced_task_root.resolve()
    forced_other_root = args.forced_other_root.resolve()
    baseline_root = args.baseline_root.resolve()
    output_png = args.output_png.resolve()
    output_csv = args.output_csv.resolve()
    output_task_csv = args.output_task_csv.resolve()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_task_csv.parent.mkdir(parents=True, exist_ok=True)

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    rows: list[dict[str, str | float]] = []
    trog_matrix_tasks = {"trog", "matrix-reasoning"}
    for task_id in tasks:
        if task_id == "vocab":
            task_root = forced_root
        elif task_id in trog_matrix_tasks:
            task_root = forced_task_root
        else:
            task_root = forced_other_root
        rows.extend(collect_rows_for_task(task_id, task_root, baseline_root))
    if "vocab" in tasks:
        maybe_append_cogvlm_vocab_row(rows, REPO_ROOT)
    rows.sort(key=lambda r: (str(r["task"]), str(r["model"])))

    if not rows:
        print("No comparable baseline/forced-binary summaries found.", flush=True)
        return 1

    # Keep vocab-only CSV schema for dashboard compatibility.
    vocab_rows = [r for r in rows if r["task"] == "vocab"]
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "baseline_vocab", "forced_binary_vocab", "delta"])
        for r in sorted(vocab_rows, key=lambda x: str(x["model"])):
            w.writerow(
                [
                    r["model"],
                    f'{float(r["baseline"]):.4f}',
                    f'{float(r["forced"]):.4f}',
                    f'{float(r["delta"]):+.4f}',
                ]
            )

    with output_task_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "task", "forced_binary_accuracy", "baseline_accuracy", "delta"])
        for r in rows:
            w.writerow(
                [
                    r["model"],
                    r["task"],
                    f'{float(r["forced"]):.4f}',
                    f'{float(r["baseline"]):.4f}',
                    f'{float(r["delta"]):+.4f}',
                ]
            )

    import matplotlib.pyplot as plt

    task_to_rows: dict[str, list[dict[str, str | float]]] = {task: [] for task in tasks}
    for r in rows:
        task_to_rows[str(r["task"])].append(r)
    fig, axes = plt.subplots(
        1,
        len(tasks),
        figsize=(max(9.5, 4.4 * len(tasks)), 5.8),
        layout="constrained",
        squeeze=False,
    )
    for idx, task_id in enumerate(tasks):
        ax = axes[0][idx]
        task_rows = sorted(task_to_rows.get(task_id, []), key=lambda x: str(x["model"]))
        if not task_rows:
            ax.set_title(task_id)
            ax.text(0.5, 0.5, "No comparable rows", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue
        labels = [str(r["model"]) for r in task_rows]
        baseline = [float(r["baseline"]) for r in task_rows]
        forced = [float(r["forced"]) for r in task_rows]
        deltas = [float(r["delta"]) for r in task_rows]
        x = range(len(labels))
        width = 0.35
        ax.bar([i - width / 2 for i in x], baseline, width=width, label="Baseline", color="#64748b")
        ax.bar([i + width / 2 for i in x], forced, width=width, label="Forced binary", color="#0ea5e9")
        for i, delta in enumerate(deltas):
            y = max(baseline[i], forced[i]) + 0.02
            ax.text(i, y, f"{delta:+.3f}", ha="center", va="bottom", fontsize=8.5, color="#334155")
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=18, ha="right")
        ax.set_ylim(0.0, 1.05)
        ax.set_title(task_id)
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)
        if idx == 0:
            ax.set_ylabel("Accuracy")
        if idx == len(tasks) - 1:
            ax.legend(loc="upper left")

    fig.suptitle("Forced-binary logits impact by task", fontsize=13)

    fig.savefig(output_png, dpi=args.dpi)
    plt.close(fig)

    print(f"Wrote {output_csv}", flush=True)
    print(f"Wrote {output_task_csv}", flush=True)
    print(f"Wrote {output_png}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
