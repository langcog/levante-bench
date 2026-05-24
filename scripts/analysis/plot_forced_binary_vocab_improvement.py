#!/usr/bin/env python3
"""Plot vocab accuracy improvement for models rerun with forced binary logits."""

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
        help="Directory containing per-model forced-binary results.",
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
        help="Output tabular summary path.",
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
    direct = baseline_root / model_slug / "summary.csv"
    if direct.exists():
        return direct
    return baseline_root / model_slug / "baseline" / "summary.csv"


def maybe_append_cogvlm_row(rows: list[tuple[str, float, float, float]], repo_root: Path) -> None:
    """Include CogVLM from validated ablation paths when available."""
    if any(model == "cogvlm" for model, *_ in rows):
        return
    baseline_summary = repo_root / "results" / "v1_additional_models" / "cogvlm_backup_pre_fix" / "summary.csv"
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
    rows.append(("cogvlm", baseline_vocab, forced_vocab, forced_vocab - baseline_vocab))


def main() -> int:
    args = parse_args()
    forced_root = args.forced_root.resolve()
    baseline_root = args.baseline_root.resolve()
    output_png = args.output_png.resolve()
    output_csv = args.output_csv.resolve()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    forced_models = sorted([p for p in forced_root.iterdir() if p.is_dir()])
    rows: list[tuple[str, float, float, float]] = []
    for model_dir in forced_models:
        model_slug = model_dir.name
        forced_vocab = read_task_accuracy(model_dir / "summary.csv", "vocab")
        baseline_vocab = read_task_accuracy(resolve_baseline_summary_path(baseline_root, model_slug), "vocab")
        if forced_vocab is None or baseline_vocab is None:
            continue
        rows.append((model_slug, baseline_vocab, forced_vocab, forced_vocab - baseline_vocab))
    maybe_append_cogvlm_row(rows, REPO_ROOT)
    rows.sort(key=lambda r: r[0])

    if not rows:
        print("No comparable baseline/forced-binary vocab summaries found.", flush=True)
        return 1

    with output_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "baseline_vocab", "forced_binary_vocab", "delta"])
        for model, base, forced, delta in rows:
            w.writerow([model, f"{base:.4f}", f"{forced:.4f}", f"{delta:+.4f}"])

    import matplotlib.pyplot as plt

    labels = [r[0] for r in rows]
    baseline = [r[1] for r in rows]
    forced = [r[2] for r in rows]
    deltas = [r[3] for r in rows]

    fig, ax = plt.subplots(figsize=(max(8.0, 1.6 * len(labels) + 4.0), 5.6), layout="constrained")
    x = range(len(labels))
    width = 0.35
    ax.bar([i - width / 2 for i in x], baseline, width=width, label="Baseline vocab", color="#64748b")
    ax.bar([i + width / 2 for i in x], forced, width=width, label="Forced binary vocab", color="#0ea5e9")

    for i, delta in enumerate(deltas):
        y = max(baseline[i], forced[i]) + 0.02
        ax.text(i, y, f"{delta:+.3f}", ha="center", va="bottom", fontsize=9, color="#334155")

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Vocab accuracy")
    ax.set_title("Forced-binary logits impact on vocab accuracy")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="upper left")

    fig.savefig(output_png, dpi=args.dpi)
    plt.close(fig)

    print(f"Wrote {output_csv}", flush=True)
    print(f"Wrote {output_png}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
