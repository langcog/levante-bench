#!/usr/bin/env python3
"""Summarise Matrix Reasoning 8B phase experiment results into a markdown table."""

from __future__ import annotations
import argparse
import csv
from pathlib import Path

ROW_ORDER = [
    ("phase_baseline.csv",       "0 — Baseline (512px)"),
    ("phase_baseline_r1024.csv", "0 — Baseline (1024px)"),
    ("phase_1_r1024.csv",        "1 — Structured prompt (1024px)"),
    ("phase_1_3.csv",            "1+3 — Structured + expert system (512px)"),
    ("phase_1_3_r1024.csv",      "1+3 — Structured + expert system (1024px)"),
    ("phase_1_4_r1024.csv",      "1+4 — Structured + rule hint (1024px)"),
    ("phase_1_3_4_r1024.csv",    "1+3+4 — Structured + expert + rule hint (1024px)"),
    ("phase_5_r1024.csv",        "5 — Describe-first (1024px)"),
    ("phase_5_3_r1024.csv",      "5+3 — Describe-first + expert system (1024px)"),
]


def summarise(results_dir: Path) -> list[dict]:
    rows = []
    baseline_acc: float | None = None

    for filename, label in ROW_ORDER:
        path = results_dir / filename
        if not path.exists():
            continue
        with open(path, newline="", encoding="utf-8") as f:
            data = list(csv.DictReader(f))
        n = len(data)
        if n == 0:
            continue
        correct = sum(1 for r in data if r["is_correct"].lower() in ("true", "1"))
        parsed = sum(1 for r in data if r["parsed"].lower() in ("true", "1"))
        acc = correct / n
        pr = parsed / n
        if baseline_acc is None:
            baseline_acc = acc
        if acc == baseline_acc and label.startswith("0"):
            delta = "—"
        else:
            delta = f"{(acc - baseline_acc) * 100:+.1f} pp"
        rows.append({
            "Phase": label,
            "N": n,
            "Accuracy": f"{acc:.1%}",
            "Parse %": f"{pr:.1%}",
            "Δ vs baseline": delta,
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, help="Path to results directory")
    args = parser.parse_args()

    results_dir = Path(args.dir)
    rows = summarise(results_dir)

    if not rows:
        print("No results found.")
        return

    col_w = {
        "Phase": max(len(r["Phase"]) for r in rows),
        "N": 4,
        "Accuracy": 8,
        "Parse %": 7,
        "Δ vs baseline": 14,
    }

    header = (
        f"| {'Phase':<{col_w['Phase']}} "
        f"| {'N':>{col_w['N']}} "
        f"| {'Accuracy':>{col_w['Accuracy']}} "
        f"| {'Parse %':>{col_w['Parse %']}} "
        f"| {'Δ vs baseline':>{col_w['Δ vs baseline']}} |"
    )
    sep = (
        f"| {'-'*col_w['Phase']} "
        f"| {'-'*col_w['N']} "
        f"| {'-'*col_w['Accuracy']} "
        f"| {'-'*col_w['Parse %']} "
        f"| {'-'*col_w['Δ vs baseline']} |"
    )

    print("\nMatrix Reasoning — InternVL3.5-8B Phase Experiment Summary")
    print("=" * (len(header)))
    print(header)
    print(sep)
    for r in rows:
        print(
            f"| {r['Phase']:<{col_w['Phase']}} "
            f"| {r['N']:>{col_w['N']}} "
            f"| {r['Accuracy']:>{col_w['Accuracy']}} "
            f"| {r['Parse %']:>{col_w['Parse %']}} "
            f"| {r['Δ vs baseline']:>{col_w['Δ vs baseline']}} |"
        )
    print()

    # Write markdown and CSV summaries
    md_path = results_dir / "phase_summary.md"
    csv_path = results_dir / "phase_summary.csv"

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Matrix Reasoning — InternVL3.5-8B Phase Experiment Summary\n\n")
        f.write("| Phase | N | Accuracy | Parse % | Δ vs baseline |\n")
        f.write("|-------|---|----------|---------|---------------|\n")
        for r in rows:
            f.write(f"| `{r['Phase']}` | {r['N']} | {r['Accuracy']} | {r['Parse %']} | {r['Δ vs baseline']} |\n")
        f.write("\n*Δ vs baseline* = accuracy difference in pp vs phase 0 (512px).\n")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Phase", "N", "Accuracy", "Parse %", "Δ vs baseline"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {md_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
