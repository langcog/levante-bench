#!/usr/bin/env python3
"""Summarise TROG 2B phase experiment results."""

from __future__ import annotations
import argparse
import csv
from pathlib import Path

ROW_ORDER = [
    ("phase_baseline.csv",   "0 — Baseline"),
    ("phase_3.csv",          "3 — Format suffix"),
    ("phase_7.csv",          "7 — Language expert system prompt"),
    ("phase_1_2_3.csv",      "1+2+3 — Structured + parsing + format (replicates 0.8B best)"),
    ("phase_1_2_3_4.csv",    "1+2+3+4 — Structural + grounding hints"),
    ("phase_1_2_7.csv",      "1+2+7 — Structural + expert system"),
    ("phase_1_2_3_6.csv",    "1+2+3+6 — Structural + grammar CoT"),
    ("phase_1_2_6_7.csv",    "1+2+7+6 — Structural + expert system + grammar CoT"),
    ("phase_1_2_3_9.csv",    "1+2+3+9 — Structural + describe-first"),
    ("phase_1_2_3_4_9.csv",  "1+2+3+4+9 — Structural + grounding hints + describe-first"),
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
        parsed  = sum(1 for r in data if r["parsed"].lower() in ("true", "1"))
        acc = correct / n
        pr  = parsed / n
        if baseline_acc is None:
            baseline_acc = acc
        delta = "—" if acc == baseline_acc and label.startswith("0") else f"{(acc - baseline_acc)*100:+.1f} pp"
        rows.append({
            "Phase": label, "N": n,
            "Accuracy": f"{acc:.1%}", "Parse %": f"{pr:.1%}",
            "Δ vs baseline": delta,
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True)
    args = parser.parse_args()

    results_dir = Path(args.dir)
    rows = summarise(results_dir)

    if not rows:
        print("No results found.")
        return

    col_w = {
        "Phase": max(len(r["Phase"]) for r in rows),
        "N": 4, "Accuracy": 8, "Parse %": 7, "Δ vs baseline": 14,
    }

    def fmt_row(r):
        return (
            f"| {r['Phase']:<{col_w['Phase']}} "
            f"| {r['N']:>{col_w['N']}} "
            f"| {r['Accuracy']:>{col_w['Accuracy']}} "
            f"| {r['Parse %']:>{col_w['Parse %']}} "
            f"| {r['Δ vs baseline']:>{col_w['Δ vs baseline']}} |"
        )

    header = (
        f"| {'Phase':<{col_w['Phase']}} | {'N':>{col_w['N']}} "
        f"| {'Accuracy':>{col_w['Accuracy']}} | {'Parse %':>{col_w['Parse %']}} "
        f"| {'Δ vs baseline':>{col_w['Δ vs baseline']}} |"
    )
    sep = "| " + " | ".join(
        "-" * w for w in [col_w["Phase"], col_w["N"], col_w["Accuracy"],
                          col_w["Parse %"], col_w["Δ vs baseline"]]
    ) + " |"

    print("\nTROG — Qwen3.5-2B Phase Experiment Summary")
    print("=" * len(header))
    print(header)
    print(sep)
    for r in rows:
        print(fmt_row(r))
    print()

    md_path = results_dir / "phase_summary.md"
    csv_path = results_dir / "phase_summary.csv"

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# TROG — Qwen3.5-2B Phase Experiment Summary\n\n")
        f.write("| Phase | N | Accuracy | Parse % | Δ vs baseline |\n")
        f.write("|-------|---|----------|---------|---------------|\n")
        for r in rows:
            f.write(f"| `{r['Phase']}` | {r['N']} | {r['Accuracy']} | {r['Parse %']} | {r['Δ vs baseline']} |\n")
        f.write("\n*Δ vs baseline* = accuracy difference in pp vs phase 0 (Qwen3.5-2B baseline).\n")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Phase", "N", "Accuracy", "Parse %", "Δ vs baseline"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {md_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
