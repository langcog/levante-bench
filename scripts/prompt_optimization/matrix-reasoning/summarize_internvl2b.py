#!/usr/bin/env python3
"""Summarise Matrix Reasoning InternVL3.5-2B phase experiment results.

Reads results/matrix-phases/phase_*.csv and writes:
  - phase_summary.csv
  - phase_summary.md
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_DIR = PROJECT_ROOT / "results" / "prompt_optimization/matrix-reasoning/internvl-3.5-2b"

ROW_ORDER = [
    "baseline",
    "1",
    "2",
    "3",
    "4",
    "5",
    "1_2_3",
    "1_2_3_4",
    "1_2_3_5",
]

PHASE_LABELS: dict[str, str] = {
    "baseline": "0 — baseline (manifest prompt)",
    "1": "1 — structured prompt (matrix + options layout)",
    "2": "2 — enhanced answer parsing",
    "3": "3 — task-specific system prompt (Raven's expert)",
    "4": "4 — rule enumeration hint",
    "5": "5 — rule CoT with answer-last (512 tokens)",
    "1_2_3": "1+2+3 — structured + parsing + system prompt",
    "1_2_3_4": "1+2+3+4 — structural + rule hint",
    "1_2_3_5": "1+2+3+5 — structural + rule CoT",
}


def stem_from_filename(path: Path) -> str | None:
    m = re.match(r"^phase_(.+)\.csv$", path.name)
    return m.group(1) if m else None


def load_metrics(csv_path: Path) -> dict:
    n = correct = parsed = 0
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            n += 1
            if row.get("is_correct", "").lower() in ("true", "1", "yes"):
                correct += 1
            if row.get("parsed", "").lower() in ("true", "1", "yes"):
                parsed += 1
    return {
        "n": n,
        "accuracy": correct / n if n else 0.0,
        "parse_rate": parsed / n if n else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=str(DEFAULT_DIR))
    args = parser.parse_args()

    out_dir = Path(args.dir)
    files = sorted(out_dir.glob("phase_*.csv"))
    files = [f for f in files if f.name != "phase_summary.csv"]

    if not files:
        print(f"No phase_*.csv files in {out_dir}", flush=True)
        return

    data: dict[str, dict] = {}
    for f in files:
        stem = stem_from_filename(f)
        if stem:
            data[stem] = load_metrics(f)

    baseline_acc = data.get("baseline", {}).get("accuracy")

    order = ROW_ORDER + sorted(k for k in data if k not in ROW_ORDER)
    rows_out = []
    for key in order:
        if key not in data:
            continue
        m = data[key]
        label = PHASE_LABELS.get(key, key)
        delta = "—"
        if baseline_acc is not None and key != "baseline":
            delta = f"{(m['accuracy'] - baseline_acc)*100:+.1f} pp"
        rows_out.append({
            "Config": f"`{key}`",
            "Description": label,
            "N": m["n"],
            "Accuracy": f"{m['accuracy']:.1%}",
            "Parse %": f"{m['parse_rate']:.1%}",
            "Δ vs baseline": delta,
        })

    header = "# Matrix Reasoning — InternVL3.5-2B phase experiments — summary\n\n"
    header += f"Source: `results/matrix-phases` · {len(rows_out)} configuration(s) with data.\n\n"

    cols = list(rows_out[0].keys())
    md_lines = ["| " + " | ".join(cols) + " |"]
    md_lines.append("|" + "|".join("---" for _ in cols) + "|")
    for r in rows_out:
        md_lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")

    md_text = header + "\n".join(md_lines) + "\n\n"
    md_text += "*Δ vs baseline* = accuracy difference in percentage points vs phase 0.\n"

    print(md_text)

    md_path = out_dir / "phase_summary.md"
    csv_path = out_dir / "phase_summary.csv"
    md_path.write_text(md_text, encoding="utf-8")
    print(f"Wrote {md_path}")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
