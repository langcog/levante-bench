#!/usr/bin/env python3
"""Summarise Mental Rotation InternVL3.5-8B phase experiment results.

Reads results/mrot-internvl8b-phases/phase_*.csv and writes:
  - phase_summary.csv
  - phase_summary.md
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DIR = PROJECT_ROOT / "results" / "prompt_optimization/mental-rotation/internvl-3.5-8b"

ROW_ORDER = [
    "baseline",
    "1", "2", "3", "4", "5",
    "1_2_3", "1_4", "1_2_3_4_5",
]

PHASE_LABELS: dict[str, str] = {
    "baseline": "0 — baseline (manifest prompts, default sys)",
    "1": "1 — structured prompt (reference / question / options)",
    "2": "2 — enhanced answer parsing",
    "3": "3 — task-specific system prompt (spatial reasoning)",
    "4": "4 — mirror awareness hint (chirality cue)",
    "5": "5 — feature-based CoT",
    "1_2_3": "1+2+3 — structured + parsing + system prompt",
    "1_4": "1+4 — structured + mirror hint",
    "1_2_3_4_5": "1+2+3+4+5 — all improvements",
}


def stem_from_filename(path: Path) -> str | None:
    m = re.match(r"^phase_(.+)\.csv$", path.name)
    return m.group(1) if m else None


def load_metrics(csv_path: Path) -> dict:
    n = correct = parsed = 0
    by_type: dict[str, dict] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            n += 1
            tt = row.get("trial_type", "UNKNOWN")
            rec = by_type.setdefault(tt, {"n": 0, "correct": 0, "parsed": 0})
            rec["n"] += 1
            if row.get("is_correct", "").lower() in ("true", "1", "yes"):
                correct += 1
                rec["correct"] += 1
            if row.get("parsed", "").lower() in ("true", "1", "yes"):
                parsed += 1
                rec["parsed"] += 1
    return {
        "n": n, "accuracy": correct / n if n else 0.0,
        "parse_rate": parsed / n if n else 0.0,
        "correct": correct, "parsed": parsed, "by_type": by_type,
    }


def sort_key(stem: str) -> tuple[int, str]:
    try:
        return (ROW_ORDER.index(stem), stem)
    except ValueError:
        return (len(ROW_ORDER), stem)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--stdout-only", action="store_true")
    args = parser.parse_args()
    out_dir: Path = args.dir

    if not out_dir.is_dir():
        print(f"Directory not found: {out_dir}", flush=True)
        return

    by_stem = {
        stem: p
        for p in sorted(out_dir.glob("phase_*.csv"))
        if (stem := stem_from_filename(p)) and p.name != "phase_summary.csv"
    }

    if not by_stem:
        print(f"No phase_*.csv files in {out_dir}", flush=True)
        return

    baseline_acc: float | None = None
    if "baseline" in by_stem:
        baseline_acc = load_metrics(by_stem["baseline"])["accuracy"]

    rows_out = []
    for stem in sorted(by_stem, key=sort_key):
        m = load_metrics(by_stem[stem])
        delta = "—" if (stem == "baseline" or baseline_acc is None) else f"{(m['accuracy'] - baseline_acc)*100:+.1f} pp"
        rows_out.append({
            "config_id": stem,
            "description": PHASE_LABELS.get(stem, stem),
            "n": m["n"], "accuracy": m["accuracy"],
            "parse_rate": m["parse_rate"],
            "correct": m["correct"], "parsed_count": m["parsed"],
            "delta_vs_baseline_pp": delta,
            "source_file": by_stem[stem].name,
        })

    try:
        rel_dir = out_dir.relative_to(PROJECT_ROOT)
    except ValueError:
        rel_dir = out_dir

    md_lines = [
        "# Mental Rotation — InternVL3.5-8B phase experiments — summary",
        "",
        f"Source: `{rel_dir}` · {len(rows_out)} configuration(s) with data.",
        "",
        "| Config | Description | N | Accuracy | Parse % | Δ vs baseline |",
        "|--------|-------------|---|---------:|--------:|---------------|",
    ] + [
        f"| `{r['config_id']}` | {r['description']} | {r['n']} | "
        f"{r['accuracy']:.1%} | {r['parse_rate']:.1%} | {r['delta_vs_baseline_pp']} |"
        for r in rows_out
    ] + ["", "*Δ vs baseline* = accuracy difference in percentage points vs phase 0.", ""]

    md_text = "\n".join(md_lines)

    if args.stdout_only:
        print(md_text)
        return

    md_path = out_dir / "phase_summary.md"
    csv_path = out_dir / "phase_summary.csv"
    md_path.write_text(md_text, encoding="utf-8")

    fieldnames = ["config_id", "description", "n", "accuracy", "parse_rate",
                  "correct", "parsed_count", "delta_vs_baseline_pp", "source_file"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows_out:
            row = {k: r[k] for k in fieldnames}
            row["accuracy"] = f"{r['accuracy']:.6f}"
            row["parse_rate"] = f"{r['parse_rate']:.6f}"
            w.writerow(row)

    print(md_text)
    print(f"\nWrote {md_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {csv_path.relative_to(PROJECT_ROOT)}")

    missing = [s for s in ROW_ORDER if s not in by_stem]
    if missing:
        print(f"\nMissing CSVs (suite incomplete): {', '.join(missing)}", flush=True)


if __name__ == "__main__":
    main()
