#!/usr/bin/env python3
"""Build a joint table of TROG phase experiment results from phase_*.csv files.

Reads results/trog-phases/phase_*.csv and writes:
  - phase_summary.csv
  - phase_summary.md

Re-run after the suite finishes (or anytime) to refresh the table.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_DIR = PROJECT_ROOT / "results" / "prompt_optimization/trog/qwen-0.8b"

ROW_ORDER = [
    "baseline",
    "1",
    "2",
    "3",
    "4",
    "5",
    "1_2_3",
    "1_5",
    "1_2_3_4_5",
]

PHASE_LABELS: dict[str, str] = {
    "baseline": "0 — baseline (manifest prompts, standard parsing)",
    "1": "1 — structured multiline prompt",
    "2": "2 — enhanced answer parsing (reverse-scan, last-letter)",
    "3": "3 — system prompt + strict format suffix",
    "4": "4 — visual grounding hints (complex types)",
    "5": "5 — sentence decomposition / CoT (complex types)",
    "1_2_3": "1+2+3 — multiline + parsing + system prompt",
    "1_5": "1+5 — multiline + decomposition",
    "1_2_3_4_5": "1+2+3+4+5 — all improvements",
}


def stem_from_filename(path: Path) -> str | None:
    m = re.match(r"^phase_(.+)\.csv$", path.name)
    return m.group(1) if m else None


def load_metrics(csv_path: Path) -> dict:
    n = 0
    correct = 0
    parsed = 0
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            n += 1
            if row.get("is_correct", "").lower() in ("true", "1", "yes"):
                correct += 1
            if row.get("parsed", "").lower() in ("true", "1", "yes"):
                parsed += 1
    acc = correct / n if n else 0.0
    pr = parsed / n if n else 0.0
    return {"n": n, "accuracy": acc, "parse_rate": pr, "correct": correct, "parsed": parsed}


def sort_key(stem: str) -> tuple[int, str]:
    try:
        idx = ROW_ORDER.index(stem)
        return (idx, stem)
    except ValueError:
        return (len(ROW_ORDER), stem)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dir",
        type=Path,
        default=DEFAULT_DIR,
        help="Directory containing phase_*.csv",
    )
    parser.add_argument(
        "--stdout-only",
        action="store_true",
        help="Print markdown table only; do not write files",
    )
    args = parser.parse_args()
    out_dir: Path = args.dir

    if not out_dir.is_dir():
        print(f"Directory not found: {out_dir}", flush=True)
        return

    files = sorted(out_dir.glob("phase_*.csv"))
    by_stem: dict[str, Path] = {}
    for p in files:
        if p.name == "phase_summary.csv":
            continue
        stem = stem_from_filename(p)
        if stem:
            by_stem[stem] = p

    if not by_stem:
        print(f"No phase_*.csv files in {out_dir}", flush=True)
        return

    stems_sorted = sorted(by_stem.keys(), key=sort_key)
    baseline_acc: float | None = None
    if "baseline" in by_stem:
        baseline_acc = load_metrics(by_stem["baseline"])["accuracy"]

    rows_out: list[dict] = []
    for stem in stems_sorted:
        path = by_stem[stem]
        m = load_metrics(path)
        label = PHASE_LABELS.get(stem, stem)
        if stem == "baseline" or baseline_acc is None:
            delta = "—"
        else:
            delta = f"{(m['accuracy'] - baseline_acc) * 100:+.1f} pp"
        rows_out.append(
            {
                "config_id": stem,
                "description": label,
                "n": m["n"],
                "accuracy": m["accuracy"],
                "parse_rate": m["parse_rate"],
                "correct": m["correct"],
                "parsed_count": m["parsed"],
                "delta_vs_baseline_pp": delta,
                "source_file": path.name,
            }
        )

    try:
        rel_dir = out_dir.relative_to(PROJECT_ROOT)
    except ValueError:
        rel_dir = out_dir
    md_lines = [
        "# TROG phase experiments — summary",
        "",
        f"Source: `{rel_dir}` · "
        f"{len(rows_out)} configuration(s) with data.",
        "",
        "| Config | Description | N | Accuracy | Parse % | Δ vs baseline |",
        "|--------|-------------|---|---------:|--------:|---------------|",
    ]
    for r in rows_out:
        md_lines.append(
            f"| `{r['config_id']}` | {r['description']} | {r['n']} | "
            f"{r['accuracy']:.1%} | {r['parse_rate']:.1%} | {r['delta_vs_baseline_pp']} |"
        )
    md_lines.append("")
    md_lines.append("*Δ vs baseline* = accuracy difference in percentage points vs phase 0.")
    md_lines.append("")
    md_text = "\n".join(md_lines)

    if args.stdout_only:
        print(md_text)
        return

    md_path = out_dir / "phase_summary.md"
    csv_path = out_dir / "phase_summary.csv"
    md_path.write_text(md_text, encoding="utf-8")

    fieldnames = [
        "config_id",
        "description",
        "n",
        "accuracy",
        "parse_rate",
        "correct",
        "parsed_count",
        "delta_vs_baseline_pp",
        "source_file",
    ]
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
        print(
            f"\nMissing CSVs (suite incomplete or different path): {', '.join(missing)}",
            flush=True,
        )


if __name__ == "__main__":
    main()
