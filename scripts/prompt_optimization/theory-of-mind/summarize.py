#!/usr/bin/env python3
"""Build a joint table of Stories (ToM) phase experiment results from phase_*.csv files.

Reads results/stories-phases/phase_*.csv and writes:
  - phase_summary.csv
  - phase_summary.md
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DIR = PROJECT_ROOT / "results" / "prompt_optimization/theory-of-mind/internvl-3.5-4b"

ROW_ORDER = [
    "baseline",
    "1", "2", "3", "4", "5", "6", "7", "8",
    "1_2_3", "1_4", "1_2_3_4_5", "1_2_3_5", "1_2_3_6",
    "3_7", "3_8", "7_8", "3_7_8",
]

PHASE_LABELS: dict[str, str] = {
    "baseline": "0 — baseline (manifest prompts, Qwen default sys)",
    "1": "1 — structured prompt (story / question / options)",
    "2": "2 — enhanced answer parsing",
    "3": "3 — task-specific system prompt (ToM reasoning)",
    "4": "4 — false belief / emotion hint",
    "5": "5 — mental state CoT (512 tokens)",
    "6": "6 — answer-last CoT (512 tokens)",
    "7": "7 — type-specific system prompt (per question type)",
    "8": "8 — perspective anchor (per-turn character hint)",
    "1_2_3": "1+2+3 — structured + parsing + system prompt",
    "1_4": "1+4 — structured + belief hint",
    "1_2_3_4_5": "1+2+3+4+5 — all improvements",
    "1_2_3_5": "1+2+3+5 — structural + CoT (512 tokens)",
    "1_2_3_6": "1+2+3+6 — structural + answer-last CoT",
    "3_7": "3+7 — generic + type-specific system prompts",
    "3_8": "3+8 — generic system + perspective anchor",
    "7_8": "7+8 — type-specific system + perspective anchor",
    "3_7_8": "3+7+8 — all system prompt variants",
}


def stem_from_filename(path: Path) -> str | None:
    m = re.match(r"^phase_(.+)\.csv$", path.name)
    return m.group(1) if m else None


def load_metrics(csv_path: Path) -> dict:
    n = 0
    correct = 0
    parsed = 0
    by_type: dict[str, dict] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
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
    acc = correct / n if n else 0.0
    pr = parsed / n if n else 0.0
    return {"n": n, "accuracy": acc, "parse_rate": pr, "correct": correct, "parsed": parsed, "by_type": by_type}


def sort_key(stem: str) -> tuple[int, str]:
    try:
        idx = ROW_ORDER.index(stem)
        return (idx, stem)
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
        rows_out.append({
            "config_id": stem,
            "description": label,
            "n": m["n"],
            "accuracy": m["accuracy"],
            "parse_rate": m["parse_rate"],
            "correct": m["correct"],
            "parsed_count": m["parsed"],
            "delta_vs_baseline_pp": delta,
            "source_file": path.name,
        })

    try:
        rel_dir = out_dir.relative_to(PROJECT_ROOT)
    except ValueError:
        rel_dir = out_dir
    md_lines = [
        "# Stories (Theory of Mind) phase experiments — summary",
        "",
        f"Source: `{rel_dir}` · {len(rows_out)} configuration(s) with data.",
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
        "config_id", "description", "n", "accuracy", "parse_rate",
        "correct", "parsed_count", "delta_vs_baseline_pp", "source_file",
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
        print(f"\nMissing CSVs (suite incomplete): {', '.join(missing)}", flush=True)


if __name__ == "__main__":
    main()
