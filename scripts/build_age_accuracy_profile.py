#!/usr/bin/env python3
"""Build the canonical age->task accuracy profile from LEVANTE child trial data.

Reads data/responses/v1/trials.csv, bins trials by task_id and rounded age (in
whole years), and writes shared/persona/age_task_accuracy.json shaped as:

    { "<task_id>": { "<ageYears>": meanAccuracy, ... }, ... }

This file is the single source of truth for the child-age persona prompts used
by BOTH levante-bench (Python) and levante-qa (TypeScript, vendored copy). Cells
with fewer than MIN_SAMPLES trials are dropped so a sparse age/task corner never
produces a misleading "expected accuracy".

Usage:
    python scripts/build_age_accuracy_profile.py
    python scripts/build_age_accuracy_profile.py --trials path/to/trials.csv --min-samples 30
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

MIN_SAMPLES_DEFAULT = 30

# Tasks that levante-qa / levante-bench actually drive. Other task_ids in the
# CSV (sre, swr, pa, ...) are ignored.
INCLUDED_TASKS = {
    "egma-math",
    "matrix-reasoning",
    "mental-rotation",
    "theory-of-mind",
    "trog",
    "vocab",
    "hearts-and-flowers",
    "same-different-selection",
    "memory-game",
}


def to_bool(value: str) -> int | None:
    v = str(value).strip().lower()
    if v in {"true", "1"}:
        return 1
    if v in {"false", "0"}:
        return 0
    return None


def main() -> None:
    here = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=Path, default=here / "data" / "responses" / "v1" / "trials.csv")
    ap.add_argument("--out", type=Path, default=here / "shared" / "persona" / "age_task_accuracy.json")
    ap.add_argument("--min-samples", type=int, default=MIN_SAMPLES_DEFAULT)
    args = ap.parse_args()

    # (task_id, ageYears) -> [n_correct, n_total]
    agg: dict[tuple[str, int], list[int]] = defaultdict(lambda: [0, 0])

    with args.trials.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            task = row.get("task_id")
            if task not in INCLUDED_TASKS:
                continue
            correct = to_bool(row.get("correct", ""))
            if correct is None:
                continue
            try:
                age_years = int(round(float(row["age"])))
            except (TypeError, ValueError):
                continue
            cell = agg[(task, age_years)]
            cell[0] += correct
            cell[1] += 1

    profile: dict[str, dict[str, float]] = defaultdict(dict)
    for (task, age_years), (n_correct, n_total) in sorted(agg.items()):
        if n_total < args.min_samples:
            continue
        profile[task][str(age_years)] = round(n_correct / n_total, 4)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(sorted(profile.items())), indent=2) + "\n")
    print(f"wrote {args.out} ({sum(len(v) for v in profile.values())} age/task cells across {len(profile)} tasks)")


if __name__ == "__main__":
    main()
