#!/usr/bin/env python3
"""Build age->mean IRT ability (theta) profile from LEVANTE child trial + IRT data.

Joins data/responses/v1/trials.csv (run_id, task_id, age) with per-task
irt_models/*_ability_scores.csv (run_id, ability, se), then bins by rounded age
in whole years and writes shared/persona/age_task_ability.json:

    { "<task_id>": { "<ageYears>": { "theta": float, "n": int }, ... }, ... }

Only tasks with an IRT ability file are included (egma-math, vocab, trog,
mental-rotation, matrix-reasoning, theory-of-mind today). Tasks without IRT
models keep accuracy-only persona hints.

Usage:
    python scripts/build_age_ability_profile.py
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

MIN_RUNS_DEFAULT = 15

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


def load_ability(irt_dir: Path, task_id: str) -> dict[str, tuple[float, float]]:
    """run_id -> (ability, se)"""
    path = irt_dir / f"{task_id}_ability_scores.csv"
    if not path.is_file():
        return {}
    out: dict[str, tuple[float, float]] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rid = row.get("run_id", "").strip()
            if not rid:
                continue
            try:
                ability = float(row["ability"])
            except (TypeError, ValueError, KeyError):
                continue
            try:
                se = float(row.get("se") or "nan")
            except (TypeError, ValueError):
                se = float("nan")
            out[rid] = (ability, se)
    return out


def main() -> None:
    here = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=Path, default=here / "data" / "responses" / "v1" / "trials.csv")
    ap.add_argument("--irt-dir", type=Path, default=here / "data" / "responses" / "v1" / "irt_models")
    ap.add_argument("--out", type=Path, default=here / "shared" / "persona" / "age_task_ability.json")
    ap.add_argument("--min-runs", type=int, default=MIN_RUNS_DEFAULT)
    args = ap.parse_args()

    # (task_id, run_id) -> age (mean if multiple trial rows)
    run_age: dict[tuple[str, str], list[float]] = defaultdict(list)
    with args.trials.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            task = row.get("task_id")
            if task not in INCLUDED_TASKS:
                continue
            rid = row.get("run_id", "").strip()
            if not rid:
                continue
            try:
                age = float(row["age"])
            except (TypeError, ValueError, KeyError):
                continue
            run_age[(task, rid)].append(age)

    # (task_id, ageYears) -> list of theta
    agg: dict[tuple[str, int], list[float]] = defaultdict(list)

    for task in INCLUDED_TASKS:
        ability_by_run = load_ability(args.irt_dir, task)
        if not ability_by_run:
            continue
        for (t, rid), ages in run_age.items():
            if t != task:
                continue
            if rid not in ability_by_run:
                continue
            theta, _se = ability_by_run[rid]
            age_years = int(round(sum(ages) / len(ages)))
            agg[(task, age_years)].append(theta)

    profile: dict[str, dict[str, dict]] = defaultdict(dict)
    for (task, age_years), thetas in sorted(agg.items()):
        if len(thetas) < args.min_runs:
            continue
        mean_theta = sum(thetas) / len(thetas)
        profile[task][str(age_years)] = {
            "theta": round(mean_theta, 4),
            "n": len(thetas),
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(sorted(profile.items())), indent=2) + "\n")
    n_cells = sum(len(v) for v in profile.values())
    print(
        f"wrote {args.out} ({n_cells} age/task cells across {len(profile)} tasks with IRT ability)"
    )


if __name__ == "__main__":
    main()
