#!/usr/bin/env python3
"""Build age->mean IRT ability (theta) profile from LEVANTE child trial + IRT data.

Joins data/responses/v1/trials.csv (run_id, task_id, age, site) with per-task
irt_models/*_ability_scores.csv (run_id, ability, se), then bins by rounded age
in whole years and writes shared/persona/age_task_ability.json:

    { "<task_id>": { "<ageYears>": { "theta": float, "n": int }, ... }, ... }

Also writes shared/persona/age_task_ability_by_country.json:

    { "<country>": { "<task_id>": { "<ageYears>": { "theta", "n" }, ... }, ... }, ... }

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
import sys
from collections import Counter, defaultdict
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
from site_country import SITE_TO_COUNTRY, site_to_country  # noqa: E402

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


def _write_profile(path: Path, profile: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(profile, indent=2) + "\n")


def main() -> None:
    here = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=Path, default=here / "data" / "responses" / "v1" / "trials.csv")
    ap.add_argument("--irt-dir", type=Path, default=here / "data" / "responses" / "v1" / "irt_models")
    ap.add_argument("--out", type=Path, default=here / "shared" / "persona" / "age_task_ability.json")
    ap.add_argument(
        "--out-by-country",
        type=Path,
        default=here / "shared" / "persona" / "age_task_ability_by_country.json",
    )
    ap.add_argument("--min-runs", type=int, default=MIN_RUNS_DEFAULT)
    args = ap.parse_args()

    # (task_id, run_id) -> ages / sites
    run_age: dict[tuple[str, str], list[float]] = defaultdict(list)
    run_sites: dict[tuple[str, str], list[str]] = defaultdict(list)
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
            site = (row.get("site") or "").strip()
            if site:
                run_sites[(task, rid)].append(site)

    # (task_id, ageYears) -> list of theta
    agg: dict[tuple[str, int], list[float]] = defaultdict(list)
    # (country, task_id, ageYears) -> list of theta
    agg_c: dict[tuple[str, str, int], list[float]] = defaultdict(list)

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
            sites = run_sites.get((task, rid), [])
            if sites:
                site = Counter(sites).most_common(1)[0][0]
                country = site_to_country(site)
                if country:
                    agg_c[(country, task, age_years)].append(theta)

    profile: dict[str, dict[str, dict]] = defaultdict(dict)
    for (task, age_years), thetas in sorted(agg.items()):
        if len(thetas) < args.min_runs:
            continue
        mean_theta = sum(thetas) / len(thetas)
        profile[task][str(age_years)] = {
            "theta": round(mean_theta, 4),
            "n": len(thetas),
        }

    by_country: dict[str, dict[str, dict[str, dict]]] = defaultdict(lambda: defaultdict(dict))
    for (country, task, age_years), thetas in sorted(agg_c.items()):
        if len(thetas) < args.min_runs:
            continue
        mean_theta = sum(thetas) / len(thetas)
        by_country[country][task][str(age_years)] = {
            "theta": round(mean_theta, 4),
            "n": len(thetas),
        }

    out_global = dict(sorted(profile.items()))
    out_country = {
        "_meta": {
            "site_to_country": SITE_TO_COUNTRY,
            "min_runs": args.min_runs,
        },
        **{c: dict(sorted(tasks.items())) for c, tasks in sorted(by_country.items())},
    }

    _write_profile(args.out, out_global)
    _write_profile(args.out_by_country, out_country)
    n_cells = sum(len(v) for v in profile.values())
    n_country = sum(len(ages) for tasks in by_country.values() for ages in tasks.values())
    print(
        f"wrote {args.out} ({n_cells} age/task cells across {len(profile)} tasks with IRT ability)"
    )
    print(
        f"wrote {args.out_by_country} ({n_country} country/age/task cells "
        f"across {len(by_country)} countries)"
    )


if __name__ == "__main__":
    main()
