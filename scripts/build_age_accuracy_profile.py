#!/usr/bin/env python3
"""Build the canonical age->task accuracy profile from LEVANTE child trial data.

Reads data/responses/<version>/trials.csv (default: v2), bins trials by task_id
and rounded age (in whole years), and writes shared/persona/age_task_accuracy.json
shaped as:

    { "<task_id>": { "<ageYears>": meanAccuracy, ... }, ... }

Also writes shared/persona/age_task_accuracy_by_country.json:

    { "<country>": { "<task_id>": { "<ageYears>": meanAccuracy, ... }, ... }, ... }

This file is the single source of truth for the child-age persona prompts used
by BOTH levante-bench (Python) and levante-qa (TypeScript, vendored copy). Cells
with fewer than MIN_SAMPLES trials are dropped so a sparse age/task corner never
produces a misleading "expected accuracy".

Usage:
    python scripts/build_age_accuracy_profile.py
    python scripts/build_age_accuracy_profile.py --version v1
    python scripts/build_age_accuracy_profile.py --trials path/to/trials.csv --min-samples 30
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
from site_country import SITE_TO_COUNTRY, site_to_country  # noqa: E402

MIN_SAMPLES_DEFAULT = 30
DEFAULT_RESPONSES_VERSION = "v2"

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


def _write_profile(path: Path, profile: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(profile, indent=2) + "\n")


def main() -> None:
    here = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--version",
        default=DEFAULT_RESPONSES_VERSION,
        help=f"data/responses/<version> (default: {DEFAULT_RESPONSES_VERSION})",
    )
    ap.add_argument("--trials", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=here / "shared" / "persona" / "age_task_accuracy.json")
    ap.add_argument(
        "--out-by-country",
        type=Path,
        default=here / "shared" / "persona" / "age_task_accuracy_by_country.json",
    )
    ap.add_argument("--min-samples", type=int, default=MIN_SAMPLES_DEFAULT)
    args = ap.parse_args()
    if args.trials is None:
        args.trials = here / "data" / "responses" / args.version / "trials.csv"

    # (task_id, ageYears) -> [n_correct, n_total]
    agg: dict[tuple[str, int], list[int]] = defaultdict(lambda: [0, 0])
    # (country, task_id, ageYears) -> [n_correct, n_total]
    agg_c: dict[tuple[str, str, int], list[int]] = defaultdict(lambda: [0, 0])

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
            country = site_to_country(row.get("site"))
            if country:
                ccell = agg_c[(country, task, age_years)]
                ccell[0] += correct
                ccell[1] += 1

    profile: dict[str, dict[str, float]] = defaultdict(dict)
    for (task, age_years), (n_correct, n_total) in sorted(agg.items()):
        if n_total < args.min_samples:
            continue
        profile[task][str(age_years)] = round(n_correct / n_total, 4)

    by_country: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    for (country, task, age_years), (n_correct, n_total) in sorted(agg_c.items()):
        if n_total < args.min_samples:
            continue
        by_country[country][task][str(age_years)] = round(n_correct / n_total, 4)

    out_global = dict(sorted(profile.items()))
    out_country = {
        "_meta": {
            "site_to_country": SITE_TO_COUNTRY,
            "min_samples": args.min_samples,
            "responses_version": args.version,
            "trials": str(args.trials),
        },
        **{c: dict(sorted(tasks.items())) for c, tasks in sorted(by_country.items())},
    }

    _write_profile(args.out, out_global)
    _write_profile(args.out_by_country, out_country)
    n_global = sum(len(v) for v in profile.values())
    n_country = sum(len(ages) for tasks in by_country.values() for ages in tasks.values())
    print(f"wrote {args.out} ({n_global} age/task cells across {len(profile)} tasks)")
    print(
        f"wrote {args.out_by_country} ({n_country} country/age/task cells "
        f"across {len(by_country)} countries) from {args.trials}"
    )


if __name__ == "__main__":
    main()
