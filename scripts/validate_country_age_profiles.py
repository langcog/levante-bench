#!/usr/bin/env python3
"""List age×country×task coverage for Child Twins grid authoring.

Reads shared/persona/age_task_{accuracy,ability}.json and the *_by_country
variants. For each requested age×country×task cell, prints θ / accuracy and
flags when the country table is missing (would fall back to global in QA).

Usage:
    python scripts/validate_country_age_profiles.py
    python scripts/validate_country_age_profiles.py --ages 6,8,10 --countries de,co,ca
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_AGES = [6, 8, 10]
DEFAULT_COUNTRIES = ["de", "co", "ca"]
DEFAULT_TASKS = [
    "egma-math",
    "matrix-reasoning",
    "mental-rotation",
    "theory-of-mind",
    "trog",
    "vocab",
    "same-different-selection",
    "hearts-and-flowers",
    "memory-game",
]


def nearest_age_key(table: dict, age: int) -> str | None:
    ages = [int(k) for k in table if str(k).lstrip("-").isdigit()]
    if not ages:
        return None
    best = min(ages, key=lambda a: abs(a - age))
    return str(best)


def load(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text())


def main() -> None:
    here = Path(__file__).resolve().parent.parent
    persona = here / "shared" / "persona"
    ap = argparse.ArgumentParser()
    ap.add_argument("--ages", default=",".join(map(str, DEFAULT_AGES)))
    ap.add_argument("--countries", default=",".join(DEFAULT_COUNTRIES))
    ap.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    ap.add_argument("--persona-dir", type=Path, default=persona)
    args = ap.parse_args()

    ages = [int(x) for x in args.ages.split(",") if x.strip()]
    countries = [c.strip().lower() for c in args.countries.split(",") if c.strip()]
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]

    acc_g = load(args.persona_dir / "age_task_accuracy.json")
    ab_g = load(args.persona_dir / "age_task_ability.json")
    acc_c = load(args.persona_dir / "age_task_accuracy_by_country.json")
    ab_c = load(args.persona_dir / "age_task_ability_by_country.json")
    for meta_key in ("_meta",):
        acc_c.pop(meta_key, None)
        ab_c.pop(meta_key, None)

    fallbacks = 0
    rows = 0
    print(
        f"{'country':<8} {'age':>3} {'task':<28} {'acc':>7} {'theta':>8} {'src':<10}"
    )
    print("-" * 72)
    for country in countries:
        for age in ages:
            for task in tasks:
                rows += 1
                c_acc = (acc_c.get(country) or {}).get(task) or {}
                c_ab = (ab_c.get(country) or {}).get(task) or {}
                g_acc = acc_g.get(task) or {}
                g_ab = ab_g.get(task) or {}

                if c_acc:
                    ak = nearest_age_key(c_acc, age)
                    acc = c_acc.get(ak) if ak else None
                    src = "country"
                else:
                    ak = nearest_age_key(g_acc, age)
                    acc = g_acc.get(ak) if ak else None
                    src = "GLOBAL"
                    fallbacks += 1

                if c_ab:
                    bk = nearest_age_key(c_ab, age)
                    cell = c_ab.get(bk) if bk else None
                    theta = cell.get("theta") if isinstance(cell, dict) else None
                    if src == "GLOBAL":
                        src = "mixed"
                else:
                    bk = nearest_age_key(g_ab, age)
                    cell = g_ab.get(bk) if bk else None
                    theta = cell.get("theta") if isinstance(cell, dict) else None

                acc_s = f"{acc:.4f}" if isinstance(acc, (int, float)) else "—"
                th_s = f"{theta:.4f}" if isinstance(theta, (int, float)) else "—"
                flag = " !" if src in ("GLOBAL", "mixed") and not c_acc else ""
                print(
                    f"{country:<8} {age:>3} {task:<28} {acc_s:>7} {th_s:>8} {src:<10}{flag}"
                )

    print("-" * 72)
    print(
        f"{rows} cells | {fallbacks} accuracy cells falling back to global "
        f"(flagged with ! when country accuracy missing)"
    )
    if fallbacks:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
