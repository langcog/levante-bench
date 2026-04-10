#!/usr/bin/env python3
"""Summarize token/position signals across shapebias runs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _norm(raw: dict[str, str]) -> dict[str, str]:
    return {str(k or "").strip().lstrip("\ufeff"): str(v or "").strip() for k, v in raw.items() if (k or "").strip()}


def _rate(num: int, den: int) -> float:
    return (num / den) if den else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True, help="Path to shapebias_detailed.csv")
    ap.add_argument("--output", type=Path, required=True, help="Path to decomposition CSV")
    args = ap.parse_args()

    buckets: dict[tuple[str, str, str], dict[str, int]] = {}
    with args.input.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            row = _norm(raw)
            model = row.get("model", "")
            decision_mode = row.get("decision_mode", "")
            prompt = row.get("prompt_condition", "")
            if not model:
                continue
            key = (model, decision_mode, prompt)
            agg = buckets.setdefault(key, {"n": 0, "decisive": 0, "shape": 0, "texture": 0, "ans1": 0, "ans2": 0})
            agg["n"] += 1
            choice = row.get("choice")
            if choice in {"shape", "texture"}:
                agg["decisive"] += 1
            if choice == "shape":
                agg["shape"] += 1
            if choice == "texture":
                agg["texture"] += 1
            ans = row.get("parsed_answer", "")
            if ans == "1":
                agg["ans1"] += 1
            elif ans == "2":
                agg["ans2"] += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "model",
                "decision_mode",
                "prompt_condition",
                "n_rows",
                "decisive_rate",
                "shape_rate_decisive",
                "token2_rate",
            ]
        )
        for (model, dm, prompt), agg in sorted(buckets.items()):
            decisive = agg["decisive"]
            answered = agg["ans1"] + agg["ans2"]
            w.writerow(
                [
                    model,
                    dm,
                    prompt,
                    agg["n"],
                    f"{_rate(decisive, agg['n']):.4f}",
                    f"{_rate(agg['shape'], decisive):.4f}",
                    f"{_rate(agg['ans2'], answered):.4f}",
                ]
            )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
