#!/usr/bin/env python3
"""Compute compact shapebias validity summary from detailed CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _norm(raw: dict[str, str]) -> dict[str, str]:
    return {str(k or "").strip().lstrip("\ufeff"): str(v or "").strip() for k, v in raw.items() if (k or "").strip()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True, help="Path to shapebias_detailed.csv")
    ap.add_argument("--output", type=Path, required=True, help="Path to output summary CSV")
    args = ap.parse_args()

    by_model: dict[str, dict[str, int]] = {}
    with args.input.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            row = _norm(raw)
            model = row.get("model", "")
            if not model:
                continue
            agg = by_model.setdefault(
                model,
                {"total": 0, "shape": 0, "texture": 0, "unclear": 0, "shape_first": 0, "texture_first": 0, "shape_on_sf": 0, "shape_on_tf": 0},
            )
            agg["total"] += 1
            choice = row.get("choice")
            if choice in {"shape", "texture", "unclear"}:
                agg[choice] += 1
            ordering = row.get("ordering")
            if ordering == "shape_first":
                agg["shape_first"] += 1
                if choice == "shape":
                    agg["shape_on_sf"] += 1
            elif ordering == "texture_first":
                agg["texture_first"] += 1
                if choice == "shape":
                    agg["shape_on_tf"] += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "model",
                "n_total",
                "n_shape",
                "n_texture",
                "n_unclear",
                "shape_rate_decisive",
                "shape_rate_shape_first",
                "shape_rate_texture_first",
                "adjusted_shape_rate_counterbalanced",
            ]
        )
        for model, agg in sorted(by_model.items()):
            decisive = agg["shape"] + agg["texture"]
            sr = (agg["shape"] / decisive) if decisive else 0.0
            sf = (agg["shape_on_sf"] / agg["shape_first"]) if agg["shape_first"] else 0.0
            tf = (agg["shape_on_tf"] / agg["texture_first"]) if agg["texture_first"] else 0.0
            adjusted = 0.5 * (sf + tf) if agg["shape_first"] and agg["texture_first"] else sr
            w.writerow([model, agg["total"], agg["shape"], agg["texture"], agg["unclear"], f"{sr:.4f}", f"{sf:.4f}", f"{tf:.4f}", f"{adjusted:.4f}"])

    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
