#!/usr/bin/env python3
"""Analyze side/ordering bias from shapebias_detailed.csv outputs."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def _norm_row(raw: dict[str, str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in raw.items():
        nk = (k or "").strip().lstrip("\ufeff")
        if not nk:
            continue
        out[nk] = (v or "").strip()
    return out


def _load_rows(path: Path, model: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            row = _norm_row(raw)
            if row.get("model") != model:
                continue
            if row.get("ordering") not in {"shape_first", "texture_first"}:
                continue
            rows.append(row)
    return rows


def _ratio(num: int, den: int) -> float:
    return num / den if den else float("nan")


def _fmt_pct(x: float) -> str:
    if math.isnan(x):
        return "N/A"
    return f"{100.0 * x:.1f}%"


def _second_pick(row: dict[str, str]) -> bool | None:
    choice = row.get("choice")
    ordering = row.get("ordering")
    if choice not in {"shape", "texture"}:
        return None
    if ordering == "shape_first":
        return choice == "texture"
    if ordering == "texture_first":
        return choice == "shape"
    return None


def analyze(rows: list[dict[str, str]]) -> dict[str, float | int | str]:
    decisive = [r for r in rows if r.get("choice") in {"shape", "texture"}]
    sf = [r for r in decisive if r.get("ordering") == "shape_first"]
    tf = [r for r in decisive if r.get("ordering") == "texture_first"]

    n_ans = sum(1 for r in decisive if r.get("parsed_answer") in {"1", "2"})
    n_1 = sum(1 for r in decisive if r.get("parsed_answer") == "1")
    n_2 = sum(1 for r in decisive if r.get("parsed_answer") == "2")

    second_bools = [x for x in (_second_pick(r) for r in decisive) if x is not None]
    n_second = sum(1 for x in second_bools if x)
    n_second_den = len(second_bools)

    sf_shape = sum(1 for r in sf if r.get("choice") == "shape")
    tf_shape = sum(1 for r in tf if r.get("choice") == "shape")
    sf_rate = _ratio(sf_shape, len(sf))
    tf_rate = _ratio(tf_shape, len(tf))
    order_gap = abs(sf_rate - tf_rate) if sf and tf else float("nan")
    adjusted_shape = 0.5 * (sf_rate + tf_rate) if sf and tf else float("nan")
    adjusted_idx = adjusted_shape - 0.5 if not math.isnan(adjusted_shape) else float("nan")

    if not math.isnan(order_gap) and not math.isnan(_ratio(n_second, n_second_den)):
        if order_gap >= 0.25 or _ratio(n_second, n_second_den) >= 0.60:
            label = "strong_side_bias"
        elif order_gap >= 0.15 or _ratio(n_second, n_second_den) >= 0.55:
            label = "moderate_side_bias"
        else:
            label = "low_side_bias"
    else:
        label = "insufficient_data"

    return {
        "n_total": len(rows),
        "n_decisive": len(decisive),
        "n_unclear": len(rows) - len(decisive),
        "n_answered_1or2": n_ans,
        "option1_count": n_1,
        "option2_count": n_2,
        "option2_rate": _ratio(n_2, n_ans),
        "second_pick_count": n_second,
        "n_second_denom": n_second_den,
        "second_pick_rate": _ratio(n_second, n_second_den),
        "shape_rate_shape_first": sf_rate,
        "shape_rate_texture_first": tf_rate,
        "order_gap_abs": order_gap,
        "adjusted_shape_rate_counterbalanced": adjusted_shape,
        "adjusted_shape_bias_index": adjusted_idx,
        "side_bias_label": label,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True, help="Path to shapebias_detailed.csv")
    ap.add_argument("--model", required=True, help="Model key in CSV (e.g. smolvlm2)")
    args = ap.parse_args()

    rows = _load_rows(args.input, args.model)
    if not rows:
        raise SystemExit(f"No shapebias rows found for model={args.model!r} in {args.input}")

    out = analyze(rows)
    print(f"Model: {args.model}")
    print(f"Input: {args.input}")
    print("")
    print(f"Decisive rows: {out['n_decisive']}/{out['n_total']} (unclear={out['n_unclear']})")
    print(
        "1/2 counts: "
        f"1={out['option1_count']} 2={out['option2_count']} "
        f"(2-rate={_fmt_pct(float(out['option2_rate']))})"
    )
    print(
        "Second-pick rate: "
        f"{out['second_pick_count']}/{out['n_second_denom']} "
        f"({_fmt_pct(float(out['second_pick_rate']))})"
    )
    print(
        "Shape rate by ordering: "
        f"shape_first={_fmt_pct(float(out['shape_rate_shape_first']))}, "
        f"texture_first={_fmt_pct(float(out['shape_rate_texture_first']))}, "
        f"gap={_fmt_pct(float(out['order_gap_abs']))}"
    )
    print(
        "Adjusted score: "
        f"{_fmt_pct(float(out['adjusted_shape_rate_counterbalanced']))} "
        f"(bias_index={float(out['adjusted_shape_bias_index']):+.3f})"
    )
    print(f"Bias label: {out['side_bias_label']}")


if __name__ == "__main__":
    main()
