#!/usr/bin/env python3
"""Estimate model age-equivalency from task accuracy curves by child age."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


LANG_SUFFIX_RE = re.compile(r"[-_]([a-z]{2})$")
LANG_ALLOWLIST = {"en", "de", "es", "fr", "pt", "it"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate age-equivalency by matching each model task accuracy to child "
            "accuracy-by-age curves."
        )
    )
    parser.add_argument("--version", default="v1", help="Results version folder under --results-root.")
    parser.add_argument(
        "--results-root",
        default="results",
        help="Results root directory containing <version>/<model>/summary.csv.",
    )
    parser.add_argument(
        "--human-age-csv",
        default="results/human-accuracy-by-age-lines.csv",
        help="Child age-accuracy CSV path.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.05,
        help="Soft weighting temperature over absolute accuracy gaps.",
    )
    parser.add_argument(
        "--output-csv",
        default="results/comparison/model_age_equivalency_accuracy.csv",
        help="Output CSV path.",
    )
    return parser.parse_args()


def parse_language(model_tag: str) -> str:
    tag = str(model_tag).lower()
    # Common instruction suffixes should not be treated as locale tags.
    if tag.endswith("-it") or tag.endswith("-instruct"):
        return "en"
    m = LANG_SUFFIX_RE.search(str(model_tag))
    if m:
        lang = m.group(1).lower()
        # Only keep known locale suffixes; otherwise default to English.
        if lang in LANG_ALLOWLIST and lang != "it":
            return lang
    return "en"


def parse_age_mid(age_bin: str) -> float | None:
    s = str(age_bin).strip()
    m_single = re.match(r"^(\d+(?:\.\d+)?)$", s)
    if m_single:
        return float(m_single.group(1)) + 0.5
    m = re.match(r"^(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)$", s)
    if not m:
        return None
    lo = float(m.group(1))
    hi = float(m.group(2))
    return (lo + hi) / 2.0


def soft_weights_from_gap(gaps: np.ndarray, temperature: float) -> np.ndarray:
    if gaps.size == 0:
        return gaps
    t = max(float(temperature), 1e-6)
    shifted = gaps - np.nanmin(gaps)
    scores = np.exp(-shifted / t)
    denom = scores.sum()
    if denom <= 0:
        return np.ones_like(scores) / len(scores)
    return scores / denom


def weighted_isotonic_increasing(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted isotonic regression (increasing) via pool-adjacent-violators."""
    if y.size == 0:
        return y
    blocks: list[dict] = []
    for i in range(len(y)):
        wi = float(w[i]) if np.isfinite(w[i]) and w[i] > 0 else 1.0
        yi = float(y[i])
        blocks.append({"start": i, "end": i, "w": wi, "mean": yi})
        while len(blocks) >= 2 and blocks[-2]["mean"] > blocks[-1]["mean"]:
            b2 = blocks.pop()
            b1 = blocks.pop()
            ww = b1["w"] + b2["w"]
            mm = (b1["mean"] * b1["w"] + b2["mean"] * b2["w"]) / ww
            blocks.append(
                {"start": b1["start"], "end": b2["end"], "w": ww, "mean": mm}
            )

    out = np.zeros_like(y, dtype=float)
    for b in blocks:
        out[b["start"] : b["end"] + 1] = b["mean"]
    return out


def extrapolate_age_from_curve(
    age_mid: np.ndarray, acc_curve: np.ndarray, model_accuracy: float
) -> float:
    """Linear extrapolation using edge segment of monotonic curve."""
    if len(age_mid) < 2:
        return float(age_mid[0]) if len(age_mid) == 1 else float("nan")
    if model_accuracy < acc_curve[0]:
        x0, x1 = float(acc_curve[0]), float(acc_curve[1])
        y0, y1 = float(age_mid[0]), float(age_mid[1])
    elif model_accuracy > acc_curve[-1]:
        x0, x1 = float(acc_curve[-2]), float(acc_curve[-1])
        y0, y1 = float(age_mid[-2]), float(age_mid[-1])
    else:
        return float("nan")
    if abs(x1 - x0) < 1e-9:
        return y0
    return y0 + (model_accuracy - x0) * (y1 - y0) / (x1 - x0)


def load_model_accuracy(results_root: Path, version: str) -> pd.DataFrame:
    version_dir = results_root / version
    if not version_dir.exists():
        raise RuntimeError(f"Results version directory not found: {version_dir}")

    rows: list[dict] = []
    for summary_path in sorted(version_dir.rglob("summary.csv")):
        model_dir = summary_path.parent
        # Skip cache or non-run helper paths.
        if any(part.lower() == "cache" for part in model_dir.parts):
            continue
        model_tag = model_dir.name
        df = pd.read_csv(summary_path)
        if not {"task_id", "accuracy"}.issubset(df.columns):
            continue
        for _, r in df.iterrows():
            rows.append(
                {
                    "task": str(r["task_id"]),
                    "model": model_tag,
                    "language": parse_language(model_tag),
                    "model_accuracy": float(r["accuracy"]),
                }
            )
    if not rows:
        raise RuntimeError(f"No summary.csv files found under {version_dir}")
    out = pd.DataFrame(rows).drop_duplicates(subset=["task", "model"], keep="last")
    return out


def load_human_curves(human_age_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(human_age_csv)
    needed = {"age_bin", "task_id", "language", "accuracy"}
    if not needed.issubset(df.columns):
        raise RuntimeError(f"Missing required columns in {human_age_csv}: {sorted(needed)}")
    out = df.rename(columns={"task_id": "task"}).copy()
    out["language"] = out["language"].astype(str).str.lower().str.strip()
    out["age_mid"] = out["age_bin"].map(parse_age_mid)
    out = out.dropna(subset=["age_mid", "accuracy", "task", "language"])
    out["accuracy"] = out["accuracy"].astype(float)
    out = out.sort_values(["task", "language", "age_mid"], kind="stable")
    return out


def main() -> int:
    args = parse_args()
    results_root = Path(args.results_root)
    human_age_csv = Path(args.human_age_csv)
    output_csv = Path(args.output_csv)

    model_acc = load_model_accuracy(results_root=results_root, version=args.version)
    human = load_human_curves(human_age_csv)

    out_rows: list[dict] = []
    for _, mrow in model_acc.iterrows():
        task = str(mrow["task"])
        model = str(mrow["model"])
        lang = str(mrow["language"])
        model_acc_value = float(mrow["model_accuracy"])

        curve = human[(human["task"] == task) & (human["language"] == lang)].copy()
        if curve.empty and lang != "en":
            curve = human[(human["task"] == task) & (human["language"] == "en")].copy()
        if curve.empty:
            continue

        curve = curve.sort_values("age_mid", kind="stable")
        acc_raw = curve["accuracy"].to_numpy(dtype=float)
        age_mid_vec = curve["age_mid"].to_numpy(dtype=float)
        weights = (
            curve["n"].to_numpy(dtype=float)
            if "n" in curve.columns
            else np.ones_like(acc_raw, dtype=float)
        )
        acc_iso = weighted_isotonic_increasing(acc_raw, weights)
        min_curve_acc = float(np.min(acc_iso))
        max_curve_acc = float(np.max(acc_iso))
        below_curve = model_acc_value < min_curve_acc
        above_curve = model_acc_value > max_curve_acc

        gaps = np.abs(acc_iso - model_acc_value)
        nearest_idx = int(np.argmin(gaps))
        nearest = curve.iloc[nearest_idx]
        extrapolated_age = extrapolate_age_from_curve(
            age_mid=age_mid_vec,
            acc_curve=acc_iso,
            model_accuracy=model_acc_value,
        )

        if below_curve:
            soft_age = float(age_mid_vec.min())
            nearest_bin = str(curve.sort_values("age_mid").iloc[0]["age_bin"])
            nearest_child_accuracy = float(acc_iso[np.argmin(age_mid_vec)])
            status = "below_youngest_bin"
        elif above_curve:
            soft_age = float(age_mid_vec.max())
            nearest_bin = str(curve.sort_values("age_mid").iloc[-1]["age_bin"])
            nearest_child_accuracy = float(acc_iso[np.argmax(age_mid_vec)])
            status = "above_oldest_bin"
        else:
            w = soft_weights_from_gap(gaps, temperature=args.temperature)
            soft_age = float(np.sum(w * age_mid_vec))
            nearest_bin = str(nearest["age_bin"])
            nearest_child_accuracy = float(acc_iso[nearest_idx])
            status = "in_range"

        out_rows.append(
            {
                "task": task,
                "model": model,
                "language": lang,
                "model_accuracy": model_acc_value,
                "nearest_age_bin": nearest_bin,
                "nearest_age_mid": float(nearest["age_mid"]),
                "nearest_child_accuracy": nearest_child_accuracy,
                "accuracy_gap": float(np.min(gaps)),
                "soft_age_eq_accuracy": soft_age,
                "extrapolated_age_eq_accuracy": float(extrapolated_age)
                if np.isfinite(extrapolated_age)
                else None,
                "age_eq_status": status,
                "curve_min_accuracy": min_curve_acc,
                "curve_max_accuracy": max_curve_acc,
                "temperature": float(args.temperature),
            }
        )

    out = pd.DataFrame(out_rows).sort_values(["task", "model"], kind="stable")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    print(f"Wrote {output_csv} ({len(out)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
