#!/usr/bin/env python3
"""Backfill <task>.npy files from existing per-task CSV outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create missing .npy task outputs from result CSVs."
    )
    parser.add_argument("--version", default="v1", help="Results version folder.")
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Results root directory (default: results).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing .npy files.",
    )
    return parser.parse_args()


def task_csvs_for_model(model_dir: Path) -> list[Path]:
    out: list[Path] = []
    for p in sorted(model_dir.glob("*.csv")):
        name = p.name
        if name == "summary.csv":
            continue
        if name.endswith("-by-type.csv"):
            continue
        out.append(p)
    return out


def infer_labels(df: pd.DataFrame) -> list[str]:
    labels: list[str] = []
    for col in ("predicted_label", "correct_label"):
        if col not in df.columns:
            continue
        for val in df[col].dropna():
            s = str(val).strip().upper()
            if len(s) == 1 and "A" <= s <= "Z" and s not in labels:
                labels.append(s)
    if not labels:
        return ["A", "B", "C", "D"]
    labels.sort()
    return labels


def csv_to_npy(csv_path: Path, npy_path: Path) -> None:
    df = pd.read_csv(csv_path)
    labels = infer_labels(df)
    label_to_idx = {label: i for i, label in enumerate(labels)}
    arr = np.zeros((len(df), len(labels)), dtype=np.float32)
    if "predicted_label" in df.columns:
        for i, val in enumerate(df["predicted_label"]):
            s = str(val).strip().upper() if pd.notna(val) else ""
            j = label_to_idx.get(s)
            if j is not None:
                arr[i, j] = 1.0
    np.save(npy_path, arr)


def main() -> int:
    args = parse_args()
    root = Path(args.results_dir) / args.version
    if not root.exists():
        raise FileNotFoundError(f"Results version not found: {root}")

    wrote = 0
    skipped = 0
    for model_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for csv_path in task_csvs_for_model(model_dir):
            task_id = csv_path.stem
            npy_path = model_dir / f"{task_id}.npy"
            if npy_path.exists() and not args.force:
                skipped += 1
                continue
            csv_to_npy(csv_path, npy_path)
            wrote += 1

    print(f"Backfill complete: wrote={wrote}, skipped_existing={skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
