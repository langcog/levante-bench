#!/usr/bin/env python3
"""Build a side-by-side vocab vs synthetic-vocab results table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan result summaries and build a table with vocab and synthetic-vocab accuracy."
        )
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        required=True,
        help="Root directory containing Slurm job outputs (job_*/...).",
    )
    parser.add_argument(
        "--version",
        default="v1",
        help="Dataset version folder under each job root (default: v1).",
    )
    parser.add_argument(
        "--expected-models",
        default="",
        help="Comma-separated model IDs to include even if missing.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Output CSV path (default: <results-root>/vocab_synthetic_table.csv).",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=None,
        help="Output Markdown path (default: <results-root>/vocab_synthetic_table.md).",
    )
    return parser.parse_args()


def _model_key(metadata: dict, summary_path: Path) -> str:
    model = str(metadata.get("model") or "").strip()
    size = str(metadata.get("model_size") or "").strip()
    if model and size:
        return f"{model}-{size}"
    if model:
        return model
    parent = summary_path.parent
    if parent.name == "baseline":
        return summary_path.parent.parent.name
    return parent.name


def _read_summary(path: Path) -> dict[str, float]:
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    out: dict[str, float] = {}
    for row in rows:
        task_id = str(row.get("task_id") or "").strip()
        if not task_id:
            continue
        try:
            out[task_id] = float(row.get("accuracy", "nan"))
        except ValueError:
            continue
    return out


def _scan_latest_by_model(results_root: Path, version: str) -> dict[str, dict]:
    candidates = []
    # job-wrapped outputs
    candidates += list(results_root.glob(f"job_*/{version}/*/summary.csv"))
    candidates += list(results_root.glob(f"job_*/{version}/*/baseline/summary.csv"))
    # direct outputs without job_* wrapper
    candidates += list(results_root.glob(f"{version}/*/summary.csv"))
    candidates += list(results_root.glob(f"{version}/*/baseline/summary.csv"))

    latest: dict[str, dict] = {}
    for summary_path in candidates:
        model_dir = (
            summary_path.parent.parent
            if summary_path.parent.name == "baseline"
            else summary_path.parent
        )
        metadata_path = model_dir / "metadata.json"
        metadata = {}
        if metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except Exception:
                metadata = {}

        key = _model_key(metadata, summary_path)
        mtime = summary_path.stat().st_mtime
        current = latest.get(key)
        if current is None or mtime > current["mtime"]:
            latest[key] = {
                "key": key,
                "summary_path": summary_path,
                "metadata_path": metadata_path if metadata_path.exists() else None,
                "metadata": metadata,
                "mtime": mtime,
                "scores": _read_summary(summary_path),
            }
    return latest


def _build_rows(latest: dict[str, dict], expected_models: list[str]) -> list[dict]:
    rows: list[dict] = []
    keys = set(latest.keys())
    keys.update(expected_models)

    for key in sorted(keys):
        rec = latest.get(key)
        if rec is None:
            rows.append(
                {
                    "model": key,
                    "vocab_accuracy": "",
                    "synthetic_vocab_accuracy": "",
                    "delta_synth_minus_vocab": "",
                    "status": "missing_summary",
                    "summary_path": "",
                }
            )
            continue

        vocab = rec["scores"].get("vocab")
        synth = rec["scores"].get("synthetic-vocab")
        delta = (synth - vocab) if (synth is not None and vocab is not None) else None
        status = "ok" if (vocab is not None and synth is not None) else "missing_task_score"
        rows.append(
            {
                "model": key,
                "vocab_accuracy": f"{vocab:.4f}" if vocab is not None else "",
                "synthetic_vocab_accuracy": f"{synth:.4f}" if synth is not None else "",
                "delta_synth_minus_vocab": f"{delta:+.4f}" if delta is not None else "",
                "status": status,
                "summary_path": str(rec["summary_path"]),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "vocab_accuracy",
        "synthetic_vocab_accuracy",
        "delta_synth_minus_vocab",
        "status",
        "summary_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Vocab vs Synthetic-Vocab Results",
        "",
        "| Model | Vocab | Synthetic-Vocab | Delta (Synth-Vocab - Vocab) | Status |",
        "|---|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['vocab_accuracy'] or '—'} | "
            f"{row['synthetic_vocab_accuracy'] or '—'} | "
            f"{row['delta_synth_minus_vocab'] or '—'} | {row['status']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = _parse_args()
    results_root = args.results_root.resolve()
    expected = [m.strip() for m in str(args.expected_models).split(",") if m.strip()]

    latest = _scan_latest_by_model(results_root, version=str(args.version))
    rows = _build_rows(latest, expected_models=expected)

    output_csv = (
        args.output_csv.resolve()
        if args.output_csv is not None
        else results_root / "vocab_synthetic_table.csv"
    )
    output_md = (
        args.output_md.resolve()
        if args.output_md is not None
        else results_root / "vocab_synthetic_table.md"
    )

    _write_csv(output_csv, rows)
    _write_markdown(output_md, rows)
    print(f"Wrote CSV: {output_csv}")
    print(f"Wrote Markdown: {output_md}")
    print(f"Rows: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
