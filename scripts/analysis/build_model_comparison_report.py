#!/usr/bin/env python3
"""Build a detailed JSON report from all results summary.csv files."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
MODEL_SIZE_RE = re.compile(r"^(?P<model>[A-Za-z0-9.-]+)_(?P<size>[0-9]+(?:\.[0-9]+)?[A-Za-z]+)$")


@dataclass
class SummaryRun:
    summary_path: Path
    relative_path: str
    run_id: str
    model: str
    size: str | None
    model_tag: str
    version: str | None
    modified_at: str
    task_metrics: dict[str, float]
    mean_accuracy: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan results/ for summary.csv files and build a model comparison JSON report.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results"),
        help="Results root to scan (default: results).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/model-comparison-report.json"),
        help="Output JSON path (default: results/model-comparison-report.json).",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indentation spaces (default: 2).",
    )
    return parser.parse_args()


def _safe_float(value: str | None) -> float | None:
    if value is None:
        return None
    s = value.strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _parse_summary_csv(path: Path) -> dict[str, float]:
    metrics: dict[str, float] = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            task_id = (row.get("task_id") or "").strip()
            acc = _safe_float(row.get("accuracy"))
            if task_id and acc is not None:
                metrics[task_id] = acc
    return metrics


def _infer_model_tag(relative_summary_path: Path) -> str:
    # Conventionally, summary.csv lives under <model_tag>/summary.csv.
    # Some historical runs append a date folder after model_tag, e.g.
    # runner-math/<ts>/smolvlm2/2026-03-24/summary.csv. In that case, choose
    # the nearest non-date folder before summary.csv.
    parts = list(relative_summary_path.parts[:-1])  # drop summary.csv
    for part in reversed(parts):
        if DATE_RE.fullmatch(part):
            continue
        return part
    return relative_summary_path.parent.name


def _split_model_and_size(model_tag: str) -> tuple[str, str | None]:
    m = MODEL_SIZE_RE.fullmatch(model_tag)
    if m:
        return m.group("model"), m.group("size")
    return model_tag, None


def _infer_version(relative_summary_path: Path) -> str | None:
    # Pick the deepest YYYY-MM-DD component in the path.
    parts = list(relative_summary_path.parts[:-1])  # drop summary.csv
    date_parts = [p for p in parts if DATE_RE.fullmatch(p)]
    return date_parts[-1] if date_parts else None


def _build_run(summary_path: Path, results_root: Path) -> SummaryRun:
    rel = summary_path.relative_to(results_root)
    model_tag = _infer_model_tag(rel)
    model, size = _split_model_and_size(model_tag)
    task_metrics = _parse_summary_csv(summary_path)
    mean_accuracy = (
        sum(task_metrics.values()) / len(task_metrics) if task_metrics else None
    )
    modified_at = datetime.fromtimestamp(
        summary_path.stat().st_mtime, tz=UTC
    ).isoformat()
    run_id = str(rel.parent).replace("\\", "/")
    return SummaryRun(
        summary_path=summary_path,
        relative_path=str(rel).replace("\\", "/"),
        run_id=run_id,
        model=model,
        size=size,
        model_tag=model_tag,
        version=_infer_version(rel),
        modified_at=modified_at,
        task_metrics=task_metrics,
        mean_accuracy=mean_accuracy,
    )


def _aggregate_runs(runs: list[SummaryRun]) -> dict[str, dict]:
    grouped: dict[str, list[SummaryRun]] = defaultdict(list)
    for run in runs:
        key = f"{run.model}|{run.size or ''}"
        grouped[key].append(run)

    out: dict[str, dict] = {}
    for key, group_runs in sorted(grouped.items()):
        model = group_runs[0].model
        size = group_runs[0].size
        task_values: dict[str, list[float]] = defaultdict(list)
        for run in group_runs:
            for task_id, acc in run.task_metrics.items():
                task_values[task_id].append(acc)

        task_stats: dict[str, dict] = {}
        for task_id, values in sorted(task_values.items()):
            task_stats[task_id] = {
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "mean": sum(values) / len(values),
                "values": values,
            }

        out[key] = {
            "model": model,
            "size": size,
            "model_tag_examples": sorted({r.model_tag for r in group_runs}),
            "run_count": len(group_runs),
            "runs": sorted(r.run_id for r in group_runs),
            "versions_seen": sorted({r.version for r in group_runs if r.version}),
            "task_stats": task_stats,
        }
    return out


def main() -> int:
    args = parse_args()
    results_root = args.results_root.resolve()
    output_json = args.output_json.resolve()

    if not results_root.exists():
        raise FileNotFoundError(f"Results root not found: {results_root}")

    summary_paths = sorted(
        [p for p in results_root.rglob("summary.csv") if p.is_file()],
        key=lambda p: p.stat().st_mtime,
    )
    runs = [_build_run(path, results_root) for path in summary_paths]
    runs_payload = [
        {
            "summary_path": str(run.summary_path),
            "relative_path": run.relative_path,
            "run_id": run.run_id,
            "model": run.model,
            "size": run.size,
            "model_tag": run.model_tag,
            "version": run.version,
            "modified_at": run.modified_at,
            "task_metrics": run.task_metrics,
            "mean_accuracy": run.mean_accuracy,
        }
        for run in runs
    ]

    payload = {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "results_root": str(results_root),
        "summary_file_count": len(runs),
        "runs": runs_payload,
        "by_model": _aggregate_runs(runs),
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(payload, indent=args.indent, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {output_json} with {len(runs)} summary file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

