#!/usr/bin/env python3
"""Run run-comparison for models/tasks when KL outputs are missing or stale."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_TASKS = [
    "egma-math",
    "matrix-reasoning",
    "mental-rotation",
    "theory-of-mind",
    "trog",
    "vocab",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate KL comparison CSVs only when needed."
    )
    parser.add_argument("--version", default="v1", help="Dataset/results version.")
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Results root used by run-comparison (default: results).",
    )
    parser.add_argument(
        "--comparison-dir",
        default="results/comparison",
        help="Directory where *_d_kl.csv files are written.",
    )
    parser.add_argument(
        "--task",
        action="append",
        dest="tasks",
        help="Task id (repeatable). Defaults to all benchmark tasks.",
    )
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        help="Model directory name under results/<version>/ (repeatable).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing.",
    )
    return parser.parse_args()


def discover_models(results_root: Path) -> list[str]:
    if not results_root.exists():
        return []
    models = []
    for p in sorted(results_root.iterdir()):
        if p.is_dir() and (p / "summary.csv").exists():
            models.append(p.name)
    return models


def needs_refresh(npy_path: Path, dkl_path: Path) -> bool:
    if not npy_path.exists():
        return False
    if not dkl_path.exists():
        return True
    return dkl_path.stat().st_mtime < npy_path.stat().st_mtime


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    results_root = project_root / args.results_dir / args.version
    comparison_dir = project_root / args.comparison_dir
    tasks = args.tasks or DEFAULT_TASKS
    models = args.models or discover_models(results_root)

    if not models:
        print(f"No models found under {results_root}", file=sys.stderr)
        return 1

    work_items: list[tuple[str, str]] = []
    for model in models:
        model_dir = results_root / model
        for task in tasks:
            npy_path = model_dir / f"{task}.npy"
            dkl_path = comparison_dir / f"{task}_{model}_d_kl.csv"
            if needs_refresh(npy_path, dkl_path):
                work_items.append((task, model))

    if not work_items:
        print("All requested KL comparison outputs are up to date.")
        return 0

    print(f"Will run {len(work_items)} KL comparison job(s).")
    for task, model in work_items:
        cmd = [
            sys.executable,
            "-m",
            "levante_bench.cli",
            "run-comparison",
            "--task",
            task,
            "--model",
            model,
            "--version",
            args.version,
            "--results-dir",
            args.results_dir,
            "--output-dir",
            args.comparison_dir,
        ]
        print(" ".join(cmd))
        if args.dry_run:
            continue
        subprocess.run(cmd, check=True, cwd=project_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
