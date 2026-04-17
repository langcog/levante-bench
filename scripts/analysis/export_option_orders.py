#!/usr/bin/env python3
"""Export per-trial option ordering for default registered tasks."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from levante_bench.config.defaults import detect_data_version
from levante_bench.config.tasks import get_task_def, list_tasks
from levante_bench.tasks import get_task_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Write machine-readable option ordering for each trial in each task "
            "using the current default task registry/code paths."
        )
    )
    parser.add_argument(
        "--version",
        default="current",
        help="Dataset/assets version (default: current).",
    )
    parser.add_argument(
        "--data-root",
        default="data",
        help="Data root passed to task datasets (default: data).",
    )
    parser.add_argument(
        "--task",
        action="append",
        dest="tasks",
        help="Task id to include (repeatable). Default: all registered tasks.",
    )
    parser.add_argument(
        "--output",
        default="results/analysis/option_orders_default.json",
        help="Output JSON path (default: results/analysis/option_orders_default.json).",
    )
    return parser.parse_args()


def _resolve_version(version_arg: str, data_root: Path) -> str:
    if str(version_arg).strip().lower() == "current":
        return detect_data_version(data_root)
    return str(version_arg)


def _trial_option_rows(trial: dict[str, Any]) -> list[dict[str, str]]:
    options = [str(v) for v in (trial.get("options") or [])]
    labels = [str(v) for v in (trial.get("option_labels") or [])]
    return [
        {"label": label, "option": option}
        for label, option in zip(labels, options)
    ]


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]

    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = project_root / data_root
    version = _resolve_version(args.version, data_root)

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = project_root / output_path

    tasks = args.tasks or list_tasks()
    tasks = sorted(set(tasks))

    payload: dict[str, Any] = {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "version": version,
        "data_root": str(data_root),
        "tasks": {},
    }

    for task_id in tasks:
        task_def = get_task_def(task_id, version, data_root=data_root, task_overrides={})
        if task_def is None:
            payload["tasks"][task_id] = {
                "error": "task_def_not_found",
                "trial_count": 0,
                "trials": [],
            }
            continue

        dataset_cls = get_task_dataset(task_id)
        if dataset_cls is None:
            payload["tasks"][task_id] = {
                "error": "dataset_not_registered",
                "trial_count": 0,
                "trials": [],
            }
            continue

        dataset = dataset_cls(task_def=task_def, version=version, data_root=data_root)
        trials_payload: list[dict[str, Any]] = []
        for idx in range(len(dataset)):
            trial = dataset[idx]
            trials_payload.append(
                {
                    "index": idx,
                    "trial_id": str(trial.get("trial_id", "")),
                    "item_uid": str(trial.get("item_uid", "")),
                    "correct_label": str(trial.get("correct_label", "")),
                    "option_order": _trial_option_rows(trial),
                }
            )

        payload["tasks"][task_id] = {
            "error": None,
            "trial_count": len(trials_payload),
            "trials": trials_payload,
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

