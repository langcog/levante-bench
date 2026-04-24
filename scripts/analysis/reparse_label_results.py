#!/usr/bin/env python3
"""Re-parse existing label-task result CSVs without rerunning model inference.

Intended for parser improvements that can recover labels from already-generated
`generated_text` outputs (e.g., Qwen truncation heuristics). The script updates
per-task CSV rows and rewrites `summary.csv` per run directory.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from levante_bench.models.qwen35 import Qwen35Model


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-parse existing label task CSVs and refresh summary.csv."
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        required=True,
        help="Root directory containing run folders (e.g., results/v1/qwen35-4B).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without writing files.",
    )
    parser.add_argument(
        "--only-unparseable",
        action="store_true",
        help="Only attempt rows currently marked parse_method=unparseable.",
    )
    return parser.parse_args()


def _option_labels_for_rows(rows: list[dict[str, str]]) -> list[str]:
    labels = sorted(
        {
            str((row.get("correct_label") or "")).strip().upper()
            for row in rows
            if str((row.get("correct_label") or "")).strip()
        }
    )
    return labels if labels else ["A", "B", "C", "D"]


def _bool_from_str(value: str | None) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _is_trial_csv(path: Path) -> bool:
    name = path.name
    if name == "summary.csv":
        return False
    if name.endswith("-by-type.csv"):
        return False
    return name.endswith(".csv")


def _iter_run_dirs(results_root: Path) -> list[Path]:
    run_dirs: list[Path] = []
    if (results_root / "summary.csv").exists():
        run_dirs.append(results_root)
    for summary in results_root.rglob("summary.csv"):
        run_dirs.append(summary.parent)
    # Deduplicate while preserving sorted order.
    return sorted(set(run_dirs))


def _reparse_task_file(path: Path, model: Qwen35Model, only_unparseable: bool, dry_run: bool) -> tuple[int, int, int]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    if not rows:
        return 0, 0, 0

    labels = _option_labels_for_rows(rows)
    changed_rows = 0
    recovered_rows = 0
    parse_method_updates = 0

    for row in rows:
        if row.get("predicted_value"):
            continue
        if only_unparseable and (row.get("parse_method") or "").strip() != "unparseable":
            continue

        prev_label = (row.get("predicted_label") or "").strip().upper()
        prev_method = (row.get("parse_method") or "").strip()
        result = model.parse_answer_result(row.get("generated_text") or "", labels)
        new_label = str(result.value).upper() if result.value is not None else ""

        row["predicted_label"] = new_label
        row["reason"] = result.reason
        row["parse_method"] = result.parse_method
        row["parse_confidence"] = result.parse_confidence
        row["parse_raw_candidate"] = result.raw_candidate
        row["is_correct"] = str(new_label == str(row.get("correct_label") or "").strip().upper())

        changed_rows += 1 if (new_label != prev_label or result.parse_method != prev_method) else 0
        recovered_rows += 1 if (not prev_label and new_label) else 0
        parse_method_updates += 1 if result.parse_method != prev_method else 0

    if not dry_run:
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    return changed_rows, recovered_rows, parse_method_updates


def _rewrite_summary(run_dir: Path, dry_run: bool) -> int:
    task_files = sorted(p for p in run_dir.glob("*.csv") if _is_trial_csv(p))
    if not task_files:
        return 0

    task_acc: list[tuple[str, float]] = []
    for path in task_files:
        with path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
        correct = sum(1 for row in rows if _bool_from_str(row.get("is_correct")))
        task_acc.append((path.stem, float(correct) / float(len(rows))))

    if not dry_run:
        summary_path = run_dir / "summary.csv"
        with summary_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["task_id", "accuracy"])
            for task_id, accuracy in task_acc:
                writer.writerow([task_id, f"{accuracy:.4f}"])

    return len(task_acc)


def main() -> int:
    args = _parse_args()
    results_root = args.results_root.resolve()
    if not results_root.exists():
        raise FileNotFoundError(f"Results root does not exist: {results_root}")

    model = Qwen35Model(model_name="Qwen/Qwen3.5-4B")
    run_dirs = _iter_run_dirs(results_root)
    if not run_dirs:
        print(f"No run directories found under {results_root}")
        return 0

    total_task_files = 0
    total_changed = 0
    total_recovered = 0
    total_method_updates = 0
    total_summaries = 0

    for run_dir in run_dirs:
        task_files = sorted(p for p in run_dir.glob("*.csv") if _is_trial_csv(p))
        if not task_files:
            continue
        run_changed = 0
        run_recovered = 0
        run_updates = 0
        for path in task_files:
            changed, recovered, method_updates = _reparse_task_file(
                path=path,
                model=model,
                only_unparseable=bool(args.only_unparseable),
                dry_run=bool(args.dry_run),
            )
            run_changed += changed
            run_recovered += recovered
            run_updates += method_updates
            total_task_files += 1
        summary_tasks = _rewrite_summary(run_dir=run_dir, dry_run=bool(args.dry_run))
        total_summaries += 1 if summary_tasks > 0 else 0

        total_changed += run_changed
        total_recovered += run_recovered
        total_method_updates += run_updates
        print(
            f"{run_dir}: changed_rows={run_changed} recovered_labels={run_recovered} "
            f"parse_method_updates={run_updates}"
        )

    mode = "DRY-RUN" if args.dry_run else "WRITE"
    print(
        f"[{mode}] run_dirs={len(run_dirs)} task_csvs={total_task_files} "
        f"changed_rows={total_changed} recovered_labels={total_recovered} "
        f"parse_method_updates={total_method_updates} summaries_rewritten={total_summaries}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
