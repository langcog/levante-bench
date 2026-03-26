"""Write evaluation outputs: per-task CSV and cross-task summary CSV."""

import csv
from pathlib import Path


def write_task_csv(output_dir: Path, task_id: str, results: list[dict]) -> Path:
    """Write per-task detailed results CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{task_id}.csv"
    fieldnames = ["trial_id", "item_uid", "generated_text", "predicted_label", "correct_label", "is_correct"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    return path


def write_summary_csv(output_dir: Path, task_accuracies: dict[str, float]) -> Path:
    """Write cross-task summary CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "summary.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["task_id", "accuracy"])
        for task_id, accuracy in task_accuracies.items():
            writer.writerow([task_id, f"{accuracy:.4f}"])
    return path
