"""Write evaluation outputs: per-task CSV and cross-task summary CSV."""

import csv
from pathlib import Path

import numpy as np

_BASE_FIELDS = [
    "trial_id",
    "item_uid",
    "generated_text",
    "reason",
    "predicted_label",
    "predicted_value",
    "predicted_slider_position",
    "parse_method",
    "parse_confidence",
    "parse_raw_candidate",
    "correct_label",
    "target_value",
    "slider_tolerance",
    "is_correct",
]

_HUMAN_FIELDS = [
    "human_correct_prop",
    "human_predicted_prop",
    "human_plurality_label",
    "human_plurality_agrees_model",
]


def write_task_csv(output_dir: Path, task_id: str, results: list[dict]) -> Path:
    """Write per-task detailed results CSV.

    Automatically includes human-comparison columns when present in results.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{task_id}.csv"

    has_human = results and any(
        r.get("human_correct_prop") is not None for r in results
    )
    fieldnames = _BASE_FIELDS + (_HUMAN_FIELDS if has_human else [])

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


def write_task_npy(
    output_dir: Path,
    task_id: str,
    results: list[dict],
    task_trials: list[dict] | None = None,
) -> Path:
    """Write per-task model choices as a .npy matrix.

    The exported matrix has shape [n_items, n_options]. Each row is one-hot over
    option labels according to the model's parsed predicted label. This restores
    compatibility for downstream comparison tooling that expects <task>.npy files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{task_id}.npy"

    if not results:
        np.save(path, np.zeros((0, 0), dtype=np.float32))
        return path

    # Prefer task trial labels to preserve dataset row order/labeling.
    trial_labels = []
    if task_trials:
        for trial in task_trials:
            labels = trial.get("option_labels") or []
            trial_labels.append([str(label).upper() for label in labels])
    else:
        for result in results:
            labels = result.get("option_labels") or []
            trial_labels.append([str(label).upper() for label in labels])

    n_rows = len(results)
    n_cols = max((len(labels) for labels in trial_labels), default=0)
    matrix = np.zeros((n_rows, n_cols), dtype=np.float32)

    for i, result in enumerate(results):
        predicted = str(result.get("predicted_label") or "").strip().upper()
        labels = trial_labels[i] if i < len(trial_labels) else []
        if predicted and predicted in labels:
            j = labels.index(predicted)
            matrix[i, j] = 1.0

    np.save(path, matrix)
    return path
