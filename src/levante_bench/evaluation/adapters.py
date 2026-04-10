"""Task-specific postprocessing hooks for runner outputs."""

from __future__ import annotations

import csv
from pathlib import Path

from levante_bench.evaluation.shapebias import CSV_FIELDS as SHAPEBIAS_CSV_FIELDS


def _chance_from_options(options: list[str] | None) -> float:
    n = len(options or [])
    if n <= 0:
        return 0.0
    return 1.0 / n


def _chance_for_trial(trial: dict) -> float:
    chance = trial.get("chance_level")
    try:
        if chance is not None:
            c = float(chance)
            if c > 0:
                return c
    except (TypeError, ValueError):
        pass
    return _chance_from_options(trial.get("options"))


def _write_math_by_type(
    model_dir: Path,
    task_results: list[dict],
    task_trials: list[dict],
) -> Path | None:
    if not task_results or not task_trials or len(task_results) != len(task_trials):
        return None

    by_type: dict[str, dict[str, float]] = {}
    for result, trial in zip(task_results, task_trials, strict=True):
        trial_type = str(trial.get("trial_type") or "UNKNOWN")
        row = by_type.setdefault(
            trial_type,
            {
                "n": 0.0,
                "correct": 0.0,
                "parsed": 0.0,
                "chance_sum": 0.0,
            },
        )
        row["n"] += 1.0
        row["correct"] += 1.0 if bool(result.get("is_correct")) else 0.0
        row["parsed"] += 1.0 if (result.get("predicted_label") is not None or result.get("predicted_value") is not None) else 0.0
        row["chance_sum"] += _chance_for_trial(trial)

    out_path = model_dir / "egma-math-by-type.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["trial_type", "n", "accuracy", "guess_baseline", "lift_vs_guess", "parse_rate"])
        for trial_type, row in sorted(by_type.items(), key=lambda kv: kv[1]["correct"] / kv[1]["n"], reverse=True):
            n = int(row["n"])
            acc = row["correct"] / row["n"] if row["n"] else 0.0
            guess = row["chance_sum"] / row["n"] if row["n"] else 0.0
            parse = row["parsed"] / row["n"] if row["n"] else 0.0
            writer.writerow([trial_type, n, acc, guess, acc - guess, parse])

    return out_path


def _write_tom_by_type(
    model_dir: Path,
    task_results: list[dict],
    task_trials: list[dict],
) -> Path | None:
    if not task_results or not task_trials or len(task_results) != len(task_trials):
        return None

    by_type: dict[str, dict[str, float]] = {}
    for result, trial in zip(task_results, task_trials, strict=True):
        trial_type = str(trial.get("trial_type") or "UNKNOWN")
        row = by_type.setdefault(
            trial_type,
            {
                "n": 0.0,
                "correct": 0.0,
                "parsed": 0.0,
                "chance_sum": 0.0,
            },
        )
        row["n"] += 1.0
        row["correct"] += 1.0 if bool(result.get("is_correct")) else 0.0
        row["parsed"] += 1.0 if (result.get("predicted_label") is not None or result.get("predicted_value") is not None) else 0.0
        row["chance_sum"] += _chance_for_trial(trial)

    out_path = model_dir / "theory-of-mind-by-type.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["trial_type", "n", "accuracy", "guess_baseline", "lift_vs_guess", "parse_rate"])
        for trial_type, row in sorted(by_type.items(), key=lambda kv: kv[1]["correct"] / kv[1]["n"], reverse=True):
            n = int(row["n"])
            acc = row["correct"] / row["n"] if row["n"] else 0.0
            guess = row["chance_sum"] / row["n"] if row["n"] else 0.0
            parse = row["parsed"] / row["n"] if row["n"] else 0.0
            writer.writerow([trial_type, n, acc, guess, acc - guess, parse])

    return out_path


def _write_shapebias_detailed(
    model_dir: Path,
    task_results: list[dict],
) -> Path | None:
    if not task_results:
        return None
    out_path = model_dir / "shapebias_detailed.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=SHAPEBIAS_CSV_FIELDS,
            extrasaction="ignore",
            restval="",
        )
        writer.writeheader()
        writer.writerows(task_results)
    return out_path


def postprocess_task_outputs(
    task_id: str,
    model_dir: Path,
    task_results: list[dict],
    task_trials: list[dict],
) -> list[Path]:
    """Write optional task-specific artifacts after per-task evaluation."""
    outputs: list[Path] = []
    if task_id == "egma-math":
        by_type = _write_math_by_type(model_dir=model_dir, task_results=task_results, task_trials=task_trials)
        if by_type is not None:
            outputs.append(by_type)
    if task_id == "theory-of-mind":
        by_type = _write_tom_by_type(model_dir=model_dir, task_results=task_results, task_trials=task_trials)
        if by_type is not None:
            outputs.append(by_type)
    if task_id == "shapebias":
        detailed = _write_shapebias_detailed(model_dir=model_dir, task_results=task_results)
        if detailed is not None:
            outputs.append(detailed)
    return outputs
