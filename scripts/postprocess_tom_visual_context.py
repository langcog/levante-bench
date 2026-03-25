#!/usr/bin/env python3
"""Deterministically derive ui_context labels for ToM visual description CSVs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fill ui_context using filename and assessment stage heuristics.")
    p.add_argument(
        "--input-csv",
        type=Path,
        default=Path("results/screenshots/theory-of-mind/theory-of-mind-visual-only-descriptions-2.2b.csv"),
    )
    p.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional output path; default overwrites input CSV.",
    )
    return p.parse_args()


def _derive_ui_context(screenshot_file: str, assessment_stage: str, corpus_prompt: str) -> tuple[str, str]:
    f = (screenshot_file or "").strip().lower()
    stage = (assessment_stage or "").strip().lower()
    prompt = (corpus_prompt or "").strip().lower()

    if "initial-load" in f:
        return "start_screen", "filename_initial_load"
    if "task-completed" in f:
        return "completion_screen", "filename_task_completed"
    if "task-started" in f:
        return "transition_screen", "filename_task_started"

    if stage == "instructions":
        if prompt.startswith("nice work! here is a new story"):
            return "transition_screen", "instructions_prompt_transition"
        return "instruction_screen", "assessment_stage_instructions"
    if stage == "test_response":
        return "question_screen", "assessment_stage_test_response"

    return "unknown", "fallback_unknown"


def run(args: argparse.Namespace) -> None:
    in_path = args.input_csv
    out_path = args.output_csv or in_path
    if not in_path.exists():
        raise FileNotFoundError(f"Missing input CSV: {in_path}")

    with open(in_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        fieldnames = list(rows[0].keys()) if rows else []

    if "ui_context_source" not in fieldnames:
        fieldnames.append("ui_context_source")

    counts: dict[str, int] = {}
    for r in rows:
        ui, source = _derive_ui_context(
            screenshot_file=r.get("screenshot_file", ""),
            assessment_stage=r.get("assessment_stage", ""),
            corpus_prompt=r.get("corpus_prompt", ""),
        )
        r["ui_context"] = ui
        r["ui_context_source"] = source
        counts[ui] = counts.get(ui, 0) + 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(
        json.dumps(
            {
                "input_csv": str(in_path),
                "output_csv": str(out_path),
                "n_rows": len(rows),
                "ui_context_counts": counts,
            },
            indent=2,
        )
    )


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()

