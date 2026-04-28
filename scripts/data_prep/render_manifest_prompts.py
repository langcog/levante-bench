#!/usr/bin/env python3
"""Render manifest full_prompt values from centralized prompt templates."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from levante_bench.data.schema import TaskDef
from levante_bench.prompts import infer_prompt_template_id, render_prompt_template_for_row
from levante_bench.tasks.theory_of_mind_manifest import TheoryOfMindDataset


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate the manifest full_prompt compatibility column from "
            "configs/prompts/*.yaml central templates."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/assets/manifest.csv"),
        help="Input manifest CSV path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output CSV path. Use a separate path for review before publishing.",
    )
    parser.add_argument(
        "--prompt-language",
        default="en",
        help="Prompt locale to render. Defaults to English compatibility prompts.",
    )
    parser.add_argument("--version", default="v1", help="Asset/corpus version to use.")
    parser.add_argument("--data-root", type=Path, default=Path("data"), help="Local data root.")
    parser.add_argument(
        "--add-template-id",
        action="store_true",
        help="Add or refresh the prompt_template_id column in the output.",
    )
    return parser.parse_args()


def _render_theory_of_mind_prompts(args: argparse.Namespace) -> dict[str, str]:
    task_def = TaskDef(
        task_id="theory-of-mind",
        benchmark_name="Stories",
        internal_name="theory-of-mind",
        prompt_language=str(args.prompt_language),
    )
    dataset = TheoryOfMindDataset(
        task_def=task_def,
        version=str(args.version),
        data_root=args.data_root,
    )
    return {str(dataset[i]["item_uid"]): str(dataset[i]["prompt"]) for i in range(len(dataset))}


def main() -> int:
    args = _parse_args()
    manifest = pd.read_csv(args.manifest, dtype=str).fillna("")
    rendered = manifest.copy()
    tom_prompts = _render_theory_of_mind_prompts(args)

    for index, row in rendered.iterrows():
        task_id = str(row.get("task", "")).strip()
        if not task_id:
            continue

        try:
            template_id = infer_prompt_template_id(task_id, row)
            if task_id == "theory-of-mind":
                full_prompt = tom_prompts.get(str(row.get("item_uid", "")).strip(), "")
                if not full_prompt:
                    continue
            else:
                full_prompt = render_prompt_template_for_row(
                    task_id,
                    row,
                    prompt_language=args.prompt_language,
                )
        except (FileNotFoundError, KeyError, ValueError):
            continue

        rendered.at[index, "full_prompt"] = full_prompt
        if args.add_template_id:
            rendered.at[index, "prompt_template_id"] = template_id

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered.to_csv(args.output, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
