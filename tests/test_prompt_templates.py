"""Tests for centralized prompt template rendering."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from levante_bench.data.schema import TaskDef
from levante_bench.prompts import render_prompt_template_for_row
from levante_bench.tasks.theory_of_mind_manifest import TheoryOfMindDataset
from levante_bench.tasks.vocab import VocabDataset


TEMPLATE_TASKS = {
    "egma-math",
    "matrix-reasoning",
    "mental-rotation",
    "theory-of-mind",
    "trog",
    "vocab",
}


def test_central_templates_reproduce_manifest_full_prompts() -> None:
    manifest = pd.read_csv("data/assets/manifest.csv", dtype=str).fillna("")
    checked = 0

    for _, row in manifest[manifest["task"].isin(TEMPLATE_TASKS)].iterrows():
        rendered = render_prompt_template_for_row(
            str(row["task"]),
            row,
            prompt_language="en",
        )
        assert rendered == str(row["full_prompt"])
        checked += 1

    assert checked > 0


def test_vocab_dataset_uses_central_localized_frame_and_item_phrase() -> None:
    task_def = TaskDef(
        task_id="vocab",
        benchmark_name="vocab",
        internal_name="vocab",
        prompt_language="es",
    )

    dataset = VocabDataset(task_def=task_def, version="v1", data_root=Path("data"))
    trial = next(dataset[i] for i in range(len(dataset)) if dataset[i]["trial_id"] == "vocab__acorn")

    assert 'Elige la imagen que corresponde al texto: "la bellota".' in trial["prompt"]
    assert "Choose the image that matches the text" not in trial["prompt"]


def test_theory_of_mind_uses_itembank_translation_with_central_answer_suffix() -> None:
    task_def = TaskDef(
        task_id="theory-of-mind",
        benchmark_name="Stories",
        internal_name="theory-of-mind",
        prompt_language="es",
    )

    dataset = TheoryOfMindDataset(task_def=task_def, version="v1", data_root=Path("data"))
    trial = dataset[0]

    assert "Esta es Marisol." in trial["prompt"]
    assert "¿Dónde buscará Marisol su libro primero?" in trial["prompt"]
    assert "Responde con A o B. A: <image1>; B: <image2>" in trial["prompt"]
    assert "Where will Madison" not in trial["prompt"]
