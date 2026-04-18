"""Unit tests for mental-rotation prompt template overrides."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from levante_bench.tasks.mental_rotation import MentalRotationDataset


def test_mental_rotation_uses_task_prompt_override(monkeypatch, tmp_path) -> None:
    prompt_override = (
        "The first image is the reference shape. "
        "The second image is option A. The third image is option B.\n\n"
        "One option is the same shape rotated to a different angle. "
        "The other is a MIRROR image (horizontally flipped).\n"
        "Which option matches the reference - just rotated, NOT flipped?\n\n"
        "Answer with one letter: A or B."
    )
    manifest = pd.DataFrame(
        [
            {
                "item_uid": "mr_item_001",
                "answer": "shape_a",
                "response_alternatives": "shape_b",
                "full_prompt": "default prompt <prompt_phrase>",
                "prompt_phrase": "ignore me",
                "prompt_image": "ref_shape",
            }
        ]
    )

    monkeypatch.setattr(
        MentalRotationDataset,
        "_load_manifest",
        lambda self: manifest,
    )
    monkeypatch.setattr(
        "levante_bench.tasks.mental_rotation.build_image_index",
        lambda _: {
            "shape_a": "/tmp/shape_a.png",
            "shape_b": "/tmp/shape_b.png",
            "ref_shape": "/tmp/ref_shape.png",
        },
    )

    task_def = SimpleNamespace(
        prompt_language="en",
        true_random_option_order=False,
        option_order_run_seed=None,
        mental_rotation_prompt_template=prompt_override,
    )
    dataset = MentalRotationDataset(task_def=task_def, version="v1", data_root=tmp_path)
    trial = dataset[0]

    assert trial["prompt"] == prompt_override
    assert trial["context_image_paths"] == []
