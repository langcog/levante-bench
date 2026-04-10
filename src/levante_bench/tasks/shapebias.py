"""Shapebias task dataset for native levante-bench run-eval."""

from __future__ import annotations

import random
from pathlib import Path

from levante_bench.data.datasets import VLMDataset
from levante_bench.evaluation.shapebias import (
    load_stimuli_rows,
    load_words,
    validate_stimulus_row,
)
from levante_bench.tasks.registry import register_task


@register_task("shapebias")
class ShapeBiasDataset(VLMDataset):
    """Dataset that expands shapebias stimuli x words into runner trials."""

    def __init__(self, task_def, version, data_root=None):
        super().__init__(task_def=task_def, version=version, data_root=data_root)
        settings = dict(getattr(task_def, "settings", {}) or {})

        default_root = Path(__file__).resolve().parents[4] / "data" / "shapebias_stimuli"
        stim_root = Path(str(settings.get("stim_root", default_root)))
        stim_set = str(settings.get("stim_set", "stimuli_A_auto_contrast"))
        num_stimuli = settings.get("num_stimuli")
        seed = int(settings.get("seed", 42))
        repeats = max(1, int(settings.get("repeats", 1)))
        ordering = str(settings.get("ordering", "both"))
        decision_mode = str(settings.get("decision_mode", "2afc"))
        prompt_condition = str(settings.get("prompt_condition", "noun_label"))
        swap_correct = bool(settings.get("swap_correct", False))
        temperature = float(settings.get("temperature", 0.0))
        max_new_tokens = int(settings.get("max_new_tokens", 128))

        stimuli = load_stimuli_rows(
            stim_root=stim_root,
            stim_set=stim_set,
            num_stimuli=int(num_stimuli) if num_stimuli is not None else None,
            seed=seed,
        )
        for stim in stimuli:
            validate_stimulus_row(stim)

        words = load_words()
        rng = random.Random(seed)
        rows: list[dict] = []

        for repeat_idx in range(1, repeats + 1):
            for stim in stimuli:
                for w in words:
                    if ordering == "both":
                        orderings = ["shape_first", "texture_first"]
                    elif ordering == "random":
                        orderings = [rng.choice(["shape_first", "texture_first"])]
                    else:
                        orderings = [ordering]
                    if decision_mode == "binary_pair":
                        orderings = ["binary_pair"]
                    for ord_name in orderings:
                        if ord_name == "shape_first":
                            a_path = stim["shape_match_path"]
                            b_path = stim["texture_match_path"]
                            a_is = "shape"
                            b_is = "texture"
                            correct = "1"
                        elif ord_name == "texture_first":
                            a_path = stim["texture_match_path"]
                            b_path = stim["shape_match_path"]
                            a_is = "texture"
                            b_is = "shape"
                            correct = "2"
                        else:
                            a_path = stim["shape_match_path"]
                            b_path = stim["texture_match_path"]
                            a_is = "shape"
                            b_is = "texture"
                            correct = "1"
                        trial_id = (
                            f"sb-{stim['stim_id']}-{w['name']}-"
                            f"{ord_name}-r{repeat_idx}"
                        )
                        rows.append(
                            {
                                "trial_id": trial_id,
                                "item_uid": trial_id,
                                "task_id": "shapebias",
                                "stim_id": stim["stim_id"],
                                "word": w["name"],
                                "word_type": w["type"],
                                "word_length": w["length"],
                                "reference_image_path": stim["reference_image_path"],
                                "a_image_path": a_path,
                                "b_image_path": b_path,
                                "ordering": ord_name,
                                "order_method": "random" if ordering == "random" else "deterministic",
                                "a_is": a_is,
                                "b_is": b_is,
                                "prompt_condition": prompt_condition,
                                "decision_mode": decision_mode,
                                "swap_correct": swap_correct,
                                "temperature": temperature,
                                "repeat": repeat_idx,
                                "max_new_tokens": max_new_tokens,
                                "stim_set": stim_set,
                                "prompt": "",  # prompt assembled by shapebias evaluator
                                "options": ["shape", "texture"],
                                "option_labels": ["1", "2"],
                                "correct_label": correct,
                                "context_image_paths": [stim["reference_image_path"]],
                                "option_image_paths": [a_path, b_path],
                                "context_type": "single_image",
                                "option_type": "image",
                            }
                        )
        self._rows = rows

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, idx):
        return dict(self._rows[idx])
