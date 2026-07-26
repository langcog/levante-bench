"""Same-Different Selection (keyed test-dimensions single-select)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from levante_bench.data.datasets import VLMDataset
from levante_bench.prompts import render_prompt_template_for_row
from levante_bench.tasks.image_index import build_image_index
from levante_bench.tasks.option_order import (
    derive_true_random_item_seed,
    deterministic_option_order,
)
from levante_bench.tasks.registry import register_task

LABELS = ["A", "B", "C", "D"]


@register_task("same-different-selection")
class SameDifferentSelectionDataset(VLMDataset):
    """Keyed SDS test-dimensions items (3-AFC/4-AFC) from the item bank."""

    def __init__(self, task_def, version, data_root=None):
        super().__init__(task_def=task_def, version=version, data_root=data_root)
        self.manifest = self._load_manifest()
        self.image_dir = (
            self.data_root
            / "assets"
            / self.version
            / "visual"
            / "same-different-selection"
        )
        self.image_index = build_image_index(self.image_dir)

    def _corpus_path(self) -> Path:
        corpus_file = str(
            self.task_def.corpus_file or "same-different-selection-item-bank.csv"
        )
        return (
            self.data_root
            / "assets"
            / self.version
            / "corpus"
            / "same-different-selection"
            / corpus_file
        )

    def _load_manifest(self) -> pd.DataFrame:
        path = self._corpus_path()
        if not path.exists():
            raise FileNotFoundError(f"SDS corpus missing: {path}")
        df = pd.read_csv(path)
        if "required_selections" in df.columns:
            df = df[df["required_selections"].astype(str).str.strip() == "1"]
        if "trial_type" in df.columns:
            df = df[df["trial_type"].astype(str).str.strip() == "test-dimensions"]
        df = df[
            df["answer"].notna()
            & df["response_alternatives"].notna()
            & df["item"].notna()
        ]
        return df.reset_index(drop=True)

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]
        answer = str(row["answer"]).split(",")[0].strip()
        alternatives = [
            value.strip()
            for value in str(row["response_alternatives"]).split(",")
            if value.strip()
        ]
        item_id = str(row.get("item_id") or row.get("audio_file") or "").strip()
        item_uid = item_id or str(row.get("item_uid") or f"sds-{idx}").strip()
        instruction = str(row["item"]).strip()

        labels = LABELS[: 1 + len(alternatives)]
        true_random = bool(getattr(self.task_def, "true_random_option_order", False))
        run_seed = getattr(self.task_def, "option_order_run_seed", None)
        true_random_seed = (
            derive_true_random_item_seed(run_seed=int(run_seed), item_key=item_uid)
            if true_random and run_seed is not None
            else None
        )
        all_options, correct_label, option_order_seed = deterministic_option_order(
            answer=answer,
            alternatives=alternatives,
            seed_value=item_uid,
            option_labels=labels,
            true_random=true_random,
            true_random_seed=true_random_seed,
        )

        option_image_paths = []
        for option in all_options:
            path = self.image_index.get(option.strip())
            if path is None:
                raise FileNotFoundError(
                    f"Image not found for '{option}' in {self.image_dir} "
                    f"(trial {item_uid})"
                )
            option_image_paths.append(str(path))

        # Prompt templates assume up to 4 image slots; pad unused ones.
        placeholders = {
            "prompt_phrase": instruction,
            "labels": ", ".join(labels[:-1] + [f"or {labels[-1]}"])
            if len(labels) > 1
            else labels[0],
            "image1": "<image1>",
            "image2": "<image2>",
            "image3": "<image3>",
            "image4": "<image4>",
        }
        prompt = render_prompt_template_for_row(
            "same-different-selection",
            row,
            prompt_language=self.prompt_language,
            placeholders=placeholders,
        )
        # Drop unused trailing option placeholders for 3-AFC items.
        if len(labels) == 3:
            prompt = prompt.replace("; D: <image4>", "").replace("D: <image4>", "")
            prompt = prompt.replace("A, B, C, or D", "A, B, or C")

        return {
            "trial_id": item_uid,
            "item_uid": item_uid,
            "prompt": prompt,
            "options": all_options,
            "option_labels": labels,
            "correct_label": correct_label,
            "option_order_seed": option_order_seed,
            "context_image_paths": [],
            "option_image_paths": option_image_paths,
            "context_type": "none",
            "option_type": "image",
        }
