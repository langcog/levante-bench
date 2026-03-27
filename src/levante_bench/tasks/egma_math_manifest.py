"""EgmaMath dataset. Context: optional image, options: text."""

import random

from pathlib import Path

import pandas as pd

from levante_bench.data.datasets import VLMDataset
from levante_bench.tasks.registry import register_task
from levante_bench.tasks.image_index import build_image_index

LABELS = ["A", "B", "C", "D"]
@register_task("egma-math")
class EgmaMathDataset(VLMDataset):
    """Reads egma-math trials from manifest.csv with text answer choices."""

    def __init__(self, task_def, version, data_root=None):
        super().__init__(task_def=task_def, version=version, data_root=data_root)
        self.manifest = self._load_manifest()
        self.image_dir = self.data_root / "assets" / self.version / "visual" / "egma-math"
        self.image_index = build_image_index(self.image_dir)

    def _load_manifest(self) -> pd.DataFrame:
        """Load and filter manifest rows for egma-math task."""
        manifest_path = self.data_root / "assets" / "manifest.csv"
        df = pd.read_csv(manifest_path)
        df = df[df["task"] == "egma-math"]
        # Number line items are not fully supported yet (see manifest notes).
        df = df[~df["trial_type"].astype(str).str.contains("Number Line", case=False, na=False)]
        return df.reset_index(drop=True)

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]

        answer = str(row["answer"]).strip()
        alternatives = str(row["response_alternatives"] or "").split(",")
        alternatives = [a.strip() for a in alternatives if a.strip()]
        all_options = [answer] + alternatives

        # Deterministic shuffle seeded by item_uid
        rng = random.Random(row["item_uid"])
        rng.shuffle(all_options)

        correct_idx = all_options.index(answer)
        correct_label = LABELS[correct_idx]

        # Build prompt from template.
        # egma-math uses <optionX> placeholders (not <imageX>) so we must substitute option text.
        # Some rows also include <prompt_image> (context image), which we convert to <image0>.
        prompt_phrase = row.get("prompt_phrase")
        prompt_phrase_s = str(prompt_phrase)
        if prompt_phrase_s in {"NA", "nan"}:
            prompt_phrase_s = ""
        prompt = str(row["full_prompt"])

        context_image_paths = []
        if "<prompt_image>" in prompt:
            prompt = prompt.replace("<prompt_image>", "<image0>")
            prompt_image = str(row.get("prompt_image", "NA")).strip()
            if prompt_image and prompt_image not in {"NA", "nan", "TODO"}:
                path = self.image_index.get(prompt_image)
                if path is None:
                    raise FileNotFoundError(
                        f"Prompt image not found for '{prompt_image}' in {self.image_dir} "
                        f"(trial {row['item_uid']})"
                    )
                context_image_paths.append(str(path))

        prompt = prompt.replace("<prompt_phrase>", prompt_phrase_s)
        for i, option in enumerate(all_options, start=1):
            prompt = prompt.replace(f"<option{i}>", str(option))

        return {
            "trial_id": row["item_uid"],
            "item_uid": row["item_uid"],
            "prompt": prompt,
            "options": all_options,
            "option_labels": LABELS[:len(all_options)],
            "correct_label": correct_label,
            "context_image_paths": context_image_paths,
            "option_image_paths": [],
            "context_type": "image" if context_image_paths else "none",
            "option_type": "text",
        }
