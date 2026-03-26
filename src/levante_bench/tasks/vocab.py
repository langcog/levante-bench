"""Vocab dataset. Context: none, options: images of words."""

import random
from pathlib import Path

import pandas as pd

from levante_bench.data.datasets import VLMDataset
from levante_bench.tasks.registry import register_task

LABELS = ["A", "B", "C", "D"]


def _build_image_index(directory: Path) -> dict[str, Path]:
    """Map image stems to paths, scanned once."""
    index = {}
    for path in directory.iterdir():
        if path.is_file():
            index[path.stem] = path
    return index


@register_task("vocab")
class VocabDataset(VLMDataset):
    """Reads vocab trials from manifest.csv. Each trial shows 4 images of
    objects; the model picks which matches the spoken word."""

    def __init__(self, task_def, version, data_root=None):
        super().__init__(task_def=task_def, version=version, data_root=data_root)
        self.manifest = self._load_manifest()
        self.image_dir = self.data_root / "assets" / self.version / "visual" / "vocab"
        self.image_index = _build_image_index(self.image_dir)

    def _load_manifest(self) -> pd.DataFrame:
        """Load and filter manifest rows for vocab task."""
        manifest_path = self.data_root / "assets" / "manifest.csv"
        df = pd.read_csv(manifest_path)
        df = df[df["task"] == "vocab"]
        df = df[df["trial_type"] == "test"]
        return df.reset_index(drop=True)

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]

        answer = row["answer"]
        alternatives = row["response_alternatives"].split(",")
        all_options = [answer] + alternatives

        # Deterministic shuffle seeded by item_uid
        rng = random.Random(row["item_uid"])
        rng.shuffle(all_options)

        correct_idx = all_options.index(answer)
        correct_label = LABELS[correct_idx]

        # Resolve option image paths from cached index
        option_image_paths = []
        for word in all_options:
            path = self.image_index.get(word.strip())
            if path is None:
                raise FileNotFoundError(
                    f"Image not found for '{word}' in {self.image_dir} "
                    f"(trial {row['item_uid']})"
                )
            option_image_paths.append(str(path))

        # Build prompt from template — keep <imageN> placeholders for interleaving
        prompt_phrase = row["prompt_phrase"]
        prompt = row["full_prompt"]
        prompt = prompt.replace("<prompt_phrase>", str(prompt_phrase))

        return {
            "trial_id": row["item_uid"],
            "item_uid": row["item_uid"],
            "prompt": prompt,
            "options": all_options,
            "option_labels": LABELS[:len(all_options)],
            "correct_label": correct_label,
            "context_image_paths": [],
            "option_image_paths": option_image_paths,
            "context_type": "none",
            "option_type": "image",
        }
