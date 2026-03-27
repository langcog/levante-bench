"""Vocab dataset. Context: none, options: images of words."""

import re
import random
from pathlib import Path

import pandas as pd

from levante_bench.data.datasets import VLMDataset
from levante_bench.tasks.image_index import build_image_index
from levante_bench.tasks.registry import register_task

LABELS = ["A", "B", "C", "D"]


def _normalize_term(term: str) -> set[str]:
    t = term.strip().lower()
    if not t:
        return set()
    compact = re.sub(r"[^a-z0-9]+", "", t)
    snake = re.sub(r"[^a-z0-9]+", "_", t).strip("_")
    dash = re.sub(r"[^a-z0-9]+", "-", t).strip("-")
    return {
        t,
        t.replace(" ", "_"),
        t.replace(" ", ""),
        t.replace("_", ""),
        snake,
        dash,
        compact,
    }


def _build_image_index(directory: Path) -> dict[str, Path]:
    """Map normalized filename variants to paths, scanned once."""
    index: dict[str, Path] = {}
    for path in directory.iterdir():
        if path.is_file():
            for key in _normalize_term(path.stem):
                index.setdefault(key, path)
    return index


def _resolve_image(term: str, image_index: dict[str, Path]) -> Path | None:
    for candidate in _normalize_term(term):
        if candidate in image_index:
            return image_index[candidate]
    return None


@register_task("vocab")
class VocabDataset(VLMDataset):
    """Reads vocab trials from manifest.csv. Each trial shows 4 images of
    objects; the model picks which matches the spoken word."""

    def __init__(self, task_def, version, data_root=None):
        super().__init__(task_def=task_def, version=version, data_root=data_root)
        self.manifest = self._load_manifest()
        self.image_dir = self.data_root / "assets" / self.version / "visual" / "vocab"
        self.image_index = build_image_index(self.image_dir)

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
            path = _resolve_image(word.strip(), self.image_index)
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
