"""Synthetic vocab dataset authored under scripts/new_vocab_assets."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from levante_bench.data.datasets import VLMDataset
from levante_bench.tasks.option_order import (
    derive_true_random_item_seed,
    deterministic_option_order,
)
from levante_bench.tasks.registry import register_task
from levante_bench.tasks.vocab import _normalize_term

LABELS = ["A", "B", "C", "D"]


def _build_image_index_recursive(directory: Path) -> dict[str, Path]:
    """Map normalized filename variants to image paths, scanned recursively."""
    index: dict[str, Path] = {}
    image_suffixes = {".png", ".jpg", ".jpeg", ".webp"}
    for path in directory.rglob("*"):
        if path.is_file() and path.suffix.lower() in image_suffixes:
            for key in _normalize_term(path.stem):
                index.setdefault(key, path)
    return index


def _resolve_image(term: str, image_index: dict[str, Path]) -> Path | None:
    for candidate in _normalize_term(term):
        if candidate in image_index:
            return image_index[candidate]
    return None


@register_task("synthetic-vocab")
class SyntheticVocabDataset(VLMDataset):
    """Synthetic-vocab trials backed by scripts/new_vocab_assets artifacts."""

    def __init__(self, task_def, version, data_root=None):
        super().__init__(task_def=task_def, version=version, data_root=data_root)
        self.assets_root = (
            Path(__file__).resolve().parent.parent.parent.parent
            / "scripts"
            / "new_vocab_assets"
            / "assets"
            / "new-vocab-2026-04-24"
        )
        self.manifest = self._load_manifest()
        self.translation_rows = self._load_translations()
        self.image_dir = self._resolve_image_dir()
        self.image_index = _build_image_index_recursive(self.image_dir)

    def _load_manifest(self) -> pd.DataFrame:
        path = self.assets_root / "manifest.csv"
        if not path.exists():
            raise FileNotFoundError(f"Synthetic vocab manifest missing: {path}")
        df = pd.read_csv(path)
        df = df[df["task"] == "vocab"]
        return df.reset_index(drop=True)

    def _load_translations(self) -> pd.DataFrame:
        path = self.assets_root / "translations" / "item-bank-translations.csv"
        if not path.exists():
            return pd.DataFrame(columns=["item_uid", "language", "full_prompt", "prompt_phrase"])
        return pd.read_csv(path, dtype=str).fillna("")

    def _resolve_image_dir(self) -> Path:
        candidates = [
            self.assets_root / "images",
            self.assets_root / "visual" / "vocab",
            self.assets_root,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return self.assets_root

    def _localized_prompt(self, item_uid: str, fallback_prompt: str, fallback_phrase: str) -> str:
        if self.prompt_language == "en" or self.translation_rows.empty:
            return self.build_localized_prompt(fallback_prompt, fallback_phrase)

        lang = re.split(r"[-_]", str(self.prompt_language).strip().lower())[0]
        rows = self.translation_rows[self.translation_rows["item_uid"].astype(str) == str(item_uid)]
        if rows.empty:
            return self.build_localized_prompt(fallback_prompt, fallback_phrase)

        lang_rows = rows[rows["language"].astype(str).str.lower() == lang]
        source = lang_rows.iloc[0] if not lang_rows.empty else rows.iloc[0]
        prompt = str(source.get("full_prompt", "")).strip() or str(fallback_prompt)
        phrase = str(source.get("prompt_phrase", "")).strip() or str(fallback_phrase)
        return self.build_localized_prompt(prompt, phrase)

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]
        answer = str(row["answer"]).strip()
        alternatives = [a.strip() for a in str(row["response_alternatives"]).split(",")]
        item_uid = str(row["item_uid"]).strip()

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
            option_labels=LABELS,
            true_random=true_random,
            true_random_seed=true_random_seed,
        )

        option_image_paths = []
        for word in all_options:
            path = _resolve_image(word, self.image_index)
            if path is None:
                raise FileNotFoundError(
                    f"Synthetic vocab image not found for '{word}' in {self.image_dir} "
                    f"(trial {item_uid})"
                )
            option_image_paths.append(str(path))

        prompt = self._localized_prompt(
            item_uid=item_uid,
            fallback_prompt=str(row.get("full_prompt", "")),
            fallback_phrase=str(row.get("prompt_phrase", "")),
        )

        return {
            "trial_id": item_uid,
            "item_uid": item_uid,
            "prompt": prompt,
            "options": all_options,
            "option_labels": LABELS[:len(all_options)],
            "correct_label": correct_label,
            "option_order_seed": option_order_seed,
            "context_image_paths": [],
            "option_image_paths": option_image_paths,
            "context_type": "none",
            "option_type": "image",
        }
