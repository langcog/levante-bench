"""VLMDataset base and task-specific subclasses."""

from pathlib import Path
from typing import Optional

import pandas as pd
from torch.utils.data import Dataset

from levante_bench.data.schema import TaskDef


class VLMDataset(Dataset):
    """Base dataset that serves standardized trial dicts.

    Subclasses set up raw data references in __init__ and implement
    __getitem__ to return a single formatted trial dict with:
        trial_id, item_uid, prompt, options, option_labels,
        correct_label, context_images, option_images,
        context_type, option_type
    """

    def __init__(self, task_def: TaskDef, version: str, data_root: Optional[Path] = None) -> None:
        self.task_def = task_def
        self.version = version
        self.data_root = data_root
        self.prompt_language = str(getattr(task_def, "prompt_language", "en") or "en")
        self._translations_exact: dict[str, str] = {}
        self._translations_normalized: dict[str, str] = {}
        self._translations_by_item_id: dict[str, str] = {}
        self._init_translations()

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> dict:
        """Return one formatted trial dict with images loaded."""
        raise NotImplementedError

    @staticmethod
    def _normalize_text(value: str) -> str:
        return " ".join(value.split()).strip()

    @staticmethod
    def _to_text(value: object) -> str:
        if value is None:
            return ""
        s = str(value)
        if s.strip().lower() in {"", "nan", "na", "none"}:
            return ""
        return s

    def _translations_path(self) -> Path | None:
        if self.data_root is None:
            return None
        return (
            Path(self.data_root)
            / "assets"
            / self.version
            / "translations"
            / "item-bank-translations.csv"
        )

    def _init_translations(self) -> None:
        if self.prompt_language == "en":
            return
        path = self._translations_path()
        if path is None or not path.exists():
            return

        df = pd.read_csv(path, dtype=str).fillna("")
        if self.prompt_language not in df.columns:
            return

        for _, row in df.iterrows():
            item_id = self._to_text(row.get("item_id", ""))
            en_text = self._to_text(row.get("en", ""))
            lang_text = self._to_text(row.get(self.prompt_language, ""))
            if not en_text or not lang_text:
                continue
            self._translations_exact.setdefault(en_text, lang_text)
            self._translations_normalized.setdefault(self._normalize_text(en_text), lang_text)
            if item_id:
                self._translations_by_item_id.setdefault(item_id, lang_text)

    def translate_text(self, text: object) -> str:
        text_s = self._to_text(text)
        if not text_s or self.prompt_language == "en":
            return text_s
        translated = self._translations_exact.get(text_s)
        if translated:
            return translated
        return self._translations_normalized.get(self._normalize_text(text_s), text_s)

    def build_localized_prompt(self, prompt_template: object, prompt_phrase: object) -> str:
        template_s = self.translate_text(prompt_template)
        phrase_s = self.translate_text(prompt_phrase)
        return template_s.replace("<prompt_phrase>", phrase_s)

    def translate_item(self, item_id: object, fallback_text: object = "") -> str:
        fallback = self._to_text(fallback_text)
        if self.prompt_language == "en":
            return fallback
        key = self._to_text(item_id)
        if not key:
            return fallback
        return self._translations_by_item_id.get(key, fallback)
