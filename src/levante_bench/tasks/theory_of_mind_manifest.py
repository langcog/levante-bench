"""Theory of Mind dataset. Context: none, options: images."""

from pathlib import Path

import pandas as pd

from levante_bench.data.datasets import VLMDataset
from levante_bench.prompts import render_prompt_template
from levante_bench.tasks.image_index import build_image_index
from levante_bench.tasks.option_order import (
    derive_true_random_item_seed,
    deterministic_option_order,
)
from levante_bench.tasks.registry import register_task

LABELS = ["A", "B", "C", "D"]
_SKIP_CONTEXT_ITEM_IDS = {"ToM-intro", "ToM-transition"}


@register_task("theory-of-mind")
class TheoryOfMindDataset(VLMDataset):
    """Reads ToM trials from manifest.csv with image answer choices."""

    def __init__(self, task_def, version, data_root=None):
        super().__init__(task_def=task_def, version=version, data_root=data_root)
        self.manifest = self._load_manifest()
        self.image_dir = self.data_root / "assets" / self.version / "visual" / "theory-of-mind"
        self.image_index = build_image_index(self.image_dir)
        self.story_prompts = self._load_story_prompts()

    def _load_manifest(self) -> pd.DataFrame:
        """Load and filter manifest rows for theory-of-mind task."""
        manifest_path = self.data_root / "assets" / "manifest.csv"
        df = pd.read_csv(manifest_path)
        df = df[df["task"] == "theory-of-mind"]
        return df.reset_index(drop=True)

    def _corpus_path(self) -> Path:
        corpus_file = self.task_def.corpus_file or "theory-of-mind-item-bank.csv"
        return (
            Path(self.data_root)
            / "assets"
            / self.version
            / "corpus"
            / "theory-of-mind"
            / str(corpus_file)
        )

    def _load_story_prompts(self) -> dict[str, dict[str, object]]:
        """Map each test item to its preceding story rows plus question row."""
        path = self._corpus_path()
        if not path.exists():
            return {}

        df = pd.read_csv(path, dtype=str).fillna("")
        story_prompts: dict[str, dict[str, object]] = {}
        current_block = None
        context_rows: list[tuple[str, str]] = []

        for _, row in df.iterrows():
            block = str(row.get("block_index", "")).strip()
            if block != current_block:
                current_block = block
                context_rows = []

            stage = str(row.get("assessment_stage", "")).strip().lower()
            item_id = str(row.get("item_id", "")).strip()
            prompt = str(row.get("prompt", "")).strip()
            if stage == "instructions":
                if prompt and item_id not in _SKIP_CONTEXT_ITEM_IDS:
                    context_rows.append((item_id, prompt))
                continue

            if stage != "test_response":
                continue

            item_uid = str(row.get("item_uid", "")).strip()
            if item_uid and prompt:
                story_prompts[item_uid] = {
                    "context_rows": list(context_rows),
                    "question_item_id": item_id,
                    "question": prompt,
                }

        return story_prompts

    def _localized_item_text(self, item_id: object, fallback_text: object) -> str:
        return self.translate_item(item_id, fallback_text) or self._to_text(fallback_text)

    def _build_story_prompt(self, row: pd.Series, n_options: int) -> str:
        story_data = self.story_prompts.get(str(row["item_uid"]).strip())
        if not story_data:
            return self._to_text(row.get("full_prompt", ""))

        context_parts = [
            self._localized_item_text(item_id, prompt)
            for item_id, prompt in story_data.get("context_rows", [])
        ]
        question = self._localized_item_text(
            story_data.get("question_item_id", ""),
            story_data.get("question", row.get("prompt", "")),
        )
        answer_options = render_prompt_template(
            "theory-of-mind",
            f"answer_options.{n_options}_images",
            prompt_language=self.prompt_language,
        )
        return " ".join([*context_parts, question, answer_options]).strip()

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]

        answer = row["answer"]
        alternatives = row["response_alternatives"].split(",")
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
            seed_value=row["item_uid"],
            option_labels=LABELS,
            true_random=true_random,
            true_random_seed=true_random_seed,
        )

        # Resolve option image paths from cached index
        option_image_paths = []
        for option in all_options:
            path = self.image_index.get(option.strip())
            if path is None:
                raise FileNotFoundError(
                    f"Image not found for '{option}' in {self.image_dir} "
                    f"(trial {row['item_uid']})"
                )
            option_image_paths.append(str(path))

        # Story/question text stays itembank-owned; only the answer suffix is centralized.
        prompt = self._build_story_prompt(row, n_options=len(all_options))
        context_image_paths = []
        prompt_image = str(row.get("prompt_image", "NA")).strip()
        if prompt_image and prompt_image not in {"NA", "nan", "TODO"}:
            path = self.image_index.get(prompt_image)
            if path is None:
                raise FileNotFoundError(
                    f"Prompt image not found for '{prompt_image}' in {self.image_dir} "
                    f"(trial {row['item_uid']})"
                )
            context_image_paths.append(str(path))

        return {
            "trial_id": row["item_uid"],
            "item_uid": row["item_uid"],
            "prompt": prompt,
            "options": all_options,
            "option_labels": LABELS[:len(all_options)],
            "correct_label": correct_label,
            "option_order_seed": option_order_seed,
            "context_image_paths": context_image_paths,
            "option_image_paths": option_image_paths,
            "context_type": "image" if context_image_paths else "none",
            "option_type": "image",
        }
