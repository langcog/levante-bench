"""TROG dataset. Context: optional image, options: images."""

import pandas as pd

from levante_bench.data.datasets import VLMDataset
from levante_bench.tasks.image_index import build_image_index
from levante_bench.tasks.option_order import deterministic_option_order
from levante_bench.tasks.registry import register_task

LABELS = ["A", "B", "C", "D"]

_ANSWER_INSTRUCTION_BY_LANG = {
    "de": 'Antworte mit A, B, C oder D. A: <image1>; B: <image2>; C: <image3>; D: <image4>',
    "de-CH": 'Antworte mit A, B, C oder D. A: <image1>; B: <image2>; C: <image3>; D: <image4>',
    "es-CO": 'Responde con A, B, C o D. A: <image1>; B: <image2>; C: <image3>; D: <image4>',
    "es-AR": 'Responde con A, B, C o D. A: <image1>; B: <image2>; C: <image3>; D: <image4>',
    "fr-CA": 'Reponds avec A, B, C ou D. A: <image1>; B: <image2>; C: <image3>; D: <image4>',
    "nl": 'Antwoord met A, B, C of D. A: <image1>; B: <image2>; C: <image3>; D: <image4>',
}


@register_task("trog")
class TrogDataset(VLMDataset):
    """Reads trog trials from manifest.csv with image answer choices."""

    def __init__(self, task_def, version, data_root=None):
        super().__init__(task_def=task_def, version=version, data_root=data_root)
        self.manifest = self._load_manifest()
        self.image_dir = self.data_root / "assets" / self.version / "visual" / "trog"
        self.image_index = build_image_index(self.image_dir)
        self.item_id_by_uid = self._build_item_id_map()

    def _build_item_id_map(self) -> dict[str, str]:
        corpus_file = str(self.task_def.corpus_file or "trog-item-bank-full-params.csv")
        path = (
            self.data_root
            / "assets"
            / self.version
            / "corpus"
            / "trog"
            / corpus_file
        )
        if not path.exists():
            return {}
        df = pd.read_csv(path)
        out: dict[str, str] = {}
        for _, row in df.iterrows():
            item_uid = str(row.get("item_uid", "")).strip()
            item_id = str(row.get("item_id", "")).strip()
            if item_uid and item_id and item_uid not in {"nan", "NA"}:
                out[item_uid] = item_id
        return out

    def _load_manifest(self) -> pd.DataFrame:
        """Load and filter manifest rows for trog task."""
        manifest_path = self.data_root / "assets" / "manifest.csv"
        df = pd.read_csv(manifest_path)
        df = df[df["task"] == "trog"]
        return df.reset_index(drop=True)

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]

        answer = row["answer"]
        alternatives = row["response_alternatives"].split(",")
        all_options, correct_label = deterministic_option_order(
            answer=answer,
            alternatives=alternatives,
            seed_value=row["item_uid"],
            option_labels=LABELS,
        )

        option_image_paths = []
        for option in all_options:
            path = self.image_index.get(option.strip())
            if path is None:
                raise FileNotFoundError(
                    f"Image not found for '{option}' in {self.image_dir} "
                    f"(trial {row['item_uid']})"
                )
            option_image_paths.append(str(path))

        answer_instruction = _ANSWER_INSTRUCTION_BY_LANG.get(
            self.prompt_language,
            'Answer with A, B, C, or D. A: <image1>; B: <image2>; C: <image3>; D: <image4>',
        )
        item_uid = str(row["item_uid"]).strip()
        item_id = self.item_id_by_uid.get(item_uid, "")
        localized_item_prompt = self.translate_item(item_id, "")
        if localized_item_prompt:
            prompt = f"{localized_item_prompt} {answer_instruction}"
        else:
            prompt_base = self.build_localized_prompt(
                prompt_template=row["prompt"],
                prompt_phrase=row["prompt_phrase"],
            )
            prompt = f'{prompt_base} "{self.translate_text(row["prompt_phrase"])}". {answer_instruction}'

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
            "context_image_paths": context_image_paths,
            "option_image_paths": option_image_paths,
            "context_type": "image" if context_image_paths else "none",
            "option_type": "image",
        }
