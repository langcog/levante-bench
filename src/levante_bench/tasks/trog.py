"""TROG (Test for Reception of Grammar) dataset. Context: none, options: 4 images."""

import csv
import random
from pathlib import Path

from levante_bench.data.datasets import VLMDataset
from levante_bench.tasks.registry import register_task

LABELS = ["A", "B", "C", "D"]


def _build_image_index(directory: Path) -> dict[str, Path]:
    """Map image stems (lowercase) to paths, scanned once."""
    index: dict[str, Path] = {}
    for path in directory.iterdir():
        if path.is_file():
            index[path.stem.lower()] = path
    return index


def _resolve_image(index: dict[str, Path], stem: str) -> Path:
    path = index.get(stem.lower())
    if path is None:
        raise FileNotFoundError(f"TROG image not found: '{stem}'")
    return path


def _split_alternatives(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def _build_prompt(instruction: str, item: str, n_options: int) -> str:
    labels = LABELS[:n_options]
    lines = [
        f"You are shown {n_options} pictures ({', '.join(labels)}).",
    ]
    if item:
        full_instruction = f"{instruction} {item}.".rstrip(".")  + "."
    else:
        full_instruction = instruction
    lines.append(full_instruction)
    lines.append(f"Return only the letter of the correct picture ({', '.join(labels)}).")
    return "\n".join(lines)


@register_task("trog")
class TrogDataset(VLMDataset):
    """TROG grammar comprehension task. Select the picture that matches a spoken sentence."""

    def __init__(self, task_def, version, data_root=None):
        super().__init__(task_def=task_def, version=version, data_root=data_root)
        self.image_dir = (
            Path(self.data_root) / "assets" / self.version / "visual" / "trog"
        )
        self.image_index = _build_image_index(self.image_dir)
        self.records = self._load_records()

    def _corpus_path(self) -> Path:
        corpus_file = self.task_def.corpus_file or "trog-item-bank-full-params.csv"
        return Path(self.data_root) / "assets" / self.version / "corpus" / "trog" / corpus_file

    def _load_records(self) -> list[dict]:
        path = self._corpus_path()
        if not path.exists():
            raise FileNotFoundError(f"TROG corpus not found: {path}")

        records: list[dict] = []
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if (row.get("assessment_stage") or "").strip().lower() != "test_response":
                    continue

                item_uid = (row.get("item_uid") or "").strip()
                answer_stem = (row.get("answer") or "").strip()
                if not item_uid or not answer_stem:
                    continue

                distractors = _split_alternatives(row.get("response_alternatives") or "")
                option_stems = [answer_stem] + [d for d in distractors if d != answer_stem]
                if len(option_stems) < 2:
                    continue

                rng = random.Random(item_uid)
                rng.shuffle(option_stems)
                gold_idx = option_stems.index(answer_stem)
                correct_label = LABELS[gold_idx]

                instruction = (row.get("prompt") or "Choose the picture of the").strip().rstrip(".")
                item_word = (row.get("item") or "").strip()
                prompt_text = _build_prompt(instruction, item_word, len(option_stems))

                try:
                    option_paths = [str(_resolve_image(self.image_index, s)) for s in option_stems]
                except FileNotFoundError:
                    continue

                records.append(
                    {
                        "trial_id": item_uid,
                        "item_uid": item_uid,
                        "prompt": prompt_text,
                        # Stems used for human-comparison matching
                        "options": option_stems,
                        "option_labels": LABELS[: len(option_stems)],
                        "correct_label": correct_label,
                        "context_image_paths": [],
                        "option_image_paths": option_paths,
                        "context_type": "none",
                        "option_type": "image",
                    }
                )
        return records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx]
