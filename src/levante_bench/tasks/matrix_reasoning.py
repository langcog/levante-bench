"""Matrix Reasoning dataset. Context: matrix puzzle image, options: 4 images."""

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
    """Look up an image by its stem (case-insensitive)."""
    path = index.get(stem.lower())
    if path is None:
        raise FileNotFoundError(
            f"Matrix-reasoning image not found: '{stem}'"
        )
    return path


def _split_alternatives(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def _build_prompt(instruction: str, n_options: int) -> str:
    labels = LABELS[:n_options]
    lines = [
        "You are shown an incomplete matrix pattern followed by candidate pieces.",
        "The first image is the matrix puzzle with a blank space.",
        f"Images 2–{n_options + 1} are the {n_options} candidate pieces "
        f"({', '.join(labels)} respectively).",
        instruction,
        f"Return only the option letter ({', '.join(labels)}).",
    ]
    return "\n".join(lines)


@register_task("matrix-reasoning")
class MatrixReasoningDataset(VLMDataset):
    """Matrix pattern-completion task. Context = puzzle image, options = 4 piece images."""

    def __init__(self, task_def, version, data_root=None):
        super().__init__(task_def=task_def, version=version, data_root=data_root)
        self.image_dir = (
            Path(self.data_root) / "assets" / self.version / "visual" / "matrix-reasoning"
        )
        self.image_index = _build_image_index(self.image_dir)
        self.records = self._load_records()

    def _corpus_path(self) -> Path:
        corpus_file = self.task_def.corpus_file or "matrix-reasoning-corpus-retest.csv"
        return Path(self.data_root) / "assets" / self.version / "corpus" / "matrix-reasoning" / corpus_file

    def _load_records(self) -> list[dict]:
        path = self._corpus_path()
        if not path.exists():
            raise FileNotFoundError(f"Matrix-reasoning corpus not found: {path}")

        records: list[dict] = []
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if (row.get("assessment_stage") or "").strip().lower() != "test_response":
                    continue

                item_uid = (row.get("item_uid") or "").strip()
                context_stem = (row.get("item") or "").strip()
                answer_stem = (row.get("answer") or "").strip()
                if not item_uid or not context_stem or not answer_stem:
                    continue

                distractors = _split_alternatives(row.get("response_alternatives") or "")
                option_stems = [answer_stem] + [d for d in distractors if d != answer_stem]
                if len(option_stems) < 2:
                    continue

                rng = random.Random(item_uid)
                rng.shuffle(option_stems)
                gold_idx = option_stems.index(answer_stem)
                correct_label = LABELS[gold_idx]

                instruction = (row.get("prompt") or "Choose the piece that best completes the pattern.").strip()
                prompt_text = _build_prompt(instruction, len(option_stems))

                try:
                    context_path = _resolve_image(self.image_index, context_stem)
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
                        "context_image_paths": [str(context_path)],
                        "option_image_paths": option_paths,
                        "context_type": "multi_image",
                        "option_type": "image",
                    }
                )
        return records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx]
