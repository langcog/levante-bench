"""EgmaMath dataset. Context: none, options: text."""

import csv
import random
from pathlib import Path

from levante_bench.data.datasets import VLMDataset
from levante_bench.tasks.registry import register_task

LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H"]


def _split_alternatives(value: str) -> list[str]:
    if not value:
        return []
    return [x.strip() for x in value.split(",") if x.strip()]


def _is_audio_dependent(row: dict[str, str]) -> bool:
    prompt = (row.get("prompt") or "").strip().lower()
    trial_type = (row.get("trial_type") or "").strip().lower()
    audio_file = (row.get("audio_file") or "").strip()
    return bool(audio_file) and ("hear" in prompt or "listening" in prompt or "identification" in trial_type)


def _build_prompt(row: dict[str, str], options: list[str]) -> str:
    trial_type = (row.get("trial_type") or "").strip()
    stem = (row.get("prompt") or "").strip()
    item = (row.get("item") or "").strip()
    lines = [
        "Solve this multiple-choice math problem.",
        "Return only the option letter (A, B, C, ...).",
    ]
    if trial_type:
        lines.append(f"Category: {trial_type}")
    if stem:
        lines.append(f"Instruction: {stem}")
    if item:
        lines.append(f"Problem: {item}")
    lines.append("Options:")
    for idx, option in enumerate(options):
        lines.append(f"{LETTERS[idx]}. {option}")
    return "\n".join(lines)


@register_task("egma-math")
class EgmaMathDataset(VLMDataset):
    """Reads egma-math corpus and serves text-only forced-choice test trials."""

    def __init__(self, task_def, version, data_root=None):
        super().__init__(task_def=task_def, version=version, data_root=data_root)
        self.records = self._load_records()

    def _corpus_path(self) -> Path:
        if not self.data_root:
            raise ValueError("EgmaMathDataset requires data_root")
        corpus_file = self.task_def.corpus_file or "test-combined-math-cat.csv"
        return (
            Path(self.data_root)
            / "assets"
            / self.version
            / "corpus"
            / "egma-math"
            / str(corpus_file)
        )

    def _load_records(self) -> list[dict]:
        path = self._corpus_path()
        if not path.exists():
            raise FileNotFoundError(f"EGMA corpus not found: {path}")

        records: list[dict] = []
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                stage = (row.get("assessment_stage") or "").strip().lower()
                if stage != "test_response":
                    continue
                if _is_audio_dependent(row):
                    continue

                answer = (row.get("answer") or "").strip()
                if not answer:
                    continue
                distractors = _split_alternatives((row.get("response_alternatives") or "").strip())
                options = [answer] + [d for d in distractors if d != answer]
                if len(options) < 2:
                    continue

                deduped: list[str] = []
                seen: set[str] = set()
                for option in options:
                    if option not in seen:
                        deduped.append(option)
                        seen.add(option)
                options = deduped

                rng = random.Random((row.get("item_uid") or "").strip() or answer)
                rng.shuffle(options)
                gold_index = options.index(answer)
                prompt_text = _build_prompt(row, options)

                records.append(
                    {
                        "trial_id": (row.get("item_uid") or row.get("item_id") or "").strip(),
                        "item_uid": (row.get("item_uid") or "").strip(),
                        "prompt": prompt_text,
                        "options": options,
                        "option_labels": LETTERS[: len(options)],
                        "correct_label": LETTERS[gold_index],
                        "context_image_paths": [],
                        "option_image_paths": [],
                        "context_type": "none",
                        "option_type": "text",
                    }
                )
        return records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx]
