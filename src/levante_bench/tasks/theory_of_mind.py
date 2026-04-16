"""Theory of Mind (Stories) dataset. Context: text story, options: text."""

import csv
from pathlib import Path

from levante_bench.data.datasets import VLMDataset
from levante_bench.tasks.option_order import deterministic_option_order
from levante_bench.tasks.registry import register_task

LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H"]


def _split_alternatives(value: str) -> list[str]:
    if not value:
        return []
    return [x.strip() for x in value.split(",") if x.strip()]


def _dedupe_keep_order(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for v in values:
        if v not in seen:
            out.append(v)
            seen.add(v)
    return out


def _build_prompt(question: str, context_lines: list[str], options: list[str]) -> str:
    lines: list[str] = []
    if context_lines:
        lines.append("Story context:")
        lines.extend(f"- {line}" for line in context_lines)
        lines.append("")
    lines.append(f"Question: {question}")
    lines.append("Options:")
    for idx, option in enumerate(options):
        lines.append(f"{LETTERS[idx]}. {option}")
    lines.append("Answer with a single letter (A, B, C, or D).")
    return "\n".join(lines)


@register_task("theory-of-mind")
class TheoryOfMindDataset(VLMDataset):
    """Reads ToM item bank and serves test_response rows with block context."""

    def __init__(self, task_def, version, data_root=None):
        super().__init__(task_def=task_def, version=version, data_root=data_root)
        self.records = self._load_records()

    def _corpus_path(self) -> Path:
        if not self.data_root:
            raise ValueError("TheoryOfMindDataset requires data_root")
        corpus_file = self.task_def.corpus_file or "theory-of-mind-item-bank.csv"
        return (
            Path(self.data_root)
            / "assets"
            / self.version
            / "corpus"
            / "theory-of-mind"
            / str(corpus_file)
        )

    def _load_records(self) -> list[dict]:
        path = self._corpus_path()
        if not path.exists():
            raise FileNotFoundError(f"ToM corpus not found: {path}")

        records: list[dict] = []
        current_block = None
        context_lines: list[str] = []

        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                block = (row.get("block_index") or "").strip()
                if block != current_block:
                    current_block = block
                    context_lines = []

                prompt = (row.get("prompt") or "").strip()
                stage = (row.get("assessment_stage") or "").strip().lower()
                if stage == "instructions":
                    if prompt:
                        context_lines.append(prompt)
                    continue
                if stage != "test_response":
                    continue

                answer = (row.get("answer") or "").strip()
                if not answer or not prompt:
                    continue

                distractors = _split_alternatives((row.get("response_alternatives") or "").strip())
                options = _dedupe_keep_order([answer] + [d for d in distractors if d != answer])
                if len(options) < 2:
                    continue

                seed_value = (row.get("item_uid") or "").strip() or prompt
                options, correct_label = deterministic_option_order(
                    answer=answer,
                    alternatives=options[1:],
                    seed_value=seed_value,
                    option_labels=LETTERS,
                )

                prompt_text = _build_prompt(prompt, context_lines, options)
                records.append(
                    {
                        "trial_id": (row.get("item_uid") or row.get("item_id") or "").strip(),
                        "item_uid": (row.get("item_uid") or "").strip(),
                        "trial_type": (row.get("trial_type") or "").strip(),
                        "prompt": prompt_text,
                        "options": options,
                        "option_labels": LETTERS[: len(options)],
                        "correct_label": correct_label,
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
