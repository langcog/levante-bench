#!/usr/bin/env python3
"""Smoke test: run Qwen3.5 on 5 vocab items and print results."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from levante_bench.models.vlm import Qwen35Model
from levante_bench.config import get_task_def
from levante_bench.tasks.vocab import VocabDataset

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
VERSION = "2026-03-26"
MODEL_ID = "Qwen/Qwen3.5-0.8B"
N_ITEMS = 5


def main():
    print(f"Loading model: {MODEL_ID}")
    model = Qwen35Model(model_name=MODEL_ID, device="cpu", dtype="bfloat16")
    model.load()
    print("Model loaded.\n")

    task_def = get_task_def("vocab", VERSION, data_root=DATA_ROOT)
    dataset = VocabDataset(task_def=task_def, version=VERSION, data_root=DATA_ROOT)
    print(f"Dataset: {len(dataset)} vocab items total. Running first {N_ITEMS}.\n")

    correct = 0
    for i in range(N_ITEMS):
        trial = dataset[i]
        trial["max_new_tokens"] = 16

        result = model.evaluate_trial(trial)

        status = "✓" if result["is_correct"] else "✗"
        print(
            f"[{i+1}/{N_ITEMS}] {status} item={result['item_uid']}"
            f"  prompt_phrase={trial['options'][['A','B','C','D'].index(trial['correct_label'])]}"
            f"  correct={result['correct_label']}"
            f"  predicted={result['predicted_label']!r}"
            f"  generated={result['generated_text']!r}"
        )
        if result["is_correct"]:
            correct += 1

    print(f"\nAccuracy: {correct}/{N_ITEMS} = {correct/N_ITEMS:.2%}")


if __name__ == "__main__":
    main()
