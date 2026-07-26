#!/usr/bin/env python3
"""Run the keyed single-select SDS items through a local VLM."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import urllib.request
from pathlib import Path

from levante_bench.runtime import load_model, run_trials

DEFAULT_CORPUS = (
    "https://storage.googleapis.com/levante-assets-dev/corpus/"
    "same-different-selection/same-different-selection-item-bank.csv"
)
LABELS = ["A", "B", "C", "D"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="smolvlm2")
    parser.add_argument(
        "--model-config",
        type=Path,
        default=Path("configs/models/smolvlm2_500m.yaml"),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--corpus", default=DEFAULT_CORPUS)
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path(
            "../levante-qa/scripts/eval/output/gcs_samediff_cache"
        ),
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument(
        "--thinking-budget",
        type=int,
        default=None,
        help="Optional override for Gemini thinking_budget (0 disables thinking).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/pilots/sds-smolvlm2-500m.json"),
    )
    return parser.parse_args()


def _read_corpus(source: str) -> str:
    path = Path(source)
    if path.is_file():
        return path.read_text(encoding="utf-8")
    with urllib.request.urlopen(source, timeout=60) as response:
        return response.read().decode("utf-8")


def _image_path(image_dir: Path, key: str) -> Path:
    for suffix in (".webp", ".png", ".jpg", ".jpeg"):
        candidate = image_dir / f"{key}{suffix}"
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"No image for {key!r} under {image_dir}")


def _build_trials(corpus: str, image_dir: Path) -> list[dict]:
    trials: list[dict] = []
    for row in csv.DictReader(io.StringIO(corpus)):
        if (row.get("required_selections") or "").strip() != "1":
            continue
        answer = (row.get("answer") or "").split(",")[0].strip()
        alternatives = [
            value.strip()
            for value in (row.get("response_alternatives") or "").split(",")
            if value.strip()
        ]
        instruction = (row.get("item") or "").strip()
        item_id = (row.get("item_id") or row.get("audio_file") or "").strip()
        choices = [answer, *alternatives]
        if (
            not item_id
            or not instruction
            or (row.get("trial_type") or "").strip() != "test-dimensions"
            or len(choices) not in {3, 4}
        ):
            continue

        order = list(range(len(choices)))
        seed = int(hashlib.md5(item_id.encode("utf-8")).hexdigest()[:8], 16)
        random.Random(seed).shuffle(order)
        ordered_keys = [choices[index] for index in order]
        correct_label = LABELS[order.index(0)]
        labels = LABELS[: len(choices)]
        image_legend = "; ".join(
            f"{label}: <image{index + 1}>"
            for index, label in enumerate(labels)
        )
        prompt = (
            f'Instruction: "{instruction}"\n'
            "Choose the one card that satisfies the instruction. "
            f"Answer with only {', '.join(labels)}.\n"
            f"{image_legend}"
        )
        trials.append(
            {
                "trial_id": item_id,
                "item_uid": item_id,
                "prompt": prompt,
                "options": ordered_keys,
                "option_labels": labels,
                "correct_label": correct_label,
                "context_image_paths": [],
                "option_image_paths": [
                    str(_image_path(image_dir, key)) for key in ordered_keys
                ],
                "answer_format": "label",
                "trial_type": (row.get("trial_type") or "").strip(),
                "chance_accuracy": 1.0 / len(choices),
            }
        )
    return trials


def _poisson_binomial_upper_tail(hits: int, probabilities: list[float]) -> float:
    distribution = [1.0]
    for probability in probabilities:
        updated = [0.0] * (len(distribution) + 1)
        for successes, mass in enumerate(distribution):
            updated[successes] += mass * (1.0 - probability)
            updated[successes + 1] += mass * probability
        distribution = updated
    return sum(distribution[hits:])


def main() -> int:
    args = _parse_args()
    trials = _build_trials(_read_corpus(args.corpus), args.image_dir)
    if args.limit:
        trials = trials[: args.limit]
    if not trials:
        raise RuntimeError("No keyed four-choice SDS trials were found.")

    print(f"Loading {args.model}; running {len(trials)} SDS trials...")
    overrides = {}
    if args.thinking_budget is not None:
        overrides["thinking_budget"] = args.thinking_budget
    model = load_model(
        model_name=args.model,
        model_config_path=args.model_config,
        model_overrides=overrides or None,
        device=args.device,
    )
    results = run_trials(
        model,
        trials,
        max_new_tokens=args.max_new_tokens,
        task_id="same-different-selection",
    )
    hits = sum(result["is_correct"] for result in results)
    parsed = sum(result["predicted_label"] is not None for result in results)
    total = len(results)
    chance_probabilities = [float(trial["chance_accuracy"]) for trial in trials]
    chance_accuracy = sum(chance_probabilities) / total
    summary = {
        "model": args.model,
        "n": total,
        "hits": hits,
        "accuracy": hits / total,
        "chance_accuracy": chance_accuracy,
        "chance_expected_hits": sum(chance_probabilities),
        "parsed": parsed,
        "exact_p_above_chance": _poisson_binomial_upper_tail(
            hits, chance_probabilities
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"summary": summary, "results": results}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
