#!/usr/bin/env python3
"""
Convert EGMA math corpus rows into SmolVLMv2-ready prompt records.

Output format is JSONL, one record per item:
{
  "item_uid": "...",
  "prompt_text": "...",
  "messages": [{"role": "user", "content": [{"type": "text", "text": "..."}]}],
  "options": ["...", "...", "...", "..."],
  "gold_index": 2,
  "gold_letter": "C",
  ...
}
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from pathlib import Path

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build SmolVLMv2 prompts from EGMA math corpus CSV.")
    p.add_argument("--corpus-csv", type=Path, required=True, help="Path to test-combined-math-cat.csv")
    p.add_argument("--output", type=Path, required=True, help="Output JSONL path")
    p.add_argument(
        "--include-practice",
        action="store_true",
        help="Include practice rows (default: only test_response rows)",
    )
    p.add_argument(
        "--include-instructions",
        action="store_true",
        help="Include instructions rows (default: exclude)",
    )
    p.add_argument(
        "--include-audio-dependent",
        action="store_true",
        help="Include rows likely requiring audio (default: exclude)",
    )
    p.add_argument(
        "--shuffle-options",
        action="store_true",
        help="Shuffle answer choices while preserving gold index (default: no shuffle)",
    )
    p.add_argument(
        "--numberline-hint",
        choices=["none", "coarse", "exact"],
        default="coarse",
        help=(
            "Optional extra hint for Number Line prompts derived from item_uid. "
            "'coarse' adds approximate location (left/middle/right), "
            "'exact' adds the marked number."
        ),
    )
    p.add_argument("--seed", type=int, default=0, help="Random seed when --shuffle-options is used")
    return p.parse_args()


def _split_alternatives(value: str) -> list[str]:
    if not value:
        return []
    return [x.strip() for x in value.split(",") if x.strip()]


def _is_audio_dependent(row: dict[str, str]) -> bool:
    prompt = (row.get("prompt") or "").strip().lower()
    trial_type = (row.get("trial_type") or "").strip().lower()
    audio_file = (row.get("audio_file") or "").strip()
    return bool(audio_file) and ("hear" in prompt or "listening" in prompt or "identification" in trial_type)


def _should_include_row(
    row: dict[str, str],
    include_practice: bool,
    include_instructions: bool,
    include_audio_dependent: bool,
) -> bool:
    stage = (row.get("assessment_stage") or "").strip().lower()
    answer = (row.get("answer") or "").strip()
    item_uid = (row.get("item_uid") or "").strip()

    if not item_uid:
        return False
    if not include_instructions and stage == "instructions":
        return False
    if not include_practice and stage and stage != "test_response":
        return False
    if not answer:
        return False
    if not include_audio_dependent and _is_audio_dependent(row):
        return False
    return True


def _numberline_hint(item_uid: str, numberline_hint: str) -> str | None:
    if numberline_hint == "none":
        return None
    # Example item_uid: math_line_4_10  -> marked=4, upper=10
    m = re.search(r"math_line_(\d+)_([0-9]+)$", item_uid)
    if not m:
        return None
    marked = int(m.group(1))
    upper = int(m.group(2))
    if numberline_hint == "exact":
        return f"The marked point is at {marked} on the number line from 0 to {upper}."
    if upper <= 0:
        return None
    ratio = marked / upper
    if ratio < 0.33:
        loc = "left side"
    elif ratio < 0.67:
        loc = "middle"
    else:
        loc = "right side"
    return f"The marked point is near the {loc} of the number line (0 to {upper})."


def _build_prompt_text(row: dict[str, str], options: list[str], numberline_hint: str) -> str:
    trial_type = (row.get("trial_type") or "").strip()
    stem = (row.get("prompt") or "").strip()
    item = (row.get("item") or "").strip()
    item_uid = (row.get("item_uid") or "").strip()

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
    hint = _numberline_hint(item_uid, numberline_hint)
    if hint is not None and trial_type.lower().startswith("number line"):
        lines.append(f"Hint: {hint}")
    lines.append("Options:")
    for i, opt in enumerate(options):
        lines.append(f"{LETTERS[i]}. {opt}")
    return "\n".join(lines)


def _record_from_row(
    row: dict[str, str],
    shuffle_options: bool,
    rng: random.Random,
    numberline_hint: str,
) -> dict[str, object] | None:
    answer = (row.get("answer") or "").strip()
    distractors = _split_alternatives((row.get("response_alternatives") or "").strip())
    options = [answer] + [d for d in distractors if d != answer]
    if len(options) < 2:
        return None

    # Keep options unique but preserve order.
    deduped: list[str] = []
    seen: set[str] = set()
    for o in options:
        if o not in seen:
            deduped.append(o)
            seen.add(o)
    options = deduped

    if shuffle_options:
        rng.shuffle(options)
    gold_index = options.index(answer)
    prompt_text = _build_prompt_text(row, options, numberline_hint=numberline_hint)

    return {
        "item_uid": (row.get("item_uid") or "").strip(),
        "task": (row.get("task") or "").strip(),
        "trial_type": (row.get("trial_type") or "").strip(),
        "assessment_stage": (row.get("assessment_stage") or "").strip(),
        "difficulty": (row.get("difficulty") or "").strip(),
        "audio_file": (row.get("audio_file") or "").strip(),
        "options": options,
        "gold_answer": answer,
        "gold_index": gold_index,
        "gold_letter": LETTERS[gold_index],
        "prompt_text": prompt_text,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt_text}],
            }
        ],
    }


def run(args: argparse.Namespace) -> tuple[int, int]:
    rng = random.Random(args.seed)
    kept = 0
    skipped = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with open(args.corpus_csv, newline="", encoding="utf-8") as f_in, open(
        args.output, "w", encoding="utf-8"
    ) as f_out:
        reader = csv.DictReader(f_in)
        for row in reader:
            if not _should_include_row(
                row=row,
                include_practice=args.include_practice,
                include_instructions=args.include_instructions,
                include_audio_dependent=args.include_audio_dependent,
            ):
                skipped += 1
                continue
            rec = _record_from_row(
                row,
                shuffle_options=args.shuffle_options,
                rng=rng,
                numberline_hint=args.numberline_hint,
            )
            if rec is None:
                skipped += 1
                continue
            f_out.write(json.dumps(rec, ensure_ascii=True) + "\n")
            kept += 1
    return kept, skipped


def main() -> None:
    args = parse_args()
    kept, skipped = run(args)
    print(f"Wrote {args.output} ({kept} prompts, {skipped} skipped)")


if __name__ == "__main__":
    main()
