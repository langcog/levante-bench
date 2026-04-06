#!/usr/bin/env python3
"""Phased experiment: test prompt/parsing improvements for egma-math on SmolVLM2.

Each phase can be toggled independently via CLI flags.
Run with --phase 0 for baseline, then --phase 1, --phase 2, etc.

Phases:
    0  Baseline (current manifest prompts, current parsing)
    1  Reformatted prompts (multiline block options)
    2  Numeric fallback parsing (value → label mapping)
    3  System prompt + explicit answer suffix
    4  Few-shot examples per trial_type
    5  Chain-of-thought for arithmetic types

Example:
    python scripts/experiment_egma_math_phases.py --phase 0
    python scripts/experiment_egma_math_phases.py --phase 1 2   # combine phases 1+2
    python scripts/experiment_egma_math_phases.py --phase 1 2 3 4 5  # all improvements
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

LABELS = ["A", "B", "C", "D"]
ARITHMETIC_TYPES = {"Addition", "Subtraction", "Multiplication", "Fraction"}

# ---------------------------------------------------------------------------
# Phase 4: few-shot examples per trial_type
# ---------------------------------------------------------------------------
FEW_SHOT_EXAMPLES: dict[str, list[dict]] = {
    "Number Identification": [
        {"problem": "7", "options": ["3", "7", "1", "9"], "answer": "B"},
    ],
    "Number Comparison": [
        {"problem": "15, 8", "options": ["15", "8"], "answer": "A"},
    ],
    "Missing Number": [
        {"problem": "3, 6, _, 12", "options": ["7", "10", "9", "8"], "answer": "C"},
    ],
    "Addition": [
        {"problem": "3+4", "options": ["8", "7", "6", "1"], "answer": "B"},
    ],
    "Subtraction": [
        {"problem": "9-3", "options": ["5", "3", "6", "12"], "answer": "C"},
    ],
    "Multiplication": [
        {"problem": "3x4", "options": ["7", "12", "34", "9"], "answer": "B"},
    ],
    "Fraction": [
        {"problem": "1/2+1/2", "options": ["1/4", "2/4", "1", "2"], "answer": "C"},
    ],
    "Counting AFC": [
        {"problem": "●●●●", "options": ["3", "4", "5", "6"], "answer": "B"},
    ],
    "Counting": [
        {"problem": "●●●", "options": ["2", "3", "4", "5"], "answer": "B"},
    ],
    "Non-symbolic Number Identification": [
        {"problem": "●●", "options": ["1", "2", "3", "4"], "answer": "B"},
    ],
    "Non-symbolic Number Comparison": [
        {"problem": "●●● vs ●●●●●", "options": ["Left", "Right"], "answer": "B"},
    ],
}


def _format_few_shot_example(ex: dict, trial_type: str) -> str:
    """Format one few-shot example as a solved Q&A pair."""
    lines = [f"Problem: {ex['problem']}", "Options:"]
    for i, opt in enumerate(ex["options"]):
        lines.append(f"{LABELS[i]}) {opt}")
    lines.append(f"Answer: {ex['answer']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompt builders per phase
# ---------------------------------------------------------------------------

def build_prompt_baseline(row: pd.Series, options: list[str]) -> str:
    """Phase 0: current manifest prompt (inline semicolons)."""
    prompt_phrase = str(row.get("prompt_phrase", ""))
    if prompt_phrase in {"NA", "nan"}:
        prompt_phrase = ""
    prompt = str(row["full_prompt"])
    if prompt in {"", "NA", "nan"}:
        prompt = str(row.get("prompt", "")).strip()
    prompt = prompt.replace("<prompt_phrase>", prompt_phrase)
    for i, option in enumerate(options, start=1):
        prompt = prompt.replace(f"<option{i}>", str(option))
    prompt = prompt.replace("<prompt_image>", "")
    return prompt


def build_prompt_phase1(row: pd.Series, options: list[str]) -> str:
    """Phase 1: reformatted multiline prompt with block options."""
    trial_type = str(row.get("trial_type", "")).strip()
    item = str(row.get("item", "")).strip()
    prompt_phrase = str(row.get("prompt_phrase", ""))
    if prompt_phrase in {"NA", "nan"}:
        prompt_phrase = ""

    if trial_type == "Number Identification":
        stem = f"Which number is {prompt_phrase}?"
    elif trial_type == "Number Comparison":
        stem = f"Which number is larger: {item}?"
    elif trial_type == "Missing Number":
        stem = f"What number completes the sequence: {item}?"
    elif trial_type in {"Addition", "Fraction"}:
        stem = f"What is {prompt_phrase}?"
    elif trial_type == "Subtraction":
        stem = f"What is {prompt_phrase}?"
    elif trial_type == "Multiplication":
        stem = f"What is {prompt_phrase}?"
    elif "counting" in trial_type.lower():
        stem = f"How many items are shown: {item}?"
    elif "non-symbolic" in trial_type.lower() and "comparison" in trial_type.lower():
        stem = f"Which group has more: {item}?"
    elif "non-symbolic" in trial_type.lower():
        stem = f"How many items: {item}?"
    else:
        stem = str(row.get("prompt", "")).strip() or f"Solve: {item}"

    lines = [stem, ""]
    for i, opt in enumerate(options):
        lines.append(f"{LABELS[i]}) {opt}")
    lines.append("")
    lines.append("Answer with one letter.")
    return "\n".join(lines)


def apply_phase3_system(prompt: str) -> str:
    """Phase 3: append explicit answer format suffix."""
    return prompt.rstrip() + "\n\nRespond with exactly one letter (A, B, C, or D). Nothing else."


def apply_phase4_fewshot(prompt: str, trial_type: str) -> str:
    """Phase 4: prepend few-shot examples for the trial type."""
    examples = FEW_SHOT_EXAMPLES.get(trial_type, [])
    if not examples:
        return prompt
    parts = ["Here are some solved examples:\n"]
    for ex in examples:
        parts.append(_format_few_shot_example(ex, trial_type))
        parts.append("")
    parts.append("Now solve this one:\n")
    parts.append(prompt)
    return "\n".join(parts)


def apply_phase5_cot(prompt: str, trial_type: str) -> str:
    """Phase 5: add chain-of-thought instruction for arithmetic types."""
    if trial_type not in ARITHMETIC_TYPES:
        return prompt
    cot = "Think step by step to compute the answer, then state your final answer as a single letter."
    return prompt.rstrip() + f"\n\n{cot}"


# ---------------------------------------------------------------------------
# Answer parsing
# ---------------------------------------------------------------------------

def parse_answer_baseline(text: str, option_labels: list[str]) -> tuple[Optional[str], str]:
    """Current parse_answer from base.py (reproduced for standalone use)."""
    text = text.strip()
    labels_upper = [la.upper() for la in option_labels]

    try:
        parsed = json.loads(text)
        answer = parsed.get("answer", "").strip().upper()
        reason = parsed.get("reason", "")
        if answer in labels_upper:
            return answer, reason
    except (json.JSONDecodeError, AttributeError):
        pass

    m = re.search(r'\{[^}]*"answer"\s*:\s*"([^"]+)"[^}]*\}', text)
    if m:
        answer = m.group(1).strip().upper()
        r = re.search(r'"reason"\s*:\s*"([^"]*)"', text)
        reason = r.group(1) if r else ""
        if answer in labels_upper:
            return answer, reason

    m = re.search(r'(?:the\s+)?(?:correct\s+)?answer\s+is\s+([A-Z])\b', text, re.IGNORECASE)
    if m:
        answer = m.group(1).upper()
        if answer in labels_upper:
            return answer, text

    if text.upper() in labels_upper:
        return text.upper(), ""

    for label in option_labels:
        if text.upper().startswith(label.upper()):
            rest = text[len(label):]
            if not rest or rest[0] in " .),:;\n":
                return label, rest.strip()

    return None, text


def parse_answer_phase2(
    text: str, option_labels: list[str], options: list[str]
) -> tuple[Optional[str], str]:
    """Phase 2: try baseline first, then fallback to matching numeric/text values to options."""
    label, reason = parse_answer_baseline(text, option_labels)
    if label is not None:
        return label, reason

    cleaned = text.strip().rstrip(".").strip()

    # Direct value match: if model outputs "5" and option B is "5"
    for i, opt in enumerate(options):
        if i >= len(option_labels):
            break
        if cleaned == opt.strip():
            return option_labels[i], text

    # Normalized numeric match: "5.0" matches "5", "2/5" matches "2/5"
    try:
        model_val = float(cleaned)
        for i, opt in enumerate(options):
            if i >= len(option_labels):
                break
            try:
                if abs(float(opt.strip()) - model_val) < 1e-9:
                    return option_labels[i], text
            except ValueError:
                continue
    except ValueError:
        pass

    # "answer is <value>" pattern where value is an option (not a letter)
    m = re.search(r'(?:the\s+)?(?:correct\s+)?answer\s+is\s+(.+?)(?:\.|$)', text, re.IGNORECASE)
    if m:
        val = m.group(1).strip()
        for i, opt in enumerate(options):
            if i >= len(option_labels):
                break
            if val == opt.strip():
                return option_labels[i], text

    # Last number in output as fallback for numeric options
    numbers = re.findall(r'[-+]?\d*\.?\d+(?:/\d+)?', text)
    if numbers:
        last_num = numbers[-1]
        for i, opt in enumerate(options):
            if i >= len(option_labels):
                break
            if last_num == opt.strip():
                return option_labels[i], text

    return None, text


# ---------------------------------------------------------------------------
# Model loading & generation
# ---------------------------------------------------------------------------

def load_model(model_id: str, device: str):
    """Load SmolVLM2 model and processor."""
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText

    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, torch_dtype=dtype
    ).to(device)
    model.eval()
    return model, processor, dtype


def generate(
    model,
    processor,
    dtype,
    device: str,
    prompt_text: str,
    max_new_tokens: int = 64,
    system_prompt: str | None = None,
) -> str:
    """Generate a response from SmolVLM2."""
    import torch

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
    messages.append({"role": "user", "content": [{"type": "text", "text": prompt_text}]})

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device, dtype=dtype)

    with torch.no_grad():
        output_ids = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)

    full_text = processor.batch_decode(output_ids, skip_special_tokens=True)[0]

    if "Assistant:" in full_text:
        text = full_text.split("Assistant:")[-1]
    else:
        text = full_text
    return re.sub(r"<\|?end\|?>.*$", "", text).strip()


# ---------------------------------------------------------------------------
# Manifest loading
# ---------------------------------------------------------------------------

def load_manifest(data_root: Path) -> pd.DataFrame:
    """Load egma-math rows from manifest.csv (excluding Number Line by default)."""
    manifest_path = data_root / "assets" / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    df = pd.read_csv(manifest_path)
    df = df[df["task"] == "egma-math"]
    df = df[~df["trial_type"].astype(str).str.contains("Number Line", case=False, na=False)]
    return df.reset_index(drop=True)


def build_trials(manifest: pd.DataFrame) -> list[dict]:
    """Build trial dicts from manifest rows with shuffled options."""
    trials = []
    for _, row in manifest.iterrows():
        answer = str(row["answer"]).strip()
        alternatives = str(row.get("response_alternatives") or "").split(",")
        alternatives = [a.strip() for a in alternatives if a.strip()]
        all_options = [answer] + alternatives

        rng = random.Random(row["item_uid"])
        rng.shuffle(all_options)
        correct_idx = all_options.index(answer)
        correct_label = LABELS[correct_idx]

        trials.append({
            "item_uid": row["item_uid"],
            "trial_type": str(row.get("trial_type", "")).strip(),
            "row": row,
            "options": all_options,
            "correct_label": correct_label,
            "answer_value": answer,
        })
    return trials


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

def run_experiment(
    args: argparse.Namespace,
    trials: list[dict],
    model,
    processor,
    dtype,
) -> list[dict]:
    """Run all trials with the selected phases and return results."""
    from tqdm import tqdm

    phases = set(args.phases)
    use_phase1 = 1 in phases
    use_phase2 = 2 in phases
    use_phase3 = 3 in phases
    use_phase4 = 4 in phases
    use_phase5 = 5 in phases

    system_prompt = None
    if use_phase3:
        system_prompt = (
            "You are solving elementary math problems. "
            "Always respond with exactly one letter: the correct option."
        )

    max_tokens = 128 if use_phase5 else 64
    results = []

    for trial in tqdm(trials, desc="Evaluating", unit="trial"):
        # Build prompt
        if use_phase1:
            prompt = build_prompt_phase1(trial["row"], trial["options"])
        else:
            prompt = build_prompt_baseline(trial["row"], trial["options"])

        if use_phase4:
            prompt = apply_phase4_fewshot(prompt, trial["trial_type"])

        if use_phase5:
            prompt = apply_phase5_cot(prompt, trial["trial_type"])

        if use_phase3:
            prompt = apply_phase3_system(prompt)

        # Generate
        response = generate(
            model, processor, dtype, args.device,
            prompt, max_new_tokens=max_tokens,
            system_prompt=system_prompt,
        )

        # Parse
        if use_phase2:
            predicted_label, reason = parse_answer_phase2(
                response, LABELS[:len(trial["options"])], trial["options"]
            )
        else:
            predicted_label, reason = parse_answer_baseline(
                response, LABELS[:len(trial["options"])]
            )

        is_correct = predicted_label == trial["correct_label"]
        results.append({
            "item_uid": trial["item_uid"],
            "trial_type": trial["trial_type"],
            "correct_label": trial["correct_label"],
            "predicted_label": predicted_label,
            "is_correct": is_correct,
            "parsed": predicted_label is not None,
            "raw_response": response,
            "prompt_preview": prompt[:200],
        })

    return results


def print_results(results: list[dict], phases: list[int]) -> None:
    """Print accuracy summary by trial type."""
    phase_str = "+".join(str(p) for p in sorted(phases)) if phases else "baseline"
    print(f"\n{'='*70}")
    print(f"  RESULTS — Phases: {phase_str}")
    print(f"{'='*70}\n")

    # By trial type
    by_type: dict[str, dict] = {}
    for r in results:
        tt = r["trial_type"]
        row = by_type.setdefault(tt, {"n": 0, "correct": 0, "parsed": 0})
        row["n"] += 1
        row["correct"] += int(r["is_correct"])
        row["parsed"] += int(r["parsed"])

    total_n = len(results)
    total_correct = sum(r["is_correct"] for r in results)
    total_parsed = sum(r["parsed"] for r in results)

    print(f"{'Trial Type':<40} {'N':>4} {'Acc':>7} {'Parse%':>7} {'Chance':>7}")
    print("-" * 70)

    for tt, row in sorted(by_type.items(), key=lambda kv: kv[1]["correct"] / max(kv[1]["n"], 1), reverse=True):
        n = row["n"]
        acc = row["correct"] / n if n else 0
        parse = row["parsed"] / n if n else 0
        chance = 0.25 if n > 2 else 0.5
        print(f"{tt:<40} {n:>4} {acc:>7.1%} {parse:>7.1%} {chance:>7.1%}")

    print("-" * 70)
    overall_acc = total_correct / total_n if total_n else 0
    overall_parse = total_parsed / total_n if total_n else 0
    print(f"{'OVERALL':<40} {total_n:>4} {overall_acc:>7.1%} {overall_parse:>7.1%}")
    print()

    # Show some failure examples
    failures = [r for r in results if not r["is_correct"]]
    unparsed = [r for r in results if not r["parsed"]]
    if unparsed:
        print(f"Unparsed responses ({len(unparsed)} total). First 5:")
        for r in unparsed[:5]:
            print(f"  [{r['trial_type']}] {r['item_uid']}")
            print(f"    Response: {r['raw_response'][:120]}")
            print(f"    Correct:  {r['correct_label']}")
        print()


def save_results(results: list[dict], output_path: Path, phases: list[int]) -> None:
    """Write detailed results to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "item_uid", "trial_type", "correct_label", "predicted_label",
            "is_correct", "parsed", "raw_response", "prompt_preview",
        ])
        writer.writeheader()
        writer.writerows(results)
    print(f"Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Phased experiment for egma-math prompt/parsing improvements.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--phase",
        nargs="+",
        type=int,
        default=[0],
        dest="phases",
        help="Phase(s) to activate: 0=baseline, 1=reformat, 2=parse-fallback, "
        "3=system-prompt, 4=few-shot, 5=cot. Combine: --phase 1 2 3",
    )
    parser.add_argument("--model-id", default="HuggingFaceTB/SmolVLM2-2.2B-Instruct")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--data-root", default=str(PROJECT_ROOT / "data"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "results" / "prompt_optimization/egma-math/qwen-0.8b"))
    parser.add_argument("--limit", type=int, default=None, help="Limit number of trials (for quick testing)")
    args = parser.parse_args()

    if args.device == "auto":
        try:
            import torch
            if torch.cuda.is_available():
                args.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                args.device = "mps"
            else:
                args.device = "cpu"
        except ImportError:
            args.device = "cpu"

    phases = sorted(set(args.phases))
    if phases == [0]:
        phases = []

    phase_names = {
        1: "reformat-prompts",
        2: "parse-fallback",
        3: "system-prompt",
        4: "few-shot",
        5: "chain-of-thought",
    }
    active = [phase_names.get(p, f"phase{p}") for p in phases]
    print(f"Active phases: {active or ['baseline']}")
    print(f"Model: {args.model_id}")
    print(f"Device: {args.device}")
    print()

    # Load data
    print("Loading manifest...")
    manifest = load_manifest(Path(args.data_root))
    trials = build_trials(manifest)
    print(f"Loaded {len(trials)} trials across {manifest['trial_type'].nunique()} types")

    if args.limit:
        trials = trials[:args.limit]
        print(f"Limited to {len(trials)} trials")

    # Load model
    print(f"\nLoading model {args.model_id}...")
    t0 = time.time()
    model, processor_obj, dtype = load_model(args.model_id, args.device)
    print(f"Model loaded in {time.time() - t0:.1f}s\n")

    # Run
    args.phases = phases
    results = run_experiment(args, trials, model, processor_obj, dtype)

    # Report
    print_results(results, phases)

    # Save
    phase_tag = "_".join(str(p) for p in phases) if phases else "baseline"
    output_path = Path(args.output_dir) / f"phase_{phase_tag}.csv"
    save_results(results, output_path, phases)


if __name__ == "__main__":
    main()
