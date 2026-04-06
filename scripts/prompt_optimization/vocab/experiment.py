#!/usr/bin/env python3
"""Phased experiment: test prompt/parsing improvements for Vocab on Qwen3.5-4B.

Each phase can be toggled independently via CLI flags.
Run with --phase 0 for baseline, then --phase 1, --phase 2, etc.

Phases:
    0  Baseline (current manifest prompts, Qwen default system prompt)
    1  Structured multiline prompt (word prominent, block option layout)
    2  Enhanced parsing (reverse-scan, "image X" patterns, last-letter fallback)
    3  Task-specific system prompt (visual vocabulary expert)
    4  Distractor awareness hint (encourage careful comparison)
    5  Visual elimination CoT (describe each image, then match)

Example:
    python scripts/experiment_vocab_phases.py --phase 0
    python scripts/experiment_vocab_phases.py --phase 1 2 3
    python scripts/experiment_vocab_phases.py --phase 1 2 3 4 5
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

LABELS = ["A", "B", "C", "D"]

QWEN_DEFAULT_SYSTEM = (
    "You are a helpful assistant. "
    "Answer with only a single letter: A, B, C, or D. Do not explain."
)


# ---------------------------------------------------------------------------
# Image resolution (mirrors vocab.py normalisation)
# ---------------------------------------------------------------------------

def _normalize_term(term: str) -> set[str]:
    t = term.strip().lower()
    if not t:
        return set()
    compact = re.sub(r"[^a-z0-9]+", "", t)
    snake = re.sub(r"[^a-z0-9]+", "_", t).strip("_")
    dash = re.sub(r"[^a-z0-9]+", "-", t).strip("-")
    return {t, t.replace(" ", "_"), t.replace(" ", ""), t.replace("_", ""),
            snake, dash, compact}


def _build_image_index(directory: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    if not directory.is_dir():
        return index
    for path in directory.iterdir():
        if path.is_file():
            for key in _normalize_term(path.stem):
                index.setdefault(key, path)
    return index


def _resolve_image(term: str, image_index: dict[str, Path]) -> Path | None:
    for candidate in _normalize_term(term):
        if candidate in image_index:
            return image_index[candidate]
    return None


# ---------------------------------------------------------------------------
# Prompt builders per phase
# ---------------------------------------------------------------------------

def build_prompt_baseline(row: dict, word: str) -> str:
    """Phase 0: current manifest prompt (single line, semicolons)."""
    full_prompt = str(row.get("full_prompt", ""))
    if full_prompt and full_prompt not in {"", "NA", "nan"}:
        return full_prompt.replace("<prompt_phrase>", word)
    return (
        f'Choose the image that matches the text: "{word}". '
        "Answer with A, B, C, or D. A: <image1>; B: <image2>; C: <image3>; D: <image4>"
    )


def build_prompt_phase1(row: dict, word: str) -> str:
    """Phase 1: structured multiline prompt with prominent word."""
    lines = [
        f'Which image shows: "{word}"?',
        "",
        "A: <image1>",
        "B: <image2>",
        "C: <image3>",
        "D: <image4>",
        "",
        "Answer with one letter.",
    ]
    return "\n".join(lines)


def apply_phase3_system(prompt: str) -> str:
    """Phase 3: append strict answer format suffix."""
    return prompt.rstrip() + "\n\nRespond with exactly one letter: A, B, C, or D. Nothing else."


def apply_phase4_distractor(prompt: str) -> str:
    """Phase 4: distractor awareness hint."""
    hint = (
        "Only one image matches the word. The other three are unrelated distractors. "
        "Look at each image carefully before choosing."
    )
    return f"{hint}\n\n{prompt}"


def apply_phase5_elimination(prompt: str, word: str) -> str:
    """Phase 5: visual elimination CoT."""
    cot = (
        f'For each image, briefly identify what it shows. '
        f'Then select the one that matches the word "{word}". '
        f'State your final answer as a single letter.'
    )
    return prompt.rstrip() + f"\n\n{cot}"


# ---------------------------------------------------------------------------
# Answer parsing
# ---------------------------------------------------------------------------

def parse_answer_baseline(text: str, option_labels: list[str]) -> tuple[Optional[str], str]:
    """Standard label parser (reproduced from base.py)."""
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
    text: str, option_labels: list[str]
) -> tuple[Optional[str], str]:
    """Phase 2: enhanced parsing with multiple fallback strategies."""
    label, reason = parse_answer_baseline(text, option_labels)
    if label is not None:
        return label, reason

    labels_upper = [la.upper() for la in option_labels]
    cleaned = text.strip()

    m = re.search(r'(?:image|option|picture|choice)\s+([A-D])\b', cleaned, re.IGNORECASE)
    if m:
        answer = m.group(1).upper()
        if answer in labels_upper:
            return answer, cleaned

    m = re.search(r'(?:choose|select|pick)\s+([A-D])\b', cleaned, re.IGNORECASE)
    if m:
        answer = m.group(1).upper()
        if answer in labels_upper:
            return answer, cleaned

    m = re.search(r'\(([A-D])\)', cleaned)
    if m:
        answer = m.group(1).upper()
        if answer in labels_upper:
            return answer, cleaned

    sentences = re.split(r"[.!?\n]", cleaned)
    for sentence in reversed(sentences):
        sentence = sentence.strip()
        for label_str in option_labels:
            if re.search(rf"\b{re.escape(label_str)}\b", sentence, re.IGNORECASE):
                return label_str.upper(), sentence

    letters = re.findall(r'\b([A-D])\b', cleaned)
    if letters:
        last = letters[-1].upper()
        if last in labels_upper:
            return last, cleaned

    return None, text


# ---------------------------------------------------------------------------
# Model loading & generation
# ---------------------------------------------------------------------------

def load_model(model_id: str, device: str):
    """Load Qwen3.5 model and processor."""
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText

    dtype = torch.bfloat16
    processor = AutoProcessor.from_pretrained(model_id, padding_side="left")
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, dtype=dtype, attn_implementation="sdpa",
    ).to(device)
    model.eval()
    return model, processor, dtype


def generate(
    model, processor, dtype, device: str,
    prompt_text: str, image_paths: list[str],
    max_new_tokens: int = 128,
    system_prompt: str | None = None,
) -> str:
    """Generate a response from Qwen3.5 with images."""
    import torch
    from PIL import Image

    pil_images = [Image.open(p).convert("RGB") for p in image_paths]
    content = _build_content(prompt_text, pil_images)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": content})

    try:
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

    inputs = processor(
        text=[text],
        images=pil_images if pil_images else None,
        return_tensors="pt",
        padding=True,
    ).to(device)

    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        output_ids = model.generate(
            **inputs, do_sample=False, max_new_tokens=max_new_tokens
        )

    generated_ids = output_ids[:, input_len:]
    raw = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return _strip_thinking(raw)


def _build_content(prompt_text: str, pil_images: list) -> list[dict]:
    """Build content list interleaving images at <imageN> placeholders."""
    content: list[dict] = []
    if pil_images and re.search(r"<image\d+>", prompt_text):
        parts = re.split(r"(<image\d+>)", prompt_text)
        for part in parts:
            m = re.match(r"<image(\d+)>", part)
            if m:
                idx = int(m.group(1)) - 1
                if idx < len(pil_images):
                    content.append({"type": "image", "image": pil_images[idx]})
            elif part.strip():
                content.append({"type": "text", "text": part.strip()})
    else:
        for img in pil_images:
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": prompt_text})
    return content


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks from Qwen3.5 outputs."""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if not cleaned and text:
        return text
    return cleaned


# ---------------------------------------------------------------------------
# Manifest loading
# ---------------------------------------------------------------------------

def load_manifest(data_root: Path) -> list[dict]:
    """Load vocab rows from manifest.csv."""
    manifest_path = data_root / "assets" / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    rows = []
    with open(manifest_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["task"] == "vocab":
                rows.append(row)
    return rows


def build_trials(manifest_rows: list[dict], image_dir: Path) -> list[dict]:
    """Build trial dicts with shuffled options and image paths."""
    image_index = _build_image_index(image_dir)
    trials = []
    skipped = 0

    for row in manifest_rows:
        answer = str(row["answer"]).strip()
        alternatives = [a.strip() for a in row["response_alternatives"].split(",") if a.strip()]
        all_options = [answer] + alternatives

        rng = random.Random(row["item_uid"])
        rng.shuffle(all_options)
        correct_idx = all_options.index(answer)
        correct_label = LABELS[correct_idx]

        option_image_paths = []
        missing = False
        for word in all_options:
            path = _resolve_image(word.strip(), image_index)
            if path is None:
                missing = True
                break
            option_image_paths.append(str(path))

        if missing:
            skipped += 1
            continue

        word = str(row.get("prompt_phrase", answer)).strip()
        if word in ("NA", "nan", ""):
            word = answer

        trials.append({
            "item_uid": row["item_uid"],
            "trial_type": str(row.get("trial_type", "test")).strip(),
            "row": row,
            "options": all_options,
            "option_image_paths": option_image_paths,
            "correct_label": correct_label,
            "word": word,
        })

    if skipped:
        print(f"  Skipped {skipped} trials with missing images", file=sys.stderr)
    return trials


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

def run_experiment(
    args: argparse.Namespace,
    trials: list[dict],
    model, processor, dtype,
) -> list[dict]:
    """Run all trials with the selected phases and return results."""
    from tqdm import tqdm

    phases = set(args.phases)
    use_phase1 = 1 in phases
    use_phase2 = 2 in phases
    use_phase3 = 3 in phases
    use_phase4 = 4 in phases
    use_phase5 = 5 in phases

    system_prompt = QWEN_DEFAULT_SYSTEM
    if use_phase3:
        system_prompt = (
            "You are a visual vocabulary expert. "
            "You match English words to their corresponding images. "
            "Always respond with exactly one letter: A, B, C, or D."
        )

    max_tokens = 256 if use_phase5 else 128
    results = []

    for trial in tqdm(trials, desc="Evaluating", unit="trial"):
        word = trial["word"]

        if use_phase1:
            prompt = build_prompt_phase1(trial["row"], word)
        else:
            prompt = build_prompt_baseline(trial["row"], word)

        if use_phase4:
            prompt = apply_phase4_distractor(prompt)

        if use_phase5:
            prompt = apply_phase5_elimination(prompt, word)

        if use_phase3:
            prompt = apply_phase3_system(prompt)

        response = generate(
            model, processor, dtype, args.device,
            prompt, trial["option_image_paths"],
            max_new_tokens=max_tokens,
            system_prompt=system_prompt,
        )

        if use_phase2:
            predicted_label, reason = parse_answer_phase2(
                response, LABELS[:len(trial["options"])]
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

    print(f"{'Trial Type':<20} {'N':>4} {'Acc':>7} {'Parse%':>7} {'Chance':>7}")
    print("-" * 50)

    for tt, row in sorted(by_type.items(), key=lambda kv: kv[1]["correct"] / max(kv[1]["n"], 1), reverse=True):
        n = row["n"]
        acc = row["correct"] / n if n else 0
        parse = row["parsed"] / n if n else 0
        print(f"{tt:<20} {n:>4} {acc:>7.1%} {parse:>7.1%} {0.25:>7.1%}")

    print("-" * 50)
    overall_acc = total_correct / total_n if total_n else 0
    overall_parse = total_parsed / total_n if total_n else 0
    print(f"{'OVERALL':<20} {total_n:>4} {overall_acc:>7.1%} {overall_parse:>7.1%}")
    print()

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
        description="Phased experiment for Vocab prompt/parsing improvements.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--phase", nargs="+", type=int, default=[0], dest="phases",
        help="Phase(s) to activate: 0=baseline, 1=multiline, 2=enhanced-parse, "
        "3=system-prompt, 4=distractor-hint, 5=elimination-cot. Combine: --phase 1 2 3",
    )
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--data-root", default=str(PROJECT_ROOT / "data"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "results" / "prompt_optimization/vocab/qwen-0.8b"))
    parser.add_argument("--limit", type=int, default=None)
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
        1: "multiline-prompt",
        2: "enhanced-parse",
        3: "system-prompt",
        4: "distractor-hint",
        5: "elimination-cot",
    }
    active = [phase_names.get(p, f"phase{p}") for p in phases]
    print(f"Active phases: {active or ['baseline']}")
    print(f"Model: {args.model_id}")
    print(f"Device: {args.device}")
    print()

    print("Loading manifest...")
    data_root = Path(args.data_root)
    manifest_rows = load_manifest(data_root)
    image_dir = data_root / "assets" / "2026-03-26" / "visual" / "vocab"
    trials = build_trials(manifest_rows, image_dir)
    print(f"Loaded {len(trials)} trials")

    trial_types = set(t["trial_type"] for t in trials)
    print(f"  Trial types: {sorted(trial_types)}")

    if args.limit:
        trials = trials[:args.limit]
        print(f"Limited to {len(trials)} trials")

    print(f"\nLoading model {args.model_id}...")
    t0 = time.time()
    model, processor_obj, dtype = load_model(args.model_id, args.device)
    print(f"Model loaded in {time.time() - t0:.1f}s\n")

    args.phases = phases
    results = run_experiment(args, trials, model, processor_obj, dtype)

    print_results(results, phases)

    phase_tag = "_".join(str(p) for p in phases) if phases else "baseline"
    output_path = Path(args.output_dir) / f"phase_{phase_tag}.csv"
    save_results(results, output_path, phases)


if __name__ == "__main__":
    main()
