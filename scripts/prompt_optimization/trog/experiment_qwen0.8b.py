#!/usr/bin/env python3
"""Phased experiment: test prompt/parsing improvements for TROG on Qwen3.5-0.8B.

Each phase can be toggled independently via CLI flags.
Run with --phase 0 for baseline, then --phase 1, --phase 2, etc.

Phases:
    0  Baseline (current manifest prompts, default parsing)
    1  Structured multiline prompt (sentence on its own line, clear option layout)
    2  Enhanced parsing (reverse-scan, "image X" patterns, last-letter fallback)
    3  System prompt + strict answer format suffix
    4  Visual grounding hints for complex grammar types
    5  Sentence decomposition / CoT for complex grammar types

Example:
    python scripts/experiment_trog_phases.py --phase 0
    python scripts/experiment_trog_phases.py --phase 1 2 3    # combine phases
    python scripts/experiment_trog_phases.py --phase 1 2 3 4 5  # all improvements
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

LABELS = ["A", "B", "C", "D"]

COMPLEX_GRAMMAR_TYPES = {
    "reversible active",
    "reversible passive",
    "relative clause",
    "embedded sentence",
    "X but not Y",
    "not only X but also Y",
    "neither X nor Y",
    "postmodified subject",
    "post modified subject",
    "reversible passive; relative clause",
    "conditional",
    "dependent clause",
    "causal",
    "gerund phrase",
    "advanced conjunction coordinating",
    "advanced preposition location",
    "advanced preposition direction",
    "compound preposition condition",
    "disjunctive",
    "temporal",
    "additive",
    "prepositional phrase",
}

SPATIAL_TYPES = {
    "in and on",
    "above and below",
    "advanced preposition location",
    "advanced preposition direction",
}

QWEN_DEFAULT_SYSTEM = (
    "You are a helpful assistant. "
    "Answer with only a single letter: A, B, C, or D. Do not explain."
)

MAX_IMAGE_SIZE = 512

# ---------------------------------------------------------------------------
# Prompt builders per phase
# ---------------------------------------------------------------------------

def build_prompt_baseline(row: dict, prompt_phrase: str) -> str:
    """Phase 0: current manifest-style prompt (single line, semicolons)."""
    full_prompt = str(row.get("full_prompt", ""))
    if full_prompt and full_prompt not in {"", "NA", "nan"}:
        prompt = full_prompt.replace("<prompt_phrase>", prompt_phrase)
    else:
        prompt_tmpl = str(row.get("prompt", "")).strip()
        prompt = f'{prompt_tmpl} "{prompt_phrase}". Answer with A, B, C, or D. A: <image1>; B: <image2>; C: <image3>; D: <image4>'
    return prompt


def build_prompt_phase1(row: dict, prompt_phrase: str) -> str:
    """Phase 1: structured multiline prompt with clear layout."""
    trial_type = str(row.get("trial_type", "")).strip()

    if trial_type in ("noun", "verb", "adjective"):
        stem = f'Which image shows: "{prompt_phrase}"?'
    elif trial_type == "two element combination":
        stem = f'Which image shows: "{prompt_phrase}"?'
    elif trial_type in SPATIAL_TYPES:
        stem = f'Which image matches: "{prompt_phrase}"?'
    else:
        stem = f'Which image matches the sentence: "{prompt_phrase}"?'

    lines = [
        stem,
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


def apply_phase4_grounding(prompt: str, trial_type: str) -> str:
    """Phase 4: visual grounding hints for complex grammar types."""
    if trial_type in ("reversible passive", "reversible active",
                      "reversible passive; relative clause"):
        hint = "Pay close attention to WHO is performing the action and WHO is receiving it."
    elif trial_type in ("relative clause", "postmodified subject",
                        "post modified subject"):
        hint = "Focus on which object or person the description modifies."
    elif trial_type in ("embedded sentence",):
        hint = "This sentence has a clause inside another clause. Identify the subject of each part."
    elif trial_type in SPATIAL_TYPES:
        hint = "Look carefully at the spatial positions of the objects in each image."
    elif trial_type in ("X but not Y", "not only X but also Y",
                        "neither X nor Y", "disjunctive"):
        hint = "The sentence specifies what IS and what IS NOT shown. Check both conditions."
    elif trial_type in COMPLEX_GRAMMAR_TYPES:
        hint = "Read the sentence carefully. Identify all subjects, actions, and objects before choosing."
    else:
        return prompt

    return f"{hint}\n\n{prompt}"


def apply_phase5_decomposition(prompt: str, trial_type: str) -> str:
    """Phase 5: sentence decomposition / CoT for complex grammar types."""
    if trial_type not in COMPLEX_GRAMMAR_TYPES:
        return prompt

    cot = (
        "Step 1: Identify the key elements in the sentence "
        "(who/what, action, recipient/location).\n"
        "Step 2: Look at each image and check which one matches ALL elements.\n"
        "Step 3: State your final answer as a single letter."
    )
    return prompt.rstrip() + f"\n\n{cot}"


# ---------------------------------------------------------------------------
# Answer parsing
# ---------------------------------------------------------------------------

def parse_answer_baseline(text: str, option_labels: list[str]) -> tuple[Optional[str], str]:
    """Standard label parser (reproduced from base.py for standalone use)."""
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

    # "Image A" / "Option A" / "Picture A" patterns
    m = re.search(r'(?:image|option|picture|choice)\s+([A-D])\b', cleaned, re.IGNORECASE)
    if m:
        answer = m.group(1).upper()
        if answer in labels_upper:
            return answer, cleaned

    # "I choose A" / "I select A" / "I pick A"
    m = re.search(r'(?:choose|select|pick)\s+([A-D])\b', cleaned, re.IGNORECASE)
    if m:
        answer = m.group(1).upper()
        if answer in labels_upper:
            return answer, cleaned

    # "(A)" pattern
    m = re.search(r'\(([A-D])\)', cleaned)
    if m:
        answer = m.group(1).upper()
        if answer in labels_upper:
            return answer, cleaned

    # Reverse sentence scan: find last sentence containing a valid label
    sentences = re.split(r"[.!?\n]", cleaned)
    for sentence in reversed(sentences):
        sentence = sentence.strip()
        for label_str in option_labels:
            if re.search(rf"\b{re.escape(label_str)}\b", sentence, re.IGNORECASE):
                return label_str.upper(), sentence

    # Last single letter A-D in the text
    letters = re.findall(r'\b([A-D])\b', cleaned)
    if letters:
        last = letters[-1].upper()
        if last in labels_upper:
            return last, cleaned

    return None, text


# ---------------------------------------------------------------------------
# Model loading & generation
# ---------------------------------------------------------------------------

def _resize_image(img, max_size: int = MAX_IMAGE_SIZE):
    """Resize PIL image so the longest side is at most max_size, preserving aspect ratio."""
    w, h = img.size
    if max(w, h) <= max_size:
        return img
    scale = max_size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    return img.resize((new_w, new_h))


def load_model(model_id: str, device: str):
    """Load Qwen3.5 model and processor."""
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText

    dtype = torch.bfloat16
    processor = AutoProcessor.from_pretrained(model_id, padding_side="left")
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        dtype=dtype,
        attn_implementation="sdpa",
    ).to(device)
    model.eval()
    return model, processor, dtype


def generate(
    model,
    processor,
    dtype,
    device: str,
    prompt_text: str,
    image_paths: list[str],
    max_new_tokens: int = 128,
    system_prompt: str | None = None,
) -> str:
    """Generate a response from Qwen3.5 with images (resized to MAX_IMAGE_SIZE)."""
    import torch
    from PIL import Image

    pil_images = [_resize_image(Image.open(p).convert("RGB")) for p in image_paths]

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


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks from Qwen3.5 outputs."""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if not cleaned and text:
        return text
    return cleaned


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


# ---------------------------------------------------------------------------
# Manifest loading
# ---------------------------------------------------------------------------

def load_manifest(data_root: Path):
    """Load TROG rows from manifest.csv and return list of dicts."""
    manifest_path = data_root / "assets" / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    rows = []
    with open(manifest_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["task"] == "trog":
                rows.append(row)
    return rows


def build_image_index(image_dir: Path) -> dict[str, Path]:
    """Map image stems and filenames to their paths."""
    index = {}
    if not image_dir.is_dir():
        return index
    for path in image_dir.iterdir():
        if path.is_file():
            index[path.stem] = path
            index[path.name] = path
    return index


def build_trials(manifest_rows: list[dict], image_dir: Path) -> list[dict]:
    """Build trial dicts from manifest rows with shuffled options and image paths."""
    image_index = build_image_index(image_dir)
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
        for option in all_options:
            path = image_index.get(option.strip())
            if path is None:
                missing = True
                break
            option_image_paths.append(str(path))

        if missing:
            skipped += 1
            continue

        prompt_phrase = str(row.get("prompt_phrase", "")).strip()
        if prompt_phrase in ("NA", "nan"):
            prompt_phrase = ""

        trials.append({
            "item_uid": row["item_uid"],
            "trial_type": str(row.get("trial_type", "")).strip(),
            "row": row,
            "options": all_options,
            "option_image_paths": option_image_paths,
            "correct_label": correct_label,
            "prompt_phrase": prompt_phrase,
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

    system_prompt = QWEN_DEFAULT_SYSTEM
    if use_phase3:
        system_prompt = (
            "You are a visual reasoning assistant. "
            "You match English sentences to the image that best depicts them. "
            "Always respond with exactly one letter: A, B, C, or D."
        )

    max_tokens = 256 if use_phase5 else 128
    results = []

    for trial in tqdm(trials, desc="Evaluating", unit="trial"):
        # Build prompt
        if use_phase1:
            prompt = build_prompt_phase1(trial["row"], trial["prompt_phrase"])
        else:
            prompt = build_prompt_baseline(trial["row"], trial["prompt_phrase"])

        if use_phase4:
            prompt = apply_phase4_grounding(prompt, trial["trial_type"])

        if use_phase5:
            prompt = apply_phase5_decomposition(prompt, trial["trial_type"])

        if use_phase3:
            prompt = apply_phase3_system(prompt)

        # Generate
        response = generate(
            model, processor, dtype, args.device,
            prompt, trial["option_image_paths"],
            max_new_tokens=max_tokens,
            system_prompt=system_prompt,
        )

        # Parse
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

    print(f"{'Trial Type':<45} {'N':>4} {'Acc':>7} {'Parse%':>7} {'Chance':>7}")
    print("-" * 75)

    for tt, row in sorted(by_type.items(), key=lambda kv: kv[1]["correct"] / max(kv[1]["n"], 1), reverse=True):
        n = row["n"]
        acc = row["correct"] / n if n else 0
        parse = row["parsed"] / n if n else 0
        print(f"{tt:<45} {n:>4} {acc:>7.1%} {parse:>7.1%} {0.25:>7.1%}")

    print("-" * 75)
    overall_acc = total_correct / total_n if total_n else 0
    overall_parse = total_parsed / total_n if total_n else 0
    print(f"{'OVERALL':<45} {total_n:>4} {overall_acc:>7.1%} {overall_parse:>7.1%}")
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
        description="Phased experiment for TROG prompt/parsing improvements.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--phase",
        nargs="+",
        type=int,
        default=[0],
        dest="phases",
        help="Phase(s) to activate: 0=baseline, 1=multiline, 2=enhanced-parse, "
        "3=system-prompt, 4=visual-grounding, 5=decomposition. Combine: --phase 1 2 3",
    )
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--data-root", default=str(PROJECT_ROOT / "data"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "results" / "prompt_optimization/trog/qwen-0.8b"))
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
        1: "multiline-prompt",
        2: "enhanced-parse",
        3: "system-prompt",
        4: "visual-grounding",
        5: "sentence-decomposition",
    }
    active = [phase_names.get(p, f"phase{p}") for p in phases]
    print(f"Active phases: {active or ['baseline']}")
    print(f"Model: {args.model_id}")
    print(f"Device: {args.device}")
    print()

    # Load data
    print("Loading manifest...")
    data_root = Path(args.data_root)
    manifest_rows = load_manifest(data_root)
    image_dir = data_root / "assets" / "2026-03-26" / "visual" / "trog"
    trials = build_trials(manifest_rows, image_dir)
    print(f"Loaded {len(trials)} trials")

    trial_types = set(t["trial_type"] for t in trials)
    print(f"  {len(trial_types)} trial types")

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
