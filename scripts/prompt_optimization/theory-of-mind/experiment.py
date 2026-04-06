#!/usr/bin/env python3
"""Phased experiment: test prompt/parsing improvements for Theory-of-Mind (Stories) on Qwen3.5-0.8B.

Each phase can be toggled independently via CLI flags.
Run with --phase 0 for baseline, then --phase 1, --phase 2, etc.

Phases:
    0  Baseline (current manifest prompts, Qwen default system prompt)
    1  Structured prompt (story / question / options clearly separated)
    2  Enhanced parsing (reverse-scan, "image X" patterns, last-letter fallback)
    3  Task-specific system prompt (Theory of Mind reasoning)
    4  False belief hint (character knowledge tracking reminder)
    5  Mental state CoT (step-by-step reasoning for beliefs and emotions)
    6  Answer-last CoT (reasoning then forced "My answer is [X]" as final line)

Example:
    python scripts/experiment_stories_phases.py --phase 0
    python scripts/experiment_stories_phases.py --phase 1 2 3
    python scripts/experiment_stories_phases.py --phase 1 2 3 4 5
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

FALSE_BELIEF_TYPES = {"false_belief_question", "attribution_question"}
EMOTION_TYPES = {"emotion_reasoning_question"}
ALL_COMPLEX_TYPES = FALSE_BELIEF_TYPES | EMOTION_TYPES | {"action_question"}


# ---------------------------------------------------------------------------
# Image resolution (mirrors image_index.py)
# ---------------------------------------------------------------------------

def _build_image_index(directory: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    if not directory.is_dir():
        return index
    for path in directory.iterdir():
        if path.is_file():
            index[path.stem] = path
            index[path.name] = path
    return index


# ---------------------------------------------------------------------------
# Prompt builders per phase
# ---------------------------------------------------------------------------

def _extract_story_and_question(full_prompt: str) -> tuple[str, str]:
    """Split full_prompt into narrative and question+options portion."""
    answer_patterns = [
        r"(Answer\s+A\s*(?:,\s*(?:or\s+)?B\s*(?:,\s*(?:or\s+)?C\s*(?:,\s*(?:or\s+)?D)?)?)?\s*(?:,?\s*or\s+[A-D])?\.\s*A:\s*<image)",
        r"(Answer\s+with\s+)",
        r"(Now\s+pick\s+one\s+of\s+the\s+)",
    ]
    for pat in answer_patterns:
        m = re.search(pat, full_prompt)
        if m:
            before = full_prompt[:m.start()].strip()
            return before, full_prompt[m.start():]

    sentences = re.split(r'(?<=[.!?])\s+', full_prompt)
    if len(sentences) > 1:
        question_idx = -1
        for i, s in enumerate(sentences):
            if "?" in s:
                question_idx = i
        if question_idx >= 0:
            story = " ".join(sentences[:question_idx])
            question = " ".join(sentences[question_idx:])
            return story, question

    return full_prompt, ""


def build_prompt_baseline(row: dict, prompt_phrase: str) -> str:
    """Phase 0: current manifest prompt."""
    full_prompt = str(row.get("full_prompt", ""))
    if full_prompt and full_prompt not in {"", "NA", "nan"}:
        prompt = full_prompt.replace("<prompt_phrase>", prompt_phrase)
    else:
        prompt = str(row.get("prompt", "")).strip()
    return prompt


def build_prompt_phase1(row: dict, prompt_phrase: str, n_options: int) -> str:
    """Phase 1: structured prompt with story/question/options separated."""
    raw = build_prompt_baseline(row, prompt_phrase)

    story_part, answer_part = _extract_story_and_question(raw)

    # Extract the actual question (last sentence ending with ?)
    sentences = re.split(r'(?<=[.!?])\s+', story_part)
    question = ""
    story_sentences = []
    for s in sentences:
        if "?" in s and not question:
            question = s
        elif question:
            question += " " + s
        else:
            story_sentences.append(s)

    if not question:
        question = sentences[-1] if sentences else raw
        story_sentences = sentences[:-1]

    story_text = re.sub(r"<image\d+>\s*", "", " ".join(story_sentences)).strip()
    question = re.sub(r"<image\d+>\s*", "", question).strip()

    option_lines = []
    for i in range(n_options):
        option_lines.append(f"{LABELS[i]}: <image{i+1}>")

    lines = []
    if story_text:
        lines.append("Story:")
        lines.append(story_text)
        lines.append("")
    lines.append(f"Question: {question}")
    lines.append("")
    lines.extend(option_lines)
    lines.append("")
    lines.append("Answer with one letter.")
    return "\n".join(lines)


def apply_phase3_system(prompt: str) -> str:
    """Phase 3: append strict answer format suffix."""
    return prompt.rstrip() + "\n\nRespond with exactly one letter. Nothing else."


def apply_phase4_belief_hint(prompt: str, trial_type: str) -> str:
    """Phase 4: false belief / attribution hint."""
    if trial_type in FALSE_BELIEF_TYPES:
        hint = (
            "Important: A character only knows what they have personally "
            "seen or been told. Events that happened while they were away "
            "are unknown to them."
        )
        return f"{hint}\n\n{prompt}"
    if trial_type in EMOTION_TYPES:
        hint = (
            "Think about how the situation would make the character feel, "
            "based on what they know and experience."
        )
        return f"{hint}\n\n{prompt}"
    return prompt


def apply_phase5_cot(prompt: str, trial_type: str) -> str:
    """Phase 5: mental state reasoning CoT."""
    if trial_type in FALSE_BELIEF_TYPES:
        cot = (
            "Step 1: What does this character KNOW (what did they see)?\n"
            "Step 2: What actually happened (the reality)?\n"
            "Step 3: Based on their knowledge (not reality), what would they think or do?\n"
            "State your final answer as a single letter."
        )
    elif trial_type in EMOTION_TYPES:
        cot = (
            "Step 1: What happened in the story?\n"
            "Step 2: How does this situation affect the character?\n"
            "Step 3: Which emotion best matches their experience?\n"
            "State your final answer as a single letter."
        )
    elif trial_type in ALL_COMPLEX_TYPES:
        cot = (
            "Think about what the characters know and feel. "
            "Then state your final answer as a single letter."
        )
    else:
        return prompt

    return prompt.rstrip() + f"\n\n{cot}"


def apply_phase6_answer_last_cot(prompt: str, trial_type: str) -> str:
    """Phase 6: answer-last CoT — reasoning steps followed by a forced final line.

    Forces the model to emit 'My answer is [X]' as the very last line so that
    even if the reasoning is long, the answer survives truncation and is always
    parseable.
    """
    if trial_type in FALSE_BELIEF_TYPES:
        cot = (
            "Step 1: What did this character SEE before they left?\n"
            "Step 2: What changed while they were gone?\n"
            "Step 3: What does the character still BELIEVE (based only on what they saw)?\n"
            "My answer is"
        )
    elif trial_type in EMOTION_TYPES:
        cot = (
            "Step 1: What happened in the story?\n"
            "Step 2: How would this make the character feel?\n"
            "My answer is"
        )
    elif trial_type in ALL_COMPLEX_TYPES:
        cot = (
            "Think step by step about the characters' knowledge and feelings.\n"
            "My answer is"
        )
    else:
        return prompt

    return prompt.rstrip() + f"\n\n{cot}"


def get_phase7_system_prompt(trial_type: str) -> str:
    """Phase 7: question-type-specific system prompt."""
    if trial_type in FALSE_BELIEF_TYPES:
        return (
            "You are reasoning about a character's beliefs in a story. "
            "IMPORTANT: The character did NOT witness certain events. "
            "Their belief is based ONLY on what they saw before leaving. "
            "Answer based on the character's knowledge, not on what actually happened. "
            "Always respond with exactly one letter."
        )
    elif trial_type in EMOTION_TYPES:
        return (
            "You are reasoning about a character's emotions in a story. "
            "Think about how the situation affects this specific character from their own perspective. "
            "Always respond with exactly one letter."
        )
    elif trial_type == "action_question":
        return (
            "You are reasoning about what a character will do next in a story. "
            "Think about what makes logical sense given what this character knows. "
            "Always respond with exactly one letter."
        )
    elif trial_type == "reality_check_question":
        return (
            "You are answering a factual question about what actually happened in a story. "
            "Always respond with exactly one letter."
        )
    else:
        return (
            "You are reasoning about characters in a story. "
            "Always respond with exactly one letter."
        )


def apply_phase8_perspective_hint(prompt: str, trial_type: str, row: dict) -> str:
    """Phase 8: append a one-line perspective anchor to the user turn."""
    char_name = ""
    m = re.search(r"Here is (\w+)\.", prompt)
    if not m:
        m = re.search(r"^(\w+) (?:put|went|placed|left|came)", prompt)
    if m:
        char_name = m.group(1)

    if trial_type in FALSE_BELIEF_TYPES:
        hint = (
            f"\nNote: {char_name} only knows what they saw before leaving. "
            f"Answer from {char_name}'s perspective, not from yours."
        ) if char_name else (
            "\nNote: The character only knows what they saw before leaving. "
            "Answer from the character's perspective, not from yours."
        )
    elif trial_type in EMOTION_TYPES:
        hint = (
            f"\nNote: Answer based on how {char_name} feels, not how you feel."
        ) if char_name else (
            "\nNote: Answer based on how the character feels, not how you feel."
        )
    else:
        return prompt

    return prompt.rstrip() + hint


# ---------------------------------------------------------------------------
# Answer parsing
# ---------------------------------------------------------------------------

def parse_answer_baseline(text: str, option_labels: list[str]) -> tuple[Optional[str], str]:
    """Standard label parser."""
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
    """Phase 2: enhanced parsing with fallbacks."""
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

MAX_IMAGE_SIZE = 512


def _resize_image(img, max_size: int = MAX_IMAGE_SIZE):
    """Resize PIL image so the longest side is at most max_size, preserving aspect ratio."""
    w, h = img.size
    if max(w, h) <= max_size:
        return img
    scale = max_size / max(w, h)
    return img.resize((int(w * scale), int(h * scale)))


def load_model(model_id: str, device: str):
    """Load InternVL3.5 model and processor."""
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText

    dtype = torch.bfloat16
    processor = AutoProcessor.from_pretrained(
        model_id, padding_side="left", trust_remote_code=True,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, dtype=dtype, attn_implementation="sdpa",
        low_cpu_mem_usage=True, trust_remote_code=True,
    ).to(device)
    model.eval()
    return model, processor, dtype


def generate(
    model, processor, dtype, device: str,
    prompt_text: str, image_paths: list[str],
    max_new_tokens: int = 128,
    system_prompt: str | None = None,
) -> str:
    """Generate a response from Qwen3.5-VL with images resized to MAX_IMAGE_SIZE."""
    import torch
    from PIL import Image

    pil_images = [_resize_image(Image.open(p).convert("RGB")) for p in image_paths] if image_paths else []
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
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if not cleaned and text:
        return text
    return cleaned


# ---------------------------------------------------------------------------
# Manifest loading
# ---------------------------------------------------------------------------

def load_manifest(data_root: Path) -> list[dict]:
    """Load theory-of-mind rows from manifest.csv."""
    manifest_path = data_root / "assets" / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    rows = []
    with open(manifest_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["task"] == "theory-of-mind":
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
        correct_label = LABELS[correct_idx] if correct_idx < len(LABELS) else "?"

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
        if prompt_phrase in ("NA", "nan", ""):
            prompt_phrase = ""

        trials.append({
            "item_uid": row["item_uid"],
            "trial_type": str(row.get("trial_type", "")).strip(),
            "row": row,
            "options": all_options,
            "option_image_paths": option_image_paths,
            "correct_label": correct_label,
            "prompt_phrase": prompt_phrase,
            "n_options": len(all_options),
            "chance_level": float(row.get("chance_level", 0.25)),
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
    from tqdm import tqdm

    phases = set(args.phases)
    use_phase1 = 1 in phases
    use_phase2 = 2 in phases
    use_phase3 = 3 in phases
    use_phase4 = 4 in phases
    use_phase5 = 5 in phases
    use_phase6 = 6 in phases
    use_phase7 = 7 in phases
    use_phase8 = 8 in phases

    system_prompt = QWEN_DEFAULT_SYSTEM
    if use_phase7:
        # Phase 7 system prompt is set per-trial inside the loop
        system_prompt = None  # placeholder; overridden below
    elif use_phase3:
        system_prompt = (
            "You are reasoning about characters in a story. "
            "Characters may have false beliefs about events they did not witness. "
            "Always respond with exactly one letter."
        )

    max_tokens = 512 if (use_phase5 or use_phase6) else 128
    results = []

    for trial in tqdm(trials, desc="Evaluating", unit="trial"):
        if use_phase1:
            prompt = build_prompt_phase1(trial["row"], trial["prompt_phrase"], trial["n_options"])
        else:
            prompt = build_prompt_baseline(trial["row"], trial["prompt_phrase"])

        if use_phase4:
            prompt = apply_phase4_belief_hint(prompt, trial["trial_type"])

        if use_phase5:
            prompt = apply_phase5_cot(prompt, trial["trial_type"])

        if use_phase6:
            prompt = apply_phase6_answer_last_cot(prompt, trial["trial_type"])

        if use_phase8:
            prompt = apply_phase8_perspective_hint(prompt, trial["trial_type"], trial["row"])

        if use_phase3:
            prompt = apply_phase3_system(prompt)

        # Phase 7: per-trial system prompt (overrides global)
        trial_system = system_prompt
        if use_phase7:
            trial_system = get_phase7_system_prompt(trial["trial_type"])

        response = generate(
            model, processor, dtype, args.device,
            prompt, trial["option_image_paths"],
            max_new_tokens=max_tokens,
            system_prompt=trial_system,
        )

        option_labels = LABELS[:trial["n_options"]]
        if use_phase2:
            predicted_label, reason = parse_answer_phase2(response, option_labels)
        else:
            predicted_label, reason = parse_answer_baseline(response, option_labels)

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
    phase_str = "+".join(str(p) for p in sorted(phases)) if phases else "baseline"
    print(f"\n{'='*70}")
    print(f"  RESULTS — Phases: {phase_str}")
    print(f"{'='*70}\n")

    by_type: dict[str, dict] = {}
    for r in results:
        tt = r["trial_type"]
        rec = by_type.setdefault(tt, {"n": 0, "correct": 0, "parsed": 0})
        rec["n"] += 1
        rec["correct"] += int(r["is_correct"])
        rec["parsed"] += int(r["parsed"])

    total_n = len(results)
    total_correct = sum(r["is_correct"] for r in results)
    total_parsed = sum(r["parsed"] for r in results)

    print(f"{'Trial Type':<35} {'N':>4} {'Acc':>7} {'Parse%':>7}")
    print("-" * 60)

    for tt, rec in sorted(by_type.items(), key=lambda kv: kv[1]["correct"] / max(kv[1]["n"], 1), reverse=True):
        n = rec["n"]
        acc = rec["correct"] / n if n else 0
        parse = rec["parsed"] / n if n else 0
        print(f"{tt:<35} {n:>4} {acc:>7.1%} {parse:>7.1%}")

    print("-" * 60)
    overall_acc = total_correct / total_n if total_n else 0
    overall_parse = total_parsed / total_n if total_n else 0
    print(f"{'OVERALL':<35} {total_n:>4} {overall_acc:>7.1%} {overall_parse:>7.1%}")
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
        description="Phased experiment for Theory-of-Mind (Stories) prompt improvements.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--phase", nargs="+", type=int, default=[0], dest="phases",
        help="Phase(s) to activate: 0=baseline, 1=structured, 2=enhanced-parse, "
        "3=system-prompt, 4=belief-hint, 5=mental-state-cot, 6=answer-last-cot, "
        "7=type-specific-system, 8=perspective-hint.",
    )
    parser.add_argument("--model-id", default="OpenGVLab/InternVL3_5-4B-HF")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--data-root", default=str(PROJECT_ROOT / "data"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "results" / "prompt_optimization/theory-of-mind/internvl-3.5-4b"))
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
        1: "structured-prompt",
        2: "enhanced-parse",
        3: "system-prompt",
        4: "belief-hint",
        5: "mental-state-cot",
        6: "answer-last-cot",
        7: "type-specific-system",
        8: "perspective-hint",
    }
    active = [phase_names.get(p, f"phase{p}") for p in phases]
    print(f"Active phases: {active or ['baseline']}")
    print(f"Model: {args.model_id}")
    print(f"Device: {args.device}")
    print()

    print("Loading manifest...")
    data_root = Path(args.data_root)
    manifest_rows = load_manifest(data_root)
    image_dir = data_root / "assets" / "2026-03-26" / "visual" / "theory-of-mind"
    trials = build_trials(manifest_rows, image_dir)
    print(f"Loaded {len(trials)} trials")

    trial_types = set(t["trial_type"] for t in trials)
    print(f"  {len(trial_types)} trial types: {sorted(trial_types)}")

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
