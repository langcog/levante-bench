#!/usr/bin/env python3
"""Phased experiment: TROG on Qwen3.5-9B.

Why 9B?
  The 2B experiment showed describe-first (Phase 9) only adds +2 pp over
  structural prompting (64.6% vs 62.6%). Literature predicts that CoT /
  describe-first scales with model capacity — meaningful gains emerge at ~7B+.
  Qwen3.5-9B is already cached, fits in 48 GB unified memory (~18 GB bfloat16),
  and shares the same multimodal architecture as the 2B, allowing a clean
  apples-to-apples comparison.

Goal: verify that describe-first gives substantially larger gains at 9B scale
vs the marginal +2 pp seen at 2B.

Phases run
----------
    0  Baseline (manifest prompt as-is)
    1+2+3+4  Best structural config from 2B run
    1+2+3+9  Structural + describe-first  ← main hypothesis
    1+2+3+4+9  Full combo
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

NEGATION_TYPES = {
    "X but not Y",
    "not only X but also Y",
    "neither X nor Y",
    "negative",
}

QWEN_DEFAULT_SYSTEM = (
    "You are a helpful assistant. "
    "Answer with only a single letter: A, B, C, or D. Do not explain."
)

LANGUAGE_EXPERT_SYSTEM = (
    "You are a language comprehension expert. "
    "Your task is to find the image that best matches the given sentence. "
    "Pay careful attention to grammatical roles: who does what to whom, "
    "spatial positions, conditions, and negations. "
    "Answer with exactly one letter: A, B, C, or D."
)

MAX_IMAGE_SIZE = 512


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------

def _resize_image(img, max_size: int = MAX_IMAGE_SIZE):
    w, h = img.size
    if max(w, h) <= max_size:
        return img
    scale = max_size / max(w, h)
    return img.resize((int(w * scale), int(h * scale)))


# ---------------------------------------------------------------------------
# Manifest loading & trial building
# ---------------------------------------------------------------------------

def load_manifest(data_root: Path) -> list[dict]:
    manifest_path = data_root / "assets" / "manifest.csv"
    rows = []
    with open(manifest_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["task"] == "trog" and row["trial_type"] != "practice":
                rows.append(row)
    return rows


def build_trials(manifest_rows: list[dict], image_dir: Path) -> list[dict]:
    index: dict[str, Path] = {}
    if image_dir.exists():
        for p in image_dir.iterdir():
            if p.is_file():
                index[p.stem] = p
                index[p.name] = p

    trials = []
    skipped = 0
    for row in manifest_rows:
        answer = str(row["answer"]).strip()
        alts = [a.strip() for a in str(row.get("response_alternatives", "")).split(",") if a.strip()]
        all_opts = [answer] + alts

        rng = random.Random(row["item_uid"])
        rng.shuffle(all_opts)
        correct_idx = all_opts.index(answer)
        correct_label = LABELS[correct_idx] if correct_idx < len(LABELS) else "?"

        image_paths = []
        missing = False
        for opt in all_opts:
            p = index.get(opt.strip())
            if p is None:
                missing = True
                break
            image_paths.append(str(p))

        prompt_phrase = str(row.get("prompt_phrase", "") or row.get("prompt", "")).strip()

        if missing:
            skipped += 1
            continue

        trials.append({
            "item_uid": row["item_uid"],
            "trial_type": str(row.get("trial_type", "")).strip(),
            "row": row,
            "image_paths": image_paths,
            "correct_label": correct_label,
            "prompt_phrase": prompt_phrase,
            "n_options": len(all_opts),
        })

    if skipped:
        print(f"  Skipped {skipped} trials (missing images)", file=sys.stderr)
    return trials


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def build_prompt_baseline(row: dict, prompt_phrase: str) -> str:
    full_prompt = str(row.get("full_prompt", ""))
    if full_prompt and full_prompt not in {"", "NA", "nan"}:
        return full_prompt.replace("<prompt_phrase>", prompt_phrase)
    prompt_tmpl = str(row.get("prompt", "")).strip()
    return f'{prompt_tmpl} "{prompt_phrase}". Answer with A, B, C, or D. A: <image1>; B: <image2>; C: <image3>; D: <image4>'


def build_prompt_phase1(row: dict, prompt_phrase: str) -> str:
    trial_type = str(row.get("trial_type", "")).strip()

    if trial_type in ("noun", "verb", "adjective", "two element combination"):
        stem = f'Which image shows: "{prompt_phrase}"?'
    elif trial_type in SPATIAL_TYPES:
        stem = f'Which image matches: "{prompt_phrase}"?'
    else:
        stem = f'Which image matches the sentence: "{prompt_phrase}"?'

    return "\n".join([
        stem, "",
        "A: <image1>",
        "B: <image2>",
        "C: <image3>",
        "D: <image4>",
        "",
        "Answer with one letter.",
    ])


def apply_phase3_format_suffix(prompt: str) -> str:
    return prompt.rstrip() + "\n\nRespond with exactly one letter: A, B, C, or D. Nothing else."


def apply_phase4_grounding(prompt: str, trial_type: str) -> str:
    if trial_type in ("reversible passive", "reversible active",
                      "reversible passive; relative clause"):
        hint = "Pay close attention to WHO is performing the action and WHO is receiving it."
    elif trial_type in ("relative clause", "postmodified subject", "post modified subject"):
        hint = "Focus on which object or person the description modifies."
    elif trial_type in ("embedded sentence",):
        hint = "This sentence has a clause inside another clause. Identify the subject of each part."
    elif trial_type in SPATIAL_TYPES:
        hint = "Look carefully at the exact spatial positions of the objects in each image."
    elif trial_type in NEGATION_TYPES:
        hint = "The sentence specifies what IS and what IS NOT shown. Verify both conditions in each image."
    elif trial_type == "conditional":
        hint = "Identify the condition (IF/WHEN part) and its result separately."
    elif trial_type in COMPLEX_GRAMMAR_TYPES:
        hint = "Read the sentence carefully. Identify all subjects, actions, and objects before choosing."
    else:
        return prompt
    return f"{hint}\n\n{prompt}"


def apply_phase6_grammar_cot(prompt: str, trial_type: str, prompt_phrase: str) -> str:
    """Phase 6: grammar role decomposition CoT targeting 0% failure types.

    Asks the model to explicitly extract grammatical roles before comparing
    to images. Different decomposition depending on trial type.
    """
    if trial_type not in COMPLEX_GRAMMAR_TYPES and trial_type not in SPATIAL_TYPES:
        return prompt

    if trial_type in SPATIAL_TYPES:
        cot = (
            "\nBefore answering:\n"
            "1. Object 1: what is it and where is it?\n"
            "2. Object 2: what is it and where is it?\n"
            "3. Which image shows exactly this spatial arrangement?\n"
            "My answer is"
        )
    elif trial_type in NEGATION_TYPES:
        cot = (
            "\nBefore answering:\n"
            "1. What MUST be present in the image?\n"
            "2. What must NOT be present?\n"
            "3. Which image satisfies both conditions?\n"
            "My answer is"
        )
    elif trial_type == "conditional":
        cot = (
            "\nBefore answering:\n"
            "1. What is the condition?\n"
            "2. What is the result if the condition is met?\n"
            "3. Which image shows this situation?\n"
            "My answer is"
        )
    elif trial_type in ("reversible active", "reversible passive",
                        "reversible passive; relative clause"):
        cot = (
            "\nBefore answering:\n"
            "1. Who is the subject (the one doing the action)?\n"
            "2. Who/what is the object (receiving the action)?\n"
            "3. Which image shows the correct direction of action?\n"
            "My answer is"
        )
    elif trial_type in ("relative clause", "postmodified subject",
                        "post modified subject", "embedded sentence"):
        cot = (
            "\nBefore answering:\n"
            "1. What is the main subject?\n"
            "2. What additional description modifies it?\n"
            "3. Which image shows a subject with that exact modifier?\n"
            "My answer is"
        )
    else:
        cot = (
            "\nBefore answering:\n"
            "1. What are the key elements in the sentence?\n"
            "2. What relationship or structure connects them?\n"
            "3. Which image matches all elements?\n"
            "My answer is"
        )

    return prompt.rstrip() + cot


def build_prompt_phase9(prompt_phrase: str, trial_type: str) -> str:
    """Phase 9: describe-first visual grounding.

    Forces the model to generate a one-sentence description of each image
    before matching to the target sentence. This is different from:
    - Phase 4 (prepends a static hint about what to look for)
    - Phase 6 (grammar CoT decomposes the sentence structure)
    Phase 9 grounds reasoning in the visual content by making the model
    commit to what it sees before reading the task sentence.
    """
    lines = [
        "Look at the four images below:",
        "",
        "A: <image1>",
        "B: <image2>",
        "C: <image3>",
        "D: <image4>",
        "",
        "Step 1: Write one sentence describing what is happening in each image.",
        "Step 2: Decide which image matches this sentence:",
        f'"{prompt_phrase}"',
        "",
        "End your response with exactly: My answer: [A/B/C/D]",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Answer parsing
# ---------------------------------------------------------------------------

def parse_answer_baseline(text: str, option_labels: list[str]) -> tuple[Optional[str], str]:
    text = text.strip()
    labels_upper = [la.upper() for la in option_labels]

    try:
        parsed = json.loads(text)
        answer = parsed.get("answer", "").strip().upper()
        if answer in labels_upper:
            return answer, parsed.get("reason", "")
    except (json.JSONDecodeError, AttributeError):
        pass

    m = re.search(r'\{[^}]*"answer"\s*:\s*"([^"]+)"[^}]*\}', text)
    if m:
        answer = m.group(1).strip().upper()
        if answer in labels_upper:
            return answer, ""

    m = re.search(r'(?:the\s+)?(?:correct\s+)?answer\s+is\s+([A-Z])\b', text, re.IGNORECASE)
    if m and m.group(1).upper() in labels_upper:
        return m.group(1).upper(), text

    if text.upper() in labels_upper:
        return text.upper(), ""

    for label in option_labels:
        if text.upper().startswith(label.upper()):
            rest = text[len(label):]
            if not rest or rest[0] in " .),:;\n":
                return label, rest.strip()

    return None, text


def parse_answer_phase2(text: str, option_labels: list[str]) -> tuple[Optional[str], str]:
    label, reason = parse_answer_baseline(text, option_labels)
    if label is not None:
        return label, reason

    labels_upper = [la.upper() for la in option_labels]
    cleaned = text.strip()

    m = re.search(r'(?:my\s+)?answer\s+(?:is\s*)?[:\-]?\s*([A-D])\b', cleaned, re.IGNORECASE)
    if m and m.group(1).upper() in labels_upper:
        return m.group(1).upper(), ""

    m = re.search(r'(?:image|option|picture|choice)\s+([A-D])\b', cleaned, re.IGNORECASE)
    if m and m.group(1).upper() in labels_upper:
        return m.group(1).upper(), cleaned

    m = re.search(r'(?:choose|select|pick)\s+([A-D])\b', cleaned, re.IGNORECASE)
    if m and m.group(1).upper() in labels_upper:
        return m.group(1).upper(), cleaned

    m = re.search(r'\(([A-D])\)', cleaned)
    if m and m.group(1).upper() in labels_upper:
        return m.group(1).upper(), cleaned

    sentences = re.split(r"[.!?\n]", cleaned)
    for sentence in reversed(sentences):
        for label_str in option_labels:
            if re.search(rf"\b{re.escape(label_str)}\b", sentence, re.IGNORECASE):
                return label_str.upper(), sentence.strip()

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
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText

    dtype = torch.bfloat16
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(model_id, padding_side="left")
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, dtype=dtype, attn_implementation="sdpa",
    ).to(device)
    model.eval()
    print(f"Model loaded in {time.time()-t0:.1f}s")
    return model, processor, dtype


def generate(
    model, processor, dtype, device: str,
    prompt_text: str,
    image_paths: list[str],
    max_new_tokens: int = 128,
    system_prompt: str | None = None,
) -> str:
    import torch
    from PIL import Image

    pil_images = [_resize_image(Image.open(p).convert("RGB")) for p in image_paths]

    # Replace <imageN> tags with processor image tokens
    parts = re.split(r"(<image\d+>)", prompt_text)
    content: list[dict] = []
    img_idx = 0
    for part in parts:
        m = re.match(r"<image(\d+)>", part)
        if m:
            tag_num = int(m.group(1))
            actual_idx = tag_num - 1
            if 0 <= actual_idx < len(pil_images):
                content.append({"type": "image", "image": pil_images[actual_idx]})
                img_idx += 1
        elif part.strip():
            content.append({"type": "text", "text": part})

    if not any(c["type"] == "image" for c in content):
        new_content: list[dict] = []
        for img in pil_images:
            new_content.append({"type": "image", "image": img})
        new_content.extend(content)
        content = new_content

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": content})

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = processor(
        text=[text], images=pil_images, return_tensors="pt", padding=True,
    ).to(device)

    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        output_ids = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)

    generated = output_ids[:, input_len:]
    return processor.batch_decode(generated, skip_special_tokens=True)[0].strip()


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

def print_results(results: list[dict], phases: list[int], model_id: str = "") -> None:
    import collections
    phase_str = "+".join(str(p) for p in sorted(phases)) if phases else "baseline"
    model_tag = model_id.split("/")[-1] if model_id else "Qwen3.5-9B"
    print(f"\n{'='*70}")
    print(f"  RESULTS — Phases: {phase_str}  (model: {model_tag})")
    print(f"{'='*70}\n")

    n = len(results)
    correct = sum(1 for r in results if r["is_correct"])
    parsed = sum(1 for r in results if r["parsed"])
    print(f"{'OVERALL':<45} {n:>4}   {correct/n:>6.1%}  {parsed/n:>6.1%}")

    by_type: dict = collections.defaultdict(lambda: {"n": 0, "correct": 0})
    for r in results:
        t = r["trial_type"]
        by_type[t]["n"] += 1
        by_type[t]["correct"] += int(r["is_correct"])

    print()
    for t in sorted(by_type):
        v = by_type[t]
        print(f"  {t:<43} {v['n']:>4}   {v['correct']/v['n']:>6.1%}")
    print()


# ---------------------------------------------------------------------------
# Experiment loop
# ---------------------------------------------------------------------------

def run_experiment(args, trials, model, processor, dtype) -> list[dict]:
    from tqdm import tqdm

    phases = set(args.phases)
    use_phase1 = 1 in phases
    use_phase2 = 2 in phases
    use_phase3 = 3 in phases
    use_phase4 = 4 in phases
    use_phase6 = 6 in phases
    use_phase7 = 7 in phases
    use_phase9 = 9 in phases

    system_prompt = QWEN_DEFAULT_SYSTEM
    if use_phase7:
        system_prompt = LANGUAGE_EXPERT_SYSTEM

    # 9B reasons verbosely by default — always use enhanced parsing and give
    # enough tokens for the model to complete its reasoning chain.
    max_tokens = 512 if use_phase9 else 256
    results = []

    for trial in tqdm(trials, desc="Evaluating", unit="trial"):
        prompt_phrase = trial["prompt_phrase"]
        trial_type = trial["trial_type"]

        if use_phase9:
            prompt = build_prompt_phase9(prompt_phrase, trial_type)
            if use_phase4:
                prompt = apply_phase4_grounding(prompt, trial_type)
        else:
            prompt = (
                build_prompt_phase1(trial["row"], prompt_phrase)
                if use_phase1
                else build_prompt_baseline(trial["row"], prompt_phrase)
            )

            if use_phase4:
                prompt = apply_phase4_grounding(prompt, trial_type)

            if use_phase6:
                prompt = apply_phase6_grammar_cot(prompt, trial_type, prompt_phrase)
            elif use_phase3:
                prompt = apply_phase3_format_suffix(prompt)

        response = generate(
            model, processor, dtype, args.device,
            prompt,
            trial["image_paths"],
            max_new_tokens=max_tokens,
            system_prompt=system_prompt,
        )

        option_labels = LABELS[:trial["n_options"]]
        # Always use enhanced parsing for 9B — the model reasons verbosely
        # and the answer appears at the end of a chain-of-thought.
        predicted_label, reason = parse_answer_phase2(response, option_labels)

        results.append({
            "item_uid": trial["item_uid"],
            "trial_type": trial_type,
            "correct_label": trial["correct_label"],
            "predicted_label": predicted_label or "",
            "is_correct": predicted_label == trial["correct_label"],
            "parsed": predicted_label is not None,
            "raw_response": response[:400],
            "prompt_preview": prompt[:120],
        })

    return results


def save_results(results: list[dict], output_dir: Path, phases: list[int]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = "_".join(str(p) for p in sorted(phases)) if phases else "baseline"
    out_path = output_dir / f"phase_{tag}.csv"
    if not results:
        return out_path
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="TROG phase experiments — Qwen3.5-2B.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--phase", nargs="+", type=int, default=[0], dest="phases",
        help="Phase(s): 0=baseline 1=structured 2=enhanced-parse 3=format-suffix 4=grounding 6=grammar-cot 7=expert-sys 9=describe-first",
    )
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--data-root", default=str(PROJECT_ROOT / "data"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "results" / "prompt_optimization/trog/qwen-3.5-9b"))
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
        1: "structured-prompt", 2: "enhanced-parse", 3: "format-suffix",
        4: "grounding", 6: "grammar-cot", 7: "expert-sys", 9: "describe-first",
    }
    active = [phase_names.get(p, f"phase{p}") for p in phases]
    print(f"Active phases: {active or ['baseline']}")
    print(f"Model: {args.model_id}")
    print(f"Device: {args.device}")
    print(f"Max image size: {MAX_IMAGE_SIZE}px")
    print()

    print("Loading manifest...")
    data_root = Path(args.data_root)
    manifest_rows = load_manifest(data_root)
    image_dir = data_root / "assets" / "2026-03-26" / "visual" / "trog"
    trials = build_trials(manifest_rows, image_dir)
    print(f"Loaded {len(trials)} trials")

    if args.limit:
        trials = trials[:args.limit]
        print(f"Limited to {args.limit} trials")

    print(f"\nLoading model {args.model_id}...")
    model, processor, dtype = load_model(args.model_id, args.device)
    print()

    results = run_experiment(args, trials, model, processor, dtype)
    save_results(results, Path(args.output_dir), phases)
    print_results(results, phases, model_id=args.model_id)


if __name__ == "__main__":
    main()
