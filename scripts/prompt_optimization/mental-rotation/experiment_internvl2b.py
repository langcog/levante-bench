#!/usr/bin/env python3
"""Phased experiment: test prompt/parsing improvements for Mental Rotation on InternVL3.5-2B.

InternVL3.5 has a stronger InternViT vision encoder that captures fine-grained spatial
features better than Qwen3.5-0.8B — the hypothesis is that its higher baseline accuracy
means prompt optimizations (mirror hint, feature CoT) will show measurable deltas.

Images are resized to MAX_IMAGE_SIZE (512px) before inference to keep speed high.

Phases:
    0  Baseline (current manifest prompts, default system prompt)
    1  Structured prompt (reference image + options clearly separated)
    2  Enhanced parsing (reverse-scan, "image X" patterns, last-letter fallback)
    3  Task-specific system prompt (spatial reasoning)
    4  Mirror awareness hint (chirality / mirror-flip cue)
    5  Feature-based CoT (identify features → check orientation → choose)

Example:
    python scripts/experiment_mrot_internvl_phases.py --phase 0
    python scripts/experiment_mrot_internvl_phases.py --phase 1 2 3
    python scripts/experiment_mrot_internvl_phases.py --phase 1 2 3 4 5
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

LABELS = ["A", "B"]
MAX_IMAGE_SIZE = 512

DEFAULT_SYSTEM = (
    "You are a helpful assistant. "
    "Answer with only a single letter: A or B. Do not explain."
)
# InternVL tends to ignore system-only instructions when images are present,
# so we also append a short instruction to the user turn.
USER_INSTRUCTION = "Reply with exactly one letter: A or B. Nothing else."


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _resize_image(img, max_size: int = MAX_IMAGE_SIZE):
    """Resize PIL image so the longest side is ≤ max_size, preserving aspect ratio."""
    w, h = img.size
    if max(w, h) <= max_size:
        return img
    scale = max_size / max(w, h)
    return img.resize((int(w * scale), int(h * scale)))


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

def build_prompt_baseline(row: dict, prompt_phrase: str) -> str:
    """Phase 0: current manifest prompt with <image0> for context."""
    full_prompt = str(row.get("full_prompt", ""))
    if full_prompt and full_prompt not in {"", "NA", "nan"}:
        prompt = full_prompt.replace("<prompt_phrase>", prompt_phrase)
    else:
        prompt = str(row.get("prompt", "")).strip()
    if "<prompt_image>" in prompt:
        prompt = prompt.replace("<prompt_image>", "<image0>")
    return prompt


def build_prompt_phase1(row: dict, prompt_phrase: str) -> str:
    """Phase 1: structured prompt with clear reference / options separation."""
    lines = [
        "Reference image:",
        "<image0>",
        "",
        "Which option shows the same shape as the reference, just rotated?",
        "",
        "A: <image1>",
        "B: <image2>",
        "",
        "Answer with one letter: A or B.",
    ]
    return "\n".join(lines)


def apply_phase3_suffix(prompt: str) -> str:
    """Phase 3: append strict answer format suffix."""
    return prompt.rstrip() + "\n\nRespond with exactly one letter: A or B. Nothing else."


def apply_phase4_mirror_hint(prompt: str) -> str:
    """Phase 4: mirror awareness hint (chirality cue)."""
    hint = (
        "One option is the same shape rotated to a different angle. "
        "The other option is its mirror image (horizontally flipped). "
        "Look carefully at the orientation and asymmetric features to "
        "distinguish between rotation and reflection."
    )
    return f"{hint}\n\n{prompt}"


def apply_phase5_cot(prompt: str, trial_type: str) -> str:
    """Phase 5: feature-based spatial reasoning CoT."""
    if trial_type == "2":
        cot = (
            "Step 1: Look at the reference silhouette — note which direction "
            "key features point (e.g., ears, beak, tail).\n"
            "Step 2: For each option, check if those features point the same "
            "way (just rotated) or are flipped.\n"
            "Step 3: Choose the option that matches the reference's chirality.\n"
            "State your final answer as a single letter."
        )
    else:
        cot = (
            "Step 1: Identify a distinctive part of the 3D shape in the reference "
            "(e.g., a protruding arm or corner).\n"
            "Step 2: For each option, check if that distinctive part is in the "
            "same relative position (rotated) or mirrored.\n"
            "Step 3: Choose the option that is a rotation, not a mirror.\n"
            "State your final answer as a single letter."
        )
    return prompt.rstrip() + f"\n\n{cot}"


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
        r = re.search(r'"reason"\s*:\s*"([^"]*)"', text)
        if answer in labels_upper:
            return answer, (r.group(1) if r else "")

    m = re.search(r'(?:the\s+)?(?:correct\s+)?answer\s+is\s+([A-B])\b', text, re.IGNORECASE)
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


def parse_answer_phase2(text: str, option_labels: list[str]) -> tuple[Optional[str], str]:
    """Phase 2: enhanced parsing with multiple fallback strategies."""
    label, reason = parse_answer_baseline(text, option_labels)
    if label is not None:
        return label, reason

    labels_upper = [la.upper() for la in option_labels]
    cleaned = text.strip()

    m = re.search(r'(?:image|option|picture|choice)\s+([A-B])\b', cleaned, re.IGNORECASE)
    if m and m.group(1).upper() in labels_upper:
        return m.group(1).upper(), cleaned

    m = re.search(r'(?:choose|select|pick)\s+([A-B])\b', cleaned, re.IGNORECASE)
    if m and m.group(1).upper() in labels_upper:
        return m.group(1).upper(), cleaned

    m = re.search(r'\(([A-B])\)', cleaned)
    if m and m.group(1).upper() in labels_upper:
        return m.group(1).upper(), cleaned

    for sentence in reversed(re.split(r"[.!?\n]", cleaned)):
        for label_str in option_labels:
            if re.search(rf"\b{re.escape(label_str)}\b", sentence.strip(), re.IGNORECASE):
                return label_str.upper(), sentence.strip()

    letters = re.findall(r'\b([A-B])\b', cleaned)
    if letters and letters[-1].upper() in labels_upper:
        return letters[-1].upper(), cleaned

    return None, text


# ---------------------------------------------------------------------------
# Model loading & generation
# ---------------------------------------------------------------------------

def load_model(model_id: str, device: str):
    """Load InternVL3.5 model and processor with trust_remote_code."""
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText

    dtype = torch.bfloat16
    processor = AutoProcessor.from_pretrained(
        model_id, padding_side="left", trust_remote_code=True,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        dtype=dtype,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)
    model.eval()
    return model, processor, dtype


def generate(
    model, processor, dtype, device: str,
    prompt_text: str, image_paths: list[str],
    max_new_tokens: int = 128,
    system_prompt: str | None = None,
    append_user_instruction: bool = True,
) -> str:
    """Generate from InternVL3.5 with resized PIL images."""
    import torch
    from PIL import Image

    pil_images = [_resize_image(Image.open(p).convert("RGB")) for p in image_paths]
    content = _build_content(prompt_text, pil_images)

    # Append per-turn instruction (InternVL ignores system-only instructions)
    if append_user_instruction:
        content.append({"type": "text", "text": USER_INSTRUCTION})

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": content})

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
            **inputs, do_sample=False, max_new_tokens=max_new_tokens,
        )

    generated_ids = output_ids[:, input_len:]
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()


def _build_content(prompt_text: str, pil_images: list) -> list[dict]:
    """Build content list interleaving images at <imageN> placeholders.

    <image0> = context/reference image
    <image1>, <image2> = option images
    """
    content: list[dict] = []
    if pil_images and re.search(r"<image\d+>", prompt_text):
        for part in re.split(r"(<image\d+>)", prompt_text):
            m = re.match(r"<image(\d+)>", part)
            if m:
                idx = int(m.group(1))
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

def load_manifest(data_root: Path) -> list[dict]:
    manifest_path = data_root / "assets" / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    rows = []
    with open(manifest_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["task"] == "mental-rotation":
                rows.append(row)
    return rows


def build_trials(manifest_rows: list[dict], image_dir: Path) -> list[dict]:
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

        option_image_paths, missing = [], False
        for option in all_options:
            path = image_index.get(option.strip())
            if path is None:
                missing = True
                break
            option_image_paths.append(str(path))

        prompt_image_stem = str(row.get("prompt_image", "")).strip()
        context_image_paths = []
        if prompt_image_stem and prompt_image_stem not in {"NA", "nan", "TODO", ""}:
            path = image_index.get(prompt_image_stem)
            if path is None:
                missing = True
            else:
                context_image_paths.append(str(path))

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
            "context_image_paths": context_image_paths,
            "correct_label": correct_label,
            "prompt_phrase": prompt_phrase,
            "n_options": len(all_options),
        })

    if skipped:
        print(f"  Skipped {skipped} trials with missing images", file=sys.stderr)
    return trials


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

def run_experiment(args, trials, model, processor, dtype) -> list[dict]:
    from tqdm import tqdm

    phases = set(args.phases)
    use_phase1 = 1 in phases
    use_phase2 = 2 in phases
    use_phase3 = 3 in phases
    use_phase4 = 4 in phases
    use_phase5 = 5 in phases

    system_prompt = DEFAULT_SYSTEM
    if use_phase3:
        system_prompt = (
            "You are a spatial reasoning assistant. You compare shapes to determine "
            "which option matches a rotated reference image. "
            "Always respond with exactly one letter: A or B."
        )

    max_tokens = 256 if use_phase5 else 64
    results = []

    for trial in tqdm(trials, desc="Evaluating", unit="trial"):
        if use_phase1:
            prompt = build_prompt_phase1(trial["row"], trial["prompt_phrase"])
        else:
            prompt = build_prompt_baseline(trial["row"], trial["prompt_phrase"])

        if use_phase4:
            prompt = apply_phase4_mirror_hint(prompt)
        if use_phase5:
            prompt = apply_phase5_cot(prompt, trial["trial_type"])
        if use_phase3:
            prompt = apply_phase3_suffix(prompt)

        image_paths = trial["context_image_paths"] + trial["option_image_paths"]

        response = generate(
            model, processor, dtype, args.device,
            prompt, image_paths,
            max_new_tokens=max_tokens,
            system_prompt=system_prompt,
            append_user_instruction=True,
        )

        option_labels = LABELS[:trial["n_options"]]
        if use_phase2:
            predicted_label, reason = parse_answer_phase2(response, option_labels)
        else:
            predicted_label, reason = parse_answer_baseline(response, option_labels)

        results.append({
            "item_uid": trial["item_uid"],
            "trial_type": trial["trial_type"],
            "correct_label": trial["correct_label"],
            "predicted_label": predicted_label,
            "is_correct": predicted_label == trial["correct_label"],
            "parsed": predicted_label is not None,
            "raw_response": response,
            "prompt_preview": prompt[:200],
        })

    return results


def print_results(results: list[dict], phases: list[int], model_id: str = "") -> None:
    phase_str = "+".join(str(p) for p in sorted(phases)) if phases else "baseline"
    model_tag = model_id.split("/")[-1] if model_id else "InternVL3.5"
    print(f"\n{'='*70}")
    print(f"  RESULTS — Phases: {phase_str}  (model: {model_tag})")
    print(f"{'='*70}\n")

    by_type: dict[str, dict] = {}
    for r in results:
        rec = by_type.setdefault(r["trial_type"], {"n": 0, "correct": 0, "parsed": 0})
        rec["n"] += 1
        rec["correct"] += int(r["is_correct"])
        rec["parsed"] += int(r["parsed"])

    type_labels = {"2": "2D silhouettes", "3": "3D shapes"}
    print(f"{'Trial Type':<25} {'N':>4} {'Acc':>7} {'Parse%':>7}")
    print("-" * 50)
    for tt, rec in sorted(by_type.items(), key=lambda kv: kv[1]["correct"] / max(kv[1]["n"], 1), reverse=True):
        n = rec["n"]
        print(f"{type_labels.get(tt, tt):<25} {n:>4} {rec['correct']/n:>7.1%} {rec['parsed']/n:>7.1%}")

    total_n = len(results)
    total_correct = sum(r["is_correct"] for r in results)
    total_parsed = sum(r["parsed"] for r in results)
    print("-" * 50)
    print(f"{'OVERALL':<25} {total_n:>4} {total_correct/total_n:>7.1%} {total_parsed/total_n:>7.1%}")
    print()

    unparsed = [r for r in results if not r["parsed"]]
    if unparsed:
        print(f"Unparsed ({len(unparsed)}). First 3:")
        for r in unparsed[:3]:
            print(f"  [{r['trial_type']}] {r['item_uid']}: {r['raw_response'][:100]}")
        print()


def save_results(results: list[dict], output_path: Path) -> None:
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
        description="Mental Rotation phase experiments — InternVL3.5-2B.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--phase", nargs="+", type=int, default=[0], dest="phases",
        help="Phase(s): 0=baseline 1=structured 2=enhanced-parse 3=sys-prompt 4=mirror-hint 5=cot",
    )
    parser.add_argument("--model-id", default="OpenGVLab/InternVL3_5-2B-HF")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--data-root", default=str(PROJECT_ROOT / "data"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "results" / "prompt_optimization/mental-rotation/internvl-3.5-2b"))
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

    phase_names = {1: "structured", 2: "enhanced-parse", 3: "sys-prompt", 4: "mirror-hint", 5: "feature-cot"}
    print(f"Active phases: {[phase_names.get(p, f'phase{p}') for p in phases] or ['baseline']}")
    print(f"Model: {args.model_id}")
    print(f"Device: {args.device}")
    print(f"Max image size: {MAX_IMAGE_SIZE}px")
    print()

    data_root = Path(args.data_root)
    print("Loading manifest...")
    manifest_rows = load_manifest(data_root)
    image_dir = data_root / "assets" / "2026-03-26" / "visual" / "mental-rotation"
    trials = build_trials(manifest_rows, image_dir)
    print(f"Loaded {len(trials)} trials — types: {sorted(set(t['trial_type'] for t in trials))}")

    if args.limit:
        trials = trials[:args.limit]
        print(f"Limited to {len(trials)} trials")

    print(f"\nLoading model {args.model_id}...")
    t0 = time.time()
    model, processor_obj, dtype = load_model(args.model_id, args.device)
    print(f"Model loaded in {time.time() - t0:.1f}s\n")

    args.phases = phases
    results = run_experiment(args, trials, model, processor_obj, dtype)
    print_results(results, phases, model_id=args.model_id)

    phase_tag = "_".join(str(p) for p in phases) if phases else "baseline"
    save_results(results, Path(args.output_dir) / f"phase_{phase_tag}.csv")


if __name__ == "__main__":
    main()
