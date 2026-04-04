#!/usr/bin/env python3
"""Phased experiment: Matrix Reasoning with InternVL3.5-8B.

New hypotheses vs the 2B run:
  - 8B has a 20x larger visual encoder (InternViT-6B) → better pattern perception
  - Higher resolution (1024px) preserves fine-grained detail lost at 512px
  - Describe-first approach grounds reasoning in explicit visual descriptions
    (literature shows VLMs reason better from text descriptions than raw pixels)

Phases
------
    0  Baseline (manifest prompt as-is, 512px)
    1  Structured prompt (explicit matrix/options layout)
    3  Expert system prompt (RPM framing — same as 2B phase 3)
    4  Rule enumeration hint (same as 2B phase 4)
    5  Describe-first: ask model to describe each row before answering

Resolution flag (independent of phases):
    --max-image-size 512   (default — matches 2B experiment)
    --max-image-size 1024  (higher resolution — new for 8B)
    --max-image-size 0     (no resize)

Combos tested:
    baseline 512, baseline 1024,
    1+3 512, 1+3 1024,
    1+4 1024, 1+3+4 1024,
    5 1024, 5+3 1024
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent

LABELS = ["A", "B", "C", "D"]

DEFAULT_MAX_IMAGE_SIZE = 512

INTERNVL_SYSTEM = "You are a helpful assistant."

EXPERT_SYSTEM = (
    "You are solving Raven's Progressive Matrices. "
    "Each puzzle shows a grid with a missing piece. "
    "Identify the visual rule — changes in shape, size, shading, number, or orientation — "
    "across rows and columns, then select the option that best completes the pattern. "
    "Always respond with exactly one letter: A, B, C, or D."
)

USER_INSTRUCTION = "\nRespond with exactly one letter: A, B, C, or D."


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------

def _resize_image(img, max_size: int = DEFAULT_MAX_IMAGE_SIZE):
    if max_size <= 0:
        return img
    w, h = img.size
    if max(w, h) <= max_size:
        return img
    scale = max_size / max(w, h)
    return img.resize((int(w * scale), int(h * scale)))


def _build_image_index(directory: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    if not directory.exists():
        return index
    for path in directory.iterdir():
        if path.is_file():
            index[path.stem] = path
            index[path.name] = path
    return index


# ---------------------------------------------------------------------------
# Manifest loading & trial building
# ---------------------------------------------------------------------------

def load_manifest(data_root: Path) -> list[dict]:
    manifest_path = data_root / "assets" / "manifest.csv"
    rows = []
    with open(manifest_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["task"] == "matrix-reasoning" and row["trial_type"] != "practice":
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

        import random
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

        prompt_image = str(row.get("prompt_image", "")).strip()
        matrix_image_path = image_index.get(prompt_image)
        if matrix_image_path is None:
            missing = True

        if missing:
            skipped += 1
            continue

        trials.append({
            "item_uid": row["item_uid"],
            "trial_type": str(row.get("trial_type", "")).strip(),
            "row": row,
            "options": all_options,
            "option_image_paths": option_image_paths,
            "matrix_image_path": str(matrix_image_path),
            "correct_label": correct_label,
            "n_options": len(all_options),
        })

    if skipped:
        print(f"  Skipped {skipped} trials with missing images", file=sys.stderr)
    return trials


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def build_prompt_baseline(row: dict) -> str:
    full = str(row.get("full_prompt", "")).strip()
    if full and full not in ("", "NA", "nan"):
        return full
    return str(row.get("prompt", "")).strip()


def build_prompt_phase1(row: dict, n_options: int) -> str:
    option_lines = "\n".join(f"{LABELS[i]}: <image{i+1}>" for i in range(n_options))
    return (
        "<prompt_image>\n\n"
        "Look at the matrix puzzle. It has a missing piece.\n\n"
        "Choose the option that best completes the pattern.\n\n"
        f"{option_lines}\n\n"
        "Answer with a single letter."
    )


def build_prompt_phase5_describe(row: dict, n_options: int) -> str:
    """Phase 5: describe-first approach.

    Ask the model to describe what it sees in each row before answering.
    Grounded in the finding that VLMs reason better from explicit visual
    descriptions than from abstract rule-finding instructions.
    """
    option_lines = "\n".join(f"{LABELS[i]}: <image{i+1}>" for i in range(n_options))
    return (
        "<prompt_image>\n\n"
        "Look at the matrix puzzle carefully.\n\n"
        "Step 1 — Describe Row 1: What objects are shown? Note their shape, count, shading, and size.\n"
        "Step 2 — Describe Row 2: Same details.\n"
        "Step 3 — Describe Row 3: What is present and what is the missing piece?\n"
        "Step 4 — Identify the rule that is consistent across all rows and columns.\n"
        "Step 5 — Choose the option that follows the rule.\n\n"
        f"{option_lines}\n\n"
        "End your response with: My answer is [letter]."
    )


def apply_phase4_rule_hint(prompt: str) -> str:
    hint = (
        "Hint: Examine how the shapes, sizes, shading, or orientation change "
        "across each row and each column. The same rule applies throughout the matrix.\n\n"
    )
    return hint + prompt


# ---------------------------------------------------------------------------
# Content builder
# ---------------------------------------------------------------------------

def _build_content(prompt_text: str, matrix_image, option_images: list) -> list[dict]:
    all_images = [matrix_image] + option_images
    has_prompt_img = "<prompt_image>" in prompt_text
    has_option_tags = bool(re.search(r"<image\d+>", prompt_text))

    content: list[dict] = []

    if has_prompt_img or has_option_tags:
        text = prompt_text.replace("<prompt_image>", "<image0>")
        parts = re.split(r"(<image\d+>)", text)
        for part in parts:
            m = re.match(r"<image(\d+)>", part)
            if m:
                idx = int(m.group(1))
                if idx < len(all_images):
                    content.append({"type": "image", "image": all_images[idx]})
            elif part.strip():
                content.append({"type": "text", "text": part.strip()})
    else:
        content.append({"type": "image", "image": matrix_image})
        content.append({"type": "text", "text": prompt_text.strip()})
        for img in option_images:
            content.append({"type": "image", "image": img})

    return content


# ---------------------------------------------------------------------------
# Model loading & generation
# ---------------------------------------------------------------------------

def load_model(model_id: str, device: str):
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText

    dtype = torch.bfloat16
    t0 = __import__("time").time()
    processor = AutoProcessor.from_pretrained(
        model_id, padding_side="left", trust_remote_code=True,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, dtype=dtype, attn_implementation="sdpa",
        low_cpu_mem_usage=True, trust_remote_code=True,
    ).to(device)
    model.eval()
    print(f"Model loaded in {__import__('time').time()-t0:.1f}s")
    return model, processor, dtype


def generate(
    model, processor, dtype, device: str,
    prompt_text: str,
    matrix_image_path: str,
    option_image_paths: list[str],
    max_new_tokens: int = 128,
    system_prompt: str | None = None,
    append_user_instruction: bool = True,
    max_image_size: int = DEFAULT_MAX_IMAGE_SIZE,
) -> str:
    import torch
    from PIL import Image

    matrix_img = _resize_image(Image.open(matrix_image_path).convert("RGB"), max_image_size)
    option_imgs = [_resize_image(Image.open(p).convert("RGB"), max_image_size) for p in option_image_paths]
    all_pil = [matrix_img] + option_imgs

    content = _build_content(prompt_text, matrix_img, option_imgs)
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
        images=all_pil,
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


# ---------------------------------------------------------------------------
# Answer parsing
# ---------------------------------------------------------------------------

def parse_answer(text: str, option_labels: list[str]) -> tuple[Optional[str], str]:
    text_stripped = text.strip()
    labels_upper = [la.upper() for la in option_labels]

    try:
        parsed = json.loads(text_stripped)
        a = parsed.get("answer", "").strip().upper()
        if a in labels_upper:
            return a, parsed.get("reason", "")
    except (json.JSONDecodeError, AttributeError):
        pass

    m = re.search(r"(?:my\s+)?answer\s+(?:is\s*)?[:\-]?\s*([A-D])\b", text_stripped, re.IGNORECASE)
    if m and m.group(1).upper() in labels_upper:
        return m.group(1).upper(), ""

    m = re.match(r"^\s*([A-D])\b", text_stripped, re.IGNORECASE)
    if m and m.group(1).upper() in labels_upper:
        return m.group(1).upper(), ""

    for tok in reversed(re.split(r"[\s,;.()\[\]]+", text_stripped)):
        if tok.upper() in labels_upper:
            return tok.upper(), ""

    for tok in re.split(r"[\s,;.()\[\]]+", text_stripped):
        if tok.upper() in labels_upper:
            return tok.upper(), ""

    return None, text_stripped[:80]


# ---------------------------------------------------------------------------
# Results printing
# ---------------------------------------------------------------------------

def print_results(results: list[dict], phases: list[int], model_id: str = "", max_image_size: int = 512) -> None:
    phase_str = "+".join(str(p) for p in sorted(phases)) if phases else "baseline"
    model_tag = model_id.split("/")[-1] if model_id else "InternVL3.5-8B"
    res_str = f"{max_image_size}px" if max_image_size > 0 else "full-res"
    print(f"\n{'='*70}")
    print(f"  RESULTS — Phases: {phase_str}  ({model_tag}, {res_str})")
    print(f"{'='*70}\n")

    n = len(results)
    correct = sum(1 for r in results if r["is_correct"])
    parsed = sum(1 for r in results if r["parsed"])
    print(f"{'OVERALL':<40} {n:>4}   {correct/n:>6.1%}  {parsed/n:>6.1%}")
    print()


# ---------------------------------------------------------------------------
# Experiment loop
# ---------------------------------------------------------------------------

def run_experiment(args, trials, model, processor, dtype) -> list[dict]:
    from tqdm import tqdm

    phases = set(args.phases)
    use_phase1 = 1 in phases
    use_phase3 = 3 in phases
    use_phase4 = 4 in phases
    use_phase5 = 5 in phases

    system_prompt = INTERNVL_SYSTEM
    if use_phase3:
        system_prompt = EXPERT_SYSTEM

    # Describe-first needs more tokens; CoT-less baseline needs less
    max_tokens = 512 if use_phase5 else 128

    results = []

    for trial in tqdm(trials, desc="Evaluating", unit="trial"):
        if use_phase5:
            prompt = build_prompt_phase5_describe(trial["row"], trial["n_options"])
        elif use_phase1:
            prompt = build_prompt_phase1(trial["row"], trial["n_options"])
        else:
            prompt = build_prompt_baseline(trial["row"])

        if use_phase4:
            prompt = apply_phase4_rule_hint(prompt)

        response = generate(
            model, processor, dtype, args.device,
            prompt,
            trial["matrix_image_path"],
            trial["option_image_paths"],
            max_new_tokens=max_tokens,
            system_prompt=system_prompt,
            append_user_instruction=not use_phase5,
            max_image_size=args.max_image_size,
        )

        option_labels = LABELS[:trial["n_options"]]
        predicted_label, reason = parse_answer(response, option_labels)

        is_correct = predicted_label == trial["correct_label"]
        results.append({
            "item_uid": trial["item_uid"],
            "trial_type": trial["trial_type"],
            "correct_label": trial["correct_label"],
            "predicted_label": predicted_label or "",
            "is_correct": is_correct,
            "parsed": predicted_label is not None,
            "raw_response": response[:300],
            "prompt_preview": prompt[:120],
            "resolution": args.max_image_size,
        })

    return results


def save_results(results: list[dict], output_dir: Path, phases: list[int], max_image_size: int) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = "_".join(str(p) for p in sorted(phases)) if phases else "baseline"
    res_tag = f"_r{max_image_size}" if max_image_size != DEFAULT_MAX_IMAGE_SIZE else ""
    out_path = output_dir / f"phase_{tag}{res_tag}.csv"
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
        description="Matrix Reasoning phase experiments — InternVL3.5-8B.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--phase", nargs="+", type=int, default=[0], dest="phases",
        help="Phase(s): 0=baseline 1=structured 3=sys-prompt 4=rule-hint 5=describe-first",
    )
    parser.add_argument("--max-image-size", type=int, default=DEFAULT_MAX_IMAGE_SIZE,
                        help="Max image longest side in pixels (0 = no resize). Default 512.")
    parser.add_argument("--model-id", default="OpenGVLab/InternVL3_5-8B-HF")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--data-root", default=str(PROJECT_ROOT / "data"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "results" / "matrix-8b-phases"))
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
        3: "sys-prompt",
        4: "rule-hint",
        5: "describe-first",
    }
    active = [phase_names.get(p, f"phase{p}") for p in phases]
    res_str = f"{args.max_image_size}px" if args.max_image_size > 0 else "full-res"
    print(f"Active phases: {active or ['baseline']}")
    print(f"Model: {args.model_id}")
    print(f"Device: {args.device}")
    print(f"Max image size: {res_str}")
    print()

    print("Loading manifest...")
    data_root = Path(args.data_root)
    manifest_rows = load_manifest(data_root)
    image_dir = data_root / "assets" / "2026-03-26" / "visual" / "matrix-reasoning"
    trials = build_trials(manifest_rows, image_dir)
    print(f"Loaded {len(trials)} trials")

    if args.limit:
        trials = trials[:args.limit]
        print(f"Limited to {args.limit} trials")

    print(f"\nLoading model {args.model_id}...")
    model, processor, dtype = load_model(args.model_id, args.device)
    print()

    results = run_experiment(args, trials, model, processor, dtype)
    save_results(results, Path(args.output_dir), phases, args.max_image_size)
    print_results(results, phases, model_id=args.model_id, max_image_size=args.max_image_size)


if __name__ == "__main__":
    main()
