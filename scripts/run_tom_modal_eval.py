#!/usr/bin/env python3
"""Evaluate ToM with no screenshots, descriptions, or screenshots."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from pathlib import Path

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus-csv", type=Path, required=True)
    p.add_argument("--visual-dir", type=Path, required=True)
    p.add_argument("--output-jsonl", type=Path, required=True)
    p.add_argument("--summary-json", type=Path, required=True)
    p.add_argument(
        "--mode",
        choices=[
            "no_screenshots",
            "screen_descriptions",
            "screenshots",
            "corpus_image_field",
        ],
        required=True,
    )
    p.add_argument("--model-id", default="HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
    p.add_argument("--device", default="cuda", choices=["auto", "cpu", "cuda"])
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--shuffle-options", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-items", type=int, default=None)
    return p.parse_args()


def _split_alts(value: str) -> list[str]:
    return [x.strip() for x in (value or "").split(",") if x.strip()]


def _extract_letter(text: str, n_options: int) -> str | None:
    m = re.search(r"\b([A-Z])\b", (text or "").upper())
    if not m:
        return None
    ch = m.group(1)
    return ch if ch in LETTERS[:n_options] else None


def _find_image(item_value: str, visual_dir: Path) -> Path | None:
    stem = (item_value or "").strip().lower()
    if not stem:
        return None
    for ext in (".webp", ".png", ".jpg", ".jpeg"):
        p = visual_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def _resolve_corpus_image(image_value: str, visual_dir: Path) -> Path | None:
    image_value = (image_value or "").strip()
    if not image_value:
        return None
    direct = Path(image_value)
    if direct.exists():
        return direct
    base = Path(image_value).name
    candidate = visual_dir / base
    if candidate.exists():
        return candidate
    stem = Path(base).stem
    if stem:
        return _find_image(stem, visual_dir)
    return None


def _prompt(question: str, options: list[str], context: list[str]) -> str:
    lines = []
    if context:
        lines.append("Story context:")
        lines.extend([f"- {c}" for c in context])
        lines.append("")
    lines.append(question.strip())
    lines.append("")
    lines.append("Options:")
    for i, opt in enumerate(options):
        lines.append(f"{LETTERS[i]}. {opt}")
    lines.append("")
    lines.append("Answer with a single letter (A, B, C, ...).")
    return "\n".join(lines)


def _load_model(model_id: str, device: str):
    import torch
    import transformers
    from transformers import AutoProcessor

    fp16 = device == "cuda" or (device == "auto" and torch.cuda.is_available())
    kwargs = {"dtype": torch.float16 if fp16 else torch.float32}
    if device == "auto":
        kwargs["device_map"] = "auto"
    model_cls = getattr(transformers, "AutoModelForImageTextToText", None) or getattr(transformers, "AutoModelForVision2Seq")
    model = model_cls.from_pretrained(model_id, **kwargs)
    processor = AutoProcessor.from_pretrained(model_id)
    if device in ("cpu", "cuda"):
        model = model.to(device)
    return model, processor


def _generate(model, processor, prompt_text: str, image_path: Path | None, max_new_tokens: int, explicit_device: str) -> str:
    from PIL import Image

    messages = [{"role": "user", "content": []}]
    images = None
    if image_path is not None:
        messages[0]["content"].append({"type": "image"})
        images = [Image.open(image_path).convert("RGB")]
    messages[0]["content"].append({"type": "text", "text": prompt_text})
    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=text, images=images, return_tensors="pt")
    model_device = model.device if explicit_device == "auto" else explicit_device
    inputs = {k: v.to(model_device) for k, v in inputs.items()}
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    new_ids = out[:, inputs["input_ids"].shape[-1] :]
    return processor.batch_decode(new_ids, skip_special_tokens=True)[0].strip()


def main() -> None:
    args = parse_args()
    import torch

    random.seed(args.seed)
    rng = random.Random(args.seed)

    rows = list(csv.DictReader(args.corpus_csv.open()))
    model, processor = _load_model(args.model_id, args.device)
    records = []

    include_desc = args.mode in ("screen_descriptions", "screenshots", "corpus_image_field")
    include_img = args.mode in ("screenshots", "corpus_image_field")
    context: list[str] = []
    current_block = None
    test_seen = 0
    image_used = 0

    for row in rows:
        stage = (row.get("assessment_stage") or "").strip().lower()
        instr = (row.get("instruction") or "").strip()
        block = (row.get("block_index") or "").strip()
        if block != current_block:
            context = []
            current_block = block
        if stage == "instructions":
            prompt_text = (row.get("prompt") or "").strip()
            if include_desc and instr:
                context.append(instr)
            elif include_desc and prompt_text:
                context.append(prompt_text)
            continue
        if stage != "test_response":
            continue
        if args.max_items is not None and test_seen >= args.max_items:
            break

        answer = (row.get("answer") or "").strip()
        options = [answer] + _split_alts(row.get("response_alternatives", ""))
        options = [o for o in options if o]
        if len(options) < 2:
            continue
        correct = answer
        if args.shuffle_options:
            rng.shuffle(options)
        correct_idx = options.index(correct)
        gold = LETTERS[correct_idx]

        image_path = None
        if include_img:
            if args.mode == "corpus_image_field":
                image_path = _resolve_corpus_image((row.get("image") or "").strip(), args.visual_dir)
            else:
                image_path = _find_image((row.get("item") or "").strip(), args.visual_dir)
            if image_path is not None:
                image_used += 1

        prompt_text = _prompt((row.get("prompt") or "").strip(), options, context)
        raw = _generate(model, processor, prompt_text, image_path, args.max_new_tokens, args.device)
        pred = _extract_letter(raw, len(options))
        ok = pred == gold if pred is not None else False
        records.append(
            {
                "item_uid": row.get("item_uid", ""),
                "scenario": row.get("scenario", ""),
                "item": row.get("item", ""),
                "mode": args.mode,
                "seed": args.seed,
                "gold": gold,
                "pred": pred,
                "correct": int(ok),
                "n_options": len(options),
                "used_image": int(image_path is not None),
                "raw_response": raw,
            }
        )
        test_seen += 1
        if include_desc and instr:
            context.append(instr)
        if len(context) > 20:
            context = context[-20:]

    n = len(records)
    acc = sum(r["correct"] for r in records) / n if n else 0.0
    parse_rate = sum(1 for r in records if r["pred"] is not None) / n if n else 0.0
    chance = sum(1.0 / r["n_options"] for r in records) / n if n else 0.0
    summary = {
        "mode": args.mode,
        "seed": args.seed,
        "n": n,
        "accuracy": acc,
        "parse_rate": parse_rate,
        "chance": chance,
        "lift_vs_chance": acc - chance,
        "images_used": image_used,
        "device": str(model.device) if hasattr(model, "device") else args.device,
        "cuda_available": bool(torch.cuda.is_available()),
    }

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
