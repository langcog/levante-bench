#!/usr/bin/env python3
"""Generate visual-only screenshot descriptions for ToM screenshots.

This script reads the existing screenshot correlation CSV (for item mapping),
runs a VLM caption pass that avoids transcribing prompt text, and writes a
separate CSV so raw visual descriptions can be compared with corpus prompts.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import torch
import transformers
from PIL import Image
from transformers import AutoProcessor

DEFAULT_MODEL_ID = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate visual-only ToM screenshot descriptions.")
    p.add_argument(
        "--input-csv",
        type=Path,
        default=Path("results/screenshots/theory-of-mind/theory-of-mind-screenshot-descriptions.csv"),
        help="Existing correlated screenshot CSV.",
    )
    p.add_argument(
        "--screenshots-dir",
        type=Path,
        default=Path("screenshots/theory-of-mind"),
        help="Directory containing screenshot PNG files.",
    )
    p.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/screenshots/theory-of-mind/theory-of-mind-visual-only-descriptions-2.2b.csv"),
        help="Output CSV for visual-only descriptions.",
    )
    p.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    p.add_argument("--device", default="cuda", choices=["auto", "cpu", "cuda"])
    p.add_argument("--max-new-tokens", type=int, default=140)
    p.add_argument("--max-items", type=int, default=None, help="Optional cap for quick tests.")
    return p.parse_args()


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _load_rows(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _load_model(model_id: str, device: str):
    use_fp16 = device == "cuda" or (device == "auto" and torch.cuda.is_available())
    model_kwargs: dict[str, Any] = {"torch_dtype": torch.float16 if use_fp16 else torch.float32}
    if device == "auto":
        model_kwargs["device_map"] = "auto"
    model_cls = getattr(transformers, "AutoModelForImageTextToText", None)
    if model_cls is None:
        model_cls = getattr(transformers, "AutoModelForVision2Seq")
    model = model_cls.from_pretrained(model_id, **model_kwargs)
    processor = AutoProcessor.from_pretrained(model_id)
    if device in ("cpu", "cuda"):
        model = model.to(device)
    return model, processor


def _extract_json(text: str) -> dict[str, Any]:
    s = (text or "").strip()
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    m = re.search(r"\{.*\}", s, flags=re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    return {
        "visual_scene_description": _normalize(s)[:300],
        "ui_context": "unknown",
        "novel_visual_info": "",
    }


def _caption_visual_only(
    *,
    model: Any,
    processor: Any,
    image_path: Path,
    device: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    img = Image.open(image_path).convert("RGB")
    instruction = (
        "Describe only the visual scene in this screenshot. "
        "Do NOT transcribe or paraphrase any on-screen prompt text. "
        "Focus on characters, objects, positions, actions, and UI layout. "
        "Return strict JSON with keys: "
        "visual_scene_description, ui_context, novel_visual_info. "
        "ui_context must be one of: start_screen, instruction_screen, question_screen, transition_screen, completion_screen, unknown. "
        "visual_scene_description should be one concise sentence. "
        "novel_visual_info should be a short phrase for details likely not present in text prompt (or empty string)."
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": instruction},
            ],
        }
    ]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[img], return_tensors="pt")
    if device in ("cpu", "cuda"):
        inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    prompt_len = inputs["input_ids"].shape[1]
    gen_ids = out_ids[:, prompt_len:]
    text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
    return _extract_json(text)


def run(args: argparse.Namespace) -> None:
    rows = _load_rows(args.input_csv)
    if args.max_items is not None:
        rows = rows[: args.max_items]
    if not rows:
        raise ValueError("No rows found in input CSV.")

    model, processor = _load_model(args.model_id, args.device)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    out_rows: list[dict[str, str]] = []
    for i, row in enumerate(rows, start=1):
        shot = (row.get("screenshot_file") or "").strip()
        if not shot:
            continue
        image_path = args.screenshots_dir / shot
        if not image_path.exists():
            out_rows.append(
                {
                    "screenshot_file": shot,
                    "item_uid": row.get("item_uid", ""),
                    "assessment_stage": row.get("assessment_stage", ""),
                    "corpus_prompt": row.get("corpus_prompt", ""),
                    "visual_scene_description": "",
                    "ui_context": "unknown",
                    "novel_visual_info": "",
                    "error": "missing_image",
                }
            )
            continue
        cap = _caption_visual_only(
            model=model,
            processor=processor,
            image_path=image_path,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
        )
        out_rows.append(
            {
                "screenshot_file": shot,
                "item_uid": row.get("item_uid", ""),
                "assessment_stage": row.get("assessment_stage", ""),
                "corpus_prompt": row.get("corpus_prompt", ""),
                "visual_scene_description": _normalize(str(cap.get("visual_scene_description", ""))),
                "ui_context": _normalize(str(cap.get("ui_context", ""))).lower() or "unknown",
                "novel_visual_info": _normalize(str(cap.get("novel_visual_info", ""))),
                "error": "",
            }
        )
        if i % 10 == 0:
            print(f"Processed {i}/{len(rows)}")

    fieldnames = [
        "screenshot_file",
        "item_uid",
        "assessment_stage",
        "corpus_prompt",
        "visual_scene_description",
        "ui_context",
        "novel_visual_info",
        "error",
    ]
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)

    print(
        json.dumps(
            {
                "input_csv": str(args.input_csv),
                "output_csv": str(args.output_csv),
                "n_rows": len(out_rows),
                "model_id": args.model_id,
            },
            indent=2,
        )
    )


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()

