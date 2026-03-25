#!/usr/bin/env python3
"""
Evaluate SmolVLM2 on vocab by showing a 2x2 image grid:
- 1 correct target image
- 3 distractor images

The correct target is randomly assigned to quadrants per item.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

LETTERS = "ABCD"
SLOT_NAMES = ["top-left", "top-right", "bottom-left", "bottom-right"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run SmolVLM2 vocab 4-image choice evaluation.")
    p.add_argument("--corpus-csv", type=Path, required=True)
    p.add_argument("--visual-dir", type=Path, required=True)
    p.add_argument("--output-jsonl", type=Path, required=True)
    p.add_argument("--summary-json", type=Path, required=True)
    p.add_argument(
        "--composite-dir",
        type=Path,
        default=Path("results/prompts/vocab-composites"),
        help="Directory to save generated 2x2 composite images",
    )
    p.add_argument("--audio-dir", type=Path, default=None, help="Optional audio directory for pronunciation clips")
    p.add_argument("--model-id", default="HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
    p.add_argument("--device", default="cuda", choices=["auto", "cpu", "cuda"])
    p.add_argument("--max-items", type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--include-practice",
        action="store_true",
        help="Include non-test rows where a valid answer/options set exists",
    )
    p.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Cell size in pixels for each quadrant image",
    )
    p.add_argument("--grid-padding", type=int, default=16, help="Padding between grid cells")
    return p.parse_args()


def _load_model(model_id: str, device: str):
    import torch
    import transformers
    from transformers import AutoProcessor

    use_fp16 = device == "cuda" or (device == "auto" and torch.cuda.is_available())
    kwargs: dict[str, Any] = {"dtype": torch.float16 if use_fp16 else torch.float32}
    if device == "auto":
        kwargs["device_map"] = "auto"
    model_cls = getattr(transformers, "AutoModelForImageTextToText", None)
    if model_cls is None:
        model_cls = getattr(transformers, "AutoModelForVision2Seq")
    model = model_cls.from_pretrained(model_id, **kwargs)
    processor = AutoProcessor.from_pretrained(model_id)
    if device in ("cpu", "cuda"):
        model = model.to(device)
    return model, processor


def _split_alts(value: str) -> list[str]:
    if not value:
        return []
    return [x.strip() for x in value.split(",") if x.strip()]


def _normalize_term(term: str) -> set[str]:
    t = term.strip().lower()
    if not t:
        return set()
    compact = re.sub(r"[^a-z0-9]+", "", t)
    snake = re.sub(r"[^a-z0-9]+", "_", t).strip("_")
    dash = re.sub(r"[^a-z0-9]+", "-", t).strip("-")
    return {
        t,
        t.replace(" ", "_"),
        t.replace(" ", ""),
        t.replace("_", ""),
        snake,
        dash,
        compact,
    }


def _build_visual_index(visual_dir: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for p in visual_dir.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in {".webp", ".png", ".jpg", ".jpeg"}:
            continue
        # Index every file by multiple normalized variants of its stem for robust matching.
        for key in _normalize_term(p.stem):
            index.setdefault(key, p)
    return index


def _resolve_image(term: str, visual_index: dict[str, Path]) -> Path | None:
    for candidate in _normalize_term(term):
        if candidate in visual_index:
            return visual_index[candidate]
    return None


def _resolve_audio(audio_file_stem: str, audio_dir: Path | None) -> Path | None:
    if audio_dir is None:
        return None
    stem = (audio_file_stem or "").strip()
    if not stem:
        return None
    for ext in (".wav", ".mp3", ".m4a", ".ogg", ".flac"):
        p = audio_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def _load_cell_image(path: Path, size: int) -> Image.Image:
    im = Image.open(path).convert("RGB")
    return ImageOps.fit(im, (size, size), method=Image.Resampling.LANCZOS)


def _build_grid(
    image_paths_by_slot: list[Path],
    out_path: Path,
    cell_size: int,
    padding: int,
) -> None:
    canvas_size = cell_size * 2 + padding * 3
    canvas = Image.new("RGB", (canvas_size, canvas_size), (255, 255, 255))
    coords = [
        (padding, padding),  # top-left
        (padding * 2 + cell_size, padding),  # top-right
        (padding, padding * 2 + cell_size),  # bottom-left
        (padding * 2 + cell_size, padding * 2 + cell_size),  # bottom-right
    ]
    for p, (x, y) in zip(image_paths_by_slot, coords):
        canvas.paste(_load_cell_image(p, cell_size), (x, y))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def _build_prompt(target: str, audio_path: Path | None) -> str:
    lines = [
        "You are given a 2x2 image grid.",
        "Quadrants are labeled as:",
        "A = top-left",
        "B = top-right",
        "C = bottom-left",
        "D = bottom-right",
        "",
    ]
    if audio_path is not None:
        # Current SmolVLM2 evaluator is image+text only, so audio availability is noted.
        lines.append(
            f'Pronunciation audio for the target exists ("{audio_path.name}") '
            "but is not playable in this evaluation."
        )
    lines.append(f'Target word: "{target}".')
    lines.append("Which quadrant is most likely to contain an image of this target word?")
    lines.append("Answer with one letter only: A, B, C, or D.")
    return "\n".join(lines)


def _extract_letter(text: str) -> str | None:
    m = re.search(r"\b([ABCD])\b", (text or "").upper())
    return m.group(1) if m else None


def _generate_one(model: Any, processor: Any, prompt_text: str, grid_path: Path, device: str, max_new_tokens: int) -> str:
    import torch

    img = Image.open(grid_path).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=text, images=[img], return_tensors="pt")
    target_device = model.device if device == "auto" else device
    inputs = {k: v.to(target_device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    gen_ids = out[:, inputs["input_ids"].shape[-1] :]
    decoded = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
    return decoded.strip()


def run(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    rows = list(csv.DictReader(args.corpus_csv.open()))
    model, processor = _load_model(args.model_id, args.device)
    visual_index = _build_visual_index(args.visual_dir)

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.composite_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    parsed = 0
    correct = 0
    skipped = 0
    chance = 0.0
    quadrant_counts = {s: 0 for s in SLOT_NAMES}
    used_audio_count = 0

    with args.output_jsonl.open("w", encoding="utf-8") as f_out:
        for row in rows:
            stage = (row.get("assessment_stage") or "").strip().lower()
            if not args.include_practice and stage != "test_response":
                continue

            answer = (row.get("answer") or "").strip()
            distractors = _split_alts((row.get("response_alternatives") or "").strip())
            options = [answer] + [d for d in distractors if d and d != answer]
            # Keep unique and require exactly 4 image candidates.
            deduped: list[str] = []
            for o in options:
                if o not in deduped:
                    deduped.append(o)
            options = deduped
            if len(options) != 4:
                skipped += 1
                continue

            image_paths = []
            missing = []
            for term in options:
                p = _resolve_image(term, visual_index)
                if p is None:
                    missing.append(term)
                image_paths.append(p)
            if missing:
                skipped += 1
                continue

            idxs = [0, 1, 2, 3]
            rng.shuffle(idxs)
            image_paths_by_slot = [image_paths[i] for i in idxs]  # slot order A,B,C,D
            term_by_slot = [options[i] for i in idxs]
            correct_slot = term_by_slot.index(answer)
            gold_letter = LETTERS[correct_slot]
            quadrant_counts[SLOT_NAMES[correct_slot]] += 1

            item_uid = (row.get("item_uid") or f"row_{total+1:04d}").strip()
            grid_path = args.composite_dir / f"{item_uid}.png"
            _build_grid(
                image_paths_by_slot=[p for p in image_paths_by_slot if p is not None],
                out_path=grid_path,
                cell_size=args.image_size,
                padding=args.grid_padding,
            )

            audio_path = _resolve_audio((row.get("audio_file") or "").strip(), args.audio_dir)
            if audio_path is not None:
                used_audio_count += 1

            prompt_text = _build_prompt(target=answer, audio_path=audio_path)
            pred_text = _generate_one(
                model=model,
                processor=processor,
                prompt_text=prompt_text,
                grid_path=grid_path,
                device=args.device,
                max_new_tokens=args.max_new_tokens,
            )
            pred_letter = _extract_letter(pred_text)
            is_correct = pred_letter == gold_letter if pred_letter is not None else False

            total += 1
            chance += 0.25
            if pred_letter is not None:
                parsed += 1
            if is_correct:
                correct += 1

            out = {
                "item_uid": item_uid,
                "audio_file": (row.get("audio_file") or "").strip(),
                "target_term": answer,
                "distractors": distractors,
                "slot_terms": {
                    "A_top_left": term_by_slot[0],
                    "B_top_right": term_by_slot[1],
                    "C_bottom_left": term_by_slot[2],
                    "D_bottom_right": term_by_slot[3],
                },
                "gold_letter": gold_letter,
                "pred_text": pred_text,
                "pred_letter": pred_letter,
                "correct": is_correct,
                "composite_image": str(grid_path),
                "audio_path": str(audio_path) if audio_path is not None else None,
                "prompt_text": prompt_text,
            }
            f_out.write(json.dumps(out, ensure_ascii=True) + "\n")

            if args.max_items is not None and total >= args.max_items:
                break

    summary = {
        "model_id": args.model_id,
        "corpus_csv": str(args.corpus_csv),
        "visual_dir": str(args.visual_dir),
        "audio_dir": str(args.audio_dir) if args.audio_dir is not None else None,
        "output_jsonl": str(args.output_jsonl),
        "n_total": total,
        "n_skipped": skipped,
        "n_parsed": parsed,
        "accuracy_all": (correct / total) if total else None,
        "accuracy_parsed_only": (correct / parsed) if parsed else None,
        "parse_rate": (parsed / total) if total else None,
        "chance_rate": (chance / total) if total else None,
        "lift_vs_chance": ((correct / total) - (chance / total)) if total else None,
        "quadrant_counts": quadrant_counts,
        "audio_available_count": used_audio_count,
        "seed": args.seed,
    }
    with args.summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
