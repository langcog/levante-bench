#!/usr/bin/env python3
"""Build tracked vocab 2x2 quadrant graphics (answer + 3 distractors)."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from pathlib import Path

from PIL import Image, ImageOps

SLOT_NAMES = ["top-left", "top-right", "bottom-left", "bottom-right"]
LETTERS = "ABCD"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build vocab quadrant graphics and manifest.")
    p.add_argument("--corpus-csv", type=Path, required=True)
    p.add_argument("--visual-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--manifest-csv", type=Path, required=True)
    p.add_argument("--summary-json", type=Path, required=True)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--max-items", type=int, default=None)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--grid-padding", type=int, default=16)
    return p.parse_args()


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
    return {t, t.replace(" ", "_"), t.replace(" ", ""), t.replace("_", ""), snake, dash, compact}


def _build_visual_index(visual_dir: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for p in visual_dir.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in {".webp", ".png", ".jpg", ".jpeg"}:
            continue
        for key in _normalize_term(p.stem):
            index.setdefault(key, p)
    return index


def _resolve_image(term: str, visual_index: dict[str, Path]) -> Path | None:
    for key in _normalize_term(term):
        if key in visual_index:
            return visual_index[key]
    return None


def _load_cell(path: Path, size: int) -> Image.Image:
    return ImageOps.fit(Image.open(path).convert("RGB"), (size, size), method=Image.Resampling.LANCZOS)


def _save_grid(image_paths_by_slot: list[Path], out_path: Path, image_size: int, padding: int) -> None:
    canvas_size = image_size * 2 + padding * 3
    canvas = Image.new("RGB", (canvas_size, canvas_size), (255, 255, 255))
    coords = [
        (padding, padding),
        (padding * 2 + image_size, padding),
        (padding, padding * 2 + image_size),
        (padding * 2 + image_size, padding * 2 + image_size),
    ]
    for p, (x, y) in zip(image_paths_by_slot, coords):
        canvas.paste(_load_cell(p, image_size), (x, y))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    rows = list(csv.DictReader(args.corpus_csv.open()))
    visual_index = _build_visual_index(args.visual_dir)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.manifest_csv.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)

    quadrant_counts = {s: 0 for s in SLOT_NAMES}
    kept = 0
    skipped = 0
    records: list[dict[str, object]] = []

    for row in rows:
        stage = (row.get("assessment_stage") or "").strip().lower()
        if stage != "test_response":
            continue
        answer = (row.get("answer") or "").strip()
        options = [answer] + _split_alts(row.get("response_alternatives", ""))
        deduped: list[str] = []
        for o in options:
            if o and o not in deduped:
                deduped.append(o)
        if len(deduped) != 4:
            skipped += 1
            continue

        source_images = []
        missing = []
        for term in deduped:
            p = _resolve_image(term, visual_index)
            if p is None:
                missing.append(term)
            source_images.append(p)
        if missing:
            skipped += 1
            continue

        order = [0, 1, 2, 3]
        rng.shuffle(order)
        slot_terms = [deduped[i] for i in order]
        slot_images = [source_images[i] for i in order]
        correct_idx = slot_terms.index(answer)
        quadrant_counts[SLOT_NAMES[correct_idx]] += 1

        item_uid = (row.get("item_uid") or f"row_{kept+1:04d}").strip()
        out_path = args.out_dir / f"{item_uid}.png"
        _save_grid([p for p in slot_images if p is not None], out_path, args.image_size, args.grid_padding)

        records.append(
            {
                "item_uid": item_uid,
                "answer": answer,
                "distractors": "|".join(deduped[1:]),
                "gold_letter": LETTERS[correct_idx],
                "gold_quadrant": SLOT_NAMES[correct_idx],
                "A_top_left_term": slot_terms[0],
                "B_top_right_term": slot_terms[1],
                "C_bottom_left_term": slot_terms[2],
                "D_bottom_right_term": slot_terms[3],
                "A_top_left_src": str(slot_images[0]),
                "B_top_right_src": str(slot_images[1]),
                "C_bottom_left_src": str(slot_images[2]),
                "D_bottom_right_src": str(slot_images[3]),
                "composite_path": str(out_path),
            }
        )
        kept += 1
        if args.max_items is not None and kept >= args.max_items:
            break

    with args.manifest_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()) if records else [])
        if records:
            writer.writeheader()
            writer.writerows(records)

    summary = {
        "corpus_csv": str(args.corpus_csv),
        "visual_dir": str(args.visual_dir),
        "out_dir": str(args.out_dir),
        "manifest_csv": str(args.manifest_csv),
        "n_total_graphics": kept,
        "n_skipped": skipped,
        "quadrant_counts": quadrant_counts,
        "seed": args.seed,
    }
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
