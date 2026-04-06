#!/usr/bin/env python3
"""Correlate ToM screenshots to corpus rows and generate per-item caption CSVs."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import requests
import torch
import transformers
from PIL import Image
from transformers import AutoProcessor

DEFAULT_MODEL_ID = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
DEFAULT_CORPUS_URL = "https://storage.googleapis.com/levante-assets-prod/corpus/theory-of-mind/theory-of-mind-item-bank.csv"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Correlate theory-of-mind screenshots to prompts and items.")
    p.add_argument(
        "--screenshots-dir",
        type=Path,
        default=Path("screenshots/theory-of-mind"),
        help="Directory containing screenshot PNGs.",
    )
    p.add_argument(
        "--corpus-csv",
        type=Path,
        default=Path("data/assets/2026-03-24/corpus/theory-of-mind/theory-of-mind-item-bank.csv"),
        help="Path to theory-of-mind-item-bank.csv.",
    )
    p.add_argument(
        "--allow-download-corpus",
        action="store_true",
        help="Download corpus CSV from public bucket if --corpus-csv is missing.",
    )
    p.add_argument(
        "--corpus-download-url",
        default=DEFAULT_CORPUS_URL,
        help="Corpus CSV URL used when --allow-download-corpus is enabled.",
    )
    p.add_argument(
        "--sequence-trace-csv",
        type=Path,
        default=Path("results/benchmark/v1/20260324-212526/tom/tom-v1-none-s1-sequence-trace.csv"),
        help="Optional sequence trace CSV for row-order alignment.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/screenshots/theory-of-mind"),
        help="Output folder for generated CSV files.",
    )
    p.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--max-new-tokens", type=int, default=180)
    return p.parse_args()


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _download_if_needed(corpus_csv: Path, url: str, allow_download: bool) -> Path:
    if corpus_csv.exists():
        return corpus_csv
    if not allow_download:
        raise FileNotFoundError(
            f"Corpus CSV not found at {corpus_csv}. Pass --allow-download-corpus to auto-download."
        )
    corpus_csv.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    corpus_csv.write_bytes(r.content)
    return corpus_csv


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
    text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, flags=re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    return {"extracted_prompt_text": "", "image_description": text[:300], "stage_guess": "unknown"}


def _caption_image(
    model: Any,
    processor: Any,
    image_path: Path,
    device: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    img = Image.open(image_path).convert("RGB")
    instruction = (
        "Analyze this screenshot from a children's story task. "
        "Return strict JSON with keys: extracted_prompt_text, image_description, stage_guess. "
        "Rules: extracted_prompt_text should copy visible on-screen prompt text exactly when readable; "
        "if no readable prompt text, use empty string. "
        "image_description should be 1 concise sentence describing the scene/UI. "
        "stage_guess should be one of initial_load, task_started, instructions, test_response, task_completed, unknown."
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


def _parse_screenshot_name(path: Path) -> tuple[int | None, str]:
    name = path.name
    m = re.search(r"theory-of-mind-(\d+)-", name)
    idx = int(m.group(1)) if m else None
    step = name.rsplit("-", 1)[-1].replace(".png", "")
    return idx, step


def _read_trace(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    return _load_csv_rows(path)


def _direct_row_match(
    screenshot_index: int | None,
    corpus_rows: list[dict[str, str]],
    trace_rows: list[dict[str, str]],
) -> tuple[dict[str, str] | None, str]:
    if screenshot_index is None:
        return None, "none"
    # Screenshots include initial-load and task-started before corpus row 0.
    row_idx = screenshot_index - 3
    if row_idx < 0 or row_idx >= len(corpus_rows):
        return None, "none"
    row = corpus_rows[row_idx]
    if trace_rows and row_idx < len(trace_rows):
        tr = trace_rows[row_idx]
        tr_stage = (tr.get("assessment_stage") or "").strip().lower()
        cr_stage = (row.get("assessment_stage") or "").strip().lower()
        if tr_stage and cr_stage and tr_stage != cr_stage:
            return None, "none"
        tr_uid = (tr.get("item_uid") or "").strip()
        cr_uid = (row.get("item_uid") or "").strip()
        if tr_uid and cr_uid and tr_uid != cr_uid:
            return None, "none"
    return row, "index"


def _prompt_match(
    extracted_prompt_text: str,
    corpus_rows: list[dict[str, str]],
    used_row_idxs: set[int],
) -> tuple[dict[str, str] | None, int | None]:
    norm = _normalize(extracted_prompt_text)
    if not norm:
        return None, None
    for i, row in enumerate(corpus_rows):
        if i in used_row_idxs:
            continue
        p = _normalize(row.get("prompt", ""))
        if p and p == norm:
            return row, i
    for i, row in enumerate(corpus_rows):
        if i in used_row_idxs:
            continue
        p = _normalize(row.get("prompt", ""))
        if p and (norm in p or p in norm):
            return row, i
    return None, None


def _row_index_lookup(corpus_rows: list[dict[str, str]]) -> dict[int, dict[str, str]]:
    return {i: r for i, r in enumerate(corpus_rows)}


def run(args: argparse.Namespace) -> None:
    corpus_path = _download_if_needed(args.corpus_csv, args.corpus_download_url, args.allow_download_corpus)
    corpus_rows = _load_csv_rows(corpus_path)
    trace_rows = _read_trace(args.sequence_trace_csv)
    row_lookup = _row_index_lookup(corpus_rows)

    model, processor = _load_model(args.model_id, args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    by_item_dir = args.output_dir / "by_item"
    by_item_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(args.screenshots_dir.glob("*.png"))
    if not images:
        raise ValueError(f"No PNG files found under {args.screenshots_dir}")

    records: list[dict[str, Any]] = []
    used_row_idxs: set[int] = set()

    for image_path in images:
        screenshot_index, step_label = _parse_screenshot_name(image_path)
        cap = _caption_image(
            model=model,
            processor=processor,
            image_path=image_path,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
        )
        extracted_prompt = (cap.get("extracted_prompt_text") or "").strip()
        description = (cap.get("image_description") or "").strip()
        stage_guess = (cap.get("stage_guess") or "").strip()

        matched_row: dict[str, str] | None = None
        match_method = "none"
        matched_idx: int | None = None

        direct_row, method = _direct_row_match(screenshot_index, corpus_rows, trace_rows)
        if direct_row is not None and method == "index":
            match_method = "index"
            matched_row = direct_row
            for i, r in row_lookup.items():
                if r is direct_row:
                    matched_idx = i
                    used_row_idxs.add(i)
                    break

        if matched_row is None:
            pm_row, pm_idx = _prompt_match(extracted_prompt, corpus_rows, used_row_idxs)
            if pm_row is not None:
                matched_row = pm_row
                matched_idx = pm_idx
                if pm_idx is not None:
                    used_row_idxs.add(pm_idx)
                match_method = "prompt"

        row = matched_row or {}
        records.append(
            {
                "screenshot_file": image_path.name,
                "screenshot_index": screenshot_index if screenshot_index is not None else "",
                "step_label": step_label,
                "extracted_prompt_text": extracted_prompt,
                "image_description": description,
                "stage_guess": stage_guess,
                "matched": bool(matched_row),
                "match_method": match_method,
                "corpus_row_index": matched_idx if matched_idx is not None else "",
                "block_index": (row.get("block_index") or "").strip(),
                "assessment_stage": (row.get("assessment_stage") or "").strip(),
                "trial_type": (row.get("trial_type") or "").strip(),
                "item_uid": (row.get("item_uid") or "").strip(),
                "corpus_prompt": (row.get("prompt") or "").strip(),
            }
        )

    master_csv = args.output_dir / "theory-of-mind-screenshot-descriptions.csv"
    fieldnames = [
        "screenshot_file",
        "screenshot_index",
        "step_label",
        "extracted_prompt_text",
        "image_description",
        "stage_guess",
        "matched",
        "match_method",
        "corpus_row_index",
        "block_index",
        "assessment_stage",
        "trial_type",
        "item_uid",
        "corpus_prompt",
    ]
    with open(master_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(records)

    # Per-item CSVs: only rows with non-empty item_uid.
    by_item: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        uid = (r.get("item_uid") or "").strip()
        if not uid:
            continue
        by_item.setdefault(uid, []).append(r)
    for uid, rows in by_item.items():
        out_csv = by_item_dir / f"{uid}.csv"
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)

    summary = {
        "screenshots_dir": str(args.screenshots_dir),
        "corpus_csv": str(corpus_path),
        "sequence_trace_csv": str(args.sequence_trace_csv) if args.sequence_trace_csv.exists() else None,
        "n_screenshots": len(images),
        "n_matched": sum(1 for r in records if r["matched"]),
        "n_item_csvs": len(by_item),
        "master_csv": str(master_csv),
        "by_item_dir": str(by_item_dir),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()

