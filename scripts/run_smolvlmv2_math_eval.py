#!/usr/bin/env python3
"""
Run SmolVLMv2 on prompt JSONL and compute multiple-choice accuracy.

Input JSONL is expected to come from scripts/build_math_prompts.py and include:
- item_uid
- options
- gold_letter (or gold_index)
- messages or prompt_text
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate SmolVLMv2 on math prompt JSONL.")
    p.add_argument("--input-jsonl", type=Path, required=True, help="Prompt JSONL path")
    p.add_argument("--output-jsonl", type=Path, required=True, help="Predictions JSONL path")
    p.add_argument("--summary-json", type=Path, required=True, help="Summary metrics JSON path")
    p.add_argument(
        "--model-id",
        default="HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        help="Hugging Face model id",
    )
    p.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Execution device",
    )
    p.add_argument("--max-items", type=int, default=None, help="Limit number of rows")
    p.add_argument("--max-new-tokens", type=int, default=8, help="Generation length")
    p.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Enable trust_remote_code when loading model/processor",
    )
    return p.parse_args()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _letter_from_record(rec: dict[str, Any]) -> str | None:
    gold_letter = rec.get("gold_letter")
    if isinstance(gold_letter, str) and gold_letter:
        return gold_letter.strip().upper()
    gold_index = rec.get("gold_index")
    if isinstance(gold_index, int) and 0 <= gold_index < len(LETTERS):
        return LETTERS[gold_index]
    return None


def _extract_letter(text: str, n_options: int) -> str | None:
    m = re.search(r"\b([A-Z])\b", text.upper())
    if not m:
        return None
    letter = m.group(1)
    if letter in LETTERS[:n_options]:
        return letter
    return None


def _messages_for_record(rec: dict[str, Any]) -> list[dict[str, Any]]:
    messages = rec.get("messages")
    if isinstance(messages, list) and messages:
        return messages
    prompt_text = rec.get("prompt_text", "")
    return [{"role": "user", "content": [{"type": "text", "text": str(prompt_text)}]}]


def _load_model(model_id: str, device: str, trust_remote_code: bool):
    import torch
    import transformers
    from transformers import AutoProcessor

    torch_dtype = torch.float16 if (device == "cuda" or (device == "auto" and torch.cuda.is_available())) else torch.float32
    model_kwargs: dict[str, Any] = {"torch_dtype": torch_dtype}
    if device == "auto":
        model_kwargs["device_map"] = "auto"
    model_cls = getattr(transformers, "AutoModelForImageTextToText", None)
    if model_cls is None:
        model_cls = getattr(transformers, "AutoModelForVision2Seq")
    model = model_cls.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
        **model_kwargs,
    )
    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
    )
    if device in ("cpu", "cuda"):
        model = model.to(device)
    return model, processor


def _generate_one(
    model: Any,
    processor: Any,
    rec: dict[str, Any],
    device: str,
    max_new_tokens: int,
) -> str:
    import torch

    messages = _messages_for_record(rec)
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, return_tensors="pt")
    if device in ("cpu", "cuda"):
        inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    prompt_len = inputs["input_ids"].shape[1]
    gen_ids = out_ids[:, prompt_len:]
    text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
    return text.strip()


def run(args: argparse.Namespace) -> None:
    rows = _read_jsonl(args.input_jsonl)
    if args.max_items is not None:
        rows = rows[: args.max_items]
    if not rows:
        raise ValueError("No records found in input JSONL")

    model, processor = _load_model(
        args.model_id,
        args.device,
        args.trust_remote_code,
    )
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    parsed = 0
    correct_total = 0
    correct_parsed = 0

    with open(args.output_jsonl, "w", encoding="utf-8") as f_out:
        for rec in rows:
            options = rec.get("options") or []
            n_options = len(options)
            if n_options < 2:
                continue
            gold_letter = _letter_from_record(rec)
            if gold_letter is None:
                continue

            pred_text = _generate_one(
                model=model,
                processor=processor,
                rec=rec,
                device=args.device,
                max_new_tokens=args.max_new_tokens,
            )
            pred_letter = _extract_letter(pred_text, n_options=n_options)
            is_correct = pred_letter == gold_letter if pred_letter is not None else False
            total += 1
            if is_correct:
                correct_total += 1
            if pred_letter is not None:
                parsed += 1
                if is_correct:
                    correct_parsed += 1

            out = {
                "item_uid": rec.get("item_uid"),
                "gold_letter": gold_letter,
                "pred_text": pred_text,
                "pred_letter": pred_letter,
                "correct": is_correct,
                "options": options,
            }
            f_out.write(json.dumps(out, ensure_ascii=True) + "\n")

    summary = {
        "model_id": args.model_id,
        "input_jsonl": str(args.input_jsonl),
        "output_jsonl": str(args.output_jsonl),
        "n_total": total,
        "n_parsed": parsed,
        "accuracy_all": (correct_total / total) if total else None,
        "accuracy_parsed_only": (correct_parsed / parsed) if parsed else None,
        "parse_rate": (parsed / total) if total else None,
    }
    with open(args.summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
