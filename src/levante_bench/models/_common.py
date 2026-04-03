"""Shared utilities for VLM model implementations."""

import re
from typing import Optional

import torch
from PIL import Image

from levante_bench.models.base import ParseResult


DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def load_pil_images(image_paths: list[str] | None) -> Optional[list]:
    """Load a list of file paths as RGB PIL Images."""
    if not image_paths:
        return None
    return [Image.open(p).convert("RGB") for p in image_paths]


def build_pil_content(
    prompt_text: str,
    pil_images: Optional[list],
) -> list[dict]:
    """Build a content list interleaving PIL images at <imageN> placeholders.

    Returns a list of ``{"type": "image"|"text", ...}`` dicts suitable for
    HuggingFace chat templates that accept PIL objects.
    """
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
        if pil_images:
            for img in pil_images:
                content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": prompt_text})
    return content


def parse_answer_with_fallback(
    base_instance,
    text: str,
    option_labels: list[str],
) -> tuple[Optional[str], str]:
    """Try the base-class parser, then scan backwards through sentences.

    Used by Qwen35Model and InternVL35Model, which may generate a brief
    chain-of-thought before stating the answer letter.
    """
    result = parse_answer_result_with_fallback(base_instance, text, option_labels)
    label = str(result.value).upper() if result.value is not None else None
    return label, result.reason


def parse_answer_result_with_fallback(
    base_instance,
    text: str,
    option_labels: list[str],
) -> ParseResult:
    """Like parse_answer_with_fallback, but returns ParseResult provenance."""
    result = super(type(base_instance), base_instance).parse_answer_result(
        text, option_labels
    )
    if result.value is not None:
        return result
    sentences = re.split(r"[.!?\n]", text)
    for sentence in reversed(sentences):
        sentence = sentence.strip()
        for label in option_labels:
            if re.search(rf"\b{re.escape(label)}\b", sentence, re.IGNORECASE):
                return ParseResult(
                    value=label.upper(),
                    reason=sentence,
                    parse_method="reverse_sentence_fallback",
                    parse_confidence="low",
                    raw_candidate=label,
                )
    return ParseResult(
        value=None,
        reason=text,
        parse_method="unparseable",
        parse_confidence="none",
    )
