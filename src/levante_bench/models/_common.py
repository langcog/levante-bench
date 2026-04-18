"""Shared utilities for VLM model implementations."""

import re
from typing import Optional

import torch
from PIL import Image


DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

_MAX_IMAGE_EDGE = 1024


def load_pil_images(
    image_paths: list[str] | None,
    max_image_edge: int | None = None,
) -> Optional[list]:
    """Load a list of file paths as RGB PIL Images."""
    if not image_paths:
        return None
    images = []
    edge_limit = int(max_image_edge) if max_image_edge is not None else _MAX_IMAGE_EDGE
    for path in image_paths:
        img = Image.open(path).convert("RGB")
        # Keep very large option images from destabilizing CUDA vision forward passes.
        if max(img.size) > edge_limit:
            img.thumbnail((edge_limit, edge_limit), Image.Resampling.LANCZOS)
        images.append(img)
    return images


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
        # If <image0> is present, the prompt uses direct indices (context=0, options=1..);
        # otherwise fall back to the 1-based convention (<image1> → pil_images[0]).
        has_image0 = "<image0>" in prompt_text
        for part in parts:
            m = re.match(r"<image(\d+)>", part)
            if m:
                n = int(m.group(1))
                idx = n if has_image0 else n - 1
                if 0 <= idx < len(pil_images):
                    content.append({"type": "image", "image": pil_images[idx]})
            elif part.strip():
                content.append({"type": "text", "text": part.strip()})
    else:
        if pil_images:
            for img in pil_images:
                content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": prompt_text})
    return content


