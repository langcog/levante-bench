"""TinyLLaVA model implementation.

Multi-image support is achieved by compositing the option images into a
labeled 2×2 grid that is passed as a single image to model.chat().

Available model IDs:
    tinyllava/TinyLLaVA-Qwen2-0.5B-SigLIP          (0.5 B)
    tinyllava/TinyLLaVA-OpenELM-450M-SigLIP-0.89B   (0.9 B)
    tinyllava/TinyLLaVA-Gemma-SigLIP-2.4B           (2.4 B)
    tinyllava/TinyLLaVA-Qwen2.5-3B-SigLIP           (3.0 B)
    tinyllava/TinyLLaVA-Phi-2-SigLIP-3.1B           (3.1 B)
"""

import re
import tempfile
from pathlib import Path
from typing import Optional

import torch
from PIL import Image, ImageDraw, ImageFont

from levante_bench.models.base import VLMModel
from levante_bench.models.registry import register
from levante_bench.models._common import DTYPE_MAP, parse_answer_with_fallback

_LABELS = ["A", "B", "C", "D"]
_CELL = 224   # each option image is resized to CELL × CELL pixels
_FONT_SIZE = 22


@register("tinyllava")
@register("tinyllava_0.5b")
@register("tinyllava_0.9b")
@register("tinyllava_2.4b")
@register("tinyllava_3b")
@register("tinyllava_3.1b")
class TinyLLaVAModel(VLMModel):
    """TinyLLaVA via HuggingFace AutoModelForCausalLM + trust_remote_code.

    TinyLLaVA exposes a single-image ``model.chat()`` API.  Multi-image
    inputs (e.g. the 4 Vocab option images) are handled by compositing them
    into a labeled 2×2 grid that the model sees as one image.  The prompt
    template's ``<imageN>`` placeholders are replaced with a grid description
    so the model knows which cells correspond to A, B, C, D.
    """

    def __init__(
        self,
        model_name: str = "tinyllava/TinyLLaVA-Phi-2-SigLIP-3.1B",
        device: str = "cpu",
        dtype: str = "bfloat16",
    ) -> None:
        super().__init__(model_name=model_name, device=device)
        self.dtype = DTYPE_MAP.get(dtype, torch.bfloat16)
        self._tmp_dir: Optional[str] = None

    # ── Loading ─────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load TinyLLaVA model and tokenizer from HuggingFace."""
        from transformers import AutoTokenizer, AutoModelForCausalLM

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            dtype=self.dtype,
            attn_implementation="eager",  # legacy model lacks _supports_sdpa
        ).to(self.device)
        self.model.eval()

        cfg = self.model.config
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_fast=False,
            model_max_length=getattr(cfg, "tokenizer_model_max_length", 2048),
            padding_side=getattr(cfg, "tokenizer_padding_side", "right"),
            trust_remote_code=True,
        )

    # ── Inference ───────────────────────────────────────────────────────────

    def generate(
        self,
        prompt_text: str,
        image_paths: list[str] | None = None,
        max_new_tokens: int = 32,
    ) -> str:
        """Generate text using TinyLLaVA's model.chat() API."""
        image, prompt = self._prepare_inputs(prompt_text, image_paths)

        output, _ = self.model.chat(
            prompt=prompt,
            image=image,
            tokenizer=self.tokenizer,
            max_new_tokens=max_new_tokens,
            temperature=0,
            num_beams=1,
        )
        return output

    # ── Image & prompt preparation ──────────────────────────────────────────

    def _prepare_inputs(
        self,
        prompt_text: str,
        image_paths: list[str] | None,
    ) -> tuple:
        """Return (image_arg, cleaned_prompt) for model.chat().

        - No images  → (None, prompt unchanged)
        - One image  → (path_str, prompt without <image1>)
        - 4 images   → (grid_path_str, prompt rewritten for A/B/C/D grid)
        """
        if not image_paths:
            return None, prompt_text

        # Remove all <imageN> placeholders from the original prompt
        clean_prompt = re.sub(r"<image\d+>", "", prompt_text).strip()

        if len(image_paths) == 1:
            return str(Path(image_paths[0]).resolve()), clean_prompt

        # Multiple images → compose a labeled grid
        grid_path = self._make_grid(image_paths)
        # Append a grid-layout hint so the model understands the labeling
        n = min(len(image_paths), 4)
        layout = ", ".join(
            f"{_LABELS[i]}={'top-left' if i==0 else 'top-right' if i==1 else 'bottom-left' if i==2 else 'bottom-right'}"
            for i in range(n)
        )
        grid_prompt = (
            f"{clean_prompt} "
            f"The image is a {2}×{(n+1)//2} grid of options ({layout}). "
            "Reply with only the letter of the correct option."
        )
        return grid_path, grid_prompt

    def _make_grid(self, image_paths: list[str]) -> str:
        """Compose up to 4 images into a labeled 2×2 grid; return temp path."""
        n = min(len(image_paths), 4)
        cols, rows = 2, (n + 1) // 2
        grid = Image.new("RGB", (cols * _CELL, rows * _CELL), color=(240, 240, 240))
        draw = ImageDraw.Draw(grid)

        # Try to load a font; fall back to default if not available
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", _FONT_SIZE)
        except OSError:
            font = ImageFont.load_default()

        for i in range(n):
            img = Image.open(image_paths[i]).convert("RGB").resize(
                (_CELL, _CELL), Image.LANCZOS
            )
            x, y = (i % cols) * _CELL, (i // cols) * _CELL
            grid.paste(img, (x, y))
            # Draw a small label badge in the top-left corner of each cell
            badge_w, badge_h = 28, 28
            draw.rectangle([x, y, x + badge_w, y + badge_h], fill=(0, 0, 0))
            draw.text((x + 6, y + 4), _LABELS[i], fill=(255, 255, 255), font=font)

        if self._tmp_dir is None:
            self._tmp_dir = tempfile.mkdtemp()
        path = str(Path(self._tmp_dir) / "grid.png")
        grid.save(path)
        return path

    # ── Output parsing ──────────────────────────────────────────────────────

    def parse_response(self, raw_output: str) -> str:
        return raw_output.strip()

    def parse_answer(self, text: str, option_labels: list[str]) -> Optional[str]:
        """Base-class parser first; falls back to reverse-sentence scan."""
        return parse_answer_with_fallback(self, text, option_labels)
