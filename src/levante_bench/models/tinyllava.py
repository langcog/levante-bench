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

from levante_bench.models.base import ParseResult, VLMModel
from levante_bench.models.registry import register
from levante_bench.models._common import (
    DTYPE_MAP,
    parse_answer_result_with_fallback,
    parse_answer_with_fallback,
)

_LABELS = ["A", "B", "C", "D"]
_CELL = 224   # each option image is resized to CELL × CELL pixels
_FONT_SIZE = 22


@register("tinyllava")
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
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                dtype=self.dtype,
                attn_implementation="eager",  # legacy model lacks _supports_sdpa
            ).to(self.device)
        except TypeError as exc:
            message = str(exc)
            tie_kwarg_mismatch = (
                "tie_weights()" in message
                and ("recompute_mapping" in message or "missing_keys" in message)
            )
            if not tie_kwarg_mismatch:
                raise

            model_cls = None
            for class_ref in (
                "modeling_tinyllava_phi.TinyLlavaForConditionalGeneration",
                "modeling_tinyllava.TinyLlavaForConditionalGeneration",
            ):
                try:
                    model_cls = get_class_from_dynamic_module(class_ref, self.model_name)
                    break
                except Exception:
                    continue
            if model_cls is None:
                raise

            if not getattr(model_cls, "_levante_tie_weights_patch", False):
                original_tie_weights = model_cls.tie_weights

                def _patched_tie_weights(self, *args, **kwargs):
                    return original_tie_weights(self)

                model_cls.tie_weights = _patched_tie_weights
                model_cls._levante_tie_weights_patch = True

            self.model = model_cls.from_pretrained(
                self.model_name,
                dtype=self.dtype,
                attn_implementation="eager",
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
        *,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        sample_seed: int | None = None,
    ) -> str:
        """Generate text using TinyLLaVA's model.chat() API."""
        if not image_paths:
            # TinyLLaVA chat() can fail when no image is provided.
            # Route text-only prompts through chat() with a tiny blank image.
            image_paths = [self._get_blank_image_path()]

        image, prompt = self._prepare_inputs(prompt_text, image_paths)

        if sample_seed is not None:
            torch.manual_seed(int(sample_seed) % (2**32))
        temp = max(float(temperature), 1e-5) if do_sample else 0.0

        output, _ = self.model.chat(
            prompt=prompt,
            image=image,
            tokenizer=self.tokenizer,
            max_new_tokens=max_new_tokens,
            temperature=temp,
            num_beams=1,
        )
        return output

    def _get_blank_image_path(self) -> str:
        """Return a persistent local blank image path for text-only trials."""
        if self._tmp_dir is None:
            self._tmp_dir = tempfile.mkdtemp()
        path = Path(self._tmp_dir) / "blank.png"
        if not path.exists():
            Image.new("RGB", (_CELL, _CELL), color=(255, 255, 255)).save(path)
        return str(path)

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
            # TinyLLaVA's chat() requires an image; for text-only tasks pass a
            # small blank white image so image_tensor is always bound.
            blank = Image.new("RGB", (64, 64), color=(255, 255, 255))
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            blank.save(tmp.name)
            return tmp.name, prompt_text

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

    def parse_answer(
        self, text: str, option_labels: list[str]
    ) -> tuple[Optional[str], str]:
        """Base-class parser first; falls back to reverse-sentence scan."""
        return parse_answer_with_fallback(self, text, option_labels)

    def parse_answer_result(self, text: str, option_labels: list[str]) -> ParseResult:
        """Parser with provenance, including reverse-sentence fallback."""
        return parse_answer_result_with_fallback(self, text, option_labels)
