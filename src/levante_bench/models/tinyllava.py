"""TinyLLaVA model implementation.

Multi-image support is achieved by compositing the option images into a
labeled 2×2 grid that is passed as a single image to model.chat().

Upstream checkpoints use HuggingFace ``trust_remote_code`` and the project-specific
``TinyLlavaForConditionalGeneration`` + ``model.chat()`` API — not the generic
``AutoModelForImageTextToText`` path, so we stay aligned with the reference
implementation.

Available model IDs:
    tinyllava/TinyLLaVA-Qwen2-0.5B-SigLIP          (0.5 B)
    tinyllava/TinyLLaVA-OpenELM-450M-SigLIP-0.89B   (0.9 B)
    tinyllava/TinyLLaVA-Gemma-SigLIP-2.4B           (2.4 B)
    tinyllava/TinyLLaVA-Qwen2.5-3B-SigLIP           (3.0 B)
    tinyllava/TinyLLaVA-Phi-2-SigLIP-3.1B           (3.1 B)
"""

from __future__ import annotations

import hashlib
import random
import re
import tempfile
from pathlib import Path
from typing import Any, Optional

import torch
from PIL import Image, ImageDraw, ImageFont

from levante_bench.models.base import VLMModel
from levante_bench.models.registry import register
from levante_bench.models._common import DTYPE_MAP

_LABELS = ["A", "B", "C", "D"]
_CELL = 224   # each option image is resized to CELL × CELL pixels
_FONT_SIZE = 22

# Prefer Linux paths on Marlowe; fall back to macOS then PIL default.
_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
)

_CELL_NAMES = ("top-left", "top-middle", "top-right", "middle-left", "middle", "middle-right", "bottom-left", "bottom-middle", "bottom-right")


def _load_grid_font() -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, _FONT_SIZE)
        except OSError:
            continue
    return ImageFont.load_default()


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
        generation: dict[str, Any] | None = None,
        device_map: str | None = None,
        attn_implementation: str = "eager",
    ) -> None:
        super().__init__(model_name=model_name, device=device)
        self.attn_implementation = str(attn_implementation or "eager")
        self.dtype = DTYPE_MAP.get(str(dtype), torch.bfloat16)
        self.device_map = str(device_map).strip() if device_map else None
        gen = dict(generation) if isinstance(generation, dict) else {}
        self._chat_generation_defaults: dict[str, Any] = {
            "temperature": 0,
            "num_beams": 1,
            **gen,
        }
        self._tmp_dir: Optional[str] = None

    # ── Loading ─────────────────────────────────────────────────────────────

    def _from_pretrained_causal_lm(self, model_cls=None):
        """Load with torch_dtype when available; optional device_map (e.g. auto on CUDA)."""
        from transformers import AutoModelForCausalLM

        kw: dict[str, Any] = {
            "trust_remote_code": True,
            "attn_implementation": self.attn_implementation,
            # Some upstream remote-code paths only honor the underscored key.
            "_attn_implementation": self.attn_implementation,
        }
        if self.device_map:
            kw["device_map"] = self.device_map

        def _load_with_kwargs(load_kwargs: dict[str, Any]):
            return (
                AutoModelForCausalLM.from_pretrained(self.model_name, **load_kwargs)
                if model_cls is None
                else model_cls.from_pretrained(self.model_name, **load_kwargs)
            )

        try:
            kw["torch_dtype"] = self.dtype
            m = _load_with_kwargs(kw)
        except TypeError:
            kw.pop("torch_dtype", None)
            kw["dtype"] = self.dtype
            m = _load_with_kwargs(kw)
        except ImportError as exc:
            # Transformers may still route to flash-attn based on checkpoint
            # defaults even when adapters request eager/sdpa. Retry explicitly.
            if "flash_attn" not in str(exc).lower() and "flashattention" not in str(exc).lower():
                raise
            kw.pop("torch_dtype", None)
            kw.pop("dtype", None)
            kw["attn_implementation"] = "eager"
            kw["_attn_implementation"] = "eager"
            try:
                kw["torch_dtype"] = self.dtype
                m = _load_with_kwargs(kw)
            except TypeError:
                kw.pop("torch_dtype", None)
                kw["dtype"] = self.dtype
                m = _load_with_kwargs(kw)
        if not self.device_map:
            m = m.to(self.device)
        return m

    def load(self) -> None:
        """Load TinyLLaVA model and tokenizer from HuggingFace."""
        from transformers import AutoTokenizer
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        try:
            self.model = self._from_pretrained_causal_lm(model_cls=None)
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
                "modeling_tinyllava_gemma.TinyLlavaForConditionalGeneration",
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

            self.model = self._from_pretrained_causal_lm(model_cls=model_cls)
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
        if not image_paths:
            # TinyLLaVA chat() can fail when no image is provided.
            # Route text-only prompts through chat() with a tiny blank image.
            image_paths = [self._get_blank_image_path()]

        image, prompt = self._prepare_inputs(prompt_text, image_paths)

        chat_kw = dict(self._chat_generation_defaults)
        chat_kw["max_new_tokens"] = max_new_tokens
        try:
            output, _ = self.model.chat(
                prompt=prompt,
                image=image,
                tokenizer=self.tokenizer,
                **chat_kw,
            )
        except RuntimeError as exc:
            msg = str(exc)
            if "CUDNN_STATUS_NOT_INITIALIZED" not in msg and "cuDNN" not in msg:
                raise
            # Some Marlowe nodes intermittently fail vision conv initialization
            # for TinyLLaVA/SigLIP. Retry once with cuDNN disabled.
            torch.cuda.empty_cache()
            with torch.backends.cudnn.flags(enabled=False):
                output, _ = self.model.chat(
                    prompt=prompt,
                    image=image,
                    tokenizer=self.tokenizer,
                    **chat_kw,
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
            return None, prompt_text

        # Remove all <imageN> placeholders from the original prompt
        clean_prompt = re.sub(r"<image\d+>", "", prompt_text).strip()

        if len(image_paths) == 1:
            return str(Path(image_paths[0]).resolve()), clean_prompt

        has_image0 = "<image0>" in prompt_text
        labels: list[str] = []
        for i, _ in enumerate(image_paths):
            if has_image0:
                if i == 0:
                    labels.append("0")
                elif 1 <= i <= 26:
                    labels.append(chr(ord("A") + i - 1))
                else:
                    labels.append(str(i))
            else:
                if i < 26:
                    labels.append(chr(ord("A") + i))
                else:
                    labels.append(str(i + 1))

        entries = list(zip(image_paths, labels))
        # Debias fixed corner priors by shuffling placement deterministically.
        # Keep context image (label 0) fixed when present.
        context_entry = None
        if entries and entries[0][1] == "0":
            context_entry = entries[0]
            entries = entries[1:]
        seed_src = f"{prompt_text}|{'|'.join(image_paths)}"
        seed = int(hashlib.sha256(seed_src.encode("utf-8")).hexdigest()[:16], 16)
        rng = random.Random(seed)
        rng.shuffle(entries)
        if context_entry is not None:
            entries = [context_entry] + entries

        # Multiple images → compose a labeled grid
        grid_path, placement = self._make_grid(entries)
        # Append a grid-layout hint so the model understands the labeling.
        layout = ", ".join(f"{label}={cell_name}" for label, cell_name in placement)
        grid_prompt = (
            f"{clean_prompt} "
            f"The image is a labeled grid ({layout})."
        )
        return grid_path, grid_prompt

    def _make_grid(self, entries: list[tuple[str, str]]) -> tuple[str, list[tuple[str, str]]]:
        """Compose images into a labeled grid; return (path, label placement)."""
        n = len(entries)
        if n <= 0:
            raise ValueError("No images to compose.")
        cols = 2 if n <= 4 else 3
        rows = (n + cols - 1) // cols
        grid = Image.new("RGB", (cols * _CELL, rows * _CELL), color=(240, 240, 240))
        draw = ImageDraw.Draw(grid)

        font = _load_grid_font()
        placement: list[tuple[str, str]] = []

        for i, (image_path, label) in enumerate(entries):
            img = Image.open(image_path).convert("RGB").resize(
                (_CELL, _CELL), Image.LANCZOS
            )
            x, y = (i % cols) * _CELL, (i // cols) * _CELL
            grid.paste(img, (x, y))
            cell_name = _CELL_NAMES[i] if i < len(_CELL_NAMES) else f"cell-{i}"
            placement.append((label, cell_name))
            # Draw a small label badge in the top-left corner of each cell
            badge_w, badge_h = 28, 28
            draw.rectangle([x, y, x + badge_w, y + badge_h], fill=(0, 0, 0))
            draw.text((x + 6, y + 4), label, fill=(255, 255, 255), font=font)

        if self._tmp_dir is None:
            self._tmp_dir = tempfile.mkdtemp()
        path = str(Path(self._tmp_dir) / "grid.png")
        grid.save(path)
        return path, placement

    # ── Output parsing ──────────────────────────────────────────────────────

    def parse_response(self, raw_output: str) -> str:
        return raw_output.strip()
