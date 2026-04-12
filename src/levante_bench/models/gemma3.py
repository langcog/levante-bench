"""Gemma 3 model implementation."""

import torch

from levante_bench.models.base import ParseResult, VLMModel
from levante_bench.models.registry import register
from levante_bench.models._common import (
    DTYPE_MAP,
    build_pil_content,
    load_pil_images,
    parse_answer_result_with_fallback,
    parse_answer_with_fallback,
)

_SYSTEM_PROMPT = (
    "You are a helpful assistant. "
    "Answer with only a single letter: A, B, C, or D. Do not explain."
)


@register("gemma3")
class Gemma3Model(VLMModel):
    """Gemma 3 via HuggingFace AutoProcessor + AutoModelForImageTextToText."""

    def __init__(
        self,
        model_name: str = "google/gemma-3-4b-it",
        device: str = "cpu",
        dtype: str = "bfloat16",
        attn_implementation: str = "sdpa",
        max_image_edge: int = 1024,
    ) -> None:
        super().__init__(model_name=model_name, device=device)
        self.dtype = DTYPE_MAP.get(dtype, torch.bfloat16)
        self.attn_implementation = attn_implementation
        self.max_image_edge = int(max_image_edge)

    def load(self) -> None:
        """Load Gemma 3 model and processor from HuggingFace."""
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_name,
            dtype=self.dtype,
            attn_implementation=self.attn_implementation,
        ).to(self.device)
        self.model.eval()

    def generate(
        self,
        prompt_text: str,
        image_paths: list[str] | None = None,
        max_new_tokens: int = 128,
    ) -> str:
        """Generate text using Gemma 3."""
        pil_images = load_pil_images(
            image_paths,
            max_image_edge=self.max_image_edge,
        )
        messages = self._build_messages(prompt_text, pil_images)

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text],
            images=pil_images if pil_images else None,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        input_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs, do_sample=False, max_new_tokens=max_new_tokens
            )

        generated_ids = output_ids[:, input_len:]
        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    def _build_messages(
        self,
        prompt_text: str,
        pil_images: list | None = None,
    ) -> list[dict]:
        """Build Gemma 3 chat messages with optional interleaved images."""
        content = build_pil_content(prompt_text, pil_images)
        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]

    def parse_response(self, raw_output: str) -> str:
        return raw_output.strip()

    def parse_answer(
        self, text: str, option_labels: list[str]
    ) -> tuple[str | None, str]:
        """Base-class parser first; falls back to reverse-sentence scan."""
        return parse_answer_with_fallback(self, text, option_labels)

    def parse_answer_result(self, text: str, option_labels: list[str]) -> ParseResult:
        """Parser with provenance, including reverse-sentence fallback."""
        return parse_answer_result_with_fallback(self, text, option_labels)
