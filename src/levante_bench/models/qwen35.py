"""Qwen3.5-VL model implementation."""

from typing import Optional

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


@register("qwen35")
@register("qwen25vl_qlora")
@register("qwen25vl_32b")
class Qwen35Model(VLMModel):
    """Qwen3.5-VL via HuggingFace AutoProcessor + AutoModelForImageTextToText.

    Images are loaded as PIL objects and passed separately from the text so
    that no extra dependency (qwen_vl_utils) is required.  The processor
    applies the Qwen3 chat template and inserts vision tokens automatically.
    Only the newly generated tokens are decoded, so parse_response is trivial.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3.5-0.8B",
        device: str = "cpu",
        dtype: str = "bfloat16",
        attn_implementation: str = "sdpa",
    ) -> None:
        super().__init__(model_name=model_name, device=device)
        self.dtype = DTYPE_MAP.get(dtype, torch.bfloat16)
        self.attn_implementation = attn_implementation

    def load(self) -> None:
        """Load Qwen3.5 model and processor from HuggingFace."""
        from transformers import AutoProcessor, AutoModelForImageTextToText

        self.processor = AutoProcessor.from_pretrained(
            self.model_name, padding_side="left"
        )
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
        max_new_tokens: int = 64,
    ) -> str:
        """Generate text using Qwen3.5-VL."""
        pil_images = load_pil_images(image_paths)
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
        pil_images: Optional[list] = None,
    ) -> list[dict]:
        """Build Qwen3.5 chat messages with system prompt and PIL images."""
        content = build_pil_content(prompt_text, pil_images)
        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]

    def parse_response(self, raw_output: str) -> str:
        """Return generated text as-is (already decoded from generated tokens only)."""
        return raw_output.strip()

    def score_choices(
        self,
        prompt_text: str,
        image_paths: list[str],
        choice_texts: tuple[str, str] = ("1", "2"),
    ) -> dict:
        """Return next-token probabilities/logits for two one-token choices."""
        pil_images = load_pil_images(image_paths)
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

        choice_ids: list[int] = []
        for choice in choice_texts:
            toks = self.processor.tokenizer.encode(choice, add_special_tokens=False)
            if len(toks) != 1:
                raise ValueError(
                    f"Choice {choice!r} must map to one token; got ids={toks}"
                )
            choice_ids.append(toks[0])

        output, elapsed = self._timed_call(lambda: self.model(**inputs))
        next_logits = output.logits[:, -1, :].float()
        selected = next_logits[:, choice_ids].squeeze(0)
        probs = torch.softmax(selected, dim=-1)
        return {
            "choice_texts": list(choice_texts),
            "choice_token_ids": choice_ids,
            "choice_logits": [float(selected[0].item()), float(selected[1].item())],
            "choice_probs": [float(probs[0].item()), float(probs[1].item())],
            "generation_time_s": elapsed,
            "model_name": self.model_name,
            "num_tokens_generated": 0,
        }

    def parse_answer(
        self, text: str, option_labels: list[str]
    ) -> tuple[Optional[str], str]:
        """Base-class parser first; falls back to reverse-sentence scan."""
        return parse_answer_with_fallback(self, text, option_labels)

    def parse_answer_result(self, text: str, option_labels: list[str]) -> ParseResult:
        """Parser with provenance, including reverse-sentence fallback."""
        return parse_answer_result_with_fallback(self, text, option_labels)
