"""VLM model implementations. One class per model family."""

import re

import torch
from PIL import Image

from levante_bench.models.base import VLMModel
from levante_bench.models.registry import register


@register("smolvlm2")
class SmolVLM2Model(VLMModel):
    """SmolVLM2 via HuggingFace AutoProcessor + AutoModelForVision2Seq."""

    def __init__(self, model_name: str = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct", device: str = "cpu") -> None:
        super().__init__(model_name=model_name, device=device)

    def load(self) -> None:
        """Load SmolVLM2 model and processor from HuggingFace."""
        from transformers import AutoProcessor, AutoModelForVision2Seq

        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = AutoModelForVision2Seq.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

    def generate(
        self,
        prompt_text: str,
        images: list[Image.Image] | None = None,
        max_new_tokens: int = 64,
    ) -> str:
        """Generate text using SmolVLM2."""
        messages = self._build_messages(prompt_text, images)
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

        # Decode only newly generated tokens
        new_tokens = output_ids[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0]

    def _build_messages(
        self,
        prompt_text: str,
        images: list[Image.Image] | None = None,
    ) -> list[dict]:
        """Build SmolVLM2 chat messages with interleaved image and text content."""
        content = []
        if images:
            for img in images:
                content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": prompt_text})
        return [{"role": "user", "content": content}]

    def parse_response(self, raw_output: str) -> str:
        """Clean SmolVLM2 output — strip whitespace and trailing assistant tags."""
        text = raw_output.strip()
        # Remove trailing assistant/end tags if present
        text = re.sub(r'<\|?end\|?>.*$', '', text).strip()
        text = re.sub(r'<\|?assistant\|?>.*$', '', text).strip()
        return text
