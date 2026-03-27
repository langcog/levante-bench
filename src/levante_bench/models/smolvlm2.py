"""SmolVLM2 model implementation."""

import re
from pathlib import Path

import torch

from levante_bench.models.base import VLMModel
from levante_bench.models.registry import register
from levante_bench.models._common import DTYPE_MAP


@register("smolvlm2")
@register("smolvlm2_256m")
@register("smolvlm2_500m")
@register("smolvlm2_2b")
class SmolVLM2Model(VLMModel):
    """SmolVLM2 via HuggingFace AutoProcessor + AutoModelForImageTextToText."""

    def __init__(
        self,
        model_name: str = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
        device: str = "cpu",
        dtype: str = "bfloat16",
        attn_implementation: str = "eager",
    ) -> None:
        super().__init__(model_name=model_name, device=device)
        self.dtype = DTYPE_MAP.get(dtype, torch.bfloat16)
        self.attn_implementation = attn_implementation

    def load(self) -> None:
        """Load SmolVLM2 model and processor from HuggingFace."""
        from transformers import AutoProcessor, AutoModelForImageTextToText

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
        max_new_tokens: int = 64,
    ) -> str:
        """Generate text using SmolVLM2."""
        messages = self._build_messages(prompt_text, image_paths)
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device, dtype=self.dtype)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs, do_sample=False, max_new_tokens=max_new_tokens
            )

        return self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]

    def _build_messages(
        self,
        prompt_text: str,
        image_paths: list[str] | None = None,
    ) -> list[dict]:
        """Build SmolVLM2 chat messages, interleaving images at <imageN> placeholders."""
        content = []
        if image_paths and re.search(r"<image\d+>", prompt_text):
            labels = ["A", "B", "C", "D", "E", "F", "G", "H"]
            parts = re.split(r"(<image\d+>)", prompt_text)
            for part in parts:
                m = re.match(r"<image(\d+)>", part)
                if m:
                    idx = int(m.group(1)) - 1
                    if idx < len(image_paths):
                        label = labels[idx] if idx < len(labels) else str(idx + 1)
                        content.append({"type": "text", "text": f"{label}:"})
                        content.append(
                            {"type": "image", "url": str(Path(image_paths[idx]).resolve())}
                        )
                elif part.strip():
                    content.append({"type": "text", "text": part.strip()})
        else:
            if image_paths:
                for path in image_paths:
                    content.append({"type": "image", "url": str(Path(path).resolve())})
            content.append({"type": "text", "text": prompt_text})
        return [{"role": "user", "content": content}]

    def parse_response(self, raw_output: str) -> str:
        """Extract only the assistant's reply from SmolVLM2 output."""
        if "Assistant:" in raw_output:
            text = raw_output.split("Assistant:")[-1]
        else:
            text = raw_output
        return re.sub(r"<\|?end\|?>.*$", "", text).strip()
