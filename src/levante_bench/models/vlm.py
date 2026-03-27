"""VLM model implementations. One class per model family."""

import re

from pathlib import Path
from typing import Optional

import torch
from PIL import Image

from levante_bench.models.base import VLMModel
from levante_bench.models.registry import register


DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


@register("smolvlm2")
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
            output_ids = self.model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)

        generated_text = self.processor.batch_decode(
            output_ids,
            skip_special_tokens=True,
        )[0]
        return generated_text

    def _build_messages(
        self,
        prompt_text: str,
        image_paths: list[str] | None = None,
    ) -> list[dict]:
        """Build SmolVLM2 chat messages, interleaving images at <imageN> placeholders."""
        content = []
        if image_paths and re.search(r'<image\d+>', prompt_text):
            # Split prompt on <imageN> placeholders and interleave with images
            labels = ["A", "B", "C", "D", "E", "F", "G", "H"]
            parts = re.split(r'(<image\d+>)', prompt_text)
            has_image0 = "<image0>" in prompt_text
            for part in parts:
                m = re.match(r'<image(\d+)>', part)
                if m:
                    n = int(m.group(1))
                    # If <image0> is present, treat image numbers as direct indices
                    # into the concatenated image_paths list (context first).
                    # Otherwise, keep the existing 1-based convention (<image1> -> image_paths[0]).
                    idx = n if has_image0 else n - 1
                    if idx < len(image_paths):
                        # <image0> is reserved for a "prompt/context" image (no label prefix).
                        if n != 0:
                            label_idx = n - 1
                            label = labels[label_idx] if label_idx < len(labels) else str(label_idx + 1)
                            content.append({"type": "text", "text": f"{label}:"})
                        content.append({"type": "image", "url": str(Path(image_paths[idx]).resolve())})
                elif part.strip():
                    content.append({"type": "text", "text": part.strip()})
        else:
            # No placeholders — images first (context), then text
            if image_paths:
                for path in image_paths:
                    content.append({"type": "image", "url": str(Path(path).resolve())})
            content.append({"type": "text", "text": prompt_text})
        return [{"role": "user", "content": content}]

    def parse_response(self, raw_output: str) -> str:
        """Extract only the assistant's reply from SmolVLM2 output."""
        # Output includes full conversation; extract after last "Assistant:"
        if "Assistant:" in raw_output:
            text = raw_output.split("Assistant:")[-1]
        else:
            text = raw_output
        text = re.sub(r'<\|?end\|?>.*$', '', text).strip()
        return text


@register("qwen35")
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
        pil_images = self._load_pil_images(image_paths)
        messages = self._build_messages(prompt_text, pil_images)

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
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
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
            )

        generated_ids = output_ids[:, input_len:]
        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    def _load_pil_images(self, image_paths: list[str] | None) -> Optional[list]:
        """Load image paths as RGB PIL Images."""
        if not image_paths:
            return None
        return [Image.open(p).convert("RGB") for p in image_paths]

    def _build_messages(
        self,
        prompt_text: str,
        pil_images: Optional[list] = None,
    ) -> list[dict]:
        """Build Qwen3.5 chat messages with PIL images at <imageN> placeholders.

        A system prompt is prepended so the model outputs only the answer
        letter rather than a chain-of-thought explanation.
        """
        content = []
        if pil_images and re.search(r'<image\d+>', prompt_text):
            parts = re.split(r'(<image\d+>)', prompt_text)
            for part in parts:
                m = re.match(r'<image(\d+)>', part)
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
        return [
            {
                "role": "system",
                "content": "You are a helpful assistant. Answer with only a single letter: A, B, C, or D. Do not explain.",
            },
            {"role": "user", "content": content},
        ]

    def parse_response(self, raw_output: str) -> str:
        """Return generated text as-is (already decoded from generated tokens only)."""
        return raw_output.strip()

    def parse_answer(self, text: str, option_labels: list[str]) -> Optional[str]:
        """Extract the answer letter from Qwen3.5 output.

        Tries the base class logic first (handles direct "A" / "A." responses).
        Falls back to scanning the end of the text for the last standalone label
        in case the model reasoned before concluding with e.g. 'The answer is B'.
        """
        result = super().parse_answer(text, option_labels)
        if result is not None:
            return result

        # Scan sentence-by-sentence from the end for the last label mention
        sentences = re.split(r'[.!?\n]', text)
        for sentence in reversed(sentences):
            sentence = sentence.strip()
            for label in option_labels:
                if re.search(rf'\b{re.escape(label)}\b', sentence, re.IGNORECASE):
                    return label
        return None


@register("internvl35")
class InternVL35Model(VLMModel):
    """InternVL3.5 via HuggingFace-native format (OpenGVLab/InternVL3_5-{size}-HF).

    Uses AutoProcessor + AutoModelForImageTextToText with trust_remote_code=True.
    Images are loaded as PIL objects.  The Qwen-based chat template is applied by
    the processor, so the generate() pipeline mirrors Qwen35Model closely.
    """

    def __init__(
        self,
        model_name: str = "OpenGVLab/InternVL3_5-1B-HF",
        device: str = "cpu",
        dtype: str = "bfloat16",
        attn_implementation: str = "sdpa",
    ) -> None:
        super().__init__(model_name=model_name, device=device)
        self.dtype = DTYPE_MAP.get(dtype, torch.bfloat16)
        self.attn_implementation = attn_implementation

    def load(self) -> None:
        """Load InternVL3.5-HF model and processor from HuggingFace."""
        from transformers import AutoProcessor, AutoModelForImageTextToText

        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            padding_side="left",
            trust_remote_code=True,
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_name,
            dtype=self.dtype,
            attn_implementation=self.attn_implementation,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()

    def generate(
        self,
        prompt_text: str,
        image_paths: list[str] | None = None,
        max_new_tokens: int = 128,
    ) -> str:
        """Generate text using InternVL3.5-HF."""
        pil_images = self._load_pil_images(image_paths)
        messages = self._build_messages(prompt_text, pil_images)

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
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
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
            )

        generated_ids = output_ids[:, input_len:]
        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    def _load_pil_images(self, image_paths: list[str] | None) -> Optional[list]:
        """Load image paths as RGB PIL Images."""
        if not image_paths:
            return None
        return [Image.open(p).convert("RGB") for p in image_paths]

    def _build_messages(
        self,
        prompt_text: str,
        pil_images: Optional[list] = None,
    ) -> list[dict]:
        """Build InternVL3.5 chat messages with PIL images at <imageN> placeholders.

        A system prompt forces single-letter answers.  The same instruction is
        also appended inside the user turn because InternVL3.5 tends to ignore
        system-only instructions when images are present.
        """
        INSTRUCTION = "Reply with exactly one letter — A, B, C, or D — and nothing else."
        content = []
        if pil_images and re.search(r'<image\d+>', prompt_text):
            parts = re.split(r'(<image\d+>)', prompt_text)
            for part in parts:
                m = re.match(r'<image(\d+)>', part)
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
        content.append({"type": "text", "text": INSTRUCTION})
        return [
            {
                "role": "system",
                "content": "You are a helpful assistant. Answer with only a single letter: A, B, C, or D. Do not explain.",
            },
            {"role": "user", "content": content},
        ]

    def parse_response(self, raw_output: str) -> str:
        """Return generated text as-is (already decoded from generated tokens only)."""
        return raw_output.strip()

    def parse_answer(self, text: str, option_labels: list[str]) -> Optional[str]:
        """Extract the answer letter, with reverse-sentence fallback."""
        result = super().parse_answer(text, option_labels)
        if result is not None:
            return result
        sentences = re.split(r'[.!?\n]', text)
        for sentence in reversed(sentences):
            sentence = sentence.strip()
            for label in option_labels:
                if re.search(rf'\b{re.escape(label)}\b', sentence, re.IGNORECASE):
                    return label
        return None
