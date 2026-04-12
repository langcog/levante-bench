"""Hosted Hugging Face Inference Providers model adapter."""

from __future__ import annotations

import base64
import io
import mimetypes
import os
import re
from pathlib import Path

import requests
from PIL import Image

from levante_bench.models._common import (
    parse_answer_result_with_fallback,
    parse_answer_with_fallback,
)
from levante_bench.models.base import ParseResult, VLMModel
from levante_bench.models.registry import register

_SYSTEM_PROMPT = (
    "You are a helpful assistant. "
    "Answer with only a single letter: A, B, C, or D. Do not explain."
)


@register("hf_hosted")
@register("qwen25vl_32b_hf")
@register("qwen25vl_72b_hf")
@register("qwen3vl_30b_hf")
@register("qwen3vl_235b_hf")
@register("aya_vision_32b_hf")
class HFHostedModel(VLMModel):
    """OpenAI-compatible chat API adapter for Hugging Face hosted models."""

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        api_key_env: str = "HF_TOKEN",
        api_base: str = "https://router.huggingface.co/v1",
        timeout_s: int = 180,
        retry_attempts: int = 4,
        max_output_tokens_cap: int = 1024,
        max_image_edge: int = 768,
        jpeg_quality: int = 75,
        max_prompt_chars: int = 12000,
    ) -> None:
        super().__init__(model_name=model_name, device=device)
        self.api_key_env = api_key_env
        self.api_base = api_base.rstrip("/")
        self.timeout_s = int(timeout_s)
        self.retry_attempts = max(1, int(retry_attempts))
        self.max_output_tokens_cap = max(1, int(max_output_tokens_cap))
        self.max_image_edge = max(256, int(max_image_edge))
        self.jpeg_quality = max(40, min(95, int(jpeg_quality)))
        self.max_prompt_chars = max(2048, int(max_prompt_chars))
        self.api_key: str | None = None

    def load(self) -> None:
        self.api_key = (
            os.getenv(self.api_key_env)
            or os.getenv("HUGGINGFACEHUB_API_TOKEN")
            or os.getenv("HUGGING_FACE_HUB_TOKEN")
        )
        if not self.api_key:
            raise RuntimeError(
                f"{self.api_key_env} is not set (or HUGGINGFACEHUB_API_TOKEN). "
                "Export one of them before running hosted HF models."
            )

    def generate(
        self,
        prompt_text: str,
        image_paths: list[str] | None = None,
        max_new_tokens: int = 64,
    ) -> str:
        if not self.api_key:
            raise RuntimeError("HF hosted model not loaded. Call load() first.")

        messages = self._build_messages(prompt_text, image_paths)
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": min(max(1, int(max_new_tokens)), self.max_output_tokens_cap),
            "temperature": 0.0,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self.api_base}/chat/completions"

        for attempt in range(self.retry_attempts):
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=self.timeout_s,
            )
            if response.status_code != 200:
                # Retry once with a shorter prompt when provider rejects context length.
                if (
                    response.status_code == 400
                    and "maximum model length" in response.text
                    and attempt < self.retry_attempts - 1
                ):
                    payload["messages"] = self._build_messages(
                        self._truncate_prompt(prompt_text),
                        image_paths,
                    )
                    continue
                if response.status_code >= 500 and attempt < self.retry_attempts - 1:
                    continue
                raise RuntimeError(
                    f"Hugging Face API error {response.status_code}: {response.text[:500]}"
                )

            data = response.json()
            choices = data.get("choices") or []
            if not choices:
                raise RuntimeError(f"Hugging Face returned no choices: {data}")
            message = choices[0].get("message") or {}
            content = message.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                text_chunks = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_chunks.append(str(item.get("text", "")))
                joined = "\n".join(t for t in text_chunks if t).strip()
                if joined:
                    return joined
            return ""
        return ""

    def _build_messages(
        self,
        prompt_text: str,
        image_paths: list[str] | None = None,
    ) -> list[dict]:
        content: list[dict] = []
        if image_paths and re.search(r"<image\d+>", prompt_text):
            parts = re.split(r"(<image\d+>)", prompt_text)
            has_image0 = "<image0>" in prompt_text
            for part in parts:
                m = re.match(r"<image(\d+)>", part)
                if m:
                    n = int(m.group(1))
                    idx = n if has_image0 else n - 1
                    if 0 <= idx < len(image_paths):
                        content.append(self._image_content(image_paths[idx]))
                elif part.strip():
                    content.append({"type": "text", "text": part.strip()})
        else:
            if image_paths:
                for path in image_paths:
                    content.append(self._image_content(path))
            if prompt_text.strip():
                content.append({"type": "text", "text": prompt_text.strip()})
        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]

    def _truncate_prompt(self, prompt_text: str) -> str:
        text = prompt_text.strip()
        if len(text) <= self.max_prompt_chars:
            return text
        head = max(512, self.max_prompt_chars // 3)
        tail = max(512, self.max_prompt_chars - head - 32)
        return (
            text[:head]
            + "\n...[prompt truncated for hosted context limit]...\n"
            + text[-tail:]
        )

    def _image_content(self, image_path: str) -> dict:
        path = Path(image_path)
        mime_type = mimetypes.guess_type(path.name)[0] or "image/png"

        try:
            img = Image.open(path).convert("RGB")
            if max(img.size) > self.max_image_edge:
                img.thumbnail((self.max_image_edge, self.max_image_edge), Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=self.jpeg_quality, optimize=True)
            raw = buf.getvalue()
            mime_type = "image/jpeg"
        except Exception:
            raw = path.read_bytes()

        b64 = base64.b64encode(raw).decode("ascii")
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{mime_type};base64,{b64}"},
        }

    def parse_response(self, raw_output: str) -> str:
        return raw_output.strip()

    def parse_answer(
        self, text: str, option_labels: list[str]
    ) -> tuple[str | None, str]:
        return parse_answer_with_fallback(self, text, option_labels)

    def parse_answer_result(self, text: str, option_labels: list[str]) -> ParseResult:
        return parse_answer_result_with_fallback(self, text, option_labels)
