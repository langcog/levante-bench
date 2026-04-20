"""Gemini Pro via Google Generative Language REST API."""

import base64
import mimetypes
import os
import re
from pathlib import Path

import requests

from levante_bench.models.base import VLMModel
from levante_bench.models.registry import register


@register("gemini_pro")
class GeminiProModel(VLMModel):
    """Gemini Pro via Google Generative Language REST API."""

    API_BASE = "https://generativelanguage.googleapis.com/v1beta"

    def __init__(
        self,
        model_name: str = "gemini-2.5-pro",
        device: str = "cpu",
        api_key_env: str = "GEMINI_API_KEY",
        timeout_s: int = 120,
        retry_attempts: int = 4,
        max_output_tokens_cap: int = 2048,
        thinking_budget: int = 128,
    ) -> None:
        super().__init__(model_name=model_name, device=device)
        self.api_key_env = api_key_env
        self.timeout_s = timeout_s
        self.retry_attempts = max(1, int(retry_attempts))
        self.max_output_tokens_cap = max(1, int(max_output_tokens_cap))
        self.thinking_budget = max(0, int(thinking_budget))
        self.api_key: str | None = None

    def load(self) -> None:
        """Validate API key presence."""
        self.api_key = os.getenv(self.api_key_env)
        if not self.api_key:
            raise RuntimeError(
                f"{self.api_key_env} is not set. Export it before running gemini_pro."
            )

    def generate(
        self,
        prompt_text: str,
        image_paths: list[str] | None = None,
        max_new_tokens: int = 64,
    ) -> str:
        """Generate text using Gemini REST API."""
        if not self.api_key:
            raise RuntimeError("Gemini model not loaded. Call load() first.")

        parts = self._build_parts(prompt_text, image_paths)
        payload = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "temperature": 0.0,
                "maxOutputTokens": int(max_new_tokens),
                "thinkingConfig": {
                    "thinkingBudget": self.thinking_budget,
                },
            },
        }

        url = f"{self.API_BASE}/models/{self.model_name}:generateContent"
        for attempt in range(self.retry_attempts):
            response = requests.post(
                url,
                params={"key": self.api_key},
                json=payload,
                timeout=self.timeout_s,
            )
            if response.status_code != 200:
                raise RuntimeError(
                    f"Gemini API error {response.status_code}: {response.text[:500]}"
                )
            data = response.json()
            candidates = data.get("candidates") or []
            if not candidates:
                raise RuntimeError(f"Gemini returned no candidates: {data}")

            candidate = candidates[0]
            content = candidate.get("content", {})
            out_parts = content.get("parts", []) if isinstance(content, dict) else []
            text_chunks = [p.get("text", "") for p in out_parts if isinstance(p, dict)]
            text = "\n".join([t for t in text_chunks if t]).strip()
            if text:
                return text

            # Gemini 2.5 can consume all output budget in thoughts and emit no text.
            # Retry with a larger output budget before giving up.
            finish_reason = str(candidate.get("finishReason") or "").upper()
            current_budget = int(payload["generationConfig"]["maxOutputTokens"])
            if (
                finish_reason == "MAX_TOKENS"
                and current_budget < self.max_output_tokens_cap
                and attempt < self.retry_attempts - 1
            ):
                payload["generationConfig"]["maxOutputTokens"] = min(
                    self.max_output_tokens_cap,
                    current_budget * 2,
                )
                continue

            return ""

        return ""

    def _build_parts(self, prompt_text: str, image_paths: list[str] | None) -> list[dict]:
        """Build Gemini parts from prompt + optional image placeholders."""
        parts: list[dict] = []
        if image_paths and re.search(r"<image\d+>", prompt_text):
            chunks = re.split(r"(<image\d+>)", prompt_text)
            has_image0 = "<image0>" in prompt_text
            for chunk in chunks:
                m = re.match(r"<image(\d+)>", chunk)
                if m:
                    n = int(m.group(1))
                    idx = n if has_image0 else n - 1
                    if 0 <= idx < len(image_paths):
                        parts.append(self._image_part(image_paths[idx]))
                elif chunk.strip():
                    parts.append({"text": chunk.strip()})
            return parts

        if image_paths:
            for p in image_paths:
                parts.append(self._image_part(p))
        if prompt_text.strip():
            parts.append({"text": prompt_text.strip()})
        return parts

    def _image_part(self, image_path: str) -> dict:
        """Encode a local image path as Gemini inline_data part."""
        path = Path(image_path)
        raw = path.read_bytes()
        mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
        b64 = base64.b64encode(raw).decode("ascii")
        return {
            "inline_data": {
                "mime_type": mime_type,
                "data": b64,
            }
        }

    def parse_response(self, raw_output: str) -> str:
        return raw_output.strip()
