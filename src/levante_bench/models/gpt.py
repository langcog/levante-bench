"""GPT via OpenAI Responses REST API."""

import base64
import mimetypes
import os
import re
from pathlib import Path

import requests

from levante_bench.models.base import VLMModel
from levante_bench.models.registry import register


@register("gpt52")
@register("gpt53")
class GPT53Model(VLMModel):
    """GPT-5.3 via OpenAI Responses REST API."""

    API_BASE = "https://api.openai.com/v1"

    def __init__(
        self,
        model_name: str = "gpt-5.3",
        device: str = "cpu",
        api_key_env: str = "OPENAI_API_KEY",
        timeout_s: int = 120,
        api_base: str = API_BASE,
        retry_attempts: int = 5,
        retry_on_5xx: bool = True,
        max_output_tokens_min: int = 512,
        max_output_tokens_cap: int = 4096,
        reasoning_effort: str | None = None,
        text_verbosity: str | None = None,
    ) -> None:
        super().__init__(model_name=model_name, device=device)
        self.api_key_env = api_key_env
        self.timeout_s = timeout_s
        self.api_base = api_base
        self.retry_attempts = max(1, int(retry_attempts))
        self.retry_on_5xx = bool(retry_on_5xx)
        self.max_output_tokens_min = max(1, int(max_output_tokens_min))
        self.max_output_tokens_cap = max(
            self.max_output_tokens_min, int(max_output_tokens_cap)
        )
        self.reasoning_effort = (
            str(reasoning_effort).strip() if reasoning_effort else None
        )
        self.text_verbosity = str(text_verbosity).strip() if text_verbosity else None
        self.api_key: str | None = None

    def load(self) -> None:
        """Validate API key presence."""
        self.api_key = os.getenv(self.api_key_env)
        if not self.api_key:
            raise RuntimeError(
                f"{self.api_key_env} is not set. Export it before running gpt53."
            )

    def generate(
        self,
        prompt_text: str,
        image_paths: list[str] | None = None,
        max_new_tokens: int = 64,
    ) -> str:
        """Generate text using OpenAI Responses API."""
        if not self.api_key:
            raise RuntimeError("GPT53 model not loaded. Call load() first.")

        content = self._build_content(prompt_text, image_paths)
        payload = {
            "model": self.model_name,
            "input": [
                {
                    "role": "user",
                    "content": content,
                }
            ],
            "max_output_tokens": max(int(max_new_tokens), self.max_output_tokens_min),
        }
        if self.reasoning_effort:
            payload["reasoning"] = {"effort": self.reasoning_effort}
        if self.text_verbosity:
            payload["text"] = {"verbosity": self.text_verbosity}

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # Retry path for GPT-5.* cases where output tokens are consumed by
        # reasoning and no final text is emitted.
        for attempt in range(self.retry_attempts):
            response = requests.post(
                f"{self.api_base}/responses",
                headers=headers,
                json=payload,
                timeout=self.timeout_s,
            )
            if response.status_code != 200:
                # Transient server errors are common on long runs.
                if (
                    self.retry_on_5xx
                    and response.status_code >= 500
                    and attempt < self.retry_attempts - 1
                ):
                    continue
                raise RuntimeError(
                    f"OpenAI API error {response.status_code}: {response.text[:500]}"
                )
            data = response.json()
            text = self._extract_response_text(data)
            if text:
                return text

            incomplete = data.get("incomplete_details") or {}
            if incomplete.get("reason") == "max_output_tokens" and int(
                payload["max_output_tokens"]
            ) < self.max_output_tokens_cap:
                payload["max_output_tokens"] = min(
                    self.max_output_tokens_cap, int(payload["max_output_tokens"]) * 2
                )
                continue

            raise RuntimeError(f"OpenAI returned empty output: {data}")

        raise RuntimeError("OpenAI response retry exhausted.")

    def _build_content(self, prompt_text: str, image_paths: list[str] | None) -> list[dict]:
        """Build Responses API content from prompt + optional image placeholders."""
        content: list[dict] = []
        if image_paths and re.search(r"<image\d+>", prompt_text):
            chunks = re.split(r"(<image\d+>)", prompt_text)
            has_image0 = "<image0>" in prompt_text
            for chunk in chunks:
                m = re.match(r"<image(\d+)>", chunk)
                if m:
                    n = int(m.group(1))
                    idx = n if has_image0 else n - 1
                    if 0 <= idx < len(image_paths):
                        content.append(self._image_content(image_paths[idx]))
                elif chunk.strip():
                    content.append({"type": "input_text", "text": chunk.strip()})
            return content

        if image_paths:
            for p in image_paths:
                content.append(self._image_content(p))
        if prompt_text.strip():
            content.append({"type": "input_text", "text": prompt_text.strip()})
        return content

    def _image_content(self, image_path: str) -> dict:
        """Encode a local image path as a data URL for input_image."""
        path = Path(image_path)
        raw = path.read_bytes()
        mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
        b64 = base64.b64encode(raw).decode("ascii")
        return {
            "type": "input_image",
            "image_url": f"data:{mime_type};base64,{b64}",
        }

    def _extract_response_text(self, data: dict) -> str:
        """Extract plain text from Responses API payload."""
        if isinstance(data.get("output_text"), str) and data.get("output_text"):
            return data["output_text"].strip()

        chunks: list[str] = []
        for item in data.get("output", []) or []:
            for part in item.get("content", []) or []:
                if part.get("type") in {"output_text", "text"}:
                    txt = part.get("text")
                    if isinstance(txt, str) and txt.strip():
                        chunks.append(txt.strip())
        return "\n".join(chunks).strip()

    def parse_response(self, raw_output: str) -> str:
        return raw_output.strip()
