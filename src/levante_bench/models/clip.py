"""CLIP similarity adapter for LEVANTE benchmark evaluation.

This adapter uses CLIP's shared embedding space for multiple-choice selection:

- image options: score prompt text (or context image) against option images
- text options: score prompt text (or context image) against option texts

CLIP remains a retrieval-style model (no generative decode), so the adapter
returns direct argmax labels from similarity scores.
"""

from __future__ import annotations

import re
import time
from typing import Any

import torch

from levante_bench.models._common import DTYPE_MAP, load_pil_images
from levante_bench.models.base import VLMModel
from levante_bench.models.registry import register

_IMAGE_PLACEHOLDER_RE = re.compile(r"<image\d+>")
_OPTION_LINE_RE = re.compile(r"^[A-H]\s*[\)\.:]\s*")


def _clean_text_for_clip(prompt: str) -> str:
    """Drop image placeholders and collapse whitespace."""
    cleaned = _IMAGE_PLACEHOLDER_RE.sub(" ", str(prompt or ""))
    return re.sub(r"\s+", " ", cleaned).strip()


def _query_text_from_prompt(prompt: str) -> str:
    """Extract query-like text from benchmark prompt.

    Many prompts include enumerated answer options (A/B/C/...) inline. For
    text-option similarity we strip those lines so query text does not simply
    echo candidate choices.
    """
    text = _clean_text_for_clip(prompt)
    if not text:
        return ""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return text
    keep = [ln for ln in lines if not _OPTION_LINE_RE.match(ln)]
    # If filtering removed everything, fall back to original cleaned text.
    return " ".join(keep).strip() or text


@register("clip_base")
class CLIPSimilarityModel(VLMModel):
    """CLIP similarity-based evaluator with text-option fallback."""

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: str = "cpu",
        dtype: str = "float32",
        prefer_context_image: bool = True,
        enable_text_option_fallback: bool = True,
        **_: Any,
    ) -> None:
        super().__init__(model_name=model_name, device=device)
        self.dtype = DTYPE_MAP.get(str(dtype).lower(), torch.float32)
        self.prefer_context_image = bool(prefer_context_image)
        self.enable_text_option_fallback = bool(enable_text_option_fallback)
        self.use_json_format = False
        self.last_generation_metadata: dict[str, Any] = {}

    def load(self) -> None:
        try:
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:
            raise ImportError(
                "CLIP requires transformers; install requirements-transformers.txt"
            ) from exc

        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        try:
            self.model = CLIPModel.from_pretrained(self.model_name, dtype=self.dtype)
        except TypeError:
            # Backward compatibility with older transformers argument naming.
            self.model = CLIPModel.from_pretrained(
                self.model_name, torch_dtype=self.dtype
            )
        self.model = self.model.to(self.device)
        self.model.eval()

    def generate(
        self,
        prompt_text: str,
        image_paths: list[str] | None = None,
        max_new_tokens: int = 64,
    ) -> str:
        raise NotImplementedError(
            "CLIPSimilarityModel uses evaluate_trial similarity scoring, not generate()."
        )

    def evaluate_trial(self, trial: dict) -> dict:
        answer_format = str(trial.get("answer_format", "label")).strip().lower()
        if answer_format != "label":
            return self._unscored_result(trial, reason=f"clip_unsupported_format:{answer_format}")

        option_labels = [str(x).upper() for x in (trial.get("option_labels") or [])]
        if not option_labels:
            return self._unscored_result(trial, reason="clip_missing_option_labels")

        option_image_paths = list(trial.get("option_image_paths") or [])
        option_texts = [str(x).strip() for x in (trial.get("options") or [])]
        context_image_paths = list(trial.get("context_image_paths") or [])

        if option_image_paths:
            return self._score_image_options(
                trial=trial,
                option_labels=option_labels,
                option_image_paths=option_image_paths,
                context_image_paths=context_image_paths,
            )

        if self.enable_text_option_fallback and option_texts:
            return self._score_text_options(
                trial=trial,
                option_labels=option_labels,
                option_texts=option_texts,
                context_image_paths=context_image_paths,
            )

        return self._unscored_result(trial, reason="clip_no_option_images")

    def _score_image_options(
        self,
        *,
        trial: dict,
        option_labels: list[str],
        option_image_paths: list[str],
        context_image_paths: list[str],
    ) -> dict:
        option_images = load_pil_images(option_image_paths)
        if not option_images:
            return self._unscored_result(trial, reason="clip_failed_to_load_option_images")

        use_context = self.prefer_context_image and bool(context_image_paths)
        context_images = load_pil_images(context_image_paths) if use_context else None

        start = time.perf_counter()
        with torch.no_grad():
            option_feats = self._encode_images(option_images)
            if use_context and context_images:
                query_feats = self._encode_images(context_images).mean(dim=0, keepdim=True)
                query_kind = "context_image"
            else:
                query_text = _query_text_from_prompt(trial.get("prompt", ""))
                if not query_text:
                    return self._unscored_result(trial, reason="clip_empty_text_query")
                query_feats = self._encode_text([query_text])
                query_kind = "prompt_text"
            sims = (query_feats @ option_feats.T).squeeze(0)
        elapsed = time.perf_counter() - start
        return self._build_label_result(
            trial=trial,
            option_labels=option_labels,
            sims=sims,
            query_kind=query_kind,
            elapsed=elapsed,
            parse_method="clip_similarity_image_options",
        )

    def _score_text_options(
        self,
        *,
        trial: dict,
        option_labels: list[str],
        option_texts: list[str],
        context_image_paths: list[str],
    ) -> dict:
        use_context = self.prefer_context_image and bool(context_image_paths)
        context_images = load_pil_images(context_image_paths) if use_context else None

        start = time.perf_counter()
        with torch.no_grad():
            option_feats = self._encode_text(option_texts)
            if use_context and context_images:
                query_feats = self._encode_images(context_images).mean(dim=0, keepdim=True)
                query_kind = "context_image_to_text_options"
            else:
                query_text = _query_text_from_prompt(trial.get("prompt", ""))
                if not query_text:
                    return self._unscored_result(trial, reason="clip_empty_text_query")
                query_feats = self._encode_text([query_text])
                query_kind = "prompt_text_to_text_options"
            sims = (query_feats @ option_feats.T).squeeze(0)
        elapsed = time.perf_counter() - start
        return self._build_label_result(
            trial=trial,
            option_labels=option_labels,
            sims=sims,
            query_kind=query_kind,
            elapsed=elapsed,
            parse_method="clip_similarity_text_options",
        )

    def _build_label_result(
        self,
        *,
        trial: dict,
        option_labels: list[str],
        sims: torch.Tensor,
        query_kind: str,
        elapsed: float,
        parse_method: str,
    ) -> dict:
        scores = [float(v) for v in sims.detach().cpu().tolist()]
        best_idx = int(max(range(len(scores)), key=lambda i: scores[i])) if scores else -1
        predicted_label = option_labels[best_idx] if (0 <= best_idx < len(option_labels)) else None
        correct_label = trial.get("correct_label")
        is_correct = bool(predicted_label and predicted_label == correct_label)

        self.last_generation_metadata = {
            "api_provider": "clip_similarity",
            "api_attempts": 1,
            "api_response_status": "ok",
            "api_finish_reason": query_kind,
        }

        generated = ", ".join(f"{lab}={score:.4f}" for lab, score in zip(option_labels, scores))
        return {
            "trial_id": trial["trial_id"],
            "item_uid": trial["item_uid"],
            "generated_text": generated,
            "predicted_label": predicted_label,
            "reason": f"clip_argmax({query_kind})",
            "parse_method": parse_method,
            "parse_confidence": "high",
            "parse_raw_candidate": predicted_label or "",
            "correct_label": correct_label,
            "is_correct": is_correct,
            "options": trial.get("options", []),
            "option_labels": trial.get("option_labels", []),
            "clip_scores": scores,
            "generation_time_s": elapsed,
            **self.last_generation_metadata,
        }

    def _encode_images(self, pil_images: list) -> torch.Tensor:
        inputs = self.processor(images=pil_images, return_tensors="pt").to(self.device)
        feats = self._unwrap_feats(self.model.get_image_features(**inputs))
        return self._l2_normalize(feats)

    def _encode_text(self, texts: list[str]) -> torch.Tensor:
        inputs = self.processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        ).to(self.device)
        feats = self._unwrap_feats(self.model.get_text_features(**inputs))
        return self._l2_normalize(feats)

    @staticmethod
    def _unwrap_feats(output) -> torch.Tensor:
        if isinstance(output, torch.Tensor):
            return output
        for attr in ("pooler_output", "image_embeds", "text_embeds", "last_hidden_state"):
            val = getattr(output, attr, None)
            if isinstance(val, torch.Tensor):
                return val
        raise TypeError(f"Unexpected CLIP feature output type: {type(output).__name__}")

    @staticmethod
    def _l2_normalize(feats: torch.Tensor) -> torch.Tensor:
        return feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    def _unscored_result(self, trial: dict, *, reason: str) -> dict:
        self.last_generation_metadata = {
            "api_provider": "clip_similarity",
            "api_attempts": 0,
            "api_response_status": "skipped",
            "api_finish_reason": reason,
        }
        return {
            "trial_id": trial["trial_id"],
            "item_uid": trial["item_uid"],
            "generated_text": "",
            "predicted_label": None,
            "predicted_value": None,
            "predicted_slider_position": None,
            "reason": reason,
            "parse_method": reason,
            "parse_confidence": "none",
            "parse_raw_candidate": "",
            "correct_label": trial.get("correct_label"),
            "target_value": trial.get("target_value"),
            "slider_tolerance": trial.get("slider_tolerance"),
            "is_correct": False,
            "options": trial.get("options", []),
            "option_labels": trial.get("option_labels", []),
            **self.last_generation_metadata,
        }
