"""CLIP image-text similarity adapter.

CLIP has no text decoder, so it cannot use the generate-then-parse path used by
generative VLMs. Instead, for label-format trials whose options are images we
score each option image against either:

  * the trial's context image (image-image similarity), when one is provided
    (e.g. mental-rotation, TROG with a stem image), or
  * the trial's text prompt (text-image similarity), otherwise (e.g. vocab,
    synthetic-vocab).

The argmax option is returned as ``predicted_label``. Numeric / slider tasks
and tasks without ``option_image_paths`` are reported as unscored
(``predicted_label=None``, ``is_correct=False``) with a clear ``parse_method``
tag so downstream summaries can distinguish "wrong" from "not applicable".
"""

from __future__ import annotations

import re
import sys
import time
from typing import Optional

import torch

from levante_bench.models._common import DTYPE_MAP, load_pil_images
from levante_bench.models.base import VLMModel
from levante_bench.models.registry import register


_IMAGE_PLACEHOLDER_RE = re.compile(r"<image\d+>")


def _clean_text_for_clip(prompt: str) -> str:
    """Strip ``<imageN>`` placeholders and collapse whitespace.

    The benchmark's prompt templates interleave ``<imageN>`` markers that bind
    option images to labels. CLIP's text encoder has no concept of images in
    text, so we drop the markers and let the model see only the natural-
    language portion of the prompt.
    """
    cleaned = _IMAGE_PLACEHOLDER_RE.sub(" ", str(prompt or ""))
    return re.sub(r"\s+", " ", cleaned).strip()


@register("clip_base")
class CLIPSimilarityModel(VLMModel):
    """CLIP-style zero-shot multiple-choice via image-text similarity.

    Defaults to ``openai/clip-vit-base-patch32``; pass a different ``hf_name``
    in the model config (e.g. ``openai/clip-vit-large-patch14`` or
    ``laion/CLIP-ViT-H-14-laion2B-s32B-b79K``) to swap backbones. The text
    encoder of stock OpenAI CLIP is English-only with a 77-token limit, both
    of which are documented limitations rather than bugs of this adapter.
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: str = "cpu",
        dtype: str = "float32",
        prefer_context_image: bool = True,
        **_: object,
    ) -> None:
        super().__init__(model_name=model_name, device=device)
        self.dtype = DTYPE_MAP.get(str(dtype).lower(), torch.float32)
        self.prefer_context_image = bool(prefer_context_image)
        self.use_json_format = False  # CLIP does not consume format prefixes
        self.last_generation_metadata: dict = {}

    def load(self) -> None:
        try:
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:  # pragma: no cover - exercised only if deps missing
            raise ImportError(
                "CLIP requires `transformers`. Install with "
                "`pip install -r requirements-transformers.txt`."
            ) from exc

        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        try:
            self.model = CLIPModel.from_pretrained(self.model_name, dtype=self.dtype)
        except TypeError:
            # Older transformers versions still expect torch_dtype.
            self.model = CLIPModel.from_pretrained(
                self.model_name, torch_dtype=self.dtype
            )
        self.model = self.model.to(self.device)
        self.model.eval()
        print(
            f"[clip] loaded {self.model_name!r} on {self.device!r} dtype={self.dtype}",
            file=sys.stderr,
        )

    # CLIP cannot generate text. The benchmark runner invokes
    # ``evaluate_trial`` (overridden below), so ``generate`` is never reached
    # on the standard path; we keep an explicit error to surface misuse.
    def generate(
        self,
        prompt_text: str,
        image_paths: list[str] | None = None,
        max_new_tokens: int = 64,
    ) -> str:
        raise NotImplementedError(
            "CLIPSimilarityModel scores choices via similarity; "
            "use evaluate_trial(...) instead of generate(...)."
        )

    def evaluate_trial(self, trial: dict) -> dict:
        """Score the trial's option images and pick the argmax label.

        Returns a result dict whose schema matches generative models so
        downstream CSV / NPY writers and human-comparison annotators do not
        need to special-case CLIP.
        """
        answer_format = str(trial.get("answer_format", "label")).strip().lower()
        option_labels = [str(l).upper() for l in trial.get("option_labels", [])]
        option_image_paths = list(trial.get("option_image_paths") or [])
        context_image_paths = list(trial.get("context_image_paths") or [])
        correct_label = trial.get("correct_label")

        if answer_format != "label":
            return self._unscored_result(
                trial=trial,
                reason=f"clip_unsupported_format:{answer_format}",
                answer_format=answer_format,
            )
        if not option_image_paths or not option_labels:
            return self._unscored_result(
                trial=trial,
                reason="clip_no_option_images",
                answer_format=answer_format,
            )

        option_images = load_pil_images(option_image_paths)
        if not option_images:
            return self._unscored_result(
                trial=trial,
                reason="clip_failed_to_load_option_images",
                answer_format=answer_format,
            )

        use_context = self.prefer_context_image and bool(context_image_paths)
        context_images = (
            load_pil_images(context_image_paths) if use_context else None
        )

        start = time.perf_counter()
        with torch.no_grad():
            option_feats = self._encode_images(option_images)
            if use_context and context_images:
                # Image-image scoring: average context-image features as the
                # query (handles single- or multi-image stems uniformly).
                query_feats = self._encode_images(context_images).mean(
                    dim=0, keepdim=True
                )
                query_kind = "context_image"
                clip_query_text = ""
            else:
                clip_query_text = _clean_text_for_clip(trial.get("prompt", ""))
                if not clip_query_text:
                    return self._unscored_result(
                        trial=trial,
                        reason="clip_empty_text_query",
                        answer_format=answer_format,
                    )
                query_feats = self._encode_text(clip_query_text)
                query_kind = "prompt_text"

            # Cosine similarity in shared CLIP space.
            sims = (query_feats @ option_feats.T).squeeze(0)
        elapsed = time.perf_counter() - start

        sims_list = [float(x) for x in sims.detach().cpu().tolist()]
        best_idx = int(max(range(len(sims_list)), key=lambda i: sims_list[i]))
        predicted_label = option_labels[best_idx] if best_idx < len(option_labels) else None
        is_correct = bool(predicted_label is not None and predicted_label == correct_label)

        scores_payload = ", ".join(
            f"{label}={score:.4f}"
            for label, score in zip(option_labels, sims_list)
        )

        self.last_generation_metadata = {
            "api_provider": "clip_similarity",
            "api_attempts": 1,
            "api_response_status": "ok",
            "api_finish_reason": query_kind,
        }

        return {
            "trial_id": trial["trial_id"],
            "item_uid": trial["item_uid"],
            "generated_text": scores_payload,
            "predicted_label": predicted_label,
            "reason": f"clip_argmax({query_kind})",
            "parse_method": "clip_similarity",
            "parse_confidence": "high",
            "parse_raw_candidate": predicted_label or "",
            "correct_label": correct_label,
            "is_correct": is_correct,
            "options": trial.get("options", []),
            "option_labels": trial.get("option_labels", []),
            "clip_query_kind": query_kind,
            "clip_query_text": clip_query_text,
            "clip_scores": sims_list,
            "generation_time_s": elapsed,
        }

    def _encode_images(self, pil_images: list) -> torch.Tensor:
        inputs = self.processor(images=pil_images, return_tensors="pt").to(self.device)
        feats = self._unwrap_feats(self.model.get_image_features(**inputs))
        return self._l2_normalize(feats)

    def _encode_text(self, text: str) -> torch.Tensor:
        inputs = self.processor(
            text=[text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        ).to(self.device)
        feats = self._unwrap_feats(self.model.get_text_features(**inputs))
        return self._l2_normalize(feats)

    @staticmethod
    def _unwrap_feats(output) -> torch.Tensor:
        """Return a tensor from CLIP feature outputs across transformers versions.

        transformers <5 returned a plain tensor from
        ``get_image_features`` / ``get_text_features``; v5+ returns a
        ``BaseModelOutputWithPooling`` whose ``pooler_output`` holds the
        projected embedding.
        """
        if isinstance(output, torch.Tensor):
            return output
        for attr in ("pooler_output", "image_embeds", "text_embeds", "last_hidden_state"):
            value = getattr(output, attr, None)
            if isinstance(value, torch.Tensor):
                return value
        raise TypeError(
            f"Unexpected CLIP feature output type: {type(output).__name__}"
        )

    @staticmethod
    def _l2_normalize(feats: torch.Tensor) -> torch.Tensor:
        return feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    def _unscored_result(
        self,
        trial: dict,
        reason: str,
        answer_format: str,
    ) -> dict:
        """Skeleton result for trials CLIP cannot score (e.g. numeric tasks)."""
        self.last_generation_metadata = {
            "api_provider": "clip_similarity",
            "api_attempts": 0,
            "api_response_status": "skipped",
            "api_finish_reason": reason,
        }
        base: dict = {
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
        }
        return base
