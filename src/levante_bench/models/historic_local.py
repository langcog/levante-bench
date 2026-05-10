"""Local historic VLM adapters (LLaVA-1.5, CogVLM, OpenFlamingo)."""

from __future__ import annotations

import re
from typing import Any, Optional

import torch
from PIL import Image, ImageDraw

from levante_bench.models._common import (
    DTYPE_MAP,
    build_pil_content,
    load_pil_images,
    should_fallback_to_sdpa,
    warn_attn_fallback,
)
from levante_bench.models.base import SYSTEM_PROMPT, VLMModel
from levante_bench.models.registry import register


@register("llava15_13b")
@register("cogvlm")
@register("openflamingo9b")
class HistoricLocalVLMModel(VLMModel):
    """Best-effort local HuggingFace adapter for older open VLM checkpoints.

    This adapter intentionally uses ``trust_remote_code=True`` and supports
    multiple loading paths, because these checkpoints differ in how they expose
    processors/chat templates across Transformers releases.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        dtype: str = "bfloat16",
        attn_implementation: str = "flash_attention_2",
        trust_remote_code: bool = True,
        device_map: str | None = None,
        max_image_edge: int | None = None,
        generation: dict[str, Any] | None = None,
        tokenizer_hf_name: str | None = None,
    ) -> None:
        super().__init__(model_name=model_name, device=device)
        self.dtype = DTYPE_MAP.get(str(dtype), torch.bfloat16)
        self.attn_implementation = str(attn_implementation or "flash_attention_2")
        self.trust_remote_code = bool(trust_remote_code)
        self.device_map = str(device_map).strip() if device_map else None
        self.max_image_edge = int(max_image_edge) if max_image_edge else None
        self.generation_defaults = dict(generation) if isinstance(generation, dict) else {}
        self.tokenizer_hf_name = str(tokenizer_hf_name).strip() if tokenizer_hf_name else None
        self.tokenizer = None

    def _load_processor(self) -> Any:
        from transformers import AutoProcessor

        kwargs = {"trust_remote_code": self.trust_remote_code}
        try:
            return AutoProcessor.from_pretrained(self.model_name, **kwargs)
        except TypeError:
            kwargs.pop("trust_remote_code", None)
            return AutoProcessor.from_pretrained(self.model_name, **kwargs)

    def _load_tokenizer(self) -> Any:
        from transformers import AutoTokenizer

        lower_name = self.model_name.lower()
        if "cogvlm" in lower_name:
            # Prefer AutoTokenizer so remote-code tokenizers can initialize when
            # available. Fall back to slow/legacy modes for older checkpoints.
            last_exc: Exception | None = None
            tokenizer_sources: list[str] = [self.model_name]
            if self.tokenizer_hf_name:
                tokenizer_sources.append(self.tokenizer_hf_name)
            # CogVLM chat checkpoints are often based on Vicuna tokenizers.
            tokenizer_sources.append("lmsys/vicuna-7b-v1.5")

            for source in tokenizer_sources:
                for kwargs in (
                    {"trust_remote_code": self.trust_remote_code, "use_fast": True},
                    {"trust_remote_code": self.trust_remote_code, "use_fast": False},
                    {"trust_remote_code": self.trust_remote_code, "use_fast": False, "legacy": True},
                ):
                    try:
                        return AutoTokenizer.from_pretrained(source, **kwargs)
                    except Exception as exc:
                        last_exc = exc
            if last_exc is not None:
                raise last_exc

        if "openflamingo" in lower_name:
            # OpenFlamingo checkpoints can expose incomplete tokenizer metadata
            # under their primary repo. Prefer a known GPT-NeoX tokenizer source.
            last_exc: Exception | None = None
            tokenizer_sources: list[str] = []
            if self.tokenizer_hf_name:
                tokenizer_sources.append(self.tokenizer_hf_name)
            tokenizer_sources.extend(
                [
                    "EleutherAI/gpt-neox-20b",
                    self.model_name,
                ]
            )
            for source in tokenizer_sources:
                for kwargs in (
                    {"trust_remote_code": self.trust_remote_code, "use_fast": False},
                    {"trust_remote_code": self.trust_remote_code, "use_fast": True},
                    {"trust_remote_code": self.trust_remote_code, "use_fast": False, "legacy": True},
                ):
                    try:
                        return AutoTokenizer.from_pretrained(source, **kwargs)
                    except Exception as exc:
                        last_exc = exc
            if last_exc is not None:
                raise last_exc

        kwargs = {"trust_remote_code": self.trust_remote_code, "use_fast": False}
        try:
            return AutoTokenizer.from_pretrained(self.model_name, **kwargs)
        except TypeError:
            kwargs.pop("trust_remote_code", None)
            try:
                return AutoTokenizer.from_pretrained(self.model_name, **kwargs)
            except TypeError:
                # Very old tokenizers may not accept use_fast in this code path.
                kwargs.pop("use_fast", None)
                return AutoTokenizer.from_pretrained(self.model_name, **kwargs)

    def _load_with_model_cls(self, model_cls: Any, attn_impl: str) -> Any:
        base_kwargs: dict[str, Any] = {
            "low_cpu_mem_usage": True,
            "trust_remote_code": self.trust_remote_code,
        }
        if self.device_map:
            base_kwargs["device_map"] = self.device_map

        # Some model classes (e.g., LlavaForConditionalGeneration on certain
        # transformers versions) reject attn_implementation/dtype kwargs.
        attempt_kwargs: list[dict[str, Any]] = []

        kw = dict(base_kwargs)
        kw["attn_implementation"] = attn_impl
        kw["torch_dtype"] = self.dtype
        attempt_kwargs.append(kw)

        kw = dict(base_kwargs)
        kw["attn_implementation"] = attn_impl
        kw["dtype"] = self.dtype
        attempt_kwargs.append(kw)

        kw = dict(base_kwargs)
        kw["torch_dtype"] = self.dtype
        attempt_kwargs.append(kw)

        kw = dict(base_kwargs)
        kw["dtype"] = self.dtype
        attempt_kwargs.append(kw)

        attempt_kwargs.append(dict(base_kwargs))

        last_exc: Exception | None = None
        for kwargs in attempt_kwargs:
            try:
                model = model_cls.from_pretrained(self.model_name, **kwargs)
                return model if self.device_map else model.to(self.device)
            except Exception as exc:
                last_exc = exc
                continue
        if last_exc is None:
            raise RuntimeError(f"Failed to load model class {model_cls} for '{self.model_name}'.")
        raise last_exc

    def _load_model(self, attn_impl: str) -> Any:
        from transformers import AutoModelForCausalLM
        import transformers as tfm

        lower_name = self.model_name.lower()
        if "llava" in lower_name:
            # LLaVA-1.5 is not supported by older AutoModelFor* multimodal maps.
            llava_cls = getattr(tfm, "LlavaForConditionalGeneration", None)
            if llava_cls is not None:
                try:
                    return self._load_with_model_cls(llava_cls, attn_impl)
                except Exception:
                    pass

        if "cogvlm" in lower_name:
            # CogVLM chat model uses custom causal-lm remote code.
            try:
                return self._load_with_model_cls(AutoModelForCausalLM, attn_impl)
            except Exception:
                pass

        last_exc: Exception | None = None
        model_classes: list[Any] = []
        image_text_cls = getattr(tfm, "AutoModelForImageTextToText", None)
        if image_text_cls is not None:
            model_classes.append(image_text_cls)
        vision2seq_cls = getattr(tfm, "AutoModelForVision2Seq", None)
        if vision2seq_cls is not None:
            model_classes.append(vision2seq_cls)
        model_classes.append(AutoModelForCausalLM)

        for model_cls in model_classes:
            try:
                return self._load_with_model_cls(model_cls, attn_impl)
            except Exception as exc:  # noqa: PERF203 - keep concrete failures for fallback ladder
                last_exc = exc
        if last_exc is None:
            raise RuntimeError(f"Could not load model '{self.model_name}'.")
        raise last_exc

    def load(self) -> None:
        processor_exc: Exception | None = None
        try:
            self.processor = self._load_processor()
        except Exception as exc:
            # CogVLM chat checkpoints often ship custom model code without an
            # AutoProcessor class. In that case we fall back to tokenizer-only.
            self.processor = None
            processor_exc = exc
        self.tokenizer = getattr(self.processor, "tokenizer", None) if self.processor else None
        if self.tokenizer is None:
            try:
                self.tokenizer = self._load_tokenizer()
            except Exception as exc:
                # Some checkpoints only work through processor-managed tokenization.
                # Keep loading unless we actually require tokenizer fallback paths.
                if self.processor is None:
                    raise RuntimeError(
                        f"Failed to load tokenizer for '{self.model_name}': {exc}"
                    ) from exc
        requested_attn = self.attn_implementation
        try:
            self.model = self._load_model(requested_attn)
        except Exception as exc:
            if not should_fallback_to_sdpa(requested_attn, exc):
                raise
            warn_attn_fallback(self.model_name, requested_attn, exc)
            self.attn_implementation = "sdpa"
            self.model = self._load_model("sdpa")
        self.model.eval()
        self._patch_generation_compat()
        if self.processor is None and processor_exc is not None:
            print(
                f"[{self.model_name}] processor unavailable ({type(processor_exc).__name__}); "
                "using tokenizer/build_conversation_input_ids fallback."
            )

    def _patch_generation_compat(self) -> None:
        """Patch model methods for remote-code / transformers compatibility."""
        if self.model is None:
            return
        lower_name = self.model_name.lower()
        if "cogvlm" not in lower_name:
            return
        # Some CogVLM remote-code revisions rely on a helper that is provided by
        # newer/other transformers generation mixins. Add a safe fallback so
        # generation does not crash when the helper is absent.
        if not hasattr(self.model, "_extract_past_from_model_output"):
            def _extract_past_from_model_output(_self, outputs, *args, **kwargs):
                if outputs is None:
                    return None
                if isinstance(outputs, dict):
                    return outputs.get("past_key_values")
                if isinstance(outputs, (list, tuple)) and len(outputs) > 1:
                    return outputs[1]
                return getattr(outputs, "past_key_values", None)

            setattr(self.model, "_extract_past_from_model_output", _extract_past_from_model_output.__get__(self.model, type(self.model)))

    def _build_messages(
        self,
        prompt_text: str,
        pil_images: Optional[list] = None,
    ) -> list[dict]:
        content = build_pil_content(prompt_text, pil_images)
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]

    def _pack_multi_images_for_cogvlm(
        self,
        prompt_text: str,
        pil_images: list[Image.Image],
    ) -> tuple[str, list[Image.Image]]:
        """Collapse multi-image trials to one panel image for CogVLM."""
        if len(pil_images) <= 1:
            return prompt_text, pil_images

        n = len(pil_images)
        cols = 2
        rows = (n + cols - 1) // cols
        cell = min(int(self.max_image_edge or 448), 512)
        pad = 8
        canvas = Image.new("RGB", (cols * cell, rows * cell), color=(240, 240, 240))
        draw = ImageDraw.Draw(canvas)

        for idx, img in enumerate(pil_images):
            tile = img.copy()
            tile.thumbnail((cell - (2 * pad), cell - (2 * pad)), Image.Resampling.LANCZOS)
            ox = (idx % cols) * cell + (cell - tile.width) // 2
            oy = (idx // cols) * cell + (cell - tile.height) // 2
            canvas.paste(tile, (ox, oy))
            # Index badge helps map prompt placeholders to merged panels.
            bx0, by0 = (idx % cols) * cell + 4, (idx // cols) * cell + 4
            bx1, by1 = bx0 + 24, by0 + 18
            draw.rectangle([bx0, by0, bx1, by1], fill=(0, 0, 0))
            draw.text((bx0 + 7, by0 + 3), str(idx), fill=(255, 255, 255))

        compact_prompt = re.sub(r"<image\d+>", "", prompt_text).strip()
        panel_hint = (
            "All images are combined into one panel image. "
            "Panels are numbered in row-major order (0, 1, 2, ...). "
            "Use those panel numbers to identify the correct option."
        )
        merged_prompt = f"{compact_prompt}\n\n{panel_hint}" if compact_prompt else panel_hint
        return merged_prompt, [canvas]

    def _build_cogvlm_model_inputs(
        self,
        prompt_text: str,
        pil_images: list[Image.Image] | None,
        tokenizer: Any,
    ) -> dict[str, Any]:
        if pil_images:
            cog_prompt, cog_images = self._pack_multi_images_for_cogvlm(prompt_text, pil_images)
            inputs = self.model.build_conversation_input_ids(
                tokenizer=tokenizer,
                query=cog_prompt,
                history=[],
                images=cog_images,
            )
        else:
            # Text-only tasks perform better through direct tokenization.
            inputs = tokenizer(prompt_text, return_tensors="pt")

        model_inputs: dict[str, Any] = {}
        for k, v in inputs.items():
            if torch.is_tensor(v):
                t = v.to(self.device)
                if t.ndim == 1:
                    t = t.unsqueeze(0)
                if k in {"images", "pixel_values"} and t.is_floating_point():
                    t = t.to(self.dtype)
                model_inputs[k] = t
            elif isinstance(v, list) and v and torch.is_tensor(v[0]):
                converted: list[torch.Tensor] = []
                for item in v:
                    t = item.to(self.device)
                    if k in {"images", "pixel_values"} and t.is_floating_point():
                        t = t.to(self.dtype)
                    converted.append(t)
                model_inputs[k] = [converted]
            else:
                model_inputs[k] = v

        if "token_type_ids" not in model_inputs:
            input_ids = model_inputs.get("input_ids")
            if torch.is_tensor(input_ids):
                model_inputs["token_type_ids"] = torch.zeros_like(input_ids, dtype=torch.long)
        return model_inputs

    def _score_cogvlm_label_choices(
        self,
        prompt_text: str,
        image_paths: list[str] | None,
        option_labels: list[str],
    ) -> tuple[str | None, dict[str, float]]:
        tokenizer = self.tokenizer or getattr(self.processor, "tokenizer", None)
        if tokenizer is None:
            return None, {}
        pil_images = load_pil_images(image_paths, max_image_edge=self.max_image_edge)
        model_inputs = self._build_cogvlm_model_inputs(prompt_text, pil_images, tokenizer)
        with torch.no_grad():
            outputs = self.model(**model_inputs, use_cache=False, return_dict=True)
        logits = outputs.logits[:, -1, :]

        scores: dict[str, float] = {}
        for label in option_labels:
            token_ids: set[int] = set()
            for variant in (label, f" {label}"):
                ids = tokenizer.encode(variant, add_special_tokens=False)
                if len(ids) == 1:
                    token_ids.add(int(ids[0]))
            if token_ids:
                cand = torch.tensor(sorted(token_ids), device=logits.device, dtype=torch.long)
                scores[label] = float(torch.max(torch.index_select(logits[0], 0, cand)).item())
            else:
                scores[label] = float("-inf")

        if not scores:
            return None, {}
        best = max(scores.items(), key=lambda kv: kv[1])[0]
        return best, scores

    def _build_forward_inputs(
        self,
        prompt_text: str,
        image_paths: list[str] | None,
    ) -> tuple[Any, dict[str, Any]] | tuple[None, None]:
        tokenizer = self.tokenizer or getattr(self.processor, "tokenizer", None)
        pil_images = load_pil_images(image_paths, max_image_edge=self.max_image_edge)
        if tokenizer is not None and not pil_images:
            inputs = tokenizer(prompt_text, return_tensors="pt")
            model_inputs: dict[str, Any] = {}
            for k, v in inputs.items():
                if torch.is_tensor(v):
                    model_inputs[k] = v.to(self.device)
                else:
                    model_inputs[k] = v
            return tokenizer, model_inputs
        if (
            self.processor is None
            and tokenizer is not None
            and hasattr(self.model, "build_conversation_input_ids")
        ):
            return tokenizer, self._build_cogvlm_model_inputs(prompt_text, pil_images, tokenizer)

        if self.processor is None:
            return None, None

        apply_template = getattr(self.processor, "apply_chat_template", None)
        if callable(apply_template):
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
            )
        else:
            inputs = self.processor(
                text=[prompt_text],
                images=pil_images if pil_images else None,
                return_tensors="pt",
                padding=True,
            )
        if hasattr(inputs, "to"):
            inputs = inputs.to(self.device)
        return tokenizer, dict(inputs)

    def _score_label_choices_from_logits(
        self,
        prompt_text: str,
        image_paths: list[str] | None,
        option_labels: list[str],
    ) -> tuple[str | None, dict[str, float]]:
        tokenizer, model_inputs = self._build_forward_inputs(prompt_text, image_paths)
        if tokenizer is None or model_inputs is None:
            return None, {}
        try:
            with torch.no_grad():
                outputs = self.model(**model_inputs, use_cache=False, return_dict=True)
        except RuntimeError as exc:
            msg = str(exc)
            if "CUDNN_STATUS_NOT_INITIALIZED" not in msg and "cuDNN" not in msg:
                raise
            # Retry once with cuDNN disabled; some Marlowe nodes intermittently
            # fail CLIP vision conv initialization in LLaVA.
            torch.cuda.empty_cache()
            with torch.backends.cudnn.flags(enabled=False):
                with torch.no_grad():
                    outputs = self.model(**model_inputs, use_cache=False, return_dict=True)
        logits = outputs.logits[:, -1, :]

        scores: dict[str, float] = {}
        for label in option_labels:
            token_ids: set[int] = set()
            for variant in (label, f" {label}"):
                ids = tokenizer.encode(variant, add_special_tokens=False)
                if len(ids) == 1:
                    token_ids.add(int(ids[0]))
            if token_ids:
                cand = torch.tensor(sorted(token_ids), device=logits.device, dtype=torch.long)
                scores[label] = float(torch.max(torch.index_select(logits[0], 0, cand)).item())
            else:
                scores[label] = float("-inf")

        if not scores:
            return None, {}
        best = max(scores.items(), key=lambda kv: kv[1])[0]
        return best, scores

    def evaluate_trial(self, trial: dict) -> dict:
        lower_name = self.model_name.lower()
        answer_format = str(trial.get("answer_format", "label")).strip().lower()
        if (
            "cogvlm" in lower_name
            and answer_format == "label"
            and trial.get("option_labels")
            and (self.processor is None)
            and hasattr(self.model, "build_conversation_input_ids")
        ):
            prompt, _, image_paths, _ = self._prepare_trial_inputs(trial)
            predicted_label, score_map = self._score_cogvlm_label_choices(
                prompt_text=prompt,
                image_paths=image_paths if image_paths else None,
                option_labels=[str(x).upper() for x in trial.get("option_labels", [])],
            )
            return {
                "trial_id": trial["trial_id"],
                "item_uid": trial["item_uid"],
                "generated_text": predicted_label or "",
                "predicted_label": predicted_label,
                "reason": "choice logits",
                "parse_method": "choice_logits",
                "parse_confidence": "high" if predicted_label is not None else "none",
                "parse_raw_candidate": (
                    "; ".join(f"{k}:{v:.3f}" for k, v in sorted(score_map.items()))
                    if score_map
                    else ""
                ),
                "correct_label": trial["correct_label"],
                "is_correct": predicted_label == trial["correct_label"],
                "options": trial.get("options", []),
                "option_labels": trial.get("option_labels", []),
            }
        if (
            "llava" in lower_name
            and answer_format == "label"
            and trial.get("option_labels")
        ):
            prompt, _, image_paths, _ = self._prepare_trial_inputs(trial)
            if not image_paths:
                # Keep text-only tasks (e.g., egma-math) on the generation path.
                return super().evaluate_trial(trial)
            try:
                predicted_label, score_map = self._score_label_choices_from_logits(
                    prompt_text=prompt,
                    image_paths=image_paths if image_paths else None,
                    option_labels=[str(x).upper() for x in trial.get("option_labels", [])],
                )
            except RuntimeError as exc:
                msg = str(exc)
                if "CUDNN_STATUS_NOT_INITIALIZED" in msg or "cuDNN" in msg:
                    # Some cluster driver/cudnn combos intermittently fail on
                    # direct vision forward passes. Fall back to standard
                    # generation so evaluation can proceed.
                    print(
                        f"[{self.model_name}] choice-logit path failed ({type(exc).__name__}: {exc}); "
                        "falling back to generation parsing.",
                    )
                    return super().evaluate_trial(trial)
                raise
            if predicted_label is not None:
                return {
                    "trial_id": trial["trial_id"],
                    "item_uid": trial["item_uid"],
                    "generated_text": predicted_label,
                    "predicted_label": predicted_label,
                    "reason": "choice logits",
                    "parse_method": "choice_logits",
                    "parse_confidence": "high",
                    "parse_raw_candidate": "; ".join(
                        f"{k}:{v:.3f}" for k, v in sorted(score_map.items())
                    ),
                    "correct_label": trial["correct_label"],
                    "is_correct": predicted_label == trial["correct_label"],
                    "options": trial.get("options", []),
                    "option_labels": trial.get("option_labels", []),
                }
        return super().evaluate_trial(trial)

    def generate(
        self,
        prompt_text: str,
        image_paths: list[str] | None = None,
        max_new_tokens: int = 128,
    ) -> str:
        pil_images = load_pil_images(image_paths, max_image_edge=self.max_image_edge)
        tokenizer = self.tokenizer or getattr(self.processor, "tokenizer", None)

        if (
            self.processor is None
            and tokenizer is not None
            and hasattr(self.model, "build_conversation_input_ids")
        ):
            # CogVLM fallback paths when AutoProcessor is unavailable.
            model_inputs = self._build_cogvlm_model_inputs(prompt_text, pil_images, tokenizer)

            gen_kwargs = {
                "do_sample": False,
                "max_new_tokens": int(max_new_tokens),
                **self.generation_defaults,
            }
            # CogVLM remote code can crash when transformers generation hands it
            # partially-populated past_key_values structures. Disabling cache
            # avoids that incompatible code path at modest speed cost.
            gen_kwargs.setdefault("use_cache", False)
            with torch.no_grad():
                output_ids = self.model.generate(**model_inputs, **gen_kwargs)

            input_ids = model_inputs.get("input_ids")
            if input_ids is not None:
                generated_ids = output_ids[:, input_ids.shape[1]:]
            else:
                generated_ids = output_ids
            return tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        apply_template = getattr(self.processor, "apply_chat_template", None) if self.processor else None
        if callable(apply_template):
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
            )
        else:
            inputs = self.processor(
                text=[prompt_text],
                images=pil_images if pil_images else None,
                return_tensors="pt",
                padding=True,
            )

        if hasattr(inputs, "to"):
            inputs = inputs.to(self.device)

        gen_kwargs = {
            "do_sample": False,
            "max_new_tokens": int(max_new_tokens),
            **self.generation_defaults,
        }
        try:
            with torch.no_grad():
                output_ids = self.model.generate(**inputs, **gen_kwargs)
        except RuntimeError as exc:
            msg = str(exc)
            has_pixels = isinstance(inputs, dict) and ("pixel_values" in inputs)
            if ("CUDNN_STATUS_NOT_INITIALIZED" not in msg and "cuDNN" not in msg) or not has_pixels:
                raise
            torch.cuda.empty_cache()
            with torch.backends.cudnn.flags(enabled=False):
                with torch.no_grad():
                    output_ids = self.model.generate(**inputs, **gen_kwargs)

        input_ids = inputs.get("input_ids")
        if input_ids is not None:
            input_len = input_ids.shape[1]
            generated_ids = output_ids[:, input_len:]
        else:
            generated_ids = output_ids

        if hasattr(self.processor, "batch_decode"):
            decoded = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
            return decoded[0] if decoded else ""

        if tokenizer is not None:
            return tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return str(generated_ids)

    def parse_response(self, raw_output: str) -> str:
        return raw_output.strip()

