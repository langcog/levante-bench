"""Local historic VLM adapters (LLaVA-1.5, CogVLM, OpenFlamingo)."""

from __future__ import annotations

from typing import Any, Optional

import torch

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
    ) -> None:
        super().__init__(model_name=model_name, device=device)
        self.dtype = DTYPE_MAP.get(str(dtype), torch.bfloat16)
        self.attn_implementation = str(attn_implementation or "flash_attention_2")
        self.trust_remote_code = bool(trust_remote_code)
        self.device_map = str(device_map).strip() if device_map else None
        self.max_image_edge = int(max_image_edge) if max_image_edge else None
        self.generation_defaults = dict(generation) if isinstance(generation, dict) else {}
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
            # CogVLM chat checkpoints are LLaMA-family and often fail through
            # AutoTokenizer fast/convert code paths.
            from transformers import LlamaTokenizer

            return LlamaTokenizer.from_pretrained(self.model_name, use_fast=False)

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
        if self.processor is None and processor_exc is not None:
            print(
                f"[{self.model_name}] processor unavailable ({type(processor_exc).__name__}); "
                "using tokenizer/build_conversation_input_ids fallback."
            )

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
            # CogVLM-style remote code path.
            inputs = self.model.build_conversation_input_ids(
                tokenizer=tokenizer,
                query=prompt_text,
                history=[],
                images=pil_images or [],
            )
            model_inputs: dict[str, Any] = {}
            for k, v in inputs.items():
                if torch.is_tensor(v):
                    model_inputs[k] = v.unsqueeze(0).to(self.device)
                elif isinstance(v, list) and v and torch.is_tensor(v[0]):
                    model_inputs[k] = [[item.to(self.device) for item in v]]
                else:
                    model_inputs[k] = v

            gen_kwargs = {
                "do_sample": False,
                "max_new_tokens": int(max_new_tokens),
                **self.generation_defaults,
            }
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

