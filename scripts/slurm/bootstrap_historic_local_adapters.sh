#!/bin/bash
# Bootstrap local historic VLM adapter files for shared environments.
#
# This script exists so teams can apply the same adapter/config backfill on
# machines that have partial branch syncs or stale checkouts.
#
# Usage:
#   bash scripts/slurm/bootstrap_historic_local_adapters.sh
#   FORCE=1 bash scripts/slurm/bootstrap_historic_local_adapters.sh

set -euo pipefail

FORCE="${FORCE:-0}"
ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

adapter_file="$ROOT_DIR/src/levante_bench/models/historic_local.py"
llava_cfg="$ROOT_DIR/configs/models/llava15_13b.yaml"
cog_cfg="$ROOT_DIR/configs/models/cogvlm.yaml"
flamingo_cfg="$ROOT_DIR/configs/models/openflamingo9b.yaml"

write_file() {
  local path="$1"
  local content="$2"
  if [[ -f "$path" && "$FORCE" != "1" ]]; then
    echo "Keep existing: $path"
    return
  fi
  mkdir -p "$(dirname "$path")"
  printf "%s" "$content" > "$path"
  echo "Wrote: $path"
}

adapter_content='"""Local historic VLM adapters (LLaVA-1.5, CogVLM, OpenFlamingo)."""

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

    def _load_processor(self) -> Any:
        from transformers import AutoProcessor
        kwargs = {"trust_remote_code": self.trust_remote_code}
        try:
            return AutoProcessor.from_pretrained(self.model_name, **kwargs)
        except TypeError:
            kwargs.pop("trust_remote_code", None)
            return AutoProcessor.from_pretrained(self.model_name, **kwargs)

    def _load_with_model_cls(self, model_cls: Any, attn_impl: str) -> Any:
        base_kwargs: dict[str, Any] = {
            "low_cpu_mem_usage": True,
            "trust_remote_code": self.trust_remote_code,
        }
        if self.device_map:
            base_kwargs["device_map"] = self.device_map

        attempt_kwargs = []

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
            raise RuntimeError(f"Failed to load model class {model_cls} for {self.model_name!r}.")
        raise last_exc

    def _load_model(self, attn_impl: str) -> Any:
        from transformers import AutoModelForCausalLM
        import transformers as tfm
        last_exc: Exception | None = None
        model_classes = []
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
            except Exception as exc:
                last_exc = exc
        if last_exc is None:
            raise RuntimeError(f"Could not load model {self.model_name!r}.")
        raise last_exc

    def load(self) -> None:
        self.processor = self._load_processor()
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

    def _patch_generation_compat(self) -> None:
        if self.model is None:
            return
        lower_name = self.model_name.lower()
        if "cogvlm" not in lower_name:
            return
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

    def generate(
        self,
        prompt_text: str,
        image_paths: list[str] | None = None,
        max_new_tokens: int = 128,
    ) -> str:
        pil_images = load_pil_images(image_paths, max_image_edge=self.max_image_edge)
        apply_template = getattr(self.processor, "apply_chat_template", None)
        if callable(apply_template):
            messages = self._build_messages(prompt_text, pil_images)
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[text], images=pil_images if pil_images else None, return_tensors="pt", padding=True)
        else:
            inputs = self.processor(text=[prompt_text], images=pil_images if pil_images else None, return_tensors="pt", padding=True)

        if hasattr(inputs, "to"):
            inputs = inputs.to(self.device)

        gen_kwargs = {"do_sample": False, "max_new_tokens": int(max_new_tokens), **self.generation_defaults}
        gen_kwargs.setdefault("use_cache", False)
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        input_ids = inputs.get("input_ids")
        if input_ids is not None:
            generated_ids = output_ids[:, input_ids.shape[1]:]
        else:
            generated_ids = output_ids

        if hasattr(self.processor, "batch_decode"):
            decoded = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
            return decoded[0] if decoded else ""

        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is not None:
            return tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return str(generated_ids)

    def parse_response(self, raw_output: str) -> str:
        return raw_output.strip()
'

llava_content='name: llava15_13b
hf_name: llava-hf/llava-1.5-13b-hf
dtype: float16
attn_implementation: sdpa
trust_remote_code: true
capabilities:
  - text_only
  - single_image
  - multi_image
'

cog_content='name: cogvlm
hf_name: THUDM/cogvlm-chat-hf
tokenizer_hf_name: lmsys/vicuna-7b-v1.5
dtype: float16
attn_implementation: sdpa
trust_remote_code: true
use_json_format: false
max_new_tokens: 16
capabilities:
  - text_only
  - single_image
  - multi_image
'

flamingo_content='name: openflamingo9b
hf_name: openflamingo/OpenFlamingo-9B-vitl-mpt7b
tokenizer_hf_name: EleutherAI/gpt-neox-20b
dtype: float16
attn_implementation: sdpa
trust_remote_code: true
capabilities:
  - text_only
  - single_image
  - multi_image
'

write_file "$adapter_file" "$adapter_content"
write_file "$llava_cfg" "$llava_content"
write_file "$cog_cfg" "$cog_content"
write_file "$flamingo_cfg" "$flamingo_content"

echo ""
echo "Bootstrap complete."
echo "Next checks:"
echo "  python -m levante_bench.cli list-models | grep -E 'llava15_13b|cogvlm|openflamingo9b'"
echo "  python -m levante_bench.cli run-eval --model llava15_13b --task vocab --version v1 --device cuda --batch-size 1"
