"""Molmo 2 model implementation."""

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


_MOLMO2_PROCESSOR_OPTIONAL_KWARGS = {
    "time_mode",
    "image_use_col_tokens",
    "use_single_crop_col_tokens",
    "use_single_crop_start_token",
    "video_use_col_tokens",
    "use_frame_special_tokens",
}


def _patch_rope_init_functions_for_molmo2() -> None:
    """Restore the legacy RoPE "default" key expected by Molmo2 remote code.

    Some newer Transformers releases removed ``ROPE_INIT_FUNCTIONS["default"]``
    while Molmo2's trusted remote modeling code still indexes that key. Keep the
    compatibility patch in our adapter instead of editing the Hugging Face cache
    on each machine.
    """
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    if "default" in ROPE_INIT_FUNCTIONS:
        return

    def _compute_default_rope_parameters(
        config=None,
        device=None,
        seq_len: int | None = None,
        layer_type: str | None = None,
    ):
        del seq_len  # The default RoPE initializer does not use current length.
        if config is None:
            raise ValueError("Molmo2 default RoPE initialization requires a config.")

        if hasattr(config, "standardize_rope_params"):
            config.standardize_rope_params()

        rope_parameters = getattr(config, "rope_parameters", None)
        if isinstance(rope_parameters, dict):
            if layer_type is not None and isinstance(rope_parameters.get(layer_type), dict):
                rope_parameters = rope_parameters[layer_type]
        else:
            rope_parameters = {}

        base = (
            rope_parameters.get("rope_theta")
            or rope_parameters.get("base")
            or getattr(config, "rope_theta", None)
            or 10000
        )
        partial_rotary_factor = rope_parameters.get(
            "partial_rotary_factor",
            getattr(config, "partial_rotary_factor", 1.0),
        )
        head_dim = getattr(config, "head_dim", None) or (
            config.hidden_size // config.num_attention_heads
        )
        dim = int(head_dim * partial_rotary_factor)
        inv_freq = 1.0 / (
            float(base)
            ** (
                torch.arange(0, dim, 2, dtype=torch.int64).to(
                    device=device,
                    dtype=torch.float,
                )
                / dim
            )
        )
        return inv_freq, 1.0

    ROPE_INIT_FUNCTIONS["default"] = _compute_default_rope_parameters


def _patch_processor_mixin_for_molmo2() -> None:
    """Allow Molmo2 remote processor optional kwargs on newer Transformers.

    Molmo2Processor forwards image/video token formatting options to
    ProcessorMixin.__init__. Some Transformers releases reject those options
    before setting them, even though the remote processor expects them as
    instance attributes.
    """
    from transformers.processing_utils import ProcessorMixin

    if getattr(ProcessorMixin, "_levante_molmo2_optional_kwargs_patch", False):
        return

    original_init = ProcessorMixin.__init__

    def _patched_init(self, *args, **kwargs):
        molmo2_kwargs = {}
        if self.__class__.__name__ == "Molmo2Processor":
            for key in list(kwargs):
                if key in _MOLMO2_PROCESSOR_OPTIONAL_KWARGS:
                    molmo2_kwargs[key] = kwargs.pop(key)

        original_init(self, *args, **kwargs)

        for key, value in molmo2_kwargs.items():
            setattr(self, key, value)

    ProcessorMixin.__init__ = _patched_init
    ProcessorMixin._levante_molmo2_optional_kwargs_patch = True


@register("molmo2")
class Molmo2Model(VLMModel):
    """Molmo 2 via HuggingFace AutoProcessor + AutoModelForImageTextToText."""

    def __init__(
        self,
        model_name: str = "allenai/Molmo2-8B",
        device: str = "cpu",
        dtype: str = "bfloat16",
        attn_implementation: str = "flash_attention_2",
    ) -> None:
        super().__init__(model_name=model_name, device=device)
        self.dtype = DTYPE_MAP.get(dtype, torch.bfloat16)
        self.attn_implementation = attn_implementation

    def load(self) -> None:
        """Load Molmo 2 model and processor from HuggingFace."""
        from transformers import AutoModelForImageTextToText, AutoProcessor

        _patch_rope_init_functions_for_molmo2()
        _patch_processor_mixin_for_molmo2()
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        requested_attn = self.attn_implementation
        try:
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_name,
                dtype=self.dtype,
                attn_implementation=requested_attn,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            ).to(self.device)
        except Exception as exc:
            if not should_fallback_to_sdpa(requested_attn, exc):
                raise
            warn_attn_fallback(self.model_name, requested_attn, exc)
            self.attn_implementation = "sdpa"
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_name,
                dtype=self.dtype,
                attn_implementation="sdpa",
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
        """Generate text using Molmo 2."""
        pil_images = load_pil_images(image_paths)
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
                pad_token_id=self._pad_token_id(),
            )

        generated_ids = output_ids[:, input_len:]
        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    def evaluate_trials_batch(self, trials: list[dict]) -> list[dict]:
        """Evaluate trials with batched tokenization/generation when possible."""
        if not trials:
            return []
        if len(trials) == 1:
            return [self.evaluate_trial(trials[0])]

        prepared = [self._prepare_trial_inputs(trial) for trial in trials]
        prompts = [item[0] for item in prepared]
        answer_formats = [item[1] for item in prepared]
        image_path_batches = [item[2] for item in prepared]
        max_new_tokens = max(item[3] for item in prepared)

        try:
            pil_batches = [
                load_pil_images(image_paths) if image_paths else None
                for image_paths in image_path_batches
            ]
            messages = [
                self._build_messages(prompt_text, pil_images)
                for prompt_text, pil_images in zip(prompts, pil_batches)
            ]
            texts = [
                self.processor.apply_chat_template(
                    message,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for message in messages
            ]

            batched_images = None if all(batch is None for batch in pil_batches) else pil_batches
            inputs = self.processor(
                text=texts,
                images=batched_images,
                return_tensors="pt",
                padding=True,
            ).to(self.device)

            if "attention_mask" in inputs:
                input_lens = inputs["attention_mask"].sum(dim=1).tolist()
            else:
                input_lens = [inputs["input_ids"].shape[1]] * len(trials)

            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=self._pad_token_id(),
                )

            results: list[dict] = []
            for trial, answer_format, in_len, row in zip(
                trials,
                answer_formats,
                input_lens,
                output_ids,
            ):
                generated_ids = row[int(in_len):]
                raw_text = self.processor.decode(
                    generated_ids,
                    skip_special_tokens=True,
                )
                clean_text = self.parse_response(raw_text)
                results.append(
                    self._build_result_from_text(
                        trial=trial,
                        clean_text=clean_text,
                        answer_format=answer_format,
                    )
                )
            return results
        except Exception:
            return [self.evaluate_trial(trial) for trial in trials]

    def _build_messages(
        self,
        prompt_text: str,
        pil_images: list | None = None,
    ) -> list[dict]:
        """Build Molmo 2 chat messages with optional interleaved images.

        Molmo2's remote chat template expects strict user/assistant alternation
        and rejects a separate system role. Fold the shared system instruction
        into the first user turn instead.
        """
        content = build_pil_content(f"{SYSTEM_PROMPT}\n\n{prompt_text}", pil_images)
        return [{"role": "user", "content": content}]

    def parse_response(self, raw_output: str) -> str:
        return raw_output.strip()

    def _pad_token_id(self) -> int | None:
        processor_tokenizer = getattr(self.processor, "tokenizer", None)
        model_gen_cfg = getattr(self.model, "generation_config", None)

        for candidate in (
            getattr(processor_tokenizer, "pad_token_id", None),
            getattr(model_gen_cfg, "pad_token_id", None),
            getattr(processor_tokenizer, "eos_token_id", None),
            getattr(model_gen_cfg, "eos_token_id", None),
        ):
            if candidate is not None:
                return int(candidate)
        return None
