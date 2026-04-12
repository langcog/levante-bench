"""InternVL3.5 model implementation (HuggingFace-native format)."""

from typing import Optional

import torch

from levante_bench.models.base import ParseResult, VLMModel
from levante_bench.models.registry import register
from levante_bench.models._common import (
    DTYPE_MAP,
    build_pil_content,
    load_pil_images,
    parse_answer_result_with_fallback,
    parse_answer_with_fallback,
)

_SYSTEM_PROMPT = (
    "You are a helpful assistant. "
    "Answer with only a single letter: A, B, C, or D. Do not explain."
)
_USER_INSTRUCTION = "Reply with exactly one letter — A, B, C, or D — and nothing else."


@register("internvl35")
class InternVL35Model(VLMModel):
    """InternVL3.5 via HuggingFace-native format (OpenGVLab/InternVL3_5-{size}-HF).

    Uses AutoProcessor + AutoModelForImageTextToText with trust_remote_code=True.
    Images are loaded as PIL objects.  The Qwen-based chat template is applied by
    the processor, so the generate() pipeline mirrors Qwen35Model closely.

    A per-message instruction is appended to the user turn because InternVL3.5
    tends to ignore system-only instructions when images are present.
    """

    def __init__(
        self,
        model_name: str = "OpenGVLab/InternVL3_5-1B-HF",
        device: str = "cpu",
        dtype: str = "bfloat16",
        attn_implementation: str = "sdpa",
        max_patches: int | None = None,
    ) -> None:
        super().__init__(model_name=model_name, device=device)
        self.dtype = DTYPE_MAP.get(dtype, torch.bfloat16)
        self.attn_implementation = attn_implementation
        self.max_patches = max_patches

    def load(self) -> None:
        """Load InternVL3.5-HF model and processor from HuggingFace."""
        from transformers import AutoProcessor, AutoModelForImageTextToText

        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            padding_side="left",
            trust_remote_code=True,
        )
        if self.max_patches is not None:
            image_processor = getattr(self.processor, "image_processor", None)
            if image_processor is not None and hasattr(image_processor, "max_patches"):
                image_processor.max_patches = int(self.max_patches)
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
        pil_images = load_pil_images(image_paths)
        messages = self._build_messages(prompt_text, pil_images)

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
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
        """Evaluate trials with batched tokenization/generation when possible.

        Falls back to per-trial execution if the processor/model stack cannot
        batch the current multimodal input layout.
        """
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
                    m, tokenize=False, add_generation_prompt=True
                )
                for m in messages
            ]

            if all(batch is None for batch in pil_batches):
                batched_images = None
            else:
                batched_images = pil_batches

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
                trials, answer_formats, input_lens, output_ids
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
            # Keep benchmark robustness over throughput when batch packing fails.
            return [self.evaluate_trial(trial) for trial in trials]

    def _build_messages(
        self,
        prompt_text: str,
        pil_images: Optional[list] = None,
    ) -> list[dict]:
        """Build InternVL3.5 chat messages with system prompt and PIL images."""
        content = build_pil_content(prompt_text, pil_images)
        content.append({"type": "text", "text": _USER_INSTRUCTION})
        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]

    def parse_response(self, raw_output: str) -> str:
        """Return generated text as-is (already decoded from generated tokens only)."""
        return raw_output.strip()

    def score_choices(
        self,
        prompt_text: str,
        image_paths: list[str],
        choice_texts: tuple[str, str] = ("1", "2"),
    ) -> dict:
        """Return next-token probabilities/logits for two one-token choices."""
        pil_images = load_pil_images(image_paths)
        messages = self._build_messages(prompt_text, pil_images)
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text],
            images=pil_images if pil_images else None,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        choice_ids: list[int] = []
        for choice in choice_texts:
            toks = self.processor.tokenizer.encode(choice, add_special_tokens=False)
            if len(toks) != 1:
                raise ValueError(
                    f"Choice {choice!r} must map to one token; got ids={toks}"
                )
            choice_ids.append(toks[0])

        output, elapsed = self._timed_call(lambda: self.model(**inputs))
        next_logits = output.logits[:, -1, :].float()
        selected = next_logits[:, choice_ids].squeeze(0)
        probs = torch.softmax(selected, dim=-1)
        return {
            "choice_texts": list(choice_texts),
            "choice_token_ids": choice_ids,
            "choice_logits": [float(selected[0].item()), float(selected[1].item())],
            "choice_probs": [float(probs[0].item()), float(probs[1].item())],
            "generation_time_s": elapsed,
            "model_name": self.model_name,
            "num_tokens_generated": 0,
        }

    def parse_answer(
        self, text: str, option_labels: list[str]
    ) -> tuple[Optional[str], str]:
        """Base-class parser first; falls back to reverse-sentence scan."""
        return parse_answer_with_fallback(self, text, option_labels)

    def parse_answer_result(self, text: str, option_labels: list[str]) -> ParseResult:
        """Parser with provenance, including reverse-sentence fallback."""
        return parse_answer_result_with_fallback(self, text, option_labels)

    def _pad_token_id(self) -> int | None:
        """Resolve a stable pad token id to avoid per-call generation warnings."""
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
