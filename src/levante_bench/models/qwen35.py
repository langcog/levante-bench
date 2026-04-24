"""Qwen3.5-VL model implementation."""

import re
from typing import Optional

import torch

from levante_bench.models.base import ParseResult, SYSTEM_PROMPT, VLMModel
from levante_bench.models.registry import register
from levante_bench.models._common import (
    DTYPE_MAP,
    build_pil_content,
    load_pil_images,
)


@register("qwen35")
@register("qwen25vl_qlora")
@register("qwen25vl_32b")
@register("qwen3vl_30b")
class Qwen35Model(VLMModel):
    """Qwen3.5-VL via HuggingFace AutoProcessor + AutoModelForImageTextToText.

    Images are loaded as PIL objects and passed separately from the text so
    that no extra dependency (qwen_vl_utils) is required.  The processor
    applies the Qwen3 chat template and inserts vision tokens automatically.
    Only the newly generated tokens are decoded, so parse_response is trivial.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3.5-0.8B",
        device: str = "cpu",
        dtype: str = "bfloat16",
        attn_implementation: str = "sdpa",
    ) -> None:
        super().__init__(model_name=model_name, device=device)
        self.dtype = DTYPE_MAP.get(dtype, torch.bfloat16)
        self.attn_implementation = attn_implementation

    def load(self) -> None:
        """Load Qwen3.5 model and processor from HuggingFace."""
        from transformers import AutoProcessor, AutoModelForImageTextToText

        self.processor = AutoProcessor.from_pretrained(
            self.model_name, padding_side="left"
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_name,
            dtype=self.dtype,
            attn_implementation=self.attn_implementation,
        ).to(self.device)
        self.model.eval()

    # Token IDs for thinking budget control (Qwen3 family)
    THINK_END_TOKEN_ID = 151668   # </think>
    IM_END_TOKEN_ID = 151645      # <|im_end|>
    EARLY_STOP_SUFFIX = (
        "\n\nConsidering the limited time by the user, "
        "I have to give the solution based on the thinking directly now."
        "\n</think>\n\n"
    )

    def generate(
        self,
        prompt_text: str,
        image_paths: list[str] | None = None,
        max_new_tokens: int = 64,
    ) -> str:
        """Generate text using Qwen3.5-VL with optional thinking budget.

        When ``thinking_budget`` is set (via YAML config), the generation is
        split into two passes: the first produces up to *thinking_budget*
        tokens of reasoning; if the model hasn't closed its ``</think>`` block
        by then, an early-stop prompt is appended and a second pass generates
        the final answer with the remaining token budget.
        """
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
        thinking_budget = getattr(self, "thinking_budget", 0)

        if thinking_budget > 0 and max_new_tokens > thinking_budget:
            return self._generate_with_budget(
                inputs, input_len, thinking_budget, max_new_tokens
            )

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs, do_sample=False, max_new_tokens=max_new_tokens
            )

        generated_ids = output_ids[:, input_len:]
        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    def _generate_with_budget(
        self,
        inputs: dict,
        input_len: int,
        thinking_budget: int,
        max_new_tokens: int,
    ) -> str:
        """Two-pass generation: capped thinking + answer."""
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs, do_sample=False, max_new_tokens=thinking_budget
            )

        new_ids = output_ids[0, input_len:].tolist()

        # If thinking already finished or generation completed, return as-is
        if (self.THINK_END_TOKEN_ID in new_ids
                or self.IM_END_TOKEN_ID in new_ids):
            return self.processor.decode(
                output_ids[0, input_len:], skip_special_tokens=True
            )

        # Thinking didn't finish — append early-stop suffix and generate again
        suffix_ids = self.processor.tokenizer.encode(
            self.EARLY_STOP_SUFFIX, add_special_tokens=False,
            return_tensors="pt",
        ).to(output_ids.device)
        extended = torch.cat([output_ids, suffix_ids], dim=-1)
        attn_mask = torch.ones_like(extended, dtype=torch.long)

        remaining = max_new_tokens - len(new_ids) - suffix_ids.shape[-1]
        if remaining <= 0:
            remaining = 64

        with torch.no_grad():
            final_ids = self.model.generate(
                input_ids=extended,
                attention_mask=attn_mask,
                do_sample=False,
                max_new_tokens=remaining,
            )

        all_new = final_ids[0, input_len:].tolist()
        # Find </think> and return only the content after it
        try:
            idx = len(all_new) - all_new[::-1].index(self.THINK_END_TOKEN_ID)
        except ValueError:
            idx = 0
        return self.processor.decode(
            all_new[idx:], skip_special_tokens=True
        ).strip()

    def evaluate_trials_batch(self, trials: list[dict]) -> list[dict]:
        """Evaluate trials with batched tokenization/generation when possible.

        Falls back to per-trial execution if batch packing fails for a
        specific multimodal layout.
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
            return [self.evaluate_trial(trial) for trial in trials]

    def _build_messages(
        self,
        prompt_text: str,
        pil_images: Optional[list] = None,
    ) -> list[dict]:
        """Build Qwen3.5 chat messages with system prompt and PIL images."""
        content = build_pil_content(prompt_text, pil_images)
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]

    def parse_response(self, raw_output: str) -> str:
        """Return generated text as-is (already decoded from generated tokens only)."""
        return raw_output.strip()

    def parse_answer_result(self, text: str, option_labels: list[str]) -> ParseResult:
        """Parse label answers with Qwen-specific fallback for truncated analyses.

        Qwen3.5 outputs sometimes enumerate "Analyze Image A/B/C/D" and get cut off
        before an explicit final answer token, which leaves the base parser with
        `unparseable`. Recover from that pattern when the evidence is clear.
        """
        result = super().parse_answer_result(text, option_labels)
        if result.value is not None:
            return result

        labels_upper = [str(label).upper() for label in option_labels]
        if not labels_upper:
            return result

        marker_re = re.compile(r"Analyze\s+Image\s+([A-Z])\s*:", re.IGNORECASE)
        markers = list(marker_re.finditer(text))
        if not markers:
            return result

        scores: dict[str, tuple[int, int]] = {}
        for idx, match in enumerate(markers):
            label = match.group(1).upper()
            if label not in labels_upper:
                continue
            seg_start = match.end()
            seg_end = markers[idx + 1].start() if idx + 1 < len(markers) else len(text)
            segment = text[seg_start:seg_end]

            neg_hits = len(re.findall(r"\bnot\b|\bisn't\b|\baren't\b|\bno\b", segment, re.IGNORECASE))
            pos_hits = len(
                re.findall(
                    r"\bclearly\b|\bcorrect\b|\bfits\b|\bdepicts?\b|\bindicates?\b|\bmatches?\b",
                    segment,
                    re.IGNORECASE,
                )
            )
            scores[label] = (neg_hits, pos_hits)

        if not scores:
            return result

        # Primary rule: if exactly one analyzed label has zero negatives while
        # at least one alternative has explicit negatives, select it.
        zero_neg = [label for label, (neg, _) in scores.items() if neg == 0]
        any_neg = any(neg > 0 for neg, _ in scores.values())
        if len(zero_neg) == 1 and any_neg:
            return ParseResult(
                value=zero_neg[0],
                reason=text,
                parse_method="qwen_analyze_image_heuristic",
                parse_confidence="low",
                raw_candidate=zero_neg[0],
            )

        # Secondary rule: choose the best (positive - negative) score if unique.
        weighted = sorted(
            ((label, pos - neg) for label, (neg, pos) in scores.items()),
            key=lambda item: item[1],
            reverse=True,
        )
        if len(weighted) >= 2 and weighted[0][1] > weighted[1][1] and weighted[0][1] > 0:
            return ParseResult(
                value=weighted[0][0],
                reason=text,
                parse_method="qwen_analyze_image_weighted",
                parse_confidence="low",
                raw_candidate=weighted[0][0],
            )

        return result

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

