"""SmolVLM2 model implementation."""

import re
import sys
from pathlib import Path

import torch

from levante_bench.models.base import VLMModel
from levante_bench.models.registry import register
from levante_bench.models._common import DTYPE_MAP


@register("smolvlm2")
class SmolVLM2Model(VLMModel):
    """SmolVLM2 via HuggingFace AutoProcessor + AutoModelForImageTextToText."""

    def __init__(
        self,
        model_name: str = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
        device: str = "cpu",
        dtype: str = "bfloat16",
        attn_implementation: str = "eager",
    ) -> None:
        super().__init__(model_name=model_name, device=device)
        self.dtype = DTYPE_MAP.get(dtype, torch.bfloat16)
        self.attn_implementation = attn_implementation
        self._batch_fallback_count = 0

    def load(self) -> None:
        """Load SmolVLM2 model and processor from HuggingFace."""
        from transformers import AutoProcessor, AutoModelForImageTextToText

        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_name,
            dtype=self.dtype,
            attn_implementation=self.attn_implementation,
        ).to(self.device)
        self.model.eval()

    def generate(
        self,
        prompt_text: str,
        image_paths: list[str] | None = None,
        max_new_tokens: int = 64,
    ) -> str:
        """Generate text using SmolVLM2."""
        messages = self._build_messages(prompt_text, image_paths)
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device, dtype=self.dtype)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs, do_sample=False, max_new_tokens=max_new_tokens
            )

        return self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]

    def evaluate_trials_batch(self, trials: list[dict]) -> list[dict]:
        """Evaluate trials with batched tokenization/generation when possible."""
        if not trials:
            return []
        if len(trials) == 1:
            return [self.evaluate_trial(trials[0])]

        prepared_trials: list[dict] = list(trials)
        prepared_inputs = [self._prepare_trial_inputs(trial) for trial in prepared_trials]

        prompts = [item[0] for item in prepared_inputs]
        answer_formats = [item[1] for item in prepared_inputs]
        image_path_batches = [item[2] for item in prepared_inputs]
        max_new_tokens = max(item[3] for item in prepared_inputs)

        try:
            messages_batch = [
                self._build_messages(prompt_text, image_paths)
                for prompt_text, image_paths in zip(prompts, image_path_batches)
            ]
            inputs = self.processor.apply_chat_template(
                messages_batch,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                padding=True,
            ).to(self.device, dtype=self.dtype)

            if "attention_mask" in inputs:
                input_lens = inputs["attention_mask"].sum(dim=1).tolist()
            else:
                input_lens = [inputs["input_ids"].shape[1]] * len(prepared_trials)

            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                )

            results: list[dict] = []
            for trial, answer_format, in_len, row in zip(
                prepared_trials, answer_formats, input_lens, output_ids
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
        except Exception as exc:
            self._batch_fallback_count += 1
            # Log fallback path to help tune batch_size and debug packing failures.
            print(
                (
                    f"[smolvlm2] batch fallback #{self._batch_fallback_count}: "
                    f"{type(exc).__name__}: {exc}"
                ),
                file=sys.stderr,
            )
            return [self.evaluate_trial(trial) for trial in trials]

    def _build_messages(
        self,
        prompt_text: str,
        image_paths: list[str] | None = None,
    ) -> list[dict]:
        """Build SmolVLM2 chat messages, interleaving images at <imageN> placeholders."""
        content = []
        if image_paths and re.search(r"<image\d+>", prompt_text):
            labels = ["A", "B", "C", "D", "E", "F", "G", "H"]
            parts = re.split(r"(<image\d+>)", prompt_text)
            # If <image0> is present, treat numbers as direct indices (context=0);
            # otherwise use 1-based (<image1> → image_paths[0]).
            has_image0 = "<image0>" in prompt_text
            for part in parts:
                m = re.match(r"<image(\d+)>", part)
                if m:
                    n = int(m.group(1))
                    idx = n if has_image0 else n - 1
                    if 0 <= idx < len(image_paths):
                        label = labels[idx] if idx < len(labels) else str(idx + 1)
                        content.append({"type": "text", "text": f"{label}:"})
                        content.append(
                            {"type": "image", "url": str(Path(image_paths[idx]).resolve())}
                        )
                elif part.strip():
                    content.append({"type": "text", "text": part.strip()})
        else:
            if image_paths:
                for path in image_paths:
                    content.append({"type": "image", "url": str(Path(path).resolve())})
            content.append({"type": "text", "text": prompt_text})
        return [{"role": "user", "content": content}]

    def parse_response(self, raw_output: str) -> str:
        """Extract only the assistant's reply from SmolVLM2 output."""
        if "Assistant:" in raw_output:
            text = raw_output.split("Assistant:")[-1]
        else:
            text = raw_output
        return re.sub(r"<\|?end\|?>.*$", "", text).strip()

    def score_choices(
        self,
        prompt_text: str,
        image_paths: list[str],
        choice_texts: tuple[str, str] = ("1", "2"),
    ) -> dict:
        """Return next-token probabilities/logits for two one-token choices."""
        messages = self._build_messages(prompt_text, image_paths)
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device, dtype=self.dtype)

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
