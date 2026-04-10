"""SmolVLM2 model implementation."""

import re
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
        prompt_profile: str = "baseline",
    ) -> None:
        super().__init__(model_name=model_name, device=device)
        self.dtype = DTYPE_MAP.get(dtype, torch.bfloat16)
        self.attn_implementation = attn_implementation
        self.prompt_profile = str(prompt_profile).strip().lower() or "baseline"

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

    def evaluate_trial(self, trial: dict) -> dict:
        """Apply task-tuned prompt upgrades before base evaluation."""
        if self.prompt_profile != "upgraded":
            return super().evaluate_trial(trial)
        trial_with_prompt = dict(trial)
        prompt = str(trial.get("prompt", ""))
        task_id = str(trial.get("task_id", "") or "").strip().lower()
        if prompt:
            trial_with_prompt["prompt"] = self._upgraded_prompt(prompt, task_id)
        return super().evaluate_trial(trial_with_prompt)

    def _upgraded_prompt(self, prompt: str, task_id: str) -> str:
        """Append concise, task-specific instructions from prompt optimization runs."""
        base_instruction = (
            "Final answer format: respond with exactly one option letter (A, B, C, or D)."
        )
        additions: list[str] = [base_instruction]

        if task_id == "vocab":
            additions.append(
                "Only one image matches the target word. Compare all options and choose the single best match."
            )
        elif task_id == "trog":
            additions.append(
                "Ground the sentence meaning in the images: who is doing what to whom, and where."
            )
            additions.append(
                "Use image details that distinguish grammar roles, then choose one letter."
            )
        elif task_id == "egma-math":
            additions.append(
                "Solve the math carefully before choosing. Double-check arithmetic and pick the best option."
            )
        elif task_id == "theory-of-mind":
            additions.append(
                "Use each character's perspective: they only know what they saw or were told."
            )
            additions.append(
                "Ignore events that happened while a character was away unless they were informed later."
            )
        elif task_id == "matrix-reasoning":
            additions.append(
                "Identify the transformation rule across rows and columns, then apply the same rule to the missing cell."
            )
            additions.append(
                "Prefer the option that best matches both row and column patterns."
            )

        return f"{prompt}\n\n" + "\n".join(additions)

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
            for part in parts:
                m = re.match(r"<image(\d+)>", part)
                if m:
                    idx = int(m.group(1)) - 1
                    if idx < len(image_paths):
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
        if self.prompt_profile == "upgraded":
            return [
                {
                    "role": "system",
                    "content": "You are a precise visual reasoning assistant.",
                },
                {"role": "user", "content": content},
            ]
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
        image_paths: list[str] | None = None,
        choice_texts: tuple[str, str] = ("1", "2"),
    ) -> dict:
        """Return next-token logits/probs for one-token choices."""
        messages = self._build_messages(prompt_text, image_paths)
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device, dtype=self.dtype)

        choice_ids: list[int] = []
        for ch in choice_texts:
            toks = self.processor.tokenizer.encode(ch, add_special_tokens=False)
            if len(toks) != 1:
                raise ValueError(f"Choice {ch!r} must map to one token; got ids={toks}")
            choice_ids.append(toks[0])

        with torch.no_grad():
            out = self.model(**inputs)
        next_logits = out.logits[:, -1, :].float()
        sel = next_logits[:, choice_ids].squeeze(0)
        probs = torch.softmax(sel, dim=-1)
        return {
            "choice_texts": list(choice_texts),
            "choice_token_ids": choice_ids,
            "choice_logits": [float(sel[0].item()), float(sel[1].item())],
            "choice_probs": [float(probs[0].item()), float(probs[1].item())],
            "generation_time_s": 0.0,
            "model_name": self.model_name,
            "num_tokens_generated": 0,
        }
