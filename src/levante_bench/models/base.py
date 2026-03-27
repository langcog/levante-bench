"""Base class for VLM evaluation models."""

import json
import re
from typing import Optional


ANSWER_FORMAT_INSTRUCTION = '\n\nRespond in JSON format: {"answer": "<letter>", "reason": "<short reason>"}'


class VLMModel:
    """Base class for all VLM models used in evaluation.

    Subclasses implement load(), generate(), _build_messages(), and
    parse_response() for model-specific behavior.
    evaluate_trial() and parse_answer() are generic.
    """

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        self.model_name = model_name
        self.device = device
        self.model = None
        self.processor = None

    def load(self) -> None:
        """Load model and processor onto device."""
        raise NotImplementedError

    def generate(
        self,
        prompt_text: str,
        image_paths: list[str] | None = None,
        max_new_tokens: int = 64,
    ) -> str:
        """Generate text given a prompt and optional images."""
        raise NotImplementedError

    def _build_messages(
        self,
        prompt_text: str,
        image_paths: list[str] | None = None,
    ) -> list[dict]:
        """Wrap prompt + images into model-specific chat message format."""
        raise NotImplementedError

    def parse_response(self, raw_output: str) -> str:
        """Clean model-specific output into plain text. Override per model."""
        return raw_output.strip()

    def evaluate_trial(self, trial: dict) -> dict:
        """Run a single trial: generate answer, parse it, return result."""
        prompt = trial["prompt"] + ANSWER_FORMAT_INSTRUCTION
        image_paths = trial.get("context_image_paths", []) + trial.get("option_image_paths", [])
        raw_output = self.generate(
            prompt_text=prompt,
            image_paths=image_paths if image_paths else None,
            max_new_tokens=trial.get("max_new_tokens", 64),
        )
        clean_text = self.parse_response(raw_output)
        predicted_label, reason = self.parse_answer(clean_text, trial["option_labels"])
        return {
            "trial_id": trial["trial_id"],
            "item_uid": trial["item_uid"],
            "generated_text": clean_text,
            "predicted_label": predicted_label,
            "reason": reason,
            "correct_label": trial["correct_label"],
            "is_correct": predicted_label == trial["correct_label"],
            # Carry option context for downstream human-comparison annotation
            "options": trial.get("options", []),
            "option_labels": trial.get("option_labels", []),
        }

    def parse_answer(self, text: str, option_labels: list[str]) -> tuple[Optional[str], str]:
        """Extract answer label and reason. Returns (label, reason).

        If parsing fails, label is None and reason is the full output.
        """
        text = text.strip()
        labels_upper = [l.upper() for l in option_labels]

        # 1. Try JSON extraction
        try:
            parsed = json.loads(text)
            answer = parsed.get("answer", "").strip().upper()
            reason = parsed.get("reason", "")
            if answer in labels_upper:
                return answer, reason
        except (json.JSONDecodeError, AttributeError):
            pass

        # 2. Try JSON embedded in text
        m = re.search(r'\{[^}]*"answer"\s*:\s*"([^"]+)"[^}]*\}', text)
        if m:
            answer = m.group(1).strip().upper()
            # Try to extract reason too
            r = re.search(r'"reason"\s*:\s*"([^"]*)"', text)
            reason = r.group(1) if r else ""
            if answer in labels_upper:
                return answer, reason

        # 3. "the answer is X" / "correct answer is X" patterns
        m = re.search(r'(?:the\s+)?(?:correct\s+)?answer\s+is\s+([A-Z])\b', text, re.IGNORECASE)
        if m:
            answer = m.group(1).upper()
            if answer in labels_upper:
                return answer, text

        # 4. Exact match (entire text is just the label)
        if text.upper() in labels_upper:
            return text.upper(), ""

        # 5. Text starts with label followed by delimiter
        for label in option_labels:
            if text.upper().startswith(label.upper()):
                rest = text[len(label):]
                if not rest or rest[0] in " .),:;\n":
                    return label, rest.strip()

        # No match — full output goes in reason
        return None, text
