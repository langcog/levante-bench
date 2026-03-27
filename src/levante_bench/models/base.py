"""Base class for VLM evaluation models."""

import re
from typing import Optional



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
        image_paths = trial.get("context_image_paths", []) + trial.get("option_image_paths", [])
        raw_output = self.generate(
            prompt_text=trial["prompt"],
            image_paths=image_paths if image_paths else None,
            max_new_tokens=trial.get("max_new_tokens", 64),
        )
        clean_text = self.parse_response(raw_output)
        predicted_label = self.parse_answer(clean_text, trial["option_labels"])
        return {
            "trial_id": trial["trial_id"],
            "item_uid": trial["item_uid"],
            "generated_text": clean_text,
            "predicted_label": predicted_label,
            "correct_label": trial["correct_label"],
            "is_correct": predicted_label == trial["correct_label"],
            # Carry option context for downstream human-comparison annotation
            "options": trial.get("options", []),
            "option_labels": trial.get("option_labels", []),
        }

    def parse_answer(self, text: str, option_labels: list[str]) -> Optional[str]:
        """Match clean text to an option label. Generic — same for all models."""
        text = text.strip()

        # Exact match
        for label in option_labels:
            if text.upper() == label.upper():
                return label

        # Starts with label followed by delimiter
        for label in option_labels:
            if text.upper().startswith(label.upper()):
                rest = text[len(label):]
                if not rest or rest[0] in " .),:;\n":
                    return label

        # First standalone label in text
        for label in option_labels:
            if re.search(rf'\b{re.escape(label)}\b', text, re.IGNORECASE):
                return label

        return None
