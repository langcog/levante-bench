"""Base class for VLM evaluation models."""

from dataclasses import dataclass
import json
import re
import time
from typing import Any, Literal, Optional


ANSWER_FORMAT_INSTRUCTION = '\n\nRespond in JSON format: {"answer": "<letter>", "reason": "<short reason>"}'
NUMERIC_ANSWER_FORMAT_INSTRUCTION = '\n\nRespond in JSON format: {"answer": "<number>", "reason": "<short reason>"}'
SLIDER_POSITION_FORMAT_INSTRUCTION = (
    "\n\nOutput only one decimal number between 0 and 1 "
    "(the slider position from left to right). No JSON or extra text."
)


@dataclass(frozen=True)
class ParseResult:
    """Canonical parse output with provenance for auditing."""

    value: Any
    reason: str
    parse_method: str
    parse_confidence: Literal["high", "medium", "low", "none"]
    raw_candidate: str = ""


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
        self.use_json_format = True

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

    def score_choices(
        self,
        prompt_text: str,
        image_paths: list[str],
        choice_texts: tuple[str, str] = ("1", "2"),
    ) -> dict:
        """Return next-token probabilities/logits for a fixed choice set.

        Models that support logit-forced evaluation should override this method.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement score_choices()."
        )

    @staticmethod
    def _timed_call(callable_fn):
        """Execute callable and return (result, elapsed_seconds)."""
        start = time.perf_counter()
        output = callable_fn()
        elapsed = time.perf_counter() - start
        return output, elapsed

    def evaluate_trial(self, trial: dict) -> dict:
        """Run a single trial: generate answer, parse it, return result."""
        prompt = trial["prompt"]
        answer_format = str(trial.get("answer_format", "label")).strip().lower()
        if self.use_json_format:
            if answer_format == "slider_position":
                prompt += SLIDER_POSITION_FORMAT_INSTRUCTION
            elif answer_format == "numeric":
                prompt += NUMERIC_ANSWER_FORMAT_INSTRUCTION
            else:
                prompt += ANSWER_FORMAT_INSTRUCTION
        image_paths = trial.get("context_image_paths", []) + trial.get("option_image_paths", [])
        raw_output = self.generate(
            prompt_text=prompt,
            image_paths=image_paths if image_paths else None,
            max_new_tokens=trial.get("max_new_tokens", 64),
        )
        clean_text = self.parse_response(raw_output)
        if answer_format in {"numeric", "slider_position"}:
            strict_json = answer_format == "slider_position"
            numeric_result = self.parse_numeric_result(
                clean_text,
                strict_json=strict_json,
                slider_mode=(answer_format == "slider_position"),
            )
            predicted_value = (
                float(numeric_result.value)
                if numeric_result.value is not None
                else None
            )
            reason = numeric_result.reason
            target_value = trial.get("target_value")
            tolerance = trial.get("slider_tolerance")
            predicted_position = None
            if answer_format == "slider_position" and predicted_value is not None:
                slider_min = trial.get("slider_min")
                slider_max = trial.get("slider_max")
                if slider_min is not None and slider_max is not None:
                    span = float(slider_max) - float(slider_min)
                    if span > 0:
                        # Clamp to [0,1] so out-of-range answers map to slider edges.
                        predicted_position = max(0.0, min(1.0, float(predicted_value)))
                        predicted_value = float(slider_min) + (predicted_position * span)
            is_correct = (
                predicted_value is not None
                and target_value is not None
                and tolerance is not None
                and abs(predicted_value - float(target_value)) < float(tolerance)
            )
            return {
                "trial_id": trial["trial_id"],
                "item_uid": trial["item_uid"],
                "generated_text": clean_text,
                "predicted_label": None,
                "predicted_value": predicted_value,
                "predicted_slider_position": predicted_position,
                "reason": reason,
                "parse_method": numeric_result.parse_method,
                "parse_confidence": numeric_result.parse_confidence,
                "parse_raw_candidate": numeric_result.raw_candidate,
                "correct_label": None,
                "target_value": target_value,
                "slider_tolerance": tolerance,
                "is_correct": is_correct,
                # Carry option context for downstream annotation
                "options": trial.get("options", []),
                "option_labels": trial.get("option_labels", []),
            }

        answer_result = self.parse_answer_result(clean_text, trial["option_labels"])
        predicted_label = (
            str(answer_result.value).upper() if answer_result.value is not None else None
        )
        reason = answer_result.reason
        return {
            "trial_id": trial["trial_id"],
            "item_uid": trial["item_uid"],
            "generated_text": clean_text,
            "predicted_label": predicted_label,
            "reason": reason,
            "parse_method": answer_result.parse_method,
            "parse_confidence": answer_result.parse_confidence,
            "parse_raw_candidate": answer_result.raw_candidate,
            "correct_label": trial["correct_label"],
            "is_correct": predicted_label == trial["correct_label"],
            # Carry option context for downstream human-comparison annotation
            "options": trial.get("options", []),
            "option_labels": trial.get("option_labels", []),
        }

    def parse_numeric_answer(
        self,
        text: str,
        strict_json: bool = False,
        slider_mode: bool = False,
    ) -> tuple[Optional[float], str]:
        """Backward-compatible numeric parser API: (value, reason)."""
        result = self.parse_numeric_result(
            text,
            strict_json=strict_json,
            slider_mode=slider_mode,
        )
        value = float(result.value) if result.value is not None else None
        return value, result.reason

    def parse_numeric_v2(
        self,
        text: str,
        strict_json: bool = False,
        slider_mode: bool = False,
    ) -> ParseResult:
        """Alias for parse_numeric_result for explicit v2 callers."""
        return self.parse_numeric_result(
            text,
            strict_json=strict_json,
            slider_mode=slider_mode,
        )

    def parse_numeric_result(
        self,
        text: str,
        strict_json: bool = False,
        slider_mode: bool = False,
    ) -> ParseResult:
        """Extract canonical numeric value with parse provenance."""
        text = text.strip()

        if slider_mode:
            # Slider mode is "semi-strict": accept only explicit scalar forms,
            # never first-number fallback from arbitrary prose.
            if re.fullmatch(r"[-+]?\d*\.?\d+", text):
                try:
                    return ParseResult(
                        value=float(text),
                        reason=text,
                        parse_method="slider_scalar",
                        parse_confidence="high",
                        raw_candidate=text,
                    )
                except ValueError:
                    pass

            try:
                parsed = json.loads(text)
                answer = parsed.get("answer")
                reason = str(parsed.get("reason", ""))
                if answer is not None and not isinstance(answer, (dict, list, tuple)):
                    return ParseResult(
                        value=float(answer),
                        reason=reason,
                        parse_method="slider_json",
                        parse_confidence="high",
                        raw_candidate=str(answer),
                    )
            except (json.JSONDecodeError, TypeError, ValueError, AttributeError):
                pass

            for pattern in (
                r'answer\s*(?:is|:)\s*"?(?P<num>[-+]?\d*\.?\d+)"?',
                r'"answer"\s*:\s*"?(?P<num>[-+]?\d*\.?\d+)"?',
            ):
                m = re.search(pattern, text, re.IGNORECASE)
                if m:
                    try:
                        return ParseResult(
                            value=float(m.group("num")),
                            reason=text,
                            parse_method="slider_explicit_pattern",
                            parse_confidence="medium",
                            raw_candidate=m.group("num"),
                        )
                    except ValueError:
                        pass
            return ParseResult(
                value=None,
                reason=text,
                parse_method="unparseable",
                parse_confidence="none",
            )

        # 1) Plain JSON
        try:
            parsed = json.loads(text)
            answer = parsed.get("answer")
            reason = str(parsed.get("reason", ""))
            if answer is not None:
                # In strict mode, accept only scalar numeric answers.
                if strict_json and isinstance(answer, (dict, list, tuple)):
                    return ParseResult(
                        value=None,
                        reason=text,
                        parse_method="strict_json_rejected_nested",
                        parse_confidence="none",
                    )
                return ParseResult(
                    value=float(answer),
                    reason=reason,
                    parse_method="strict_json" if strict_json else "json",
                    parse_confidence="high",
                    raw_candidate=str(answer),
                )
        except (json.JSONDecodeError, TypeError, ValueError, AttributeError):
            pass

        if strict_json:
            # Strict mode accepts only explicit scalar answer fields.
            m = re.search(r'"answer"\s*:\s*"?(?P<num>[-+]?\d*\.?\d+)"?', text)
            if m:
                try:
                    return ParseResult(
                        value=float(m.group("num")),
                        reason=text,
                        parse_method="strict_json_embedded_answer",
                        parse_confidence="medium",
                        raw_candidate=m.group("num"),
                    )
                except ValueError:
                    pass
            return ParseResult(
                value=None,
                reason=text,
                parse_method="unparseable",
                parse_confidence="none",
            )

        # 2) Embedded JSON answer
        m = re.search(r'"answer"\s*:\s*"?(?P<num>[-+]?\d*\.?\d+)"?', text)
        if m:
            try:
                return ParseResult(
                    value=float(m.group("num")),
                    reason=text,
                    parse_method="embedded_json_answer",
                    parse_confidence="medium",
                    raw_candidate=m.group("num"),
                )
            except ValueError:
                pass

        # 3) First standalone number fallback
        m = re.search(r"(?P<num>[-+]?\d*\.?\d+)", text)
        if m:
            try:
                return ParseResult(
                    value=float(m.group("num")),
                    reason=text,
                    parse_method="first_number_fallback",
                    parse_confidence="low",
                    raw_candidate=m.group("num"),
                )
            except ValueError:
                pass

        return ParseResult(
            value=None,
            reason=text,
            parse_method="unparseable",
            parse_confidence="none",
        )

    def parse_answer(self, text: str, option_labels: list[str]) -> tuple[Optional[str], str]:
        """Backward-compatible label parser API: (label, reason)."""
        result = self.parse_answer_result(text, option_labels)
        label = str(result.value).upper() if result.value is not None else None
        return label, result.reason

    def parse_answer_v2(self, text: str, option_labels: list[str]) -> ParseResult:
        """Alias for parse_answer_result for explicit v2 callers."""
        return self.parse_answer_result(text, option_labels)

    def parse_answer_result(self, text: str, option_labels: list[str]) -> ParseResult:
        """Extract canonical answer label with parse provenance."""
        text = text.strip()
        labels_upper = [l.upper() for l in option_labels]

        # 1. Try JSON extraction
        try:
            parsed = json.loads(text)
            answer = parsed.get("answer", "").strip().upper()
            reason = parsed.get("reason", "")
            if answer in labels_upper:
                return ParseResult(
                    value=answer,
                    reason=reason,
                    parse_method="strict_json",
                    parse_confidence="high",
                    raw_candidate=str(parsed.get("answer", "")),
                )
        except (json.JSONDecodeError, AttributeError):
            pass

        # 2. Embedded/truncated JSON answer field (works on incomplete JSON too).
        m = re.search(r'"answer"\s*:\s*"?(?P<label>[A-Z])\b', text, re.IGNORECASE)
        if m:
            answer = m.group("label").upper()
            if answer in labels_upper:
                r = re.search(r'"reason"\s*:\s*"(?P<reason>[^"]*)"', text, re.IGNORECASE)
                reason = r.group("reason") if r else text
                return ParseResult(
                    value=answer,
                    reason=reason,
                    parse_method="embedded_json_answer",
                    parse_confidence="medium",
                    raw_candidate=m.group("label"),
                )

        # 3. Common natural-language patterns.
        # Require punctuation/end-of-text after the label to avoid false
        # positives like "The correct answer is A bird."
        label_suffix = r"(?=\s*(?:[)\].,:;!?]|$))"
        phrase_patterns = (
            rf"\b(?:the\s+)?(?:correct\s+)?answer\s+is\s+(?P<label>[A-Z]){label_suffix}",
            rf"\b(?:the\s+)?(?:correct\s+)?option\s+is\s+(?P<label>[A-Z]){label_suffix}",
            rf"\b(?:my\s+)?answer\s*[:=]\s*(?P<label>[A-Z]){label_suffix}",
            rf"\b(?:the\s+)?(?:correct\s+)?option\s*[:=]\s*(?P<label>[A-Z]){label_suffix}",
            rf"\b(?:final\s+)?answer\s*(?:is|:|=|->|=>|-)\s*\(?\s*(?P<label>[A-Z])\s*\)?{label_suffix}",
            rf"\b(?:choose|pick|select)\s+(?:option\s+)?(?P<label>[A-Z]){label_suffix}",
            rf"\boption\s+(?P<label>[A-Z])(?:\s+is\s+correct)?{label_suffix}",
        )
        for pattern in phrase_patterns:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                answer = m.group("label").upper()
                if answer in labels_upper:
                    return ParseResult(
                        value=answer,
                        reason=text,
                        parse_method="explicit_phrase",
                        parse_confidence="medium",
                        raw_candidate=m.group("label"),
                    )

        # 4. Single label wrapped by punctuation/noise (e.g., "; A:")
        m = re.search(
            r'^[\s\W_]*(?P<label>[A-Z])[\s\W_]*$',
            text,
            re.IGNORECASE,
        )
        if m:
            answer = m.group("label").upper()
            if answer in labels_upper:
                return ParseResult(
                    value=answer,
                    reason=text,
                    parse_method="punctuated_single_label",
                    parse_confidence="low",
                    raw_candidate=m.group("label"),
                )

        # 5. Exact match (entire text is just the label)
        if text.upper() in labels_upper:
            return ParseResult(
                value=text.upper(),
                reason="",
                parse_method="exact_label",
                parse_confidence="high",
                raw_candidate=text,
            )

        # 6. Text starts with label followed by delimiter
        for label in option_labels:
            if text.upper().startswith(label.upper()):
                rest = text[len(label):]
                if not rest or rest[0] in " .),:;\n":
                    return ParseResult(
                        value=label.upper(),
                        reason=rest.strip(),
                        parse_method="prefix_label",
                        parse_confidence="low",
                        raw_candidate=label,
                    )

        # No match — full output goes in reason
        return ParseResult(
            value=None,
            reason=text,
            parse_method="unparseable",
            parse_confidence="none",
        )
