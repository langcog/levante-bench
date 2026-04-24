"""Base class for VLM evaluation models."""

from dataclasses import dataclass
import json
import re
import time
from typing import Any, Literal, Optional

import json_repair


SYSTEM_PROMPT = "You are a helpful assistant."

# ── Format instructions (JSON path) ────────────────────────────────────────
# Label instructions are templates: {labels} is substituted with a prose
# enumeration of the trial's option_labels (e.g., "A, B, C, or D").
ANSWER_FORMAT_INSTRUCTION = (
    '\n\nRespond in JSON format: {{"answer": "<one of {labels}>", "reason": "<short reason>"}}'
)
NUMERIC_ANSWER_FORMAT_INSTRUCTION = (
    '\n\nRespond in JSON format: {"answer": "<number>", "reason": "<short reason>"}'
)
SLIDER_POSITION_FORMAT_INSTRUCTION = (
    "\n\nOutput only one decimal number between 0 and 1 "
    "(the slider position from left to right). No JSON or extra text."
)

# Short reinforcement prepended to the user message so small models see the
# output-format constraint before the task framing (they often drift into CoT
# if it only appears as a trailing suffix).
LABEL_FORMAT_PREFIX = (
    'Respond only with the JSON object '
    '{{"answer":"<one of {labels}>","reason":"<short reason>"}}. No preamble.\n\n'
)
NUMERIC_FORMAT_PREFIX = (
    'Respond only with the JSON object '
    '{"answer":"<number>","reason":"<short reason>"}. No preamble.\n\n'
)
SLIDER_FORMAT_PREFIX = "Respond only with a single decimal between 0 and 1. No preamble.\n\n"

# ── Format instructions (simple / non-JSON path) ───────────────────────────
# Used when the model's YAML sets `use_json_format: false`. Designed for
# very small models (e.g., SmolVLM2-500M) that echo JSON template placeholders
# like `<letter>` instead of filling them in. The letters list is substituted
# from the trial's option_labels so 2-option tasks say "A or B" and 4-option
# tasks say "A, B, C, or D".
LABEL_SIMPLE_INSTRUCTION = "\n\nAnswer with only the letter {labels}."
NUMERIC_SIMPLE_INSTRUCTION = "\n\nAnswer with only a number."

# Default token budgets per answer format. Label/numeric allow a bit of CoT
# before the JSON object; slider just needs a scalar.
DEFAULT_MAX_NEW_TOKENS = {
    "label": 128,
    "numeric": 128,
    "slider_position": 32,
}


def _try_json_repair(text: str) -> Any:
    """Attempt to parse possibly-malformed JSON with json-repair.

    Returns the parsed value (typically a dict) or ``None`` if repair failed
    or produced an empty string (which json-repair uses to signal "no JSON").
    """
    if not text:
        return None
    try:
        parsed = json_repair.loads(text)
    except Exception:
        return None
    if parsed == "" or parsed == {} or parsed == []:
        return None
    return parsed


def _format_labels_prose(option_labels: list[str]) -> str:
    """Render option labels as a natural-language list for prompt templates.

    ['A'] → 'A'; ['A', 'B'] → 'A or B'; ['A', 'B', 'C', 'D'] → 'A, B, C, or D'.
    Falls back to 'A, B, C, or D' if no labels are supplied (defensive).
    """
    if not option_labels:
        return "A, B, C, or D"
    labels = [str(l) for l in option_labels]
    if len(labels) == 1:
        return labels[0]
    if len(labels) == 2:
        return f"{labels[0]} or {labels[1]}"
    return ", ".join(labels[:-1]) + f", or {labels[-1]}"


def _coerce_numeric(value: Any) -> Optional[float]:
    """Coerce a repaired answer value to float, or return None."""
    if value is None or isinstance(value, (dict, list, tuple)):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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
        prompt, answer_format, image_paths, max_new_tokens = self._prepare_trial_inputs(trial)
        raw_output = self.generate(
            prompt_text=prompt,
            image_paths=image_paths if image_paths else None,
            max_new_tokens=max_new_tokens,
        )
        clean_text = self.parse_response(raw_output)
        return self._build_result_from_text(
            trial=trial,
            clean_text=clean_text,
            answer_format=answer_format,
        )

    def _prepare_trial_inputs(self, trial: dict) -> tuple[str, str, list[str], int]:
        """Build canonical prompt/input payload for a trial.

        Prompt construction follows two axes:
          - answer_format: label | numeric | slider_position
          - self.use_json_format: True → JSON template; False → simple instruction
            (terse, letter/number-only; used for tiny models that echo JSON
            placeholders rather than filling them in).
        Label instructions are option-aware: the trial's option_labels are
        substituted into {labels} (e.g., "A or B" for mental-rotation,
        "A, B, C, or D" for vocab).
        """
        prompt = trial["prompt"]
        answer_format = str(trial.get("answer_format", "label")).strip().lower()
        labels_prose = _format_labels_prose(trial.get("option_labels", []))

        if self.use_json_format:
            if answer_format == "slider_position":
                prompt = SLIDER_FORMAT_PREFIX + prompt + SLIDER_POSITION_FORMAT_INSTRUCTION
            elif answer_format == "numeric":
                prompt = NUMERIC_FORMAT_PREFIX + prompt + NUMERIC_ANSWER_FORMAT_INSTRUCTION
            else:
                prompt = (
                    LABEL_FORMAT_PREFIX.format(labels=labels_prose)
                    + prompt
                    + ANSWER_FORMAT_INSTRUCTION.format(labels=labels_prose)
                )
        else:
            if answer_format == "slider_position":
                prompt = prompt + SLIDER_POSITION_FORMAT_INSTRUCTION
            elif answer_format == "numeric":
                prompt = prompt + NUMERIC_SIMPLE_INSTRUCTION
            else:
                prompt = prompt + LABEL_SIMPLE_INSTRUCTION.format(labels=labels_prose)

        image_paths = trial.get("context_image_paths", []) + trial.get("option_image_paths", [])
        default_budget = DEFAULT_MAX_NEW_TOKENS.get(answer_format, 128)
        max_new_tokens = int(trial.get("max_new_tokens", default_budget))
        return prompt, answer_format, image_paths, max_new_tokens

    def _build_result_from_text(
        self,
        trial: dict,
        clean_text: str,
        answer_format: str,
    ) -> dict:
        """Convert parsed model text into the canonical benchmark result row."""
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

    def evaluate_trials_batch(self, trials: list[dict]) -> list[dict]:
        """Evaluate a batch of trials.

        Default implementation preserves existing behavior by evaluating each
        trial independently. Model adapters can override this for true batched
        generation.
        """
        return [self.evaluate_trial(trial) for trial in trials]

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
                return ParseResult(
                    value=float(text),
                    reason=text,
                    parse_method="slider_scalar",
                    parse_confidence="high",
                    raw_candidate=text,
                )

            parsed = _try_json_repair(text)
            if isinstance(parsed, dict) and "answer" in parsed:
                numeric = _coerce_numeric(parsed.get("answer"))
                if numeric is not None:
                    return ParseResult(
                        value=numeric,
                        reason=str(parsed.get("reason", "")),
                        parse_method="slider_json",
                        parse_confidence="high",
                        raw_candidate=str(parsed.get("answer")),
                    )

            # Explicit "answer is X" prose (still semi-strict: must name "answer").
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

        parsed = _try_json_repair(text)
        if isinstance(parsed, dict) and "answer" in parsed:
            answer = parsed.get("answer")
            if strict_json and isinstance(answer, (dict, list, tuple)):
                return ParseResult(
                    value=None,
                    reason=text,
                    parse_method="strict_json_rejected_nested",
                    parse_confidence="none",
                )
            numeric = _coerce_numeric(answer)
            if numeric is not None:
                return ParseResult(
                    value=numeric,
                    reason=str(parsed.get("reason", "")),
                    parse_method="strict_json" if strict_json else "json_repair",
                    parse_confidence="high",
                    raw_candidate=str(answer),
                )

        if strict_json:
            return ParseResult(
                value=None,
                reason=text,
                parse_method="unparseable",
                parse_confidence="none",
            )

        # Last-resort: first standalone number anywhere in the text.
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
        """Extract canonical answer label with parse provenance.

        Layers, first hit wins:
          1. json-repair (handles strict JSON, markdown fences, trailing prose,
             single quotes, unquoted keys, trailing commas, truncated JSON,
             unescaped inner quotes).
          2. Natural-language phrases (``The answer is B``, ``Final answer: C``).
          3. Trailing sentence that is itself a lone label (``…reasoning. B.``).
          4. Single label wrapped in noise (``; A :``).
          5. Exact label (``A``).
          6. Prefix label followed by delimiter (``B) because…``).
        """
        text = text.strip()
        labels_upper = [l.upper() for l in option_labels]

        # 1. json-repair layer.
        parsed = _try_json_repair(text)
        if isinstance(parsed, dict) and "answer" in parsed:
            raw_answer = parsed.get("answer", "")
            if not isinstance(raw_answer, (dict, list, tuple)):
                answer = str(raw_answer).strip().upper()
                if answer in labels_upper:
                    return ParseResult(
                        value=answer,
                        reason=str(parsed.get("reason", "")),
                        parse_method="json_repair",
                        parse_confidence="high",
                        raw_candidate=str(raw_answer),
                    )

        # 2. Natural-language phrases. The label terminator must be punctuation,
        # end-of-text, or a connector word ("because", "since", "as", "so"),
        # so we reject "The correct answer is A bird." but accept
        # "The answer is B because …".
        label_terminator = (
            r"(?=\s*(?:[)\].,:;!?]|$|\b(?:because|since|as|so|and|therefore)\b))"
        )
        phrase_patterns = (
            rf"\b(?:the\s+)?(?:correct\s+)?answer\s+is\s+(?P<label>[A-Z]){label_terminator}",
            rf"\b(?:the\s+)?(?:correct\s+)?option\s+is\s+(?P<label>[A-Z]){label_terminator}",
            rf"\b(?:my\s+)?answer\s*[:=]\s*(?P<label>[A-Z]){label_terminator}",
            rf"\b(?:the\s+)?(?:correct\s+)?option\s*[:=]\s*(?P<label>[A-Z]){label_terminator}",
            rf"\b(?:final\s+)?answer\s*(?:is|:|=|->|=>|-)\s*\(?\s*(?P<label>[A-Z])\s*\)?{label_terminator}",
            rf"\b(?:choose|pick|select)\s+(?:option\s+)?(?P<label>[A-Z]){label_terminator}",
            rf"\boption\s+(?P<label>[A-Z])(?:\s+is\s+correct)?{label_terminator}",
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

        # 2.5: Last affirmative option mention (for verbose reasoning models).
        # Catches "Option C matches the pattern" or "Option A is correct" buried
        # in extended CoT.  We take the *last* match to favour the model's final
        # conclusion over earlier speculative mentions.
        affirmative_pattern = re.compile(
            r'\boption\s+(?P<label>[A-Z])\b\s+(?:matches|fits|shows|is\s+(?:the\s+)?'
            r'correct|is\s+the\s+answer|corresponds|completes|seems?\s+correct|'
            r'looks?\s+correct|appears?\s+correct|works)',
            re.IGNORECASE,
        )
        aff_matches = list(affirmative_pattern.finditer(text))
        if aff_matches:
            last = aff_matches[-1]
            answer = last.group("label").upper()
            if answer in labels_upper:
                return ParseResult(
                    value=answer,
                    reason=text,
                    parse_method="affirmative_option_mention",
                    parse_confidence="low",
                    raw_candidate=last.group("label"),
                )

        # 3. Trailing sentence is itself a lone label (with optional punctuation).
        # Catches "…chain of thought. B." without scanning for arbitrary label
        # mentions earlier in the text.
        sentences = [s.strip() for s in re.split(r"[.!?\n]", text) if s.strip()]
        if sentences:
            m = re.match(
                r"^[\s\W_]*(?P<label>[A-Z])[\s\W_]*$",
                sentences[-1],
                re.IGNORECASE,
            )
            if m:
                answer = m.group("label").upper()
                if answer in labels_upper and len(sentences) > 1:
                    return ParseResult(
                        value=answer,
                        reason=text,
                        parse_method="trailing_sentence_label",
                        parse_confidence="medium",
                        raw_candidate=m.group("label"),
                    )

        # 4. Single label wrapped by punctuation/noise (e.g., "; A:").
        m = re.search(r"^[\s\W_]*(?P<label>[A-Z])[\s\W_]*$", text, re.IGNORECASE)
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

        # 5. Exact match (entire text is just the label).
        if text.upper() in labels_upper:
            return ParseResult(
                value=text.upper(),
                reason="",
                parse_method="exact_label",
                parse_confidence="high",
                raw_candidate=text,
            )

        # 6. Text starts with label followed by delimiter.
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

        return ParseResult(
            value=None,
            reason=text,
            parse_method="unparseable",
            parse_confidence="none",
        )
