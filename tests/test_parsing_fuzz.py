"""Property-based tests for parsing logic."""

from __future__ import annotations

import json

import pytest
from hypothesis import given, settings, strategies as st

from levante_bench.models.base import VLMModel


LABELS = ["A", "B", "C", "D"]


@settings(max_examples=80, deadline=None)
@given(
    label=st.sampled_from(LABELS),
    use_lower=st.booleans(),
    reason=st.text(min_size=0, max_size=40),
)
def test_parse_answer_json_fuzz(label: str, use_lower: bool, reason: str) -> None:
    model = VLMModel(model_name="dummy")
    variant = label.lower() if use_lower else label
    text = json.dumps({"answer": variant, "reason": reason})
    parsed, parsed_reason = model.parse_answer(text, LABELS)
    assert parsed == label
    assert parsed_reason == reason


@settings(max_examples=80, deadline=None)
@given(label=st.sampled_from(LABELS), prefix=st.text(min_size=0, max_size=20))
def test_parse_answer_prefix_delimiter_fuzz(label: str, prefix: str) -> None:
    model = VLMModel(model_name="dummy")
    text = f"{label}) {prefix}"
    parsed, _ = model.parse_answer(text, LABELS)
    assert parsed == label


@settings(max_examples=120, deadline=None)
@given(
    text=st.text(
        alphabet=st.sampled_from(list("efghijklmnopqrstuvwxyz0123456789 _-.,:;")),
        min_size=0,
        max_size=80,
    )
)
def test_parse_answer_non_label_text_returns_none(text: str) -> None:
    model = VLMModel(model_name="dummy")
    parsed, _ = model.parse_answer(text, LABELS)
    assert parsed is None


@settings(max_examples=100, deadline=None)
@given(num=st.floats(allow_nan=False, allow_infinity=False, width=32))
def test_parse_numeric_answer_plain_float_fuzz(num: float) -> None:
    model = VLMModel(model_name="dummy")
    text = f"{float(num):.8f}"
    parsed, _ = model.parse_numeric_answer(text, strict_json=False, slider_mode=False)
    assert parsed == pytest.approx(float(text))


@settings(max_examples=100, deadline=None)
@given(num=st.floats(allow_nan=False, allow_infinity=False, width=32))
def test_parse_numeric_answer_strict_json_fuzz(num: float) -> None:
    model = VLMModel(model_name="dummy")
    text = json.dumps({"answer": float(num), "reason": "ok"})
    parsed, _ = model.parse_numeric_answer(text, strict_json=True, slider_mode=False)
    assert parsed == pytest.approx(float(num))


class _SliderMockModel(VLMModel):
    def __init__(self, raw_output: str) -> None:
        super().__init__(model_name="mock", device="cpu")
        self.raw_output = raw_output

    def load(self) -> None:
        return None

    def generate(
        self,
        prompt_text: str,
        image_paths: list[str] | None = None,
        max_new_tokens: int = 64,
    ) -> str:
        return self.raw_output

    def _build_messages(
        self,
        prompt_text: str,
        image_paths: list[str] | None = None,
    ) -> list[dict]:
        return [{"role": "user", "content": [{"type": "text", "text": prompt_text}]}]


@settings(max_examples=80, deadline=None)
@given(
    pos=st.floats(
        min_value=-3,
        max_value=3,
        allow_subnormal=False,
        allow_nan=False,
        allow_infinity=False,
        width=32,
    )
)
def test_slider_position_clamp_fuzz(pos: float) -> None:
    model = _SliderMockModel(f"{float(pos):.8f}")
    trial = {
        "trial_id": "slider-fuzz",
        "item_uid": "u-slider",
        "prompt": "slider",
        "answer_format": "slider_position",
        "slider_min": 10.0,
        "slider_max": 30.0,
        "target_value": 10.0,
        "slider_tolerance": 100.0,
    }
    result = model.evaluate_trial(trial)
    assert result["predicted_slider_position"] is not None
    assert 0.0 <= result["predicted_slider_position"] <= 1.0
    assert 10.0 <= result["predicted_value"] <= 30.0
