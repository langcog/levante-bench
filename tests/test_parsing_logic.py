"""Unit tests for answer parsing logic in VLMModel."""

from __future__ import annotations

import pytest

from levante_bench.models.base import VLMModel
from levante_bench.models.internvl35 import InternVL35Model
from levante_bench.models.qwen35 import Qwen35Model


@pytest.mark.parametrize(
    "text,labels,expected",
    [
        ('{"answer": "B", "reason": "because"}', ["A", "B", "C", "D"], "B"),
        ('{"answer": "b"}', ["A", "B", "C", "D"], "B"),
        ('{"answer":"C","reason":"truncated"', ["A", "B", "C", "D"], "C"),
        ("The correct answer is C.", ["A", "B", "C", "D"], "C"),
        ("The correct option is D.", ["A", "B", "C", "D"], "D"),
        ("Answer: A", ["A", "B", "C", "D"], "A"),
        ("my answer=B", ["A", "B", "C", "D"], "B"),
        ("B.", ["A", "B", "C", "D"], "B"),
        ("B) The largest one", ["A", "B", "C", "D"], "B"),
        ("The correct answer is A bird.", ["A", "B", "C", "D"], None),
        ("Category B is wrong.", ["A", "B", "C", "D"], None),
        ("I think the answer might be A bird", ["A", "B", "C", "D"], None),
        ("None of the above", ["A", "B", "C", "D"], None),
        ("", ["A", "B", "C", "D"], None),
    ],
)
def test_parse_answer_branches(text: str, labels: list[str], expected: str | None) -> None:
    model = VLMModel(model_name="dummy")
    label, _ = model.parse_answer(text, labels)
    assert label == expected


@pytest.mark.parametrize(
    "text,strict_json,slider_mode,expected",
    [
        ("0.75", False, False, 0.75),
        ("-1.5", False, False, -1.5),
        ('{"answer": 2.5, "reason":"ok"}', False, False, 2.5),
        ('{"answer":"-3.0"}', False, False, -3.0),
        ('noise {"answer":"4.25"} trailing', False, False, 4.25),
        ("score is 7.0 now", False, False, 7.0),
        ('{"answer":{"value": 1.2}}', True, False, None),
        ('{"answer":"1.2"}', True, False, 1.2),
        ("1.5", False, True, 1.5),
        ('{"answer":"0.25"}', False, True, 0.25),
        ("answer is 0.40", False, True, 0.40),
        ("nonsense", False, True, None),
    ],
)
def test_parse_numeric_answer_modes(
    text: str,
    strict_json: bool,
    slider_mode: bool,
    expected: float | None,
) -> None:
    model = VLMModel(model_name="dummy")
    value, _ = model.parse_numeric_answer(
        text, strict_json=strict_json, slider_mode=slider_mode
    )
    if expected is None:
        assert value is None
    else:
        assert value == pytest.approx(expected)


def test_parse_answer_v2_includes_provenance() -> None:
    model = VLMModel(model_name="dummy")
    result = model.parse_answer_v2('{"answer":"b","reason":"because"}', ["A", "B", "C", "D"])
    assert result.value == "B"
    assert result.parse_method == "strict_json"
    assert result.parse_confidence == "high"


def test_parse_numeric_v2_includes_provenance() -> None:
    model = VLMModel(model_name="dummy")
    result = model.parse_numeric_v2('{"answer":"2.75"}', strict_json=True)
    assert result.value == pytest.approx(2.75)
    assert result.parse_method == "strict_json"
    assert result.parse_confidence == "high"


@pytest.mark.parametrize("model_cls", [Qwen35Model, InternVL35Model])
def test_model_specific_parse_answer_prefers_last_sentence(model_cls) -> None:
    model = model_cls(model_name="dummy")
    label, reason = model.parse_answer(
        "The shape looks like a circle. B.",
        ["A", "B", "C", "D"],
    )
    assert label == "B"
    assert reason in {"", "B"}


@pytest.mark.parametrize("model_cls", [Qwen35Model, InternVL35Model])
def test_model_specific_parse_answer_reverse_scan_handles_conflict(model_cls) -> None:
    model = model_cls(model_name="dummy")
    label, _ = model.parse_answer(
        "Option B is wrong. Therefore A",
        ["A", "B", "C", "D"],
    )
    assert label == "A"
