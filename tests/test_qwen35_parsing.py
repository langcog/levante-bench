"""Unit tests for Qwen3.5-specific parsing fallbacks."""

from __future__ import annotations

from levante_bench.models.qwen35 import Qwen35Model


def test_qwen35_recovers_label_from_analyze_image_pattern() -> None:
    model = Qwen35Model(model_name="Qwen/Qwen3.5-4B")
    text = (
        'The user wants me to identify "apple".\n'
        "1. **Analyze Image A:** This is not an apple.\n"
        "2. **Analyze Image B:** This is not an apple.\n"
        "3. **Analyze Image C:** This is clearly an apple with a stem.\n"
        "4. **Analyze Image D:** This is not an apple.\n"
    )
    result = model.parse_answer_result(text, ["A", "B", "C", "D"])
    assert result.value == "C"
    assert result.parse_method == "qwen_analyze_image_heuristic"


def test_qwen35_keeps_unparseable_when_signal_is_ambiguous() -> None:
    model = Qwen35Model(model_name="Qwen/Qwen3.5-4B")
    text = (
        "The user asks to choose A or B.\n"
        "1. **Analyze Image A:** The image contains a rabbit shape.\n"
        "2. **Analyze Image B:** The image contains a rabbit shape too.\n"
    )
    result = model.parse_answer_result(text, ["A", "B"])
    assert result.value is None
    assert result.parse_method == "unparseable"
