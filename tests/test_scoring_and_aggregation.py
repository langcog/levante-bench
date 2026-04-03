"""Unit tests for trial scoring and by-type aggregation outputs."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from levante_bench.evaluation.adapters import _write_math_by_type, _write_tom_by_type
from levante_bench.models.base import VLMModel


class MockModel(VLMModel):
    """Test double that returns a fixed raw output."""

    def __init__(self, raw_output: str) -> None:
        super().__init__(model_name="mock", device="cpu")
        self.raw_output = raw_output

    def load(self) -> None:  # pragma: no cover - not used in these unit tests
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


class PromptCapturingModel(MockModel):
    def __init__(self, raw_output: str) -> None:
        super().__init__(raw_output=raw_output)
        self.last_prompt_text: str | None = None

    def generate(
        self,
        prompt_text: str,
        image_paths: list[str] | None = None,
        max_new_tokens: int = 64,
    ) -> str:
        self.last_prompt_text = prompt_text
        return super().generate(
            prompt_text=prompt_text,
            image_paths=image_paths,
            max_new_tokens=max_new_tokens,
        )


def test_evaluate_trial_label_is_correct_true() -> None:
    model = MockModel('{"answer":"B"}')
    trial = {
        "trial_id": "t1",
        "item_uid": "u1",
        "prompt": "pick one",
        "option_labels": ["A", "B", "C", "D"],
        "correct_label": "B",
        "answer_format": "label",
    }

    result = model.evaluate_trial(trial)
    assert result["predicted_label"] == "B"
    assert result["parse_method"] in {"strict_json", "embedded_json_answer"}
    assert result["parse_confidence"] in {"high", "medium"}
    assert result["is_correct"] is True


def test_evaluate_trial_numeric_is_correct_false() -> None:
    model = MockModel('{"answer":"3.2"}')
    trial = {
        "trial_id": "t2",
        "item_uid": "u2",
        "prompt": "number?",
        "answer_format": "numeric",
        "target_value": 5.0,
        "slider_tolerance": 0.1,
    }

    result = model.evaluate_trial(trial)
    assert result["predicted_value"] == pytest.approx(3.2)
    assert result["parse_method"] in {"json", "strict_json", "embedded_json_answer"}
    assert result["is_correct"] is False


def test_evaluate_trial_slider_clamps_position_to_unit_interval() -> None:
    model = MockModel("1.5")
    trial = {
        "trial_id": "t3",
        "item_uid": "u3",
        "prompt": "slider",
        "answer_format": "slider_position",
        "slider_min": 10.0,
        "slider_max": 20.0,
        "target_value": 20.0,
        "slider_tolerance": 0.01,
    }

    result = model.evaluate_trial(trial)
    assert result["predicted_slider_position"] == pytest.approx(1.0)
    assert result["predicted_value"] == pytest.approx(20.0)
    assert result["is_correct"] is True


def test_evaluate_trial_label_without_json_format_instruction() -> None:
    model = PromptCapturingModel("B")
    model.use_json_format = False
    trial = {
        "trial_id": "t4",
        "item_uid": "u4",
        "prompt": "choose one",
        "option_labels": ["A", "B", "C", "D"],
        "correct_label": "B",
        "answer_format": "label",
    }

    result = model.evaluate_trial(trial)
    assert model.last_prompt_text == "choose one"
    assert result["predicted_label"] == "B"
    assert result["is_correct"] is True


def _read_rows(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_write_math_by_type_outputs_expected_metrics(tmp_path: Path) -> None:
    task_results = [
        {"is_correct": True, "predicted_label": "A", "predicted_value": None},
        {"is_correct": False, "predicted_label": None, "predicted_value": None},
        {"is_correct": False, "predicted_label": "B", "predicted_value": None},
    ]
    task_trials = [
        {"trial_type": "algebra", "options": ["A", "B", "C", "D"]},
        {"trial_type": "algebra", "options": ["A", "B", "C", "D"]},
        {"trial_type": "geometry", "options": ["A", "B"], "chance_level": 0.5},
    ]

    out = _write_math_by_type(tmp_path, task_results, task_trials)
    assert out is not None and out.exists()

    rows = _read_rows(out)
    assert rows[0]["trial_type"] == "algebra"
    assert rows[1]["trial_type"] == "geometry"
    by_type = {r["trial_type"]: r for r in rows}

    # algebra: 1/2 correct, parse 1/2, chance 1/4
    assert by_type["algebra"]["n"] == "2"
    assert float(by_type["algebra"]["accuracy"]) == pytest.approx(0.5)
    assert float(by_type["algebra"]["guess_baseline"]) == pytest.approx(0.25)
    assert float(by_type["algebra"]["lift_vs_guess"]) == pytest.approx(0.25)
    assert float(by_type["algebra"]["parse_rate"]) == pytest.approx(0.5)

    # geometry: 0/1 correct, parse 1/1, chance comes from chance_level
    assert by_type["geometry"]["n"] == "1"
    assert float(by_type["geometry"]["accuracy"]) == pytest.approx(0.0)
    assert float(by_type["geometry"]["guess_baseline"]) == pytest.approx(0.5)
    assert float(by_type["geometry"]["lift_vs_guess"]) == pytest.approx(-0.5)
    assert float(by_type["geometry"]["parse_rate"]) == pytest.approx(1.0)


def test_write_tom_by_type_mismatched_lengths_returns_none(tmp_path: Path) -> None:
    out = _write_tom_by_type(
        tmp_path,
        task_results=[{"is_correct": True, "predicted_label": "A"}],
        task_trials=[],
    )
    assert out is None
