"""Tests for SmolVLM2 JSON answer instruction styles."""

from levante_bench.models.smolvlm2 import SmolVLM2Model


def test_smolvlm2_json_instruction_uses_placeholder_by_default() -> None:
    model = SmolVLM2Model(json_answer_placeholder=True)
    model.use_json_format = True
    prompt, _, _, _ = model._prepare_trial_inputs(
        {
            "prompt": "Choose the answer.",
            "answer_format": "label",
            "context_image_paths": [],
            "option_image_paths": [],
        }
    )
    assert '"answer": "<letter>"' in prompt


def test_smolvlm2_json_instruction_can_use_explicit_letter_example() -> None:
    model = SmolVLM2Model(json_answer_placeholder=False)
    model.use_json_format = True
    prompt, _, _, _ = model._prepare_trial_inputs(
        {
            "prompt": "Choose the answer.",
            "answer_format": "label",
            "context_image_paths": [],
            "option_image_paths": [],
        }
    )
    assert '"answer": "A"' in prompt
    assert '"answer": "<letter>"' not in prompt
