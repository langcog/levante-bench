"""API-model compatibility tests for the unified prompt templates.

Each API model (Gemini, GPT-5.3, HF-hosted) overrides only the payload-build
step — not `_prepare_trial_inputs`. So to verify the unified prompt flows
correctly, we call `_prepare_trial_inputs` to get the prompt string, pass it
to the model's builder, and inspect the returned payload. No HTTP, no mocks.
"""

from __future__ import annotations

from levante_bench.models.gemini import GeminiProModel
from levante_bench.models.gpt import GPT53Model
from levante_bench.models.hf_hosted import HFHostedModel


def _label_trial(option_labels: list[str]) -> dict:
    return {
        "trial_id": "t",
        "item_uid": "u",
        "prompt": "pick one",
        "option_labels": option_labels,
        "correct_label": option_labels[0],
        "answer_format": "label",
    }


def _prompt_for(model, trial: dict) -> str:
    prompt, *_ = model._prepare_trial_inputs(trial)
    return prompt


def _gemini_payload_text(model: GeminiProModel, prompt: str) -> str:
    parts = model._build_parts(prompt, image_paths=None)
    return "\n".join(p.get("text", "") for p in parts if isinstance(p, dict))


def _gpt_payload_text(model: GPT53Model, prompt: str) -> str:
    content = model._build_content(prompt, image_paths=None)
    return "\n".join(
        b.get("text", "")
        for b in content
        if isinstance(b, dict) and b.get("type") in {"text", "input_text"}
    )


def _hf_payload_text(model: HFHostedModel, prompt: str) -> str:
    messages = model._build_messages(prompt, image_paths=None)
    chunks: list[str] = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, str):
            chunks.append(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    chunks.append(str(item.get("text", "")))
    return "\n".join(chunks)


# ── Gemini ───────────────────────────────────────────────────────────────

def test_gemini_json_payload_has_option_aware_instruction() -> None:
    model = GeminiProModel(model_name="gemini-2.5-pro")
    prompt = _prompt_for(model, _label_trial(["A", "B"]))
    text = _gemini_payload_text(model, prompt)
    assert "<one of A or B>" in text
    assert text.count("<one of A or B>") == 2  # prefix + suffix
    assert "<one of A, B, C, or D>" not in text


def test_gemini_simple_mode_payload_has_letter_only_instruction() -> None:
    model = GeminiProModel(model_name="gemini-2.5-pro")
    model.use_json_format = False
    prompt = _prompt_for(model, _label_trial(["A", "B", "C", "D"]))
    text = _gemini_payload_text(model, prompt)
    assert "Answer with only the letter A, B, C, or D." in text
    assert '{"answer"' not in text


# ── GPT-5.3 (Responses API) ──────────────────────────────────────────────

def test_gpt53_json_payload_has_option_aware_instruction() -> None:
    model = GPT53Model(model_name="gpt-5.3", max_output_tokens_min=8)
    prompt = _prompt_for(model, _label_trial(["A", "B"]))
    text = _gpt_payload_text(model, prompt)
    assert "<one of A or B>" in text
    assert text.count("<one of A or B>") == 2
    assert "<one of A, B, C, or D>" not in text


def test_gpt53_simple_mode_payload_has_letter_only_instruction() -> None:
    model = GPT53Model(model_name="gpt-5.3", max_output_tokens_min=8)
    model.use_json_format = False
    prompt = _prompt_for(model, _label_trial(["A", "B", "C", "D"]))
    text = _gpt_payload_text(model, prompt)
    assert "Answer with only the letter A, B, C, or D." in text
    assert '{"answer"' not in text


# ── HF Hosted (OpenAI-compat chat endpoint) ──────────────────────────────

def test_hf_hosted_json_payload_has_option_aware_instruction() -> None:
    model = HFHostedModel(model_name="dummy/model")
    prompt = _prompt_for(model, _label_trial(["A", "B"]))
    text = _hf_payload_text(model, prompt)
    assert "<one of A or B>" in text
    assert text.count("<one of A or B>") == 2
    assert "<one of A, B, C, or D>" not in text


def test_hf_hosted_simple_mode_payload_has_letter_only_instruction() -> None:
    model = HFHostedModel(model_name="dummy/model")
    model.use_json_format = False
    prompt = _prompt_for(model, _label_trial(["A", "B", "C", "D"]))
    text = _hf_payload_text(model, prompt)
    assert "Answer with only the letter A, B, C, or D." in text
    assert '{"answer"' not in text
