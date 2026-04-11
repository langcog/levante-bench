"""Unit tests for external runtime API helpers."""

from __future__ import annotations

import pytest

from levante_bench.runtime import api


def test_load_model_uses_name_from_inline_config(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_resolve_model_config(model_name, **kwargs):
        captured["model_name"] = model_name
        captured["kwargs"] = kwargs
        return {"hf_name": "dummy/hf-model"}

    def fake_build_model(model_name, model_cfg, device, auto_load):
        captured["build_model_name"] = model_name
        captured["build_model_cfg"] = model_cfg
        captured["device"] = device
        captured["auto_load"] = auto_load
        return object()

    monkeypatch.setattr(api, "resolve_model_config", fake_resolve_model_config)
    monkeypatch.setattr(api, "build_model", fake_build_model)

    model = api.load_model(
        model_config={"name": "qwen35", "hf_name": "ignored/by/resolve"},
        device="cpu",
        auto_load=False,
    )

    assert model is not None
    assert captured["model_name"] == "qwen35"
    assert captured["build_model_name"] == "qwen35"
    assert captured["device"] == "cpu"
    assert captured["auto_load"] is False


def test_run_trials_sets_defaults_without_overwriting_existing() -> None:
    class DummyModel:
        def evaluate_trial(self, trial):
            return trial

    trials = [
        {
            "trial_id": "t1",
            "item_uid": "u1",
            "prompt": "p1",
            "option_labels": ["A", "B"],
            "correct_label": "A",
        },
        {
            "trial_id": "t2",
            "item_uid": "u2",
            "prompt": "p2",
            "option_labels": ["A", "B"],
            "correct_label": "B",
            "max_new_tokens": 5,
            "task_id": "already-there",
        },
    ]

    out = api.run_trials(
        model=DummyModel(),
        trials=trials,
        max_new_tokens=64,
        task_id="custom",
    )

    assert out[0]["max_new_tokens"] == 64
    assert out[0]["task_id"] == "custom"
    assert out[1]["max_new_tokens"] == 5
    assert out[1]["task_id"] == "already-there"


def test_run_logit_forced_12_without_swap() -> None:
    class DummyModel:
        def score_choices(self, prompt_text, image_paths, choice_texts):
            return {
                "choice_probs": [0.8, 0.2],
                "choice_logits": [1.2, 0.3],
                "model_name": "dummy",
                "generation_time_s": 0.4,
            }

    out = api.run_logit_forced_12(
        model=DummyModel(),
        prompt_text="pick 1 or 2",
        image_paths=["ref.png", "a.png", "b.png"],
        swap_correct=False,
    )
    assert out["predicted_choice"] == "1"
    assert out["choice_probs"] == [0.8, 0.2]
    assert out["choice_logits"] == [1.2, 0.3]


def test_run_logit_forced_12_with_swap_correction() -> None:
    class DummyModel:
        def __init__(self):
            self.calls = 0

        def score_choices(self, prompt_text, image_paths, choice_texts):
            self.calls += 1
            if self.calls == 1:
                return {
                    "choice_probs": [0.6, 0.4],
                    "choice_logits": [0.7, 0.5],
                    "model_name": "dummy",
                    "generation_time_s": 0.5,
                }
            return {
                "choice_probs": [0.3, 0.7],
                "choice_logits": [0.2, 0.9],
                "model_name": "dummy",
                "generation_time_s": 0.5,
            }

    model = DummyModel()
    out = api.run_logit_forced_12(
        model=model,
        prompt_text="pick 1 or 2",
        image_paths=["ref.png", "a.png", "b.png"],
        swap_correct=True,
    )
    assert out["predicted_choice"] == "1"
    assert out["choice_probs"] == pytest.approx([0.65, 0.35])
    assert "swap_corrected" in out


def test_score_choices_raises_for_unsupported_model() -> None:
    class DummyModel:
        def score_choices(self, prompt_text, image_paths, choice_texts):
            raise NotImplementedError

    with pytest.raises(ValueError, match="does not support score_choices"):
        api.score_choices(
            model=DummyModel(),
            prompt_text="pick 1 or 2",
            image_paths=["ref.png", "a.png", "b.png"],
            choice_texts=("1", "2"),
        )


def test_score_choices_validates_choice_texts_with_tokenizer() -> None:
    class DummyTokenizer:
        def encode(self, text, add_special_tokens=False):
            return [1, 2] if text == "bad" else [1]

    class DummyProcessor:
        tokenizer = DummyTokenizer()

    class DummyModel:
        processor = DummyProcessor()

        def score_choices(self, prompt_text, image_paths, choice_texts):
            return {}

    with pytest.raises(ValueError, match="one-token choices"):
        api.score_choices(
            model=DummyModel(),
            prompt_text="pick",
            image_paths=["ref.png", "a.png", "b.png"],
            choice_texts=("bad", "2"),
        )
