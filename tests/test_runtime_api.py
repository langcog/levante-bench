"""Unit tests for external runtime API helpers."""

from __future__ import annotations

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
