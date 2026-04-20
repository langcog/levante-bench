"""Tests for runner cache behavior and GPT53 retry logic."""

from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

from omegaconf import OmegaConf

from levante_bench.evaluation import runner
from levante_bench.models.gpt import GPT53Model


def test_run_eval_uses_cache_on_second_pass(monkeypatch, tmp_path: Path) -> None:
    calls = {"evaluate_trial": 0}

    class DummyDataset:
        def __init__(self, *args, **kwargs) -> None:
            self._trial = {
                "trial_id": "t1",
                "item_uid": "u1",
                "prompt": "choose",
                "options": ["opt-a", "opt-b"],
                "option_labels": ["A", "B"],
                "correct_label": "A",
                "answer_format": "label",
            }

        def __len__(self) -> int:
            return 1

        def __getitem__(self, idx: int) -> dict:
            return dict(self._trial)

    class DummyModel:
        def __init__(self, model_name: str, device: str) -> None:
            self.model_name = model_name
            self.device = device
            self.use_json_format = True

        def load(self) -> None:
            return None

        def evaluate_trial(self, trial: dict) -> dict:
            calls["evaluate_trial"] += 1
            return {
                "trial_id": trial["trial_id"],
                "item_uid": trial["item_uid"],
                "generated_text": "A",
                "predicted_label": "A",
                "reason": "",
                "correct_label": trial["correct_label"],
                "is_correct": True,
                "options": trial["options"],
                "option_labels": trial["option_labels"],
            }

        def evaluate_trials_batch(self, trials: list[dict]) -> list[dict]:
            return [self.evaluate_trial(trial) for trial in trials]

    monkeypatch.setattr(
        runner,
        "load_model_config",
        lambda _: OmegaConf.create({"hf_name": "dummy/hf", "max_new_tokens": 8}),
    )
    monkeypatch.setattr(
        runner,
        "resolve_model_config",
        lambda model_name, model_overrides=None, model_config=None: {
            "hf_name": "dummy/hf",
            "max_new_tokens": 8,
        },
    )
    monkeypatch.setattr(
        runner,
        "build_model",
        lambda model_name, model_cfg, device, auto_load=True: DummyModel(
            model_name=model_cfg["hf_name"],
            device=device,
        ),
    )
    monkeypatch.setattr(runner, "load_task_config", lambda _: {"context_type": "none"})
    monkeypatch.setattr(
        runner,
        "get_task_def",
        lambda *args, **kwargs: SimpleNamespace(human_response_path=None),
    )
    monkeypatch.setattr(runner, "get_task_dataset", lambda _: DummyDataset)
    monkeypatch.setattr(runner, "write_task_csv", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, "postprocess_task_outputs", lambda **kwargs: [])
    monkeypatch.setattr(
        runner, "write_summary_csv", lambda model_dir, _: model_dir / "summary.csv"
    )
    monkeypatch.setattr(runner, "tqdm", lambda iterable, **kwargs: iterable)

    cfg = OmegaConf.create(
        {
            "data_root": str(tmp_path / "data"),
            "output_dir": str(tmp_path / "results"),
            "version": "unit-test",
            "device": "cpu",
            "models": ["dummy"],
            "tasks": ["dummy-task"],
        }
    )

    runner.run_eval(cfg)
    runner.run_eval(cfg)

    assert calls["evaluate_trial"] == 1


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict | None = None, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self) -> dict:
        return self._payload


def test_gpt53_retries_on_5xx_then_succeeds(monkeypatch) -> None:
    import levante_bench.models.gpt as vlm_module

    model = GPT53Model(
        model_name="gpt-5.3",
        retry_attempts=3,
        max_output_tokens_min=8,
        max_output_tokens_cap=64,
    )
    model.api_key = "test-key"

    responses = [
        _FakeResponse(status_code=502, text="bad gateway"),
        _FakeResponse(status_code=200, payload={"output_text": "B"}),
    ]
    seen_tokens: list[int] = []

    def fake_post(url: str, headers: dict, json: dict, timeout: int):
        seen_tokens.append(int(json["max_output_tokens"]))
        return responses.pop(0)

    monkeypatch.setattr(vlm_module.requests, "post", fake_post)

    out = model.generate("pick one", max_new_tokens=4)

    assert out == "B"
    assert seen_tokens == [8, 8]


def test_gpt53_doubles_tokens_after_incomplete_response(monkeypatch) -> None:
    import levante_bench.models.gpt as vlm_module

    model = GPT53Model(
        model_name="gpt-5.3",
        retry_attempts=3,
        max_output_tokens_min=8,
        max_output_tokens_cap=32,
    )
    model.api_key = "test-key"

    responses = [
        _FakeResponse(
            status_code=200,
            payload={
                "output": [],
                "incomplete_details": {"reason": "max_output_tokens"},
            },
        ),
        _FakeResponse(status_code=200, payload={"output_text": "C"}),
    ]
    seen_tokens: list[int] = []

    def fake_post(url: str, headers: dict, json: dict, timeout: int):
        seen_tokens.append(int(json["max_output_tokens"]))
        return responses.pop(0)

    monkeypatch.setattr(vlm_module.requests, "post", fake_post)

    out = model.generate("pick one", max_new_tokens=4)

    assert out == "C"
    assert seen_tokens == [8, 16]
