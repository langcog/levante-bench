"""Unit tests for runner utility behavior."""

from __future__ import annotations
from pathlib import Path

from omegaconf import OmegaConf

from levante_bench.evaluation import runner


def test_resolve_device_non_auto_passthrough() -> None:
    assert runner.resolve_device("cpu") == "cpu"
    assert runner.resolve_device("cuda") == "cuda"


def test_resolve_device_auto_without_torch_defaults_cpu(monkeypatch) -> None:
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("no torch")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert runner.resolve_device("auto") == "cpu"


def test_run_eval_merges_overrides_and_filters_constructor_kwargs(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    class DummyModel:
        def __init__(self, model_name: str, device: str, keep_me: int = 0) -> None:
            captured["model_name"] = model_name
            captured["device"] = device
            captured["keep_me"] = keep_me

        def load(self) -> None:
            captured["loaded"] = True

    monkeypatch.setattr(
        runner,
        "load_model_config",
        lambda model_name: OmegaConf.create(
            {
                "hf_name": "base/hf-model",
                "size": "tiny",
                "keep_me": 1,
                "drop_me": 2,
                "max_new_tokens": 64,
                "use_json_format": True,
            }
        ),
    )
    monkeypatch.setattr(runner, "get_model_class", lambda model_name: DummyModel)
    monkeypatch.setattr(runner, "load_cache", lambda path: {})
    monkeypatch.setattr(runner, "write_summary_csv", lambda model_dir, _: model_dir / "summary.csv")

    cfg = OmegaConf.create(
        {
            "data_root": str(tmp_path / "data"),
            "output_dir": str(tmp_path / "out"),
            "version": "unit-test",
            "device": "cpu",
            "models": [{"name": "dummy", "keep_me": 99, "drop_me": 123}],
            "tasks": [],
        }
    )

    results = runner.run_eval(cfg)

    assert captured["model_name"] == "base/hf-model"
    assert captured["device"] == "cpu"
    assert captured["keep_me"] == 99
    assert captured["loaded"] is True
    assert "dummy" in results


def test_run_eval_applies_global_and_task_specific_overrides(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured_overrides: dict[str, dict] = {}

    class DummyModel:
        def __init__(self, model_name: str, device: str) -> None:
            pass

        def load(self) -> None:
            pass

    monkeypatch.setattr(
        runner,
        "load_model_config",
        lambda model_name: OmegaConf.create(
            {
                "hf_name": "base/hf-model",
                "size": "tiny",
                "max_new_tokens": 64,
                "use_json_format": True,
            }
        ),
    )
    monkeypatch.setattr(runner, "get_model_class", lambda model_name: DummyModel)
    monkeypatch.setattr(runner, "load_task_config", lambda task_id: {"context_type": "none"})
    monkeypatch.setattr(runner, "load_cache", lambda path: {})
    monkeypatch.setattr(runner, "write_summary_csv", lambda model_dir, _: model_dir / "summary.csv")

    def _capture_get_task_def(task_id, version, data_root=None, task_overrides=None):
        captured_overrides[task_id] = dict(task_overrides or {})
        return None

    monkeypatch.setattr(runner, "get_task_def", _capture_get_task_def)

    cfg = OmegaConf.create(
        {
            "data_root": str(tmp_path / "data"),
            "output_dir": str(tmp_path / "out"),
            "version": "unit-test",
            "device": "cpu",
            "models": ["dummy"],
            "tasks": ["trog"],
            "task_overrides": {
                "__all__": {"prompt_language": "de"},
                "trog": {"include_numberline": False},
            },
        }
    )

    runner.run_eval(cfg)

    assert captured_overrides["trog"] == {
        "prompt_language": "de",
        "include_numberline": False,
    }
