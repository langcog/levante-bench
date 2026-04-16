"""Unit tests for runner utility behavior."""

from __future__ import annotations
import json
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
        pass

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
    def _fake_build_model(model_name, model_cfg, device, auto_load):
        captured["model_name"] = model_name
        captured["device"] = device
        captured["keep_me"] = model_cfg.get("keep_me")
        captured["loaded"] = auto_load
        return DummyModel()

    monkeypatch.setattr(runner, "build_model", _fake_build_model)
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

    assert captured["model_name"] == "dummy"
    assert captured["device"] == "cpu"
    assert captured["keep_me"] == 99
    assert captured["loaded"] is True
    assert "dummy" in results
    metadata_path = results["dummy"].parent / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["model_label"] == "dummy-tiny"
    assert metadata["prompt_language"] == "en"


def test_run_eval_applies_global_and_task_specific_overrides(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured_overrides: dict[str, dict] = {}

    class DummyModel:
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
    monkeypatch.setattr(
        runner,
        "build_model",
        lambda model_name, model_cfg, device, auto_load: DummyModel(),
    )
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
        "true_random_option_order": False,
        "option_order_run_seed": None,
    }


def test_run_eval_appends_non_english_language_suffix_to_model_dir(
    monkeypatch,
    tmp_path: Path,
) -> None:
    class DummyModel:
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
    monkeypatch.setattr(
        runner,
        "build_model",
        lambda model_name, model_cfg, device, auto_load: DummyModel(),
    )
    monkeypatch.setattr(runner, "load_cache", lambda path: {})
    monkeypatch.setattr(runner, "write_summary_csv", lambda model_dir, _: model_dir / "summary.csv")

    cfg = OmegaConf.create(
        {
            "data_root": str(tmp_path / "data"),
            "output_dir": str(tmp_path / "out"),
            "version": "unit-test",
            "device": "cpu",
            "models": ["dummy"],
            "tasks": [],
            "task_overrides": {"__all__": {"prompt_language": "de-CH"}},
        }
    )

    results = runner.run_eval(cfg)
    assert results["dummy"].parent.name == "dummy-tiny-de"


def test_run_eval_normalizes_legacy_output_dir_suffix(
    monkeypatch,
    tmp_path: Path,
) -> None:
    class DummyModel:
        pass

    monkeypatch.setattr(
        runner,
        "load_model_config",
        lambda model_name: OmegaConf.create(
            {
                "hf_name": "base/hf-model",
                "size": "E4B-it",
                "max_new_tokens": 64,
                "use_json_format": True,
            }
        ),
    )
    monkeypatch.setattr(
        runner,
        "build_model",
        lambda model_name, model_cfg, device, auto_load: DummyModel(),
    )
    monkeypatch.setattr(runner, "load_cache", lambda path: {})
    monkeypatch.setattr(runner, "write_summary_csv", lambda model_dir, _: model_dir / "summary.csv")

    cfg = OmegaConf.create(
        {
            "data_root": str(tmp_path / "data"),
            "output_dir": str(tmp_path / "results" / "gemma4-v1"),
            "version": "v1",
            "device": "cpu",
            "models": ["gemma4"],
            "tasks": [],
        }
    )

    results = runner.run_eval(cfg)
    assert results["gemma4"] == tmp_path / "results" / "v1" / "gemma4-E4B-it" / "summary.csv"


def test_run_eval_true_random_writes_numbered_run_subdirs(
    monkeypatch,
    tmp_path: Path,
) -> None:
    class DummyModel:
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
    monkeypatch.setattr(
        runner,
        "build_model",
        lambda model_name, model_cfg, device, auto_load: DummyModel(),
    )
    monkeypatch.setattr(runner, "load_cache", lambda path: {})
    monkeypatch.setattr(runner, "write_summary_csv", lambda model_dir, _: model_dir / "summary.csv")

    cfg = OmegaConf.create(
        {
            "data_root": str(tmp_path / "data"),
            "output_dir": str(tmp_path / "out"),
            "version": "unit-test",
            "device": "cpu",
            "models": ["dummy"],
            "tasks": [],
            "num_runs": 2,
            "true_random_option_order": True,
        }
    )

    results = runner.run_eval(cfg)
    assert "dummy:0001" in results
    assert "dummy:0002" in results
    assert results["dummy:0001"].parent.name == "0001"
    assert results["dummy:0002"].parent.name == "0002"
