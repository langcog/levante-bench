"""Tests for run-trials-jsonl CLI command."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from levante_bench import cli


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_cmd_run_trials_jsonl_writes_output(monkeypatch, tmp_path: Path) -> None:
    input_path = tmp_path / "trials.jsonl"
    output_path = tmp_path / "results.jsonl"
    _write_jsonl(
        input_path,
        [
            {
                "trial_id": "t1",
                "item_uid": "u1",
                "prompt": "p",
                "option_labels": ["A", "B"],
                "correct_label": "A",
            }
        ],
    )

    monkeypatch.setattr(cli, "load_model", lambda **kwargs: object())
    monkeypatch.setattr(
        cli,
        "run_trials",
        lambda **kwargs: [{"trial_id": "t1", "predicted_label": "A", "is_correct": True}],
    )

    args = argparse.Namespace(
        input_jsonl=str(input_path),
        output_jsonl=str(output_path),
        model="qwen35",
        model_config_path=None,
        configs_root=None,
        device="cpu",
        max_new_tokens=32,
        task_id="custom-task",
    )
    rc = cli.cmd_run_trials_jsonl(args)
    assert rc == 0

    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    out = json.loads(lines[0])
    assert out["trial_id"] == "t1"
    assert out["predicted_label"] == "A"


def test_cmd_run_trials_jsonl_missing_input_returns_error(tmp_path: Path) -> None:
    args = argparse.Namespace(
        input_jsonl=str(tmp_path / "missing.jsonl"),
        output_jsonl=str(tmp_path / "out.jsonl"),
        model="qwen35",
        model_config_path=None,
        configs_root=None,
        device="cpu",
        max_new_tokens=None,
        task_id=None,
    )
    rc = cli.cmd_run_trials_jsonl(args)
    assert rc == 1


def test_cmd_run_trials_jsonl_validates_required_fields(tmp_path: Path) -> None:
    input_path = tmp_path / "bad.jsonl"
    output_path = tmp_path / "out.jsonl"
    _write_jsonl(
        input_path,
        [
            {
                "trial_id": "t1",
                "item_uid": "u1",
                "prompt": "p",
                "correct_label": "A",
            }
        ],
    )
    args = argparse.Namespace(
        input_jsonl=str(input_path),
        output_jsonl=str(output_path),
        model="qwen35",
        model_config_path=None,
        configs_root=None,
        device="cpu",
        max_new_tokens=None,
        task_id=None,
    )
    rc = cli.cmd_run_trials_jsonl(args)
    assert rc == 1
    assert not output_path.exists()
