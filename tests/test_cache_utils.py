"""Unit tests for evaluation cache helpers."""

from __future__ import annotations

from pathlib import Path

from levante_bench.evaluation.cache import load_cache, save_cache, trial_hash


def test_trial_hash_is_stable_for_same_trial_content() -> None:
    trial = {
        "trial_id": "t1",
        "item_uid": "u1",
        "prompt": "pick one",
        "options": ["red", "blue"],
        "option_labels": ["A", "B"],
    }
    assert trial_hash(trial) == trial_hash(dict(trial))


def test_trial_hash_changes_when_key_fields_change() -> None:
    base = {
        "trial_id": "t1",
        "item_uid": "u1",
        "prompt": "pick one",
        "options": ["red", "blue"],
        "option_labels": ["A", "B"],
    }
    changed_prompt = dict(base)
    changed_prompt["prompt"] = "pick two"
    changed_options = dict(base)
    changed_options["options"] = ["red", "green"]
    changed_seed = dict(base)
    changed_seed["option_order_seed"] = "123"

    assert trial_hash(base) != trial_hash(changed_prompt)
    assert trial_hash(base) != trial_hash(changed_options)
    assert trial_hash(base) != trial_hash(changed_seed)


def test_load_cache_missing_file_returns_empty_dict(tmp_path: Path) -> None:
    cache_path = tmp_path / "cache" / "responses.json"
    assert load_cache(cache_path) == {}


def test_save_and_load_cache_round_trip(tmp_path: Path) -> None:
    cache_path = tmp_path / "cache" / "responses.json"
    payload = {
        "abc123": {"trial_id": "t1", "predicted_label": "A", "is_correct": True},
        "def456": {"trial_id": "t2", "predicted_value": 3.14, "is_correct": False},
    }

    save_cache(cache_path, payload)
    loaded = load_cache(cache_path)

    assert cache_path.exists()
    assert loaded == payload
