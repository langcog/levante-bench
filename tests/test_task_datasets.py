"""Tests for task dataset loading and trial dict correctness.

No model or GPU needed. Verifies each registered task produces valid,
well-formed trial dicts from manifest data.

Usage:
    python tests/test_task_datasets.py
    python tests/test_task_datasets.py --task vocab
    python tests/test_task_datasets.py --version 2026-03-24
"""

import argparse
import re
import sys
from pathlib import Path

REQUIRED_FIELDS = {
    "trial_id": str,
    "item_uid": str,
    "prompt": str,
    "options": list,
    "option_labels": list,
    "correct_label": str,
    "context_image_paths": list,
    "option_image_paths": list,
    "context_type": str,
    "option_type": str,
}

VALID_CONTEXT_TYPES = {"none", "image", "multi_image"}  # multi_image unused currently but valid
VALID_OPTION_TYPES = {"text", "image"}


def run_test(name, fn):
    try:
        fn()
        print(f"  PASS  {name}")
        return True
    except Exception as e:
        print(f"  FAIL  {name}: {e}")
        return False


# ── Individual tests ────────────────────────────────────────────────────────

def test_loads(task_id, version, data_root):
    """Dataset loads and has >0 trials."""
    from levante_bench.config import get_task_def
    from levante_bench.tasks import get_task_dataset

    task_def = get_task_def(task_id, version, data_root=data_root)
    assert task_def is not None, f"No task def for {task_id}"
    ds_cls = get_task_dataset(task_id)
    assert ds_cls is not None, f"No dataset class registered for {task_id}"
    ds = ds_cls(task_def=task_def, version=version, data_root=data_root)
    assert len(ds) > 0, "Dataset is empty"
    return ds


def test_trial_fields(trial):
    """Trial dict has all required keys with correct types."""
    for field, expected_type in REQUIRED_FIELDS.items():
        assert field in trial, f"Missing field '{field}'"
        assert isinstance(trial[field], expected_type), (
            f"'{field}' is {type(trial[field]).__name__}, expected {expected_type.__name__}"
        )


def test_trial_values(trial):
    """Values are valid: >=2 options, correct_label in labels, valid types."""
    assert trial["context_type"] in VALID_CONTEXT_TYPES, (
        f"Invalid context_type: {trial['context_type']}"
    )
    assert trial["option_type"] in VALID_OPTION_TYPES, (
        f"Invalid option_type: {trial['option_type']}"
    )
    assert len(trial["options"]) >= 2, (
        f"Need >=2 options, got {len(trial['options'])}"
    )
    assert len(trial["option_labels"]) == len(trial["options"]), (
        f"Labels ({len(trial['option_labels'])}) != options ({len(trial['options'])})"
    )
    assert trial["correct_label"] in trial["option_labels"], (
        f"correct_label '{trial['correct_label']}' not in {trial['option_labels']}"
    )
    assert trial["prompt"].strip(), "Prompt is empty"


def test_image_paths_exist(trial):
    """All referenced image paths point to real files."""
    for p in trial["context_image_paths"]:
        assert Path(p).exists(), f"Context image missing: {p}"
    for p in trial["option_image_paths"]:
        assert Path(p).exists(), f"Option image missing: {p}"


def test_image_count_matches_type(trial):
    """Image count is consistent with option_type."""
    if trial["option_type"] == "image":
        assert len(trial["option_image_paths"]) == len(trial["options"]), (
            f"option_type=image but {len(trial['option_image_paths'])} paths "
            f"for {len(trial['options'])} options"
        )
    else:
        assert len(trial["option_image_paths"]) == 0, (
            f"option_type=text but has {len(trial['option_image_paths'])} option images"
        )


def test_placeholders_resolved(trial):
    """No unresolved template placeholders in prompt."""
    prompt = trial["prompt"]
    assert "<prompt_phrase>" not in prompt, "Unresolved <prompt_phrase>"
    assert "<prompt_image>" not in prompt, "Unresolved <prompt_image>"
    if trial["option_type"] == "text":
        for i in range(1, 9):
            assert f"<option{i}>" not in prompt, f"Unresolved <option{i}>"


def test_image_interleaving(trial):
    """Image-option tasks have <imageN> placeholders for model interleaving."""
    if trial["option_type"] == "image" and trial["option_image_paths"]:
        placeholders = re.findall(r'<image\d+>', trial["prompt"])
        n_option_imgs = len(trial["option_image_paths"])
        assert len(placeholders) >= n_option_imgs, (
            f"Need >={n_option_imgs} <imageN> placeholders, found {len(placeholders)}"
        )


def test_shuffle_determinism(task_id, version, data_root):
    """Loading same trial twice gives identical option order."""
    from levante_bench.config import get_task_def
    from levante_bench.tasks import get_task_dataset

    task_def = get_task_def(task_id, version, data_root=data_root)
    ds_cls = get_task_dataset(task_id)
    ds1 = ds_cls(task_def=task_def, version=version, data_root=data_root)
    ds2 = ds_cls(task_def=task_def, version=version, data_root=data_root)
    assert ds1[0]["options"] == ds2[0]["options"], "Option order not deterministic"
    assert ds1[0]["correct_label"] == ds2[0]["correct_label"], "Correct label changed"


def test_sample_trials(ds, n=5):
    """First N trials all pass field/value/image checks."""
    n = min(n, len(ds))
    for i in range(n):
        t = ds[i]
        test_trial_fields(t)
        test_trial_values(t)
        test_image_paths_exist(t)
        test_image_count_matches_type(t)


# ── Runner ──────────────────────────────────────────────────────────────────

def run_all(task_ids, version, data_root):
    passed, failed = 0, 0

    for task_id in task_ids:
        print(f"\n--- {task_id} ---")

        ds = None
        try:
            ds = test_loads(task_id, version, data_root)
            print(f"  PASS  loads ({len(ds)} trials)")
            passed += 1
        except Exception as e:
            print(f"  FAIL  loads: {e}")
            failed += 1
            continue

        trial = ds[0]
        tests = [
            ("trial_fields", lambda: test_trial_fields(trial)),
            ("trial_values", lambda: test_trial_values(trial)),
            ("image_paths_exist", lambda: test_image_paths_exist(trial)),
            ("image_count_matches_type", lambda: test_image_count_matches_type(trial)),
            ("placeholders_resolved", lambda: test_placeholders_resolved(trial)),
            ("image_interleaving", lambda: test_image_interleaving(trial)),
            ("shuffle_determinism", lambda: test_shuffle_determinism(task_id, version, data_root)),
            ("sample_trials", lambda: test_sample_trials(ds)),
        ]
        for name, fn in tests:
            if run_test(name, fn):
                passed += 1
            else:
                failed += 1

    return passed, failed


def main():
    parser = argparse.ArgumentParser(description="Test task datasets")
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--version", type=str, default="2026-03-24")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    from levante_bench.tasks import list_task_datasets
    all_tasks = list_task_datasets()

    if args.task:
        assert args.task in all_tasks, f"'{args.task}' not registered. Available: {all_tasks}"
        task_ids = [args.task]
    else:
        task_ids = all_tasks

    print("=" * 50)
    print("TASK DATASET TESTS")
    print("=" * 50)

    p, f = run_all(task_ids, args.version, data_root)

    print("\n" + "=" * 50)
    print(f"TOTAL: {p} passed, {f} failed")
    print("=" * 50)
    return 1 if f > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
