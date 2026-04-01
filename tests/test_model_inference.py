"""Tests for model inference across all task prompt formats.

Loads a model once, then verifies it correctly handles each task's
prompt format: text-only, image options, context images, mixed.

Usage:
    python tests/test_model_inference.py --model smolvlm2 --device cuda
    python tests/test_model_inference.py --model smolvlm2 --device cuda --task vocab
"""

import argparse
import sys
from pathlib import Path


def run_test(name, fn):
    try:
        fn()
        print(f"  PASS  {name}")
        return True
    except Exception as e:
        print(f"  FAIL  {name}: {e}")
        return False


# ── Tests ───────────────────────────────────────────────────────────────────

def test_model_loads(model_id, device):
    """Model class exists, loads weights without error."""
    from levante_bench.models import get_model_class

    model_cls = get_model_class(model_id)
    assert model_cls is not None, f"'{model_id}' not registered"
    model = model_cls(device=device)
    model.load()
    return model


def test_parse_answer(model):
    """parse_answer extracts labels from common output formats."""
    labels = ["A", "B", "C", "D"]
    cases = [
        ('{"answer": "A", "reason": "it matches"}', "A"),
        ('{"answer": "B"}', "B"),
        ('{"answer": "C", "reason": "truncated"', "C"),
        ("The correct answer is C", "C"),
        ("The answer is D", "D"),
        ("Answer: B", "B"),
        ("correct option is A", "A"),
        ("A", "A"),
        ("B.", "B"),
        ("", None),
        ("no label here", None),
    ]
    for text, expected in cases:
        label, reason = model.parse_answer(text, labels)
        assert label == expected, (
            f"parse_answer({text!r}) = {label!r}, expected {expected!r}"
        )


def test_build_messages(model, trial):
    """Model builds valid chat messages with correct image count."""
    image_paths = trial.get("context_image_paths", []) + trial.get("option_image_paths", [])
    messages = model._build_messages(
        trial["prompt"], image_paths if image_paths else None
    )
    assert isinstance(messages, list) and len(messages) > 0
    content = messages[0]["content"]

    has_text = any(c["type"] == "text" for c in content)
    assert has_text, "No text element in messages"

    n_images_in_msg = sum(1 for c in content if c["type"] == "image")
    if image_paths:
        assert n_images_in_msg == len(image_paths), (
            f"Expected {len(image_paths)} images, got {n_images_in_msg}"
        )

    # For image-option tasks, verify images are interleaved (not all at start)
    if trial["option_type"] == "image" and len(image_paths) > 1:
        types = [c["type"] for c in content]
        first_img = types.index("image") if "image" in types else -1
        last_text = len(types) - 1 - types[::-1].index("text") if "text" in types else -1
        # Images should not all come before text (interleaving check)
        assert first_img < last_text, (
            "Images should be interleaved with text labels, not all before text"
        )


def test_generate(model, trial):
    """Model generates non-empty response and returns valid result dict."""
    result = model.evaluate_trial(trial)
    assert isinstance(result, dict)
    assert result["generated_text"].strip(), "Generated text is empty"
    assert result["predicted_label"] is None or result["predicted_label"] in trial["option_labels"]
    assert isinstance(result["is_correct"], bool)
    return result


# ── Runner ──────────────────────────────────────────────────────────────────

def run_all(model_id, device, task_ids, version, data_root):
    passed, failed = 0, 0

    # Load model once
    model = None
    try:
        model = test_model_loads(model_id, device)
        print(f"  PASS  model_loads ({model_id})")
        passed += 1
    except Exception as e:
        print(f"  FAIL  model_loads: {e}")
        return 0, 1

    if run_test("parse_answer", lambda: test_parse_answer(model)):
        passed += 1
    else:
        failed += 1

    # Test each task
    from levante_bench.config import get_task_def
    from levante_bench.tasks import get_task_dataset

    for task_id in task_ids:
        print(f"\n--- {model_id} x {task_id} ---")

        try:
            task_def = get_task_def(task_id, version, data_root=data_root)
            ds = get_task_dataset(task_id)(
                task_def=task_def, version=version, data_root=data_root
            )
            trial = ds[0]
        except Exception as e:
            print(f"  FAIL  load_trial: {e}")
            failed += 1
            continue

        # Show what kind of prompt this task produces
        n_ctx = len(trial["context_image_paths"])
        n_opt = len(trial["option_image_paths"])
        print(f"  info  option_type={trial['option_type']}, "
              f"context_imgs={n_ctx}, option_imgs={n_opt}")

        if run_test("build_messages", lambda: test_build_messages(model, trial)):
            passed += 1
        else:
            failed += 1

        if run_test("generate", lambda: test_generate(model, trial)):
            passed += 1
        else:
            failed += 1

    return passed, failed


def main():
    parser = argparse.ArgumentParser(description="Test model inference across tasks")
    parser.add_argument("--model", type=str, default="smolvlm2")
    parser.add_argument("--device", type=str, default="cuda")
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
    print(f"MODEL INFERENCE TESTS ({args.model})")
    print("=" * 50)

    p, f = run_all(args.model, args.device, task_ids, args.version, data_root)

    print("\n" + "=" * 50)
    print(f"TOTAL: {p} passed, {f} failed")
    print("=" * 50)
    return 1 if f > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
