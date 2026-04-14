#!/usr/bin/env python3
"""Prompt Robustness Sweep — Preliminary experiment.

Runs a Task Framing (TF) × Output Format (OF) matrix on a stratified subset
of trials for each (model, task) pair.  Uses the existing levante_bench model
infrastructure so any registered model works out of the box.

Usage:
    python scripts/prompt_robustness_sweep.py \\
        --models qwen35 smolvlm2 internvl35 \\
        --tasks mental-rotation vocab trog \\
        --subset-fraction 0.3 \\
        --output-dir results/prompt_robustness

    # Single model, all tasks, full trials:
    python scripts/prompt_robustness_sweep.py \\
        --models qwen35 --model-sizes 0.8B \\
        --subset-fraction 1.0
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from levante_bench.config import get_task_def, load_model_config, load_task_config
from levante_bench.config.defaults import detect_data_version
from levante_bench.runtime.modeling import build_model, resolve_model_config
from levante_bench.tasks import get_task_dataset

PROMPTS_DIR = PROJECT_ROOT / "configs" / "prompts"


# ── Prompt config loading ─────────────────────────────────────────────

def load_prompt_config(task_id: str) -> dict:
    """Load the TF × OF config for a task."""
    yaml_name = task_id.replace("-", "_") + ".yaml"
    path = PROMPTS_DIR / yaml_name
    if not path.exists():
        raise FileNotFoundError(f"No prompt config for task '{task_id}' at {path}")
    with open(path) as f:
        return yaml.safe_load(f)


# ── Prompt assembly ───────────────────────────────────────────────────

def _image_tag(idx: int) -> str:
    return f"<image{idx}>"


def assemble_prompt(
    tf_template: str,
    of_config: dict,
    trial: dict,
    prompt_config: dict,
) -> tuple[str, str | None, list[str]]:
    """Build (prompt_text, system_prompt, image_paths) from TF + OF + trial.

    Returns the fully assembled prompt with image placeholders, the system
    prompt (or None), and the ordered list of image file paths.
    """
    task_id = prompt_config["task_id"]
    labels = prompt_config["labels"]
    n_options = prompt_config["n_options"]

    context_paths = trial.get("context_image_paths", [])
    option_paths = trial.get("option_image_paths", [])
    image_paths: list[str] = []

    prompt_text = tf_template

    # ToM: {full_prompt} is the entire story from the manifest
    if "{full_prompt}" in prompt_text:
        full_prompt = trial.get("prompt", "")
        # Replace image placeholders from manifest to our numbered scheme
        for i in range(1, 5):
            full_prompt = full_prompt.replace(f"<image{i}>", f"{{opt_{chr(96+i)}_image}}")
        prompt_text = prompt_text.replace("{full_prompt}", full_prompt)

    # Substitute prompt_phrase
    prompt_phrase = trial.get("prompt_phrase", "")
    prompt_text = prompt_text.replace("{prompt_phrase}", str(prompt_phrase))

    # Substitute text options (egma-math): trial["options"] is already shuffled
    options = trial.get("options", [])
    for i, opt in enumerate(options):
        prompt_text = prompt_text.replace(f"{{option{i+1}}}", str(opt))

    # Context/reference image
    img_idx = 0
    if context_paths:
        prompt_text = prompt_text.replace("{ref_image}", _image_tag(img_idx))
        prompt_text = prompt_text.replace("{context_image}", _image_tag(img_idx))
        image_paths.append(context_paths[0])
        img_idx += 1
    else:
        prompt_text = prompt_text.replace("{ref_image}", "")
        prompt_text = prompt_text.replace("{context_image}", "")

    # Option images
    option_labels = ["a", "b", "c", "d"]
    for i, opt_path in enumerate(option_paths):
        placeholder = f"{{opt_{option_labels[i]}_image}}"
        prompt_text = prompt_text.replace(placeholder, _image_tag(img_idx))
        image_paths.append(opt_path)
        img_idx += 1

    # Append output format suffix
    of_suffix = of_config.get("suffix", "")
    of_suffix = of_suffix.replace("{labels}", labels)
    # Escape JSON braces that were doubled in YAML
    prompt_text = prompt_text.strip() + of_suffix

    # System prompt
    system = of_config.get("system")
    if system:
        system = system.replace("{labels}", labels)

    # Clean up whitespace
    prompt_text = re.sub(r"\n{3,}", "\n\n", prompt_text).strip()

    return prompt_text, system, image_paths


# ── Trial loading via existing infrastructure ─────────────────────────

def load_trials(task_id: str, data_root: Path, version: str) -> list[dict]:
    """Load all trials for a task using the registered dataset class.

    Enriches each trial dict with ``prompt_phrase`` and text ``options``
    from the raw manifest so that TF templates can reference them.
    """
    task_cfg = load_task_config(task_id)
    if task_cfg is None:
        raise ValueError(f"No config for task {task_id}")
    task_def = get_task_def(task_id, version, data_root=data_root)
    if task_def is None:
        raise ValueError(f"No task def for {task_id} version={version}")
    dataset_cls = get_task_dataset(task_id)
    if dataset_cls is None:
        raise ValueError(f"No dataset for {task_id}")
    dataset = dataset_cls(task_def=task_def, version=version, data_root=data_root)

    # Build a lookup from item_uid → manifest row for prompt_phrase / options
    import pandas as pd
    manifest_path = data_root / "assets" / "manifest.csv"
    manifest_df = pd.read_csv(manifest_path)
    manifest_df = manifest_df[manifest_df["task"] == task_id]
    manifest_lookup = {}
    for _, row in manifest_df.iterrows():
        uid = str(row.get("item_uid", "")).strip()
        if uid:
            manifest_lookup[uid] = row

    trials = []
    for i in range(len(dataset)):
        trial = dataset[i]
        uid = trial.get("item_uid", "")
        mrow = manifest_lookup.get(uid)
        if mrow is not None:
            if "prompt_phrase" not in trial:
                phrase = str(mrow.get("prompt_phrase", ""))
                if phrase in ("nan", "NA", ""):
                    phrase = ""
                trial["prompt_phrase"] = phrase
        trials.append(trial)
    return trials


def stratified_subset(trials: list[dict], fraction: float, seed: int = 42) -> list[dict]:
    """Select a stratified subset preserving correct_label distribution."""
    if fraction >= 1.0:
        return trials
    import random
    rng = random.Random(seed)
    by_label: dict[str, list[dict]] = {}
    for t in trials:
        label = t.get("correct_label", "?")
        by_label.setdefault(label, []).append(t)

    subset = []
    for label, group in sorted(by_label.items()):
        k = max(1, int(len(group) * fraction))
        subset.extend(rng.sample(group, min(k, len(group))))
    rng.shuffle(subset)
    return subset


# ── Evaluation ────────────────────────────────────────────────────────

def parse_answer_from_text(text: str, option_labels: list[str]) -> str | None:
    """Lightweight answer parser covering JSON, tags, phrases, bare letter."""
    text = text.strip()
    labels_upper = [l.upper() for l in option_labels]

    # <answer>X</answer> tags
    m = re.search(r"<answer>\s*([A-Z])\s*</answer>", text, re.IGNORECASE)
    if m and m.group(1).upper() in labels_upper:
        return m.group(1).upper()

    # JSON
    try:
        parsed = json.loads(text)
        answer = parsed.get("answer", "").strip().upper()
        if answer in labels_upper:
            return answer
    except (json.JSONDecodeError, AttributeError):
        pass

    # Embedded JSON
    m = re.search(r'"answer"\s*:\s*"?([A-Z])\b', text, re.IGNORECASE)
    if m and m.group(1).upper() in labels_upper:
        return m.group(1).upper()

    # "The answer is X" patterns
    for pat in [
        r"\b(?:the\s+)?(?:correct\s+)?answer\s+is\s+([A-Z])\b",
        r"\b(?:final\s+)?answer\s*[:=]\s*\(?\s*([A-Z])\s*\)?\b",
    ]:
        m = re.search(pat, text, re.IGNORECASE)
        if m and m.group(1).upper() in labels_upper:
            return m.group(1).upper()

    # Strip <think> blocks, then check for bare letter
    clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if clean.upper() in labels_upper:
        return clean.upper()

    # First letter in text
    m = re.match(r"^\s*([A-Z])\s*[)\].:,;]?\s*$", clean, re.IGNORECASE)
    if m and m.group(1).upper() in labels_upper:
        return m.group(1).upper()

    return None


def run_sweep_cell(
    model,
    trial: dict,
    prompt_text: str,
    system_prompt: str | None,
    image_paths: list[str],
    max_new_tokens: int = 128,
) -> dict:
    """Run one (trial, TF, OF) cell and return result dict."""
    # Build messages manually to inject system prompt
    if hasattr(model, "_build_messages"):
        messages = model._build_messages(prompt_text, image_paths or None)
        if system_prompt and messages:
            messages.insert(0, {"role": "system", "content": system_prompt})

    raw_output = model.generate(
        prompt_text=prompt_text,
        image_paths=image_paths if image_paths else None,
        max_new_tokens=max_new_tokens,
    )
    clean_text = model.parse_response(raw_output)
    option_labels = trial.get("option_labels", ["A", "B", "C", "D"])
    predicted = parse_answer_from_text(clean_text, option_labels)

    return {
        "predicted_label": predicted,
        "correct_label": trial["correct_label"],
        "is_correct": predicted == trial["correct_label"] if predicted else False,
        "parse_success": predicted is not None,
        "generated_text": clean_text[:500],
    }


# ── Main sweep ────────────────────────────────────────────────────────

def run_sweep(args):
    data_root = PROJECT_ROOT / "data"
    version = detect_data_version(data_root)
    print(f"Data version: {version}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []

    for model_name in args.models:
        model_size = args.model_sizes if args.model_sizes else None

        base_cfg = load_model_config(model_name)
        if base_cfg is None:
            print(f"  Skip model {model_name}: no config", file=sys.stderr)
            continue

        overrides = {}
        if model_size:
            overrides["size"] = model_size

        model_cfg = resolve_model_config(
            model_name=model_name,
            model_overrides=overrides,
            model_config=base_cfg,
        )

        device = args.device
        if device == "auto":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else (
                    "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
                )
            except ImportError:
                device = "cpu"

        print(f"\n{'='*60}")
        print(f"Model: {model_name} (size={model_size or 'default'}, device={device})")
        print(f"{'='*60}")

        try:
            model = build_model(
                model_name=model_name,
                model_cfg=model_cfg,
                device=device,
                auto_load=True,
            )
        except Exception as exc:
            print(f"  Failed to load {model_name}: {exc}", file=sys.stderr)
            continue

        max_new_tokens = int(model_cfg.get("max_new_tokens", 128))
        model_family = str(model_cfg.get("family", model_name))

        for task_id in args.tasks:
            print(f"\n  Task: {task_id}")

            try:
                prompt_cfg = load_prompt_config(task_id)
            except FileNotFoundError as e:
                print(f"    Skip: {e}", file=sys.stderr)
                continue

            try:
                trials = load_trials(task_id, data_root, version)
            except Exception as e:
                print(f"    Skip: {e}", file=sys.stderr)
                continue

            subset = stratified_subset(trials, args.subset_fraction)
            print(f"    Trials: {len(subset)}/{len(trials)} (subset={args.subset_fraction})")
            print(f"    Label distribution: {Counter(t['correct_label'] for t in subset)}")

            tf_configs = prompt_cfg.get("task_framings", {})
            of_configs = prompt_cfg.get("output_formats", {})

            for tf_name, tf_cfg in tf_configs.items():
                for of_name, of_cfg in of_configs.items():
                    cell_name = f"{tf_name}×{of_name}"
                    t0 = time.time()

                    correct = 0
                    parsed = 0
                    predictions = Counter()

                    for trial in subset:
                        prompt_text, system_prompt, image_paths = assemble_prompt(
                            tf_template=tf_cfg["template"],
                            of_config=of_cfg,
                            trial=trial,
                            prompt_config=prompt_cfg,
                        )

                        # Determine max_new_tokens based on OF
                        tokens = max_new_tokens
                        if of_name in ("OF3_cot", "OF4_tags"):
                            tokens = max(tokens, 256)

                        result = run_sweep_cell(
                            model=model,
                            trial=trial,
                            prompt_text=prompt_text,
                            system_prompt=system_prompt,
                            image_paths=image_paths,
                            max_new_tokens=tokens,
                        )

                        if result["parse_success"]:
                            parsed += 1
                            predictions[result["predicted_label"]] += 1
                        if result["is_correct"]:
                            correct += 1

                    elapsed = time.time() - t0
                    n = len(subset)
                    acc = correct / n if n else 0
                    parse_rate = parsed / n if n else 0

                    # Response bias
                    if parsed > 0:
                        most_common_label, most_common_count = predictions.most_common(1)[0]
                        bias_ratio = most_common_count / parsed
                    else:
                        most_common_label = "?"
                        bias_ratio = 0.0

                    row = {
                        "model": model_name,
                        "model_size": model_size or "",
                        "model_family": model_family,
                        "task": task_id,
                        "task_framing": tf_name,
                        "output_format": of_name,
                        "n_trials": n,
                        "n_correct": correct,
                        "accuracy": round(acc, 4),
                        "n_parsed": parsed,
                        "parse_rate": round(parse_rate, 4),
                        "dominant_label": most_common_label,
                        "bias_ratio": round(bias_ratio, 4),
                        "elapsed_s": round(elapsed, 1),
                    }
                    all_rows.append(row)

                    sig = "*" if acc > 0.5 + 2 / (n ** 0.5) else ""
                    print(
                        f"    {cell_name:30s} "
                        f"acc={acc:.1%} parse={parse_rate:.1%} "
                        f"bias={most_common_label}:{bias_ratio:.0%} "
                        f"({elapsed:.0f}s) {sig}"
                    )

        # Free model memory
        del model
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            import gc
            gc.collect()
        except ImportError:
            pass

    # Save results
    results_path = output_dir / "sweep_results.csv"
    if all_rows:
        with open(results_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nSaved {len(all_rows)} rows to {results_path}")

    # Also save as JSON for richer downstream analysis
    json_path = output_dir / "sweep_results.json"
    with open(json_path, "w") as f:
        json.dump(all_rows, f, indent=2)
    print(f"Saved {json_path}")

    return all_rows


def main():
    parser = argparse.ArgumentParser(description="Prompt Robustness Sweep")
    parser.add_argument(
        "--models", nargs="+", required=True,
        help="Model names (registered in configs/models/)",
    )
    parser.add_argument(
        "--model-sizes", default=None,
        help="Model size override (e.g., 0.8B, 2.2B)",
    )
    parser.add_argument(
        "--tasks", nargs="+",
        default=["mental-rotation", "matrix-reasoning", "vocab", "trog",
                 "theory-of-mind", "egma-math"],
        help="Task IDs to sweep",
    )
    parser.add_argument(
        "--subset-fraction", type=float, default=0.3,
        help="Fraction of trials to use (stratified, default 0.3)",
    )
    parser.add_argument(
        "--output-dir", default="results/prompt_robustness",
        help="Output directory for results",
    )
    parser.add_argument(
        "--device", default="auto",
        help="Device (auto, cpu, cuda, mps)",
    )
    args = parser.parse_args()
    run_sweep(args)


if __name__ == "__main__":
    main()
