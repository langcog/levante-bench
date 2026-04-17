#!/usr/bin/env python
"""
Item-level difficulty correlation: VLM accuracy vs. IRT difficulty from children.

Reuses infrastructure from prompt_robustness_sweep.py.
Runs the best prompt (TF×OF from sweep) on ALL items for selected tasks/models,
saves per-item predictions, then correlates with IRT difficulty.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import yaml

from levante_bench.config import get_task_def, load_model_config, load_task_config
from levante_bench.config.defaults import detect_data_version
from levante_bench.runtime.modeling import build_model, resolve_model_config
from levante_bench.tasks import get_task_dataset

PROMPTS_DIR = PROJECT_ROOT / "configs" / "prompts"

BEST_PROMPTS = {
    "vocab":            ("TF1_minimal", "OF1_bare"),
    "trog":             ("TF1_minimal", "OF2_json"),
    "egma-math":        ("TF1_minimal", "OF2_json"),
    "mental-rotation":  ("TF1_minimal", "OF1_bare"),
}

IRT_FILES = {
    "vocab":           "data/assets/2026-03-26/corpus/vocab/vocab-item-bank.csv",
    "trog":            "data/assets/2026-03-26/corpus/trog/trog-item-bank-full-params.csv",
    "egma-math":       "data/assets/2026-03-26/corpus/egma-math/test-combined-math-cat.csv",
    "mental-rotation": "data/assets/2026-03-26/corpus/mental-rotation/mental-rotation-item-bank.csv",
}


def load_irt_difficulty(task_id: str) -> dict[str, float]:
    path = PROJECT_ROOT / IRT_FILES[task_id]
    lookup = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            uid = row.get("item_uid", "").strip()
            d = row.get("d", "").strip()
            diff = row.get("difficulty", "").strip()
            val = d if d and d not in ("NA", "None", "") else diff
            if val and val not in ("NA", "None", ""):
                try:
                    lookup[uid] = float(val)
                except ValueError:
                    pass
    return lookup


def load_prompt_config(task_id: str) -> dict:
    yaml_name = task_id.replace("-", "_") + ".yaml"
    path = PROMPTS_DIR / yaml_name
    with open(path) as f:
        return yaml.safe_load(f)


def _image_tag(idx: int) -> str:
    return f"<image{idx}>"


def assemble_prompt(
    tf_template: str,
    of_config: dict,
    trial: dict,
    prompt_config: dict,
) -> tuple[str, str | None, list[str]]:
    """Copied from prompt_robustness_sweep.py — identical logic."""
    labels = prompt_config["labels"]

    context_paths = trial.get("context_image_paths", [])
    option_paths = trial.get("option_image_paths", [])
    image_paths: list[str] = []

    prompt_text = tf_template

    if "{full_prompt}" in prompt_text:
        full_prompt = trial.get("prompt", "")
        for i in range(1, 5):
            full_prompt = full_prompt.replace(f"<image{i}>", f"{{opt_{chr(96+i)}_image}}")
        prompt_text = prompt_text.replace("{full_prompt}", full_prompt)

    prompt_phrase = trial.get("prompt_phrase", "")
    prompt_text = prompt_text.replace("{prompt_phrase}", str(prompt_phrase))

    options = trial.get("options", [])
    for i, opt in enumerate(options):
        prompt_text = prompt_text.replace(f"{{option{i+1}}}", str(opt))

    img_idx = 0
    if context_paths:
        prompt_text = prompt_text.replace("{ref_image}", _image_tag(img_idx))
        prompt_text = prompt_text.replace("{context_image}", _image_tag(img_idx))
        image_paths.append(context_paths[0])
        img_idx += 1
    else:
        prompt_text = prompt_text.replace("{ref_image}", "")
        prompt_text = prompt_text.replace("{context_image}", "")

    option_labels = ["a", "b", "c", "d"]
    for i, opt_path in enumerate(option_paths):
        placeholder = f"{{opt_{option_labels[i]}_image}}"
        prompt_text = prompt_text.replace(placeholder, _image_tag(img_idx))
        image_paths.append(opt_path)
        img_idx += 1

    of_suffix = of_config.get("suffix", "")
    of_suffix = of_suffix.replace("{labels}", labels)
    prompt_text = prompt_text.strip() + of_suffix

    system = of_config.get("system")
    if system:
        system = system.replace("{labels}", labels)

    prompt_text = re.sub(r"\n{3,}", "\n\n", prompt_text).strip()
    return prompt_text, system, image_paths


def load_trials(task_id: str, data_root: Path, version: str) -> list[dict]:
    """Load trials — same as sweep."""
    import pandas as pd
    task_cfg = load_task_config(task_id)
    task_def = get_task_def(task_id, version, data_root=data_root)
    dataset_cls = get_task_dataset(task_id)
    dataset = dataset_cls(task_def=task_def, version=version, data_root=data_root)

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


def parse_answer_from_text(text: str, option_labels: list[str]) -> str | None:
    """Copied from sweep — identical parse logic."""
    if not text:
        return None
    text_clean = text.strip()
    if text_clean and text_clean[0] in option_labels:
        if len(text_clean) == 1 or not text_clean[1].isalpha():
            return text_clean[0]
    for pat in [
        r'"answer"\s*:\s*"([A-Z])"',
        r"\b(?:the\s+)?(?:correct\s+)?answer\s+is\s+([A-Z])\b",
        r"\b(?:final\s+)?answer\s*[:=]\s*\(?\s*([A-Z])\s*\)?\b",
        r"<answer>\s*([A-Z])\s*</answer>",
    ]:
        m = re.search(pat, text_clean, re.IGNORECASE)
        if m:
            letter = m.group(1).upper()
            if letter in option_labels:
                return letter
    for label in option_labels:
        if re.search(rf"\b{label}\b", text_clean):
            return label
    return None


def main():
    parser = argparse.ArgumentParser(description="Item-level difficulty correlation")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--model-sizes", default=None)
    parser.add_argument("--tasks", nargs="+",
                        default=["vocab", "trog", "egma-math", "mental-rotation"])
    parser.add_argument("--output-dir", default="results/item_difficulty")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    data_root = PROJECT_ROOT / "data"
    version = detect_data_version(data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

    all_item_rows = []

    for model_name in args.models:
        model_size = args.model_sizes

        print(f"\n{'='*60}")
        print(f"Model: {model_name} (size={model_size or 'default'}, device={device})")
        print(f"{'='*60}")

        model_cfg = resolve_model_config(
            model_name, {"size": model_size} if model_size else None)

        try:
            model = build_model(model_name, model_cfg, device=device, auto_load=True)
        except Exception as exc:
            print(f"  FAILED to load model: {exc}", file=sys.stderr)
            continue

        max_new_tokens = int(model_cfg.get("max_new_tokens", 128))

        for task_id in args.tasks:
            if task_id not in BEST_PROMPTS:
                print(f"  Skip {task_id}: no best prompt defined")
                continue
            if task_id not in IRT_FILES:
                print(f"  Skip {task_id}: no IRT data")
                continue

            tf_name, of_name = BEST_PROMPTS[task_id]
            print(f"\n  Task: {task_id} (prompt: {tf_name}×{of_name})")

            prompt_cfg = load_prompt_config(task_id)
            tf_template = prompt_cfg["task_framings"][tf_name]["template"]
            of_config = prompt_cfg["output_formats"][of_name]

            irt = load_irt_difficulty(task_id)
            print(f"    IRT difficulty available for {len(irt)} items")

            trials = load_trials(task_id, data_root, version)
            print(f"    Total trials: {len(trials)}")

            t0 = time.time()
            for i, trial in enumerate(trials):
                uid = trial.get("item_uid", f"item_{i}")
                prompt_text, system_prompt, image_paths = assemble_prompt(
                    tf_template, of_config, trial, prompt_cfg)

                try:
                    raw_output = model.generate(
                        prompt_text=prompt_text,
                        image_paths=image_paths if image_paths else None,
                        max_new_tokens=max_new_tokens,
                    )
                    clean = model.parse_response(raw_output)
                except Exception as e:
                    clean = ""
                    print(f"    ERROR on {uid}: {e}", flush=True)

                option_labels = trial.get("option_labels", ["A", "B", "C", "D"])
                predicted = parse_answer_from_text(clean, option_labels)
                correct_label = trial["correct_label"]
                is_correct = predicted == correct_label if predicted else False

                item_difficulty = irt.get(uid)

                all_item_rows.append({
                    "model": model_name,
                    "model_size": model_size or "",
                    "task": task_id,
                    "item_uid": uid,
                    "tf": tf_name,
                    "of": of_name,
                    "predicted": predicted or "",
                    "correct_label": correct_label,
                    "is_correct": int(is_correct),
                    "parsed": int(predicted is not None),
                    "irt_difficulty": item_difficulty if item_difficulty is not None else "",
                })

                if (i + 1) % 20 == 0:
                    elapsed = time.time() - t0
                    n_done = i + 1
                    task_rows = [r for r in all_item_rows
                                 if r["model"] == model_name and r["task"] == task_id]
                    acc_so_far = sum(r["is_correct"] for r in task_rows) / n_done
                    print(f"    {n_done}/{len(trials)} ({elapsed:.0f}s, "
                          f"acc={acc_so_far:.1%})", flush=True)

            elapsed = time.time() - t0
            task_items = [r for r in all_item_rows
                          if r["model"] == model_name and r["task"] == task_id]
            with_irt = [r for r in task_items if r["irt_difficulty"] != ""]
            correct_count = sum(r["is_correct"] for r in task_items)
            parsed_count = sum(r["parsed"] for r in task_items)
            acc = correct_count / len(task_items) if task_items else 0
            print(f"    Done: {correct_count}/{len(task_items)} correct "
                  f"({acc:.1%}), {parsed_count} parsed, "
                  f"{len(with_irt)} with IRT, {elapsed:.0f}s")

            if len(with_irt) >= 10:
                difficulties = [float(r["irt_difficulty"]) for r in with_irt]
                correctness = [r["is_correct"] for r in with_irt]
                r_val, p_val = stats.pointbiserialr(correctness, difficulties)
                rho, rho_p = stats.spearmanr(correctness, difficulties)
                print(f"    Correlation: r_pb={r_val:.3f} (p={p_val:.3f}), "
                      f"rho={rho:.3f} (p={rho_p:.3f})")

        del model
        try:
            import torch, gc
            torch.cuda.empty_cache()
            gc.collect()
        except:
            pass

    items_path = output_dir / "item_results.csv"
    if all_item_rows:
        with open(items_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_item_rows[0].keys())
            writer.writeheader()
            writer.writerows(all_item_rows)
        print(f"\nSaved {len(all_item_rows)} item rows to {items_path}")

    print(f"\n{'='*60}")
    print("CORRELATION SUMMARY")
    print(f"{'='*60}")
    by_model_task = defaultdict(list)
    for r in all_item_rows:
        if r["irt_difficulty"] != "":
            by_model_task[(r["model"], r["task"])].append(r)

    summary_rows = []
    for (model_name, task_id), items in sorted(by_model_task.items()):
        if len(items) < 10:
            continue
        difficulties = [float(r["irt_difficulty"]) for r in items]
        correctness = [r["is_correct"] for r in items]
        n = len(items)
        acc = sum(correctness) / n
        r_val, p_val = stats.pointbiserialr(correctness, difficulties)
        rho, rho_p = stats.spearmanr(correctness, difficulties)
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        print(f"  {model_name:15s} {task_id:20s} n={n:3d} acc={acc:.1%} "
              f"r_pb={r_val:+.3f}{sig:3s} rho={rho:+.3f}")
        summary_rows.append({
            "model": model_name, "task": task_id, "n_items": n,
            "accuracy": round(acc, 4),
            "r_pointbiserial": round(r_val, 4), "p_pointbiserial": round(p_val, 4),
            "rho_spearman": round(rho, 4), "p_spearman": round(rho_p, 4),
        })

    summary_path = output_dir / "correlation_summary.csv"
    if summary_rows:
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    main()
