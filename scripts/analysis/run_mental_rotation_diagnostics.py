#!/usr/bin/env python3
"""Run targeted diagnostics for mental-rotation model failures.

This script evaluates five diagnostic axes on a configurable subset:
1) model baseline comparison
2) resolution sensitivity
3) prompt variant sensitivity
4) repeat consistency (proxy for self-consistency under greedy decoding)
5) angle/trial-type difficulty breakdowns
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import random
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

from levante_bench.config import get_task_def, load_model_config, load_task_config
from levante_bench.evaluation.runner import resolve_device
from levante_bench.models import get_model_class
from levante_bench.tasks import get_task_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["smolvlm2_2b"],
        help="Model config names in configs/models (e.g., smolvlm2_2b qwen35_0.8b).",
    )
    parser.add_argument("--version", default="2026-03-25")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--sample-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--resolution-sides",
        nargs="*",
        type=int,
        default=[256, 512],
        help="Longest-side values used for resized-image diagnostics.",
    )
    parser.add_argument(
        "--prompt-variants",
        nargs="*",
        default=["structured", "mirror_hint", "feature_cot"],
        choices=["structured", "mirror_hint", "feature_cot"],
    )
    parser.add_argument(
        "--repeat-consistency-runs",
        type=int,
        default=3,
        help="Repeated greedy decodes on baseline prompt.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Override model max_new_tokens from config.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/analysis/mental_rotation_diagnostics",
    )
    return parser.parse_args()


def parse_angle(item_uid: str) -> int:
    s = str(item_uid)
    patterns = [
        r"_(\d{3})_",
        r"-(\d{3})(?:_|$)",
        r"r(\d{3})",
        r"R[pn]-(\d{3})",
    ]
    for pattern in patterns:
        m = re.search(pattern, s, flags=re.IGNORECASE)
        if m:
            return int(m.group(1))
    return 0


def normalize_angle(angle: int) -> int:
    angle = int(angle) % 360
    return min(angle, 360 - angle)


def angle_bucket(angle_norm: int) -> str:
    if angle_norm <= 40:
        return "easy_0_40"
    if angle_norm <= 80:
        return "mid_80"
    return "hard_120_160"


def build_prompt_variant(base_prompt: str, variant: str) -> str:
    if variant == "structured":
        return (
            "Reference image:\n"
            "<image0>\n\n"
            "Which option shows the same object as the reference, rotated but not mirrored?\n"
            "A: <image1>\n"
            "B: <image2>\n\n"
            "Respond with exactly one letter: A or B."
        )
    if variant == "mirror_hint":
        hint = (
            "One option is a true rotation of the reference. The other is a mirror reflection. "
            "Track asymmetric parts and their clockwise order to distinguish rotation vs mirror."
        )
        return f"{hint}\n\n{base_prompt}"
    if variant == "feature_cot":
        cot = (
            "Step 1: Identify one asymmetric landmark in the reference.\n"
            "Step 2: Compare each option and track whether the landmark order is preserved.\n"
            "Step 3: Pick the option that preserves chirality under rotation.\n"
            "Final answer: one letter only (A or B)."
        )
        return f"{base_prompt}\n\n{cot}"
    return base_prompt


def resize_image_cached(src: str, max_side: int, cache_dir: Path) -> str:
    src_path = Path(src)
    key = hashlib.sha1(f"{src_path.resolve()}::{max_side}".encode("utf-8")).hexdigest()
    out_path = cache_dir / f"{key}.png"
    if out_path.exists():
        return str(out_path)
    with Image.open(src_path).convert("RGB") as img:
        w, h = img.size
        m = max(w, h)
        if m > max_side:
            scale = max_side / m
            new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(out_path)
    return str(out_path)


def maybe_resize_paths(paths: list[str], max_side: int | None, cache_root: Path) -> list[str]:
    if max_side is None:
        return paths
    cache_dir = cache_root / str(max_side)
    return [resize_image_cached(p, max_side, cache_dir) for p in paths]


def load_model_from_config(model_cfg_name: str, device: str, max_new_tokens_override: int | None):
    cfg = load_model_config(model_cfg_name)
    if cfg is None:
        raise ValueError(f"Model config not found: {model_cfg_name}")
    cfg = dict(cfg)
    family_name = str(cfg.get("family") or cfg["name"])
    model_cls = get_model_class(family_name)
    if model_cls is None:
        raise ValueError(f"Model class not registered: {family_name}")

    ctor_cfg = {
        k: v
        for k, v in cfg.items()
        if k
        not in {
            "name",
            "family",
            "params_b",
            "hf_name",
            "size",
            "max_new_tokens",
            "use_json_format",
            "capabilities",
        }
    }
    sig = inspect.signature(model_cls.__init__)
    accepted = set(sig.parameters.keys()) - {"self", "model_name", "device"}
    if not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        ctor_cfg = {k: v for k, v in ctor_cfg.items() if k in accepted}

    model = model_cls(model_name=cfg["hf_name"], device=device, **ctor_cfg)
    model.use_json_format = bool(cfg.get("use_json_format", True))
    model.load()
    max_new_tokens = int(max_new_tokens_override or cfg.get("max_new_tokens", 64))
    label = str(cfg.get("name", model_cfg_name))
    return model, max_new_tokens, label


def select_trials(dataset, sample_size: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows = []
    for idx in range(len(dataset)):
        trial = dataset[idx]
        row = dataset.manifest.iloc[idx]
        angle = parse_angle(str(trial["item_uid"]))
        angle_norm = normalize_angle(angle)
        rows.append(
            {
                "idx": idx,
                "trial": trial,
                "trial_type": str(row.get("trial_type", "")),
                "angle": angle,
                "angle_norm": angle_norm,
                "angle_bucket": angle_bucket(angle_norm),
            }
        )
    if sample_size <= 0 or sample_size >= len(rows):
        return rows

    by_type: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_type.setdefault(r["trial_type"], []).append(r)
    selected: list[dict[str, Any]] = []
    for t, group in by_type.items():
        n = max(1, round(sample_size * len(group) / len(rows)))
        rng.shuffle(group)
        selected.extend(group[: min(n, len(group))])
    # adjust exact size
    dedup = {r["idx"]: r for r in selected}
    selected = list(dedup.values())
    if len(selected) < sample_size:
        remaining = [r for r in rows if r["idx"] not in dedup]
        rng.shuffle(remaining)
        selected.extend(remaining[: sample_size - len(selected)])
    elif len(selected) > sample_size:
        rng.shuffle(selected)
        selected = selected[:sample_size]
    selected.sort(key=lambda x: x["idx"])
    return selected


def evaluate_once(
    model,
    max_new_tokens: int,
    trial: dict,
    prompt_text: str,
    image_paths: list[str],
) -> dict[str, Any]:
    t0 = time.time()
    raw = model.generate(
        prompt_text=prompt_text,
        image_paths=image_paths if image_paths else None,
        max_new_tokens=max_new_tokens,
    )
    clean = model.parse_response(raw)
    label, reason = model.parse_answer(clean, trial["option_labels"])
    elapsed = time.time() - t0
    pred = label.upper() if isinstance(label, str) else None
    return {
        "predicted_label": pred,
        "is_correct": pred == trial["correct_label"],
        "parseable": pred is not None,
        "reason": reason,
        "generated_text": clean,
        "latency_s": elapsed,
    }


def summarize(df: pd.DataFrame) -> dict[str, Any]:
    overall = (
        df.groupby(["model_label", "condition"], as_index=False)
        .agg(
            n=("is_correct", "size"),
            acc=("is_correct", "mean"),
            parse_rate=("parseable", "mean"),
            latency_s=("latency_s", "mean"),
        )
        .sort_values(["model_label", "condition"])
    )
    by_type = (
        df.groupby(["model_label", "condition", "trial_type"], as_index=False)
        .agg(n=("is_correct", "size"), acc=("is_correct", "mean"))
        .sort_values(["model_label", "condition", "trial_type"])
    )
    by_angle = (
        df.groupby(["model_label", "condition", "angle_bucket"], as_index=False)
        .agg(n=("is_correct", "size"), acc=("is_correct", "mean"))
        .sort_values(["model_label", "condition", "angle_bucket"])
    )
    return {
        "overall": overall.to_dict(orient="records"),
        "by_trial_type": by_type.to_dict(orient="records"),
        "by_angle_bucket": by_angle.to_dict(orient="records"),
    }


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = Path.cwd() / data_root
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = Path.cwd() / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    run_tag = time.strftime("%Y%m%d-%H%M%S")
    run_dir = out_dir / run_tag
    run_dir.mkdir(parents=True, exist_ok=True)

    task_cfg = load_task_config("mental-rotation")
    if task_cfg is None:
        raise RuntimeError("Task config missing: mental-rotation")
    task_def = get_task_def("mental-rotation", args.version, data_root=data_root)
    dataset_cls = get_task_dataset("mental-rotation")
    dataset = dataset_cls(task_def=task_def, version=args.version, data_root=data_root)
    selected = select_trials(dataset, args.sample_size, args.seed)

    cache_root = run_dir / "resized_cache"
    records: list[dict[str, Any]] = []
    consistency_records: list[dict[str, Any]] = []

    for model_cfg_name in args.models:
        model, max_new_tokens, model_label = load_model_from_config(
            model_cfg_name=model_cfg_name,
            device=device,
            max_new_tokens_override=args.max_new_tokens,
        )
        print(f"[diagnostics] model={model_cfg_name} label={model_label} device={device}")

        for row in selected:
            trial = row["trial"]
            base_prompt = trial["prompt"]
            base_paths = trial["context_image_paths"] + trial["option_image_paths"]

            # Baseline native resolution
            r = evaluate_once(model, max_new_tokens, trial, base_prompt, base_paths)
            records.append(
                {
                    "model_cfg": model_cfg_name,
                    "model_label": model_label,
                    "condition": "baseline_native",
                    "prompt_variant": "baseline",
                    "resolution": "native",
                    "item_uid": trial["item_uid"],
                    "trial_type": row["trial_type"],
                    "angle": row["angle"],
                    "angle_norm": row["angle_norm"],
                    "angle_bucket": row["angle_bucket"],
                    "correct_label": trial["correct_label"],
                    **r,
                }
            )

            # Resolution sensitivity on baseline prompt
            for max_side in args.resolution_sides:
                resized = maybe_resize_paths(base_paths, max_side, cache_root)
                rr = evaluate_once(model, max_new_tokens, trial, base_prompt, resized)
                records.append(
                    {
                        "model_cfg": model_cfg_name,
                        "model_label": model_label,
                        "condition": f"resolution_{max_side}",
                        "prompt_variant": "baseline",
                        "resolution": str(max_side),
                        "item_uid": trial["item_uid"],
                        "trial_type": row["trial_type"],
                        "angle": row["angle"],
                        "angle_norm": row["angle_norm"],
                        "angle_bucket": row["angle_bucket"],
                        "correct_label": trial["correct_label"],
                        **rr,
                    }
                )

            # Prompt variants at native resolution
            for variant in args.prompt_variants:
                prompt_variant = build_prompt_variant(base_prompt, variant)
                rv = evaluate_once(model, max_new_tokens, trial, prompt_variant, base_paths)
                records.append(
                    {
                        "model_cfg": model_cfg_name,
                        "model_label": model_label,
                        "condition": f"prompt_{variant}",
                        "prompt_variant": variant,
                        "resolution": "native",
                        "item_uid": trial["item_uid"],
                        "trial_type": row["trial_type"],
                        "angle": row["angle"],
                        "angle_norm": row["angle_norm"],
                        "angle_bucket": row["angle_bucket"],
                        "correct_label": trial["correct_label"],
                        **rv,
                    }
                )

            # Repeat consistency (greedy, same prompt/images)
            if args.repeat_consistency_runs > 1:
                labels = []
                texts = []
                for _ in range(args.repeat_consistency_runs):
                    rc = evaluate_once(model, max_new_tokens, trial, base_prompt, base_paths)
                    labels.append(rc["predicted_label"])
                    texts.append(rc["generated_text"])
                unique_labels = len({x for x in labels})
                consistency_records.append(
                    {
                        "model_cfg": model_cfg_name,
                        "model_label": model_label,
                        "item_uid": trial["item_uid"],
                        "trial_type": row["trial_type"],
                        "angle_norm": row["angle_norm"],
                        "runs": args.repeat_consistency_runs,
                        "unique_labels": unique_labels,
                        "all_same_label": int(unique_labels == 1),
                        "labels": json.dumps(labels),
                        "texts": json.dumps(texts),
                    }
                )

    df = pd.DataFrame(records)
    detail_path = run_dir / "diagnostic_details.csv"
    df.to_csv(detail_path, index=False)

    summary_obj = summarize(df)
    summary_path = run_dir / "diagnostic_summary.json"
    summary_path.write_text(json.dumps(summary_obj, indent=2), encoding="utf-8")

    consistency_path = run_dir / "repeat_consistency.csv"
    if consistency_records:
        pd.DataFrame(consistency_records).to_csv(consistency_path, index=False)
    else:
        consistency_path.write_text("runs=1; no repeat consistency computed\n", encoding="utf-8")

    # Human-readable digest
    digest_lines = []
    digest_lines.append(f"run_dir: {run_dir}")
    digest_lines.append(f"models: {', '.join(args.models)}")
    digest_lines.append(f"device: {device}")
    digest_lines.append(f"n_trials: {len(selected)}")
    digest_lines.append("")
    digest_lines.append("Overall accuracy by condition:")
    overall = pd.DataFrame(summary_obj["overall"])
    if not overall.empty:
        for _, r in overall.iterrows():
            digest_lines.append(
                f"- {r['model_label']} | {r['condition']}: "
                f"acc={r['acc']:.3f} ({int(r['n'])} items), parse={r['parse_rate']:.3f}, "
                f"latency={r['latency_s']:.2f}s"
            )
    if consistency_records:
        cdf = pd.DataFrame(consistency_records)
        digest_lines.append("")
        digest_lines.append("Repeat consistency (greedy):")
        for model_label, g in cdf.groupby("model_label"):
            digest_lines.append(
                f"- {model_label}: all_same_label_rate={g['all_same_label'].mean():.3f} "
                f"({len(g)} trials, runs={args.repeat_consistency_runs})"
            )
        digest_lines.append(
            "- Note: current model adapters use greedy decoding; this consistency check "
            "is a determinism proxy, not stochastic self-consistency voting."
        )

    digest_path = run_dir / "digest.md"
    digest_path.write_text("\n".join(digest_lines) + "\n", encoding="utf-8")

    print(f"[done] wrote {detail_path}")
    print(f"[done] wrote {summary_path}")
    print(f"[done] wrote {consistency_path}")
    print(f"[done] wrote {digest_path}")


if __name__ == "__main__":
    main()
