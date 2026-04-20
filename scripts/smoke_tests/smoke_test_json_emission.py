#!/usr/bin/env python3
"""Smoke test: measure JSON emission + parse rate across local models.

Runs a small sample of trials from each of the 5 multi-modal tasks and
reports how many trials emit a valid JSON object (json-repair parseable),
how many produce a parsed answer, and how many are correct.

Usage:
    python scripts/smoke_tests/smoke_test_json_emission.py \
        --model qwen35 \
        --hf-name Qwen/Qwen3.5-0.8B \
        --n-per-task 4
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from levante_bench.config import detect_data_version, get_task_def
from levante_bench.models.base import _try_json_repair
from levante_bench.tasks.matrix_reasoning import MatrixReasoningDataset
from levante_bench.tasks.mental_rotation import MentalRotationDataset
from levante_bench.tasks.theory_of_mind import TheoryOfMindDataset
from levante_bench.tasks.trog import TrogDataset
from levante_bench.tasks.vocab import VocabDataset

DATA_ROOT = REPO_ROOT / "data"

DATASETS = {
    "matrix-reasoning": MatrixReasoningDataset,
    "mental-rotation": MentalRotationDataset,
    "vocab": VocabDataset,
    "trog": TrogDataset,
    "theory-of-mind": TheoryOfMindDataset,
}


def _build_model(model_family: str, hf_name: str, device: str, dtype: str):
    """Instantiate a model by family name."""
    if model_family == "qwen35":
        from levante_bench.models.qwen35 import Qwen35Model
        return Qwen35Model(model_name=hf_name, device=device, dtype=dtype)
    if model_family == "internvl35":
        from levante_bench.models.internvl35 import InternVL35Model
        return InternVL35Model(model_name=hf_name, device=device, dtype=dtype)
    if model_family == "gemma3":
        from levante_bench.models.gemma3 import Gemma3Model
        return Gemma3Model(model_name=hf_name, device=device, dtype=dtype)
    if model_family == "gemma4":
        from levante_bench.models.gemma4 import Gemma4Model
        return Gemma4Model(model_name=hf_name, device=device, dtype=dtype)
    if model_family == "smolvlm2":
        from levante_bench.models.smolvlm2 import SmolVLM2Model
        return SmolVLM2Model(model_name=hf_name, device=device, dtype=dtype)
    if model_family == "tinyllava":
        from levante_bench.models.tinyllava import TinyLLaVAModel
        return TinyLLaVAModel(model_name=hf_name, device=device, dtype=dtype)
    raise ValueError(f"Unknown model family: {model_family}")


def has_json(text: str) -> bool:
    parsed = _try_json_repair(text)
    return isinstance(parsed, dict) and "answer" in parsed


def run_model(
    model_family: str,
    hf_name: str,
    n_per_task: int,
    device: str,
    dtype: str,
    format_mode: str = "json",
) -> dict:
    version = detect_data_version(DATA_ROOT)
    print(f"\n{'='*72}\n{model_family} ({hf_name}) on {device} [{format_mode}]\n{'='*72}")
    model = _build_model(model_family, hf_name, device, dtype)
    # JSON format is the unified default; other modes are experimental for
    # very small models that struggle with the JSON template.
    model.use_json_format = format_mode == "json"
    model.load()

    summary: dict[str, dict[str, int]] = {}
    for task_name, ds_cls in DATASETS.items():
        task_def = get_task_def(task_name, version, data_root=DATA_ROOT)
        dataset = ds_cls(task_def=task_def, version=version, data_root=DATA_ROOT)
        n = min(n_per_task, len(dataset))
        stats = {"total": n, "json": 0, "parsed": 0, "correct": 0}
        print(f"\n-- {task_name} ({n}/{len(dataset)}) --")
        for i in range(n):
            trial = dataset[i]
            try:
                result = model.evaluate_trial(trial)
            except Exception as exc:
                print(f"  [{i+1}/{n}] ERROR: {exc!r}")
                continue
            raw = result["generated_text"]
            json_ok = has_json(raw)
            parsed = (
                result.get("predicted_label") is not None
                or result.get("predicted_value") is not None
            )
            stats["json"] += int(json_ok)
            stats["parsed"] += int(parsed)
            stats["correct"] += int(bool(result["is_correct"]))
            status = "✓" if result["is_correct"] else "✗"
            pred = result.get("predicted_label") or result.get("predicted_value")
            raw_preview = re.sub(r"\s+", " ", raw.strip())[:90]
            print(
                f"  [{i+1}/{n}] {status} pred={pred!r} "
                f"json={'Y' if json_ok else 'N'} raw={raw_preview!r}"
            )
        summary[task_name] = stats

    # Per-model totals
    tot = {k: sum(s[k] for s in summary.values()) for k in ("total", "json", "parsed", "correct")}
    print(f"\n--- {model_family} totals ---")
    print(f"{'task':<20} {'n':>3} {'json':>5} {'parsed':>6} {'correct':>7}")
    for task_name, s in summary.items():
        print(f"{task_name:<20} {s['total']:>3} {s['json']:>5} {s['parsed']:>6} {s['correct']:>7}")
    print(
        f"{'TOTAL':<20} {tot['total']:>3} {tot['json']:>5} {tot['parsed']:>6} {tot['correct']:>7}"
    )
    return {"family": model_family, "hf_name": hf_name, "per_task": summary, "total": tot}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model family (qwen35, internvl35, ...)")
    parser.add_argument("--hf-name", required=True, help="HF repo identifier")
    parser.add_argument("--n-per-task", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument(
        "--format",
        choices=["json", "simple"],
        default="json",
        help=(
            "json = full JSON prompt (option-aware); "
            "simple = terse letter-only instruction (for tiny models that echo JSON placeholders)."
        ),
    )
    args = parser.parse_args()
    run_model(
        args.model,
        args.hf_name,
        args.n_per_task,
        args.device,
        args.dtype,
        format_mode=args.format,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
