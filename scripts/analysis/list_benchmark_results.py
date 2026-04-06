#!/usr/bin/env python3
"""Summarize benchmark outputs and deltas across runs."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class V1Run:
    run_id: str
    path: Path
    max_items_math: int | None
    max_items_tom: int | None
    math_accuracy_all: float | None
    tom_none_mean_accuracy: float | None
    tom_state_model_mean_accuracy: float | None


@dataclass
class VocabRun:
    run_id: str
    path: Path
    n_total: int | None
    accuracy_all: float | None
    lift_vs_chance: float | None


@dataclass
class PromptRun:
    run_id: str
    group: str
    path: Path
    n_total: int | None
    accuracy_all: float | None
    parse_rate: float | None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="List benchmark outputs and compare with prior runs.")
    p.add_argument(
        "--results-root",
        type=Path,
        default=Path("results/benchmark"),
        help="Benchmark results root (default: results/benchmark).",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Max runs to show per benchmark type (default: 10).",
    )
    p.add_argument(
        "--prompts-root",
        type=Path,
        default=Path("results/prompts"),
        help="Prompt experiment results root (default: results/prompts).",
    )
    return p.parse_args()


def _safe_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _safe_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _load_v1_run(run_dir: Path) -> V1Run | None:
    summary_csv = run_dir / "benchmark_summary.csv"
    if not summary_csv.exists():
        return None
    metrics: dict[str, str] = {}
    with open(summary_csv, newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for row in rd:
            metrics[(row.get("metric") or "").strip()] = (row.get("value") or "").strip()

    run_meta = run_dir / "run_metadata.json"
    meta: dict[str, object] = {}
    if run_meta.exists():
        meta = json.loads(run_meta.read_text(encoding="utf-8"))

    return V1Run(
        run_id=run_dir.name,
        path=run_dir,
        max_items_math=_safe_int(meta.get("max_items_math")),
        max_items_tom=_safe_int(meta.get("max_items_tom")),
        math_accuracy_all=_safe_float(metrics.get("math_accuracy_all")),
        tom_none_mean_accuracy=_safe_float(metrics.get("tom_none_mean_accuracy")),
        tom_state_model_mean_accuracy=_safe_float(metrics.get("tom_state_model_mean_accuracy")),
    )


def _load_vocab_run(run_dir: Path) -> VocabRun | None:
    summary_json = run_dir / "vocab-summary.json"
    if not summary_json.exists():
        return None
    data = json.loads(summary_json.read_text(encoding="utf-8"))
    return VocabRun(
        run_id=run_dir.name,
        path=run_dir,
        n_total=_safe_int(data.get("n_total")),
        accuracy_all=_safe_float(str(data.get("accuracy_all", ""))),
        lift_vs_chance=_safe_float(str(data.get("lift_vs_chance", ""))),
    )


def _prompt_group_from_id(run_id: str) -> str:
    low = run_id.lower()
    if "numberline" in low:
        return "math-numberline"
    if "math" in low:
        return "math"
    if "tom" in low:
        return "tom"
    if "vocab" in low:
        return "vocab"
    return "other"


def _load_prompt_run(summary_json: Path) -> PromptRun | None:
    try:
        data = json.loads(summary_json.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if "accuracy_all" not in data:
        return None
    run_id = summary_json.stem
    if run_id.endswith("-summary"):
        run_id = run_id[: -len("-summary")]
    return PromptRun(
        run_id=run_id,
        group=_prompt_group_from_id(run_id),
        path=summary_json,
        n_total=_safe_int(data.get("n_total")),
        accuracy_all=_safe_float(str(data.get("accuracy_all", ""))),
        parse_rate=_safe_float(str(data.get("parse_rate", ""))),
    )


def _fmt_float(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.4f}"


def _fmt_delta(current: float | None, previous: float | None) -> str:
    if current is None or previous is None:
        return "-"
    delta = current - previous
    return f"{delta:+.4f}"


def _run_type(max_items_math: int | None, max_items_tom: int | None) -> str:
    if max_items_math is None and max_items_tom is None:
        return "full"
    return "smoke/partial"


def _print_v1_table(runs: list[V1Run], limit: int) -> None:
    print("\nV1 benchmark runs")
    print(
        "run_id | type | math_acc | d_math | tom_none | d_tom_none | tom_state_model | d_tom_state | path"
    )
    print("-" * 128)
    shown = runs[-limit:]
    for i, run in enumerate(shown):
        prev = shown[i - 1] if i > 0 else None
        print(
            " | ".join(
                [
                    run.run_id,
                    _run_type(run.max_items_math, run.max_items_tom),
                    _fmt_float(run.math_accuracy_all),
                    _fmt_delta(run.math_accuracy_all, prev.math_accuracy_all if prev else None),
                    _fmt_float(run.tom_none_mean_accuracy),
                    _fmt_delta(run.tom_none_mean_accuracy, prev.tom_none_mean_accuracy if prev else None),
                    _fmt_float(run.tom_state_model_mean_accuracy),
                    _fmt_delta(
                        run.tom_state_model_mean_accuracy,
                        prev.tom_state_model_mean_accuracy if prev else None,
                    ),
                    str(run.path),
                ]
            )
        )


def _print_vocab_table(runs: list[VocabRun], limit: int) -> None:
    print("\nVocab benchmark runs")
    print("run_id | n_total | accuracy | d_accuracy | lift_vs_chance | d_lift | path")
    print("-" * 128)
    shown = runs[-limit:]
    for i, run in enumerate(shown):
        prev = shown[i - 1] if i > 0 else None
        print(
            " | ".join(
                [
                    run.run_id,
                    str(run.n_total) if run.n_total is not None else "-",
                    _fmt_float(run.accuracy_all),
                    _fmt_delta(run.accuracy_all, prev.accuracy_all if prev else None),
                    _fmt_float(run.lift_vs_chance),
                    _fmt_delta(run.lift_vs_chance, prev.lift_vs_chance if prev else None),
                    str(run.path),
                ]
            )
        )


def _print_prompt_tables(runs: list[PromptRun], limit: int) -> None:
    print("\nPrompt experiment runs (results/prompts)")
    if not runs:
        print("- none found")
        return
    groups = sorted({r.group for r in runs})
    for group in groups:
        group_runs = [r for r in runs if r.group == group]
        shown = group_runs[-limit:]
        print(f"\n[{group}]")
        print("run_id | n_total | accuracy | d_accuracy | parse_rate | path")
        print("-" * 128)
        for i, run in enumerate(shown):
            prev = shown[i - 1] if i > 0 else None
            print(
                " | ".join(
                    [
                        run.run_id,
                        str(run.n_total) if run.n_total is not None else "-",
                        _fmt_float(run.accuracy_all),
                        _fmt_delta(run.accuracy_all, prev.accuracy_all if prev else None),
                        _fmt_float(run.parse_rate),
                        str(run.path),
                    ]
                )
            )


def main() -> None:
    args = parse_args()
    root = args.results_root
    v1_root = root / "v1"
    vocab_root = root / "vocab"
    prompts_root = args.prompts_root

    v1_runs: list[V1Run] = []
    if v1_root.exists():
        for run_dir in sorted([p for p in v1_root.iterdir() if p.is_dir()]):
            run = _load_v1_run(run_dir)
            if run is not None:
                v1_runs.append(run)

    vocab_runs: list[VocabRun] = []
    if vocab_root.exists():
        for run_dir in sorted([p for p in vocab_root.iterdir() if p.is_dir()]):
            run = _load_vocab_run(run_dir)
            if run is not None:
                vocab_runs.append(run)

    prompt_runs: list[PromptRun] = []
    if prompts_root.exists():
        summary_paths = sorted(
            [p for p in prompts_root.glob("*summary.json") if p.is_file()],
            key=lambda p: p.stat().st_mtime,
        )
        for summary_path in summary_paths:
            run = _load_prompt_run(summary_path)
            if run is not None:
                prompt_runs.append(run)

    print(f"Results root: {root}")
    if v1_runs:
        _print_v1_table(v1_runs, args.limit)
    else:
        print("\nV1 benchmark runs\n- none found")

    if vocab_runs:
        _print_vocab_table(vocab_runs, args.limit)
    else:
        print("\nVocab benchmark runs\n- none found")

    _print_prompt_tables(prompt_runs, args.limit)


if __name__ == "__main__":
    main()
