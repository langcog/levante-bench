#!/usr/bin/env python3
"""Run multi-seed ToM robustness experiments and aggregate metrics."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run ToM robustness across variants/seeds.")
    p.add_argument("--corpus-csv", type=Path, required=True, help="Path to theory-of-mind-item-bank.csv")
    p.add_argument(
        "--variants",
        default="none,oracle,model",
        help="Comma-separated memory modes to compare (none,oracle,model).",
    )
    p.add_argument(
        "--seeds",
        default="1,2,3,4,5",
        help="Comma-separated shuffle seeds.",
    )
    p.add_argument("--device", default="cuda", choices=["auto", "cpu", "cuda"], help="Execution device")
    p.add_argument(
        "--model-id",
        default="HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        help="Hugging Face model id",
    )
    p.add_argument(
        "--memory-format",
        default="detailed",
        choices=["detailed", "compact", "belief_state"],
        help="Memory representation used for oracle/model modes.",
    )
    p.add_argument("--history-window", type=int, default=8, help="History window for compact/belief_state.")
    p.add_argument(
        "--reasoning-instruction",
        default="standard",
        choices=["standard", "facts_only"],
        help="Prompt instruction style.",
    )
    p.add_argument(
        "--template-style",
        default="standard",
        choices=["standard", "trial_aware"],
        help="Template style forwarded to ToM evaluator.",
    )
    p.add_argument(
        "--two-stage",
        action="store_true",
        help="Enable two-stage reasoning in ToM evaluator.",
    )
    p.add_argument(
        "--analysis-max-new-tokens",
        type=int,
        default=64,
        help="Stage-1 token budget when two-stage is enabled.",
    )
    p.add_argument("--max-items", type=int, default=None, help="Optional cap for quick smoke runs")
    p.add_argument(
        "--output-prefix",
        default="tom-robust",
        help="Prefix for produced files in results/prompts/",
    )
    p.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/prompts"),
        help="Directory to store run outputs.",
    )
    return p.parse_args()


def _run_one(args: argparse.Namespace, variant: str, seed: int) -> Path:
    out_pred = args.results_dir / f"{args.output_prefix}-{variant}-s{seed}-preds.jsonl"
    out_sum = args.results_dir / f"{args.output_prefix}-{variant}-s{seed}-summary.json"
    out_trace = args.results_dir / f"{args.output_prefix}-{variant}-s{seed}-sequence-trace.csv"
    cmd = [
        sys.executable,
        "scripts/run_smolvlmv2_tom_eval.py",
        "--corpus-csv",
        str(args.corpus_csv),
        "--output-jsonl",
        str(out_pred),
        "--summary-json",
        str(out_sum),
        "--memory-mode",
        variant,
        "--memory-format",
        args.memory_format,
        "--history-window",
        str(args.history_window),
        "--reasoning-instruction",
        args.reasoning_instruction,
        "--template-style",
        args.template_style,
        "--shuffle-options",
        "--seed",
        str(seed),
        "--device",
        args.device,
        "--model-id",
        args.model_id,
        "--strict-order-check",
        "--sequence-trace-csv",
        str(out_trace),
    ]
    if args.two_stage:
        cmd.extend(["--two-stage", "--analysis-max-new-tokens", str(args.analysis_max_new_tokens)])
    if args.max_items is not None:
        cmd.extend(["--max-items", str(args.max_items)])
    subprocess.run(cmd, check=True)
    return out_sum


def _aggregate(
    output_summaries: list[tuple[str, int, Path]],
    out_per_run: Path,
    out_summary: Path,
) -> None:
    rows: list[dict[str, object]] = []
    for variant, seed, path in output_summaries:
        s = json.loads(path.read_text())
        rows.append(
            {
                "variant": variant,
                "seed": seed,
                "n_total": s["n_total"],
                "accuracy_all": s["accuracy_all"],
                "parse_rate": s["parse_rate"],
            }
        )

    with open(out_per_run, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["variant", "seed", "n_total", "accuracy_all", "parse_rate"])
        w.writeheader()
        w.writerows(rows)

    by_variant: dict[str, list[float]] = {}
    for r in rows:
        by_variant.setdefault(str(r["variant"]), []).append(float(r["accuracy_all"]))
    agg_rows = []
    for v in sorted(by_variant):
        vals = by_variant[v]
        agg_rows.append(
            {
                "variant": v,
                "mean_accuracy": statistics.mean(vals),
                "std_accuracy": statistics.pstdev(vals),
                "min_accuracy": min(vals),
                "max_accuracy": max(vals),
            }
        )
    with open(out_summary, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["variant", "mean_accuracy", "std_accuracy", "min_accuracy", "max_accuracy"],
        )
        w.writeheader()
        w.writerows(agg_rows)


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    variants = [x.strip() for x in args.variants.split(",") if x.strip()]
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]

    output_summaries: list[tuple[str, int, Path]] = []
    for v in variants:
        for s in seeds:
            summary_path = _run_one(args, variant=v, seed=s)
            output_summaries.append((v, s, summary_path))

    out_per_run = args.results_dir / f"{args.output_prefix}-per-run.csv"
    out_summary = args.results_dir / f"{args.output_prefix}-summary.csv"
    _aggregate(output_summaries, out_per_run=out_per_run, out_summary=out_summary)
    print(f"Wrote {out_per_run}")
    print(f"Wrote {out_summary}")


if __name__ == "__main__":
    main()
