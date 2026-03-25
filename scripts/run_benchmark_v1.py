#!/usr/bin/env python3
"""Run benchmark v1 and write a timestamped result bundle."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run benchmark v1 (math + ToM).")
    p.add_argument("--data-version", default="2026-03-24", help="Data/assets version folder name")
    p.add_argument("--model-id", default="HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
    p.add_argument("--device", default="cuda", choices=["auto", "cpu", "cuda"])
    p.add_argument("--tom-seeds", default="1,2,3,4,5", help="Comma-separated seeds for ToM robustness")
    p.add_argument("--tom-variants", default="none,state_model", help="Comma-separated ToM variants")
    p.add_argument(
        "--tom-template-style",
        default="standard",
        choices=["standard", "trial_aware"],
        help="Template style for ToM evaluator inside robustness runs.",
    )
    p.add_argument("--tom-two-stage", action="store_true", help="Enable two-stage ToM evaluation.")
    p.add_argument("--tom-analysis-max-new-tokens", type=int, default=64)
    p.add_argument("--max-items-math", type=int, default=None, help="Optional cap for math items")
    p.add_argument("--max-items-tom", type=int, default=None, help="Optional cap for ToM items")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional explicit output directory. Default: results/benchmark/v1/<timestamp>",
    )
    return p.parse_args()


def run_cmd(cmd: list[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=str(cwd), check=True)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _get_git_commit(repo_root: Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            text=True,
        ).strip()
        return out
    except Exception:
        return None


def _chance_from_options(options: list[str]) -> float:
    n = len(options)
    return 1.0 / n if n > 0 else 0.0


def _build_tom_item_metadata(corpus_path: Path) -> dict[str, dict[str, object]]:
    rows = []
    with open(corpus_path, newline="", encoding="utf-8") as f:
        rd = list(csv.DictReader(f))
    for block in sorted({(r.get("block_index") or "").strip() for r in rd}):
        block_rows = [r for r in rd if (r.get("block_index") or "").strip() == block]
        last_instr_idx = None
        q_idx = 0
        for i, r in enumerate(block_rows):
            stage = (r.get("assessment_stage") or "").strip().lower()
            if stage == "instructions":
                last_instr_idx = i
                continue
            if stage != "test_response":
                continue
            uid = (r.get("item_uid") or "").strip()
            ans = (r.get("answer") or "").strip()
            if not uid or not ans:
                continue
            tt = (r.get("trial_type") or "").strip()
            alts = [x.strip() for x in (r.get("response_alternatives") or "").split(",") if x.strip()]
            opts = []
            for o in [ans] + alts:
                if o not in opts:
                    opts.append(o)
            if len(opts) < 2:
                continue
            dist = None if last_instr_idx is None else (i - last_instr_idx)
            rows.append(
                {
                    "item_uid": uid,
                    "trial_type": tt,
                    "block_index": block,
                    "q_idx_in_block": q_idx,
                    "distance_from_last_instruction": dist,
                    "n_options": len(opts),
                    "chance": 1.0 / len(opts),
                }
            )
            q_idx += 1
    return {r["item_uid"]: r for r in rows}


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_root = args.output_dir or (repo_root / "results" / "benchmark" / "v1" / timestamp)
    out_math = out_root / "math"
    out_tom = out_root / "tom"
    out_math.mkdir(parents=True, exist_ok=True)
    out_tom.mkdir(parents=True, exist_ok=True)

    data_version = args.data_version
    math_corpus = repo_root / "data" / "assets" / data_version / "corpus" / "egma-math" / "test-combined-math-cat.csv"
    tom_corpus = repo_root / "data" / "assets" / data_version / "corpus" / "theory-of-mind" / "theory-of-mind-item-bank.csv"

    # ---- Math pipeline ----
    math_prompts = out_math / "egma-math-prompts.jsonl"
    math_prompts_eval = out_math / "egma-math-prompts-eval.jsonl"
    math_preds = out_math / "egma-math-preds.jsonl"
    math_summary = out_math / "egma-math-summary.json"
    math_by_type_csv = out_math / "egma-math-by-type.csv"
    math_by_type_png = out_math / "egma-math-by-type.png"

    cmd = [
        sys.executable,
        "scripts/build_math_prompts.py",
        "--corpus-csv",
        str(math_corpus),
        "--output",
        str(math_prompts),
    ]
    run_cmd(cmd, repo_root)

    # If max-items-math is set, create a truncated prompts file to keep row counts aligned.
    if args.max_items_math is not None:
        prompt_lines = [l for l in math_prompts.read_text(encoding="utf-8").splitlines() if l.strip()]
        prompt_lines = prompt_lines[: args.max_items_math]
        math_prompts_eval.write_text("\n".join(prompt_lines) + "\n", encoding="utf-8")
    else:
        math_prompts_eval = math_prompts

    cmd = [
        sys.executable,
        "scripts/run_smolvlmv2_math_eval.py",
        "--input-jsonl",
        str(math_prompts_eval),
        "--output-jsonl",
        str(math_preds),
        "--summary-json",
        str(math_summary),
        "--model-id",
        args.model_id,
        "--device",
        args.device,
    ]
    run_cmd(cmd, repo_root)

    cmd = [
        sys.executable,
        "scripts/analyze_math_type_accuracy.py",
        "--prompts-jsonl",
        str(math_prompts_eval),
        "--preds-jsonl",
        str(math_preds),
        "--output-csv",
        str(math_by_type_csv),
        "--output-png",
        str(math_by_type_png),
    ]
    run_cmd(cmd, repo_root)

    # math chance baseline (overall)
    math_prompt_rows = [json.loads(l) for l in math_prompts_eval.read_text().splitlines() if l.strip()]
    math_chance = statistics.mean(_chance_from_options(r.get("options") or []) for r in math_prompt_rows)
    math_sum = read_json(math_summary)
    math_acc = float(math_sum["accuracy_all"])

    # ---- ToM robustness pipeline ----
    tom_prefix = "tom-v1"
    cmd = [
        sys.executable,
        "scripts/run_tom_robustness.py",
        "--corpus-csv",
        str(tom_corpus),
        "--variants",
        args.tom_variants,
        "--seeds",
        args.tom_seeds,
        "--device",
        args.device,
        "--model-id",
        args.model_id,
        "--template-style",
        args.tom_template_style,
        "--output-prefix",
        tom_prefix,
        "--results-dir",
        str(out_tom),
    ]
    if args.tom_two_stage:
        cmd.extend(["--two-stage", "--analysis-max-new-tokens", str(args.tom_analysis_max_new_tokens)])
    if args.max_items_tom is not None:
        cmd.extend(["--max-items", str(args.max_items_tom)])
    run_cmd(cmd, repo_root)

    tom_summary_csv = out_tom / f"{tom_prefix}-summary.csv"
    tom_per_run_csv = out_tom / f"{tom_prefix}-per-run.csv"

    # Compute ToM chance baseline from evaluated uid set in first variant/seed output.
    first_variant = args.tom_variants.split(",")[0].strip()
    first_seed = args.tom_seeds.split(",")[0].strip()
    first_preds = out_tom / f"{tom_prefix}-{first_variant}-s{first_seed}-preds.jsonl"
    uid_eval = [json.loads(l).get("item_uid") for l in first_preds.read_text().splitlines() if l.strip()]
    uid_eval = [u for u in uid_eval if u]

    uid_to_chance: dict[str, float] = {}
    with open(tom_corpus, newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            uid = (r.get("item_uid") or "").strip()
            if not uid:
                continue
            if (r.get("assessment_stage") or "").strip().lower() != "test_response":
                continue
            ans = (r.get("answer") or "").strip()
            if not ans:
                continue
            alts = [x.strip() for x in (r.get("response_alternatives") or "").split(",") if x.strip()]
            opts = []
            for o in [ans] + alts:
                if o not in opts:
                    opts.append(o)
            if len(opts) >= 2:
                uid_to_chance[uid] = 1.0 / len(opts)
    tom_chance = statistics.mean(uid_to_chance[u] for u in uid_eval if u in uid_to_chance)

    # Memory-critical subset metadata and chance baseline.
    uid_meta = _build_tom_item_metadata(tom_corpus)
    memcrit_uids = [
        uid
        for uid, m in uid_meta.items()
        if m["trial_type"] == "false_belief_question"
        and m["distance_from_last_instruction"] is not None
        and int(m["distance_from_last_instruction"]) >= 2
    ]
    second_order_uids = [
        uid
        for uid, m in uid_meta.items()
        if m["trial_type"] == "false_belief_question" and str(uid).startswith("tom_second_order")
    ]
    memcrit_chance = statistics.mean(uid_meta[u]["chance"] for u in memcrit_uids) if memcrit_uids else None
    second_order_chance = statistics.mean(uid_meta[u]["chance"] for u in second_order_uids) if second_order_uids else None

    # Add chance/lift table for ToM summary variants.
    tom_vs_chance_csv = out_tom / f"{tom_prefix}-summary-vs-chance.csv"
    with open(tom_summary_csv, newline="", encoding="utf-8") as f_in, open(
        tom_vs_chance_csv, "w", newline="", encoding="utf-8"
    ) as f_out:
        rd = csv.DictReader(f_in)
        fieldnames = list(rd.fieldnames or []) + ["chance_baseline", "lift_vs_chance"]
        wr = csv.DictWriter(f_out, fieldnames=fieldnames)
        wr.writeheader()
        for r in rd:
            mean_acc = float(r["mean_accuracy"])
            r["chance_baseline"] = tom_chance
            r["lift_vs_chance"] = mean_acc - tom_chance
            wr.writerow(r)

    # Memory-critical summary across per-run ToM outputs.
    memcrit_summary_csv = out_tom / f"{tom_prefix}-memory-critical-summary.csv"
    mem_rows: list[dict[str, object]] = []
    for variant in [v.strip() for v in args.tom_variants.split(",") if v.strip()]:
        seed_vals: list[float] = []
        seed_vals_second: list[float] = []
        for seed in [int(s.strip()) for s in args.tom_seeds.split(",") if s.strip()]:
            preds_path = out_tom / f"{tom_prefix}-{variant}-s{seed}-preds.jsonl"
            preds = [json.loads(l) for l in preds_path.read_text(encoding="utf-8").splitlines() if l.strip()]
            by_uid = {p.get("item_uid"): p for p in preds if p.get("item_uid")}
            vals = [1.0 if by_uid[u].get("correct") else 0.0 for u in memcrit_uids if u in by_uid]
            vals2 = [1.0 if by_uid[u].get("correct") else 0.0 for u in second_order_uids if u in by_uid]
            if vals:
                seed_vals.append(sum(vals) / len(vals))
            if vals2:
                seed_vals_second.append(sum(vals2) / len(vals2))
        if seed_vals:
            mem_rows.append(
                {
                    "variant": variant,
                    "subset": "memory_critical_false_belief_distance_ge_2",
                    "n_items": len(memcrit_uids),
                    "mean_accuracy": statistics.mean(seed_vals),
                    "std_accuracy": statistics.pstdev(seed_vals),
                    "chance_baseline": memcrit_chance,
                    "lift_vs_chance": (statistics.mean(seed_vals) - memcrit_chance) if memcrit_chance is not None else None,
                }
            )
        if seed_vals_second:
            mem_rows.append(
                {
                    "variant": variant,
                    "subset": "second_order_false_belief",
                    "n_items": len(second_order_uids),
                    "mean_accuracy": statistics.mean(seed_vals_second),
                    "std_accuracy": statistics.pstdev(seed_vals_second),
                    "chance_baseline": second_order_chance,
                    "lift_vs_chance": (statistics.mean(seed_vals_second) - second_order_chance)
                    if second_order_chance is not None
                    else None,
                }
            )
    with open(memcrit_summary_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "variant",
            "subset",
            "n_items",
            "mean_accuracy",
            "std_accuracy",
            "chance_baseline",
            "lift_vs_chance",
        ]
        wr = csv.DictWriter(f, fieldnames=fieldnames)
        wr.writeheader()
        wr.writerows(mem_rows)

    # ---- Top-level metadata + summary ----
    metadata = {
        "benchmark_version": "v1",
        "timestamp": timestamp,
        "data_version": data_version,
        "model_id": args.model_id,
        "device": args.device,
        "tom_variants": [v.strip() for v in args.tom_variants.split(",") if v.strip()],
        "tom_seeds": [int(s.strip()) for s in args.tom_seeds.split(",") if s.strip()],
        "tom_template_style": args.tom_template_style,
        "tom_two_stage": args.tom_two_stage,
        "tom_analysis_max_new_tokens": args.tom_analysis_max_new_tokens,
        "max_items_math": args.max_items_math,
        "max_items_tom": args.max_items_tom,
        "git_commit": _get_git_commit(repo_root),
        "paths": {
            "math_summary_json": str(math_summary.relative_to(repo_root)),
            "math_by_type_csv": str(math_by_type_csv.relative_to(repo_root)),
            "tom_summary_csv": str(tom_summary_csv.relative_to(repo_root)),
            "tom_per_run_csv": str(tom_per_run_csv.relative_to(repo_root)),
            "tom_summary_vs_chance_csv": str(tom_vs_chance_csv.relative_to(repo_root)),
            "tom_memory_critical_summary_csv": str(memcrit_summary_csv.relative_to(repo_root)),
        },
    }
    (out_root / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # benchmark_summary.csv
    bench_summary = out_root / "benchmark_summary.csv"
    with open(bench_summary, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["math_accuracy_all", math_acc])
        w.writerow(["math_chance_baseline", math_chance])
        w.writerow(["math_lift_vs_chance", math_acc - math_chance])
        # Append ToM variant means/stds and chance lift.
        with open(tom_summary_csv, newline="", encoding="utf-8") as tf:
            rd = csv.DictReader(tf)
            for r in rd:
                variant = r["variant"]
                mean_acc = float(r["mean_accuracy"])
                std_acc = float(r["std_accuracy"])
                w.writerow([f"tom_{variant}_mean_accuracy", mean_acc])
                w.writerow([f"tom_{variant}_std_accuracy", std_acc])
                w.writerow([f"tom_{variant}_chance_baseline", tom_chance])
                w.writerow([f"tom_{variant}_lift_vs_chance", mean_acc - tom_chance])
        # Append memory-critical metrics.
        with open(memcrit_summary_csv, newline="", encoding="utf-8") as mf:
            rd = csv.DictReader(mf)
            for r in rd:
                variant = r["variant"]
                subset = r["subset"]
                mean_acc = float(r["mean_accuracy"])
                std_acc = float(r["std_accuracy"])
                chance = float(r["chance_baseline"])
                lift = float(r["lift_vs_chance"])
                key = f"tom_{variant}_{subset}"
                w.writerow([f"{key}_mean_accuracy", mean_acc])
                w.writerow([f"{key}_std_accuracy", std_acc])
                w.writerow([f"{key}_chance_baseline", chance])
                w.writerow([f"{key}_lift_vs_chance", lift])

    print(f"Benchmark bundle written to: {out_root}")


if __name__ == "__main__":
    main()
