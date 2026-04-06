"""Math response distribution — multi trial-type run.

Picks one representative item per trial type, runs:
  Axis A: K stochastic samples (fixed prompt, fixed option order)
  Axis B: S greedy passes varying option order (shuffle seeds)

Usage:
    python scripts/run_math_dist_prototype.py [--k 30] [--shuffle-seeds 12] \
        [--trial-types "Missing Number,Addition,Fraction,Subtraction,Number Comparison"] \
        [--models qwen35] [--output-dir results/math-dist]
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Best-prompt builder: Phase 1 multiline + Phase 5 CoT on arithmetic types
# (from experiment_egma_math_phases.py — Phase 5 best at 57 %, Phase 1 at 50.8 %)
# ---------------------------------------------------------------------------
_MC_LABELS = ["A", "B", "C", "D", "E", "F", "G", "H"]
_ARITHMETIC_TYPES = {"Addition", "Subtraction", "Multiplication", "Fraction"}


def _build_best_prompt(row: dict, options: list[str]) -> str:
    trial_type = str(row.get("trial_type") or "").strip()
    item = str(row.get("item") or "").strip()
    prompt_phrase = str(row.get("prompt_phrase") or "").strip()
    if prompt_phrase in ("NA", "nan", ""):
        prompt_phrase = item
    type_lower = trial_type.lower()
    if trial_type == "Number Identification":
        stem = f"Which number is {prompt_phrase}?"
    elif trial_type == "Number Comparison":
        stem = f"Which number is larger: {item}?"
    elif trial_type == "Missing Number":
        stem = f"What number completes the sequence: {item}?"
    elif trial_type in _ARITHMETIC_TYPES:
        stem = f"What is {prompt_phrase}?"
    elif "counting" in type_lower:
        stem = f"How many items are shown: {item}?"
    elif "non-symbolic" in type_lower and "comparison" in type_lower:
        stem = f"Which group has more: {item}?"
    elif "non-symbolic" in type_lower:
        stem = f"How many items: {item}?"
    else:
        stem = str(row.get("prompt") or "").strip() or f"Solve: {item}"
    lines = [stem, ""]
    for i, opt in enumerate(options):
        lines.append(f"{_MC_LABELS[i]}) {opt}")
    lines.append("")
    lines.append("Answer with one letter.")
    prompt = "\n".join(lines)
    if trial_type in _ARITHMETIC_TYPES:
        cot = "Think step by step to compute the answer, then state your final answer as a single letter."
        prompt = prompt.rstrip() + f"\n\n{cot}"
    return prompt


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
_DEFAULT_TRIAL_TYPES = "Missing Number,Addition,Fraction,Subtraction,Number Comparison"

def parse_args():
    p = argparse.ArgumentParser(description="Math response distribution — multi trial-type")
    p.add_argument("--k", type=int, default=30, help="Stochastic samples per item (Axis A)")
    p.add_argument("--temp", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--shuffle-seeds", type=int, default=12, help="Greedy passes varying option order (Axis B)")
    p.add_argument("--trial-types", default=_DEFAULT_TRIAL_TYPES,
                   help="Comma-separated trial types to include")
    p.add_argument("--models", nargs="+", default=["qwen35"])
    p.add_argument("--output-dir", default=str(PROJECT_ROOT / "results" / "math-dist"))
    p.add_argument("--data-root", default=str(PROJECT_ROOT / "data"))
    p.add_argument("--device", default="auto")
    return p.parse_args()


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def load_model(model_id: str, device: str):
    from omegaconf import OmegaConf
    from levante_bench.config.loader import load_model_config
    from levante_bench.models import get_model_class
    cfg_dc = load_model_config(model_id)
    if cfg_dc is None:
        raise ValueError(f"No config for '{model_id}'")
    cfg = OmegaConf.to_container(cfg_dc, resolve=True)
    reg_name = cfg.get("name", model_id)
    cls = get_model_class(reg_name)
    if cls is None:
        raise ValueError(f"Model class not registered: '{reg_name}'")
    kwargs = dict(model_name=cfg["hf_name"], device=device)
    if "dtype" in cfg:
        kwargs["dtype"] = cfg["dtype"]
    if "attn_implementation" in cfg:
        kwargs["attn_implementation"] = cfg["attn_implementation"]
    m = cls(**kwargs)
    print(f"  Loading {model_id} ({cfg['hf_name']}) …", flush=True)
    m.load()
    return m, cfg


def free_model(model, device: str) -> None:
    try:
        import torch
        del model.model
        if hasattr(model, "processor"):
            del model.processor
        if device == "mps":
            torch.mps.empty_cache()
        elif device == "cuda":
            torch.cuda.empty_cache()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def bar_chart(hist: Counter, letter_order: list, title: str, out_path: Path,
              gold_label: str | None = None) -> None:
    keys = [k for k in letter_order if hist.get(k, 0) > 0]
    extras = sorted(k for k in hist if k not in letter_order and hist[k] > 0)
    keys += extras
    vals = [hist[k] for k in keys]
    def _color(k):
        if k == "_unparsed":
            return "#C44E52"
        if k == gold_label:
            return "#55A868"
        return "#4C72B0"
    colors = [_color(k) for k in keys]
    fig, ax = plt.subplots(figsize=(6, 3))
    bars = ax.bar(keys, vals, color=colors)
    ax.set_ylabel("Count", fontsize=9)
    ax.set_xlabel("Parsed label", fontsize=9)
    ax.set_title(title, fontsize=9)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02 * max(vals or [1]), str(v), ha="center", fontsize=8)
    if gold_label:
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(color="#55A868", label=f"Gold ({gold_label})")], fontsize=8)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def summary_grid(all_results: list[dict], out_path: Path, k: int, temp: float) -> None:
    """One panel per (model, trial_type): stochastic histogram + robustness scatter."""
    models = sorted(set(r["model"] for r in all_results))
    trial_types = sorted(set(r["trial_type"] for r in all_results),
                         key=lambda tt: [r["trial_type"] for r in all_results].index(tt))
    ncols = len(trial_types)
    nrows = len(models) * 2  # row 0: stochastic hist, row 1: robustness scatter
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(3.5 * ncols, 3 * nrows),
        squeeze=False,
    )
    for col, tt in enumerate(trial_types):
        for mrow, mid in enumerate(models):
            subset = [r for r in all_results if r["model"] == mid and r["trial_type"] == tt]
            if not subset:
                continue
            ref = subset[0]
            letter_order = ref["option_labels"] + ["_unparsed"]
            gold = ref["gold_label"]

            # ── Axis A: stochastic histogram ────────────────────────────────
            ax_hist = axes[mrow * 2][col]
            stoch = [r for r in subset if r["axis"] == "A"]
            hist: Counter = Counter()
            for r in stoch:
                hist[r["label"]] += r["count"]
            total = sum(hist.values())
            keys = [k for k in letter_order if hist.get(k, 0) > 0]
            keys += sorted(k for k in hist if k not in letter_order and hist[k] > 0)
            vals = [hist[k] for k in keys]
            colors = ["#55A868" if k == gold else "#C44E52" if k == "_unparsed" else "#4C72B0"
                      for k in keys]
            ax_hist.bar(keys, vals, color=colors)
            for i, v in enumerate(vals):
                ax_hist.text(i, v + 0.02 * max(vals or [1]), str(v), ha="center", fontsize=7)
            gold_pct = hist.get(gold, 0) / total if total else 0
            gold_entropy = -sum((c / total) * math.log2(c / total) for c in hist.values() if c > 0)
            ax_hist.set_title(
                f"{mid} | {tt}\ngold={gold} ({gold_pct:.0%})  H={gold_entropy:.2f}b",
                fontsize=8,
            )
            ax_hist.set_ylabel(f"K={k} samples", fontsize=7)

            # ── Axis B: robustness scatter ───────────────────────────────────
            ax_rob = axes[mrow * 2 + 1][col]
            rob = [r for r in subset if r["axis"] == "B"]
            if rob:
                xs = [r["shuffle_seed"] for r in rob]
                ys = [r["correct"] for r in rob]
                acc = sum(ys) / len(ys)
                ax_rob.scatter(xs, ys, alpha=0.7, s=40)
                ax_rob.axhline(acc, color="orange", linestyle="--", linewidth=1)
                ax_rob.set_title(f"Robustness acc={acc:.0%}", fontsize=8)
                ax_rob.set_ylabel("Correct (0/1)", fontsize=7)
                ax_rob.set_yticks([0, 1])
                ax_rob.set_xlabel("shuffle_seed", fontsize=7)

    plt.suptitle(
        f"EGMA-math response distributions — Qwen3.5-0.8B — best prompt\n"
        f"Axis A: K={k} stochastic (T={temp}) | Axis B: greedy shuffle seeds",
        fontsize=10, y=1.01,
    )
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSummary grid saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    target_types = [t.strip() for t in args.trial_types.split(",") if t.strip()]

    print(f"\n{'='*62}")
    print(f"Math Response Distribution — Multi trial-type")
    print(f"  device       : {device}")
    print(f"  model(s)     : {args.models}")
    print(f"  trial types  : {target_types}")
    print(f"  K samples    : {args.k}  (T={args.temp}, top_p={args.top_p})")
    print(f"  shuffle seeds: {args.shuffle_seeds}")
    print(f"  output dir   : {out_dir}")
    print(f"{'='*62}\n")

    from levante_bench.config.defaults import detect_data_version
    from levante_bench.evaluation.stochastic import (
        collect_parsed_labels,
        greedy_predicted_label,
        label_histogram,
    )
    from levante_bench.tasks.egma_math import (
        egma_math_corpus_path,
        egma_math_trial_from_row,
        iter_egma_math_corpus_rows,
        math_shuffle_rng,
    )

    data_root = Path(args.data_root)
    version = detect_data_version(data_root)
    corpus_path = egma_math_corpus_path(data_root, version)
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus not found: {corpus_path}")
    print(f"Corpus : {corpus_path}")
    print(f"Version: {version}\n")

    # ── Select one representative item per trial type ──────────────────────
    selected: dict[str, dict] = {}  # trial_type -> raw corpus row
    for row in iter_egma_math_corpus_rows(corpus_path):
        tt = str(row.get("trial_type") or "").strip()
        if tt not in target_types or tt in selected:
            continue
        rng = math_shuffle_rng(row, 0)
        trial = egma_math_trial_from_row(row, rng)
        if trial and len(trial.get("option_labels", [])) >= 4:
            selected[tt] = row
        if len(selected) == len(target_types):
            break

    missing = [tt for tt in target_types if tt not in selected]
    if missing:
        print(f"WARNING: no eligible item found for: {missing}")

    print("Items selected:")
    for tt, row in selected.items():
        uid = row.get("item_uid", "?")
        trial0 = egma_math_trial_from_row(row, math_shuffle_rng(row, 0))
        gold = trial0["correct_label"] if trial0 else "?"
        opts = dict(zip(trial0["option_labels"], trial0["options"])) if trial0 else {}
        print(f"  {tt:30}  {uid}  gold={gold}  {opts}")
    print()

    shuffle_seeds = list(range(args.shuffle_seeds))
    all_results: list[dict] = []
    axis_a_csv = out_dir / "axis_a_stochastic.csv"
    axis_b_csv = out_dir / "axis_b_robustness.csv"

    # ── Load model once and run all trial types ────────────────────────────
    for mid in args.models:
        print(f"\n{'─'*62}")
        print(f"Model: {mid}")
        print(f"{'─'*62}")
        model, cfg = load_model(mid, device)
        mtok = int(cfg.get("max_new_tokens", 128))

        for tt in target_types:
            row = selected.get(tt)
            if row is None:
                continue
            trial_ref = egma_math_trial_from_row(row, math_shuffle_rng(row, 0))
            assert trial_ref is not None
            trial_ref["prompt"] = _build_best_prompt(row, trial_ref["options"])
            trial_ref["max_new_tokens"] = mtok
            item_uid = trial_ref["item_uid"]
            gold = trial_ref["correct_label"]
            option_labels = list(trial_ref["option_labels"])

            print(f"\n  ── {tt} ({item_uid})  gold={gold} ──")
            print(f"  Prompt:\n{trial_ref['prompt']}")

            # ── Axis A: stochastic ─────────────────────────────────────────
            labels = collect_parsed_labels(
                model,
                trial_ref,
                n_samples=args.k,
                do_sample=True,
                temperature=args.temp,
                top_p=args.top_p,
                base_seed=0,
                max_new_tokens=mtok,
            )
            hist = label_histogram(labels)
            total = sum(hist.values())
            gold_pct = hist.get(gold, 0) / total if total else 0
            entropy = -sum((c / total) * math.log2(c / total) for c in hist.values() if c > 0)
            letter_order = option_labels + ["_unparsed"]
            print(f"  A) {dict((k, hist[k]) for k in letter_order if hist.get(k,0)>0)}"
                  f"  gold={gold_pct:.0%}  H={entropy:.2f}b")

            for label, count in sorted(hist.items()):
                all_results.append({
                    "axis": "A",
                    "model": mid,
                    "trial_type": tt,
                    "item_uid": item_uid,
                    "gold_label": gold,
                    "option_labels": option_labels,
                    "label": label,
                    "count": count,
                    "proportion": count / total if total else 0,
                    "is_gold": label == gold,
                    "shuffle_seed": None,
                    "correct": None,
                    "chosen_option": None,
                })

            bar_chart(
                hist, letter_order,
                f"{mid} | {tt} — stochastic K={args.k} T={args.temp}\n"
                f"item={item_uid}  gold={gold}  H={entropy:.2f}b",
                out_dir / f"axis_a_{mid}_{tt.replace(' ','_')}.png",
                gold_label=gold,
            )

            # ── Axis B: robustness ─────────────────────────────────────────
            correct_count = 0
            for s in shuffle_seeds:
                trial_s = egma_math_trial_from_row(row, math_shuffle_rng(row, s))
                if trial_s is None:
                    continue
                trial_s["prompt"] = _build_best_prompt(row, trial_s["options"])
                trial_s["max_new_tokens"] = mtok
                pred = greedy_predicted_label(model, trial_s, max_new_tokens=mtok)
                gold_s = trial_s["correct_label"]
                correct = pred == gold_s
                if correct:
                    correct_count += 1
                chosen_text = ""
                if pred:
                    for i, lb in enumerate(trial_s["option_labels"]):
                        if str(lb).upper() == str(pred).upper():
                            chosen_text = str(trial_s["options"][i])[:60]
                            break
                all_results.append({
                    "axis": "B",
                    "model": mid,
                    "trial_type": tt,
                    "item_uid": item_uid,
                    "gold_label": gold_s,
                    "option_labels": option_labels,
                    "label": pred,
                    "count": 1,
                    "proportion": float(correct),
                    "is_gold": correct,
                    "shuffle_seed": s,
                    "correct": int(correct),
                    "chosen_option": chosen_text,
                })
            rob_acc = correct_count / len(shuffle_seeds)
            print(f"  B) robustness {rob_acc:.0%} ({correct_count}/{len(shuffle_seeds)} seeds)")

        free_model(model, device)

    # ── Write CSVs ─────────────────────────────────────────────────────────
    axis_a_rows = [r for r in all_results if r["axis"] == "A"]
    axis_b_rows = [r for r in all_results if r["axis"] == "B"]

    if axis_a_rows:
        with open(axis_a_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(axis_a_rows[0].keys()))
            writer.writeheader()
            writer.writerows(axis_a_rows)
        print(f"\nAxis A CSV: {axis_a_csv}")

    if axis_b_rows:
        with open(axis_b_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(axis_b_rows[0].keys()))
            writer.writeheader()
            writer.writerows(axis_b_rows)
        print(f"Axis B CSV: {axis_b_csv}")

    # ── Summary grid ───────────────────────────────────────────────────────
    summary_grid(all_results, out_dir / "summary_grid.png", k=args.k, temp=args.temp)

    # ── Console summary ────────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print("FINAL SUMMARY")
    print(f"{'='*62}")
    for mid in args.models:
        print(f"\nModel: {mid}")
        print(f"  {'Trial Type':30}  {'Gold%':6}  {'Entropy':8}  {'Rob%':6}")
        print(f"  {'─'*30}  {'─'*6}  {'─'*8}  {'─'*6}")
        for tt in target_types:
            a_rows = [r for r in all_results if r["axis"] == "A" and r["model"] == mid and r["trial_type"] == tt]
            b_rows = [r for r in all_results if r["axis"] == "B" and r["model"] == mid and r["trial_type"] == tt]
            if not a_rows:
                continue
            total_a = sum(r["count"] for r in a_rows)
            gold_lbl = a_rows[0]["gold_label"]
            gold_n = sum(r["count"] for r in a_rows if r["is_gold"])
            gold_pct = gold_n / total_a if total_a else 0
            entropy = -sum((r["count"] / total_a) * math.log2(r["count"] / total_a)
                           for r in a_rows if r["count"] > 0)
            rob_acc = sum(r["correct"] for r in b_rows) / len(b_rows) if b_rows else float("nan")
            print(f"  {tt:30}  {gold_pct:6.0%}  {entropy:8.2f}b  {rob_acc:6.0%}")

    print(f"\nOutputs: {out_dir}")


if __name__ == "__main__":
    main()
