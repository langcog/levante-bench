#!/usr/bin/env python3
"""Compare child age trends with historic VLM milestone trends by task."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

ALL_SITES_CAPTION = (
    "Caption: Left panel shows historic VLM performance by task. For each curated model "
    "checkpoint, the x value is the proxy milestone year from the historic-frontier model "
    "specification and the y value is that model's task accuracy read from summary.csv. "
    "Right panel shows children's calibrated task performance across all available sites. "
    "For each task, trial metadata from data/responses/v1/trials.csv is joined to the task's "
    "IRT person scores in data/responses/v1/irt_models/*_ability_scores.csv by run_id; "
    "children are binned by age, and each point is the mean task-specific IRT ability "
    "(theta) for children in that age bin. Error bars show standard error of the mean "
    "ability. The two y-axes are intentionally different: VLMs use raw accuracy, while "
    "children use IRT ability to normalize for differences in item difficulty."
)

from plot_historic_frontier_models import (  # noqa: E402
    DEFAULT_MODEL_SPECS,
    TASK_COLORS,
    TASK_LEGEND_LABELS,
    TASK_ORDER,
    collect_series,
)
from plot_human_ability_by_age_lines import build_age_bins, infer_language  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=REPO_ROOT,
        help="Project root path.",
    )
    parser.add_argument(
        "--version",
        default="v1",
        help="Data/results version.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=REPO_ROOT / "results" / "v1",
        help="Directory containing historic model summary.csv files.",
    )
    parser.add_argument(
        "--human-ability-csv",
        type=Path,
        default=REPO_ROOT / "results" / "analysis" / "human_ability_by_age_lines.csv",
        help="CSV from plot_human_ability_by_age_lines.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "analysis",
        help="Directory for comparison charts and CSVs.",
    )
    parser.add_argument("--min-age", type=float, default=5.0)
    parser.add_argument("--max-age", type=float, default=14.75)
    parser.add_argument("--bin-width", type=float, default=1.0)
    parser.add_argument("--min-trials", type=int, default=20)
    parser.add_argument("--dpi", type=int, default=160)
    return parser.parse_args()


def to_bool(series: pd.Series) -> pd.Series:
    normalized = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .replace({"true": "1", "false": "0"})
    )
    numeric = pd.to_numeric(normalized, errors="coerce")
    return numeric == 1


def load_vlm_task_series(results_root: Path) -> tuple[pd.DataFrame, list[str]]:
    rows, missing = collect_series(results_root, DEFAULT_MODEL_SPECS)
    out_rows: list[dict] = []
    for year, slug, label, acc in rows:
        for task_id in TASK_ORDER:
            out_rows.append(
                {
                    "model": slug,
                    "label": label,
                    "proxy_year": year,
                    "task_id": task_id,
                    "accuracy": acc[task_id],
                }
            )
    return pd.DataFrame(out_rows), missing


def load_best_available_vlm_series(
    requested_root: Path, project_root: Path
) -> tuple[pd.DataFrame, list[str], Path]:
    vlm, missing = load_vlm_task_series(requested_root)
    n_models = int(vlm["model"].nunique()) if not vlm.empty else 0
    fallback_root = project_root / "results" / "v1_additional_models"
    if n_models >= 2 or requested_root == fallback_root or not fallback_root.is_dir():
        return vlm, missing, requested_root

    fallback_vlm, fallback_missing = load_vlm_task_series(fallback_root)
    fallback_n_models = int(fallback_vlm["model"].nunique()) if not fallback_vlm.empty else 0
    if fallback_n_models > n_models:
        return fallback_vlm, fallback_missing, fallback_root
    return vlm, missing, requested_root


def aggregate_human_accuracy(
    project_root: Path,
    version: str,
    *,
    min_age: float,
    max_age: float,
    bin_width: float,
    min_trials: int,
) -> pd.DataFrame:
    trials_path = project_root / "data" / "responses" / version / "trials.csv"
    trials = pd.read_csv(trials_path, low_memory=False)
    needed = {"task_id", "run_id", "age", "correct"}
    missing = needed - set(trials.columns)
    if missing:
        raise RuntimeError(f"Missing required columns in {trials_path}: {sorted(missing)}")

    trials = trials[trials["task_id"].isin(TASK_ORDER)].copy()
    trials["age"] = pd.to_numeric(trials["age"], errors="coerce")
    trials = trials.dropna(subset=["task_id", "run_id", "age", "correct"])
    trials = trials[(trials["age"] >= min_age) & (trials["age"] <= max_age)].copy()
    trials["language"] = infer_language(trials)
    trials["is_correct"] = to_bool(trials["correct"])
    trials = build_age_bins(trials, min_age=min_age, max_age=max_age, bin_width=bin_width)
    trials = trials[trials["age_bin"].notna()].copy()
    trials["age_bin"] = trials["age_bin"].astype(str)

    frames: list[pd.DataFrame] = []
    for scope, data in (
        ("english", trials[trials["language"] == "en"].copy()),
        ("all", trials.copy()),
    ):
        if data.empty:
            continue
        data["scope"] = scope
        data["group"] = "en" if scope == "english" else "all"
        agg = (
            data.groupby(["scope", "group", "age_bin", "task_id"], observed=True, as_index=False)
            .agg(
                n_trials=("is_correct", "size"),
                n_children=("run_id", "nunique"),
                age_mean=("age", "mean"),
                accuracy=("is_correct", "mean"),
            )
            .copy()
        )
        agg = agg[agg["n_trials"] >= min_trials].copy()
        frames.append(agg)
    return pd.concat(frames, ignore_index=True).sort_values(
        ["scope", "task_id", "age_mean"], kind="stable"
    )


def fit_slope(df: pd.DataFrame, x_col: str, y_col: str) -> float | None:
    data = df[[x_col, y_col]].dropna().copy()
    if len(data) < 2:
        return None
    x = data[x_col].to_numpy(dtype=float)
    y = data[y_col].to_numpy(dtype=float)
    if np.nanmax(x) - np.nanmin(x) <= 1e-9:
        return None
    return float(np.polyfit(x, y, deg=1)[0])


def build_learning_rate_summary(
    vlm: pd.DataFrame, ability: pd.DataFrame, accuracy: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict] = []
    ability_en = ability[(ability["scope"] == "english") & (ability["group"] == "en")].copy()
    ability_all = ability[(ability["scope"] == "all") & (ability["group"] == "all")].copy()
    accuracy_en = accuracy[(accuracy["scope"] == "english") & (accuracy["group"] == "en")].copy()

    for task_id in TASK_ORDER:
        child_ability_en = ability_en[ability_en["task_id"] == task_id]
        child_ability_all = ability_all[ability_all["task_id"] == task_id]
        child_ability = child_ability_en if not child_ability_en.empty else child_ability_all
        child_ability_scope = "english" if not child_ability_en.empty else "all"
        child_acc = accuracy_en[accuracy_en["task_id"] == task_id]
        task_vlm = vlm[vlm["task_id"] == task_id]
        rows.append(
            {
                "task_id": task_id,
                "child_ability_scope": child_ability_scope if not child_ability.empty else None,
                "child_ability_slope_per_year": fit_slope(
                    child_ability, "age_mean", "ability_mean"
                ),
                "child_ability_english_slope_per_year": fit_slope(
                    child_ability_en, "age_mean", "ability_mean"
                ),
                "child_ability_all_sites_slope_per_year": fit_slope(
                    child_ability_all, "age_mean", "ability_mean"
                ),
                "child_accuracy_slope_per_year": fit_slope(child_acc, "age_mean", "accuracy"),
                "vlm_accuracy_slope_per_proxy_year": fit_slope(
                    task_vlm, "proxy_year", "accuracy"
                ),
                "child_ability_first": child_ability.sort_values("age_mean")[
                    "ability_mean"
                ].iloc[0]
                if not child_ability.empty
                else None,
                "child_ability_last": child_ability.sort_values("age_mean")[
                    "ability_mean"
                ].iloc[-1]
                if not child_ability.empty
                else None,
                "vlm_accuracy_first": task_vlm.sort_values("proxy_year")["accuracy"].iloc[0]
                if not task_vlm.empty
                else None,
                "vlm_accuracy_last": task_vlm.sort_values("proxy_year")["accuracy"].iloc[-1]
                if not task_vlm.empty
                else None,
                "n_child_age_bins": int(len(child_ability)),
                "n_child_english_age_bins": int(len(child_ability_en)),
                "n_child_all_sites_age_bins": int(len(child_ability_all)),
                "n_vlm_milestones": int(len(task_vlm)),
            }
        )

    out = pd.DataFrame(rows)
    for col in (
        "child_ability_slope_per_year",
        "child_accuracy_slope_per_year",
        "vlm_accuracy_slope_per_proxy_year",
    ):
        out[f"{col}_rank_desc"] = out[col].rank(ascending=False, method="min")
    return out


def plot_paired_comparison(
    vlm: pd.DataFrame,
    ability: pd.DataFrame,
    out_path: Path,
    *,
    scope: str,
    group: str,
    dpi: int,
    caption: str | None = None,
) -> None:
    import textwrap

    import matplotlib.pyplot as plt

    child = ability[(ability["scope"] == scope) & (ability["group"] == group)].copy()
    if child.empty:
        raise RuntimeError(f"No child ability data for {scope}/{group}")

    fig_height = 7.8 if caption else 6.2
    layout = None if caption else "constrained"
    fig, axes = plt.subplots(1, 2, figsize=(15, fig_height), layout=layout)

    for i, task_id in enumerate(TASK_ORDER):
        color = TASK_COLORS[i % len(TASK_COLORS)]
        vlm_rows = vlm[vlm["task_id"] == task_id].sort_values("proxy_year")
        child_rows = child[child["task_id"] == task_id].sort_values("age_mean")

        axes[0].plot(
            vlm_rows["proxy_year"],
            vlm_rows["accuracy"],
            marker="o",
            linewidth=2.0,
            markersize=5.5,
            color=color,
            label=TASK_LEGEND_LABELS[task_id],
            alpha=0.92,
        )
        axes[1].errorbar(
            child_rows["age_mean"],
            child_rows["ability_mean"],
            yerr=child_rows["ability_sem"],
            marker="o",
            linewidth=2.0,
            markersize=5.5,
            capsize=2.5,
            color=color,
            label=TASK_LEGEND_LABELS[task_id],
            alpha=0.92,
        )

    axes[0].set_title("Historic VLMs by task")
    axes[0].set_xlabel("Proxy milestone (fractional year)")
    axes[0].set_ylabel("Task accuracy")
    axes[0].set_ylim(0.0, 1.02)
    axes[0].yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")

    axes[1].set_title("Children by task")
    axes[1].set_xlabel("Child age (years)")
    axes[1].set_ylabel("IRT ability (task-specific theta)")
    axes[1].axhline(0, color="#64748b", linewidth=1.0, alpha=0.45)

    for ax in axes:
        ax.grid(True, alpha=0.35, linestyle="--")
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=9, framealpha=0.92)
    fig.suptitle("LEVANTE Task Trends: VLM Milestones vs Child Development")
    if caption:
        fig.subplots_adjust(left=0.06, right=0.83, top=0.88, bottom=0.25, wspace=0.12)
    note = caption or (
        "Left and right y-axes are intentionally different: VLMs use accuracy, "
        "children use IRT ability to normalize item difficulty. Compare task ordering "
        "and trend shape, not absolute y-values."
    )
    if caption:
        note = "\n".join(textwrap.wrap(note, width=185))
    fig.text(
        0.02,
        0.04 if caption else 0.02,
        note,
        fontsize=8 if caption else 8,
        color="#475569",
        ha="left",
        va="bottom",
        wrap=not bool(caption),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_child_accuracy_companion(
    accuracy: pd.DataFrame, out_path: Path, *, scope: str, group: str, dpi: int
) -> None:
    import matplotlib.pyplot as plt

    data = accuracy[(accuracy["scope"] == scope) & (accuracy["group"] == group)].copy()
    if data.empty:
        raise RuntimeError(f"No human accuracy data for {scope}/{group}")

    fig, ax = plt.subplots(figsize=(12, 6.2), layout="constrained")
    for i, task_id in enumerate(TASK_ORDER):
        rows = data[data["task_id"] == task_id].sort_values("age_mean")
        if rows.empty:
            continue
        ax.plot(
            rows["age_mean"],
            rows["accuracy"],
            marker="o",
            linewidth=2.0,
            markersize=5.5,
            color=TASK_COLORS[i % len(TASK_COLORS)],
            label=TASK_LEGEND_LABELS[task_id],
            alpha=0.92,
        )
    ax.set_title("Children - Raw Accuracy by Age (companion metric)")
    ax.set_xlabel("Child age (years)")
    ax.set_ylabel("Raw trial accuracy")
    ax.set_ylim(0.0, 1.02)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.grid(True, alpha=0.35, linestyle="--")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=9, framealpha=0.92)
    fig.text(
        0.02,
        0.02,
        "Raw accuracy is included as a bridge to VLM accuracy, but it is less calibrated "
        "than IRT ability because item difficulty varies by child/task.",
        fontsize=8,
        color="#475569",
        wrap=True,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def write_research_notes(summary: pd.DataFrame, out_path: Path) -> None:
    lines = [
        "# Children vs Historic VLM Learning-Rate Notes",
        "",
        "These rankings are a first-pass descriptive scaffold for research, not a causal model.",
        "Child slopes use English-only IRT ability by age when available, with all-sites ability as a documented fallback for sparse tasks; VLM slopes use curated proxy years from the historic frontier chart.",
        "",
        "## Fastest child ability slopes",
        "",
    ]
    child_ranked = summary.sort_values(
        "child_ability_slope_per_year", ascending=False, na_position="last"
    )
    for _, row in child_ranked.iterrows():
        lines.append(
            f"- {row['task_id']}: {row['child_ability_slope_per_year']:.4f} theta/year"
            if pd.notna(row["child_ability_slope_per_year"])
            else f"- {row['task_id']}: insufficient child ability bins"
        )
    lines.extend(["", "## Fastest VLM accuracy slopes", ""])
    vlm_ranked = summary.sort_values(
        "vlm_accuracy_slope_per_proxy_year", ascending=False, na_position="last"
    )
    for _, row in vlm_ranked.iterrows():
        lines.append(
            f"- {row['task_id']}: {row['vlm_accuracy_slope_per_proxy_year']:.4f} accuracy/proxy-year"
            if pd.notna(row["vlm_accuracy_slope_per_proxy_year"])
            else f"- {row['task_id']}: insufficient VLM milestones"
        )
    lines.extend(
        [
            "",
            "## Research Caveats",
            "",
            "- Child ability is calibrated within task, not equated across tasks.",
            "- Raw child accuracy remains useful for visual intuition but is confounded by adaptive/easier item assignment.",
            "- VLM proxy years are curated ordering metadata, not training exposure or compute.",
            "- Ceiling effects can flatten both accuracy and ability trends, especially on easier tasks.",
        ]
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    output_dir = args.output_dir.resolve()

    vlm, missing, vlm_root = load_best_available_vlm_series(
        args.results_root.resolve(), project_root=project_root
    )
    if vlm.empty:
        raise RuntimeError("No historic VLM task series found.")
    if missing:
        print(f"Skipped historic VLMs without summary.csv: {', '.join(missing)}")
    print(f"Using historic VLM results root: {vlm_root}")

    ability = pd.read_csv(args.human_ability_csv)
    accuracy = aggregate_human_accuracy(
        project_root=project_root,
        version=args.version,
        min_age=args.min_age,
        max_age=args.max_age,
        bin_width=args.bin_width,
        min_trials=args.min_trials,
    )
    summary = build_learning_rate_summary(vlm=vlm, ability=ability, accuracy=accuracy)

    output_dir.mkdir(parents=True, exist_ok=True)
    vlm_path = output_dir / "historic_vlm_task_series.csv"
    accuracy_path = output_dir / "human_accuracy_by_age_task_lines.csv"
    summary_path = output_dir / "child_vlm_learning_rate_summary.csv"
    notes_path = output_dir / "child_vlm_learning_rate_notes.md"

    vlm.to_csv(vlm_path, index=False)
    accuracy.to_csv(accuracy_path, index=False)
    summary.to_csv(summary_path, index=False)
    write_research_notes(summary, notes_path)

    plot_paired_comparison(
        vlm,
        ability,
        output_dir / "child_vlm_task_trends_english.png",
        scope="english",
        group="en",
        dpi=args.dpi,
    )
    plot_paired_comparison(
        vlm,
        ability,
        output_dir / "child_vlm_task_trends_all_sites.png",
        scope="all",
        group="all",
        dpi=args.dpi,
    )
    plot_paired_comparison(
        vlm,
        ability,
        output_dir / "child_vlm_task_trends_all_sites_captioned.png",
        scope="all",
        group="all",
        dpi=args.dpi,
        caption=ALL_SITES_CAPTION,
    )
    caption_path = output_dir / "child_vlm_task_trends_all_sites_caption.txt"
    caption_path.write_text(ALL_SITES_CAPTION + "\n", encoding="utf-8")
    plot_child_accuracy_companion(
        accuracy,
        output_dir / "human_accuracy_by_age_task_lines_english.png",
        scope="english",
        group="en",
        dpi=args.dpi,
    )
    plot_child_accuracy_companion(
        accuracy,
        output_dir / "human_accuracy_by_age_task_lines_all_sites.png",
        scope="all",
        group="all",
        dpi=args.dpi,
    )

    print(f"Wrote {vlm_path}")
    print(f"Wrote {accuracy_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {notes_path}")
    print(f"Wrote {output_dir / 'child_vlm_task_trends_english.png'}")
    print(f"Wrote {output_dir / 'child_vlm_task_trends_all_sites.png'}")
    print(f"Wrote {output_dir / 'child_vlm_task_trends_all_sites_captioned.png'}")
    print(f"Wrote {caption_path}")
    print(f"Wrote {output_dir / 'human_accuracy_by_age_task_lines_english.png'}")
    print(f"Wrote {output_dir / 'human_accuracy_by_age_task_lines_all_sites.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
