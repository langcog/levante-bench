#!/usr/bin/env python3
"""Plot difficulty-calibrated child ability by age bin for LEVANTE tasks."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]

TASK_ORDER: tuple[str, ...] = (
    "egma-math",
    "matrix-reasoning",
    "mental-rotation",
    "theory-of-mind",
    "trog",
    "vocab",
)

TASK_LEGEND_LABELS: dict[str, str] = {
    "egma-math": "Egma math",
    "matrix-reasoning": "Matrix reasoning",
    "mental-rotation": "Mental rotation",
    "theory-of-mind": "Theory of mind",
    "trog": "TROG",
    "vocab": "Vocab",
}

TASK_COLORS: tuple[str, ...] = (
    "#dc2626",
    "#ea580c",
    "#ca8a04",
    "#16a34a",
    "#2563eb",
    "#7c3aed",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--version",
        default="v3",
        help="Data version under data/responses (default: v3).",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=REPO_ROOT,
        help="Project root path.",
    )
    parser.add_argument(
        "--min-age",
        type=float,
        default=5.0,
        help="Minimum child age to include.",
    )
    parser.add_argument(
        "--max-age",
        type=float,
        default=14.75,
        help="Maximum child age to include.",
    )
    parser.add_argument(
        "--bin-width",
        type=float,
        default=1.0,
        help="Age bin width in years.",
    )
    parser.add_argument(
        "--min-children",
        type=int,
        default=5,
        help="Minimum unique children per age/task group to plot.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "analysis",
        help="Output directory for charts and CSVs.",
    )
    parser.add_argument("--dpi", type=int, default=160, help="Figure DPI.")
    return parser.parse_args()


def infer_language(df: pd.DataFrame) -> pd.Series:
    for col in ("language", "prompt_language", "lang", "locale"):
        if col in df.columns:
            out = df[col].astype(str).str.strip().str.lower()
            return out.replace({"": "unknown", "nan": "unknown"})

    empty = pd.Series([""] * len(df), index=df.index, dtype="object")
    dataset = df["dataset"].astype(str).str.lower() if "dataset" in df.columns else empty
    site = df["site"].astype(str).str.lower() if "site" in df.columns else empty
    merged = (dataset + " " + site).str.strip()

    lang = pd.Series(["unknown"] * len(df), index=df.index, dtype="object")
    lang = lang.mask(merged.str.contains(r"\bde\b|german|mpieva", regex=True), "de")
    lang = lang.mask(merged.str.contains(r"\bco\b|spanish|uniandes", regex=True), "es")
    lang = lang.mask(merged.str.contains(r"\bca\b|western|english", regex=True), "en")
    return lang


def build_age_bins(
    df: pd.DataFrame, min_age: float, max_age: float, bin_width: float
) -> pd.DataFrame:
    if bin_width <= 0:
        raise ValueError("--bin-width must be > 0")
    edges = list(np.arange(float(min_age), float(max_age) + bin_width, bin_width))
    if len(edges) < 2:
        edges = [float(min_age), float(max_age) + float(bin_width)]

    labels: list[str] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mid = (lo + hi) / 2.0
        labels.append(str(int(mid)) if abs(mid - round(mid)) < 1e-9 else f"{mid:g}")

    out = df.copy()
    out["age_bin"] = pd.cut(
        out["age"],
        bins=edges,
        labels=labels,
        include_lowest=True,
        right=False,
    )
    return out


def load_child_ability(
    project_root: Path, version: str, min_age: float, max_age: float, bin_width: float
) -> pd.DataFrame:
    response_dir = project_root / "data" / "responses" / version
    trials_path = response_dir / "trials.csv"
    irt_dir = response_dir / "irt_models"
    if not trials_path.is_file():
        raise FileNotFoundError(f"Trials CSV not found: {trials_path}")

    trials = pd.read_csv(trials_path, low_memory=False)
    needed = {"task_id", "run_id", "age"}
    missing = needed - set(trials.columns)
    if missing:
        raise RuntimeError(f"Missing required columns in {trials_path}: {sorted(missing)}")

    trials = trials[trials["task_id"].isin(TASK_ORDER)].copy()
    trials["age"] = pd.to_numeric(trials["age"], errors="coerce")
    trials = trials.dropna(subset=["task_id", "run_id", "age"])
    trials = trials[(trials["age"] >= min_age) & (trials["age"] <= max_age)].copy()
    trials["language"] = infer_language(trials)
    if "site" not in trials.columns:
        trials["site"] = "unknown"
    if "dataset" not in trials.columns:
        trials["dataset"] = "unknown"

    child_meta = (
        trials.groupby(["task_id", "run_id"], as_index=False)
        .agg(
            age=("age", "mean"),
            language=("language", lambda s: s.mode().iat[0] if not s.mode().empty else "unknown"),
            site=("site", lambda s: s.mode().iat[0] if not s.mode().empty else "unknown"),
            dataset=("dataset", lambda s: s.mode().iat[0] if not s.mode().empty else "unknown"),
        )
        .copy()
    )

    frames: list[pd.DataFrame] = []
    for task_id in TASK_ORDER:
        ability_path = irt_dir / f"{task_id}_ability_scores.csv"
        if not ability_path.is_file():
            continue
        ability = pd.read_csv(ability_path, usecols=["run_id", "ability", "se"])
        ability = ability.dropna(subset=["run_id", "ability"]).copy()
        ability["task_id"] = task_id
        merged = child_meta[child_meta["task_id"] == task_id].merge(
            ability, on=["task_id", "run_id"], how="inner"
        )
        frames.append(merged)

    if not frames:
        raise RuntimeError(f"No task ability scores found under {irt_dir}")

    out = pd.concat(frames, ignore_index=True)
    out = build_age_bins(out, min_age=min_age, max_age=max_age, bin_width=bin_width)
    out = out[out["age_bin"].notna()].copy()
    out["age_bin"] = out["age_bin"].astype(str)
    return out


def aggregate_ability(child: pd.DataFrame, scope: str, min_children: int) -> pd.DataFrame:
    if scope == "english":
        data = child[child["language"] == "en"].copy()
        data["group"] = "en"
    elif scope == "all":
        data = child.copy()
        data["group"] = "all"
    elif scope == "language":
        data = child.copy()
        data["group"] = data["language"].astype(str)
    else:
        raise ValueError(f"Unknown scope: {scope}")

    if data.empty:
        return pd.DataFrame()

    agg = (
        data.groupby(["scope", "group", "age_bin", "task_id"], observed=True, as_index=False)
        .agg(
            n_children=("run_id", "nunique"),
            age_mean=("age", "mean"),
            ability_mean=("ability", "mean"),
            ability_median=("ability", "median"),
            ability_sd=("ability", "std"),
            ability_se_mean=("se", "mean"),
        )
        .copy()
    )
    agg["ability_sem"] = agg["ability_sd"] / np.sqrt(agg["n_children"].clip(lower=1))
    agg = agg[agg["n_children"] >= min_children].copy()
    return agg.sort_values(["scope", "group", "task_id", "age_mean"], kind="stable")


def plot_by_task_lines(
    agg: pd.DataFrame, out_path: Path, title: str, *, group: str | None, dpi: int
) -> None:
    import matplotlib.pyplot as plt

    data = agg.copy()
    if group is not None:
        data = data[data["group"] == group].copy()
    if data.empty:
        raise RuntimeError(f"No data available for plot: {title}")

    fig, ax = plt.subplots(figsize=(12, 6.2), layout="constrained")
    for i, task_id in enumerate(TASK_ORDER):
        task_rows = data[data["task_id"] == task_id].sort_values("age_mean")
        if task_rows.empty:
            continue
        color = TASK_COLORS[i % len(TASK_COLORS)]
        ax.errorbar(
            task_rows["age_mean"],
            task_rows["ability_mean"],
            yerr=task_rows["ability_sem"],
            color=color,
            marker="o",
            linewidth=2.0,
            markersize=5.5,
            capsize=2.5,
            label=TASK_LEGEND_LABELS[task_id],
            alpha=0.92,
        )

    ax.axhline(0, color="#64748b", linewidth=1.0, alpha=0.45)
    ax.set_title(title)
    ax.set_xlabel("Child age (years)")
    ax.set_ylabel("IRT ability (task-specific theta)")
    ax.grid(True, alpha=0.35, linestyle="--")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=9, framealpha=0.92)
    fig.text(
        0.02,
        0.02,
        "Ability is calibrated within each task, so compare age trends within task; "
        "vertical levels are not equated across tasks. Error bars show SEM.",
        fontsize=8,
        color="#475569",
        wrap=True,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_language_sensitivity(agg: pd.DataFrame, out_path: Path, dpi: int) -> None:
    import matplotlib.pyplot as plt

    data = agg[agg["scope"] == "language"].copy()
    if data.empty:
        raise RuntimeError("No language sensitivity data available.")

    groups = sorted(data["group"].dropna().unique())
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, layout="constrained")
    for ax, task_id in zip(axes.ravel(), TASK_ORDER):
        task_rows = data[data["task_id"] == task_id].copy()
        for group in groups:
            rows = task_rows[task_rows["group"] == group].sort_values("age_mean")
            if rows.empty:
                continue
            ax.plot(
                rows["age_mean"],
                rows["ability_mean"],
                marker="o",
                linewidth=1.8,
                markersize=4.5,
                label=group,
                alpha=0.9,
            )
        ax.axhline(0, color="#64748b", linewidth=0.8, alpha=0.35)
        ax.set_title(TASK_LEGEND_LABELS[task_id])
        ax.grid(True, alpha=0.25, linestyle="--")
    axes[1, 0].set_xlabel("Child age (years)")
    axes[1, 1].set_xlabel("Child age (years)")
    axes[1, 2].set_xlabel("Child age (years)")
    axes[0, 0].set_ylabel("IRT ability")
    axes[1, 0].set_ylabel("IRT ability")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=max(len(labels), 1), frameon=False)
    fig.suptitle("Child IRT Ability by Age and Language/Site Inference")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    output_dir = args.output_dir.resolve()

    child = load_child_ability(
        project_root=project_root,
        version=args.version,
        min_age=args.min_age,
        max_age=args.max_age,
        bin_width=args.bin_width,
    )
    child_path = output_dir / "human_ability_by_age_child_level.csv"
    child_path.parent.mkdir(parents=True, exist_ok=True)
    child.to_csv(child_path, index=False)

    frames = []
    for scope in ("english", "all", "language"):
        scoped = child.copy()
        scoped["scope"] = scope
        frames.append(aggregate_ability(scoped, scope=scope, min_children=args.min_children))
    agg = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    agg_path = output_dir / "human_ability_by_age_lines.csv"
    agg.to_csv(agg_path, index=False)

    plot_by_task_lines(
        agg[agg["scope"] == "english"],
        output_dir / "human_ability_by_age_lines_english.png",
        "LEVANTE Children - IRT Ability by Age (English-only)",
        group="en",
        dpi=args.dpi,
    )
    plot_by_task_lines(
        agg[agg["scope"] == "all"],
        output_dir / "human_ability_by_age_lines_all_sites.png",
        "LEVANTE Children - IRT Ability by Age (All sites)",
        group="all",
        dpi=args.dpi,
    )
    plot_language_sensitivity(
        agg[agg["scope"] == "language"],
        output_dir / "human_ability_by_age_language_sensitivity.png",
        dpi=args.dpi,
    )

    print(f"Wrote {child_path}")
    print(f"Wrote {agg_path}")
    print(f"Wrote {output_dir / 'human_ability_by_age_lines_english.png'}")
    print(f"Wrote {output_dir / 'human_ability_by_age_lines_all_sites.png'}")
    print(f"Wrote {output_dir / 'human_ability_by_age_language_sensitivity.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
