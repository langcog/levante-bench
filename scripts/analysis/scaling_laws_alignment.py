#!/usr/bin/env python3
"""Scaling Laws for Cognitive Alignment.

Analyses how three alignment metrics — accuracy, item-level correlation
(point-biserial r), and distribution-level KL divergence — scale with model
parameters across the LEVANTE-bench v1 models.

Produces:
  paper/figures/scaling_alignment.pdf   — 3-panel main figure
  paper/figures/scaling_crosstask.pdf   — supplementary cross-task figure
  results/analysis/scaling_laws_summary.csv — fitted parameters

All data is fetched from the public GCS bucket (no gcloud SDK needed).
"""

from __future__ import annotations

import argparse
import io
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from scipy import stats
from scipy.optimize import curve_fit

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BUCKET = "levante-bench"
BUCKET_URL = f"https://storage.googleapis.com/{BUCKET}"
GCS_API = f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o"

V1_MODELS: list[dict[str, Any]] = [
    {"bucket_name": "smolvlm2-256M",       "params_b": 0.256,  "family": "smolvlm2"},
    {"bucket_name": "smolvlm2-500M",       "params_b": 0.5,    "family": "smolvlm2"},
    {"bucket_name": "smolvlm2-2.2B",       "params_b": 2.2,    "family": "smolvlm2"},
    {"bucket_name": "qwen35-0.8B",         "params_b": 0.8,    "family": "qwen35"},
    {"bucket_name": "qwen35-2B",           "params_b": 2.0,    "family": "qwen35"},
    {"bucket_name": "qwen35-4B",           "params_b": 4.0,    "family": "qwen35"},
    {"bucket_name": "qwen35-9B",           "params_b": 9.0,    "family": "qwen35"},
    {"bucket_name": "qwen35-27B",          "params_b": 27.0,   "family": "qwen35"},
    {"bucket_name": "internvl35-1B",       "params_b": 1.0,    "family": "internvl35"},
    {"bucket_name": "internvl35-2B",       "params_b": 2.0,    "family": "internvl35"},
    {"bucket_name": "internvl35-4B",       "params_b": 4.0,    "family": "internvl35"},
    {"bucket_name": "internvl35-8B",       "params_b": 8.0,    "family": "internvl35"},
    {"bucket_name": "internvl35-14B",      "params_b": 14.0,   "family": "internvl35"},
    {"bucket_name": "internvl35-38B",      "params_b": 38.0,   "family": "internvl35"},
    {"bucket_name": "gemma4-E2B-it",       "params_b": 2.0,    "family": "gemma4"},
    {"bucket_name": "gemma4-E4B-it",       "params_b": 4.0,    "family": "gemma4"},
    {"bucket_name": "gemma4-26B-A4B-it",   "params_b": 26.0,   "family": "gemma4"},
    {"bucket_name": "tinyllava-3.1B",      "params_b": 3.1,    "family": "tinyllava"},
    {"bucket_name": "gpt53",               "params_b": 200.0,  "family": "gpt"},
    {"bucket_name": "gemini_pro",          "params_b": 200.0,  "family": "gemini"},
]

# Map bucket model name → KL CSV model identifier used in comparison/ filenames.
_KL_MODEL_MAP: dict[str, str] = {
    "smolvlm2-256M":       "smolvlm2-256M",
    "smolvlm2-500M":       "smolvlm2-500M",
    "smolvlm2-2.2B":       "smolvlm2-2_2B",
    "qwen35-0.8B":         "qwen35_0_8B",
    "qwen35-2B":           "qwen35_2B",
    "qwen35-4B":           "qwen35_4B",
    "internvl35-1B":       "internvl35_1B",
    "gemma4-E2B-it":       "gemma4-E2B-it",
    "gemma4-E4B-it":       "gemma4-E4B-it",
    "gpt53":               "gpt53",
    "gemini_pro":          "gemini_pro",
    "tinyllava-3.1B":      "tinyllava_3_1B",
}

# Closest-ability-bin file for age-equivalency analysis
CLOSEST_BIN_URL = f"{BUCKET_URL}/results/comparison/closest_ability_bin_by_model_task.csv"

TASKS = [
    "egma-math", "matrix-reasoning", "mental-rotation",
    "theory-of-mind", "trog", "vocab",
]
TASK_LABELS = {
    "egma-math": "Math",
    "matrix-reasoning": "Matrix",
    "mental-rotation": "Mental rot.",
    "theory-of-mind": "ToM",
    "trog": "TROG",
    "vocab": "Vocab",
}
TASK_DOMAIN = {
    "egma-math": "language",
    "trog": "language",
    "vocab": "language",
    "theory-of-mind": "language",
    "matrix-reasoning": "spatial",
    "mental-rotation": "spatial",
}

FAMILY_COLORS = {
    "smolvlm2":   "#4C72B0",
    "qwen35":     "#DD8452",
    "internvl35": "#55A868",
    "gemma4":     "#C44E52",
    "tinyllava":  "#8172B2",
    "gpt":        "#937860",
    "gemini":     "#DA8BC3",
}
FAMILY_MARKERS = {
    "smolvlm2":   "o",
    "qwen35":     "s",
    "internvl35": "D",
    "gemma4":     "^",
    "tinyllava":  "v",
    "gpt":        "*",
    "gemini":     "P",
}

# ---------------------------------------------------------------------------
# Data fetching helpers
# ---------------------------------------------------------------------------

def _fetch_csv(url: str) -> pd.DataFrame | None:
    """Fetch a CSV from a URL, return DataFrame or None on failure."""
    try:
        r = requests.get(url, timeout=15)
        if r.status_code == 200 and "task_id" in r.text[:200]:
            return pd.read_csv(io.StringIO(r.text))
        if r.status_code == 200 and len(r.text) > 10:
            return pd.read_csv(io.StringIO(r.text))
    except Exception:
        pass
    return None


def _fetch_text(url: str) -> str | None:
    try:
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            return r.text
    except Exception:
        pass
    return None


def fetch_accuracy_data() -> pd.DataFrame:
    """Download summary.csv for all v1 models and build a tidy DataFrame."""
    rows = []
    for m in V1_MODELS:
        url = f"{BUCKET_URL}/results/v1/{m['bucket_name']}/baseline/summary.csv"
        df = _fetch_csv(url)
        if df is not None:
            df["model"] = m["bucket_name"]
            df["params_b"] = m["params_b"]
            df["family"] = m["family"]
            rows.append(df)
        else:
            print(f"  [WARN] No summary for {m['bucket_name']}")
    return pd.concat(rows, ignore_index=True)


def fetch_kl_data() -> pd.DataFrame:
    """Download D_KL CSVs from the bucket comparison directory.

    Only fetches for models in our V1_MODELS list that have a KL name mapping.
    """
    rows = []
    for m in V1_MODELS:
        kl_name = _KL_MODEL_MAP.get(m["bucket_name"])
        if kl_name is None:
            continue
        for task in TASKS:
            url = f"{BUCKET_URL}/results/comparison/{task}_{kl_name}_d_kl.csv"
            df = _fetch_csv(url)
            if df is not None:
                df["params_b"] = m["params_b"]
                df["family"] = m["family"]
                df["bucket_name"] = m["bucket_name"]
                rows.append(df)
            else:
                print(f"  [WARN] No D_KL for {task}/{kl_name}")
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def fetch_item_data() -> pd.DataFrame:
    """Download per-item task CSVs for computing item-level correlations."""
    rows = []
    for m in V1_MODELS:
        for task in TASKS:
            url = f"{BUCKET_URL}/results/v1/{m['bucket_name']}/baseline/{task}.csv"
            df = _fetch_csv(url)
            if df is not None and "is_correct" in df.columns:
                df["model"] = m["bucket_name"]
                df["params_b"] = m["params_b"]
                df["family"] = m["family"]
                df["task"] = task
                rows.append(df[["model", "params_b", "family", "task",
                                "item_uid", "is_correct"]])
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def fetch_closest_ability_bin() -> pd.DataFrame:
    """Download age-equivalency data (closest ability bin per model × task)."""
    df = _fetch_csv(CLOSEST_BIN_URL)
    if df is not None:
        return df
    print("  [WARN] Could not fetch closest_ability_bin_by_model_task.csv")
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Curve fitting
# ---------------------------------------------------------------------------

def log_linear(x: np.ndarray, a: float, b: float) -> np.ndarray:
    return a * np.log10(x) + b


def sigmoid(x: np.ndarray, L: float, k: float, x0: float, b: float) -> np.ndarray:
    return L / (1.0 + np.exp(-k * (np.log10(x) - x0))) + b


def _r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def _aic(n: int, k: int, rss: float) -> float:
    if rss <= 0 or n <= k:
        return np.inf
    return n * np.log(rss / n) + 2 * k


def fit_scaling(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """Fit log-linear and sigmoid, return best model info."""
    result: dict[str, Any] = {"n": len(x)}

    # Log-linear fit
    try:
        popt_ll, _ = curve_fit(log_linear, x, y)
        y_pred_ll = log_linear(x, *popt_ll)
        r2_ll = _r_squared(y, y_pred_ll)
        rss_ll = np.sum((y - y_pred_ll) ** 2)
        aic_ll = _aic(len(x), 2, rss_ll)
        result["loglinear"] = {
            "params": {"a": popt_ll[0], "b": popt_ll[1]},
            "r2": r2_ll, "aic": aic_ll,
        }
    except Exception:
        result["loglinear"] = None

    # Sigmoid fit
    try:
        p0 = [np.max(y) - np.min(y), 2.0, np.median(np.log10(x)), np.min(y)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt_sig, _ = curve_fit(
                sigmoid, x, y, p0=p0, maxfev=10000,
                bounds=([0, 0.01, -3, -1], [2, 20, 4, 1]),
            )
        y_pred_sig = sigmoid(x, *popt_sig)
        r2_sig = _r_squared(y, y_pred_sig)
        rss_sig = np.sum((y - y_pred_sig) ** 2)
        aic_sig = _aic(len(x), 4, rss_sig)
        result["sigmoid"] = {
            "params": {"L": popt_sig[0], "k": popt_sig[1],
                       "x0": popt_sig[2], "b": popt_sig[3]},
            "r2": r2_sig, "aic": aic_sig,
        }
    except Exception:
        result["sigmoid"] = None

    # Pick best
    ll = result.get("loglinear")
    sig = result.get("sigmoid")
    if ll and sig:
        result["best"] = "sigmoid" if sig["aic"] < ll["aic"] else "loglinear"
    elif ll:
        result["best"] = "loglinear"
    elif sig:
        result["best"] = "sigmoid"
    else:
        result["best"] = None

    return result


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def compute_item_correlations(
    item_df: pd.DataFrame, kl_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute item-level correlation between model accuracy and human D_KL.

    Since IRT item parameters aren't on the bucket, this uses mean D_KL across
    ability bins as an item-level divergence proxy: higher D_KL means the model
    diverges more from human response distributions. This is not an item
    difficulty sign convention.
    """
    if item_df.empty or kl_df.empty:
        return pd.DataFrame()

    item_df = item_df.copy()
    item_df["is_correct"] = item_df["is_correct"].map(
        {True: 1, False: 0, "True": 1, "False": 0}
    ).fillna(0).astype(int)

    model_item_acc = item_df.groupby(["model", "task", "params_b", "family", "item_uid"]).agg(
        item_acc=("is_correct", "mean"),
        n_trials=("is_correct", "count"),
    ).reset_index()

    item_difficulty = kl_df.groupby(["task", "item_uid"]).agg(
        mean_item_dkl=("D_KL", "mean"),
    ).reset_index()

    kl_model_map_inv = {v: k for k, v in _KL_MODEL_MAP.items()}

    results = []
    for (model, task, params_b, family), grp in model_item_acc.groupby(
        ["model", "task", "params_b", "family"]
    ):
        diff = item_difficulty[item_difficulty["task"] == task]
        if diff.empty:
            continue
        merged = grp.merge(diff, on="item_uid", how="inner", suffixes=("", "_kl"))
        if len(merged) < 10:
            continue
        try:
            if merged["item_acc"].std() < 1e-10 or merged["mean_item_dkl"].std() < 1e-10:
                continue
            r_val, p_val = stats.pearsonr(
                merged["item_acc"].values, merged["mean_item_dkl"].values
            )
            results.append({
                "model": model, "task": task, "params_b": params_b,
                "family": family, "r_pb": r_val, "p_value": p_val,
                "n_items": len(merged),
            })
        except Exception:
            continue

    return pd.DataFrame(results)


def compute_mean_kl(kl_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate D_KL per model × task (mean over items and ability bins)."""
    if kl_df.empty:
        return pd.DataFrame()
    agg = kl_df.groupby(["bucket_name", "task", "params_b", "family"]).agg(
        mean_dkl=("D_KL", "mean"),
        median_dkl=("D_KL", "median"),
        n_pairs=("D_KL", "count"),
    ).reset_index()
    agg.rename(columns={"bucket_name": "model"}, inplace=True)
    return agg


def compute_cross_task_alignment(
    acc_df: pd.DataFrame, human_task_acc: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Spearman rho between model and human task-accuracy vectors."""
    if human_task_acc is None:
        human_task_acc = {
            "egma-math": 0.72, "matrix-reasoning": 0.55,
            "mental-rotation": 0.65, "theory-of-mind": 0.60,
            "trog": 0.68, "vocab": 0.70,
        }

    human_vec = np.array([human_task_acc[t] for t in TASKS])
    results = []
    for model, grp in acc_df.groupby("model"):
        model_acc = {}
        for _, row in grp.iterrows():
            model_acc[row["task_id"]] = row["accuracy"]
        if len(model_acc) < len(TASKS):
            continue
        model_vec = np.array([model_acc.get(t, np.nan) for t in TASKS])
        if np.any(np.isnan(model_vec)):
            continue
        rho, p = stats.spearmanr(model_vec, human_vec)
        m_info = next((m for m in V1_MODELS if m["bucket_name"] == model), None)
        if m_info:
            results.append({
                "model": model, "params_b": m_info["params_b"],
                "family": m_info["family"], "spearman_rho": rho, "p_value": p,
            })
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_scatter(
    ax: plt.Axes, df: pd.DataFrame, x_col: str, y_col: str,
    fit_result: dict | None = None, ylabel: str = "", title: str = "",
    invert_y: bool = False,
) -> None:
    for family, color in FAMILY_COLORS.items():
        mask = df["family"] == family
        sub = df[mask]
        if sub.empty:
            continue
        marker = FAMILY_MARKERS.get(family, "o")
        ax.scatter(
            sub[x_col], sub[y_col],
            c=color, marker=marker, s=50, alpha=0.8,
            label=family, edgecolors="white", linewidths=0.3,
            zorder=3,
        )

    if fit_result and fit_result.get("best"):
        x_range = np.logspace(
            np.log10(df[x_col].min() * 0.8),
            np.log10(df[x_col].max() * 1.2),
            200,
        )
        best = fit_result["best"]
        info = fit_result[best]
        if best == "loglinear":
            y_pred = log_linear(x_range, **info["params"])
        else:
            y_pred = sigmoid(x_range, **info["params"])
        ax.plot(x_range, y_pred, "k--", alpha=0.5, linewidth=1.2, zorder=2)
        r2 = info["r2"]
        ax.text(
            0.03, 0.95 if not invert_y else 0.05,
            f"{best}\n$R^2$={r2:.3f}",
            transform=ax.transAxes, fontsize=7,
            va="top" if not invert_y else "bottom",
            ha="left", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
        )

    ax.set_xscale("log")
    ax.set_xlabel("Parameters (B)", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    if invert_y:
        ax.invert_yaxis()
    ax.grid(True, alpha=0.2)


def plot_main_figure(
    acc_df: pd.DataFrame,
    corr_df: pd.DataFrame,
    kl_agg: pd.DataFrame,
    fit_results: dict,
    output_path: Path,
) -> None:
    """3-panel figure: accuracy, item r_pb, and D_KL vs log(params)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Panel A: Accuracy (mean across tasks per model)
    if not acc_df.empty:
        agg_acc = acc_df.groupby(["model", "params_b", "family"]).agg(
            mean_acc=("accuracy", "mean")
        ).reset_index()
        _plot_scatter(
            axes[0], agg_acc, "params_b", "mean_acc",
            fit_result=fit_results.get("accuracy"),
            ylabel="Mean accuracy", title="A. Accuracy scaling",
        )
        axes[0].set_ylim(0, 1.05)

    # Panel B: Item-level model-human divergence correlation (mean r_pb across tasks per model)
    if not corr_df.empty:
        agg_corr = corr_df.groupby(["model", "params_b", "family"]).agg(
            mean_rpb=("r_pb", "mean")
        ).reset_index()
        _plot_scatter(
            axes[1], agg_corr, "params_b", "mean_rpb",
            fit_result=fit_results.get("item_corr"),
            ylabel="Mean $r_{pb}$ (item divergence)", title="B. Item alignment scaling",
        )

    # Panel C: KL divergence
    if not kl_agg.empty:
        agg_kl = kl_agg.groupby(["model", "params_b", "family"]).agg(
            mean_dkl=("mean_dkl", "mean")
        ).reset_index()
        _plot_scatter(
            axes[2], agg_kl, "params_b", "mean_dkl",
            fit_result=fit_results.get("kl"),
            ylabel="Mean $D_{KL}$", title="C. Distribution alignment scaling",
            invert_y=True,
        )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=len(FAMILY_COLORS),
        fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.02),
    )
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved {output_path}")
    plt.close(fig)


def plot_accuracy_by_task(
    acc_df: pd.DataFrame,
    fit_results_by_task: dict,
    output_path: Path,
) -> None:
    """Per-task accuracy vs params, with fitted curves."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes_flat = axes.flatten()

    for i, task in enumerate(TASKS):
        ax = axes_flat[i]
        sub = acc_df[acc_df["task_id"] == task].copy()
        if sub.empty:
            continue

        for family, color in FAMILY_COLORS.items():
            mask = sub["family"] == family
            s = sub[mask]
            if s.empty:
                continue
            marker = FAMILY_MARKERS.get(family, "o")
            ax.scatter(
                s["params_b"], s["accuracy"],
                c=color, marker=marker, s=50, alpha=0.8,
                label=family, edgecolors="white", linewidths=0.3, zorder=3,
            )

        fit_r = fit_results_by_task.get(task)
        if fit_r and fit_r.get("best"):
            x_range = np.logspace(
                np.log10(sub["params_b"].min() * 0.8),
                np.log10(sub["params_b"].max() * 1.2), 200,
            )
            best = fit_r["best"]
            info = fit_r[best]
            if best == "loglinear":
                y_pred = log_linear(x_range, **info["params"])
            else:
                y_pred = sigmoid(x_range, **info["params"])
            ax.plot(x_range, y_pred, "k--", alpha=0.5, linewidth=1.2, zorder=2)

            slope_txt = ""
            if best == "loglinear":
                slope_txt = f"slope={info['params']['a']:.3f}"
            r2 = info["r2"]
            ax.text(
                0.03, 0.95,
                f"{best} $R^2$={r2:.3f}\n{slope_txt}".strip(),
                transform=ax.transAxes, fontsize=7, va="top", ha="left",
                fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
            )

        domain = TASK_DOMAIN[task]
        ax.set_xscale("log")
        ax.set_title(
            f"{TASK_LABELS[task]} ({domain})",
            fontsize=10, fontweight="bold",
        )
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Parameters (B)", fontsize=8)
        ax.set_ylabel("Accuracy", fontsize=8)
        ax.grid(True, alpha=0.2)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=len(FAMILY_COLORS),
        fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.01),
    )
    fig.suptitle("Accuracy scaling by task", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved {output_path}")
    plt.close(fig)


def plot_cross_task(ct_df: pd.DataFrame, output_path: Path) -> None:
    """Cross-task alignment: Spearman rho vs log(params)."""
    if ct_df.empty:
        print("  [SKIP] No cross-task data to plot.")
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    for family, color in FAMILY_COLORS.items():
        mask = ct_df["family"] == family
        sub = ct_df[mask]
        if sub.empty:
            continue
        marker = FAMILY_MARKERS.get(family, "o")
        ax.scatter(
            sub["params_b"], sub["spearman_rho"],
            c=color, marker=marker, s=60, alpha=0.8,
            label=family, edgecolors="white", linewidths=0.3, zorder=3,
        )

    if len(ct_df) >= 3:
        try:
            popt, _ = curve_fit(log_linear, ct_df["params_b"].values,
                                ct_df["spearman_rho"].values)
            x_r = np.logspace(
                np.log10(ct_df["params_b"].min() * 0.8),
                np.log10(ct_df["params_b"].max() * 1.2), 200,
            )
            ax.plot(x_r, log_linear(x_r, *popt), "k--", alpha=0.5, linewidth=1.2)
            y_pred = log_linear(ct_df["params_b"].values, *popt)
            r2 = _r_squared(ct_df["spearman_rho"].values, y_pred)
            ax.text(
                0.03, 0.05,
                f"slope={popt[0]:.3f}, $R^2$={r2:.3f}",
                transform=ax.transAxes, fontsize=8, va="bottom",
                fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
            )
        except Exception:
            pass

    ax.axhline(0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("Parameters (B)", fontsize=10)
    ax.set_ylabel("Spearman $\\rho$ (model vs human task accuracies)", fontsize=10)
    ax.set_title("Cross-task alignment vs model scale", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved {output_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).resolve().parent.parent.parent,
        help="Project root (default: auto-detected).",
    )
    args = parser.parse_args()
    root = args.output_dir
    fig_dir = root / "paper" / "figures"
    analysis_dir = root / "results" / "analysis"
    fig_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # 1. Fetch accuracy data
    print("Fetching accuracy data...")
    acc_df = fetch_accuracy_data()
    print(f"  {len(acc_df)} rows ({acc_df['model'].nunique()} models)")

    # 2. Fetch KL data
    print("Fetching KL divergence data...")
    kl_df = fetch_kl_data()
    kl_agg = compute_mean_kl(kl_df)
    print(f"  {len(kl_df)} raw KL rows → {len(kl_agg)} model×task cells")

    # 3. Fetch item-level data
    print("Fetching item-level data...")
    item_df = fetch_item_data()
    print(f"  {len(item_df)} item rows")

    # 3b. Fetch closest ability bin data
    print("Fetching closest ability bin data...")
    cab_df = fetch_closest_ability_bin()
    print(f"  {len(cab_df)} rows")

    # 4. Compute item-level correlations (using D_KL as divergence proxy)
    print("Computing item-level correlations...")
    corr_df = compute_item_correlations(item_df, kl_df)
    print(f"  {len(corr_df)} model×task correlation cells")

    # 5. Compute cross-task alignment
    print("Computing cross-task alignment...")
    ct_df = compute_cross_task_alignment(acc_df)
    print(f"  {len(ct_df)} models with cross-task rho")

    # 6. Fit scaling curves
    print("Fitting scaling curves...")
    fit_results: dict[str, Any] = {}
    summary_rows = []

    # Overall accuracy
    if not acc_df.empty:
        agg = acc_df.groupby(["model", "params_b"]).agg(
            mean_acc=("accuracy", "mean")).reset_index()
        fit_r = fit_scaling(agg["params_b"].values, agg["mean_acc"].values)
        fit_results["accuracy"] = fit_r
        for model_type in ["loglinear", "sigmoid"]:
            if fit_r.get(model_type):
                summary_rows.append({
                    "metric": "accuracy_mean", "task": "all",
                    "fit_type": model_type,
                    "r2": fit_r[model_type]["r2"],
                    "aic": fit_r[model_type]["aic"],
                    "is_best": model_type == fit_r["best"],
                    **{f"param_{k}": v for k, v in fit_r[model_type]["params"].items()},
                })

    # Per-task accuracy fits
    fit_by_task: dict[str, dict] = {}
    for task in TASKS:
        sub = acc_df[acc_df["task_id"] == task]
        if len(sub) < 5:
            continue
        fit_r = fit_scaling(sub["params_b"].values, sub["accuracy"].values)
        fit_by_task[task] = fit_r
        for model_type in ["loglinear", "sigmoid"]:
            if fit_r.get(model_type):
                summary_rows.append({
                    "metric": "accuracy", "task": task,
                    "fit_type": model_type,
                    "r2": fit_r[model_type]["r2"],
                    "aic": fit_r[model_type]["aic"],
                    "is_best": model_type == fit_r["best"],
                    **{f"param_{k}": v for k, v in fit_r[model_type]["params"].items()},
                })

    # Item correlation scaling
    if not corr_df.empty:
        agg = corr_df.groupby(["model", "params_b"]).agg(
            mean_rpb=("r_pb", "mean")).reset_index()
        if len(agg) >= 3:
            fit_r = fit_scaling(agg["params_b"].values, agg["mean_rpb"].values)
            fit_results["item_corr"] = fit_r

    # KL scaling
    if not kl_agg.empty:
        agg = kl_agg.groupby(["model", "params_b"]).agg(
            mean_dkl=("mean_dkl", "mean")).reset_index()
        if len(agg) >= 3:
            fit_r = fit_scaling(agg["params_b"].values, agg["mean_dkl"].values)
            fit_results["kl"] = fit_r

    # 7. Generate figures
    print("Generating figures...")
    plot_main_figure(acc_df, corr_df, kl_agg, fit_results,
                     fig_dir / "scaling_alignment.pdf")
    plot_accuracy_by_task(acc_df, fit_by_task,
                          fig_dir / "scaling_accuracy_by_task.pdf")
    plot_cross_task(ct_df, fig_dir / "scaling_crosstask.pdf")

    # 8. Save summary CSV
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        out_path = analysis_dir / "scaling_laws_summary.csv"
        summary_df.to_csv(out_path, index=False)
        print(f"Saved {out_path}")

    # 9. Print key results
    print("\n" + "=" * 70)
    print("KEY RESULTS")
    print("=" * 70)
    if fit_by_task:
        print("\nPer-task accuracy scaling (log-linear slopes):")
        for task in TASKS:
            ft = fit_by_task.get(task, {})
            ll = ft.get("loglinear")
            if ll:
                domain = TASK_DOMAIN[task]
                print(f"  {TASK_LABELS[task]:12s} ({domain:8s}): "
                      f"slope={ll['params']['a']:+.4f}  R²={ll['r2']:.3f}")

    if not ct_df.empty:
        print(f"\nCross-task alignment: mean rho = {ct_df['spearman_rho'].mean():.3f}")
        rho_corr, rho_p = stats.spearmanr(
            ct_df["params_b"].values, ct_df["spearman_rho"].values)
        print(f"  Correlation with log(params): rho={rho_corr:.3f}, p={rho_p:.3f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
