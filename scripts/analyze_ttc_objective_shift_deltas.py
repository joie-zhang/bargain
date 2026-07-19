#!/usr/bin/env python3
"""Matched TTC objective-shift deltas within family/game/order cells."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCORED_RUNS = (
    PROJECT_ROOT
    / "analysis/neurips_revision_20260504/ttc_objective_shift/ttc_objective_shift_scored_runs.csv"
)
OUT_DIR = PROJECT_ROOT / "analysis/neurips_revision_20260504/ttc_objective_shift"
GRAPHICS_DIR = PROJECT_ROOT / "overleaf/neurips/graphics"

FAMILY_LABELS = {
    "gpt-5": "GPT-5",
    "claude-sonnet-4-6": "Claude Sonnet 4.6",
    "gemini-3-flash": "Gemini 3 Flash",
}
FAMILY_MARKERS = {
    "gpt-5": "o",
    "claude-sonnet-4-6": "s",
    "gemini-3-flash": "^",
}
FAMILY_COLORS = {
    "gpt-5": "#2563eb",
    "claude-sonnet-4-6": "#dc2626",
    "gemini-3-flash": "#16a34a",
}
WEAK_STRONG = {
    "gpt-5": ("minimal", "high"),
    "claude-sonnet-4-6": ("low", "max"),
    "gemini-3-flash": ("minimal", "high"),
}
DELTA_COLS = [
    "passable_language_per_1k",
    "self_interest_language_per_1k",
    "passable_minus_self_per_1k",
    "concession_language_per_1k",
    "refusal_language_per_1k",
    "target_utility",
    "utility_gap",
    "absolute_payoff_gap",
    "payoff_gini_corrected",
    "payoff_variance",
    "target_abs_fair_excess",
    "target_fair_excess",
    "fairness_distance",
    "round",
]


def load_matched_deltas() -> pd.DataFrame:
    scored = pd.read_csv(SCORED_RUNS)
    rows = []
    for family, (weak, strong) in WEAK_STRONG.items():
        sub = scored[scored["family"].eq(family)].copy()
        weak_df = sub[sub["level"].eq(weak)].set_index(["game_cell", "order"])
        strong_df = sub[sub["level"].eq(strong)].set_index(["game_cell", "order"])
        common = sorted(set(weak_df.index) & set(strong_df.index))
        for key in common:
            w = weak_df.loc[key]
            s = strong_df.loc[key]
            row = {
                "family": family,
                "weak_level": weak,
                "strong_level": strong,
                "game_cell": key[0],
                "order": key[1],
                "game": s["game"],
                "delta_tokens": float(s["target_compute_tokens_per_call"]) - float(w["target_compute_tokens_per_call"]),
            }
            for col in DELTA_COLS:
                row[f"delta_{col}"] = float(s[col]) - float(w[col])
                row[f"weak_{col}"] = float(w[col])
                row[f"strong_{col}"] = float(s[col])
            rows.append(row)
    frame = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)
    out = OUT_DIR / "ttc_objective_shift_weak_strong_deltas.csv"
    frame.to_csv(out, index=False)
    return frame


def fit_stats(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return {"slope": math.nan, "intercept": math.nan, "r": math.nan}
    slope, intercept = np.polyfit(x, y, 1)
    r = float(np.corrcoef(x, y)[0, 1])
    return {"slope": float(slope), "intercept": float(intercept), "r": r}


def scatter_delta_grid(frame: pd.DataFrame) -> Path:
    specs = [
        ("delta_utility_gap", "Delta target - baseline"),
        ("delta_absolute_payoff_gap", "Delta absolute payoff gap"),
        ("delta_payoff_gini_corrected", "Delta corrected Gini"),
        ("delta_target_abs_fair_excess", "Delta |target fair-share excess|"),
        ("delta_target_utility", "Delta target payoff"),
        ("delta_payoff_variance", "Delta payoff variance"),
    ]
    x_col = "delta_passable_minus_self_per_1k"
    fig, axes = plt.subplots(2, 3, figsize=(14.2, 8.0))
    axes = axes.ravel()
    x = frame[x_col].to_numpy(dtype=float)

    for ax, (y_col, ylabel) in zip(axes, specs, strict=True):
        y = frame[y_col].to_numpy(dtype=float)
        for family, sub in frame.groupby("family"):
            ax.scatter(
                sub[x_col],
                sub[y_col],
                s=46,
                marker=FAMILY_MARKERS.get(family, "o"),
                color=FAMILY_COLORS.get(family, "#475569"),
                edgecolor="white",
                linewidth=0.6,
                alpha=0.78,
                label=FAMILY_LABELS.get(family, family),
            )
        stats = fit_stats(x, y)
        if math.isfinite(stats["slope"]):
            xs = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), 100)
            ax.plot(xs, stats["slope"] * xs + stats["intercept"], color="#111827", linewidth=1.7, alpha=0.85)
        ax.axhline(0.0, color="#94a3b8", linewidth=0.9, alpha=0.8)
        ax.axvline(0.0, color="#94a3b8", linewidth=0.9, alpha=0.8)
        ax.text(
            0.03,
            0.95,
            f"r={stats['r']:+.2f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            bbox={"facecolor": "white", "edgecolor": "#d1d5db", "alpha": 0.9, "pad": 3},
        )
        ax.set_xlabel("Delta passable minus self-interest language / 1k words")
        ax.set_ylabel(ylabel)
        ax.grid(True, color="#d1d5db", alpha=0.42, linewidth=0.85)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=True, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Weak-to-strong TTC changes within matched game/order cells", fontsize=18, y=0.98)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    out = GRAPHICS_DIR / "ttc_objective_shift_weak_strong_delta_scatter.png"
    fig.savefig(out, dpi=240, bbox_inches="tight", pad_inches=0.14)
    plt.close(fig)
    return out


def bin_delta_plot(frame: pd.DataFrame) -> Path:
    data = frame.copy()
    data["passable_delta_bin"] = pd.qcut(
        data["delta_passable_minus_self_per_1k"].rank(method="first"),
        4,
        labels=["Q1", "Q2", "Q3", "Q4"],
    ).astype(str)
    summary = (
        data.groupby("passable_delta_bin", dropna=False)
        .agg(
            n=("family", "size"),
            delta_passable_minus_self_per_1k=("delta_passable_minus_self_per_1k", "mean"),
            delta_target_utility=("delta_target_utility", "mean"),
            delta_utility_gap=("delta_utility_gap", "mean"),
            delta_absolute_payoff_gap=("delta_absolute_payoff_gap", "mean"),
            delta_payoff_gini_corrected=("delta_payoff_gini_corrected", "mean"),
            delta_target_abs_fair_excess=("delta_target_abs_fair_excess", "mean"),
            delta_payoff_variance=("delta_payoff_variance", "mean"),
        )
        .reset_index()
        .sort_values("passable_delta_bin")
    )
    summary.to_csv(OUT_DIR / "ttc_objective_shift_weak_strong_delta_quartiles.csv", index=False)

    specs = [
        ("delta_target_utility", "Delta target payoff"),
        ("delta_utility_gap", "Delta target - baseline"),
        ("delta_absolute_payoff_gap", "Delta absolute gap"),
        ("delta_payoff_gini_corrected", "Delta corrected Gini"),
        ("delta_target_abs_fair_excess", "Delta |target fair-share excess|"),
        ("delta_payoff_variance", "Delta payoff variance"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13.8, 7.2))
    axes = axes.ravel()
    xs = np.arange(len(summary))
    labels = list(summary["passable_delta_bin"])
    for ax, (metric, ylabel) in zip(axes, specs, strict=True):
        ax.plot(xs, summary[metric], marker="o", color="#334155", linewidth=2.0)
        ax.axhline(0.0, color="#94a3b8", linewidth=0.9)
        ax.set_xticks(xs, labels)
        ax.set_xlabel("Delta passable-vs-self language quartile")
        ax.set_ylabel(ylabel)
        ax.grid(True, color="#d1d5db", alpha=0.42, linewidth=0.85)
    fig.suptitle("Matched weak-to-strong outcomes by objective-shift language", fontsize=18, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = GRAPHICS_DIR / "ttc_objective_shift_weak_strong_delta_quartiles.png"
    fig.savefig(out, dpi=240, bbox_inches="tight", pad_inches=0.14)
    plt.close(fig)
    return out


def main() -> None:
    frame = load_matched_deltas()
    print(frame.shape)
    print(scatter_delta_grid(frame))
    print(bin_delta_plot(frame))
    cols = [
        "delta_passable_minus_self_per_1k",
        "delta_target_utility",
        "delta_utility_gap",
        "delta_absolute_payoff_gap",
        "delta_payoff_gini_corrected",
        "delta_target_abs_fair_excess",
        "delta_payoff_variance",
    ]
    print(frame[cols].corr()["delta_passable_minus_self_per_1k"].sort_values().to_string())


if __name__ == "__main__":
    main()
