#!/usr/bin/env python3
"""Regenerate the ICML homogeneous-adversary main-paper panels."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GRAPHICS_DIR = PROJECT_ROOT / "overleaf/icml_aiwild_template/graphics/n_gt_2_report"
GINI_SUMMARY = GRAPHICS_DIR / "homogeneous_adversary_baseline_only_vs_all_payoff_gini_summary.csv"
ROLE_SUMMARY = GRAPHICS_DIR / "role_payoff_with_within_run_variance_bars_summary.csv"
GINI_OUT = GRAPHICS_DIR / "homogeneous_adversary_baseline_only_vs_all_payoff_gini.png"
ROLE_OUT = GRAPHICS_DIR / "homogeneous_adversary_role_payoff_vs_adversary_elo_within_run_variance_bars_tall_y40_60_no_errorbars.png"
COMBINED_OUT = GRAPHICS_DIR / "homogeneous_adversary_gini_and_role_payoff.png"

FIGSIZE = (5.8, 4.9)
DPI = 320
SUBPLOT_KW = dict(left=0.27, right=0.975, bottom=0.18, top=0.96)


def style_axes(ax: plt.Axes) -> None:
    ax.grid(True, axis="y", alpha=0.28, linewidth=0.7)
    ax.tick_params(axis="both", labelsize=13)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def plot_gini() -> None:
    summary = pd.read_csv(GINI_SUMMARY)
    summary = summary[summary["scope"].eq("overall")].sort_values("bucket_x").reset_index(drop=True)

    x = np.arange(len(summary))
    y = summary["baseline_only_payoff_gini_mean"].to_numpy(dtype=float)
    yerr = np.nan_to_num(summary["baseline_only_payoff_gini_sem"].to_numpy(dtype=float), nan=0.0)
    labels = summary["bucket_label"].astype(str).tolist()

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.bar(
        x,
        y,
        width=0.6,
        yerr=yerr,
        capsize=4,
        color="#4E79A7",
        alpha=0.86,
        edgecolor="black",
        linewidth=0.7,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=14)
    ax.set_xlabel("Adversary Elo", fontsize=16, labelpad=8)
    ax.set_ylabel("Gini inequality\namongst the N-1 baseline agents", fontsize=16, labelpad=10)
    ax.set_ylim(0.10, 0.23)
    ax.set_yticks(np.arange(0.10, 0.231, 0.02))
    style_axes(ax)
    fig.subplots_adjust(**SUBPLOT_KW)
    fig.savefig(GINI_OUT, dpi=DPI)
    plt.close(fig)


def plot_role_payoffs() -> None:
    summary = pd.read_csv(ROLE_SUMMARY)
    summary = (
        summary[summary["scenario"].eq("homogeneous_adversary")]
        .sort_values("bucket_x")
        .reset_index(drop=True)
    )

    x = np.arange(len(summary))
    high = summary["high_role_payoff_mean"].to_numpy(dtype=float)
    low = summary["low_role_payoff_mean"].to_numpy(dtype=float)
    labels = summary["bucket_label"].astype(str).tolist()

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(x, high, color="#D95F02", marker="o", markersize=6.0, linewidth=2.6, label="Adversary")
    ax.plot(x, low, color="#4E79A7", marker="o", markersize=6.0, linewidth=2.6, label="Baseline agents")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=14)
    ax.set_xlabel("Adversary Elo", fontsize=16, labelpad=8)
    ax.set_ylabel("Payoff", fontsize=16, labelpad=10)
    ax.set_ylim(40.0, 60.0)
    ax.set_yticks(np.arange(40.0, 60.1, 2.5))
    ax.legend(loc="upper left", frameon=False, fontsize=15, handlelength=1.6)
    style_axes(ax)
    fig.subplots_adjust(**SUBPLOT_KW)
    fig.savefig(ROLE_OUT, dpi=DPI)
    plt.close(fig)


def combine_panels() -> None:
    gap_px = 32
    with Image.open(GINI_OUT) as left, Image.open(ROLE_OUT) as right:
        left = left.convert("RGB")
        right = right.convert("RGB")
        width = left.width + gap_px + right.width
        height = max(left.height, right.height)
        combined = Image.new("RGB", (width, height), "white")
        combined.paste(left, (0, 0))
        combined.paste(right, (left.width + gap_px, 0))
        combined.save(COMBINED_OUT)


def main() -> None:
    plot_gini()
    plot_role_payoffs()
    combine_panels()
    print(f"Wrote {GINI_OUT}")
    print(f"Wrote {ROLE_OUT}")
    print(f"Wrote {COMBINED_OUT}")


if __name__ == "__main__":
    main()
