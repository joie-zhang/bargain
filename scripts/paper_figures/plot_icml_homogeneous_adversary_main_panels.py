#!/usr/bin/env python3
"""Regenerate the ICML homogeneous-adversary Gini appendix figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GRAPHICS_DIR = PROJECT_ROOT / "overleaf/icml_aiwild_template/graphics/n_gt_2_report"
GINI_SUMMARY = GRAPHICS_DIR / "homogeneous_adversary_baseline_only_vs_all_payoff_gini_summary.csv"
GINI_OUT = GRAPHICS_DIR / "homogeneous_adversary_baseline_only_vs_all_payoff_gini.png"

FIGSIZE = (6.4, 5.15)
DPI = 320
BLUE = "#4E79A7"
BLUE_DARK = "#2F6085"
GRID = "#D1D5DB"

MODEL_LABELS = {
    1240: "Nova\nMicro\n1240",
    1317: "GPT-4o\nmini\n1317",
    1389: "Sonnet\n4\n1389",
    1448: "Gemini\n2.5 Pro\n1448",
    1484: "GPT-5.4\nHigh\n1484",
}


def style_axes(ax: plt.Axes) -> None:
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", color=GRID, alpha=0.52, linewidth=0.75)
    ax.tick_params(axis="both", labelsize=14, width=1.45, length=5.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.45)
    ax.spines["bottom"].set_linewidth(1.45)


def plot_gini() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    summary = pd.read_csv(GINI_SUMMARY)
    summary = summary[summary["scope"].eq("overall")].sort_values("bucket_x").reset_index(drop=True)
    summary["elo"] = pd.to_numeric(summary["bucket_x"], errors="raise").round().astype(int)
    if summary["elo"].tolist() != list(MODEL_LABELS):
        raise RuntimeError(f"Expected five model Elo values {list(MODEL_LABELS)}, found {summary['elo'].tolist()}")
    if not summary["n_runs"].eq(260).all():
        raise RuntimeError("Expected 260 homogeneous-adversary runs per model")

    x = np.arange(len(summary))
    y = summary["baseline_only_payoff_gini_mean"].to_numpy(dtype=float)
    yerr = np.nan_to_num(summary["baseline_only_payoff_gini_sem"].to_numpy(dtype=float), nan=0.0)
    labels = summary["elo"].map(MODEL_LABELS).tolist()

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.bar(
        x,
        y,
        width=0.66,
        yerr=yerr,
        capsize=5,
        color=BLUE,
        alpha=0.78,
        edgecolor=BLUE_DARK,
        linewidth=1.0,
        error_kw={"ecolor": BLUE_DARK, "elinewidth": 1.55, "capthick": 1.55},
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12.5, linespacing=0.95)
    ax.set_xlim(-0.48, len(summary) - 0.52)
    ax.set_xlabel("Adversary model and Arena Elo", fontsize=18, labelpad=9)
    ax.set_ylabel("Baseline-agent\nGini inequality", fontsize=18, labelpad=11)
    ax.set_ylim(0.10, 0.23)
    ax.set_yticks(np.arange(0.10, 0.231, 0.02))
    style_axes(ax)
    fig.subplots_adjust(left=0.17, right=0.985, bottom=0.285, top=0.975)
    fig.savefig(
        GINI_OUT,
        dpi=DPI,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close(fig)


def main() -> None:
    plot_gini()
    print(f"Wrote {GINI_OUT}")


if __name__ == "__main__":
    main()
