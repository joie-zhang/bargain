#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "analysis/recreated_figures/figure7_from_script"
OUT_DIR = ROOT / "analysis/recreated_figures/figure7_label_edits"
OUT_PATH = OUT_DIR / "heterogeneous_vs_homogeneous_gini_bars_label_edits.png"


def main() -> None:
    het = pd.read_csv(DATA_DIR / "heterogeneous_gini_run_metrics.csv")
    hom = pd.read_csv(DATA_DIR / "homogeneous_gini_run_metrics.csv")

    het_mean = float(het["payoff_gini_corrected"].mean())
    het_sem = float(het["payoff_gini_corrected"].sem())
    hom_mean = float(hom["payoff_gini_corrected"].mean())
    hom_sem = float(hom["payoff_gini_corrected"].sem())

    model_summary = (
        hom.groupby(["model_short", "model_elo"], as_index=False)["payoff_gini_corrected"]
        .agg(["mean", "sem"])
        .reset_index()
        .sort_values("model_elo")
    )

    # The original Figure 7 fit line matches a run-level linear fit.
    slope, intercept = np.polyfit(
        hom["model_elo"].to_numpy(dtype=float),
        hom["payoff_gini_corrected"].to_numpy(dtype=float),
        1,
    )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 17,
            "axes.labelsize": 18,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 13,
        }
    )

    # Preserve the paper asset canvas: 2520 x 990 px at 180 dpi.
    fig = plt.figure(figsize=(14, 5.5), dpi=180)
    ax_bar = fig.add_axes([0.067, 0.124, 0.224, 0.762])
    ax_scatter = fig.add_axes([0.392, 0.124, 0.583, 0.762])

    red = "#d62728"
    green = "#2ca02c"

    ax_bar.bar(
        [0, 1],
        [het_mean, hom_mean],
        yerr=[het_sem, hom_sem],
        color=[red, green],
        edgecolor=[red, green],
        width=0.65,
        ecolor="black",
        capsize=4,
    )
    ax_bar.set_xticks([0, 1])
    ax_bar.set_xticklabels(
        ["Heterogeneous", "Homogenous Control\n(Monocultures)"],
        fontsize=11.8,
    )
    ax_bar.set_ylabel("Gini inequality")
    ax_bar.set_ylim(0, 0.227)
    ax_bar.set_yticks(np.arange(0, 0.21, 0.05))
    ax_bar.grid(True, axis="y", alpha=0.25)

    x = model_summary["model_elo"].to_numpy(dtype=float)
    y = model_summary["mean"].to_numpy(dtype=float)
    yerr = model_summary["sem"].to_numpy(dtype=float)
    ax_scatter.errorbar(
        x,
        y,
        yerr=yerr,
        fmt="o",
        color=green,
        ecolor=green,
        elinewidth=1.4,
        capsize=4,
        markersize=6,
        zorder=3,
    )

    fit_x = np.array([x.min(), x.max()])
    ax_scatter.plot(
        fit_x,
        intercept + slope * fit_x,
        color=green,
        alpha=0.6,
        linewidth=1.4,
        label="Homogenous Control (Monocultures)",
    )
    ax_scatter.axhspan(het_mean - het_sem, het_mean + het_sem, color=red, alpha=0.12)
    ax_scatter.axhline(
        het_mean,
        color=red,
        linestyle="--",
        linewidth=1.7,
        label="Heterogeneous mean",
    )

    labels = {
        "Nova Micro": (4, 0.028, "left"),
        "Claude 3 Haiku": (4, -0.018, "left"),
        "GPT-5 nano": (4, 0.020, "left"),
        "Opus 4.5": (4, -0.010, "left"),
        "GPT-5.2 Chat": (4, 0.030, "left"),
        "Opus 4.6": (5, -0.014, "left"),
    }
    for row in model_summary.itertuples(index=False):
        if row.model_short not in labels:
            continue
        dx, dy, ha = labels[row.model_short]
        ax_scatter.annotate(
            row.model_short,
            xy=(row.model_elo, row.mean),
            xytext=(row.model_elo + dx, row.mean + dy),
            textcoords="data",
            ha=ha,
            va="center",
            fontsize=12.5,
            arrowprops={"arrowstyle": "-", "color": "#999999", "lw": 0.7},
        )

    ax_scatter.set_xlabel("Monoculture model Elo")
    ax_scatter.set_ylabel("Gini inequality")
    ax_scatter.set_xlim(1210, 1529)
    ax_scatter.set_ylim(-0.025, 0.525)
    ax_scatter.set_xticks([1250, 1300, 1350, 1400, 1450, 1500])
    ax_scatter.set_yticks(np.arange(0, 0.51, 0.1))
    ax_scatter.grid(True, alpha=0.25)

    handles, labels = ax_scatter.get_legend_handles_labels()
    order = [labels.index("Heterogeneous mean"), labels.index("Homogenous Control (Monocultures)")]
    ax_scatter.legend(
        [handles[i] for i in order],
        [labels[i] for i in order],
        loc="upper right",
        frameon=False,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH)
    plt.close(fig)
    print(OUT_PATH)


if __name__ == "__main__":
    main()
