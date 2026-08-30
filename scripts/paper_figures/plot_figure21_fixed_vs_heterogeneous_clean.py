#!/usr/bin/env python3
"""Render the fixed-baseline and heterogeneous N=2 comparison."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
AUDIT_DIR = ROOT / "reproduction_audit" / "fig21_fixed_vs_random_pairings"
FIXED_ROWS = (
    ROOT
    / "reproduction_audit"
    / "fig10_bilateral_adversary_payoff"
    / "generated"
    / "primary_run_lineage.csv"
)
HETEROGENEOUS_ROWS = AUDIT_DIR / "heterogeneous_agent_rows_from_raw.csv"
CANONICAL_OUTPUT = (
    ROOT
    / "experiments"
    / "results"
    / "n2_plus_multiagent_comparison_analysis_20260505"
    / "plots_multiagent"
    / "n2_baseline_vs_heterogeneous_pairings.png"
)
PAPER_OUTPUT = (
    ROOT
    / "overleaf"
    / "icml_aiwild_template"
    / "graphics"
    / "n_gt_2_report"
    / "n2_baseline_vs_heterogeneous_pairings.png"
)

GAME_ORDER = ("game1", "game2", "game3")
GAME_TITLES = {
    "game1": "Game 1: Item Allocation",
    "game2": "Game 2: Diplomatic Treaty",
    "game3": "Game 3: Co-Funding",
}
SERIES = (
    ("Fixed GPT-5-nano", "#e45756"),
    ("Heterogeneous pairings", "#2f80bd"),
)


def sem(values: pd.Series) -> float:
    """Return the usual standard error of the mean for a plotted model cell."""
    clean = values.dropna()
    if len(clean) < 2:
        return float("nan")
    return float(clean.std(ddof=1) / np.sqrt(len(clean)))


def aggregate() -> tuple[pd.DataFrame, pd.DataFrame]:
    fixed = pd.read_csv(FIXED_ROWS)
    heterogeneous = pd.read_csv(HETEROGENEOUS_ROWS)

    if len(fixed) != 1500:
        raise ValueError("Unexpected primary fixed-baseline table.")
    if len(heterogeneous) != 520:
        raise ValueError("Unexpected heterogeneous reconstruction table.")

    fixed_agg = (
        fixed.groupby(
            ["game_id", "adversary_model", "adversary_elo"],
            dropna=False,
        )
        .agg(
            payoff=("adversary_utility", "mean"),
            payoff_sem=("adversary_utility", sem),
            observations=("adversary_utility", "count"),
        )
        .reset_index()
        .rename(columns={"game_id": "game", "adversary_elo": "elo"})
    )

    heterogeneous = heterogeneous[
        heterogeneous["experiment_family"].eq("heterogeneous_random")
        & heterogeneous["n_agents"].eq(2)
    ].copy()
    heterogeneous_agg = (
        heterogeneous.groupby(
            ["game_label", "model", "model_short", "elo"], dropna=False
        )
        .agg(
            payoff=("final_utility", "mean"),
            payoff_sem=("final_utility", sem),
            observations=("final_utility", "count"),
        )
        .reset_index()
        .rename(columns={"game_label": "game"})
    )
    return fixed_agg, heterogeneous_agg


def plot_series(
    ax: plt.Axes,
    frame: pd.DataFrame,
    *,
    label: str,
    color: str,
) -> None:
    frame = frame.sort_values("elo")
    x = frame["elo"].to_numpy(dtype=float)
    y = frame["payoff"].to_numpy(dtype=float)
    yerr = frame["payoff_sem"].to_numpy(dtype=float)

    ax.errorbar(
        x,
        y,
        yerr=yerr,
        fmt="none",
        ecolor=to_rgba(color, 0.27),
        elinewidth=1.45,
        capsize=2.8,
        capthick=1.25,
        zorder=1,
    )
    ax.scatter(
        x,
        y,
        s=66,
        color=color,
        alpha=0.48,
        edgecolors="none",
        label=label,
        zorder=2,
    )

    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() >= 2:
        slope, intercept = np.polyfit(x[finite], y[finite], 1)
        fit_x = np.linspace(x[finite].min(), x[finite].max(), 200)
        ax.plot(
            fit_x,
            slope * fit_x + intercept,
            color=color,
            linestyle="--",
            linewidth=3.25,
            alpha=0.98,
            zorder=3,
        )


def render(outputs: tuple[Path, ...] = (CANONICAL_OUTPUT, PAPER_OUTPUT)) -> tuple[Path, ...]:
    fixed, heterogeneous = aggregate()
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 5.05), sharey=False)

    for index, (ax, game) in enumerate(zip(axes, GAME_ORDER)):
        plot_series(
            ax,
            fixed[fixed["game"].eq(game)],
            label=SERIES[0][0],
            color=SERIES[0][1],
        )
        plot_series(
            ax,
            heterogeneous[heterogeneous["game"].eq(game)],
            label=SERIES[1][0],
            color=SERIES[1][1],
        )

        ax.set_title(GAME_TITLES[game], fontsize=21, fontweight="normal", pad=12)
        ax.set_xlabel("Arena Elo", fontsize=18, fontweight="normal", labelpad=9)
        ax.set_ylabel(
            "Mean Payoff" if index == 0 else "",
            fontsize=18,
            fontweight="normal",
            labelpad=10,
        )
        ax.tick_params(axis="both", which="major", labelsize=14.5, width=1.1, length=5)
        ax.grid(True, color="#9aa0a6", alpha=0.18, linewidth=0.9)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_color("#555555")
            spine.set_linewidth(1.05)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=2,
        frameon=False,
        fontsize=18.5,
        handletextpad=0.7,
        columnspacing=2.2,
        markerscale=1.1,
    )
    fig.subplots_adjust(left=0.066, right=0.992, bottom=0.28, top=0.91, wspace=0.16)

    for output in outputs:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return outputs


if __name__ == "__main__":
    for path in render():
        print(path)
