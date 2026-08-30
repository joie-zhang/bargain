#!/usr/bin/env python3
"""Render the streamlined ICML-AI-WILD Appendix Figure 29.

The renderer uses the preserved May 5 fitted payoff-performance-Elo table, so
it changes only the presentation of the paper figure—not the fitted ratings.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    PROJECT_ROOT
    / "experiments/results/n2_plus_multiagent_comparison_analysis_20260505"
    / "tables_multiagent/heterogeneous_performance_elo_rankings.csv"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "experiments/results/n2_plus_multiagent_comparison_analysis_20260505"
    / "plots_multiagent/heterogeneous_performance_elo_vs_arena_by_game_n.png"
)

GAME_ORDER = ("game1", "game2", "game3")
N_ORDER = (2, 4, 6, 8, 10)
N_COLORS = {
    2: "#1f77b4",
    4: "#d62728",
    6: "#2ca02c",
    8: "#9467bd",
    10: "#ff7f0e",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_plot_data(path: Path) -> pd.DataFrame:
    rankings = pd.read_csv(path)
    required = {
        "arena_elo",
        "performance_elo",
        "game_label",
        "n_agents",
        "competition_band",
        "scope",
    }
    missing = required.difference(rankings.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    plot_data = rankings[
        rankings["scope"].eq("by_n")
        & rankings["competition_band"].eq("all")
        & rankings["game_label"].isin(GAME_ORDER)
        & rankings["n_agents"].isin(N_ORDER)
    ].copy()
    if len(plot_data) != 356:
        raise ValueError(f"Expected 356 plotted ratings, found {len(plot_data)}")

    panel_counts = plot_data.groupby(["game_label", "n_agents"]).size()
    expected_panels = {(game, n) for game in GAME_ORDER for n in N_ORDER}
    if set(panel_counts.index) != expected_panels:
        raise ValueError("The retained table does not contain all 15 Figure 29 panels")
    return plot_data


def render(plot_data: pd.DataFrame, output: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 13.5,
            "font.weight": "normal",
            "axes.labelweight": "normal",
            "axes.titleweight": "normal",
            "axes.edgecolor": "#555555",
            "axes.linewidth": 0.9,
        }
    )

    fig, axes = plt.subplots(
        3,
        5,
        figsize=(13.5, 7.6),
        sharex=True,
        sharey=True,
    )

    for row, game in enumerate(GAME_ORDER):
        for col, n_agents in enumerate(N_ORDER):
            ax = axes[row, col]
            panel = plot_data[
                plot_data["game_label"].eq(game)
                & plot_data["n_agents"].eq(n_agents)
            ].sort_values("arena_elo")
            color = N_COLORS[n_agents]

            # Show descriptive point estimates without invalid model-wise
            # curvature bars; pairwise outcomes within a run are dependent.
            ax.scatter(
                panel["arena_elo"],
                panel["performance_elo"],
                s=78,
                color=color,
                edgecolors="none",
                alpha=0.26,
                zorder=2,
            )

            slope, intercept = np.polyfit(
                panel["arena_elo"].to_numpy(dtype=float),
                panel["performance_elo"].to_numpy(dtype=float),
                1,
            )
            fit_x = np.linspace(1235.0, 1510.0, 200)
            ax.plot(
                fit_x,
                slope * fit_x + intercept,
                color=color,
                linewidth=3.4,
                alpha=0.98,
                solid_capstyle="round",
                zorder=3,
            )

            ax.set_xlim(1225, 1515)
            ax.set_ylim(1180, 1780)
            ax.set_xticks([1300, 1450])
            ax.set_yticks([1200, 1400, 1600, 1750])
            ax.grid(True, color="#d9d9d9", linewidth=0.7, alpha=0.50)
            ax.set_axisbelow(True)

            ax.tick_params(
                axis="both",
                labelsize=13.5,
                width=0.9,
                length=3.5,
                colors="#333333",
            )
            ax.tick_params(labelleft=col == 0, labelbottom=row == 2)

            if row == 0:
                ax.set_title(f"N = {n_agents}", fontsize=18, pad=7, fontweight="normal")
            if row == 2:
                ax.set_xlabel("Arena Elo", fontsize=16.5, labelpad=7, fontweight="normal")
            if col == 0:
                ax.text(
                    0.035,
                    0.945,
                    f"Game {row + 1}",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=15.0,
                    fontweight="normal",
                    color="#222222",
                )

    fig.supylabel(
        "Payoff performance Elo",
        x=0.012,
        fontsize=18.0,
        fontweight="normal",
    )

    # Tight panel spacing is intentional: labels are shared by row/column, so
    # the axes can occupy the space previously used by repeated text.
    fig.subplots_adjust(
        left=0.076,
        right=0.995,
        bottom=0.105,
        top=0.944,
        wspace=0.055,
        hspace=0.090,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    plot_data = load_plot_data(args.input.resolve())
    output = args.output.resolve()
    render(plot_data, output)
    print(f"Rendered {len(plot_data)} ratings across 15 panels")
    print(f"Wrote: {output}")


if __name__ == "__main__":
    main()
