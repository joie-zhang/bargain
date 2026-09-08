#!/usr/bin/env python3
"""Render the paper's heterogeneous competition figure as a spacious 5x3 grid.

The input table is the raw-run-derived aggregate retained by the Figure 25
reproduction audit.  Each point is a model/cell mean and each dashed curve is
an OLS fit within one exact game-specific competition setting.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    REPO_ROOT
    / "reproduction_audit/fig25_heterogeneous_competition/plotted_aggregate.csv"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "experiments/results/figure_iteration_20260802/multiagent/"
    / "heterogeneous_payoff_vs_arena_elo_by_competition_5x3_clean.png"
)

GAME_ORDER = ("game1", "game2", "game3")
GAME_TITLES = {
    "game1": "Item Allocation",
    "game2": "Diplomacy Treaty",
    "game3": "Co-Funding",
}
N_ORDER = (2, 4, 6, 8, 10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_aggregate(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "game_label",
        "n_agents",
        "elo",
        "competition_ci_rounded",
        "competition_label_ci",
        "final_utility",
        "final_utility_sem",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")
    if len(frame) != 1487:
        raise ValueError(f"Expected 1,487 plotted aggregate rows; found {len(frame)}")
    if set(frame["game_label"].unique()) != set(GAME_ORDER):
        raise ValueError("Unexpected game coverage in the Figure 25 aggregate")
    if set(frame["n_agents"].unique()) != set(N_ORDER):
        raise ValueError("Unexpected group-size coverage in the Figure 25 aggregate")
    return frame


def add_competition_series(ax: plt.Axes, subset: pd.DataFrame, color: object) -> None:
    subset = subset.sort_values("elo")
    x = subset["elo"].to_numpy(dtype=float)
    y = subset["final_utility"].to_numpy(dtype=float)
    yerr = subset["final_utility_sem"].fillna(0.0).clip(lower=0.0).to_numpy(dtype=float)

    # Means and their SEM remain visible, but deliberately recede behind the fits.
    ax.errorbar(
        x,
        y,
        yerr=yerr,
        fmt="o",
        linestyle="none",
        markersize=6.0,
        markeredgewidth=0.0,
        color=color,
        ecolor=color,
        elinewidth=0.9,
        capsize=1.8,
        alpha=0.24,
        zorder=2,
    )

    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 2 or np.unique(x[valid]).size < 2:
        return
    slope, intercept = np.polyfit(x[valid], y[valid], deg=1)
    fit_x = np.linspace(float(x[valid].min()), float(x[valid].max()), 120)
    ax.plot(
        fit_x,
        slope * fit_x + intercept,
        color=color,
        linestyle="--",
        linewidth=3.4,
        alpha=0.96,
        dash_capstyle="round",
        zorder=3,
    )


def render(frame: pd.DataFrame, output: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.linewidth": 1.2,
            "xtick.major.width": 1.1,
            "ytick.major.width": 1.1,
        }
    )

    figure, axes = plt.subplots(
        len(N_ORDER),
        len(GAME_ORDER),
        figsize=(14.2, 15.6),
        dpi=220,
        sharex=True,
        sharey="col",
    )
    figure.subplots_adjust(
        left=0.155,
        right=0.885,
        bottom=0.075,
        top=0.955,
        wspace=0.20,
        hspace=0.15,
    )

    colormap = plt.get_cmap("viridis")
    normalization = Normalize(vmin=0.0, vmax=1.0)

    for row, n_agents in enumerate(N_ORDER):
        for col, game in enumerate(GAME_ORDER):
            ax = axes[row, col]
            panel = frame[
                frame["game_label"].eq(game) & frame["n_agents"].eq(n_agents)
            ]
            for competition, subset in panel.groupby("competition_ci_rounded", sort=True):
                add_competition_series(ax, subset, colormap(normalization(float(competition))))

            if row == 0:
                ax.set_title(GAME_TITLES[game], fontsize=23, pad=12)
            if row != len(N_ORDER) - 1:
                ax.tick_params(axis="x", labelbottom=False)

            ax.tick_params(axis="both", which="major", labelsize=16, length=5.5)
            ax.grid(True, color="#9ca3af", linewidth=0.75, alpha=0.20)
            ax.set_axisbelow(True)
            for spine in ax.spines.values():
                spine.set_color("#374151")

    # Group-size labels identify rows without repeating titles inside 15 panels.
    for row, n_agents in enumerate(N_ORDER):
        position = axes[row, 0].get_position()
        figure.text(
            0.064,
            (position.y0 + position.y1) / 2.0,
            f"n = {n_agents}",
            ha="center",
            va="center",
            fontsize=20,
        )

    figure.supxlabel("Arena Elo", fontsize=21, y=0.025)
    figure.supylabel("Mean Model Payoff", fontsize=21, x=0.012)

    colorbar_axis = figure.add_axes([0.915, 0.16, 0.022, 0.68])
    scalar_mappable = ScalarMappable(norm=normalization, cmap=colormap)
    scalar_mappable.set_array([])
    colorbar = figure.colorbar(scalar_mappable, cax=colorbar_axis)
    colorbar.set_label("Competition Index", fontsize=20, labelpad=16)
    colorbar.set_ticks(np.linspace(0.0, 1.0, 6))
    colorbar.ax.tick_params(labelsize=16, width=1.1, length=5.0)

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, facecolor="white", bbox_inches="tight", pad_inches=0.12)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    aggregate = load_aggregate(args.input.resolve())
    render(aggregate, args.output.resolve())
    print(f"Rendered {len(aggregate):,} aggregate model/cell rows")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
