#!/usr/bin/env python3
"""Render the polished 5-by-3 Appendix Figure 24 from raw-derived means.

The input table is the audited aggregation of 1,300 homogeneous-adversary
experiment JSON files: five adversary models, four runs per plotted point,
three games, and N in {2, 4, 6, 8, 10}.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    REPO_ROOT
    / "reproduction_audit/fig24_hom_adversary_competition/aggregated_plot_data.csv"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "overleaf/icml_aiwild_template/graphics/n_gt_2_report/"
    / "hom_adversary_payoff_vs_elo_by_competition_3x5.png"
)

GAME_ORDER = ("game1", "game2", "game3")
GAME_LABELS = {
    "game1": "Item Allocation",
    "game2": "Diplomacy Treaty",
    "game3": "Co-funding",
}
N_ORDER = (2, 4, 6, 8, 10)
CANVAS_PX = (4200, 4900)
DPI = 300


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def validate_data(frame: pd.DataFrame) -> None:
    required = {
        "game_label",
        "n_agents",
        "adversary_model",
        "adversary_elo",
        "competition_ci_rounded",
        "adversary_utility",
        "adversary_utility_sem",
        "run_count",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if len(frame) != 325:
        raise ValueError(f"Expected 325 plotted means; found {len(frame)}")
    if set(frame["game_label"].unique()) != set(GAME_ORDER):
        raise ValueError("Unexpected game coverage")
    if set(frame["n_agents"].unique()) != set(N_ORDER):
        raise ValueError("Unexpected N coverage")
    if not frame["run_count"].eq(4).all():
        raise ValueError("Every plotted mean must aggregate exactly four raw runs")
    cell_sizes = frame.groupby(
        ["game_label", "n_agents", "competition_ci_rounded"], dropna=False
    ).size()
    if not cell_sizes.eq(5).all():
        raise ValueError("Every competition series must contain five Elo points")


def linear_fit(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    x = frame["adversary_elo"].to_numpy(dtype=float)
    y = frame["adversary_utility"].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    fit_x = np.linspace(float(x.min()), float(x.max()), 120)
    return fit_x, slope * fit_x + intercept


def make_figure(frame: pd.DataFrame, output: Path) -> None:
    width_px, height_px = CANVAS_PX
    fig = plt.figure(
        figsize=(width_px / DPI, height_px / DPI),
        dpi=DPI,
        facecolor="white",
    )
    grid = fig.add_gridspec(
        5,
        4,
        width_ratios=[1, 1, 1, 0.045],
        left=0.105,
        right=0.955,
        bottom=0.075,
        top=0.955,
        wspace=0.17,
        hspace=0.17,
    )
    axes = np.empty((5, 3), dtype=object)
    cmap = plt.colormaps["viridis"]
    # The three game-specific indices are normalized competition quantities,
    # so one scale makes colors directly comparable throughout the grid.
    norm = Normalize(vmin=0.0, vmax=1.0)

    for row, n_agents in enumerate(N_ORDER):
        for col, game in enumerate(GAME_ORDER):
            game_frame = frame[frame["game_label"].eq(game)]
            share_x = axes[0, col] if row else None
            share_y = axes[0, col] if row else None
            ax = fig.add_subplot(grid[row, col], sharex=share_x, sharey=share_y)
            axes[row, col] = ax
            panel = game_frame[game_frame["n_agents"].eq(n_agents)]

            for competition, series in panel.groupby("competition_ci_rounded"):
                series = series.sort_values("adversary_elo")
                color = cmap(norm(float(competition)))

                # Keep observations and uncertainty visible but quiet.
                ax.errorbar(
                    series["adversary_elo"],
                    series["adversary_utility"],
                    yerr=series["adversary_utility_sem"].clip(lower=0),
                    fmt="o-",
                    color=color,
                    ecolor=color,
                    markersize=6.8,
                    markeredgewidth=0,
                    linewidth=1.0,
                    elinewidth=0.9,
                    capsize=2.5,
                    capthick=0.9,
                    alpha=0.23,
                    zorder=2,
                )

                # The fitted relationship is the visual emphasis.
                fit_x, fit_y = linear_fit(series)
                ax.plot(
                    fit_x,
                    fit_y,
                    color=color,
                    linestyle="--",
                    linewidth=3.3,
                    alpha=0.98,
                    zorder=3,
                )

            if row == 0:
                ax.set_title(GAME_LABELS[game], fontsize=18, pad=11)
            if row == len(N_ORDER) - 1:
                ax.set_xlabel("Adversary Elo", fontsize=16, labelpad=8)
            else:
                ax.tick_params(axis="x", labelbottom=False)
            if col == 0:
                ax.set_ylabel(
                    f"n={n_agents}\nAdversary Payoff",
                    fontsize=16,
                    labelpad=14,
                )

            ax.tick_params(axis="both", which="major", labelsize=13.5, width=1.1, length=5)
            ax.grid(True, alpha=0.18, linewidth=0.8)
            for spine in ax.spines.values():
                spine.set_color("#555555")
                spine.set_linewidth(1.0)

    color_axis = fig.add_subplot(grid[:, 3])
    colorbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=color_axis)
    colorbar.ax.set_title("Competition\nIndex", fontsize=16, pad=12)
    colorbar.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    colorbar.ax.tick_params(labelsize=13.5, width=1.0, length=5)
    colorbar.outline.set_linewidth(0.9)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=DPI, facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.input.resolve())
    validate_data(frame)
    make_figure(frame, args.output.resolve())
    print(f"Loaded {len(frame)} plotted means from 1,300 raw homogeneous-adversary runs.")
    print(f"Saved Figure 24 to {args.output.resolve()}")


if __name__ == "__main__":
    main()
