#!/usr/bin/env python3
"""Render the ICML Figure 16 Llama-baseline result as a compact 1x3 overlay."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    PROJECT_ROOT
    / "experiments/results/appendix_llama33_baseline_analysis_20260503/overall_by_model_game.csv"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "overleaf/icml_aiwild_template/graphics/appendix_llama/"
    / "llama33_overall_utility_overlay_1x3.png"
)

ADVERSARY_COLOR = "#b45309"
BASELINE_COLOR = "#2563eb"
GAME_ORDER = ("game1", "game2", "game3")
GAME_TITLES = {"game1": "Game 1", "game2": "Game 2", "game3": "Game 3"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def add_fit(ax: plt.Axes, x: pd.Series, y: pd.Series, color: str) -> None:
    clean = pd.DataFrame({"x": x, "y": y}).dropna()
    slope, intercept = np.polyfit(clean["x"], clean["y"], deg=1)
    fit_x = np.linspace(float(clean["x"].min()), float(clean["x"].max()), 200)
    ax.plot(
        fit_x,
        slope * fit_x + intercept,
        color=color,
        linestyle="--",
        linewidth=1.45,
        alpha=0.9,
        zorder=2,
    )


def validate(frame: pd.DataFrame) -> None:
    required = {
        "game_id",
        "adversary_elo",
        "adversary_utility_mean",
        "baseline_utility_mean",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    counts = frame.groupby("game_id").size().to_dict()
    expected = {game_id: 10 for game_id in GAME_ORDER}
    if counts != expected:
        raise ValueError(f"Expected ten model means per game, found {counts}")


def render(frame: pd.DataFrame, output: Path) -> None:
    plt.rcParams.update(
        {
            "font.size": 15.5,
            "axes.titlesize": 18,
            "axes.labelsize": 17,
            "xtick.labelsize": 14.5,
            "ytick.labelsize": 14.5,
            "legend.fontsize": 14.5,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.25), sharex=True, sharey=False)

    for ax, game_id in zip(axes, GAME_ORDER, strict=True):
        game = frame[frame["game_id"].eq(game_id)].sort_values("adversary_elo")
        x = game["adversary_elo"]
        adversary = game["adversary_utility_mean"]
        baseline = game["baseline_utility_mean"]

        ax.plot(
            x,
            adversary,
            color=ADVERSARY_COLOR,
            marker="o",
            markersize=7.2,
            markeredgecolor="white",
            markeredgewidth=0.65,
            linewidth=2.0,
            zorder=4,
        )
        ax.plot(
            x,
            baseline,
            color=BASELINE_COLOR,
            marker="o",
            markersize=7.2,
            markeredgecolor="white",
            markeredgewidth=0.65,
            linewidth=2.0,
            zorder=4,
        )
        add_fit(ax, x, adversary, ADVERSARY_COLOR)
        add_fit(ax, x, baseline, BASELINE_COLOR)

        combined = pd.concat([adversary, baseline], ignore_index=True)
        span = float(combined.max() - combined.min())
        padding = max(1.0, 0.09 * span)
        ax.set_ylim(float(combined.min()) - padding, float(combined.max()) + padding)
        ax.set_xlim(float(x.min()) - 8, float(x.max()) + 8)
        ax.set_title(GAME_TITLES[game_id], fontweight="normal", pad=4)
        ax.tick_params(axis="both", pad=2)
        ax.grid(alpha=0.22, linewidth=0.8)
        ax.set_axisbelow(True)

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=ADVERSARY_COLOR,
            marker="o",
            markersize=7.2,
            linewidth=2.0,
            label="Adversary utility",
        ),
        Line2D(
            [0],
            [0],
            color=BASELINE_COLOR,
            marker="o",
            markersize=7.2,
            linewidth=2.0,
            label="Llama 3.3 70B baseline utility",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=2,
        frameon=False,
        handlelength=2.5,
        columnspacing=2.2,
    )
    fig.supxlabel("Adversary Elo", y=0.065, fontweight="normal")
    fig.supylabel("Mean Utility", x=0.022, fontweight="normal")
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.20, top=0.74, wspace=0.30)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    frame = pd.read_csv(input_path)
    validate(frame)
    render(frame, output_path)
    print(f"Read: {input_path}")
    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()
