#!/usr/bin/env python3
"""Render the cleaned one-row ICML appendix rounds-to-consensus figure.

The input is the raw-derived primary-run table produced by
``scripts/analyze_n2_baseline_comparison.py``. Each panel aggregates rounds to
consensus by adversary model within one game.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    PROJECT_ROOT
    / "experiments/results/n2_baseline_comparison_analysis_20260505/primary_runs_with_metrics.csv"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "experiments/results/n2_baseline_comparison_analysis_20260505/gpt5_nano/"
    / "07_08_rounds_to_consensus_combined.png"
)

GAME_ORDER = ("game1", "game2", "game3")
GAME_TITLES = {
    "game1": "Game 1: Item Allocation",
    "game2": "Game 2: Diplomatic Treaty",
    "game3": "Game 3: Co-Funding",
}
POINT_COLOR = "#0f766e"
FIT_COLOR = "#1f2937"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_gpt5_rounds(input_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(input_path)
    required = {
        "baseline_key",
        "game_id",
        "competition_value",
        "competition_label",
        "adversary_model",
        "adversary_elo",
        "rounds_to_consensus",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns in {input_path}: {missing}")

    frame = frame[frame["baseline_key"].eq("gpt5_nano")].copy()
    if len(frame) != 1500:
        raise ValueError(f"Expected 1,500 GPT-5-nano primary runs, found {len(frame)}")
    return frame


def aggregate_rounds(frame: pd.DataFrame) -> pd.DataFrame:
    overall = (
        frame.groupby(["game_id", "adversary_model", "adversary_elo"], as_index=False)
        .agg(mean_rounds=("rounds_to_consensus", "mean"))
        .sort_values(["game_id", "adversary_elo"])
    )
    return overall


def add_trend(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    *,
    color: object,
    linewidth: float,
    label: str | None = None,
) -> None:
    finite = np.isfinite(x) & np.isfinite(y)
    x_clean = x[finite]
    y_clean = y[finite]
    if x_clean.size < 2 or np.unique(x_clean).size < 2:
        return
    slope, intercept = np.polyfit(x_clean, y_clean, deg=1)
    trend_x = np.linspace(float(x_clean.min()), float(x_clean.max()), 200)
    ax.plot(
        trend_x,
        slope * trend_x + intercept,
        color=color,
        linestyle="--",
        linewidth=linewidth,
        alpha=0.98,
        label=label,
        zorder=3,
    )


def style_axis(ax: plt.Axes, *, show_y_label: bool) -> None:
    ax.set_ylabel("Rounds to Consensus" if show_y_label else "", fontsize=18, labelpad=9)
    ax.tick_params(axis="y", which="major", labelsize=15.5, width=1.1)
    ax.set_xlabel("Adversary Elo", fontsize=18, labelpad=8)
    ax.tick_params(axis="x", which="major", labelsize=15.5, width=1.1)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    ax.grid(alpha=0.18, linewidth=0.9)
    ax.margins(x=0.035)


def render(overall: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.45), sharey=False, constrained_layout=True)

    for panel_index, game_id in enumerate(GAME_ORDER):
        ax = axes[panel_index]

        game_overall = overall[overall["game_id"].eq(game_id)].sort_values("adversary_elo")
        defined_overall = game_overall.dropna(subset=["adversary_elo", "mean_rounds"])
        ax.scatter(
            defined_overall["adversary_elo"],
            defined_overall["mean_rounds"],
            s=56,
            color=POINT_COLOR,
            alpha=0.82,
            edgecolors="none",
            zorder=2,
        )
        add_trend(
            ax,
            game_overall["adversary_elo"].to_numpy(dtype=float),
            game_overall["mean_rounds"].to_numpy(dtype=float),
            color=FIT_COLOR,
            linewidth=2.5,
        )
        ax.set_title(
            GAME_TITLES[game_id],
            fontsize=24,
            pad=13,
        )
        style_axis(ax, show_y_label=panel_index == 0)

    fig.set_constrained_layout_pads(w_pad=0.025, h_pad=0.025, wspace=0.035, hspace=0.0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    frame = load_gpt5_rounds(input_path)
    overall = aggregate_rounds(frame)
    render(overall, output_path)
    print(f"Wrote {output_path}")
    print(
        f"Aggregated {len(overall)} overall rows from {len(frame)} primary runs"
    )


if __name__ == "__main__":
    main()
