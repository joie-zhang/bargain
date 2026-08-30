#!/usr/bin/env python3
"""Render the top-only Figure 30 fleet-dilution diagnostic for the paper."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    REPO_ROOT
    / "reproduction_audit/fig30_multiagent_dilution/output/pooled_plot_points.csv"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "experiments/results/figure_iteration_20260802/multiagent/"
    / "hom_adversary_dilution_advantage_vs_n_top_only_clean.png"
)

GAME_ORDER = ("game1", "game2", "game3")
GAME_TITLES = {
    "game1": "Game 1: Item Allocation",
    "game2": "Game 2: Diplomatic Treaty",
    "game3": "Game 3: Co-Funding",
}
N_ORDER = (2, 4, 6, 8, 10)
MODEL_ORDER = (
    "amazon-nova-micro-v1.0",
    "claude-sonnet-4-20250514",
    "gemini-2.5-pro",
    "gpt-4o-mini-2024-07-18",
    "gpt-5.4-high",
)
MODEL_LABELS = {
    "amazon-nova-micro-v1.0": "Nova Micro",
    "claude-sonnet-4-20250514": "Sonnet 4",
    "gemini-2.5-pro": "Gemini 2.5 Pro",
    "gpt-4o-mini-2024-07-18": "GPT-4o mini",
    "gpt-5.4-high": "GPT-5.4 High",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_points(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "game_label",
        "n_agents",
        "adversary_model",
        "adversary_advantage",
        "adversary_advantage_sem",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")
    if len(frame) != 75:
        raise ValueError(f"Expected 75 pooled Figure 30 points; found {len(frame)}")
    if set(frame["game_label"].unique()) != set(GAME_ORDER):
        raise ValueError("Unexpected game coverage")
    if set(frame["n_agents"].unique()) != set(N_ORDER):
        raise ValueError("Unexpected group-size coverage")
    if set(frame["adversary_model"].unique()) != set(MODEL_ORDER):
        raise ValueError("Unexpected adversary-model coverage")
    return frame


def render(frame: pd.DataFrame, output: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.weight": "normal",
            "axes.titleweight": "normal",
            "axes.labelweight": "normal",
            "axes.linewidth": 1.2,
        }
    )

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(15.8, 5.6),
        dpi=220,
        sharey=True,
    )
    colors = {
        model: plt.cm.tab10(index)
        for index, model in enumerate(MODEL_ORDER)
    }

    for column, (ax, game) in enumerate(zip(axes, GAME_ORDER)):
        game_frame = frame[frame["game_label"].eq(game)]
        for model in MODEL_ORDER:
            series = game_frame[
                game_frame["adversary_model"].eq(model)
            ].sort_values("n_agents")
            ax.errorbar(
                series["n_agents"],
                series["adversary_advantage"],
                yerr=series["adversary_advantage_sem"].fillna(0.0).clip(lower=0.0),
                color=colors[model],
                marker="o",
                markersize=6.5,
                linewidth=2.2,
                elinewidth=1.3,
                capsize=3.8,
                capthick=1.2,
                alpha=0.92,
                label=MODEL_LABELS[model],
            )

        ax.axhline(0.0, color="#4b5563", linewidth=1.1)
        ax.set_title(GAME_TITLES[game], fontsize=22, fontweight="normal", pad=12)
        ax.set_xlabel("Number of Agents", fontsize=19, fontweight="normal", labelpad=9)
        ax.set_xticks(N_ORDER)
        ax.tick_params(axis="x", labelsize=16, width=1.1, length=5.0)
        ax.tick_params(axis="y", labelsize=16, width=1.1, length=5.0)
        ax.grid(True, color="#9ca3af", linewidth=0.75, alpha=0.22)
        ax.set_axisbelow(True)

        if column == 0:
            ax.set_ylabel(
                "Adversary Payoff Advantage",
                fontsize=19,
                fontweight="normal",
                labelpad=11,
            )
        else:
            ax.set_ylabel("")
            ax.tick_params(axis="y", which="both", left=False, labelleft=False)

    handles, labels = axes[0].get_legend_handles_labels()
    legend = figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.018),
        ncol=5,
        fontsize=16,
        frameon=False,
        columnspacing=1.7,
        handlelength=2.1,
        handletextpad=0.55,
    )
    for text in legend.get_texts():
        text.set_fontweight("normal")

    figure.subplots_adjust(left=0.075, right=0.99, top=0.89, bottom=0.265, wspace=0.12)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, facecolor="white", bbox_inches="tight", pad_inches=0.12)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    points = load_points(args.input.resolve())
    render(points, args.output.resolve())
    print(f"Rendered {len(points)} pooled Figure 30 points")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
