#!/usr/bin/env python3
"""Render the paper's Figure 31 from its archived group-size summary table."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    PROJECT_ROOT
    / "experiments/results/n2_plus_multiagent_comparison_analysis_20260505"
    / "tables_multiagent/general_n_dynamics_summary.csv"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "experiments/results/n2_plus_multiagent_comparison_analysis_20260505"
    / "plots_multiagent/general_shifted_gini_vs_n.png"
)

GAME_ORDER = ["game1", "game2", "game3"]
GAME_TITLES = {
    "game1": "Game 1: Item Allocation",
    "game2": "Game 2: Diplomatic Treaty",
    "game3": "Game 3: Co-Funding",
}
FAMILY_ORDER = [
    "homogeneous_control",
    "homogeneous_adversary",
    "heterogeneous_random",
]
FAMILY_LABELS = {
    "homogeneous_control": "Homogeneous control",
    "homogeneous_adversary": "Homogeneous adversary",
    "heterogeneous_random": "Heterogeneous",
}
FAMILY_COLORS = {
    "homogeneous_control": "#1f77b4",
    "homogeneous_adversary": "#d62728",
    "heterogeneous_random": "#2ca02c",
}
N_ORDER = [2, 4, 6, 8, 10]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_summary(path: Path) -> pd.DataFrame:
    summary = pd.read_csv(path)
    required = {
        "game_label",
        "n_agents",
        "experiment_family",
        "shifted_gini",
        "shifted_gini_sem",
    }
    missing = sorted(required.difference(summary.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    selected = summary[
        summary["game_label"].isin(GAME_ORDER)
        & summary["n_agents"].isin(N_ORDER)
        & summary["experiment_family"].isin(FAMILY_ORDER)
    ].copy()
    expected_rows = len(GAME_ORDER) * len(N_ORDER) * len(FAMILY_ORDER)
    if len(selected) != expected_rows:
        raise ValueError(f"Expected {expected_rows} Figure 31 cells, found {len(selected)}")
    return selected


def render(summary: pd.DataFrame, output: Path) -> None:
    plt.rcParams.update(
        {
            "font.size": 15,
            "font.weight": "normal",
            "axes.titleweight": "normal",
            "axes.labelweight": "normal",
            "axes.titlesize": 19,
            "axes.labelsize": 17,
            "xtick.labelsize": 14.5,
            "ytick.labelsize": 14.5,
            "legend.fontsize": 16,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(15.4, 5.25), sharey=False)

    for ax, game in zip(axes, GAME_ORDER, strict=True):
        game_df = summary[summary["game_label"].eq(game)]
        for family in FAMILY_ORDER:
            series = game_df[game_df["experiment_family"].eq(family)].sort_values("n_agents")
            ax.errorbar(
                series["n_agents"],
                series["shifted_gini"],
                yerr=series["shifted_gini_sem"],
                color=FAMILY_COLORS[family],
                label=FAMILY_LABELS[family],
                marker="o",
                markersize=7.5,
                linewidth=2.0,
                elinewidth=1.5,
                capsize=4.5,
                capthick=1.4,
            )
        ax.set_title(GAME_TITLES[game], pad=10, fontweight="normal")
        ax.set_xlabel("N agents", labelpad=7, fontweight="normal")
        ax.set_ylabel("Shifted utility Gini", labelpad=8, fontweight="normal")
        ax.set_xticks(N_ORDER)
        ax.tick_params(axis="both", which="major", width=1.0, length=5)
        ax.grid(True, alpha=0.22)

    handles, labels = axes[-1].get_legend_handles_labels()
    legend = fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=3,
        frameon=False,
        prop={"size": 16, "weight": "normal"},
        handlelength=2.5,
        columnspacing=2.4,
    )
    fig.subplots_adjust(left=0.065, right=0.99, top=0.90, bottom=0.24, wspace=0.23)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight", bbox_extra_artists=(legend,))
    plt.close(fig)


def main() -> None:
    args = parse_args()
    summary = load_summary(args.input)
    render(summary, args.output)
    print(f"Rendered {len(summary)} cells to {args.output.resolve()}")


if __name__ == "__main__":
    main()
