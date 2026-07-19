#!/usr/bin/env python3
"""Regenerate the main-text Figure 4 welfare-by-competition plot."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_CSV = PROJECT_ROOT / "experiments/results/n2_baseline_comparison_analysis_20260505/all_runs_with_metrics.csv"
OUTPUT_PATH = PROJECT_ROOT / "overleaf/neurips/graphics/n2_gpt5_nano/10_total_welfare_by_competition_ewma.png"
ATTAINABLE_CSV = (
    PROJECT_ROOT
    / "experiments/results/figure_iteration_20260507/gpt5_nano/"
    / "figure4_welfare_by_competition_attainable_welfare.csv"
)

GAME_ORDER = ["game1", "game2", "game3"]
GAME_LABELS = {
    "game1": "Game 1: Item Allocation",
    "game2": "Game 2: Diplomacy",
    "game3": "Game 3: Co-funding",
}
PALETTES = {
    "game1": {
        "c=0": "#9ecae1",
        "c=0.25": "#6baed6",
        "c=0.5": "#4292c6",
        "c=0.75": "#2171b5",
        "c=0.9": "#08519c",
        "c=0.95": "#084594",
        "c=1": "#08306b",
    },
    "game2": {
        "CI2=0": "#9ecae1",
        "CI2=0.25": "#6baed6",
        "CI2=0.5": "#2171b5",
        "CI2=1": "#08306b",
    },
    "game3": {
        "CI3=0": "#9ecae1",
        "CI3=0.2": "#6baed6",
        "CI3=0.4": "#2171b5",
        "CI3=0.8": "#08306b",
    },
}


def ewm(values: pd.Series, alpha: float = 0.24) -> pd.Series:
    return values.ewm(alpha=alpha, adjust=False).mean()


def label_order(df: pd.DataFrame, game_id: str) -> list[str]:
    labels = (
        df[df["game_id"].eq(game_id)][["competition_label", "competition_value"]]
        .drop_duplicates()
        .sort_values("competition_value")["competition_label"]
        .tolist()
    )
    return labels


def attainable_lines(df: pd.DataFrame) -> pd.DataFrame:
    positive = df[df["optimal_social_welfare"].fillna(0) > 0].copy()
    rows = (
        positive.groupby(["game_id", "competition_label", "competition_value"], as_index=False)
        .agg(
            attainable_welfare_line_max_positive_optimal=("optimal_social_welfare", "max"),
            mean_positive_optimal_welfare=("optimal_social_welfare", "mean"),
            n_positive_optimal_rows=("optimal_social_welfare", "size"),
        )
    )
    counts = (
        df.groupby(["game_id", "competition_label", "competition_value"], as_index=False)
        .agg(n_rows=("optimal_social_welfare", "size"))
    )
    return rows.merge(counts, on=["game_id", "competition_label", "competition_value"], how="left")


def main() -> None:
    df = pd.read_csv(INPUT_CSV)
    df = (
        df[df["baseline_key"].eq("gpt5_nano")]
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["adversary_elo", "payoff_social_welfare", "competition_label", "competition_value"])
        .copy()
    )
    lines = attainable_lines(df)
    ATTAINABLE_CSV.parent.mkdir(parents=True, exist_ok=True)
    lines.to_csv(ATTAINABLE_CSV, index=False)

    fig, axes = plt.subplots(1, 3, figsize=(19.8, 5.9), sharex=False, sharey=False)
    fig.patch.set_facecolor("white")

    for ax, game_id in zip(axes, GAME_ORDER):
        game_df = df[df["game_id"].eq(game_id)]
        game_lines = lines[lines["game_id"].eq(game_id)]
        labels = label_order(df, game_id)

        for label in labels:
            label_df = game_df[game_df["competition_label"].eq(label)].sort_values("adversary_elo")
            color = PALETTES[game_id].get(label, "#2171b5")
            per_elo = (
                label_df.groupby("adversary_elo", as_index=False)
                .agg(mean_welfare=("payoff_social_welfare", "mean"))
                .sort_values("adversary_elo")
            )
            if per_elo.empty:
                continue

            ax.scatter(
                label_df["adversary_elo"],
                label_df["payoff_social_welfare"],
                s=7,
                color=color,
                alpha=0.09,
                linewidths=0,
                zorder=1,
            )
            ax.plot(
                per_elo["adversary_elo"],
                per_elo["mean_welfare"],
                color=color,
                alpha=0.16,
                linewidth=1.0,
                zorder=2,
            )
            ax.plot(
                per_elo["adversary_elo"],
                ewm(per_elo["mean_welfare"]),
                marker="o",
                markersize=5.2,
                linewidth=2.35,
                color=color,
                markeredgecolor="white",
                markeredgewidth=0.75,
                label=label,
                zorder=4,
            )

            row = game_lines[game_lines["competition_label"].eq(label)]
            if not row.empty:
                ax.axhline(
                    float(row["attainable_welfare_line_max_positive_optimal"].iloc[0]),
                    color=color,
                    linestyle=(0, (4.0, 3.2)),
                    linewidth=1.25,
                    alpha=0.42,
                    zorder=0,
                )

        ax.set_title(GAME_LABELS[game_id], fontsize=27, pad=14)
        ax.set_xlabel("Adversary Elo", fontsize=24, labelpad=8)
        ax.tick_params(axis="both", labelsize=19)
        ax.xaxis.set_major_locator(MultipleLocator(100))
        ax.set_xlim(1085, 1515)
        ax.grid(True, color="#d1d5db", alpha=0.38, linewidth=0.85)
        if game_id == "game1":
            ax.set_ylabel("Total Welfare", fontsize=24, labelpad=10)
            ax.set_ylim(-4, 212)
            legend_loc = "lower right"
            ncol = 2
        elif game_id == "game2":
            ax.set_ylim(-4, 212)
            legend_loc = "lower right"
            ncol = 1
        else:
            ax.set_ylim(-30, 158)
            legend_loc = "upper left"
            ncol = 1
        legend = ax.legend(
            title="Competition",
            loc=legend_loc,
            ncol=ncol,
            fontsize=16.5,
            title_fontsize=17.5,
            frameon=True,
            facecolor="white",
            framealpha=0.92,
            handlelength=1.8,
            handletextpad=0.45,
            columnspacing=0.8,
            borderpad=0.55,
        )
        legend.get_frame().set_edgecolor("#d1d5db")

    fig.subplots_adjust(left=0.065, right=0.995, top=0.84, bottom=0.16, wspace=0.17)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=260, bbox_inches="tight")
    plt.close(fig)
    print(OUTPUT_PATH)
    print(ATTAINABLE_CSV)


if __name__ == "__main__":
    main()
