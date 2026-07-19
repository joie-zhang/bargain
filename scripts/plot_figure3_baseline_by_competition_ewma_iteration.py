#!/usr/bin/env python3
"""Iteration plots for paper Figure 3.

This creates the GPT-5-nano baseline-payoff-by-competition figure used in the
main text, plus an appendix companion that shows raw per-Elo means at the exact
competition settings. The main figure keeps only the smoothed competition-band
curves so it stays legible at NeurIPS column width.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator


PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_CSV = PROJECT_ROOT / "experiments/results/n2_baseline_comparison_analysis_20260505/all_runs_with_metrics.csv"
ITERATION_DIR = PROJECT_ROOT / "experiments/results/figure_iteration_20260507/gpt5_nano"
MAIN_OUTPUT_PATH = (
    PROJECT_ROOT
    / "overleaf/neurips/graphics/n2_gpt5_nano/04_baseline_payoff_by_competition.png"
)
ITERATION_OUTPUT_PATH = ITERATION_DIR / "figure3_baseline_payoff_by_competition_iteration.png"
APPENDIX_RAW_OUTPUT_PATH = (
    PROJECT_ROOT
    / "overleaf/neurips/graphics/n2_gpt5_nano/04_baseline_payoff_by_competition_raw_per_elo.png"
)

GAME_LABELS = {
    "game1": "Game 1: Item Allocation",
    "game2": "Game 2: Diplomacy",
    "game3": "Game 3: Co-funding",
}

COMPETITION_ORDER = ["0.0", "0.1-0.3", "0.4-0.6", "0.7-0.9", "1.0"]
TEAL_COLORS = {
    "0.0": "#164e4f",
    "0.1-0.3": "#146b6d",
    "0.4-0.6": "#00858c",
    "0.7-0.9": "#00a3ad",
    "1.0": "#16b8c7",
}
RAW_PALETTES = {
    "game1": {
        "c=0": "#083344",
        "c=0.25": "#155e75",
        "c=0.5": "#0e7490",
        "c=0.75": "#0891b2",
        "c=0.9": "#06b6d4",
        "c=0.95": "#22d3ee",
        "c=1": "#67e8f9",
    },
    "game2": {
        "CI2=0": "#083344",
        "CI2=0.25": "#0e7490",
        "CI2=0.5": "#06b6d4",
        "CI2=1": "#67e8f9",
    },
    "game3": {
        "CI3=0": "#083344",
        "CI3=0.2": "#0e7490",
        "CI3=0.4": "#06b6d4",
        "CI3=0.8": "#67e8f9",
    },
}


def competition_band(value: float) -> str:
    if np.isclose(value, 0.0):
        return "0.0"
    if value <= 0.3:
        return "0.1-0.3"
    if value <= 0.6:
        return "0.4-0.6"
    if value <= 0.9:
        return "0.7-0.9"
    return "1.0"


def ewm(values: pd.Series, alpha: float = 0.24) -> pd.Series:
    return values.ewm(alpha=alpha, adjust=False).mean()


def draw_main(df: pd.DataFrame) -> None:
    df["competition_band"] = df["competition_value"].astype(float).map(competition_band)

    fig, axes = plt.subplots(1, 3, figsize=(19.8, 5.7), sharex=False, sharey=False)
    fig.patch.set_facecolor("white")

    for ax, game_id in zip(axes, ["game1", "game2", "game3"]):
        game_df = (
            df[df["game_id"].eq(game_id)]
            .replace([np.inf, -np.inf], np.nan)
            .dropna(subset=["adversary_elo", "baseline_utility", "competition_band"])
            .copy()
        )

        for band in COMPETITION_ORDER:
            band_df = game_df[game_df["competition_band"].eq(band)].sort_values("adversary_elo")
            if band_df.empty:
                continue
            color = TEAL_COLORS[band]

            per_elo = (
                band_df.groupby("adversary_elo", as_index=False)
                .agg(mean_payoff=("baseline_utility", "mean"))
                .sort_values("adversary_elo")
            )

            per_elo["smoothed_payoff"] = ewm(per_elo["mean_payoff"])
            ax.plot(
                per_elo["adversary_elo"],
                per_elo["smoothed_payoff"],
                marker="o",
                markersize=6.0,
                linewidth=2.7,
                color=color,
                alpha=0.96,
                markeredgecolor="white",
                markeredgewidth=0.9,
                zorder=3,
                label=band,
            )

        ax.set_title(GAME_LABELS[game_id], fontsize=27, pad=14)
        ax.set_xlabel("Adversary Elo", fontsize=24, labelpad=8)
        if game_id == "game1":
            ax.set_ylabel("Baseline Model Payoff", fontsize=24, labelpad=10)
        else:
            ax.set_ylabel("")
        ax.grid(True, color="#d1d5db", alpha=0.42, linewidth=0.85)
        ax.tick_params(axis="both", labelsize=19)
        ax.xaxis.set_major_locator(MultipleLocator(100))
        ax.set_xlim(1085, 1515)
        ymin, ymax = ax.get_ylim()
        if game_id in {"game1", "game2"}:
            ax.set_ylim(max(-2, ymin), min(104, max(102, ymax)))
        else:
            ax.set_ylim(min(-16, ymin), max(86, ymax))
    legend_handles = [
        Line2D(
            [0],
            [0],
            color=TEAL_COLORS[band],
            marker="o",
            markersize=7.5,
            linewidth=3.0,
            markeredgecolor="white",
            markeredgewidth=0.9,
            label=band,
        )
        for band in COMPETITION_ORDER
    ]
    legend = fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.53, 0.0),
        ncol=5,
        fontsize=19,
        title="Competition",
        title_fontsize=19,
        frameon=True,
        facecolor="white",
        framealpha=0.94,
        borderpad=0.45,
        handlelength=1.55,
        handletextpad=0.45,
        columnspacing=0.75,
    )
    legend.get_frame().set_edgecolor("#d1d5db")

    fig.subplots_adjust(left=0.065, right=0.995, top=0.84, bottom=0.31, wspace=0.17)
    for path in [MAIN_OUTPUT_PATH, ITERATION_OUTPUT_PATH]:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=260, bbox_inches="tight")
        print(path)
    plt.close(fig)


def draw_appendix_raw(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(19.8, 5.7), sharex=False, sharey=False)
    fig.patch.set_facecolor("white")

    for ax, game_id in zip(axes, ["game1", "game2", "game3"]):
        game_df = (
            df[df["game_id"].eq(game_id)]
            .replace([np.inf, -np.inf], np.nan)
            .dropna(subset=["adversary_elo", "baseline_utility", "competition_label", "competition_value"])
            .copy()
        )
        labels = (
            game_df[["competition_label", "competition_value"]]
            .drop_duplicates()
            .sort_values("competition_value")["competition_label"]
            .tolist()
        )
        for label in labels:
            label_df = game_df[game_df["competition_label"].eq(label)]
            per_elo = (
                label_df.groupby("adversary_elo", as_index=False)
                .agg(mean_payoff=("baseline_utility", "mean"))
                .sort_values("adversary_elo")
            )
            color = RAW_PALETTES.get(game_id, {}).get(label, "#0e7490")
            ax.plot(
                per_elo["adversary_elo"],
                per_elo["mean_payoff"],
                marker="o",
                markersize=5.3,
                linewidth=1.5,
                color=color,
                alpha=0.72,
                markeredgecolor="white",
                markeredgewidth=0.65,
                label=label,
            )

        ax.set_title(GAME_LABELS[game_id], fontsize=26, pad=14)
        ax.set_xlabel("Adversary Elo", fontsize=23, labelpad=8)
        if game_id == "game1":
            ax.set_ylabel("Baseline Model Payoff", fontsize=23, labelpad=10)
        ax.grid(True, color="#d1d5db", alpha=0.36, linewidth=0.85)
        ax.tick_params(axis="both", labelsize=18)
        ax.xaxis.set_major_locator(MultipleLocator(100))
        ax.set_xlim(1085, 1515)
        ymin, ymax = ax.get_ylim()
        if game_id in {"game1", "game2"}:
            ax.set_ylim(max(-2, ymin), min(104, max(102, ymax)))
        else:
            ax.set_ylim(min(-16, ymin), max(86, ymax))
        ncol = 2 if len(labels) > 4 else 1
        legend = ax.legend(
            title="Competition",
            fontsize=13.5,
            title_fontsize=14.5,
            ncol=ncol,
            frameon=True,
            facecolor="white",
            framealpha=0.9,
            loc="best",
            handlelength=1.65,
            handletextpad=0.45,
            columnspacing=0.8,
            borderpad=0.45,
        )
        legend.get_frame().set_edgecolor("#d1d5db")

    fig.subplots_adjust(left=0.065, right=0.995, top=0.84, bottom=0.18, wspace=0.17)
    APPENDIX_RAW_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(APPENDIX_RAW_OUTPUT_PATH, dpi=260, bbox_inches="tight")
    plt.close(fig)
    print(APPENDIX_RAW_OUTPUT_PATH)


def main() -> None:
    df = pd.read_csv(INPUT_CSV)
    df = df[df["baseline_key"].eq("gpt5_nano")].copy()
    draw_main(df)
    draw_appendix_raw(df)


if __name__ == "__main__":
    main()
