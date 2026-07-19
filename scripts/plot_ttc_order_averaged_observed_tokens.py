#!/usr/bin/env python3
"""Create the order-averaged TTC observed-token scatter for the main text."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_CSV = (
    PROJECT_ROOT
    / "experiments/results/ttc_native_scaling_20260502_212943/monitoring/partial_results_latest.csv"
)
ORDER_AVG_CSV = PROJECT_ROOT / "analysis/neurips_revision_20260504/ttc_order_averaged.csv"
OUTPUT_PATH = PROJECT_ROOT / "overleaf/neurips/graphics/ttc_order_averaged_target_payoff_vs_compute.png"

GAME_ORDER = ["game1", "game2", "game3"]
GAME_LABELS = {
    "game1": "Item allocation",
    "game2": "Diplomacy",
    "game3": "Co-funding",
}
FAMILY_ORDER = ["gpt-5", "claude-sonnet-4-6", "gemini-3-flash"]
FAMILY_LABELS = {
    "gpt-5": "GPT-5",
    "claude-sonnet-4-6": "Claude Sonnet 4.6",
    "gemini-3-flash": "Gemini 3 Flash",
}
EFFORT_ORDER = ["minimal", "low", "medium", "high", "max"]
EFFORT_LABELS = {
    "minimal": "Minimal",
    "low": "Low",
    "medium": "Medium",
    "high": "High",
    "max": "Max",
}
EFFORT_COLORS = {
    "minimal": "#64748b",
    "low": "#2563eb",
    "medium": "#0f766e",
    "high": "#f97316",
    "max": "#7c3aed",
}
GAME_MARKERS = {
    "game1": "o",
    "game2": "s",
    "game3": "^",
}


def load_order_average() -> pd.DataFrame:
    if ORDER_AVG_CSV.exists():
        return pd.read_csv(ORDER_AVG_CSV)

    df = pd.read_csv(RESULTS_CSV)
    order_avg = (
        df.groupby(["family", "provider", "level", "level_index", "game", "game_cell"], dropna=False)
        .agg(
            order_count=("order", "nunique"),
            run_count=("config_id", "size"),
            target_utility=("target_utility", "mean"),
            baseline_utility=("baseline_utility", "mean"),
            utility_gap=("utility_gap", "mean"),
            target_compute_tokens_per_call=("target_compute_tokens_per_call", "mean"),
            target_output_tokens_per_call=("target_output_tokens_per_call", "mean"),
            target_reasoning_tokens_raw_per_call=("target_reasoning_tokens_raw_per_call", "mean"),
            consensus_rate=("consensus", "mean"),
            mean_round=("round", "mean"),
        )
        .reset_index()
        .sort_values(["family", "game_cell", "level_index"])
    )
    ORDER_AVG_CSV.parent.mkdir(parents=True, exist_ok=True)
    order_avg.to_csv(ORDER_AVG_CSV, index=False)
    return order_avg


def main() -> None:
    order_avg = load_order_average()
    order_avg = (
        order_avg.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["game", "family", "target_compute_tokens_per_call", "target_utility"])
        .copy()
    )

    fig, axes = plt.subplots(1, 3, figsize=(15.8, 5.15), sharex=True, sharey=True)
    fig.patch.set_facecolor("white")

    x_max = max(4000, float(order_avg["target_compute_tokens_per_call"].max()) * 1.06)

    for ax, family in zip(axes, FAMILY_ORDER):
        family_df = order_avg[order_avg["family"].eq(family)].copy()
        if family_df.empty:
            continue

        for _, cell_df in family_df.groupby("game_cell", sort=False):
            cell_df = cell_df.sort_values("level_index")
            if len(cell_df) > 1:
                ax.plot(
                    cell_df["target_compute_tokens_per_call"],
                    cell_df["target_utility"],
                    color="#94a3b8",
                    alpha=0.18,
                    linewidth=1.2,
                    zorder=1,
                )

        for game in GAME_ORDER:
            game_df = family_df[family_df["game"].eq(game)]
            if game_df.empty:
                continue
            point_colors = [EFFORT_COLORS.get(level, "#111827") for level in game_df["level"]]
            ax.scatter(
                game_df["target_compute_tokens_per_call"],
                game_df["target_utility"],
                s=132,
                marker=GAME_MARKERS[game],
                c=point_colors,
                alpha=0.86,
                edgecolor="white",
                linewidth=0.95,
                zorder=3,
            )

        ax.set_title(FAMILY_LABELS[family], fontsize=25, pad=10)
        ax.tick_params(axis="both", labelsize=16)
        ax.grid(True, color="#d1d5db", alpha=0.42, linewidth=0.85)
        ax.set_xlim(-95, x_max)
        ax.set_ylim(-5, 105)

    axes[0].set_ylabel("Target payoff", fontsize=21, labelpad=12)
    fig.supxlabel("Observed target tokens/call", fontsize=20, y=0.20)
    effort_handles = [
        Line2D(
            [0],
            [0],
            color=EFFORT_COLORS[effort],
            marker="o",
            linestyle="",
            markersize=10.8,
            markeredgecolor="white",
            markeredgewidth=0.9,
            label=EFFORT_LABELS[effort],
        )
        for effort in EFFORT_ORDER
    ]
    game_handles = [
        Line2D(
            [0],
            [0],
            color="#334155",
            marker=GAME_MARKERS[game],
            linestyle="",
            markersize=10.8,
            markeredgecolor="white",
            markeredgewidth=0.9,
            label=GAME_LABELS[game],
        )
        for game in GAME_ORDER
    ]
    effort_legend = fig.legend(
        handles=effort_handles,
        loc="lower center",
        bbox_to_anchor=(0.37, 0.005),
        ncol=len(effort_handles),
        title="Reasoning effort",
        title_fontsize=14.5,
        fontsize=14.2,
        frameon=True,
        facecolor="white",
        framealpha=0.94,
        columnspacing=1.05,
        handletextpad=0.45,
        borderpad=0.45,
    )
    effort_legend.get_frame().set_edgecolor("#d1d5db")

    game_legend = fig.legend(
        handles=game_handles,
        loc="lower center",
        bbox_to_anchor=(0.78, 0.005),
        ncol=len(game_handles),
        title="Game",
        title_fontsize=14.5,
        fontsize=14.2,
        frameon=True,
        facecolor="white",
        framealpha=0.94,
        columnspacing=1.05,
        handletextpad=0.45,
        borderpad=0.45,
    )
    game_legend.get_frame().set_edgecolor("#d1d5db")

    fig.subplots_adjust(left=0.088, right=0.995, top=0.84, bottom=0.34, wspace=0.11)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=260, bbox_inches="tight", pad_inches=0.16)
    plt.close(fig)
    print(OUTPUT_PATH)
    print(ORDER_AVG_CSV)


if __name__ == "__main__":
    main()
