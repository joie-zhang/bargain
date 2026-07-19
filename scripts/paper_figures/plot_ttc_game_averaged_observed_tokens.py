#!/usr/bin/env python3
"""Create game-averaged TTC observed-token summaries for Figure 6 iteration."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ORDER_AVG_CSV = PROJECT_ROOT / "analysis/neurips_revision_20260504/ttc_order_averaged.csv"
SUMMARY_CSV = PROJECT_ROOT / "analysis/neurips_revision_20260504/ttc_game_averaged_by_effort.csv"
PANEL_OUTPUT_PATH = (
    PROJECT_ROOT
    / "overleaf/neurips/graphics/ttc_game_averaged_target_payoff_vs_compute.png"
)
SINGLE_OUTPUT_PATH = (
    PROJECT_ROOT
    / "overleaf/neurips/graphics/ttc_game_averaged_target_payoff_vs_compute_single_panel.png"
)
PANEL_WITH_POINTS_OUTPUT_PATH = (
    PROJECT_ROOT
    / "overleaf/neurips/graphics/ttc_game_averaged_target_payoff_vs_compute_gray_points.png"
)

FAMILY_ORDER = ["gpt-5", "claude-sonnet-4-6", "gemini-3-flash"]
FAMILY_LABELS = {
    "gpt-5": "GPT-5",
    "claude-sonnet-4-6": "Claude Sonnet 4.6",
    "gemini-3-flash": "Gemini 3 Flash",
}
FAMILY_COLORS = {
    "gpt-5": "#2563eb",
    "claude-sonnet-4-6": "#dc2626",
    "gemini-3-flash": "#16a34a",
}
FAMILY_MARKERS = {
    "gpt-5": "o",
    "claude-sonnet-4-6": "s",
    "gemini-3-flash": "^",
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


def sem(series: pd.Series) -> float:
    values = series.dropna()
    if len(values) <= 1:
        return 0.0
    return float(values.std(ddof=1) / np.sqrt(len(values)))


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    order_avg = pd.read_csv(ORDER_AVG_CSV)
    order_avg = (
        order_avg.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["family", "level", "level_index", "game_cell", "target_compute_tokens_per_call", "target_utility"])
        .copy()
    )

    summary = (
        order_avg.groupby(["family", "provider", "level", "level_index"], dropna=False)
        .agg(
            game_cell_count=("game_cell", "nunique"),
            target_utility_mean=("target_utility", "mean"),
            target_utility_sem=("target_utility", sem),
            target_utility_sd=("target_utility", "std"),
            target_tokens_mean=("target_compute_tokens_per_call", "mean"),
            target_tokens_sem=("target_compute_tokens_per_call", sem),
            target_tokens_sd=("target_compute_tokens_per_call", "std"),
            consensus_rate_mean=("consensus_rate", "mean"),
            mean_round=("mean_round", "mean"),
        )
        .reset_index()
        .sort_values(["family", "level_index"])
    )
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_CSV, index=False)
    return order_avg, summary


def plot_panels(
    summary: pd.DataFrame,
    output_path: Path,
    order_avg: pd.DataFrame | None = None,
    *,
    x_label_override: str | None = None,
    y_label_override: str | None = None,
    title_fontsize: float = 24,
    axis_label_fontsize: float = 20,
    x_label_fontsize: float = 19,
    tick_label_fontsize: float = 15,
    x_label_y: float = 0.10,
    legend_y: float = -0.07,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.6, 4.9), sharex=False, sharey=True)
    fig.patch.set_facecolor("white")

    y_lim = (48, 82)
    if order_avg is not None:
        y_lim = (-5, 105)

    # Per-family x-range computed below from the family's own data.

    for ax, family in zip(axes, FAMILY_ORDER):
        family_df = summary[summary["family"].eq(family)].sort_values("level_index")
        if family_df.empty:
            continue

        if order_avg is not None:
            raw_family_df = order_avg[order_avg["family"].eq(family)]
            for effort in EFFORT_ORDER:
                raw_effort_df = raw_family_df[raw_family_df["level"].eq(effort)]
                if raw_effort_df.empty:
                    continue
                ax.scatter(
                    raw_effort_df["target_compute_tokens_per_call"],
                    raw_effort_df["target_utility"],
                    s=28,
                    color=EFFORT_COLORS[effort],
                    alpha=0.18,
                    edgecolor="none",
                    zorder=1,
                )

        ax.plot(
            family_df["target_tokens_mean"],
            family_df["target_utility_mean"],
            color="#475569",
            linewidth=2.55,
            alpha=0.60,
            zorder=2,
        )
        for _, row in family_df.iterrows():
            effort = str(row["level"])
            ax.errorbar(
                row["target_tokens_mean"],
                row["target_utility_mean"],
                yerr=row["target_utility_sem"],
                fmt=FAMILY_MARKERS[family],
                markersize=10.8,
                color=EFFORT_COLORS.get(effort, "#475569"),
                markeredgecolor="white",
                markeredgewidth=0.95,
                elinewidth=1.55,
                capsize=4.2,
                alpha=0.95,
                zorder=3,
            )

        ax.set_title(FAMILY_LABELS[family], fontsize=title_fontsize, pad=10)
        ax.tick_params(axis="both", labelsize=tick_label_fontsize)
        ax.grid(True, color="#d1d5db", alpha=0.42, linewidth=0.85)
        # Per-family x range: fit the family's own token range with a small margin.
        token_min = float(family_df["target_tokens_mean"].min())
        token_max = float(family_df["target_tokens_mean"].max())
        if order_avg is not None:
            raw_family_df = order_avg[order_avg["family"].eq(family)]
            if not raw_family_df.empty:
                token_min = min(token_min, float(raw_family_df["target_compute_tokens_per_call"].min()))
                token_max = max(token_max, float(raw_family_df["target_compute_tokens_per_call"].max()))
        token_span = max(token_max - token_min, 1.0)
        pad_left = max(token_span * 0.08, 50.0)
        pad_right = max(token_span * 0.10, 80.0)
        ax.set_xlim(max(token_min - pad_left, -70), token_max + pad_right)
        ax.set_ylim(*y_lim)

    y_label = y_label_override or ("Target payoff" if order_avg is not None else "Mean target payoff")
    x_label = x_label_override or (
        "Observed target tokens/call" if order_avg is not None else "Mean observed target tokens/call"
    )
    axes[0].set_ylabel(y_label, fontsize=axis_label_fontsize, labelpad=10)
    # Move x-axis title closer to the tick labels (was y=0.07) to free vertical space
    # between the title and the legend (which sits at bbox_to_anchor y=-0.07).
    fig.supxlabel(x_label, fontsize=x_label_fontsize, y=x_label_y)

    handles = [
        Line2D(
            [0],
            [0],
            color=EFFORT_COLORS[effort],
            marker="o",
            linestyle="",
            markersize=10.5,
            markeredgecolor="white",
            markeredgewidth=0.9,
            label=EFFORT_LABELS[effort],
        )
        for effort in EFFORT_ORDER
        if effort in set(summary["level"])
    ]
    legend = fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, legend_y),
        ncol=len(handles),
        title="Reasoning effort",
        title_fontsize=13.5,
        fontsize=13.2,
        frameon=True,
        facecolor="white",
        framealpha=0.94,
        columnspacing=1.0,
        handletextpad=0.45,
        borderpad=0.45,
    )
    legend.get_frame().set_edgecolor("#d1d5db")

    # Increase bottom margin slightly so the raised supxlabel and the legend both
    # have room (was bottom=0.25).
    fig.subplots_adjust(left=0.08, right=0.995, top=0.84, bottom=0.28, wspace=0.14)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=260, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def main() -> None:
    order_avg, summary = load_data()
    plot_panels(
        summary,
        PANEL_OUTPUT_PATH,
        x_label_override="Inference Tokens",
        y_label_override="Payoff",
    )
    plot_panels(summary, PANEL_WITH_POINTS_OUTPUT_PATH, order_avg)
    print(PANEL_OUTPUT_PATH)
    print(PANEL_WITH_POINTS_OUTPUT_PATH)
    print(SUMMARY_CSV)


if __name__ == "__main__":
    main()
