#!/usr/bin/env python3
"""Render Figure 2 candidate with much larger plot text.

This is intentionally a standalone candidate renderer. It does not update the
paper asset.
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


ROOT = Path(__file__).resolve().parents[2]
INPUT_CSV = ROOT / "experiments/results/n2_baseline_comparison_analysis_20260505/all_runs_with_metrics.csv"
OUT_PATH = ROOT / "analysis/recreated_figures/figure2_bilateral_overview_combined_large_fonts.png"

GAME_ORDER = ["game1", "game2", "game3"]
LEFT_LABELS = {
    "game1": "Game 1: Item allocation",
    "game2": "Game 2: Diplomatic Treaty",
    "game3": "Game 3: Co-funding",
}
RIGHT_TITLES = {
    "game1": "G1: Item allocation",
    "game2": "G2: Diplomatic Treaty",
    "game3": "G3: Co-funding",
}
COLORS = {
    "game1": "#2b7bba",
    "game2": "#e63946",
    "game3": "#2ca02c",
    "cooperative": "#0b4f63",
    "competitive": "#48c7df",
}


def sem(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if len(clean) <= 1:
        return 0.0
    return float(clean.std(ddof=1) / np.sqrt(len(clean)))


def fit_line(x: pd.Series, y: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    x_arr = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    y_arr = pd.to_numeric(y, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    if mask.sum() < 2:
        return np.array([]), np.array([])
    coef = np.polyfit(x_arr[mask], y_arr[mask], deg=1)
    xs = np.linspace(float(x_arr[mask].min()), float(x_arr[mask].max()), 100)
    ys = coef[0] * xs + coef[1]
    return xs, ys


def style_axis(
    ax: plt.Axes,
    *,
    title: str | None,
    xlabel: str,
    ylabel: str | None,
    title_size: int = 20,
    xlabel_size: int = 17,
    ylabel_size: int = 18,
    tick_size: int = 13,
    sparse_x_ticks: bool = False,
) -> None:
    if title:
        ax.set_title(title, fontsize=title_size, pad=9)
    ax.set_xlabel(xlabel, fontsize=xlabel_size, labelpad=6)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=ylabel_size, labelpad=8)
    ax.tick_params(axis="both", labelsize=tick_size, length=4.5, width=0.8)
    ax.grid(True, color="#d1d5db", alpha=0.52, linewidth=0.75)
    if sparse_x_ticks:
        ax.set_xticks([1100, 1300, 1500])
    else:
        ax.xaxis.set_major_locator(MultipleLocator(100))
    ax.set_xlim(1088, 1512)
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)


def draw_left(ax: plt.Axes, df: pd.DataFrame) -> list[Line2D]:
    handles: list[Line2D] = []
    for game_id in GAME_ORDER:
        game_df = df[df["game_id"].eq(game_id)]
        agg = (
            game_df.groupby(["adversary_model", "adversary_elo"], as_index=False)
            .agg(mean=("adversary_utility", "mean"), err=("adversary_utility", sem))
            .sort_values("adversary_elo")
        )
        color = COLORS[game_id]
        ax.errorbar(
            agg["adversary_elo"],
            agg["mean"],
            yerr=agg["err"],
            fmt="o",
            markersize=4.7,
            markerfacecolor="white",
            markeredgewidth=1.05,
            color=color,
            ecolor=color,
            elinewidth=0.9,
            capsize=2.2,
            alpha=0.72,
            zorder=3,
        )
        xs, ys = fit_line(agg["adversary_elo"], agg["mean"])
        if len(xs):
            ax.plot(xs, ys, color=color, linestyle="--", linewidth=2.8, zorder=4)
        handles.append(Line2D([0], [0], color=color, linestyle="--", linewidth=3.0, label=LEFT_LABELS[game_id]))

    ax.set_ylim(-10, 102)
    style_axis(
        ax,
        title=None,
        xlabel="Adversary Elo",
        ylabel="Adversary payoff",
        xlabel_size=18,
        ylabel_size=18,
        tick_size=14,
    )
    return handles


def draw_right(ax: plt.Axes, df: pd.DataFrame, game_id: str) -> None:
    game_df = df[df["game_id"].eq(game_id)].copy()
    comp_min = float(game_df["competition_value"].min())
    comp_max = float(game_df["competition_value"].max())
    series = [
        (comp_min, "Max Cooperative", COLORS["cooperative"]),
        (comp_max, "Max Competitive", COLORS["competitive"]),
    ]
    for comp_value, label, color in series:
        sub = game_df[np.isclose(game_df["competition_value"].astype(float), comp_value)]
        per_elo = (
            sub.groupby("adversary_elo", as_index=False)
            .agg(mean=("baseline_utility", "mean"), err=("baseline_utility", sem))
            .sort_values("adversary_elo")
        )
        per_elo["smooth"] = per_elo["mean"].ewm(alpha=0.24, adjust=False).mean()
        x = per_elo["adversary_elo"].to_numpy(dtype=float)
        y = per_elo["smooth"].to_numpy(dtype=float)
        err = per_elo["err"].to_numpy(dtype=float)
        ax.fill_between(x, y - err, y + err, color=color, alpha=0.17, linewidth=0, zorder=1)
        ax.plot(
            x,
            y,
            color=color,
            marker="o",
            markersize=4.2,
            linewidth=2.0,
            markeredgecolor="white",
            markeredgewidth=0.55,
            alpha=0.96,
            label=label,
            zorder=3,
        )

    style_axis(
        ax,
            title=RIGHT_TITLES[game_id],
            xlabel="Adversary Elo",
            ylabel="Baseline payoff" if game_id == "game1" else None,
            title_size=15,
            xlabel_size=15,
            ylabel_size=18,
            tick_size=12,
            sparse_x_ticks=True,
    )
    if game_id in {"game1", "game2"}:
        ax.set_ylim(-5, 105)
    else:
        ax.set_ylim(-5, 65)
    if game_id != "game1":
        ax.tick_params(labelleft=False)


def main() -> None:
    df = pd.read_csv(INPUT_CSV)
    df = df[df["baseline_key"].eq("gpt5_nano")].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["game_id", "adversary_elo", "adversary_utility", "baseline_utility", "competition_value"])

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.unicode_minus": False,
            "mathtext.fontset": "dejavusans",
        }
    )

    fig = plt.figure(figsize=(14.84, 5.68), dpi=300)
    plot_bottom = 0.295
    plot_height = 0.56
    lhs_width = plot_height * (5.68 / 14.84)
    left_ax = fig.add_axes([0.18, plot_bottom, lhs_width, plot_height])
    left_handles = draw_left(left_ax, df)

    rhs_width = 0.128
    rhs_gap = 0.03
    rhs_start = 0.49
    right_axes = [
        fig.add_axes([rhs_start + idx * (rhs_width + rhs_gap), plot_bottom, rhs_width, plot_height])
        for idx in range(3)
    ]
    for ax, game_id in zip(right_axes, GAME_ORDER, strict=True):
        draw_right(ax, df, game_id)

    left_legend = fig.legend(
        handles=left_handles,
        loc="lower center",
        bbox_to_anchor=(0.325, 0.075),
        ncol=3,
        fontsize=14.0,
        frameon=False,
        handlelength=1.65,
        columnspacing=1.35,
        handletextpad=0.46,
    )
    for text in left_legend.get_texts():
        text.set_va("center")

    right_handles = [
        Line2D([0], [0], color=COLORS["cooperative"], marker="o", markersize=4.6, linewidth=2.1, label="Max Cooperative"),
        Line2D([0], [0], color=COLORS["competitive"], marker="o", markersize=4.6, linewidth=2.1, label="Max Competitive"),
    ]
    fig.legend(
        handles=right_handles,
        loc="lower left",
        bbox_to_anchor=(0.645, 0.075),
        ncol=2,
        fontsize=14.0,
        frameon=False,
        handlelength=1.65,
        columnspacing=1.08,
        handletextpad=0.46,
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=300, facecolor="white")
    plt.close(fig)
    print(OUT_PATH)


if __name__ == "__main__":
    main()
