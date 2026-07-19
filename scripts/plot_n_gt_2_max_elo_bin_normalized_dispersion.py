#!/usr/bin/env python3
"""Range-normalized dispersion plots for the max-Elo-bin analysis."""

from __future__ import annotations

import math
import textwrap
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "overleaf/neurips/graphics/n_gt_2_report"
RUN_METRICS = OUT_DIR / "heterogeneous_max_elo_bin_variance_selected_run_metrics.csv"
BIN_TABLE = OUT_DIR / "heterogeneous_max_elo_bin_variance_selected_bins.csv"
AGENT_TABLE = (
    PROJECT_ROOT
    / "experiments/results/n2_plus_multiagent_comparison_analysis_20260505"
    / "tables_multiagent/heterogeneous_agents_fresh.csv"
)

N_ORDER = [2, 4, 6, 8, 10]
GAME_ORDER = ["game1", "game2", "game3"]
GAME_LABELS = {"game1": "Game 1", "game2": "Game 2", "game3": "Game 3"}
COMPETITION_ORDER = ["cooperative", "middle", "competitive"]
COMPETITION_LABELS = {
    "cooperative": "Cooperative",
    "middle": "Middle",
    "competitive": "Competitive",
}
COMPETITION_COLORS = {
    "cooperative": "#2F6FED",
    "middle": "#8C8C8C",
    "competitive": "#D62728",
}
N_COLORS = {
    2: "#4E79A7",
    4: "#F28E2B",
    6: "#59A14F",
    8: "#B07AA1",
    10: "#E15759",
}
BIN_COLORS = ["#386CB0", "#7FC97F", "#F0027F"]

PAYOFF_RANGE = 100.0
POINT_ALPHA = 0.16
POINT_SIZE = 18
LINE_WIDTH = 2.6
Y_LIMIT_QUANTILE = 0.97


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    runs = pd.read_csv(RUN_METRICS)
    bins = pd.read_csv(BIN_TABLE)
    agents = pd.read_csv(AGENT_TABLE)
    roster_elos = pd.to_numeric(agents["elo"], errors="coerce").dropna().drop_duplicates()
    elo_min = float(roster_elos.min())
    elo_max = float(roster_elos.max())
    elo_range = elo_max - elo_min

    runs["elo_std"] = np.sqrt(runs["elo_variance"].clip(lower=0))
    runs["payoff_std"] = np.sqrt(runs["payoff_variance"].clip(lower=0))
    runs["elo_dispersion_norm"] = runs["elo_std"] / elo_range
    runs["payoff_dispersion_norm"] = runs["payoff_std"] / PAYOFF_RANGE
    metadata = {
        "elo_min": elo_min,
        "elo_max": elo_max,
        "elo_range": elo_range,
        "payoff_range": PAYOFF_RANGE,
    }

    runs.to_csv(OUT_DIR / "heterogeneous_max_elo_bin_normalized_dispersion_run_metrics.csv", index=False)
    pd.DataFrame([metadata]).to_csv(
        OUT_DIR / "heterogeneous_max_elo_bin_normalized_dispersion_metadata.csv",
        index=False,
    )
    return runs, bins, metadata


def fit_line(frame: pd.DataFrame) -> dict[str, float]:
    data = frame[["elo_dispersion_norm", "payoff_dispersion_norm"]].replace([np.inf, -np.inf], np.nan).dropna()
    if (
        len(data) < 3
        or data["elo_dispersion_norm"].nunique() < 2
        or data["payoff_dispersion_norm"].nunique() < 2
    ):
        return {
            "slope": math.nan,
            "intercept": math.nan,
            "slope_stderr": math.nan,
            "p_value": math.nan,
            "pearson_r": math.nan,
            "r_squared": math.nan,
        }
    result = stats.linregress(
        data["elo_dispersion_norm"].to_numpy(float),
        data["payoff_dispersion_norm"].to_numpy(float),
    )
    return {
        "slope": float(result.slope),
        "intercept": float(result.intercept),
        "slope_stderr": float(result.stderr),
        "p_value": float(result.pvalue),
        "pearson_r": float(result.rvalue),
        "r_squared": float(result.rvalue * result.rvalue),
    }


def fit_summary(frame: pd.DataFrame, group_cols: list[str], scope: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, sub in frame.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        fit = fit_line(sub)
        row = dict(zip(group_cols, keys, strict=True))
        row.update(
            {
                "scope": scope,
                "n_runs": len(sub),
                "elo_dispersion_norm_min": float(sub["elo_dispersion_norm"].min()),
                "elo_dispersion_norm_max": float(sub["elo_dispersion_norm"].max()),
                "payoff_dispersion_norm_min": float(sub["payoff_dispersion_norm"].min()),
                "payoff_dispersion_norm_max": float(sub["payoff_dispersion_norm"].max()),
                "slope_payoff_pp_per_elo_pp": fit["slope"],
                "slope_stderr": fit["slope_stderr"],
                "p_value": fit["p_value"],
                "intercept": fit["intercept"],
                "pearson_r": fit["pearson_r"],
                "r_squared": fit["r_squared"],
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def trend_y_lim(frame: pd.DataFrame) -> tuple[float, float]:
    fit = fit_line(frame)
    fitted_max = 0.0
    if math.isfinite(fit["slope"]) and len(frame) > 0:
        xs = [
            float(frame["elo_dispersion_norm"].min()),
            float(frame["elo_dispersion_norm"].max()),
        ]
        fitted_max = max(0.0, *(fit["slope"] * x + fit["intercept"] for x in xs))
    robust_data_max = float(frame["payoff_dispersion_norm"].quantile(Y_LIMIT_QUANTILE))
    raw_max = max(0.10, robust_data_max, fitted_max) * 1.12
    rounded_max = math.ceil(raw_max / 0.05) * 0.05
    return (0.0, rounded_max)


def row_label(row: pd.Series) -> str:
    models = "\n".join(textwrap.wrap(row["model_shorts"], width=28))
    return f"{row['bin_label']}\n{models}\nPayoff SD"


def draw_panel(
    ax: plt.Axes,
    sub: pd.DataFrame,
    *,
    point_color: str,
    x_lim: tuple[float, float],
    y_lim: tuple[float, float],
    title: str | None = None,
    show_stats: bool = True,
) -> None:
    ax.scatter(
        sub["elo_dispersion_norm"],
        sub["payoff_dispersion_norm"],
        s=POINT_SIZE,
        color=point_color,
        edgecolor="none",
        alpha=POINT_ALPHA,
    )
    fit = fit_line(sub)
    if math.isfinite(fit["slope"]):
        xs = np.linspace(
            float(sub["elo_dispersion_norm"].min()),
            float(sub["elo_dispersion_norm"].max()),
            120,
        )
        ax.plot(xs, fit["slope"] * xs + fit["intercept"], color="#111111", lw=LINE_WIDTH, alpha=0.95)
    if title is not None:
        ax.set_title(title, fontsize=12, pad=8)
    if show_stats:
        if math.isfinite(fit["pearson_r"]):
            label = f"n={len(sub)}\nr={fit['pearson_r']:+.2f}\nslope={fit['slope']:+.2f}"
        else:
            label = f"n={len(sub)}\nr=NA\nslope=NA"
        ax.text(
            0.04,
            0.96,
            label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.5,
            color="#222222",
            bbox={"facecolor": "white", "edgecolor": "#D0D0D0", "alpha": 0.86, "pad": 2.5},
        )
    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.grid(True, color="#D9DEE7", alpha=0.55, linewidth=0.7)
    ax.tick_params(axis="both", labelsize=8.5)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def plot_overall(runs: pd.DataFrame, bins: pd.DataFrame) -> Path:
    x_lim = (0.0, float(runs["elo_dispersion_norm"].max()) * 1.05)
    y_lim = trend_y_lim(runs)
    fig, axes = plt.subplots(1, len(bins), figsize=(8.7, 6.7), sharex=True, sharey=True)
    for ax, (_, bin_row), color in zip(axes, bins.iterrows(), BIN_COLORS, strict=True):
        sub = runs[runs["max_elo_bin"].eq(bin_row["max_elo_bin"])]
        title = f"{bin_row['bin_label']}\n{bin_row['n_models']} max models, n={len(sub)}"
        draw_panel(ax, sub, point_color=color, x_lim=x_lim, y_lim=y_lim, title=title)
    axes[0].set_ylabel("Payoff SD (% of 0-100 range)", fontsize=11, labelpad=7)
    fig.supxlabel("Elo SD (% of model range)", fontsize=12, y=0.02)
    fig.suptitle("Range-normalized payoff dispersion vs Elo dispersion", fontsize=15, y=1.02)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96), w_pad=1.2)
    out_path = OUT_DIR / "heterogeneous_max_elo_bin_normalized_dispersion_overall.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_by_n(runs: pd.DataFrame, bins: pd.DataFrame) -> Path:
    x_lim = (0.0, float(runs["elo_dispersion_norm"].max()) * 1.05)
    y_lim = trend_y_lim(runs)
    fig, axes = plt.subplots(len(bins), len(N_ORDER), figsize=(13.4, 15.4), sharex=True, sharey=True)
    for row_idx, (_, bin_row) in enumerate(bins.iterrows()):
        for col_idx, n_agents in enumerate(N_ORDER):
            ax = axes[row_idx, col_idx]
            sub = runs[runs["max_elo_bin"].eq(bin_row["max_elo_bin"]) & runs["n_agents"].eq(n_agents)]
            draw_panel(
                ax,
                sub,
                point_color=N_COLORS[n_agents],
                x_lim=x_lim,
                y_lim=y_lim,
                title=f"N={n_agents}" if row_idx == 0 else None,
            )
            if col_idx == 0:
                ax.set_ylabel(row_label(bin_row), fontsize=9.2, labelpad=8)
    fig.supxlabel("Elo SD (% of model range)", fontsize=12, y=0.03)
    fig.suptitle("Range-normalized dispersion by group size", fontsize=15, y=0.995)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97), w_pad=0.9, h_pad=1.1)
    out_path = OUT_DIR / "heterogeneous_max_elo_bin_normalized_dispersion_by_n.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_by_game(runs: pd.DataFrame, bins: pd.DataFrame) -> Path:
    x_lim = (0.0, float(runs["elo_dispersion_norm"].max()) * 1.05)
    y_lim = trend_y_lim(runs)
    fig, axes = plt.subplots(len(bins), len(GAME_ORDER), figsize=(9.2, 13.6), sharex=True, sharey=True)
    game_colors = {"game1": "#4E79A7", "game2": "#59A14F", "game3": "#E15759"}
    for row_idx, (_, bin_row) in enumerate(bins.iterrows()):
        for col_idx, game in enumerate(GAME_ORDER):
            ax = axes[row_idx, col_idx]
            sub = runs[runs["max_elo_bin"].eq(bin_row["max_elo_bin"]) & runs["game_label"].eq(game)]
            draw_panel(
                ax,
                sub,
                point_color=game_colors[game],
                x_lim=x_lim,
                y_lim=y_lim,
                title=GAME_LABELS[game] if row_idx == 0 else None,
            )
            if col_idx == 0:
                ax.set_ylabel(row_label(bin_row), fontsize=9.2, labelpad=8)
    fig.supxlabel("Elo SD (% of model range)", fontsize=12, y=0.03)
    fig.suptitle("Range-normalized dispersion by game", fontsize=15, y=0.995)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97), w_pad=1.0, h_pad=1.1)
    out_path = OUT_DIR / "heterogeneous_max_elo_bin_normalized_dispersion_by_game.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_by_competition(runs: pd.DataFrame, bins: pd.DataFrame) -> Path:
    x_lim = (0.0, float(runs["elo_dispersion_norm"].max()) * 1.05)
    y_lim = trend_y_lim(runs)
    fig, axes = plt.subplots(len(bins), len(COMPETITION_ORDER), figsize=(9.2, 13.6), sharex=True, sharey=True)
    for row_idx, (_, bin_row) in enumerate(bins.iterrows()):
        for col_idx, competition_band in enumerate(COMPETITION_ORDER):
            ax = axes[row_idx, col_idx]
            sub = runs[
                runs["max_elo_bin"].eq(bin_row["max_elo_bin"])
                & runs["competition_band"].eq(competition_band)
            ]
            draw_panel(
                ax,
                sub,
                point_color=COMPETITION_COLORS[competition_band],
                x_lim=x_lim,
                y_lim=y_lim,
                title=COMPETITION_LABELS[competition_band] if row_idx == 0 else None,
            )
            if col_idx == 0:
                ax.set_ylabel(row_label(bin_row), fontsize=9.2, labelpad=8)
    fig.supxlabel("Elo SD (% of model range)", fontsize=12, y=0.03)
    fig.suptitle("Range-normalized dispersion by competition", fontsize=15, y=0.995)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97), w_pad=1.0, h_pad=1.1)
    out_path = OUT_DIR / "heterogeneous_max_elo_bin_normalized_dispersion_by_competition.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    runs, bins, metadata = load_data()

    group_base = ["max_elo_bin_order", "max_elo_bin", "max_elo_bin_label", "max_elo_bin_models"]
    overall_summary = fit_summary(runs, group_base, "overall").sort_values("max_elo_bin_order")
    by_n_summary = fit_summary(runs, group_base + ["n_agents"], "by_n").sort_values(
        ["max_elo_bin_order", "n_agents"]
    )
    by_game_summary = fit_summary(runs, group_base + ["game_label"], "by_game").sort_values(
        ["max_elo_bin_order", "game_label"]
    )
    by_competition_summary = fit_summary(
        runs,
        group_base + ["competition_band"],
        "by_competition",
    ).sort_values(["max_elo_bin_order", "competition_band"])

    pd.concat([overall_summary, by_n_summary, by_game_summary, by_competition_summary], ignore_index=True).to_csv(
        OUT_DIR / "heterogeneous_max_elo_bin_normalized_dispersion_fit_summary.csv",
        index=False,
    )
    overall_summary.to_csv(
        OUT_DIR / "heterogeneous_max_elo_bin_normalized_dispersion_overall_fit_summary.csv",
        index=False,
    )
    by_n_summary.to_csv(
        OUT_DIR / "heterogeneous_max_elo_bin_normalized_dispersion_by_n_fit_summary.csv",
        index=False,
    )
    by_game_summary.to_csv(
        OUT_DIR / "heterogeneous_max_elo_bin_normalized_dispersion_by_game_fit_summary.csv",
        index=False,
    )
    by_competition_summary.to_csv(
        OUT_DIR / "heterogeneous_max_elo_bin_normalized_dispersion_by_competition_fit_summary.csv",
        index=False,
    )

    paths = [
        plot_overall(runs, bins),
        plot_by_n(runs, bins),
        plot_by_game(runs, bins),
        plot_by_competition(runs, bins),
    ]

    print(
        "Normalization: "
        f"Elo SD / {metadata['elo_range']:.0f} Elo points "
        f"({metadata['elo_min']:.0f}-{metadata['elo_max']:.0f}); "
        f"payoff SD / {metadata['payoff_range']:.0f} payoff points"
    )
    print("\nOverall normalized-dispersion fit summary:")
    cols = [
        "max_elo_bin_label",
        "n_runs",
        "slope_payoff_pp_per_elo_pp",
        "p_value",
        "pearson_r",
        "r_squared",
    ]
    print(overall_summary[cols].round(4).to_string(index=False))
    print("\nWrote:")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
