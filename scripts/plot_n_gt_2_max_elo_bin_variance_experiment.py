#!/usr/bin/env python3
"""Max-Elo-bin variance experiment for heterogeneous N-player runs.

This repeats the fixed-maximum-Elo analysis, but uses bins of strongest-model
Elo instead of exact strongest models. We keep only runs whose strongest model
falls in the top half of the model Elo roster, then split that upper half into
three equal model-count bins.
"""

from __future__ import annotations

import math
import textwrap
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AGENT_TABLE = (
    PROJECT_ROOT
    / "experiments/results/n2_plus_multiagent_comparison_analysis_20260505"
    / "tables_multiagent/heterogeneous_agents_fresh.csv"
)
OUT_DIR = PROJECT_ROOT / "overleaf/neurips/graphics/n_gt_2_report"

N_ORDER = [2, 4, 6, 8, 10]
GAME_ORDER = ["game1", "game2", "game3"]
GAME_LABELS = {
    "game1": "Game 1",
    "game2": "Game 2",
    "game3": "Game 3",
}
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
TREND_POINT_ALPHA = 0.16
TREND_POINT_SIZE = 18
TREND_LINE_WIDTH = 2.6
Y_LIMIT_QUANTILE = 0.95


def _var(values: pd.Series) -> float:
    return float(np.var(values.to_numpy(dtype=float), ddof=0))


def load_agents() -> pd.DataFrame:
    agents = pd.read_csv(AGENT_TABLE)
    agents = agents[agents["experiment_family"].eq("heterogeneous_random")].copy()
    for col in ["n_agents", "elo", "final_utility", "competition_ci"]:
        agents[col] = pd.to_numeric(agents[col], errors="coerce")
    agents = agents.dropna(
        subset=["run_key", "game_label", "n_agents", "elo", "final_utility", "competition_ci"]
    )
    agents["n_agents"] = agents["n_agents"].astype(int)
    return agents


def load_run_metrics(agents: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for run_key, group in agents.groupby("run_key", sort=False):
        max_elo = float(group["elo"].max())
        top = group[group["elo"].eq(max_elo)].sort_values(["model_short", "model"])
        rows.append(
            {
                "run_key": run_key,
                "config_id": int(group["config_id"].iloc[0]),
                "game_label": group["game_label"].iloc[0],
                "n_agents": int(group["n_agents"].iloc[0]),
                "competition_ci": float(group["competition_ci"].iloc[0]),
                "competition_label_ci": group["competition_label_ci"].iloc[0],
                "competition_band": group["competition_band"].iloc[0],
                "max_elo": max_elo,
                "max_model": top["model"].iloc[0],
                "max_model_short": top["model_short"].iloc[0],
                "mean_elo": float(group["elo"].mean()),
                "min_elo": float(group["elo"].min()),
                "elo_variance": _var(group["elo"]),
                "payoff_variance": _var(group["final_utility"]),
                "mean_payoff": float(group["final_utility"].mean()),
                "n_agents_observed": int(group["final_utility"].count()),
                "model_count": int(group["model"].nunique()),
            }
        )
    return pd.DataFrame(rows).sort_values(["max_elo", "n_agents", "game_label", "config_id"])


def choose_max_elo_bins(agents: pd.DataFrame, run_metrics: pd.DataFrame) -> pd.DataFrame:
    model_roster = (
        agents[["model", "model_short", "elo"]]
        .drop_duplicates()
        .sort_values(["elo", "model_short"])
        .reset_index(drop=True)
    )
    model_roster["elo_rank_low_to_high"] = np.arange(1, len(model_roster) + 1)
    upper_half = model_roster.iloc[len(model_roster) // 2 :].copy().reset_index(drop=True)

    # Three bins, four models per bin for the 24-model roster.
    bin_rows: list[dict[str, object]] = []
    for bin_index, start in enumerate(range(0, len(upper_half), 4), start=1):
        models = upper_half.iloc[start : start + 4].copy()
        if models.empty:
            continue
        runs = run_metrics[run_metrics["max_model"].isin(models["model"])]
        if len(models) < 2 or runs.empty:
            continue
        bin_label = f"Bin {bin_index}: Elo {int(models['elo'].min())}-{int(models['elo'].max())}"
        bin_rows.append(
            {
                "max_elo_bin": f"upper_bin_{bin_index}",
                "bin_order": bin_index,
                "bin_label": bin_label,
                "elo_min": float(models["elo"].min()),
                "elo_max": float(models["elo"].max()),
                "n_models": int(len(models)),
                "model_shorts": "; ".join(models["model_short"].tolist()),
                "models": "; ".join(models["model"].tolist()),
                "n_runs": int(len(runs)),
            }
        )
    return pd.DataFrame(bin_rows)


def assign_bins(run_metrics: pd.DataFrame, bins: pd.DataFrame) -> pd.DataFrame:
    selected = run_metrics.copy()
    selected["max_elo_bin"] = pd.NA
    selected["max_elo_bin_label"] = pd.NA
    selected["max_elo_bin_order"] = pd.NA
    selected["max_elo_bin_model_count"] = pd.NA
    selected["max_elo_bin_models"] = pd.NA
    for _, row in bins.iterrows():
        mask = selected["max_elo"].between(row["elo_min"], row["elo_max"], inclusive="both")
        selected.loc[mask, "max_elo_bin"] = row["max_elo_bin"]
        selected.loc[mask, "max_elo_bin_label"] = row["bin_label"]
        selected.loc[mask, "max_elo_bin_order"] = int(row["bin_order"])
        selected.loc[mask, "max_elo_bin_model_count"] = int(row["n_models"])
        selected.loc[mask, "max_elo_bin_models"] = row["model_shorts"]
    selected = selected.dropna(subset=["max_elo_bin"]).copy()
    selected["max_elo_bin_order"] = selected["max_elo_bin_order"].astype(int)
    selected["max_elo_bin_model_count"] = selected["max_elo_bin_model_count"].astype(int)
    return selected.sort_values(["max_elo_bin_order", "n_agents", "game_label", "config_id"])


def fit_line(frame: pd.DataFrame) -> dict[str, float]:
    data = frame[["elo_variance", "payoff_variance"]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < 3 or data["elo_variance"].nunique() < 2 or data["payoff_variance"].nunique() < 2:
        return {
            "slope": math.nan,
            "intercept": math.nan,
            "slope_stderr": math.nan,
            "p_value": math.nan,
            "pearson_r": math.nan,
            "r_squared": math.nan,
        }
    x = data["elo_variance"].to_numpy(dtype=float)
    y = data["payoff_variance"].to_numpy(dtype=float)
    result = stats.linregress(x, y)
    pearson_r = float(result.rvalue)
    return {
        "slope": float(result.slope),
        "intercept": float(result.intercept),
        "slope_stderr": float(result.stderr),
        "p_value": float(result.pvalue),
        "pearson_r": pearson_r,
        "r_squared": pearson_r * pearson_r,
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
                "elo_variance_min": float(sub["elo_variance"].min()),
                "elo_variance_max": float(sub["elo_variance"].max()),
                "payoff_variance_min": float(sub["payoff_variance"].min()),
                "payoff_variance_max": float(sub["payoff_variance"].max()),
                "slope_payoff_var_per_elo_var": fit["slope"],
                "slope_payoff_var_per_1000_elo_var": fit["slope"] * 1000.0
                if math.isfinite(fit["slope"])
                else math.nan,
                "slope_stderr": fit["slope_stderr"],
                "slope_stderr_per_1000_elo_var": fit["slope_stderr"] * 1000.0
                if math.isfinite(fit["slope_stderr"])
                else math.nan,
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
        xs = [float(frame["elo_variance"].min()), float(frame["elo_variance"].max())]
        fitted_max = max(0.0, *(fit["slope"] * x + fit["intercept"] for x in xs))
    robust_data_max = float(frame["payoff_variance"].quantile(Y_LIMIT_QUANTILE))
    raw_max = max(100.0, robust_data_max, fitted_max) * 1.15
    rounded_max = math.ceil(raw_max / 50.0) * 50.0
    return (0.0, rounded_max)


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
        sub["elo_variance"],
        sub["payoff_variance"],
        s=TREND_POINT_SIZE,
        color=point_color,
        edgecolor="none",
        linewidth=0,
        alpha=TREND_POINT_ALPHA,
    )
    fit = fit_line(sub)
    if math.isfinite(fit["slope"]):
        xs = np.linspace(float(sub["elo_variance"].min()), float(sub["elo_variance"].max()), 120)
        ax.plot(xs, fit["slope"] * xs + fit["intercept"], color="#111111", lw=TREND_LINE_WIDTH, alpha=0.95)
    if title is not None:
        ax.set_title(title, fontsize=12, pad=8)
    if show_stats:
        if math.isfinite(fit["pearson_r"]):
            label = (
                f"n={len(sub)}\n"
                f"r={fit['pearson_r']:+.2f}\n"
                f"slope={fit['slope'] * 1000.0:+.1f}/1k"
            )
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
    ax.grid(True, color="#D9DEE7", alpha=0.55, linewidth=0.7)
    ax.tick_params(axis="both", labelsize=8.5)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def row_label(row: pd.Series) -> str:
    models = "\n".join(textwrap.wrap(row["model_shorts"], width=28))
    return f"{row['bin_label']}\n{models}\nPayoff variance"


def plot_overall(selected_runs: pd.DataFrame, bins: pd.DataFrame) -> Path:
    x_lim = (0.0, float(selected_runs["elo_variance"].max()) * 1.05)
    y_lim = trend_y_lim(selected_runs)
    fig, axes = plt.subplots(1, len(bins), figsize=(8.7, 6.7), sharex=True, sharey=True)
    for ax, (_, bin_row), color in zip(axes, bins.iterrows(), BIN_COLORS, strict=True):
        sub = selected_runs[selected_runs["max_elo_bin"].eq(bin_row["max_elo_bin"])]
        title = f"{bin_row['bin_label']}\n{bin_row['n_models']} max models, n={len(sub)}"
        draw_panel(ax, sub, point_color=color, x_lim=x_lim, y_lim=y_lim, title=title)
    axes[0].set_ylabel("Within-run payoff variance", fontsize=11, labelpad=7)
    fig.supxlabel("Within-roster Arena Elo variance", fontsize=12, y=0.02)
    fig.suptitle(
        "Payoff variance vs Elo variance, conditional on strongest-model Elo bin",
        fontsize=15,
        y=1.02,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.96), w_pad=1.2)
    out_path = OUT_DIR / "heterogeneous_max_elo_bin_variance_overall_trend_zoomed.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_by_n(selected_runs: pd.DataFrame, bins: pd.DataFrame) -> Path:
    x_lim = (0.0, float(selected_runs["elo_variance"].max()) * 1.05)
    y_lim = trend_y_lim(selected_runs)
    fig, axes = plt.subplots(len(bins), len(N_ORDER), figsize=(13.4, 15.4), sharex=True, sharey=True)
    for row_idx, (_, bin_row) in enumerate(bins.iterrows()):
        for col_idx, n_agents in enumerate(N_ORDER):
            ax = axes[row_idx, col_idx]
            sub = selected_runs[
                selected_runs["max_elo_bin"].eq(bin_row["max_elo_bin"])
                & selected_runs["n_agents"].eq(n_agents)
            ]
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
    fig.supxlabel("Within-roster Arena Elo variance", fontsize=12, y=0.03)
    fig.suptitle("Max-Elo-bin experiment, broken down by group size", fontsize=15, y=0.995)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97), w_pad=0.9, h_pad=1.1)
    out_path = OUT_DIR / "heterogeneous_max_elo_bin_variance_by_n_trend_zoomed.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_by_game(selected_runs: pd.DataFrame, bins: pd.DataFrame) -> Path:
    x_lim = (0.0, float(selected_runs["elo_variance"].max()) * 1.05)
    y_lim = trend_y_lim(selected_runs)
    fig, axes = plt.subplots(len(bins), len(GAME_ORDER), figsize=(9.2, 13.6), sharex=True, sharey=True)
    game_colors = {"game1": "#4E79A7", "game2": "#59A14F", "game3": "#E15759"}
    for row_idx, (_, bin_row) in enumerate(bins.iterrows()):
        for col_idx, game in enumerate(GAME_ORDER):
            ax = axes[row_idx, col_idx]
            sub = selected_runs[
                selected_runs["max_elo_bin"].eq(bin_row["max_elo_bin"])
                & selected_runs["game_label"].eq(game)
            ]
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
    fig.supxlabel("Within-roster Arena Elo variance", fontsize=12, y=0.03)
    fig.suptitle("Max-Elo-bin experiment, broken down by game", fontsize=15, y=0.995)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97), w_pad=1.0, h_pad=1.1)
    out_path = OUT_DIR / "heterogeneous_max_elo_bin_variance_by_game_trend_zoomed.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_by_competition(selected_runs: pd.DataFrame, bins: pd.DataFrame) -> Path:
    x_lim = (0.0, float(selected_runs["elo_variance"].max()) * 1.05)
    y_lim = trend_y_lim(selected_runs)
    fig, axes = plt.subplots(
        len(bins),
        len(COMPETITION_ORDER),
        figsize=(9.2, 13.6),
        sharex=True,
        sharey=True,
    )
    for row_idx, (_, bin_row) in enumerate(bins.iterrows()):
        for col_idx, competition_band in enumerate(COMPETITION_ORDER):
            ax = axes[row_idx, col_idx]
            sub = selected_runs[
                selected_runs["max_elo_bin"].eq(bin_row["max_elo_bin"])
                & selected_runs["competition_band"].eq(competition_band)
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
    fig.supxlabel("Within-roster Arena Elo variance", fontsize=12, y=0.03)
    fig.suptitle("Max-Elo-bin experiment, broken down by competition band", fontsize=15, y=0.995)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97), w_pad=1.0, h_pad=1.1)
    out_path = OUT_DIR / "heterogeneous_max_elo_bin_variance_by_competition_trend_zoomed.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    agents = load_agents()
    run_metrics = load_run_metrics(agents)
    bins = choose_max_elo_bins(agents, run_metrics)
    selected_runs = assign_bins(run_metrics, bins)

    run_metrics.to_csv(OUT_DIR / "heterogeneous_max_elo_bin_variance_all_run_metrics.csv", index=False)
    bins.to_csv(OUT_DIR / "heterogeneous_max_elo_bin_variance_selected_bins.csv", index=False)
    selected_runs.to_csv(OUT_DIR / "heterogeneous_max_elo_bin_variance_selected_run_metrics.csv", index=False)

    group_base = ["max_elo_bin_order", "max_elo_bin", "max_elo_bin_label", "max_elo_bin_models"]
    overall_summary = fit_summary(selected_runs, group_base, "overall").sort_values("max_elo_bin_order")
    by_n_summary = fit_summary(selected_runs, group_base + ["n_agents"], "by_n").sort_values(
        ["max_elo_bin_order", "n_agents"]
    )
    by_game_summary = fit_summary(selected_runs, group_base + ["game_label"], "by_game").sort_values(
        ["max_elo_bin_order", "game_label"]
    )
    by_competition_summary = fit_summary(
        selected_runs,
        group_base + ["competition_band"],
        "by_competition",
    ).sort_values(["max_elo_bin_order", "competition_band"])
    pd.concat([overall_summary, by_n_summary, by_game_summary, by_competition_summary], ignore_index=True).to_csv(
        OUT_DIR / "heterogeneous_max_elo_bin_variance_fit_summary.csv",
        index=False,
    )
    overall_summary.to_csv(OUT_DIR / "heterogeneous_max_elo_bin_variance_overall_fit_summary.csv", index=False)
    by_n_summary.to_csv(OUT_DIR / "heterogeneous_max_elo_bin_variance_by_n_fit_summary.csv", index=False)
    by_game_summary.to_csv(OUT_DIR / "heterogeneous_max_elo_bin_variance_by_game_fit_summary.csv", index=False)
    by_competition_summary.to_csv(
        OUT_DIR / "heterogeneous_max_elo_bin_variance_by_competition_fit_summary.csv",
        index=False,
    )

    paths = [
        plot_overall(selected_runs, bins),
        plot_by_n(selected_runs, bins),
        plot_by_game(selected_runs, bins),
        plot_by_competition(selected_runs, bins),
    ]

    print("Selected strongest-model Elo bins:")
    print(bins[["bin_order", "bin_label", "n_models", "n_runs", "model_shorts"]].to_string(index=False))
    print("\nOverall max-Elo-bin fit summary:")
    cols = [
        "max_elo_bin_label",
        "n_runs",
        "slope_payoff_var_per_1000_elo_var",
        "pearson_r",
        "r_squared",
    ]
    print(overall_summary[cols].round(4).to_string(index=False))
    print("\nWrote:")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
