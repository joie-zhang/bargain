#!/usr/bin/env python3
"""Fixed-maximum-Elo variance experiment for heterogeneous N-player runs.

For selected anchor models, keep only runs where that model is the strongest
model in the sampled roster. Within those fixed-max strata, plot within-roster
Elo variance against within-run payoff variance.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AGENT_TABLE = (
    PROJECT_ROOT
    / "experiments/results/n2_plus_multiagent_comparison_analysis_20260505"
    / "tables_multiagent/heterogeneous_agents_fresh.csv"
)
OUT_DIR = PROJECT_ROOT / "overleaf/neurips/graphics/n_gt_2_report"

MIN_ANCHOR_RUNS = 50
N_ANCHORS = 5
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
ANCHOR_COLORS = ["#386CB0", "#7FC97F", "#FDC086", "#BEAED4", "#F0027F"]


def _var(values: pd.Series) -> float:
    return float(np.var(values.to_numpy(dtype=float), ddof=0))


def _short_title(label: str, max_chars: int = 18) -> str:
    if len(label) <= max_chars:
        return label
    return label[: max_chars - 1] + "..."


def load_run_metrics() -> pd.DataFrame:
    agents = pd.read_csv(AGENT_TABLE)
    agents = agents[agents["experiment_family"].eq("heterogeneous_random")].copy()
    for col in ["n_agents", "elo", "final_utility", "competition_ci"]:
        agents[col] = pd.to_numeric(agents[col], errors="coerce")
    agents = agents.dropna(
        subset=["run_key", "game_label", "n_agents", "elo", "final_utility", "competition_ci"]
    )
    agents["n_agents"] = agents["n_agents"].astype(int)

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


def choose_anchor_models(run_metrics: pd.DataFrame) -> pd.DataFrame:
    counts = (
        run_metrics.groupby(["max_elo", "max_model", "max_model_short"], dropna=False)
        .agg(
            total_runs=("run_key", "count"),
            elo_variance_min=("elo_variance", "min"),
            elo_variance_max=("elo_variance", "max"),
            payoff_variance_min=("payoff_variance", "min"),
            payoff_variance_max=("payoff_variance", "max"),
        )
        .reset_index()
        .sort_values("max_elo")
    )
    counts["eligible"] = counts["total_runs"].ge(MIN_ANCHOR_RUNS)
    eligible = counts[counts["eligible"]].reset_index(drop=True)
    if len(eligible) < N_ANCHORS:
        raise RuntimeError(
            f"Only {len(eligible)} max-Elo models have at least {MIN_ANCHOR_RUNS} runs; "
            f"need {N_ANCHORS}."
        )

    positions = np.linspace(0, len(eligible) - 1, N_ANCHORS)
    selected_indices: list[int] = []
    used: set[int] = set()
    for position in positions:
        idx = int(round(position))
        if idx in used:
            for candidate in range(len(eligible)):
                if candidate not in used:
                    idx = candidate
                    break
        selected_indices.append(idx)
        used.add(idx)

    selected = eligible.iloc[selected_indices].copy()
    selected["anchor_order"] = range(1, len(selected) + 1)
    return selected.sort_values("anchor_order")


def fit_line(frame: pd.DataFrame) -> dict[str, float]:
    data = frame[["elo_variance", "payoff_variance"]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < 3 or data["elo_variance"].nunique() < 2 or data["payoff_variance"].nunique() < 2:
        return {
            "slope": math.nan,
            "intercept": math.nan,
            "pearson_r": math.nan,
            "r_squared": math.nan,
        }
    x = data["elo_variance"].to_numpy(dtype=float)
    y = data["payoff_variance"].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    pearson_r = float(np.corrcoef(x, y)[0, 1])
    return {
        "slope": float(slope),
        "intercept": float(intercept),
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
                "intercept": fit["intercept"],
                "pearson_r": fit["pearson_r"],
                "r_squared": fit["r_squared"],
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


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
        s=28,
        color=point_color,
        edgecolor="white",
        linewidth=0.45,
        alpha=0.68,
    )
    fit = fit_line(sub)
    if math.isfinite(fit["slope"]):
        xs = np.linspace(float(sub["elo_variance"].min()), float(sub["elo_variance"].max()), 120)
        ax.plot(xs, fit["slope"] * xs + fit["intercept"], color="#111111", lw=1.8, alpha=0.9)
    if title is not None:
        ax.set_title(title, fontsize=12, pad=8)
    if show_stats:
        if math.isfinite(fit["pearson_r"]):
            label = f"n={len(sub)}\nr={fit['pearson_r']:+.2f}"
        else:
            label = f"n={len(sub)}\nr=NA"
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


def plot_overall(selected_runs: pd.DataFrame, anchors: pd.DataFrame) -> Path:
    x_lim = (0.0, float(selected_runs["elo_variance"].max()) * 1.05)
    y_lim = (0.0, max(float(selected_runs["payoff_variance"].max()) * 1.08, 100.0))
    fig, axes = plt.subplots(1, len(anchors), figsize=(19.0, 4.6), sharex=True, sharey=True)

    for ax, (_, anchor), color in zip(axes, anchors.iterrows(), ANCHOR_COLORS, strict=True):
        sub = selected_runs[selected_runs["max_model"].eq(anchor["max_model"])]
        title = f"{_short_title(anchor['max_model_short'])}\nmax Elo={anchor['max_elo']:.0f}"
        draw_panel(ax, sub, point_color=color, x_lim=x_lim, y_lim=y_lim, title=title)

    axes[0].set_ylabel("Within-run payoff variance", fontsize=11, labelpad=7)
    fig.supxlabel("Within-roster Arena Elo variance", fontsize=12, y=0.02)
    fig.suptitle(
        "Payoff variance vs Elo variance, conditional on selected strongest model",
        fontsize=16,
        y=1.04,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.96), w_pad=1.2)
    out_path = OUT_DIR / "heterogeneous_fixed_max_elo_variance_overall.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_by_n(selected_runs: pd.DataFrame, anchors: pd.DataFrame) -> Path:
    x_lim = (0.0, float(selected_runs["elo_variance"].max()) * 1.05)
    y_lim = (0.0, max(float(selected_runs["payoff_variance"].max()) * 1.08, 100.0))
    fig, axes = plt.subplots(
        len(anchors),
        len(N_ORDER),
        figsize=(18.5, 15.2),
        sharex=True,
        sharey=True,
    )
    for row_idx, (_, anchor) in enumerate(anchors.iterrows()):
        for col_idx, n_agents in enumerate(N_ORDER):
            ax = axes[row_idx, col_idx]
            sub = selected_runs[
                selected_runs["max_model"].eq(anchor["max_model"])
                & selected_runs["n_agents"].eq(n_agents)
            ]
            draw_panel(
                ax,
                sub,
                point_color=N_COLORS[n_agents],
                x_lim=x_lim,
                y_lim=y_lim,
                title=f"N={n_agents}" if row_idx == 0 else None,
                show_stats=True,
            )
            if col_idx == 0:
                ax.set_ylabel(
                    f"{_short_title(anchor['max_model_short'], 15)}\nPayoff variance",
                    fontsize=10,
                    labelpad=8,
                )

    fig.supxlabel("Within-roster Arena Elo variance", fontsize=12, y=0.03)
    fig.suptitle(
        "Fixed-maximum-Elo experiment, broken down by group size",
        fontsize=16,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.97), w_pad=0.9, h_pad=1.1)
    out_path = OUT_DIR / "heterogeneous_fixed_max_elo_variance_by_n.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_by_game(selected_runs: pd.DataFrame, anchors: pd.DataFrame) -> Path:
    x_lim = (0.0, float(selected_runs["elo_variance"].max()) * 1.05)
    y_lim = (0.0, max(float(selected_runs["payoff_variance"].max()) * 1.08, 100.0))
    fig, axes = plt.subplots(
        len(anchors),
        len(GAME_ORDER),
        figsize=(13.5, 15.2),
        sharex=True,
        sharey=True,
    )
    game_colors = {"game1": "#4E79A7", "game2": "#59A14F", "game3": "#E15759"}
    for row_idx, (_, anchor) in enumerate(anchors.iterrows()):
        for col_idx, game in enumerate(GAME_ORDER):
            ax = axes[row_idx, col_idx]
            sub = selected_runs[
                selected_runs["max_model"].eq(anchor["max_model"])
                & selected_runs["game_label"].eq(game)
            ]
            draw_panel(
                ax,
                sub,
                point_color=game_colors[game],
                x_lim=x_lim,
                y_lim=y_lim,
                title=GAME_LABELS[game] if row_idx == 0 else None,
                show_stats=True,
            )
            if col_idx == 0:
                ax.set_ylabel(
                    f"{_short_title(anchor['max_model_short'], 15)}\nPayoff variance",
                    fontsize=10,
                    labelpad=8,
                )

    fig.supxlabel("Within-roster Arena Elo variance", fontsize=12, y=0.03)
    fig.suptitle(
        "Fixed-maximum-Elo experiment, broken down by game",
        fontsize=16,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.97), w_pad=1.0, h_pad=1.1)
    out_path = OUT_DIR / "heterogeneous_fixed_max_elo_variance_by_game.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_by_competition(selected_runs: pd.DataFrame, anchors: pd.DataFrame) -> Path:
    x_lim = (0.0, float(selected_runs["elo_variance"].max()) * 1.05)
    y_lim = (0.0, max(float(selected_runs["payoff_variance"].max()) * 1.08, 100.0))
    fig, axes = plt.subplots(
        len(anchors),
        len(COMPETITION_ORDER),
        figsize=(13.5, 15.2),
        sharex=True,
        sharey=True,
    )
    for row_idx, (_, anchor) in enumerate(anchors.iterrows()):
        for col_idx, competition_band in enumerate(COMPETITION_ORDER):
            ax = axes[row_idx, col_idx]
            sub = selected_runs[
                selected_runs["max_model"].eq(anchor["max_model"])
                & selected_runs["competition_band"].eq(competition_band)
            ]
            draw_panel(
                ax,
                sub,
                point_color=COMPETITION_COLORS[competition_band],
                x_lim=x_lim,
                y_lim=y_lim,
                title=COMPETITION_LABELS[competition_band] if row_idx == 0 else None,
                show_stats=True,
            )
            if col_idx == 0:
                ax.set_ylabel(
                    f"{_short_title(anchor['max_model_short'], 15)}\nPayoff variance",
                    fontsize=10,
                    labelpad=8,
                )

    fig.supxlabel("Within-roster Arena Elo variance", fontsize=12, y=0.03)
    fig.suptitle(
        "Fixed-maximum-Elo experiment, broken down by competition band",
        fontsize=16,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.97), w_pad=1.0, h_pad=1.1)
    out_path = OUT_DIR / "heterogeneous_fixed_max_elo_variance_by_competition.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    run_metrics = load_run_metrics()
    anchors = choose_anchor_models(run_metrics)
    selected_runs = run_metrics[run_metrics["max_model"].isin(anchors["max_model"])].copy()

    run_metrics.to_csv(OUT_DIR / "heterogeneous_fixed_max_elo_variance_all_run_metrics.csv", index=False)
    anchors.to_csv(OUT_DIR / "heterogeneous_fixed_max_elo_variance_selected_anchors.csv", index=False)
    selected_runs.to_csv(OUT_DIR / "heterogeneous_fixed_max_elo_variance_selected_run_metrics.csv", index=False)

    overall_summary = fit_summary(
        selected_runs,
        ["max_elo", "max_model_short", "max_model"],
        "overall",
    ).sort_values("max_elo")
    by_n_summary = fit_summary(
        selected_runs,
        ["max_elo", "max_model_short", "max_model", "n_agents"],
        "by_n",
    ).sort_values(["max_elo", "n_agents"])
    by_game_summary = fit_summary(
        selected_runs,
        ["max_elo", "max_model_short", "max_model", "game_label"],
        "by_game",
    ).sort_values(["max_elo", "game_label"])
    by_competition_summary = fit_summary(
        selected_runs,
        ["max_elo", "max_model_short", "max_model", "competition_band"],
        "by_competition",
    ).sort_values(["max_elo", "competition_band"])
    pd.concat([overall_summary, by_n_summary, by_game_summary, by_competition_summary], ignore_index=True).to_csv(
        OUT_DIR / "heterogeneous_fixed_max_elo_variance_fit_summary.csv",
        index=False,
    )
    overall_summary.to_csv(OUT_DIR / "heterogeneous_fixed_max_elo_variance_overall_fit_summary.csv", index=False)
    by_n_summary.to_csv(OUT_DIR / "heterogeneous_fixed_max_elo_variance_by_n_fit_summary.csv", index=False)
    by_game_summary.to_csv(OUT_DIR / "heterogeneous_fixed_max_elo_variance_by_game_fit_summary.csv", index=False)
    by_competition_summary.to_csv(
        OUT_DIR / "heterogeneous_fixed_max_elo_variance_by_competition_fit_summary.csv",
        index=False,
    )

    paths = [
        plot_overall(selected_runs, anchors),
        plot_by_n(selected_runs, anchors),
        plot_by_game(selected_runs, anchors),
        plot_by_competition(selected_runs, anchors),
    ]

    print("Selected max-Elo anchors:")
    print(anchors[["anchor_order", "max_model_short", "max_elo", "total_runs"]].to_string(index=False))
    print("\nOverall fixed-max fit summary:")
    cols = [
        "max_model_short",
        "max_elo",
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
