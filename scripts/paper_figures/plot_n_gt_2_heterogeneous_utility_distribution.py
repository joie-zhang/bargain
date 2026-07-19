#!/usr/bin/env python3
"""Recreate N>2 heterogeneous payoff Figure 7 and plot utility by Elo bucket.

The new utility-distribution plot uses the heterogeneous agent table produced by
the N=2 + N>2 analysis bundle. For each N and Elo bucket, it first averages
agent utility within each game, then averages those three game means so that the
three games contribute equally.
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


PROJECT_ROOT = Path(__file__).resolve().parents[2]
AGENT_TABLE = (
    PROJECT_ROOT
    / "experiments/results/n2_plus_multiagent_comparison_analysis_20260505"
    / "tables_multiagent/heterogeneous_agents_fresh.csv"
)
OVERLEAF_OUT = PROJECT_ROOT / "overleaf/neurips/graphics/n_gt_2_report"
ITERATION_OUT = (
    PROJECT_ROOT
    / "experiments/results/figure_iteration_20260626/multiagent_utility_distribution"
)

GAME_ORDER = ["game1", "game2", "game3"]
GAME_TITLES = {
    "game1": "Game 1: Item allocation",
    "game2": "Game 2: Diplomacy",
    "game3": "Game 3: Cofunding",
}
N_ORDER = [2, 4, 6, 8, 10]
N_COLORS = {
    2: "#1f77b4",
    4: "#d62728",
    6: "#2ca02c",
    8: "#9467bd",
    10: "#ff7f0e",
}
ELO_BIN_COUNT = 10
BAR_COLOR = "#5B8DB8"


def sem(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if len(clean) <= 1:
        return 0.0
    return float(clean.std(ddof=1) / math.sqrt(len(clean)))


def linear_fit(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    data = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < 2 or data["x"].nunique() < 2:
        return math.nan, math.nan
    slope, intercept = np.polyfit(data["x"].to_numpy(dtype=float), data["y"].to_numpy(dtype=float), 1)
    return float(slope), float(intercept)


def load_heterogeneous_agents() -> pd.DataFrame:
    agents = pd.read_csv(AGENT_TABLE)
    agents = agents[agents["experiment_family"].eq("heterogeneous_random")].copy()
    numeric_cols = ["n_agents", "elo", "final_utility"]
    for col in numeric_cols:
        agents[col] = pd.to_numeric(agents[col], errors="coerce")
    agents = agents.dropna(subset=["game_label", "n_agents", "model", "elo", "final_utility"])
    agents["n_agents"] = agents["n_agents"].astype(int)
    agents["elo"] = agents["elo"].astype(int)
    return agents


def aggregate_model_payoff(agents: pd.DataFrame) -> pd.DataFrame:
    return (
        agents.groupby(["game_label", "n_agents", "model", "model_short", "elo"], dropna=False)
        .agg(
            obs_count=("final_utility", "count"),
            final_utility=("final_utility", "mean"),
            final_utility_sem=("final_utility", sem),
        )
        .reset_index()
    )


def recreate_clean_figure7(agents: pd.DataFrame) -> Path:
    agg = aggregate_model_payoff(agents)
    fig, axes = plt.subplots(1, 3, figsize=(18.4, 5.87), sharey=False)

    for ax, game in zip(axes, GAME_ORDER):
        game_df = agg[agg["game_label"].eq(game)]
        for n in N_ORDER:
            sub = game_df[game_df["n_agents"].eq(n)].sort_values("elo")
            if sub.empty:
                continue
            color = N_COLORS[n]
            ax.errorbar(
                sub["elo"],
                sub["final_utility"],
                yerr=sub["final_utility_sem"].fillna(0.0),
                fmt="o",
                ms=4.0,
                color=color,
                ecolor=color,
                elinewidth=0.9,
                capsize=1.7,
                alpha=0.35,
                linestyle="none",
            )
            slope, intercept = linear_fit(sub["elo"], sub["final_utility"])
            if math.isfinite(slope):
                xs = np.linspace(float(sub["elo"].min()), float(sub["elo"].max()), 120)
                ax.plot(xs, slope * xs + intercept, color=color, linestyle="--", lw=2.1, alpha=0.86)

        ax.set_title(GAME_TITLES[game], fontsize=24, pad=10)
        ax.set_xlabel("Arena Elo", fontsize=20, labelpad=10)
        ax.set_ylabel("Mean model payoff", fontsize=20, labelpad=10)
        ax.tick_params(axis="both", labelsize=16)
        ax.grid(True, alpha=0.18, linewidth=0.7)

    legend_handles = [
        Line2D([0], [0], color=N_COLORS[n], marker="o", linestyle="--", lw=2.1, ms=5.0, label=f"N={n}")
        for n in N_ORDER
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.06),
        ncol=len(N_ORDER),
        fontsize=18,
        frameon=False,
        handlelength=2.5,
        columnspacing=2.3,
    )
    fig.tight_layout(rect=(0, 0.12, 1, 1))

    out_path = OVERLEAF_OUT / "heterogeneous_payoff_vs_arena_elo_by_n_recreated.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    ITERATION_OUT.mkdir(parents=True, exist_ok=True)
    fig_copy = ITERATION_OUT / out_path.name
    fig_copy.write_bytes(out_path.read_bytes())
    return out_path


def assign_elo_buckets(agents: pd.DataFrame, n_bins: int = ELO_BIN_COUNT) -> tuple[pd.DataFrame, pd.DataFrame]:
    models = (
        agents[["model", "model_short", "elo"]]
        .drop_duplicates()
        .sort_values(["elo", "model"])
        .reset_index(drop=True)
    )
    edges = np.linspace(float(models["elo"].min()), float(models["elo"].max()), n_bins + 1)
    models["bucket_order"] = pd.cut(
        models["elo"],
        bins=edges,
        labels=False,
        include_lowest=True,
        duplicates="drop",
    ).astype(int)
    models["elo_bucket"] = models["bucket_order"].map(lambda value: f"elo_bin_{int(value) + 1:02d}")
    ranges = (
        models.groupby(["elo_bucket", "bucket_order"], as_index=False)
        .agg(
            elo_min=("elo", "min"),
            elo_max=("elo", "max"),
            model_count=("model", "count"),
        )
        .sort_values("bucket_order")
    )
    ranges["bucket_label"] = ranges.apply(
        lambda row: f"{int(row['elo_min'])}\n{int(row['elo_max'])}",
        axis=1,
    )
    model_map = models[["model", "elo_bucket"]]
    agents = agents.merge(model_map, on="model", how="left")
    agents = agents.merge(ranges[["elo_bucket", "bucket_order", "bucket_label"]], on="elo_bucket", how="left")
    return agents, ranges


def compute_bucket_summary(agents: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    game_means = (
        agents.groupby(["game_label", "n_agents", "elo_bucket", "bucket_order", "bucket_label"], dropna=False)
        .agg(
            game_mean_utility=("final_utility", "mean"),
            agent_obs=("final_utility", "count"),
            run_count=("run_key", "nunique"),
            model_count=("model", "nunique"),
        )
        .reset_index()
    )
    summary = (
        game_means.groupby(["n_agents", "elo_bucket", "bucket_order", "bucket_label"], dropna=False)
        .agg(
            mean_utility=("game_mean_utility", "mean"),
            utility_sem_across_games=("game_mean_utility", sem),
            n_games=("game_label", "nunique"),
            total_agent_obs=("agent_obs", "sum"),
            total_run_count=("run_count", "sum"),
            model_count=("model_count", "max"),
        )
        .reset_index()
        .sort_values(["n_agents", "bucket_order"])
    )
    return summary, game_means.sort_values(["n_agents", "bucket_order", "game_label"])


def plot_bucket_utility(summary: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 5, figsize=(18.2, 3.8), sharey=True)
    bucket_orders = sorted(summary["bucket_order"].dropna().astype(int).unique())
    x_positions = np.arange(len(bucket_orders))

    for ax, n in zip(axes, N_ORDER):
        sub = summary[summary["n_agents"].eq(n)].sort_values("bucket_order")
        ax.bar(
            x_positions,
            sub["mean_utility"],
            color=BAR_COLOR,
            edgecolor="#333333",
            linewidth=0.6,
            alpha=0.86,
        )
        ax.set_title(f"N={n}", fontsize=14, pad=8)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(sub["bucket_label"].tolist(), fontsize=6.8, rotation=0)
        ax.tick_params(axis="y", labelsize=9)
        ax.grid(True, axis="y", alpha=0.22, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.set_ylim(40, 70)
        ax.set_yticks([40, 50, 60, 70])
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        if ax is axes[0]:
            ax.set_ylabel("Mean utility\n(avg. over games)", fontsize=11, labelpad=8)

    fig.supxlabel("Arena Elo bucket", fontsize=12, y=0.02)
    fig.suptitle("Heterogeneous utility by Elo bucket as group size increases", fontsize=16, y=1.04)
    fig.tight_layout(rect=(0, 0.04, 1, 0.98), w_pad=1.0)

    out_path = OVERLEAF_OUT / "heterogeneous_utility_by_elo_bucket_by_n.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    ITERATION_OUT.mkdir(parents=True, exist_ok=True)
    fig_copy = ITERATION_OUT / out_path.name
    fig_copy.write_bytes(out_path.read_bytes())
    return out_path


def main() -> None:
    OVERLEAF_OUT.mkdir(parents=True, exist_ok=True)
    ITERATION_OUT.mkdir(parents=True, exist_ok=True)

    agents = load_heterogeneous_agents()
    figure7_path = recreate_clean_figure7(agents)

    bucketed_agents, bucket_ranges = assign_elo_buckets(agents)
    summary, game_means = compute_bucket_summary(bucketed_agents)
    bucket_plot_path = plot_bucket_utility(summary)

    for out_dir in [OVERLEAF_OUT, ITERATION_OUT]:
        bucket_ranges.to_csv(out_dir / "heterogeneous_elo_bucket_ranges.csv", index=False)
        summary.to_csv(out_dir / "heterogeneous_utility_by_elo_bucket_by_n.csv", index=False)
        game_means.to_csv(out_dir / "heterogeneous_utility_by_elo_bucket_by_n_game_means.csv", index=False)

    print(f"Wrote {figure7_path}")
    print(f"Wrote {bucket_plot_path}")
    print(f"Wrote summaries to {OVERLEAF_OUT} and {ITERATION_OUT}")


if __name__ == "__main__":
    main()
