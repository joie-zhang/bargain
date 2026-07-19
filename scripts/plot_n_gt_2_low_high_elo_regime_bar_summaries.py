#!/usr/bin/env python3
"""Summary-only bar plots for low- vs high-Elo heterogeneous rosters.

For a fraction f, bottom-f rosters contain only models from the bottom f of the
heterogeneous model roster by Arena Elo. Top-f rosters contain only models from
the top f. When the bottom and top sets overlap, the bar plots use the exclusive
comparison: bottom-only vs top-only.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AGENT_TABLE = (
    PROJECT_ROOT
    / "experiments/results/n2_plus_multiagent_comparison_analysis_20260505"
    / "tables_multiagent/heterogeneous_agents_fresh.csv"
)
OVERLEAF_OUT = PROJECT_ROOT / "overleaf/neurips/graphics/n_gt_2_report"
ITERATION_OUT = (
    PROJECT_ROOT
    / "experiments/results/figure_iteration_20260626/low_high_elo_regime_bar_summaries"
)

FRACTIONS = [0.60, 0.50]
N_ORDER = [2, 4, 6, 8, 10]
GAME_ORDER = ["game1", "game2", "game3"]
GAME_TITLES = {"game1": "Game 1", "game2": "Game 2", "game3": "Game 3"}
METRICS = [
    ("elo_variance", "Mean Elo variance"),
    ("payoff_variance", "Mean payoff variance"),
]
REGIME_ORDER = ["bottom", "top"]
COLORS = {"bottom": "#4E79A7", "top": "#E15759"}


def population_variance(values: pd.Series) -> float:
    arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if arr.size == 0:
        return math.nan
    return float(np.var(arr, ddof=0))


def sem(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) <= 1:
        return 0.0
    return float(clean.std(ddof=1) / math.sqrt(len(clean)))


def load_agents_and_models() -> tuple[pd.DataFrame, pd.DataFrame]:
    agents = pd.read_csv(AGENT_TABLE)
    agents = agents[agents["experiment_family"].eq("heterogeneous_random")].copy()
    for col in ["n_agents", "elo", "final_utility", "competition_ci"]:
        agents[col] = pd.to_numeric(agents[col], errors="coerce")
    agents = agents.dropna(subset=["run_key", "model", "model_short", "elo", "final_utility", "n_agents", "game_label"])
    agents["n_agents"] = agents["n_agents"].astype(int)
    agents["elo"] = agents["elo"].astype(int)

    models = (
        agents[["model", "model_short", "elo"]]
        .drop_duplicates()
        .sort_values(["elo", "model"])
        .reset_index(drop=True)
    )
    models["rank_1based"] = models.index + 1
    return agents, models


def build_run_metrics(agents: pd.DataFrame) -> pd.DataFrame:
    return (
        agents.groupby(["run_key", "game_label", "n_agents", "competition_ci", "competition_label_ci"], dropna=False)
        .agg(
            models=("model", lambda values: tuple(sorted(set(values)))),
            model_count=("model", "nunique"),
            mean_roster_elo=("elo", "mean"),
            elo_variance=("elo", population_variance),
            payoff_variance=("final_utility", population_variance),
            mean_payoff=("final_utility", "mean"),
        )
        .reset_index()
    )


def classify_runs(run_metrics: pd.DataFrame, models: pd.DataFrame, fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    cut_count = math.ceil(fraction * len(models))
    bottom_models = set(models.loc[models["rank_1based"].le(cut_count), "model"])
    top_models = set(models.loc[models["rank_1based"].ge(len(models) - cut_count + 1), "model"])

    model_flags = models.copy()
    pct = int(round(fraction * 100))
    model_flags[f"bottom{pct}_model"] = model_flags["model"].isin(bottom_models)
    model_flags[f"top{pct}_model"] = model_flags["model"].isin(top_models)

    frame = run_metrics.copy()
    frame["fraction"] = fraction
    frame["cut_count"] = cut_count
    frame["all_bottom"] = frame["models"].map(lambda values: all(model in bottom_models for model in values))
    frame["all_top"] = frame["models"].map(lambda values: all(model in top_models for model in values))
    frame["regime"] = np.select(
        [
            frame["all_bottom"] & ~frame["all_top"],
            frame["all_top"] & ~frame["all_bottom"],
            frame["all_bottom"] & frame["all_top"],
        ],
        ["bottom", "top", "overlap"],
        default="mixed",
    )
    return frame, model_flags


def summarize(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    summary = (
        frame[frame["regime"].isin(REGIME_ORDER)]
        .groupby(group_cols + ["regime"], dropna=False)
        .agg(
            run_count=("run_key", "count"),
            mean_roster_elo=("mean_roster_elo", "mean"),
            elo_variance=("elo_variance", "mean"),
            elo_variance_sem=("elo_variance", sem),
            payoff_variance=("payoff_variance", "mean"),
            payoff_variance_sem=("payoff_variance", sem),
            mean_payoff=("mean_payoff", "mean"),
        )
        .reset_index()
    )
    return summary


def delta_summary(summary: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grouped = summary.groupby(group_cols, dropna=False) if group_cols else [((), summary)]
    for keys, sub in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        if not {"bottom", "top"}.issubset(set(sub["regime"])):
            continue
        bottom = sub[sub["regime"].eq("bottom")].iloc[0]
        top = sub[sub["regime"].eq("top")].iloc[0]
        row = {col: key for col, key in zip(group_cols, keys)}
        row.update(
            {
                "bottom_runs": int(bottom["run_count"]),
                "top_runs": int(top["run_count"]),
                "bottom_mean_roster_elo": bottom["mean_roster_elo"],
                "top_mean_roster_elo": top["mean_roster_elo"],
                "bottom_elo_variance": bottom["elo_variance"],
                "top_elo_variance": top["elo_variance"],
                "elo_variance_diff_top_minus_bottom": top["elo_variance"] - bottom["elo_variance"],
                "elo_variance_pct_diff_vs_bottom": 100.0 * (top["elo_variance"] - bottom["elo_variance"]) / bottom["elo_variance"],
                "bottom_payoff_variance": bottom["payoff_variance"],
                "top_payoff_variance": top["payoff_variance"],
                "payoff_variance_diff_top_minus_bottom": top["payoff_variance"] - bottom["payoff_variance"],
                "payoff_variance_pct_diff_vs_bottom": 100.0 * (top["payoff_variance"] - bottom["payoff_variance"]) / bottom["payoff_variance"],
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def annotate_bar(ax: plt.Axes, bar: plt.Rectangle, value: float, run_count: int) -> None:
    ymax = ax.get_ylim()[1]
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + ymax * 0.025,
        f"{value:.1f}\nn={run_count}",
        ha="center",
        va="bottom",
        fontsize=8,
    )


def plot_aggregate(summary: pd.DataFrame, fraction: float) -> Path:
    pct = int(round(fraction * 100))
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.8))
    labels = [f"Bottom {pct}%", f"Top {pct}%"]
    for ax, (metric, title) in zip(axes, METRICS):
        sub = summary.set_index("regime").loc[REGIME_ORDER].reset_index()
        values = sub[metric].to_numpy(dtype=float)
        errors = sub[f"{metric}_sem"].to_numpy(dtype=float)
        bars = ax.bar(
            [0, 1],
            values,
            yerr=errors,
            color=[COLORS["bottom"], COLORS["top"]],
            alpha=0.86,
            edgecolor="#333333",
            linewidth=0.6,
            capsize=4,
        )
        ax.set_title(title, fontsize=12)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylim(0, max(values + errors) * 1.28)
        ax.grid(True, axis="y", alpha=0.22, linewidth=0.6)
        ax.set_axisbelow(True)
        for bar, value, run_count in zip(bars, values, sub["run_count"].astype(int)):
            annotate_bar(ax, bar, value, run_count)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
    fig.suptitle(f"Bottom vs top {pct}% Elo rosters: final means", fontsize=14, y=1.03)
    fig.tight_layout()
    out_path = OVERLEAF_OUT / f"heterogeneous_low_high_elo_regime_{pct}_aggregate_bar.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_breakdown(summary: pd.DataFrame, fraction: float, group_col: str, group_order: list[object], title_prefix: str) -> Path:
    pct = int(round(fraction * 100))
    fig, axes = plt.subplots(2, len(group_order), figsize=(3.15 * len(group_order), 6.3), sharey="row")
    if len(group_order) == 1:
        axes = np.asarray(axes).reshape(2, 1)
    width = 0.70
    for row_idx, (metric, metric_title) in enumerate(METRICS):
        row_values = summary[metric].to_numpy(dtype=float)
        row_errors = summary[f"{metric}_sem"].to_numpy(dtype=float)
        row_max = max(row_values + row_errors) * 1.32
        for col_idx, group_value in enumerate(group_order):
            ax = axes[row_idx, col_idx]
            sub = summary[summary[group_col].eq(group_value)].set_index("regime")
            if not set(REGIME_ORDER).issubset(sub.index):
                ax.set_visible(False)
                continue
            sub = sub.loc[REGIME_ORDER].reset_index()
            values = sub[metric].to_numpy(dtype=float)
            errors = sub[f"{metric}_sem"].to_numpy(dtype=float)
            bars = ax.bar(
                [0, 1],
                values,
                width=width,
                yerr=errors,
                color=[COLORS["bottom"], COLORS["top"]],
                alpha=0.86,
                edgecolor="#333333",
                linewidth=0.6,
                capsize=3,
            )
            ax.set_ylim(0, row_max)
            if row_idx == 0:
                title = f"N={group_value}" if group_col == "n_agents" else GAME_TITLES[str(group_value)]
                ax.set_title(title, fontsize=11)
            if col_idx == 0:
                ax.set_ylabel(metric_title, fontsize=10)
            ax.set_xticks([0, 1])
            ax.set_xticklabels([f"Bot {pct}", f"Top {pct}"], fontsize=8)
            ax.grid(True, axis="y", alpha=0.22, linewidth=0.6)
            ax.set_axisbelow(True)
            for bar, value, run_count in zip(bars, values, sub["run_count"].astype(int)):
                annotate_bar(ax, bar, value, run_count)
            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)
    fig.suptitle(f"{title_prefix}: bottom vs top {pct}% Elo rosters", fontsize=14, y=1.01)
    fig.tight_layout()
    suffix = "by_n" if group_col == "n_agents" else "by_game"
    out_path = OVERLEAF_OUT / f"heterogeneous_low_high_elo_regime_{pct}_{suffix}_bar.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_frame(frame: pd.DataFrame, filename: str) -> None:
    for out_dir in [OVERLEAF_OUT, ITERATION_OUT]:
        out_dir.mkdir(parents=True, exist_ok=True)
        frame.to_csv(out_dir / filename, index=False)


def main() -> None:
    OVERLEAF_OUT.mkdir(parents=True, exist_ok=True)
    ITERATION_OUT.mkdir(parents=True, exist_ok=True)
    agents, models = load_agents_and_models()
    base_run_metrics = build_run_metrics(agents)
    plot_paths: list[Path] = []
    all_deltas: list[pd.DataFrame] = []

    for fraction in FRACTIONS:
        pct = int(round(fraction * 100))
        run_metrics, model_flags = classify_runs(base_run_metrics, models, fraction)
        aggregate = summarize(run_metrics, [])
        by_n = summarize(run_metrics, ["n_agents"])
        by_game = summarize(run_metrics, ["game_label"])
        aggregate_delta = delta_summary(aggregate, [])
        by_n_delta = delta_summary(by_n, ["n_agents"])
        by_game_delta = delta_summary(by_game, ["game_label"])
        for frame, scope in [(aggregate_delta, "aggregate"), (by_n_delta, "by_n"), (by_game_delta, "by_game")]:
            if not frame.empty:
                frame.insert(0, "fraction", fraction)
                frame.insert(1, "scope", scope)
                all_deltas.append(frame)

        plot_paths.extend(
            [
                plot_aggregate(aggregate, fraction),
                plot_breakdown(by_n, fraction, "n_agents", N_ORDER, "By N"),
                plot_breakdown(by_game, fraction, "game_label", GAME_ORDER, "By game"),
            ]
        )

        save_frame(model_flags, f"heterogeneous_low_high_elo_regime_{pct}_model_flags.csv")
        save_frame(
            run_metrics.drop(columns=["models"]),
            f"heterogeneous_low_high_elo_regime_{pct}_run_metrics.csv",
        )
        save_frame(aggregate, f"heterogeneous_low_high_elo_regime_{pct}_aggregate_bar_summary.csv")
        save_frame(by_n, f"heterogeneous_low_high_elo_regime_{pct}_by_n_bar_summary.csv")
        save_frame(by_game, f"heterogeneous_low_high_elo_regime_{pct}_by_game_bar_summary.csv")
        save_frame(aggregate_delta, f"heterogeneous_low_high_elo_regime_{pct}_aggregate_delta.csv")
        save_frame(by_n_delta, f"heterogeneous_low_high_elo_regime_{pct}_by_n_delta.csv")
        save_frame(by_game_delta, f"heterogeneous_low_high_elo_regime_{pct}_by_game_delta.csv")

        print(f"\nBottom/top {pct}% aggregate delta:")
        print(aggregate_delta.to_string(index=False))

    if all_deltas:
        combined_delta = pd.concat(all_deltas, ignore_index=True, sort=False)
        save_frame(combined_delta, "heterogeneous_low_high_elo_regime_50_60_combined_delta.csv")

    for path in plot_paths:
        (ITERATION_OUT / path.name).write_bytes(path.read_bytes())
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
