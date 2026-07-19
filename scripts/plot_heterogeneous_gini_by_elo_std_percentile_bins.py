#!/usr/bin/env python3
"""Binned bar plots of heterogeneous payoff Gini by Elo-std percentile bins."""

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
OUT_DIR = PROJECT_ROOT / "overleaf/neurips/graphics/n_gt_2_report"

N_ORDER = [2, 4, 6, 8, 10]
GAME_ORDER = ["game1", "game2", "game3"]
GAME_TITLES = {"game1": "Game 1", "game2": "Game 2", "game3": "Game 3"}
COMPETITION_ORDER = ["cooperative", "middle", "competitive"]
COMPETITION_TITLES = {
    "cooperative": "Low competition",
    "middle": "Medium competition",
    "competitive": "High competition",
}

N_BINS = 10
PERCENTILE_LABELS = [f"{10 * i}-{10 * (i + 1)}%" for i in range(N_BINS)]
BAR_COLOR = "#4E79A7"
BAR_EDGE = "#2F5F8F"


def sem(values: pd.Series) -> float:
    clean = values.replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 2:
        return math.nan
    return float(clean.std(ddof=1) / math.sqrt(len(clean)))


def gini_shifted_corrected(values: pd.Series | np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return math.nan
    if float(arr.min()) < 0:
        arr = arr - float(arr.min())
    if arr.size < 2 or np.allclose(arr, arr[0]) or np.allclose(arr, 0.0):
        return 0.0
    mean_value = float(arr.mean())
    if math.isclose(mean_value, 0.0):
        return 0.0
    raw_gini = float(np.mean(np.abs(arr[:, None] - arr[None, :])) / (2.0 * mean_value))
    return min(raw_gini * float(arr.size / (arr.size - 1)), 1.0)


def load_run_metrics() -> pd.DataFrame:
    agents = pd.read_csv(AGENT_TABLE)
    agents = agents[agents["experiment_family"].eq("heterogeneous_random")].copy()
    for col in ["n_agents", "elo", "final_utility", "competition_ci"]:
        agents[col] = pd.to_numeric(agents[col], errors="coerce")
    agents = agents.dropna(subset=["run_key", "game_label", "n_agents", "elo", "final_utility"]).copy()
    agents["n_agents"] = agents["n_agents"].astype(int)

    return (
        agents.groupby(
            [
                "run_key",
                "config_id",
                "game_label",
                "n_agents",
                "competition_ci",
                "competition_label_ci",
                "competition_band",
            ],
            dropna=False,
        )
        .agg(
            elo_std=("elo", lambda s: float(np.std(s.to_numpy(dtype=float), ddof=0))),
            elo_variance=("elo", lambda s: float(np.var(s.to_numpy(dtype=float), ddof=0))),
            mean_roster_elo=("elo", "mean"),
            max_roster_elo=("elo", "max"),
            payoff_gini_corrected=("final_utility", gini_shifted_corrected),
            average_payoff=("final_utility", "mean"),
            payoff_variance=("final_utility", lambda s: float(np.var(s.to_numpy(dtype=float), ddof=0))),
            n_agents_observed=("final_utility", "count"),
            model_count=("model", "nunique"),
        )
        .reset_index()
        .sort_values(["n_agents", "game_label", "config_id"])
    )


def subset_frame(frame: pd.DataFrame, filters: dict[str, object]) -> pd.DataFrame:
    out = frame
    for col, value in filters.items():
        out = out[out[col].eq(value)]
    return out


def add_percentile_bins(sub: pd.DataFrame) -> pd.DataFrame:
    out = sub.sort_values(["elo_std", "run_key"], kind="mergesort").reset_index(drop=True).copy()
    if out.empty:
        out["elo_std_percentile_bin"] = pd.Series(dtype=int)
        out["elo_std_percentile_label"] = pd.Series(dtype=str)
        return out
    out["elo_std_percentile_bin"] = np.minimum(
        np.floor(np.arange(len(out)) * N_BINS / len(out)).astype(int),
        N_BINS - 1,
    )
    out["elo_std_percentile_label"] = out["elo_std_percentile_bin"].map(
        {idx: label for idx, label in enumerate(PERCENTILE_LABELS)}
    )
    return out


def summarize_bins(frame: pd.DataFrame, filters: dict[str, object] | None = None) -> pd.DataFrame:
    if filters is None:
        filters = {}
    sub = add_percentile_bins(subset_frame(frame, filters))
    rows: list[dict[str, object]] = []
    for idx, label in enumerate(PERCENTILE_LABELS):
        bucket = sub[sub["elo_std_percentile_bin"].eq(idx)]
        rows.append(
            {
                "bin_index": idx,
                "elo_std_percentile_bin": label,
                "n_runs": int(len(bucket)),
                "elo_std_min": float(bucket["elo_std"].min()) if len(bucket) else math.nan,
                "elo_std_max": float(bucket["elo_std"].max()) if len(bucket) else math.nan,
                "elo_std_mean": float(bucket["elo_std"].mean()) if len(bucket) else math.nan,
                "payoff_gini_corrected_mean": float(bucket["payoff_gini_corrected"].mean()) if len(bucket) else math.nan,
                "payoff_gini_corrected_sem": sem(bucket["payoff_gini_corrected"]) if len(bucket) else math.nan,
                "payoff_gini_corrected_median": float(bucket["payoff_gini_corrected"].median()) if len(bucket) else math.nan,
                "average_payoff_mean": float(bucket["average_payoff"].mean()) if len(bucket) else math.nan,
            }
        )
    return pd.DataFrame(rows)


def y_max_for_summaries(summaries: list[pd.DataFrame]) -> float:
    values: list[float] = []
    for summary in summaries:
        mean = summary["payoff_gini_corrected_mean"].to_numpy(dtype=float)
        err = np.nan_to_num(summary["payoff_gini_corrected_sem"].to_numpy(dtype=float), nan=0.0)
        finite = np.isfinite(mean)
        if finite.any():
            values.append(float(np.nanmax(mean[finite] + err[finite])))
    if not values:
        return 0.05
    return min(1.0, max(values) * 1.22 + 0.015)


def draw_bar_axis(
    ax: plt.Axes,
    summary: pd.DataFrame,
    y_max: float,
    show_counts: bool = True,
) -> None:
    x = np.arange(N_BINS)
    mean = summary["payoff_gini_corrected_mean"].to_numpy(dtype=float)
    err = np.nan_to_num(summary["payoff_gini_corrected_sem"].to_numpy(dtype=float), nan=0.0)
    n_runs = summary["n_runs"].to_numpy(dtype=int)
    valid = np.isfinite(mean)
    ax.bar(
        x[valid],
        mean[valid],
        width=0.78,
        yerr=err[valid],
        capsize=2.0,
        color=BAR_COLOR,
        edgecolor=BAR_EDGE,
        linewidth=0.45,
        alpha=0.88,
    )
    if show_counts:
        for xpos, value, count in zip(x[valid], mean[valid], n_runs[valid], strict=False):
            if count > 0:
                ax.text(xpos, value + y_max * 0.025, f"n={count}", ha="center", va="bottom", fontsize=5.8, rotation=90)
    ax.set_ylim(0, y_max)
    ax.set_xticks(x)
    ax.set_xticklabels(PERCENTILE_LABELS, rotation=45, ha="right", fontsize=6.8)
    ax.grid(True, axis="y", alpha=0.24, linewidth=0.55)
    ax.tick_params(axis="y", labelsize=7.5)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def add_metadata(summary: pd.DataFrame, scope: str, row_label: str, col_label: str, filters: dict[str, object]) -> pd.DataFrame:
    out = summary.copy()
    out["scope"] = scope
    out["row_label"] = row_label
    out["col_label"] = col_label
    out["filters"] = ";".join(f"{key}={value}" for key, value in filters.items())
    return out


def plot_overall(frame: pd.DataFrame) -> tuple[Path, pd.DataFrame]:
    summary = summarize_bins(frame)
    y_max = y_max_for_summaries([summary])
    fig, ax = plt.subplots(figsize=(7.2, 4.9))
    draw_bar_axis(ax, summary, y_max, show_counts=True)
    ax.set_xlabel("Within-roster Elo standard deviation percentile bin", fontsize=10.5)
    ax.set_ylabel("Mean corrected payoff Gini", fontsize=10.5)
    ax.set_title("Heterogeneous runs: corrected payoff Gini by Elo-spread percentile", fontsize=13, pad=10)
    fig.tight_layout()
    out_path = OUT_DIR / "heterogeneous_payoff_gini_by_elo_std_percentile_bin_overall.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path, add_metadata(summary, "overall", "All", "All", {})


def plot_grid(
    frame: pd.DataFrame,
    row_groups: list[tuple[str, dict[str, object], dict[str, object]]],
    col_groups: list[tuple[str, dict[str, object], dict[str, object]]],
    scope: str,
    title: str,
    filename: str,
    figsize: tuple[float, float],
) -> tuple[Path, pd.DataFrame]:
    summaries: dict[tuple[int, int], pd.DataFrame] = {}
    summary_rows: list[pd.DataFrame] = []
    for row_idx, (row_label, row_filter, row_extra) in enumerate(row_groups):
        for col_idx, (col_label, col_filter, col_extra) in enumerate(col_groups):
            filters = {**row_filter, **col_filter}
            summary = summarize_bins(frame, filters)
            summaries[(row_idx, col_idx)] = summary
            metadata = add_metadata(summary, scope, row_label, col_label, filters)
            for key, value in {**row_extra, **col_extra}.items():
                metadata[key] = value
            summary_rows.append(metadata)

    y_max = y_max_for_summaries(list(summaries.values()))
    fig, axes = plt.subplots(len(row_groups), len(col_groups), figsize=figsize, sharex=True, sharey=True, squeeze=False)
    show_counts = len(row_groups) * len(col_groups) <= 5
    for row_idx, (row_label, _, _) in enumerate(row_groups):
        for col_idx, (col_label, _, _) in enumerate(col_groups):
            ax = axes[row_idx, col_idx]
            draw_bar_axis(ax, summaries[(row_idx, col_idx)], y_max, show_counts=show_counts)
            if row_idx == 0:
                ax.set_title(col_label, fontsize=9, pad=5)
            if col_idx == 0:
                ax.set_ylabel(f"{row_label}\nMean Gini" if len(row_groups) > 1 else "Mean Gini", fontsize=8.5)
            if row_idx == len(row_groups) - 1:
                ax.set_xlabel("Elo std percentile", fontsize=8.5)
            else:
                ax.tick_params(axis="x", labelbottom=False)
    fig.suptitle(title, fontsize=13, y=1.01)
    fig.tight_layout()
    out_path = OUT_DIR / filename
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path, pd.concat(summary_rows, ignore_index=True)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    run_metrics = load_run_metrics()

    paths: list[Path] = []
    summaries: list[pd.DataFrame] = []

    path, summary = plot_overall(run_metrics)
    paths.append(path)
    summaries.append(summary)

    n_cols = [(f"N={n}", {"n_agents": n}, {"n_agents": n}) for n in N_ORDER]
    game_cols = [(GAME_TITLES[g], {"game_label": g}, {"game_label": g}) for g in GAME_ORDER]
    comp_cols = [(COMPETITION_TITLES[c], {"competition_band": c}, {"competition_band": c}) for c in COMPETITION_ORDER]
    all_rows = [("All", {}, {})]

    specs = [
        (
            all_rows,
            game_cols,
            "by_game",
            "Heterogeneous corrected payoff Gini by Elo-spread percentile, by game",
            "heterogeneous_payoff_gini_by_elo_std_percentile_bin_by_game.png",
            (11.0, 4.0),
        ),
        (
            all_rows,
            n_cols,
            "by_n",
            "Heterogeneous corrected payoff Gini by Elo-spread percentile, by N",
            "heterogeneous_payoff_gini_by_elo_std_percentile_bin_by_n.png",
            (16.0, 4.0),
        ),
        (
            all_rows,
            comp_cols,
            "by_competition",
            "Heterogeneous corrected payoff Gini by Elo-spread percentile, by competition band",
            "heterogeneous_payoff_gini_by_elo_std_percentile_bin_by_competition.png",
            (11.0, 4.0),
        ),
        (
            game_cols,
            n_cols,
            "by_game_n",
            "Heterogeneous corrected payoff Gini by Elo-spread percentile, by game and N",
            "heterogeneous_payoff_gini_by_elo_std_percentile_bin_by_game_n.png",
            (16.0, 8.2),
        ),
        (
            comp_cols,
            game_cols,
            "by_competition_game",
            "Heterogeneous corrected payoff Gini by Elo-spread percentile, by competition band and game",
            "heterogeneous_payoff_gini_by_elo_std_percentile_bin_by_competition_game.png",
            (11.0, 8.2),
        ),
    ]

    for row_groups, col_groups, scope, title, filename, figsize in specs:
        path, summary = plot_grid(run_metrics, row_groups, col_groups, scope, title, filename, figsize)
        paths.append(path)
        summaries.append(summary)

    for n in N_ORDER:
        filtered = subset_frame(run_metrics, {"n_agents": n})
        path, summary = plot_grid(
            filtered,
            game_cols,
            comp_cols,
            f"by_game_competition_n{n}",
            f"Heterogeneous corrected payoff Gini by Elo-spread percentile, by game and competition band for N={n}",
            f"heterogeneous_payoff_gini_by_elo_std_percentile_bin_by_game_competition_n{n}.png",
            (11.0, 8.2),
        )
        summary["n_agents"] = n
        paths.append(path)
        summaries.append(summary)

    run_metrics_path = OUT_DIR / "heterogeneous_payoff_gini_by_elo_std_percentile_bin_run_metrics.csv"
    summary_path = OUT_DIR / "heterogeneous_payoff_gini_by_elo_std_percentile_bin_summary.csv"
    run_metrics.to_csv(run_metrics_path, index=False)
    pd.concat(summaries, ignore_index=True).to_csv(summary_path, index=False)

    for path in paths:
        print(f"Wrote {path}")
    print(f"Wrote {run_metrics_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
