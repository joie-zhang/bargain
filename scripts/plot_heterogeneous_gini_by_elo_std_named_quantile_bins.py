#!/usr/bin/env python3
"""Bar plots of heterogeneous payoff Gini by Elo-std tertiles/quartiles/quintiles."""

from __future__ import annotations

import math
from dataclasses import dataclass
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

BAR_COLOR = "#4E79A7"
BAR_EDGE = "#2F5F8F"


@dataclass(frozen=True)
class BinConfig:
    n_bins: int
    name_singular: str
    name_plural: str
    file_tag: str
    short_prefix: str


BIN_CONFIGS = [
    BinConfig(3, "tertile", "tertiles", "tertile", "T"),
    BinConfig(4, "quartile", "quartiles", "quartile", "Q"),
    BinConfig(5, "quintile", "quintiles", "quintile", "Qn"),
]


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


def percentile_bounds(config: BinConfig, idx: int) -> tuple[float, float]:
    return 100.0 * idx / config.n_bins, 100.0 * (idx + 1) / config.n_bins


def bin_label(config: BinConfig, idx: int) -> str:
    left, right = percentile_bounds(config, idx)
    return f"{config.name_singular.title()} {idx + 1}\n{left:.0f}-{right:.0f}%"


def add_quantile_bins(sub: pd.DataFrame, config: BinConfig) -> pd.DataFrame:
    out = sub.sort_values(["elo_std", "run_key"], kind="mergesort").reset_index(drop=True).copy()
    if out.empty:
        out["elo_std_quantile_bin"] = pd.Series(dtype=int)
        out["elo_std_quantile_label"] = pd.Series(dtype=str)
        return out
    out["elo_std_quantile_bin"] = np.minimum(
        np.floor(np.arange(len(out)) * config.n_bins / len(out)).astype(int),
        config.n_bins - 1,
    )
    out["elo_std_quantile_label"] = out["elo_std_quantile_bin"].map(
        {idx: bin_label(config, idx) for idx in range(config.n_bins)}
    )
    return out


def summarize_bins(
    frame: pd.DataFrame,
    config: BinConfig,
    filters: dict[str, object] | None = None,
) -> pd.DataFrame:
    if filters is None:
        filters = {}
    sub = add_quantile_bins(subset_frame(frame, filters), config)
    rows: list[dict[str, object]] = []
    for idx in range(config.n_bins):
        bucket = sub[sub["elo_std_quantile_bin"].eq(idx)]
        pct_left, pct_right = percentile_bounds(config, idx)
        rows.append(
            {
                "bin_family": config.name_singular,
                "bin_index": idx,
                "elo_std_quantile_label": bin_label(config, idx).replace("\n", " "),
                "percentile_left": pct_left,
                "percentile_right": pct_right,
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
    config: BinConfig,
    y_max: float,
    show_counts: bool = True,
) -> None:
    x = np.arange(config.n_bins)
    mean = summary["payoff_gini_corrected_mean"].to_numpy(dtype=float)
    err = np.nan_to_num(summary["payoff_gini_corrected_sem"].to_numpy(dtype=float), nan=0.0)
    n_runs = summary["n_runs"].to_numpy(dtype=int)
    valid = np.isfinite(mean)
    ax.bar(
        x[valid],
        mean[valid],
        width=0.72,
        yerr=err[valid],
        capsize=2.4,
        color=BAR_COLOR,
        edgecolor=BAR_EDGE,
        linewidth=0.45,
        alpha=0.88,
    )
    if show_counts:
        for xpos, value, count in zip(x[valid], mean[valid], n_runs[valid], strict=False):
            if count > 0:
                ax.text(xpos, value + y_max * 0.025, f"n={count}", ha="center", va="bottom", fontsize=6.2, rotation=90)
    ax.set_ylim(0, y_max)
    ax.set_xticks(x)
    ax.set_xticklabels([bin_label(config, idx) for idx in range(config.n_bins)], rotation=35, ha="right", fontsize=7)
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


def plot_overall(frame: pd.DataFrame, config: BinConfig) -> tuple[Path, pd.DataFrame]:
    summary = summarize_bins(frame, config)
    y_max = y_max_for_summaries([summary])
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    draw_bar_axis(ax, summary, config, y_max, show_counts=True)
    ax.set_xlabel(f"Within-roster Elo standard deviation {config.name_singular}", fontsize=10.5)
    ax.set_ylabel("Mean corrected payoff Gini", fontsize=10.5)
    ax.set_title(
        f"Heterogeneous runs: corrected payoff Gini by Elo-spread {config.name_singular}",
        fontsize=13,
        pad=10,
    )
    fig.tight_layout()
    out_path = OUT_DIR / f"heterogeneous_payoff_gini_by_elo_std_{config.file_tag}_overall.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path, add_metadata(summary, "overall", "All", "All", {})


def plot_grid(
    frame: pd.DataFrame,
    config: BinConfig,
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
            summary = summarize_bins(frame, config, filters)
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
            draw_bar_axis(ax, summaries[(row_idx, col_idx)], config, y_max, show_counts=show_counts)
            if row_idx == 0:
                ax.set_title(col_label, fontsize=9, pad=5)
            if col_idx == 0:
                ax.set_ylabel(f"{row_label}\nMean Gini" if len(row_groups) > 1 else "Mean Gini", fontsize=8.5)
            if row_idx == len(row_groups) - 1:
                ax.set_xlabel(config.name_singular.title(), fontsize=8.5)
            else:
                ax.tick_params(axis="x", labelbottom=False)
    fig.suptitle(title, fontsize=13, y=1.01)
    fig.tight_layout()
    out_path = OUT_DIR / filename
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path, pd.concat(summary_rows, ignore_index=True)


def make_outputs_for_config(run_metrics: pd.DataFrame, config: BinConfig) -> tuple[list[Path], pd.DataFrame]:
    paths: list[Path] = []
    summaries: list[pd.DataFrame] = []

    path, summary = plot_overall(run_metrics, config)
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
            f"Heterogeneous corrected payoff Gini by Elo-spread {config.name_singular}, by game",
            f"heterogeneous_payoff_gini_by_elo_std_{config.file_tag}_by_game.png",
            (10.2, 4.0),
        ),
        (
            all_rows,
            n_cols,
            "by_n",
            f"Heterogeneous corrected payoff Gini by Elo-spread {config.name_singular}, by N",
            f"heterogeneous_payoff_gini_by_elo_std_{config.file_tag}_by_n.png",
            (14.6, 4.0),
        ),
        (
            all_rows,
            comp_cols,
            "by_competition",
            f"Heterogeneous corrected payoff Gini by Elo-spread {config.name_singular}, by competition band",
            f"heterogeneous_payoff_gini_by_elo_std_{config.file_tag}_by_competition.png",
            (10.2, 4.0),
        ),
        (
            game_cols,
            n_cols,
            "by_game_n",
            f"Heterogeneous corrected payoff Gini by Elo-spread {config.name_singular}, by game and N",
            f"heterogeneous_payoff_gini_by_elo_std_{config.file_tag}_by_game_n.png",
            (14.6, 8.0),
        ),
        (
            comp_cols,
            game_cols,
            "by_competition_game",
            f"Heterogeneous corrected payoff Gini by Elo-spread {config.name_singular}, by competition band and game",
            f"heterogeneous_payoff_gini_by_elo_std_{config.file_tag}_by_competition_game.png",
            (10.2, 8.0),
        ),
    ]

    for row_groups, col_groups, scope, title, filename, figsize in specs:
        path, summary = plot_grid(run_metrics, config, row_groups, col_groups, scope, title, filename, figsize)
        paths.append(path)
        summaries.append(summary)

    for n in N_ORDER:
        filtered = subset_frame(run_metrics, {"n_agents": n})
        path, summary = plot_grid(
            filtered,
            config,
            game_cols,
            comp_cols,
            f"by_game_competition_n{n}",
            f"Heterogeneous corrected payoff Gini by Elo-spread {config.name_singular}, by game and competition band for N={n}",
            f"heterogeneous_payoff_gini_by_elo_std_{config.file_tag}_by_game_competition_n{n}.png",
            (10.2, 8.0),
        )
        summary["n_agents"] = n
        paths.append(path)
        summaries.append(summary)

    return paths, pd.concat(summaries, ignore_index=True)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    run_metrics = load_run_metrics()
    run_metrics_path = OUT_DIR / "heterogeneous_payoff_gini_by_elo_std_named_quantile_bin_run_metrics.csv"
    run_metrics.to_csv(run_metrics_path, index=False)

    all_paths: list[Path] = []
    all_summaries: list[pd.DataFrame] = []
    for config in BIN_CONFIGS:
        paths, summary = make_outputs_for_config(run_metrics, config)
        summary_path = OUT_DIR / f"heterogeneous_payoff_gini_by_elo_std_{config.file_tag}_summary.csv"
        summary.to_csv(summary_path, index=False)
        all_paths.extend(paths)
        all_paths.append(summary_path)
        all_summaries.append(summary)

    combined_summary_path = OUT_DIR / "heterogeneous_payoff_gini_by_elo_std_named_quantile_bin_summary.csv"
    pd.concat(all_summaries, ignore_index=True).to_csv(combined_summary_path, index=False)

    for path in all_paths:
        print(f"Wrote {path}")
    print(f"Wrote {run_metrics_path}")
    print(f"Wrote {combined_summary_path}")


if __name__ == "__main__":
    main()
