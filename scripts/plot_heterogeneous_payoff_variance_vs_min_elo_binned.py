#!/usr/bin/env python3
"""Binned bar versions of payoff variance vs minimum roster Elo."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "overleaf/neurips/graphics/n_gt_2_report"
RUN_METRICS_PATH = OUT_DIR / "heterogeneous_payoff_variance_vs_min_elo_run_metrics.csv"

N_ORDER = [2, 4, 6, 8, 10]
N_FOCUS_ORDER = [8, 10]
GAME_ORDER = ["game1", "game2", "game3"]
GAME_TITLES = {"game1": "Game 1", "game2": "Game 2", "game3": "Game 3"}
COMPETITION_ORDER = ["cooperative", "middle", "competitive"]
COMPETITION_TITLES = {
    "cooperative": "Low competition",
    "middle": "Medium competition",
    "competitive": "High competition",
}

# Fixed bins keep the x-axis comparable across all breakdowns.
BIN_EDGES = [1235, 1280, 1320, 1360, 1400, 1440, 1505]
BIN_LABELS = ["1235-1280", "1280-1320", "1320-1360", "1360-1400", "1400-1440", "1440-1505"]
BAR_COLOR = "#4E79A7"


def sem(values: pd.Series) -> float:
    clean = values.replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 2:
        return math.nan
    return float(clean.std(ddof=1) / math.sqrt(len(clean)))


def load_run_metrics() -> pd.DataFrame:
    if not RUN_METRICS_PATH.exists():
        raise FileNotFoundError(
            f"Missing {RUN_METRICS_PATH}. Run plot_heterogeneous_payoff_variance_vs_min_elo.py first."
        )
    runs = pd.read_csv(RUN_METRICS_PATH)
    runs["min_elo_bin"] = pd.cut(
        runs["min_roster_elo"],
        bins=BIN_EDGES,
        labels=BIN_LABELS,
        include_lowest=True,
        right=False,
    )
    return runs


def subset_frame(frame: pd.DataFrame, filters: dict[str, object]) -> pd.DataFrame:
    out = frame
    for col, value in filters.items():
        out = out[out[col].eq(value)]
    return out


def summarize_bins(frame: pd.DataFrame, filters: dict[str, object] | None = None) -> pd.DataFrame:
    if filters is None:
        filters = {}
    sub = subset_frame(frame, filters)
    rows: list[dict[str, object]] = []
    for idx, label in enumerate(BIN_LABELS):
        bucket = sub[sub["min_elo_bin"].astype(str).eq(label)]
        rows.append(
            {
                "bin_sort": idx,
                "min_elo_bin": label,
                "bin_left": BIN_EDGES[idx],
                "bin_right": BIN_EDGES[idx + 1],
                "n_runs": int(len(bucket)),
                "payoff_variance_mean": float(bucket["payoff_variance"].mean()) if len(bucket) else math.nan,
                "payoff_variance_sem": sem(bucket["payoff_variance"]) if len(bucket) else math.nan,
                "payoff_variance_median": float(bucket["payoff_variance"].median()) if len(bucket) else math.nan,
                "average_payoff_mean": float(bucket["average_payoff"].mean()) if len(bucket) else math.nan,
            }
        )
    return pd.DataFrame(rows)


def draw_binned_axis(
    ax: plt.Axes,
    summary: pd.DataFrame,
    y_max: float,
    show_counts: bool = True,
) -> None:
    summary = summary.sort_values("bin_sort").reset_index(drop=True)
    x = np.arange(len(summary))
    means = summary["payoff_variance_mean"].to_numpy(dtype=float)
    sems = np.nan_to_num(summary["payoff_variance_sem"].to_numpy(dtype=float), nan=0.0)
    counts = summary["n_runs"].to_numpy(dtype=int)
    valid = np.isfinite(means)
    ax.bar(
        x[valid],
        means[valid],
        yerr=sems[valid],
        capsize=2.5,
        color=BAR_COLOR,
        alpha=0.86,
        width=0.72,
    )
    if show_counts:
        for xpos, mean, sem_value, n_runs in zip(x, means, sems, counts, strict=False):
            if np.isfinite(mean) and n_runs > 0:
                ax.text(
                    xpos,
                    mean + sem_value + y_max * 0.018,
                    f"n={n_runs}",
                    ha="center",
                    va="bottom",
                    fontsize=6.5,
                    rotation=90 if y_max < 200 else 0,
                )
    ax.set_ylim(0, y_max)
    ax.set_xticks(x)
    ax.set_xticklabels(BIN_LABELS, rotation=35, ha="right", fontsize=7.5)
    ax.grid(True, axis="y", alpha=0.23, linewidth=0.55)
    ax.tick_params(axis="y", labelsize=7.5)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def y_max_from_summaries(summaries: list[pd.DataFrame]) -> float:
    values: list[float] = []
    for summary in summaries:
        mean = summary["payoff_variance_mean"].to_numpy(dtype=float)
        err = np.nan_to_num(summary["payoff_variance_sem"].to_numpy(dtype=float), nan=0.0)
        finite = np.isfinite(mean)
        if finite.any():
            values.append(float(np.nanmax(mean[finite] + err[finite])))
    if not values:
        return 1.0
    return max(values) * 1.22


def plot_overall(
    frame: pd.DataFrame,
    title: str = "Heterogeneous runs: payoff variance by minimum roster Elo bucket",
    filename: str = "heterogeneous_payoff_variance_vs_min_elo_binned_overall.png",
    scope: str = "overall",
) -> tuple[Path, pd.DataFrame]:
    summary = summarize_bins(frame)
    y_max = y_max_from_summaries([summary])
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    draw_binned_axis(ax, summary, y_max)
    ax.set_xlabel("Minimum roster Elo bucket", fontsize=10)
    ax.set_ylabel("Mean payoff variance", fontsize=10)
    ax.set_title(title, fontsize=13, pad=10)
    fig.tight_layout()
    out_path = OUT_DIR / filename
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    summary["scope"] = scope
    summary["row_label"] = "All"
    summary["col_label"] = "All"
    return out_path, summary


def plot_grid(
    frame: pd.DataFrame,
    row_groups: list[tuple[str, dict[str, object], dict[str, object]]],
    col_groups: list[tuple[str, dict[str, object], dict[str, object]]],
    scope: str,
    title: str,
    filename: str,
    figsize: tuple[float, float],
) -> tuple[Path, pd.DataFrame]:
    summaries: list[pd.DataFrame] = []
    cells: list[tuple[str, str, pd.DataFrame]] = []
    for row_label, row_filter, row_extra in row_groups:
        for col_label, col_filter, col_extra in col_groups:
            filters = {**row_filter, **col_filter}
            summary = summarize_bins(frame, filters)
            summary["scope"] = scope
            summary["row_label"] = row_label
            summary["col_label"] = col_label
            for key, value in {**row_extra, **col_extra}.items():
                summary[key] = value
            summaries.append(summary)
            cells.append((row_label, col_label, summary))

    y_max = y_max_from_summaries(summaries)
    fig, axes = plt.subplots(len(row_groups), len(col_groups), figsize=figsize, sharey=True, squeeze=False)
    for row_idx, (row_label, _, _) in enumerate(row_groups):
        for col_idx, (col_label, _, _) in enumerate(col_groups):
            ax = axes[row_idx, col_idx]
            summary = next(s for r, c, s in cells if r == row_label and c == col_label)
            draw_binned_axis(ax, summary, y_max, show_counts=True)
            if row_idx == 0:
                ax.set_title(col_label, fontsize=9, pad=5)
            if col_idx == 0:
                ax.set_ylabel(f"{row_label}\nMean payoff variance" if len(row_groups) > 1 else "Mean payoff variance", fontsize=8.5)
            if row_idx == len(row_groups) - 1:
                ax.set_xlabel("Min Elo bucket", fontsize=8.5)
            else:
                ax.tick_params(axis="x", labelbottom=False)
    fig.suptitle(title, fontsize=13, y=1.01)
    fig.tight_layout()
    out_path = OUT_DIR / filename
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path, pd.concat(summaries, ignore_index=True)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = load_run_metrics()
    paths: list[Path] = []
    summaries: list[pd.DataFrame] = []

    path, summary = plot_overall(frame)
    paths.append(path)
    summaries.append(summary)

    n_cols = [(f"N={n}", {"n_agents": n}, {"n_agents": n}) for n in N_ORDER]
    game_cols = [(GAME_TITLES[g], {"game_label": g}, {"game_label": g}) for g in GAME_ORDER]
    comp_cols = [(COMPETITION_TITLES[c], {"competition_band": c}, {"competition_band": c}) for c in COMPETITION_ORDER]
    all_rows = [("All", {}, {})]

    plot_specs = [
        (
            all_rows,
            game_cols,
            "by_game",
            "Heterogeneous payoff variance by min-Elo bucket, by game",
            "heterogeneous_payoff_variance_vs_min_elo_binned_by_game.png",
            (11.0, 4.0),
        ),
        (
            all_rows,
            n_cols,
            "by_n",
            "Heterogeneous payoff variance by min-Elo bucket, by N",
            "heterogeneous_payoff_variance_vs_min_elo_binned_by_n.png",
            (16.0, 4.0),
        ),
        (
            all_rows,
            comp_cols,
            "by_competition",
            "Heterogeneous payoff variance by min-Elo bucket, by competition band",
            "heterogeneous_payoff_variance_vs_min_elo_binned_by_competition.png",
            (11.0, 4.0),
        ),
        (
            game_cols,
            n_cols,
            "by_game_n",
            "Heterogeneous payoff variance by min-Elo bucket, by game and N",
            "heterogeneous_payoff_variance_vs_min_elo_binned_by_game_n.png",
            (16.0, 8.0),
        ),
        (
            comp_cols,
            game_cols,
            "by_competition_game",
            "Heterogeneous payoff variance by min-Elo bucket, by competition band and game",
            "heterogeneous_payoff_variance_vs_min_elo_binned_by_competition_game.png",
            (11.0, 8.0),
        ),
    ]

    for row_groups, col_groups, scope, title, filename, figsize in plot_specs:
        path, summary = plot_grid(frame, row_groups, col_groups, scope, title, filename, figsize)
        paths.append(path)
        summaries.append(summary)

    for n in N_ORDER:
        path, summary = plot_grid(
            subset_frame(frame, {"n_agents": n}),
            game_cols,
            comp_cols,
            f"by_game_competition_n{n}",
            f"Heterogeneous payoff variance by min-Elo bucket, by game and competition band for N={n}",
            f"heterogeneous_payoff_variance_vs_min_elo_binned_by_game_competition_n{n}.png",
            (11.0, 8.0),
        )
        summary["n_agents"] = n
        paths.append(path)
        summaries.append(summary)

    summary_path = OUT_DIR / "heterogeneous_payoff_variance_vs_min_elo_binned_summary.csv"
    pd.concat(summaries, ignore_index=True).to_csv(summary_path, index=False)
    for path in paths:
        print(f"Wrote {path}")
    print(f"Wrote {summary_path}")

    focus_frame = frame[frame["n_agents"].isin(N_FOCUS_ORDER)].copy()
    focus_paths: list[Path] = []
    focus_summaries: list[pd.DataFrame] = []

    path, summary = plot_overall(
        focus_frame,
        title="Heterogeneous N=8/10 runs: payoff variance by minimum roster Elo bucket",
        filename="heterogeneous_payoff_variance_vs_min_elo_binned_n8_n10_overall.png",
        scope="overall_n8_n10",
    )
    focus_paths.append(path)
    focus_summaries.append(summary)

    focus_n_cols = [(f"N={n}", {"n_agents": n}, {"n_agents": n}) for n in N_FOCUS_ORDER]
    focus_plot_specs = [
        (
            all_rows,
            game_cols,
            "by_game_n8_n10",
            "Heterogeneous N=8/10 payoff variance by min-Elo bucket, by game",
            "heterogeneous_payoff_variance_vs_min_elo_binned_n8_n10_by_game.png",
            (11.0, 4.0),
        ),
        (
            all_rows,
            focus_n_cols,
            "by_n_n8_n10",
            "Heterogeneous N=8/10 payoff variance by min-Elo bucket, by N",
            "heterogeneous_payoff_variance_vs_min_elo_binned_n8_n10_by_n.png",
            (8.0, 4.0),
        ),
        (
            all_rows,
            comp_cols,
            "by_competition_n8_n10",
            "Heterogeneous N=8/10 payoff variance by min-Elo bucket, by competition band",
            "heterogeneous_payoff_variance_vs_min_elo_binned_n8_n10_by_competition.png",
            (11.0, 4.0),
        ),
        (
            game_cols,
            focus_n_cols,
            "by_game_n_n8_n10",
            "Heterogeneous N=8/10 payoff variance by min-Elo bucket, by game and N",
            "heterogeneous_payoff_variance_vs_min_elo_binned_n8_n10_by_game_n.png",
            (8.0, 8.0),
        ),
        (
            comp_cols,
            game_cols,
            "by_competition_game_n8_n10",
            "Heterogeneous N=8/10 payoff variance by min-Elo bucket, by competition band and game",
            "heterogeneous_payoff_variance_vs_min_elo_binned_n8_n10_by_competition_game.png",
            (11.0, 8.0),
        ),
    ]

    for row_groups, col_groups, scope, title, filename, figsize in focus_plot_specs:
        path, summary = plot_grid(focus_frame, row_groups, col_groups, scope, title, filename, figsize)
        focus_paths.append(path)
        focus_summaries.append(summary)

    for n in N_FOCUS_ORDER:
        path, summary = plot_grid(
            subset_frame(focus_frame, {"n_agents": n}),
            game_cols,
            comp_cols,
            f"by_game_competition_n{n}_n8_n10",
            f"Heterogeneous N={n} payoff variance by min-Elo bucket, by game and competition band",
            f"heterogeneous_payoff_variance_vs_min_elo_binned_n8_n10_by_game_competition_n{n}.png",
            (11.0, 8.0),
        )
        summary["n_agents"] = n
        focus_paths.append(path)
        focus_summaries.append(summary)

    focus_summary_path = OUT_DIR / "heterogeneous_payoff_variance_vs_min_elo_binned_n8_n10_summary.csv"
    pd.concat(focus_summaries, ignore_index=True).to_csv(focus_summary_path, index=False)
    for path in focus_paths:
        print(f"Wrote {path}")
    print(f"Wrote {focus_summary_path}")


if __name__ == "__main__":
    main()
