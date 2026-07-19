#!/usr/bin/env python3
"""Plot group payoff variance conditional on each model appearing in the group.

Each point is a model. The x-axis is that model's Arena Elo. The y-axis is the
mean within-run payoff variance among all agents in heterogeneous runs that
include the model. This is not the variance of the model's own payoff over time.
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
    / "experiments/results/figure_iteration_20260626/included_model_group_payoff_variance_vs_elo"
)

N_ORDER = [2, 4, 6, 8, 10]
N_COLORS = {
    2: "#4E79A7",
    4: "#F28E2B",
    6: "#59A14F",
    8: "#B07AA1",
    10: "#E15759",
}
GAME_ORDER = ["game1", "game2", "game3"]
GAME_TITLES = {
    "game1": "Game 1",
    "game2": "Game 2",
    "game3": "Game 3",
}
GAME_COLORS = {
    "game1": "#4E79A7",
    "game2": "#59A14F",
    "game3": "#E15759",
}


def population_variance(values: pd.Series) -> float:
    arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if arr.size == 0:
        return math.nan
    return float(np.var(arr, ddof=0))


def load_model_run_inclusions() -> pd.DataFrame:
    agents = pd.read_csv(AGENT_TABLE)
    agents = agents[agents["experiment_family"].eq("heterogeneous_random")].copy()
    for col in ["n_agents", "elo", "final_utility", "competition_ci"]:
        agents[col] = pd.to_numeric(agents[col], errors="coerce")
    agents = agents.dropna(subset=["run_key", "model", "model_short", "elo", "final_utility", "n_agents", "game_label"])
    agents["n_agents"] = agents["n_agents"].astype(int)
    agents["elo"] = agents["elo"].astype(int)

    run_vars = (
        agents.groupby(["run_key", "game_label", "n_agents", "competition_ci", "competition_label_ci"], dropna=False)
        .agg(
            group_payoff_variance=("final_utility", population_variance),
            group_mean_payoff=("final_utility", "mean"),
            agents_in_run=("final_utility", "count"),
        )
        .reset_index()
    )
    inclusions = agents[["run_key", "model", "model_short", "elo"]].drop_duplicates()
    return inclusions.merge(run_vars, on="run_key", how="inner").sort_values(["model", "run_key"])


def fit_line(frame: pd.DataFrame, y_col: str = "mean_group_payoff_variance") -> dict[str, float]:
    data = frame[["elo", y_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < 2 or data["elo"].nunique() < 2:
        return {"slope": math.nan, "intercept": math.nan, "pearson_r": math.nan, "r_squared": math.nan}
    x = data["elo"].to_numpy(dtype=float)
    y = data[y_col].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    pearson_r = float(np.corrcoef(x, y)[0, 1])
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "pearson_r": pearson_r,
        "r_squared": pearson_r * pearson_r,
    }


def add_fit(ax: plt.Axes, frame: pd.DataFrame, y_col: str = "mean_group_payoff_variance") -> dict[str, float]:
    fit = fit_line(frame, y_col=y_col)
    if math.isfinite(fit["slope"]):
        xs = np.linspace(float(frame["elo"].min()), float(frame["elo"].max()), 160)
        ys = fit["slope"] * xs + fit["intercept"]
        ax.plot(xs, ys, color="#111111", lw=1.8, alpha=0.9)
    ax.text(
        0.04,
        0.96,
        f"r={fit['pearson_r']:+.2f}\n$R^2$={fit['r_squared']:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        bbox={"facecolor": "white", "edgecolor": "#CCCCCC", "alpha": 0.9, "pad": 3.0},
    )
    return fit


def style_axis(ax: plt.Axes, title: str, ylabel: bool = False) -> None:
    ax.set_title(title, fontsize=13, pad=8)
    ax.set_xlabel("Model Arena Elo", fontsize=10)
    if ylabel:
        ax.set_ylabel("Mean within-run payoff variance\nfor groups including model", fontsize=10)
    ax.grid(True, alpha=0.22, linewidth=0.6)
    ax.tick_params(axis="both", labelsize=8.5)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def summarize_cells(inclusions: pd.DataFrame, dims: list[str], equal_game_average: bool) -> pd.DataFrame:
    group_cols = dims + ["model", "model_short", "elo"]
    if equal_game_average and "game_label" not in dims:
        game_dims = dims + ["game_label", "model", "model_short", "elo"]
        game_means = (
            inclusions.groupby(game_dims, dropna=False)
            .agg(
                game_mean_group_payoff_variance=("group_payoff_variance", "mean"),
                game_run_count=("run_key", "nunique"),
            )
            .reset_index()
        )
        summary = (
            game_means.groupby(group_cols, dropna=False)
            .agg(
                mean_group_payoff_variance=("game_mean_group_payoff_variance", "mean"),
                game_count=("game_label", "nunique"),
                run_count=("game_run_count", "sum"),
            )
            .reset_index()
        )
        raw_counts = (
            inclusions.groupby(group_cols, dropna=False)
            .agg(raw_inclusion_rows=("run_key", "count"), raw_run_count=("run_key", "nunique"))
            .reset_index()
        )
        summary = summary.merge(raw_counts, on=group_cols, how="left")
        summary["averaging"] = "equal_game_mean"
        return summary.sort_values(group_cols)

    summary = (
        inclusions.groupby(group_cols, dropna=False)
        .agg(
            mean_group_payoff_variance=("group_payoff_variance", "mean"),
            run_count=("run_key", "nunique"),
            raw_inclusion_rows=("run_key", "count"),
            game_count=("game_label", "nunique"),
        )
        .reset_index()
        .sort_values(group_cols)
    )
    summary["raw_run_count"] = summary["run_count"]
    summary["averaging"] = "cell_mean"
    return summary


def make_fit_row(scope: dict[str, object], frame: pd.DataFrame) -> dict[str, object]:
    fit = fit_line(frame)
    return {
        **scope,
        "n_models": len(frame),
        "run_count": int(frame["run_count"].sum()),
        "slope_payoff_var_per_elo": fit["slope"],
        "slope_payoff_var_per_100_elo": fit["slope"] * 100 if math.isfinite(fit["slope"]) else math.nan,
        "intercept": fit["intercept"],
        "pearson_r": fit["pearson_r"],
        "r_squared": fit["r_squared"],
    }


def save_frame(frame: pd.DataFrame, filename: str) -> None:
    for out_dir in [OVERLEAF_OUT, ITERATION_OUT]:
        out_dir.mkdir(parents=True, exist_ok=True)
        frame.to_csv(out_dir / filename, index=False)


def plot_overall(overall: pd.DataFrame) -> tuple[Path, pd.DataFrame]:
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    ax.scatter(overall["elo"], overall["mean_group_payoff_variance"], s=34, color="#4E79A7", alpha=0.78, linewidths=0)
    fit = add_fit(ax, overall)
    for _, row in overall.iterrows():
        ax.annotate(
            str(row["model_short"]),
            (float(row["elo"]), float(row["mean_group_payoff_variance"])),
            xytext=(2, 2),
            textcoords="offset points",
            fontsize=6.0,
            alpha=0.62,
        )
    style_axis(ax, "Groups including model: payoff variance vs model Elo", ylabel=True)
    fig.tight_layout()
    out_path = OVERLEAF_OUT / "heterogeneous_included_model_group_payoff_variance_vs_elo_overall.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    fit_summary = pd.DataFrame([make_fit_row({"scope": "overall_equal_game_average"}, overall)])
    return out_path, fit_summary


def plot_by_n(by_n: pd.DataFrame) -> tuple[Path, pd.DataFrame]:
    fig, axes = plt.subplots(1, 5, figsize=(18.5, 4.2), sharex=True, sharey=True)
    rows: list[dict[str, object]] = []
    for ax, n in zip(axes, N_ORDER):
        sub = by_n[by_n["n_agents"].eq(n)]
        ax.scatter(sub["elo"], sub["mean_group_payoff_variance"], s=24, color=N_COLORS[n], alpha=0.72, linewidths=0)
        add_fit(ax, sub)
        rows.append(make_fit_row({"n_agents": n, "scope": "by_n_equal_game_average"}, sub))
        style_axis(ax, f"N={n}", ylabel=ax is axes[0])
    fig.suptitle("Groups including model: payoff variance vs model Elo, by N", fontsize=15, y=1.03)
    fig.tight_layout()
    out_path = OVERLEAF_OUT / "heterogeneous_included_model_group_payoff_variance_vs_elo_by_n.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path, pd.DataFrame(rows)


def plot_by_game(by_game: pd.DataFrame) -> tuple[Path, pd.DataFrame]:
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.2), sharex=True, sharey=False)
    rows: list[dict[str, object]] = []
    for ax, game in zip(axes, GAME_ORDER):
        sub = by_game[by_game["game_label"].eq(game)]
        ax.scatter(sub["elo"], sub["mean_group_payoff_variance"], s=26, color=GAME_COLORS[game], alpha=0.74, linewidths=0)
        add_fit(ax, sub)
        rows.append(make_fit_row({"game_label": game, "scope": "by_game"}, sub))
        style_axis(ax, GAME_TITLES[game], ylabel=ax is axes[0])
    fig.suptitle("Groups including model: payoff variance vs model Elo, by game", fontsize=15, y=1.03)
    fig.tight_layout()
    out_path = OVERLEAF_OUT / "heterogeneous_included_model_group_payoff_variance_vs_elo_by_game.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path, pd.DataFrame(rows)


def plot_by_game_n(by_game_n: pd.DataFrame) -> tuple[Path, pd.DataFrame]:
    fig, axes = plt.subplots(3, 5, figsize=(18.7, 9.0), sharex=True, sharey=False)
    rows: list[dict[str, object]] = []
    for row_idx, game in enumerate(GAME_ORDER):
        for col_idx, n in enumerate(N_ORDER):
            ax = axes[row_idx, col_idx]
            sub = by_game_n[(by_game_n["game_label"].eq(game)) & (by_game_n["n_agents"].eq(n))]
            ax.scatter(sub["elo"], sub["mean_group_payoff_variance"], s=18, color=GAME_COLORS[game], alpha=0.70, linewidths=0)
            add_fit(ax, sub)
            rows.append(make_fit_row({"game_label": game, "n_agents": n, "scope": "by_game_n"}, sub))
            style_axis(ax, f"{GAME_TITLES[game]}, N={n}", ylabel=col_idx == 0)
            if row_idx < len(GAME_ORDER) - 1:
                ax.set_xlabel("")
    fig.suptitle("Groups including model: payoff variance vs model Elo, by game and N", fontsize=15, y=1.01)
    fig.tight_layout()
    out_path = OVERLEAF_OUT / "heterogeneous_included_model_group_payoff_variance_vs_elo_by_game_n.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path, pd.DataFrame(rows)


def main() -> None:
    OVERLEAF_OUT.mkdir(parents=True, exist_ok=True)
    ITERATION_OUT.mkdir(parents=True, exist_ok=True)
    inclusions = load_model_run_inclusions()

    overall = summarize_cells(inclusions, [], equal_game_average=True)
    by_n = summarize_cells(inclusions, ["n_agents"], equal_game_average=True)
    by_game = summarize_cells(inclusions, ["game_label"], equal_game_average=False)
    by_game_n = summarize_cells(inclusions, ["game_label", "n_agents"], equal_game_average=False)

    overall_path, overall_fit = plot_overall(overall)
    by_n_path, by_n_fit = plot_by_n(by_n)
    by_game_path, by_game_fit = plot_by_game(by_game)
    by_game_n_path, by_game_n_fit = plot_by_game_n(by_game_n)

    outputs = [
        (inclusions, "heterogeneous_included_model_run_group_payoff_variances.csv"),
        (overall, "heterogeneous_included_model_group_payoff_variance_vs_elo_overall.csv"),
        (by_n, "heterogeneous_included_model_group_payoff_variance_vs_elo_by_n.csv"),
        (by_game, "heterogeneous_included_model_group_payoff_variance_vs_elo_by_game.csv"),
        (by_game_n, "heterogeneous_included_model_group_payoff_variance_vs_elo_by_game_n.csv"),
        (overall_fit, "heterogeneous_included_model_group_payoff_variance_vs_elo_overall_fit_summary.csv"),
        (by_n_fit, "heterogeneous_included_model_group_payoff_variance_vs_elo_by_n_fit_summary.csv"),
        (by_game_fit, "heterogeneous_included_model_group_payoff_variance_vs_elo_by_game_fit_summary.csv"),
        (by_game_n_fit, "heterogeneous_included_model_group_payoff_variance_vs_elo_by_game_n_fit_summary.csv"),
    ]
    for frame, filename in outputs:
        save_frame(frame, filename)

    for path in [overall_path, by_n_path, by_game_path, by_game_n_path]:
        (ITERATION_OUT / path.name).write_bytes(path.read_bytes())
        print(f"Wrote {path}")

    print("\nOverall fit:")
    print(overall_fit.to_string(index=False))
    print("\nBy-N fit:")
    print(by_n_fit.to_string(index=False))
    print("\nBy-game fit:")
    print(by_game_fit.to_string(index=False))


if __name__ == "__main__":
    main()
