#!/usr/bin/env python3
"""Plot within-run payoff variance against mean roster Elo.

Each point is one completed heterogeneous game outcome. The x-axis is the
average Arena Elo of the models in that roster; the y-axis is the variance of
final payoffs among agents in that run.
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
OVERLEAF_OUT = PROJECT_ROOT / "overleaf/neurips/graphics/n_gt_2_report"
ITERATION_OUT = (
    PROJECT_ROOT
    / "experiments/results/figure_iteration_20260626/group_payoff_variance_vs_mean_elo"
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


def load_run_metrics() -> pd.DataFrame:
    agents = pd.read_csv(AGENT_TABLE)
    agents = agents[agents["experiment_family"].eq("heterogeneous_random")].copy()
    for col in ["n_agents", "elo", "final_utility", "competition_ci"]:
        agents[col] = pd.to_numeric(agents[col], errors="coerce")
    agents = agents.dropna(subset=["run_key", "game_label", "n_agents", "elo", "final_utility"])
    agents["n_agents"] = agents["n_agents"].astype(int)

    run_metrics = (
        agents.groupby(
            ["run_key", "config_id", "game_label", "n_agents", "competition_ci", "competition_label_ci"],
            dropna=False,
        )
        .agg(
            mean_roster_elo=("elo", "mean"),
            elo_variance=("elo", population_variance),
            payoff_variance=("final_utility", population_variance),
            mean_payoff=("final_utility", "mean"),
            n_agents_observed=("final_utility", "count"),
            model_count=("model", "nunique"),
        )
        .reset_index()
        .sort_values(["game_label", "n_agents", "config_id"])
    )
    return run_metrics


def fit_line(frame: pd.DataFrame) -> dict[str, float]:
    data = frame[["mean_roster_elo", "payoff_variance"]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < 2 or data["mean_roster_elo"].nunique() < 2:
        return {"slope": math.nan, "intercept": math.nan, "pearson_r": math.nan, "r_squared": math.nan}
    x = data["mean_roster_elo"].to_numpy(dtype=float)
    y = data["payoff_variance"].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    pearson_r = float(np.corrcoef(x, y)[0, 1])
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "pearson_r": pearson_r,
        "r_squared": pearson_r * pearson_r,
    }


def add_fit(ax: plt.Axes, frame: pd.DataFrame) -> dict[str, float]:
    fit = fit_line(frame)
    if math.isfinite(fit["slope"]):
        xs = np.linspace(float(frame["mean_roster_elo"].min()), float(frame["mean_roster_elo"].max()), 160)
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


def fit_summary_row(scope: dict[str, object], frame: pd.DataFrame) -> dict[str, object]:
    fit = fit_line(frame)
    return {
        **scope,
        "n_runs": len(frame),
        "slope_payoff_var_per_mean_elo": fit["slope"],
        "slope_payoff_var_per_100_mean_elo": fit["slope"] * 100 if math.isfinite(fit["slope"]) else math.nan,
        "intercept": fit["intercept"],
        "pearson_r": fit["pearson_r"],
        "r_squared": fit["r_squared"],
    }


def style_axis(ax: plt.Axes, title: str, ylabel: bool = False) -> None:
    ax.set_title(title, fontsize=13, pad=8)
    ax.set_xlabel("Mean roster Arena Elo", fontsize=10)
    if ylabel:
        ax.set_ylabel("Within-run payoff variance", fontsize=10)
    ax.grid(True, alpha=0.22, linewidth=0.6)
    ax.tick_params(axis="both", labelsize=8.5)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def save_frame(frame: pd.DataFrame, filename: str) -> None:
    for out_dir in [OVERLEAF_OUT, ITERATION_OUT]:
        out_dir.mkdir(parents=True, exist_ok=True)
        frame.to_csv(out_dir / filename, index=False)


def plot_overall(run_metrics: pd.DataFrame) -> tuple[Path, pd.DataFrame]:
    fig, ax = plt.subplots(figsize=(7.6, 5.3))
    for n in N_ORDER:
        sub = run_metrics[run_metrics["n_agents"].eq(n)]
        ax.scatter(
            sub["mean_roster_elo"],
            sub["payoff_variance"],
            s=17,
            color=N_COLORS[n],
            alpha=0.34,
            linewidths=0,
            label=f"N={n}",
        )
    add_fit(ax, run_metrics)
    style_axis(ax, "Heterogeneous runs: payoff variance vs mean roster Elo", ylabel=True)
    ax.legend(loc="upper right", frameon=False, fontsize=9, ncol=2)
    fig.tight_layout()
    out_path = OVERLEAF_OUT / "heterogeneous_group_payoff_variance_vs_mean_elo_overall.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path, pd.DataFrame([fit_summary_row({"scope": "overall"}, run_metrics)])


def plot_by_n(run_metrics: pd.DataFrame) -> tuple[Path, pd.DataFrame]:
    fig, axes = plt.subplots(1, 5, figsize=(18.5, 4.2), sharex=True, sharey=True)
    rows: list[dict[str, object]] = []
    for ax, n in zip(axes, N_ORDER):
        sub = run_metrics[run_metrics["n_agents"].eq(n)]
        for game in GAME_ORDER:
            game_sub = sub[sub["game_label"].eq(game)]
            ax.scatter(
                game_sub["mean_roster_elo"],
                game_sub["payoff_variance"],
                s=17,
                color=GAME_COLORS[game],
                alpha=0.38,
                linewidths=0,
            )
        add_fit(ax, sub)
        rows.append(fit_summary_row({"scope": "by_n", "n_agents": n}, sub))
        style_axis(ax, f"N={n}", ylabel=ax is axes[0])
    handles = [
        Line2D([0], [0], marker="o", linestyle="none", color=GAME_COLORS[game], label=GAME_TITLES[game], markersize=5)
        for game in GAME_ORDER
    ]
    handles.append(Line2D([0], [0], color="#111111", lw=1.8, label="Linear fit"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=4, frameon=False, fontsize=9)
    fig.suptitle("Payoff variance vs mean roster Elo, broken down by N", fontsize=15, y=1.03)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    out_path = OVERLEAF_OUT / "heterogeneous_group_payoff_variance_vs_mean_elo_by_n.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path, pd.DataFrame(rows)


def plot_by_game(run_metrics: pd.DataFrame) -> tuple[Path, pd.DataFrame]:
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.3), sharex=True, sharey=False)
    rows: list[dict[str, object]] = []
    for ax, game in zip(axes, GAME_ORDER):
        sub = run_metrics[run_metrics["game_label"].eq(game)]
        for n in N_ORDER:
            n_sub = sub[sub["n_agents"].eq(n)]
            ax.scatter(
                n_sub["mean_roster_elo"],
                n_sub["payoff_variance"],
                s=18,
                color=N_COLORS[n],
                alpha=0.42,
                linewidths=0,
            )
        add_fit(ax, sub)
        rows.append(fit_summary_row({"scope": "by_game", "game_label": game}, sub))
        style_axis(ax, GAME_TITLES[game], ylabel=ax is axes[0])
    handles = [
        Line2D([0], [0], marker="o", linestyle="none", color=N_COLORS[n], label=f"N={n}", markersize=5)
        for n in N_ORDER
    ]
    handles.append(Line2D([0], [0], color="#111111", lw=1.8, label="Linear fit"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.10), ncol=6, frameon=False, fontsize=9)
    fig.suptitle("Payoff variance vs mean roster Elo, broken down by game", fontsize=15, y=1.03)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    out_path = OVERLEAF_OUT / "heterogeneous_group_payoff_variance_vs_mean_elo_by_game.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path, pd.DataFrame(rows)


def plot_by_game_n(run_metrics: pd.DataFrame) -> tuple[Path, pd.DataFrame]:
    fig, axes = plt.subplots(3, 5, figsize=(18.7, 9.0), sharex=True, sharey=False)
    rows: list[dict[str, object]] = []
    for row_idx, game in enumerate(GAME_ORDER):
        for col_idx, n in enumerate(N_ORDER):
            ax = axes[row_idx, col_idx]
            sub = run_metrics[(run_metrics["game_label"].eq(game)) & (run_metrics["n_agents"].eq(n))]
            colors = plt.cm.viridis(
                plt.Normalize(
                    vmin=float(sub["competition_ci"].min()),
                    vmax=float(sub["competition_ci"].max()),
                )(sub["competition_ci"].to_numpy(dtype=float))
            )
            ax.scatter(sub["mean_roster_elo"], sub["payoff_variance"], s=16, color=colors, alpha=0.55, linewidths=0)
            add_fit(ax, sub)
            rows.append(fit_summary_row({"scope": "by_game_n", "game_label": game, "n_agents": n}, sub))
            style_axis(ax, f"{GAME_TITLES[game]}, N={n}", ylabel=col_idx == 0)
            if row_idx < len(GAME_ORDER) - 1:
                ax.set_xlabel("")
    fig.suptitle("Payoff variance vs mean roster Elo, broken down by game and N", fontsize=15, y=1.01)
    fig.tight_layout()
    out_path = OVERLEAF_OUT / "heterogeneous_group_payoff_variance_vs_mean_elo_by_game_n.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path, pd.DataFrame(rows)


def main() -> None:
    OVERLEAF_OUT.mkdir(parents=True, exist_ok=True)
    ITERATION_OUT.mkdir(parents=True, exist_ok=True)
    run_metrics = load_run_metrics()

    overall_path, overall_fit = plot_overall(run_metrics)
    by_n_path, by_n_fit = plot_by_n(run_metrics)
    by_game_path, by_game_fit = plot_by_game(run_metrics)
    by_game_n_path, by_game_n_fit = plot_by_game_n(run_metrics)

    for frame, filename in [
        (run_metrics, "heterogeneous_group_payoff_variance_vs_mean_elo_run_metrics.csv"),
        (overall_fit, "heterogeneous_group_payoff_variance_vs_mean_elo_overall_fit_summary.csv"),
        (by_n_fit, "heterogeneous_group_payoff_variance_vs_mean_elo_by_n_fit_summary.csv"),
        (by_game_fit, "heterogeneous_group_payoff_variance_vs_mean_elo_by_game_fit_summary.csv"),
        (by_game_n_fit, "heterogeneous_group_payoff_variance_vs_mean_elo_by_game_n_fit_summary.csv"),
    ]:
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
