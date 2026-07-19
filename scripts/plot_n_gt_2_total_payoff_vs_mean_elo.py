#!/usr/bin/env python3
"""Plot total payoff against mean roster Elo for heterogeneous N>2 runs."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

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


def load_run_metrics() -> pd.DataFrame:
    agents = pd.read_csv(AGENT_TABLE)
    agents = agents[agents["experiment_family"].eq("heterogeneous_random")].copy()
    for col in ["n_agents", "elo", "final_utility", "competition_ci"]:
        agents[col] = pd.to_numeric(agents[col], errors="coerce")
    agents = agents.dropna(subset=["run_key", "game_label", "n_agents", "elo", "final_utility"]).copy()
    agents["n_agents"] = agents["n_agents"].astype(int)

    run_metrics = (
        agents.groupby(
            ["run_key", "config_id", "game_label", "n_agents", "competition_ci", "competition_label_ci"],
            dropna=False,
        )
        .agg(
            mean_roster_elo=("elo", "mean"),
            min_roster_elo=("elo", "min"),
            max_roster_elo=("elo", "max"),
            elo_std=("elo", lambda s: float(np.std(pd.to_numeric(s, errors="coerce").dropna(), ddof=0))),
            total_payoff=("final_utility", "sum"),
            mean_payoff=("final_utility", "mean"),
            payoff_variance=("final_utility", lambda s: float(np.var(pd.to_numeric(s, errors="coerce").dropna(), ddof=0))),
            n_agents_observed=("final_utility", "count"),
            model_count=("model", "nunique"),
        )
        .reset_index()
        .sort_values(["game_label", "n_agents", "config_id"])
    )
    return run_metrics


def fit_line(frame: pd.DataFrame) -> dict[str, float]:
    data = frame[["mean_roster_elo", "total_payoff"]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < 3 or data["mean_roster_elo"].nunique() < 2:
        return {
            "slope": math.nan,
            "intercept": math.nan,
            "pearson_r": math.nan,
            "r_squared": math.nan,
            "p_value": math.nan,
            "stderr": math.nan,
        }
    fit = stats.linregress(data["mean_roster_elo"].to_numpy(dtype=float), data["total_payoff"].to_numpy(dtype=float))
    return {
        "slope": float(fit.slope),
        "intercept": float(fit.intercept),
        "pearson_r": float(fit.rvalue),
        "r_squared": float(fit.rvalue**2),
        "p_value": float(fit.pvalue),
        "stderr": float(fit.stderr),
    }


def p_text(p_value: float) -> str:
    if not math.isfinite(p_value):
        return "p=NA"
    if p_value < 0.001:
        return "p<0.001"
    return f"p={p_value:.3f}"


def add_fit(ax: plt.Axes, frame: pd.DataFrame, color: str = "#111111", box: bool = True) -> dict[str, float]:
    fit = fit_line(frame)
    if math.isfinite(fit["slope"]):
        xs = np.linspace(float(frame["mean_roster_elo"].min()), float(frame["mean_roster_elo"].max()), 160)
        ys = fit["slope"] * xs + fit["intercept"]
        ax.plot(xs, ys, color=color, lw=1.9, alpha=0.92)
    if box:
        ax.text(
            0.04,
            0.96,
            f"slope/100={fit['slope'] * 100:+.1f}\nr={fit['pearson_r']:+.2f}\n{p_text(fit['p_value'])}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "#CCCCCC", "alpha": 0.9, "pad": 3.0},
        )
    return fit


def fit_summary_row(scope: dict[str, object], frame: pd.DataFrame) -> dict[str, object]:
    fit = fit_line(frame)
    return {
        **scope,
        "n_runs": int(len(frame)),
        "mean_roster_elo_min": float(frame["mean_roster_elo"].min()) if len(frame) else math.nan,
        "mean_roster_elo_max": float(frame["mean_roster_elo"].max()) if len(frame) else math.nan,
        "total_payoff_mean": float(frame["total_payoff"].mean()) if len(frame) else math.nan,
        "slope_total_payoff_per_mean_elo": fit["slope"],
        "slope_total_payoff_per_100_mean_elo": fit["slope"] * 100 if math.isfinite(fit["slope"]) else math.nan,
        "intercept": fit["intercept"],
        "pearson_r": fit["pearson_r"],
        "r_squared": fit["r_squared"],
        "p_value": fit["p_value"],
        "stderr": fit["stderr"],
    }


def style_axis(ax: plt.Axes, title: str, ylabel: bool = False) -> None:
    ax.set_title(title, fontsize=13, pad=8)
    ax.set_xlabel("Mean roster Elo", fontsize=10)
    if ylabel:
        ax.set_ylabel("Total payoff", fontsize=10)
    ax.grid(True, alpha=0.22, linewidth=0.6)
    ax.tick_params(axis="both", labelsize=8.5)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def plot_overall(run_metrics: pd.DataFrame) -> tuple[Path, pd.DataFrame]:
    fig, ax = plt.subplots(figsize=(7.6, 5.3))
    for n in N_ORDER:
        sub = run_metrics[run_metrics["n_agents"].eq(n)]
        ax.scatter(
            sub["mean_roster_elo"],
            sub["total_payoff"],
            s=18,
            color=N_COLORS[n],
            alpha=0.34,
            linewidths=0,
            label=f"N={n}",
        )
    add_fit(ax, run_metrics)
    style_axis(ax, "Heterogeneous runs: total payoff vs mean roster Elo", ylabel=True)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=9)
    fig.tight_layout()
    out_path = OUT_DIR / "heterogeneous_total_payoff_vs_mean_elo_overall.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path, pd.DataFrame([fit_summary_row({"scope": "overall"}, run_metrics)])


def plot_by_n(run_metrics: pd.DataFrame) -> tuple[Path, pd.DataFrame]:
    fig, axes = plt.subplots(1, 5, figsize=(18.7, 4.2), sharex=True, sharey=False)
    rows: list[dict[str, object]] = []
    for ax, n in zip(axes, N_ORDER, strict=True):
        sub = run_metrics[run_metrics["n_agents"].eq(n)]
        for game in GAME_ORDER:
            game_sub = sub[sub["game_label"].eq(game)]
            ax.scatter(
                game_sub["mean_roster_elo"],
                game_sub["total_payoff"],
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
    handles.append(Line2D([0], [0], color="#111111", lw=1.9, label="Linear fit"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=4, frameon=False, fontsize=9)
    fig.suptitle("Total payoff vs mean roster Elo, broken down by N", fontsize=15, y=1.03)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    out_path = OUT_DIR / "heterogeneous_total_payoff_vs_mean_elo_by_n.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path, pd.DataFrame(rows)


def plot_by_game(run_metrics: pd.DataFrame) -> tuple[Path, pd.DataFrame]:
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.3), sharex=True, sharey=False)
    rows: list[dict[str, object]] = []
    for ax, game in zip(axes, GAME_ORDER, strict=True):
        sub = run_metrics[run_metrics["game_label"].eq(game)]
        for n in N_ORDER:
            n_sub = sub[sub["n_agents"].eq(n)]
            ax.scatter(
                n_sub["mean_roster_elo"],
                n_sub["total_payoff"],
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
    handles.append(Line2D([0], [0], color="#111111", lw=1.9, label="Linear fit"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.10), ncol=6, frameon=False, fontsize=9)
    fig.suptitle("Total payoff vs mean roster Elo, broken down by game", fontsize=15, y=1.03)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    out_path = OUT_DIR / "heterogeneous_total_payoff_vs_mean_elo_by_game.png"
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
            norm = plt.Normalize(vmin=float(run_metrics["competition_ci"].min()), vmax=float(run_metrics["competition_ci"].max()))
            colors = plt.cm.viridis(norm(sub["competition_ci"].to_numpy(dtype=float)))
            ax.scatter(sub["mean_roster_elo"], sub["total_payoff"], s=16, color=colors, alpha=0.55, linewidths=0)
            add_fit(ax, sub)
            rows.append(fit_summary_row({"scope": "by_game_n", "game_label": game, "n_agents": n}, sub))
            style_axis(ax, f"{GAME_TITLES[game]}, N={n}", ylabel=col_idx == 0)
            if row_idx < len(GAME_ORDER) - 1:
                ax.set_xlabel("")
    fig.suptitle("Total payoff vs mean roster Elo, broken down by game and N", fontsize=15, y=1.01)
    fig.tight_layout()
    out_path = OUT_DIR / "heterogeneous_total_payoff_vs_mean_elo_by_game_n.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path, pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    run_metrics = load_run_metrics()

    overall_path, overall_fit = plot_overall(run_metrics)
    by_n_path, by_n_fit = plot_by_n(run_metrics)
    by_game_path, by_game_fit = plot_by_game(run_metrics)
    by_game_n_path, by_game_n_fit = plot_by_game_n(run_metrics)

    run_metrics_path = OUT_DIR / "heterogeneous_total_payoff_vs_mean_elo_run_metrics.csv"
    run_metrics.to_csv(run_metrics_path, index=False)

    for frame, filename in [
        (overall_fit, "heterogeneous_total_payoff_vs_mean_elo_overall_fit_summary.csv"),
        (by_n_fit, "heterogeneous_total_payoff_vs_mean_elo_by_n_fit_summary.csv"),
        (by_game_fit, "heterogeneous_total_payoff_vs_mean_elo_by_game_fit_summary.csv"),
        (by_game_n_fit, "heterogeneous_total_payoff_vs_mean_elo_by_game_n_fit_summary.csv"),
    ]:
        frame.to_csv(OUT_DIR / filename, index=False)

    print(f"Wrote {run_metrics_path}")
    for path in [overall_path, by_n_path, by_game_path, by_game_n_path]:
        print(f"Wrote {path}")
    print()
    print(overall_fit.to_string(index=False))
    print()
    print(by_n_fit.to_string(index=False))
    print()
    print(by_game_fit.to_string(index=False))


if __name__ == "__main__":
    main()
