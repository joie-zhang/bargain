#!/usr/bin/env python3
"""Plot within-roster Elo variance against within-run payoff variance.

Each point is one completed heterogeneous run. The x-value is the population
variance of Arena Elo among the agents in that roster; the y-value is the
population variance of realized utility among those same agents.
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
    / "experiments/results/figure_iteration_20260626/multiagent_variance_correlation"
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
GAME_LABELS = {
    "game1": "Game 1",
    "game2": "Game 2",
    "game3": "Game 3",
}
GAME_FILE_STEMS = {
    "game1": "game1",
    "game2": "game2",
    "game3": "game3",
}
GAME_COLORS = {
    "game1": "#4E79A7",
    "game2": "#59A14F",
    "game3": "#E15759",
}


def load_run_variances() -> pd.DataFrame:
    agents = pd.read_csv(AGENT_TABLE)
    agents = agents[agents["experiment_family"].eq("heterogeneous_random")].copy()
    for col in ["n_agents", "elo", "final_utility"]:
        agents[col] = pd.to_numeric(agents[col], errors="coerce")
    agents = agents.dropna(subset=["run_key", "game_label", "n_agents", "elo", "final_utility"])
    agents["n_agents"] = agents["n_agents"].astype(int)

    run_vars = (
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
            elo_variance=("elo", lambda values: float(np.var(values.to_numpy(dtype=float), ddof=0))),
            payoff_variance=("final_utility", lambda values: float(np.var(values.to_numpy(dtype=float), ddof=0))),
            mean_payoff=("final_utility", "mean"),
            n_agents_observed=("final_utility", "count"),
            model_count=("model", "nunique"),
        )
        .reset_index()
    )
    return run_vars.sort_values(["n_agents", "game_label", "config_id"])


def fit_line(frame: pd.DataFrame) -> dict[str, float]:
    data = frame[["elo_variance", "payoff_variance"]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < 2 or data["elo_variance"].nunique() < 2:
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


def compute_competition_slopes(run_vars: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_cols = ["n_agents", "game_label", "competition_ci", "competition_label_ci", "competition_band"]
    for keys, sub in run_vars.groupby(group_cols, dropna=False):
        n_agents, game_label, competition_ci, competition_label_ci, competition_band = keys
        fit = fit_line(sub)
        rows.append(
            {
                "n_agents": int(n_agents),
                "game_label": game_label,
                "competition_ci": competition_ci,
                "competition_label_ci": competition_label_ci,
                "competition_band": competition_band,
                "n_runs": len(sub),
                "slope_payoff_var_per_elo_var": fit["slope"],
                "slope_payoff_var_per_1000_elo_var": fit["slope"] * 1000.0
                if math.isfinite(fit["slope"])
                else math.nan,
                "intercept": fit["intercept"],
                "pearson_r": fit["pearson_r"],
                "r_squared": fit["r_squared"],
            }
        )
    return pd.DataFrame(rows).sort_values(["n_agents", "game_label", "competition_ci"])


def compute_game_n_fit_summary(run_vars: pd.DataFrame, high_only: bool = False) -> pd.DataFrame:
    frame = run_vars[run_vars["competition_band"].eq("competitive")].copy() if high_only else run_vars.copy()
    rows: list[dict[str, object]] = []
    for (game_label, n_agents), sub in frame.groupby(["game_label", "n_agents"], dropna=False):
        fit = fit_line(sub)
        competition_labels = ", ".join(str(label) for label in sorted(sub["competition_label_ci"].dropna().unique()))
        rows.append(
            {
                "game_label": game_label,
                "n_agents": int(n_agents),
                "competition_scope": "high_competition" if high_only else "all_competition",
                "competition_labels": competition_labels,
                "n_runs": len(sub),
                "slope_payoff_var_per_elo_var": fit["slope"],
                "slope_payoff_var_per_1000_elo_var": fit["slope"] * 1000.0
                if math.isfinite(fit["slope"])
                else math.nan,
                "intercept": fit["intercept"],
                "pearson_r": fit["pearson_r"],
                "r_squared": fit["r_squared"],
            }
        )
    return pd.DataFrame(rows).sort_values(["game_label", "n_agents"])


def plot_variance_correlation(run_vars: pd.DataFrame) -> tuple[Path, pd.DataFrame]:
    fit_rows: list[dict[str, object]] = []
    fig, axes = plt.subplots(1, 5, figsize=(19.0, 4.25), sharex=True, sharey=True)

    x_max = float(run_vars["elo_variance"].max()) * 1.04
    y_max = max(float(run_vars["payoff_variance"].max()) * 1.05, 100.0)

    for ax, n in zip(axes, N_ORDER):
        sub_n = run_vars[run_vars["n_agents"].eq(n)]
        for game in GAME_ORDER:
            sub = sub_n[sub_n["game_label"].eq(game)]
            ax.scatter(
                sub["elo_variance"],
                sub["payoff_variance"],
                s=17,
                color=GAME_COLORS[game],
                alpha=0.38,
                linewidths=0,
                label=GAME_LABELS[game],
            )

        fit = fit_line(sub_n)
        fit_rows.append(
            {
                "n_agents": n,
                "n_runs": len(sub_n),
                "slope_payoff_var_per_elo_var": fit["slope"],
                "slope_payoff_var_per_1000_elo_var": fit["slope"] * 1000.0
                if math.isfinite(fit["slope"])
                else math.nan,
                "intercept": fit["intercept"],
                "pearson_r": fit["pearson_r"],
                "r_squared": fit["r_squared"],
            }
        )
        if math.isfinite(fit["slope"]):
            xs = np.linspace(float(sub_n["elo_variance"].min()), float(sub_n["elo_variance"].max()), 150)
            ys = fit["slope"] * xs + fit["intercept"]
            ax.plot(xs, ys, color="#111111", lw=1.8, alpha=0.86)

        ax.text(
            0.04,
            0.96,
            f"r={fit['pearson_r']:+.2f}\n$R^2$={fit['r_squared']:.2f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "#CCCCCC", "alpha": 0.88, "pad": 3.0},
        )
        ax.set_title(f"N={n}", fontsize=14, pad=8)
        ax.grid(True, alpha=0.22, linewidth=0.6)
        ax.tick_params(axis="both", labelsize=9)
        ax.set_xlim(0, x_max)
        ax.set_ylim(0, y_max)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    axes[0].set_ylabel("Within-run payoff variance", fontsize=11, labelpad=8)
    fig.supxlabel("Within-roster Arena Elo variance", fontsize=12, y=0.03)
    fig.suptitle("Does capability dispersion predict payoff dispersion in heterogeneous groups?", fontsize=16, y=1.03)
    handles = [
        Line2D([0], [0], marker="o", linestyle="none", color=GAME_COLORS[game], label=GAME_LABELS[game], markersize=6)
        for game in GAME_ORDER
    ]
    handles.append(Line2D([0], [0], color="#111111", lw=1.8, label="Linear fit"))
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=4,
        frameon=False,
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.97), w_pad=1.1)

    out_path = OVERLEAF_OUT / "heterogeneous_elo_variance_vs_payoff_variance_by_n.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    fit_summary = pd.DataFrame(fit_rows)
    return out_path, fit_summary


def plot_all_n_all_games_variance_correlation(run_vars: pd.DataFrame) -> tuple[Path, pd.DataFrame]:
    fit = fit_line(run_vars)
    fig, ax = plt.subplots(figsize=(7.6, 5.3))

    for n in N_ORDER:
        sub = run_vars[run_vars["n_agents"].eq(n)]
        ax.scatter(
            sub["elo_variance"],
            sub["payoff_variance"],
            s=17,
            color=N_COLORS[n],
            alpha=0.34,
            linewidths=0,
            label=f"N={n}",
        )

    if math.isfinite(fit["slope"]):
        xs = np.linspace(float(run_vars["elo_variance"].min()), float(run_vars["elo_variance"].max()), 200)
        ys = fit["slope"] * xs + fit["intercept"]
        ax.plot(xs, ys, color="#111111", lw=2.1, alpha=0.9, label="Linear fit")

    ax.text(
        0.03,
        0.97,
        (
            f"runs={len(run_vars)}\n"
            f"r={fit['pearson_r']:+.2f}\n"
            f"$R^2$={fit['r_squared']:.3f}\n"
            f"slope={fit['slope'] * 1000.0:+.2f}/1000 Elo-var"
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={"facecolor": "white", "edgecolor": "#CCCCCC", "alpha": 0.9, "pad": 4.0},
    )
    ax.set_title("Heterogeneous runs: Elo variance vs payoff variance", fontsize=15, pad=10)
    ax.set_xlabel("Within-roster Arena Elo variance", fontsize=12)
    ax.set_ylabel("Within-run payoff variance", fontsize=12)
    ax.grid(True, alpha=0.22, linewidth=0.6)
    ax.tick_params(axis="both", labelsize=10)
    ax.set_xlim(0, float(run_vars["elo_variance"].max()) * 1.04)
    ax.set_ylim(0, float(run_vars["payoff_variance"].max()) * 1.05)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.legend(loc="upper right", frameon=False, fontsize=9, ncol=2)
    fig.tight_layout()

    out_path = OVERLEAF_OUT / "heterogeneous_elo_variance_vs_payoff_variance_all_n_all_games.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    fit_summary = pd.DataFrame(
        [
            {
                "scope": "all_n_all_games",
                "n_runs": len(run_vars),
                "slope_payoff_var_per_elo_var": fit["slope"],
                "slope_payoff_var_per_1000_elo_var": fit["slope"] * 1000.0
                if math.isfinite(fit["slope"])
                else math.nan,
                "intercept": fit["intercept"],
                "pearson_r": fit["pearson_r"],
                "r_squared": fit["r_squared"],
            }
        ]
    )
    return out_path, fit_summary


def plot_game_variance_correlation(run_vars: pd.DataFrame, game: str, high_only: bool = False) -> tuple[Path, pd.DataFrame]:
    game_frame = run_vars[run_vars["game_label"].eq(game)].copy()
    if high_only:
        game_frame = game_frame[game_frame["competition_band"].eq("competitive")].copy()

    fit_summary = compute_game_n_fit_summary(game_frame, high_only=high_only)
    scope_label = "high competition" if high_only else "all competition levels"
    scope_stem = "high_competition" if high_only else "all_competition"
    fig, axes = plt.subplots(1, 5, figsize=(19.0, 4.25), sharex=True, sharey=True)

    x_max = max(float(game_frame["elo_variance"].max()) * 1.04, 100.0)
    y_max = max(float(game_frame["payoff_variance"].max()) * 1.05, 100.0)
    cmap = plt.cm.viridis
    norm = plt.Normalize(
        vmin=float(game_frame["competition_ci"].min()),
        vmax=float(game_frame["competition_ci"].max()),
    )

    for ax, n in zip(axes, N_ORDER):
        sub_n = game_frame[game_frame["n_agents"].eq(n)]
        colors = cmap(norm(sub_n["competition_ci"].to_numpy(dtype=float)))
        ax.scatter(
            sub_n["elo_variance"],
            sub_n["payoff_variance"],
            s=24 if high_only else 18,
            color=colors,
            alpha=0.55 if high_only else 0.42,
            linewidths=0,
        )

        fit = fit_line(sub_n)
        if math.isfinite(fit["slope"]):
            xs = np.linspace(float(sub_n["elo_variance"].min()), float(sub_n["elo_variance"].max()), 150)
            ys = fit["slope"] * xs + fit["intercept"]
            ax.plot(xs, ys, color="#111111", lw=1.9, alpha=0.9)

        ax.text(
            0.04,
            0.96,
            f"r={fit['pearson_r']:+.2f}\n$R^2$={fit['r_squared']:.2f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "#CCCCCC", "alpha": 0.88, "pad": 3.0},
        )
        if high_only:
            labels = ", ".join(str(label) for label in sorted(sub_n["competition_label_ci"].dropna().unique()))
            ax.text(
                0.04,
                0.05,
                labels,
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=8,
                color="#444444",
            )
        ax.set_title(f"N={n}", fontsize=14, pad=8)
        ax.grid(True, alpha=0.22, linewidth=0.6)
        ax.tick_params(axis="both", labelsize=9)
        ax.set_xlim(0, x_max)
        ax.set_ylim(0, y_max)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    axes[0].set_ylabel("Within-run payoff variance", fontsize=11, labelpad=8)
    if high_only:
        fig.supxlabel("Within-roster Arena Elo variance", fontsize=12, y=0.03)
    else:
        axes[2].set_xlabel("Within-roster Arena Elo variance", fontsize=12, labelpad=14)
    fig.suptitle(
        f"{GAME_LABELS[game]}: Elo variance vs payoff variance ({scope_label})",
        fontsize=16,
        y=1.03,
    )
    if not high_only:
        fig.subplots_adjust(left=0.045, right=0.995, top=0.82, bottom=0.31, wspace=0.08)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cax = fig.add_axes([0.34, 0.075, 0.36, 0.035])
        cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
        cbar.set_label("Competition index", fontsize=10)
        cbar.ax.tick_params(labelsize=8)
    else:
        fig.tight_layout(rect=(0, 0.08, 1, 0.97), w_pad=1.1)

    out_path = (
        OVERLEAF_OUT
        / f"heterogeneous_elo_variance_vs_payoff_variance_{GAME_FILE_STEMS[game]}_{scope_stem}_by_n.png"
    )
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path, fit_summary


def plot_high_competition_variance_correlation(run_vars: pd.DataFrame) -> tuple[Path, pd.DataFrame]:
    high = run_vars[run_vars["competition_band"].eq("competitive")].copy()
    fit_rows: list[dict[str, object]] = []
    fig, axes = plt.subplots(1, 5, figsize=(19.0, 4.25), sharex=True, sharey=True)

    x_max = max(float(high["elo_variance"].max()) * 1.04, 100.0)
    y_max = max(float(high["payoff_variance"].max()) * 1.05, 100.0)

    for ax, n in zip(axes, N_ORDER):
        sub_n = high[high["n_agents"].eq(n)]
        for game in GAME_ORDER:
            sub = sub_n[sub_n["game_label"].eq(game)]
            ax.scatter(
                sub["elo_variance"],
                sub["payoff_variance"],
                s=24,
                color=GAME_COLORS[game],
                alpha=0.50,
                linewidths=0,
                label=GAME_LABELS[game],
            )

        fit = fit_line(sub_n)
        fit_rows.append(
            {
                "n_agents": n,
                "n_runs": len(sub_n),
                "slope_payoff_var_per_elo_var": fit["slope"],
                "slope_payoff_var_per_1000_elo_var": fit["slope"] * 1000.0
                if math.isfinite(fit["slope"])
                else math.nan,
                "intercept": fit["intercept"],
                "pearson_r": fit["pearson_r"],
                "r_squared": fit["r_squared"],
            }
        )
        if math.isfinite(fit["slope"]):
            xs = np.linspace(float(sub_n["elo_variance"].min()), float(sub_n["elo_variance"].max()), 150)
            ys = fit["slope"] * xs + fit["intercept"]
            ax.plot(xs, ys, color="#111111", lw=1.9, alpha=0.9)

        ax.text(
            0.04,
            0.96,
            f"r={fit['pearson_r']:+.2f}\n$R^2$={fit['r_squared']:.2f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "#CCCCCC", "alpha": 0.88, "pad": 3.0},
        )
        ax.set_title(f"N={n}", fontsize=14, pad=8)
        ax.grid(True, alpha=0.22, linewidth=0.6)
        ax.tick_params(axis="both", labelsize=9)
        ax.set_xlim(0, x_max)
        ax.set_ylim(0, y_max)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    axes[0].set_ylabel("Within-run payoff variance", fontsize=11, labelpad=8)
    fig.supxlabel("Within-roster Arena Elo variance", fontsize=12, y=0.03)
    fig.suptitle("High-competition heterogeneous runs: Elo variance vs payoff variance", fontsize=16, y=1.03)
    handles = [
        Line2D([0], [0], marker="o", linestyle="none", color=GAME_COLORS[game], label=GAME_LABELS[game], markersize=6)
        for game in GAME_ORDER
    ]
    handles.append(Line2D([0], [0], color="#111111", lw=1.9, label="Linear fit"))
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=4,
        frameon=False,
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.97), w_pad=1.1)

    out_path = OVERLEAF_OUT / "heterogeneous_elo_variance_vs_payoff_variance_high_competition_by_n.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path, pd.DataFrame(fit_rows)


def plot_game_competition_slopes(slopes: pd.DataFrame, game: str) -> Path:
    game_slopes = slopes[slopes["game_label"].eq(game)].copy()
    fig, axes = plt.subplots(1, 5, figsize=(19.0, 4.2), sharey=True)
    finite = game_slopes["slope_payoff_var_per_1000_elo_var"].replace([np.inf, -np.inf], np.nan).dropna()
    y_abs = max(5.0, float(finite.abs().max()) * 1.1)
    color = GAME_COLORS[game]

    for ax, n in zip(axes, N_ORDER):
        sub = game_slopes[game_slopes["n_agents"].eq(n)].sort_values("competition_ci")
        ax.plot(
            sub["competition_ci"],
            sub["slope_payoff_var_per_1000_elo_var"],
            marker="o",
            ms=4.8,
            lw=1.5,
            color=color,
            alpha=0.92,
        )
        ax.axhline(0, color="#333333", lw=0.9, alpha=0.65)
        ax.set_title(f"N={n}", fontsize=14, pad=8)
        ax.set_xlabel("Competition index", fontsize=10)
        ax.grid(True, alpha=0.22, linewidth=0.6)
        ax.tick_params(axis="both", labelsize=9)
        ax.set_ylim(-y_abs, y_abs)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    axes[0].set_ylabel("Slope: payoff variance per 1000 Elo-variance units", fontsize=11, labelpad=8)
    fig.suptitle(f"{GAME_LABELS[game]}: Elo-payoff variance slope by competition level", fontsize=16, y=1.03)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97), w_pad=1.1)

    out_path = (
        OVERLEAF_OUT
        / f"heterogeneous_elo_payoff_variance_slopes_by_competition_{GAME_FILE_STEMS[game]}_by_n.png"
    )
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_competition_slopes(slopes: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 5, figsize=(19.0, 4.2), sharey=True)
    finite = slopes["slope_payoff_var_per_1000_elo_var"].replace([np.inf, -np.inf], np.nan).dropna()
    y_abs = max(5.0, float(finite.abs().max()) * 1.1)

    for ax, n in zip(axes, N_ORDER):
        sub_n = slopes[slopes["n_agents"].eq(n)]
        for game in GAME_ORDER:
            sub = sub_n[sub_n["game_label"].eq(game)].sort_values("competition_ci")
            ax.plot(
                sub["competition_ci"],
                sub["slope_payoff_var_per_1000_elo_var"],
                marker="o",
                ms=4.4,
                lw=1.4,
                color=GAME_COLORS[game],
                label=GAME_LABELS[game],
                alpha=0.9,
            )
        ax.axhline(0, color="#333333", lw=0.9, alpha=0.65)
        ax.set_title(f"N={n}", fontsize=14, pad=8)
        ax.set_xlabel("Competition index", fontsize=10)
        ax.grid(True, alpha=0.22, linewidth=0.6)
        ax.tick_params(axis="both", labelsize=9)
        ax.set_ylim(-y_abs, y_abs)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    axes[0].set_ylabel("Slope: payoff variance per 1000 Elo-variance units", fontsize=11, labelpad=8)
    fig.suptitle("Elo-payoff variance slope by competition level", fontsize=16, y=1.03)
    handles = [
        Line2D([0], [0], marker="o", color=GAME_COLORS[game], lw=1.4, label=GAME_LABELS[game], markersize=5)
        for game in GAME_ORDER
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=3,
        frameon=False,
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.97), w_pad=1.1)

    out_path = OVERLEAF_OUT / "heterogeneous_elo_payoff_variance_slopes_by_competition_and_n.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    OVERLEAF_OUT.mkdir(parents=True, exist_ok=True)
    ITERATION_OUT.mkdir(parents=True, exist_ok=True)

    run_vars = load_run_variances()
    aggregate_plot_path, aggregate_fit_summary = plot_all_n_all_games_variance_correlation(run_vars)
    plot_path, fit_summary = plot_variance_correlation(run_vars)
    high_plot_path, high_fit_summary = plot_high_competition_variance_correlation(run_vars)
    competition_slopes = compute_competition_slopes(run_vars)
    slope_plot_path = plot_competition_slopes(competition_slopes)
    game_plot_paths: list[Path] = []
    game_fit_summaries: list[pd.DataFrame] = []
    game_high_fit_summaries: list[pd.DataFrame] = []
    for game in GAME_ORDER:
        game_plot_path, game_fit_summary = plot_game_variance_correlation(run_vars, game, high_only=False)
        game_high_plot_path, game_high_fit_summary = plot_game_variance_correlation(run_vars, game, high_only=True)
        game_slope_path = plot_game_competition_slopes(competition_slopes, game)
        game_plot_paths.extend([game_plot_path, game_high_plot_path, game_slope_path])
        game_fit_summaries.append(game_fit_summary)
        game_high_fit_summaries.append(game_high_fit_summary)
    game_fit_summary = pd.concat(game_fit_summaries, ignore_index=True)
    game_high_fit_summary = pd.concat(game_high_fit_summaries, ignore_index=True)

    for out_dir in [OVERLEAF_OUT, ITERATION_OUT]:
        run_vars.to_csv(out_dir / "heterogeneous_run_elo_payoff_variances.csv", index=False)
        aggregate_fit_summary.to_csv(
            out_dir / "heterogeneous_elo_payoff_variance_fit_summary_all_n_all_games.csv",
            index=False,
        )
        fit_summary.to_csv(out_dir / "heterogeneous_elo_payoff_variance_fit_summary.csv", index=False)
        high_fit_summary.to_csv(out_dir / "heterogeneous_elo_payoff_variance_high_competition_fit_summary.csv", index=False)
        competition_slopes.to_csv(out_dir / "heterogeneous_elo_payoff_variance_slopes_by_competition.csv", index=False)
        game_fit_summary.to_csv(out_dir / "heterogeneous_elo_payoff_variance_fit_summary_by_game.csv", index=False)
        game_high_fit_summary.to_csv(
            out_dir / "heterogeneous_elo_payoff_variance_high_competition_fit_summary_by_game.csv",
            index=False,
        )
    (ITERATION_OUT / aggregate_plot_path.name).write_bytes(aggregate_plot_path.read_bytes())
    (ITERATION_OUT / plot_path.name).write_bytes(plot_path.read_bytes())
    (ITERATION_OUT / high_plot_path.name).write_bytes(high_plot_path.read_bytes())
    (ITERATION_OUT / slope_plot_path.name).write_bytes(slope_plot_path.read_bytes())
    for path in game_plot_paths:
        (ITERATION_OUT / path.name).write_bytes(path.read_bytes())

    print(f"Wrote {aggregate_plot_path}")
    print(f"Wrote {plot_path}")
    print(f"Wrote {high_plot_path}")
    print(f"Wrote {slope_plot_path}")
    for path in game_plot_paths:
        print(f"Wrote {path}")
    print(f"Wrote summaries to {OVERLEAF_OUT} and {ITERATION_OUT}")
    print("\nAll-N all-game aggregate fit:")
    print(aggregate_fit_summary.to_string(index=False))
    print("\nAll-competition pooled fit:")
    print(fit_summary.to_string(index=False))
    print("\nHigh-competition pooled fit:")
    print(high_fit_summary.to_string(index=False))


if __name__ == "__main__":
    main()
