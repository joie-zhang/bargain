#!/usr/bin/env python3
"""Plot average payoff against mean/max roster Elo for heterogeneous N>2 runs."""

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
COMPETITION_BAND_ORDER = ["cooperative", "middle", "competitive"]
COMPETITION_BAND_TITLES = {
    "cooperative": "Low competition",
    "middle": "Medium competition",
    "competitive": "High competition",
}
X_SPECS = [
    ("mean_roster_elo", "Mean roster Elo", "mean_elo"),
    ("max_roster_elo", "Max roster Elo", "max_elo"),
]


def load_run_metrics() -> pd.DataFrame:
    agents = pd.read_csv(AGENT_TABLE)
    agents = agents[agents["experiment_family"].eq("heterogeneous_random")].copy()
    for col in ["n_agents", "elo", "final_utility", "competition_ci"]:
        agents[col] = pd.to_numeric(agents[col], errors="coerce")
    agents = agents.dropna(subset=["run_key", "game_label", "n_agents", "elo", "final_utility"]).copy()
    agents["n_agents"] = agents["n_agents"].astype(int)

    run_metrics = (
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
            mean_roster_elo=("elo", "mean"),
            min_roster_elo=("elo", "min"),
            max_roster_elo=("elo", "max"),
            elo_std=("elo", lambda s: float(np.std(pd.to_numeric(s, errors="coerce").dropna(), ddof=0))),
            total_payoff=("final_utility", "sum"),
            average_payoff=("final_utility", "mean"),
            payoff_variance=("final_utility", lambda s: float(np.var(pd.to_numeric(s, errors="coerce").dropna(), ddof=0))),
            n_agents_observed=("final_utility", "count"),
            model_count=("model", "nunique"),
        )
        .reset_index()
        .sort_values(["game_label", "n_agents", "config_id"])
    )
    return run_metrics


def fit_line(frame: pd.DataFrame, x_col: str) -> dict[str, float]:
    data = frame[[x_col, "average_payoff"]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < 3 or data[x_col].nunique() < 2:
        return {
            "slope": math.nan,
            "intercept": math.nan,
            "pearson_r": math.nan,
            "r_squared": math.nan,
            "p_value": math.nan,
            "stderr": math.nan,
        }
    fit = stats.linregress(data[x_col].to_numpy(dtype=float), data["average_payoff"].to_numpy(dtype=float))
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


def add_fit(ax: plt.Axes, frame: pd.DataFrame, x_col: str, box: bool = True, color: str = "#111111") -> dict[str, float]:
    fit = fit_line(frame, x_col)
    if math.isfinite(fit["slope"]):
        xs = np.linspace(float(frame[x_col].min()), float(frame[x_col].max()), 160)
        ys = fit["slope"] * xs + fit["intercept"]
        ax.plot(xs, ys, color=color, lw=1.9, alpha=0.92)
    if box:
        ax.text(
            0.04,
            0.96,
            f"slope/100={fit['slope'] * 100:+.2f}\nr={fit['pearson_r']:+.2f}\n{p_text(fit['p_value'])}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "#CCCCCC", "alpha": 0.9, "pad": 3.0},
        )
    return fit


def fit_summary_row(scope: dict[str, object], frame: pd.DataFrame, x_col: str) -> dict[str, object]:
    fit = fit_line(frame, x_col)
    return {
        **scope,
        "x_col": x_col,
        "n_runs": int(len(frame)),
        "x_min": float(frame[x_col].min()) if len(frame) else math.nan,
        "x_max": float(frame[x_col].max()) if len(frame) else math.nan,
        "average_payoff_mean": float(frame["average_payoff"].mean()) if len(frame) else math.nan,
        "slope_average_payoff_per_elo": fit["slope"],
        "slope_average_payoff_per_100_elo": fit["slope"] * 100 if math.isfinite(fit["slope"]) else math.nan,
        "intercept": fit["intercept"],
        "pearson_r": fit["pearson_r"],
        "r_squared": fit["r_squared"],
        "p_value": fit["p_value"],
        "stderr": fit["stderr"],
    }


def style_axis(ax: plt.Axes, title: str, x_label: str, ylabel: bool = False) -> None:
    ax.set_title(title, fontsize=13, pad=8)
    ax.set_xlabel(x_label, fontsize=10)
    if ylabel:
        ax.set_ylabel("Average payoff per agent", fontsize=10)
    ax.grid(True, alpha=0.22, linewidth=0.6)
    ax.tick_params(axis="both", labelsize=8.5)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def plot_overall(run_metrics: pd.DataFrame, x_col: str, x_label: str, file_tag: str) -> tuple[Path, pd.DataFrame]:
    fig, ax = plt.subplots(figsize=(7.6, 5.3))
    for n in N_ORDER:
        sub = run_metrics[run_metrics["n_agents"].eq(n)]
        ax.scatter(
            sub[x_col],
            sub["average_payoff"],
            s=18,
            color=N_COLORS[n],
            alpha=0.34,
            linewidths=0,
            label=f"N={n}",
        )
    add_fit(ax, run_metrics, x_col)
    style_axis(ax, f"Heterogeneous runs: average payoff vs {x_label.lower()}", x_label, ylabel=True)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=9)
    fig.tight_layout()
    out_path = OUT_DIR / f"heterogeneous_average_payoff_vs_{file_tag}_overall.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path, pd.DataFrame([fit_summary_row({"scope": "overall"}, run_metrics, x_col)])


def plot_by_n(run_metrics: pd.DataFrame, x_col: str, x_label: str, file_tag: str) -> tuple[Path, pd.DataFrame]:
    fig, axes = plt.subplots(1, 5, figsize=(18.7, 4.2), sharex=True, sharey=True)
    rows: list[dict[str, object]] = []
    for ax, n in zip(axes, N_ORDER, strict=True):
        sub = run_metrics[run_metrics["n_agents"].eq(n)]
        for game in GAME_ORDER:
            game_sub = sub[sub["game_label"].eq(game)]
            ax.scatter(
                game_sub[x_col],
                game_sub["average_payoff"],
                s=17,
                color=GAME_COLORS[game],
                alpha=0.38,
                linewidths=0,
            )
        add_fit(ax, sub, x_col)
        rows.append(fit_summary_row({"scope": "by_n", "n_agents": n}, sub, x_col))
        style_axis(ax, f"N={n}", x_label, ylabel=ax is axes[0])
    handles = [
        Line2D([0], [0], marker="o", linestyle="none", color=GAME_COLORS[game], label=GAME_TITLES[game], markersize=5)
        for game in GAME_ORDER
    ]
    handles.append(Line2D([0], [0], color="#111111", lw=1.9, label="Linear fit"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=4, frameon=False, fontsize=9)
    fig.suptitle(f"Average payoff vs {x_label.lower()}, broken down by N", fontsize=15, y=1.03)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    out_path = OUT_DIR / f"heterogeneous_average_payoff_vs_{file_tag}_by_n.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path, pd.DataFrame(rows)


def plot_by_game(run_metrics: pd.DataFrame, x_col: str, x_label: str, file_tag: str) -> tuple[Path, pd.DataFrame]:
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.3), sharex=True, sharey=True)
    rows: list[dict[str, object]] = []
    for ax, game in zip(axes, GAME_ORDER, strict=True):
        sub = run_metrics[run_metrics["game_label"].eq(game)]
        for n in N_ORDER:
            n_sub = sub[sub["n_agents"].eq(n)]
            ax.scatter(
                n_sub[x_col],
                n_sub["average_payoff"],
                s=18,
                color=N_COLORS[n],
                alpha=0.42,
                linewidths=0,
            )
        add_fit(ax, sub, x_col)
        rows.append(fit_summary_row({"scope": "by_game", "game_label": game}, sub, x_col))
        style_axis(ax, GAME_TITLES[game], x_label, ylabel=ax is axes[0])
    handles = [
        Line2D([0], [0], marker="o", linestyle="none", color=N_COLORS[n], label=f"N={n}", markersize=5)
        for n in N_ORDER
    ]
    handles.append(Line2D([0], [0], color="#111111", lw=1.9, label="Linear fit"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.10), ncol=6, frameon=False, fontsize=9)
    fig.suptitle(f"Average payoff vs {x_label.lower()}, broken down by game", fontsize=15, y=1.03)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    out_path = OUT_DIR / f"heterogeneous_average_payoff_vs_{file_tag}_by_game.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path, pd.DataFrame(rows)


def plot_by_game_n(run_metrics: pd.DataFrame, x_col: str, x_label: str, file_tag: str) -> tuple[Path, pd.DataFrame]:
    fig, axes = plt.subplots(3, 5, figsize=(18.7, 9.0), sharex=True, sharey=True)
    rows: list[dict[str, object]] = []
    norm = plt.Normalize(vmin=float(run_metrics["competition_ci"].min()), vmax=float(run_metrics["competition_ci"].max()))
    for row_idx, game in enumerate(GAME_ORDER):
        for col_idx, n in enumerate(N_ORDER):
            ax = axes[row_idx, col_idx]
            sub = run_metrics[(run_metrics["game_label"].eq(game)) & (run_metrics["n_agents"].eq(n))]
            colors = plt.cm.viridis(norm(sub["competition_ci"].to_numpy(dtype=float)))
            ax.scatter(sub[x_col], sub["average_payoff"], s=16, color=colors, alpha=0.55, linewidths=0)
            add_fit(ax, sub, x_col)
            rows.append(fit_summary_row({"scope": "by_game_n", "game_label": game, "n_agents": n}, sub, x_col))
            style_axis(ax, f"{GAME_TITLES[game]}, N={n}", x_label, ylabel=col_idx == 0)
            if row_idx < len(GAME_ORDER) - 1:
                ax.set_xlabel("")
    fig.suptitle(f"Average payoff vs {x_label.lower()}, broken down by game and N", fontsize=15, y=1.01)
    fig.tight_layout()
    out_path = OUT_DIR / f"heterogeneous_average_payoff_vs_{file_tag}_by_game_n.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path, pd.DataFrame(rows)


def competition_bands(run_metrics: pd.DataFrame) -> list[str]:
    present = set(run_metrics["competition_band"].dropna().astype(str))
    ordered = [band for band in COMPETITION_BAND_ORDER if band in present]
    ordered.extend(sorted(present.difference(ordered)))
    return ordered


def plot_by_competition(run_metrics: pd.DataFrame, x_col: str, x_label: str, file_tag: str) -> tuple[Path, pd.DataFrame]:
    bands = competition_bands(run_metrics)
    fig, axes = plt.subplots(1, len(bands), figsize=(13.2, 4.3), sharex=True, sharey=True)
    axes_flat = np.asarray(axes).ravel()
    rows: list[dict[str, object]] = []
    for ax, band in zip(axes_flat, bands, strict=False):
        sub = run_metrics[run_metrics["competition_band"].eq(band)]
        for n in N_ORDER:
            n_sub = sub[sub["n_agents"].eq(n)]
            ax.scatter(
                n_sub[x_col],
                n_sub["average_payoff"],
                s=12,
                color=N_COLORS[n],
                alpha=0.38,
                linewidths=0,
            )
        add_fit(ax, sub, x_col, box=True)
        rows.append(
            fit_summary_row(
                {
                    "scope": "by_competition",
                    "competition_band": band,
                    "competition_band_title": COMPETITION_BAND_TITLES.get(band, band),
                },
                sub,
                x_col,
            )
        )
        ax.set_title(f"{COMPETITION_BAND_TITLES.get(band, band)}\nn={len(sub)}", fontsize=12, pad=6)
        ax.set_xlabel(x_label, fontsize=10)
        ax.grid(True, alpha=0.20, linewidth=0.5)
        ax.tick_params(axis="both", labelsize=8)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
    axes_flat[0].set_ylabel("Average payoff per agent", fontsize=10)
    handles = [
        Line2D([0], [0], marker="o", linestyle="none", color=N_COLORS[n], label=f"N={n}", markersize=5)
        for n in N_ORDER
    ]
    handles.append(Line2D([0], [0], color="#111111", lw=1.9, label="Linear fit"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=6, frameon=False, fontsize=9)
    fig.suptitle(f"Average payoff vs {x_label.lower()}, broken down by competition band", fontsize=15, y=1.03)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    out_path = OUT_DIR / f"heterogeneous_average_payoff_vs_{file_tag}_by_competition.png"
    fig.savefig(out_path, dpi=210, bbox_inches="tight")
    plt.close(fig)
    return out_path, pd.DataFrame(rows)


def plot_by_competition_n(run_metrics: pd.DataFrame, x_col: str, x_label: str, file_tag: str) -> tuple[Path, pd.DataFrame]:
    bands = competition_bands(run_metrics)
    fig, axes = plt.subplots(len(N_ORDER), len(bands), figsize=(12.6, 11.0), sharex=True, sharey=True)
    rows: list[dict[str, object]] = []
    for row_idx, n in enumerate(N_ORDER):
        for col_idx, band in enumerate(bands):
            ax = axes[row_idx, col_idx]
            sub = run_metrics[
                run_metrics["n_agents"].eq(n)
                & run_metrics["competition_band"].eq(band)
            ]
            for game in GAME_ORDER:
                game_sub = sub[sub["game_label"].eq(game)]
                ax.scatter(
                    game_sub[x_col],
                    game_sub["average_payoff"],
                    s=10,
                    color=GAME_COLORS[game],
                    alpha=0.48,
                    linewidths=0,
                )
            add_fit(ax, sub, x_col, box=False)
            rows.append(
                fit_summary_row(
                    {
                        "scope": "by_competition_n",
                        "competition_band": band,
                        "competition_band_title": COMPETITION_BAND_TITLES.get(band, band),
                        "n_agents": n,
                    },
                    sub,
                    x_col,
                )
            )
            if row_idx == 0:
                ax.set_title(COMPETITION_BAND_TITLES.get(band, band), fontsize=11, pad=5)
            if col_idx == 0:
                ax.set_ylabel(f"N={n}\nAvg payoff", fontsize=9)
            if row_idx == len(N_ORDER) - 1:
                ax.set_xlabel(x_label, fontsize=9)
            if len(sub) < 3:
                ax.text(0.5, 0.5, f"n={len(sub)}", transform=ax.transAxes, ha="center", va="center", fontsize=8, color="#777777")
            ax.grid(True, alpha=0.18, linewidth=0.45)
            ax.tick_params(axis="both", labelsize=7.5)
            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)
    handles = [
        Line2D([0], [0], marker="o", linestyle="none", color=GAME_COLORS[game], label=GAME_TITLES[game], markersize=5)
        for game in GAME_ORDER
    ]
    handles.append(Line2D([0], [0], color="#111111", lw=1.9, label="Linear fit"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.01), ncol=4, frameon=False, fontsize=9)
    fig.suptitle(f"Average payoff vs {x_label.lower()}, broken down by competition band and N", fontsize=15, y=1.02)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out_path = OUT_DIR / f"heterogeneous_average_payoff_vs_{file_tag}_by_competition_n.png"
    fig.savefig(out_path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return out_path, pd.DataFrame(rows)


def plot_by_competition_game(run_metrics: pd.DataFrame, x_col: str, x_label: str, file_tag: str) -> tuple[Path, pd.DataFrame]:
    bands = competition_bands(run_metrics)
    fig, axes = plt.subplots(len(GAME_ORDER), len(bands), figsize=(12.6, 7.6), sharex=True, sharey=True)
    rows: list[dict[str, object]] = []
    for row_idx, game in enumerate(GAME_ORDER):
        for col_idx, band in enumerate(bands):
            ax = axes[row_idx, col_idx]
            sub = run_metrics[
                run_metrics["game_label"].eq(game)
                & run_metrics["competition_band"].eq(band)
            ]
            for n in N_ORDER:
                n_sub = sub[sub["n_agents"].eq(n)]
                ax.scatter(
                    n_sub[x_col],
                    n_sub["average_payoff"],
                    s=10,
                    color=N_COLORS[n],
                    alpha=0.48,
                    linewidths=0,
                )
            add_fit(ax, sub, x_col, box=False)
            rows.append(
                fit_summary_row(
                    {
                        "scope": "by_competition_game",
                        "competition_band": band,
                        "competition_band_title": COMPETITION_BAND_TITLES.get(band, band),
                        "game_label": game,
                    },
                    sub,
                    x_col,
                )
            )
            if row_idx == 0:
                ax.set_title(COMPETITION_BAND_TITLES.get(band, band), fontsize=11, pad=5)
            if col_idx == 0:
                ax.set_ylabel(f"{GAME_TITLES[game]}\nAvg payoff", fontsize=9)
            if row_idx == len(GAME_ORDER) - 1:
                ax.set_xlabel(x_label, fontsize=9)
            if len(sub) < 3:
                ax.text(0.5, 0.5, f"n={len(sub)}", transform=ax.transAxes, ha="center", va="center", fontsize=8, color="#777777")
            ax.grid(True, alpha=0.18, linewidth=0.45)
            ax.tick_params(axis="both", labelsize=7.5)
            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)
    handles = [
        Line2D([0], [0], marker="o", linestyle="none", color=N_COLORS[n], label=f"N={n}", markersize=5)
        for n in N_ORDER
    ]
    handles.append(Line2D([0], [0], color="#111111", lw=1.9, label="Linear fit"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.015), ncol=6, frameon=False, fontsize=9)
    fig.suptitle(f"Average payoff vs {x_label.lower()}, broken down by competition band and game", fontsize=15, y=1.025)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out_path = OUT_DIR / f"heterogeneous_average_payoff_vs_{file_tag}_by_competition_game.png"
    fig.savefig(out_path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return out_path, pd.DataFrame(rows)


def plot_by_competition_n_game(run_metrics: pd.DataFrame, x_col: str, x_label: str, file_tag: str) -> tuple[Path, pd.DataFrame]:
    bands = competition_bands(run_metrics)
    columns = [(game, band) for game in GAME_ORDER for band in bands]
    fig, axes = plt.subplots(len(N_ORDER), len(columns), figsize=(25.5, 11.0), sharex=True, sharey=True)
    rows: list[dict[str, object]] = []
    for row_idx, n in enumerate(N_ORDER):
        for col_idx, (game, band) in enumerate(columns):
            ax = axes[row_idx, col_idx]
            sub = run_metrics[
                run_metrics["n_agents"].eq(n)
                & run_metrics["game_label"].eq(game)
                & run_metrics["competition_band"].eq(band)
            ]
            ax.scatter(
                sub[x_col],
                sub["average_payoff"],
                s=12,
                color=GAME_COLORS[game],
                alpha=0.52,
                linewidths=0,
            )
            add_fit(ax, sub, x_col, box=False)
            rows.append(
                fit_summary_row(
                    {
                        "scope": "by_competition_n_game",
                        "competition_band": band,
                        "competition_band_title": COMPETITION_BAND_TITLES.get(band, band),
                        "n_agents": n,
                        "game_label": game,
                    },
                    sub,
                    x_col,
                )
            )
            if row_idx == 0:
                ax.set_title(
                    f"{GAME_TITLES[game]}\n{COMPETITION_BAND_TITLES.get(band, band).replace(' competition', '')}",
                    fontsize=8,
                    pad=4,
                )
            if col_idx == 0:
                ax.set_ylabel(f"N={n}\nAvg payoff", fontsize=8)
            if row_idx == len(N_ORDER) - 1:
                ax.set_xlabel(x_label, fontsize=7)
            if len(sub) < 3:
                ax.text(
                    0.5,
                    0.5,
                    f"n={len(sub)}",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="#777777",
                )
            ax.grid(True, alpha=0.17, linewidth=0.4)
            ax.tick_params(axis="both", labelsize=6.5)
            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)
    handles = [
        Line2D([0], [0], marker="o", linestyle="none", color=GAME_COLORS[game], label=GAME_TITLES[game], markersize=5)
        for game in GAME_ORDER
    ]
    handles.append(Line2D([0], [0], color="#111111", lw=1.9, label="Linear fit"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.012), ncol=4, frameon=False, fontsize=9)
    fig.suptitle(
        f"Average payoff vs {x_label.lower()}, broken down by competition band, N, and game",
        fontsize=15,
        y=1.02,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    out_path = OUT_DIR / f"heterogeneous_average_payoff_vs_{file_tag}_by_competition_n_game.png"
    fig.savefig(out_path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return out_path, pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    run_metrics = load_run_metrics()
    run_metrics_path = OUT_DIR / "heterogeneous_average_payoff_vs_roster_elo_run_metrics.csv"
    run_metrics.to_csv(run_metrics_path, index=False)

    all_fits: list[pd.DataFrame] = []
    plot_paths: list[Path] = []
    for x_col, x_label, file_tag in X_SPECS:
        overall_path, overall_fit = plot_overall(run_metrics, x_col, x_label, file_tag)
        by_n_path, by_n_fit = plot_by_n(run_metrics, x_col, x_label, file_tag)
        by_game_path, by_game_fit = plot_by_game(run_metrics, x_col, x_label, file_tag)
        by_game_n_path, by_game_n_fit = plot_by_game_n(run_metrics, x_col, x_label, file_tag)
        by_comp_path, by_comp_fit = plot_by_competition(run_metrics, x_col, x_label, file_tag)
        by_comp_n_path, by_comp_n_fit = plot_by_competition_n(run_metrics, x_col, x_label, file_tag)
        by_comp_game_path, by_comp_game_fit = plot_by_competition_game(run_metrics, x_col, x_label, file_tag)
        by_comp_n_game_path, by_comp_n_game_fit = plot_by_competition_n_game(run_metrics, x_col, x_label, file_tag)
        plot_paths.extend([
            overall_path,
            by_n_path,
            by_game_path,
            by_game_n_path,
            by_comp_path,
            by_comp_n_path,
            by_comp_game_path,
            by_comp_n_game_path,
        ])
        all_fits.extend([
            overall_fit,
            by_n_fit,
            by_game_fit,
            by_game_n_fit,
            by_comp_fit,
            by_comp_n_fit,
            by_comp_game_fit,
            by_comp_n_game_fit,
        ])

    fit_summary = pd.concat(all_fits, ignore_index=True)
    fit_summary_path = OUT_DIR / "heterogeneous_average_payoff_vs_roster_elo_fit_summary.csv"
    fit_summary.to_csv(fit_summary_path, index=False)

    print(f"Wrote {run_metrics_path}")
    print(f"Wrote {fit_summary_path}")
    for path in plot_paths:
        print(f"Wrote {path}")
    print()
    print(fit_summary[fit_summary["scope"].isin(["overall", "by_game"])].to_string(index=False))


if __name__ == "__main__":
    main()
