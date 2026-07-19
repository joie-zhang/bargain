#!/usr/bin/env python3
"""Plot heterogeneous corrected payoff Gini against within-roster Elo variance."""

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
GAME_ORDER = ["game1", "game2", "game3"]
GAME_TITLES = {"game1": "Game 1", "game2": "Game 2", "game3": "Game 3"}
COMPETITION_ORDER = ["cooperative", "middle", "competitive"]
COMPETITION_TITLES = {
    "cooperative": "Low competition",
    "middle": "Medium competition",
    "competitive": "High competition",
}

POINT_COLOR = "#4E79A7"
FIT_COLOR = "#111111"


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
            elo_variance=("elo", lambda s: float(np.var(s.to_numpy(dtype=float), ddof=0))),
            elo_std=("elo", lambda s: float(np.std(s.to_numpy(dtype=float), ddof=0))),
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
    return run_metrics


def fit_line(frame: pd.DataFrame) -> dict[str, float]:
    data = frame[["elo_variance", "payoff_gini_corrected"]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < 3 or data["elo_variance"].nunique() < 2:
        return {
            "n_runs": int(len(data)),
            "slope": math.nan,
            "intercept": math.nan,
            "pearson_r": math.nan,
            "r_squared": math.nan,
            "p_value": math.nan,
            "stderr": math.nan,
        }
    fit = stats.linregress(
        data["elo_variance"].to_numpy(dtype=float),
        data["payoff_gini_corrected"].to_numpy(dtype=float),
    )
    return {
        "n_runs": int(len(data)),
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


def subset_frame(frame: pd.DataFrame, filters: dict[str, object]) -> pd.DataFrame:
    out = frame
    for col, value in filters.items():
        out = out[out[col].eq(value)]
    return out


def draw_scatter_axis(
    ax: plt.Axes,
    frame: pd.DataFrame,
    x_lim: tuple[float, float],
    y_lim: tuple[float, float],
    annotate: bool = True,
    point_size: float = 20.0,
) -> dict[str, float]:
    fit = fit_line(frame)
    if len(frame):
        ax.scatter(
            frame["elo_variance"],
            frame["payoff_gini_corrected"],
            s=point_size,
            alpha=0.38,
            linewidths=0,
            color=POINT_COLOR,
        )
    if math.isfinite(fit["slope"]) and len(frame):
        xs = np.linspace(float(frame["elo_variance"].min()), float(frame["elo_variance"].max()), 160)
        ax.plot(xs, fit["slope"] * xs + fit["intercept"], color=FIT_COLOR, linewidth=1.6)
    if annotate:
        text = (
            f"r={fit['pearson_r']:.2f}\n"
            f"{p_text(fit['p_value'])}\n"
            f"n={fit['n_runs']}"
        )
        ax.text(
            0.03,
            0.95,
            text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7.2,
            bbox={"facecolor": "white", "edgecolor": "#BBBBBB", "alpha": 0.82, "pad": 2.5},
        )
    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)
    ax.grid(True, alpha=0.22, linewidth=0.55)
    ax.tick_params(axis="both", labelsize=7.5)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    return fit


def fit_row(scope: str, row_label: str, col_label: str, filters: dict[str, object], frame: pd.DataFrame) -> dict[str, object]:
    sub = subset_frame(frame, filters)
    fit = fit_line(sub)
    row: dict[str, object] = {
        "scope": scope,
        "row_label": row_label,
        "col_label": col_label,
        "filters": ";".join(f"{key}={value}" for key, value in filters.items()),
    }
    row.update(fit)
    return row


def plot_grid(
    frame: pd.DataFrame,
    row_groups: list[tuple[str, dict[str, object], dict[str, object]]],
    col_groups: list[tuple[str, dict[str, object], dict[str, object]]],
    scope: str,
    title: str,
    filename: str,
    figsize: tuple[float, float],
    x_lim: tuple[float, float],
    y_lim: tuple[float, float],
    annotate: bool = True,
) -> tuple[Path, pd.DataFrame]:
    fig, axes = plt.subplots(len(row_groups), len(col_groups), figsize=figsize, sharex=True, sharey=True, squeeze=False)
    fit_rows: list[dict[str, object]] = []
    for row_idx, (row_label, row_filter, row_extra) in enumerate(row_groups):
        for col_idx, (col_label, col_filter, col_extra) in enumerate(col_groups):
            filters = {**row_filter, **col_filter}
            sub = subset_frame(frame, filters)
            ax = axes[row_idx, col_idx]
            draw_scatter_axis(ax, sub, x_lim, y_lim, annotate=annotate, point_size=18.0)
            if row_idx == 0:
                ax.set_title(col_label, fontsize=9, pad=5)
            if col_idx == 0:
                ax.set_ylabel(f"{row_label}\nCorrected Gini" if len(row_groups) > 1 else "Corrected Gini", fontsize=8.5)
            if row_idx == len(row_groups) - 1:
                ax.set_xlabel("Elo variance", fontsize=8.5)
            else:
                ax.tick_params(axis="x", labelbottom=False)
            row = fit_row(scope, row_label, col_label, filters, frame)
            row.update(row_extra)
            row.update(col_extra)
            fit_rows.append(row)
    fig.suptitle(title, fontsize=13, y=1.01)
    fig.tight_layout()
    out_path = OUT_DIR / filename
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path, pd.DataFrame(fit_rows)


def plot_overall(frame: pd.DataFrame, x_lim: tuple[float, float], y_lim: tuple[float, float]) -> tuple[Path, pd.DataFrame]:
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    fit = draw_scatter_axis(ax, frame, x_lim, y_lim, annotate=False, point_size=24.0)
    if math.isfinite(fit["slope"]):
        ax.text(
            0.03,
            0.95,
            (
                f"slope={fit['slope']:.6f}\n"
                f"r={fit['pearson_r']:.2f}, R^2={fit['r_squared']:.2f}\n"
                f"{p_text(fit['p_value'])}, n={fit['n_runs']}"
            ),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "#BBBBBB", "alpha": 0.88, "pad": 4},
        )
    ax.set_xlabel("Within-roster Elo variance", fontsize=11)
    ax.set_ylabel("Corrected payoff Gini", fontsize=11)
    ax.set_title("Heterogeneous runs: corrected payoff Gini vs Elo variance", fontsize=13, pad=10)
    legend = [Line2D([0], [0], marker="o", color="none", markerfacecolor=POINT_COLOR, alpha=0.45, markersize=7, label="Runs")]
    legend.append(Line2D([0], [0], color=FIT_COLOR, lw=1.8, label="Linear fit"))
    ax.legend(handles=legend, frameon=False, fontsize=9, loc="upper right")
    fig.tight_layout()
    out_path = OUT_DIR / "heterogeneous_payoff_gini_vs_elo_variance_overall.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    row = {"scope": "overall", "row_label": "All", "col_label": "All", "filters": ""}
    row.update(fit)
    return out_path, pd.DataFrame([row])


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    run_metrics = load_run_metrics()
    x_lim = (0.0, float(run_metrics["elo_variance"].max()) * 1.04)
    y_lim = (0.0, min(1.0, float(run_metrics["payoff_gini_corrected"].max()) * 1.08 + 0.01))

    all_rows: list[pd.DataFrame] = []
    paths: list[Path] = []

    path, fits = plot_overall(run_metrics, x_lim, y_lim)
    paths.append(path)
    all_rows.append(fits)

    n_cols = [(f"N={n}", {"n_agents": n}, {"n_agents": n}) for n in N_ORDER]
    game_cols = [(GAME_TITLES[g], {"game_label": g}, {"game_label": g}) for g in GAME_ORDER]
    comp_cols = [(COMPETITION_TITLES[c], {"competition_band": c}, {"competition_band": c}) for c in COMPETITION_ORDER]
    all_rows_group = [("All", {}, {})]

    plot_specs = [
        (
            all_rows_group,
            game_cols,
            "by_game",
            "Heterogeneous corrected payoff Gini vs Elo variance, by game",
            "heterogeneous_payoff_gini_vs_elo_variance_by_game.png",
            (11.0, 3.8),
        ),
        (
            all_rows_group,
            n_cols,
            "by_n",
            "Heterogeneous corrected payoff Gini vs Elo variance, by N",
            "heterogeneous_payoff_gini_vs_elo_variance_by_n.png",
            (16.0, 3.8),
        ),
        (
            all_rows_group,
            comp_cols,
            "by_competition",
            "Heterogeneous corrected payoff Gini vs Elo variance, by competition band",
            "heterogeneous_payoff_gini_vs_elo_variance_by_competition.png",
            (11.0, 3.8),
        ),
        (
            game_cols,
            n_cols,
            "by_game_n",
            "Heterogeneous corrected payoff Gini vs Elo variance, by game and N",
            "heterogeneous_payoff_gini_vs_elo_variance_by_game_n.png",
            (16.0, 8.0),
        ),
        (
            comp_cols,
            game_cols,
            "by_competition_game",
            "Heterogeneous corrected payoff Gini vs Elo variance, by competition band and game",
            "heterogeneous_payoff_gini_vs_elo_variance_by_competition_game.png",
            (11.0, 8.0),
        ),
    ]
    for row_groups, col_groups, scope, title, filename, figsize in plot_specs:
        path, fits = plot_grid(run_metrics, row_groups, col_groups, scope, title, filename, figsize, x_lim, y_lim)
        paths.append(path)
        all_rows.append(fits)

    for n in N_ORDER:
        filtered = subset_frame(run_metrics, {"n_agents": n})
        path, fits = plot_grid(
            filtered,
            game_cols,
            comp_cols,
            f"by_game_competition_n{n}",
            f"Heterogeneous corrected payoff Gini vs Elo variance, by game and competition band for N={n}",
            f"heterogeneous_payoff_gini_vs_elo_variance_by_game_competition_n{n}.png",
            (11.0, 8.0),
            x_lim,
            y_lim,
        )
        fits["n_agents"] = n
        paths.append(path)
        all_rows.append(fits)

    run_metrics_path = OUT_DIR / "heterogeneous_payoff_gini_vs_elo_variance_run_metrics.csv"
    fit_summary_path = OUT_DIR / "heterogeneous_payoff_gini_vs_elo_variance_fit_summary.csv"
    run_metrics.to_csv(run_metrics_path, index=False)
    pd.concat(all_rows, ignore_index=True).to_csv(fit_summary_path, index=False)

    for path in paths:
        print(f"Wrote {path}")
    print(f"Wrote {run_metrics_path}")
    print(f"Wrote {fit_summary_path}")
    overall = all_rows[0].iloc[0]
    print(
        "Overall fit: "
        f"n={int(overall['n_runs'])}, slope={overall['slope']:.8f}, "
        f"r={overall['pearson_r']:.4f}, R^2={overall['r_squared']:.4f}, p={overall['p_value']:.6g}"
    )


if __name__ == "__main__":
    main()
