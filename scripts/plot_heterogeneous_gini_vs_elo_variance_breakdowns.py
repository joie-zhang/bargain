#!/usr/bin/env python3
"""Plot heterogeneous payoff Gini against within-roster Elo variance."""

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
GAME_TITLES = {"game1": "Game 1", "game2": "Game 2", "game3": "Game 3"}
GAME_COLORS = {"game1": "#4E79A7", "game2": "#59A14F", "game3": "#E15759"}
COMPETITION_ORDER = ["cooperative", "middle", "competitive"]
COMPETITION_TITLES = {
    "cooperative": "Low competition",
    "middle": "Medium competition",
    "competitive": "High competition",
}


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
            average_payoff=("final_utility", "mean"),
            payoff_variance=("final_utility", lambda s: float(np.var(s.to_numpy(dtype=float), ddof=0))),
            payoff_gini_corrected=("final_utility", gini_shifted_corrected),
            n_agents_observed=("final_utility", "count"),
            model_count=("model", "nunique"),
        )
        .reset_index()
        .sort_values(["game_label", "n_agents", "competition_ci", "config_id"])
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
    fit = stats.linregress(data["elo_variance"].to_numpy(dtype=float), data["payoff_gini_corrected"].to_numpy(dtype=float))
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


def annotate_fit(ax: plt.Axes, fit: dict[str, float], compact: bool) -> None:
    if compact:
        text = f"r={fit['pearson_r']:.2f}\n{p_text(fit['p_value'])}\nn={fit['n_runs']}"
        fontsize = 6.5
    else:
        text = (
            f"slope={fit['slope']:.2e}\n"
            f"r={fit['pearson_r']:.2f}, R^2={fit['r_squared']:.2f}\n"
            f"{p_text(fit['p_value'])}, n={fit['n_runs']}"
        )
        fontsize = 8.5
    ax.text(
        0.04,
        0.05,
        text,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=fontsize,
        bbox={"facecolor": "white", "edgecolor": "#BBBBBB", "alpha": 0.82, "pad": 2.5},
    )


def draw_axis(
    ax: plt.Axes,
    frame: pd.DataFrame,
    color_by: str = "n_agents",
    compact_annotation: bool = False,
    show_legend: bool = False,
) -> dict[str, float]:
    if color_by == "n_agents":
        groups = [(n, frame[frame["n_agents"].eq(n)], N_COLORS[n], f"N={n}") for n in N_ORDER]
    elif color_by == "game_label":
        groups = [(g, frame[frame["game_label"].eq(g)], GAME_COLORS[g], GAME_TITLES[g]) for g in GAME_ORDER]
    else:
        groups = [("all", frame, "#4E79A7", "Runs")]

    for _, sub, color, label in groups:
        if sub.empty:
            continue
        ax.scatter(
            sub["elo_variance"],
            sub["payoff_gini_corrected"],
            s=18 if compact_annotation else 24,
            alpha=0.38,
            linewidths=0,
            color=color,
            label=label,
        )

    fit = fit_line(frame)
    if math.isfinite(fit["slope"]):
        xs = np.linspace(float(frame["elo_variance"].min()), float(frame["elo_variance"].max()), 160)
        ys = fit["slope"] * xs + fit["intercept"]
        ax.plot(xs, ys, color="#111111", linewidth=1.8 if compact_annotation else 2.2)
    annotate_fit(ax, fit, compact_annotation)

    ax.grid(True, alpha=0.22, linewidth=0.55)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    if show_legend:
        ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    return fit


def set_common_axes(axes: np.ndarray, run_metrics: pd.DataFrame) -> None:
    x_max = max(float(run_metrics["elo_variance"].max()) * 1.04, 1.0)
    y_max = min(max(float(run_metrics["payoff_gini_corrected"].max()) * 1.08, 0.05), 1.02)
    for ax in axes.ravel():
        ax.set_xlim(0, x_max)
        ax.set_ylim(0, y_max)


def save_overall(run_metrics: pd.DataFrame) -> tuple[Path, list[dict[str, object]]]:
    fig, ax = plt.subplots(figsize=(6.4, 5.3))
    fit = draw_axis(ax, run_metrics, color_by="n_agents", compact_annotation=False, show_legend=True)
    ax.set_xlabel("Within-roster Elo variance", fontsize=11)
    ax.set_ylabel("Corrected payoff Gini", fontsize=11)
    ax.set_title("Heterogeneous runs: payoff Gini vs Elo variance", fontsize=13, pad=10)
    set_common_axes(np.asarray([[ax]]), run_metrics)
    fig.tight_layout()
    out_path = OUT_DIR / "heterogeneous_payoff_gini_vs_elo_variance_overall.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path, [{"scope": "overall", "row_label": "All", "col_label": "All", **fit}]


def save_one_row(
    run_metrics: pd.DataFrame,
    groups: list[tuple[str, dict[str, object]]],
    scope: str,
    title: str,
    filename: str,
    color_by: str = "n_agents",
) -> tuple[Path, list[dict[str, object]]]:
    fig, axes = plt.subplots(1, len(groups), figsize=(4.25 * len(groups), 4.1), sharex=True, sharey=True, squeeze=False)
    fit_rows: list[dict[str, object]] = []
    for ax, (label, filters) in zip(axes.ravel(), groups, strict=True):
        sub = apply_filters(run_metrics, filters)
        fit = draw_axis(ax, sub, color_by=color_by, compact_annotation=True)
        ax.set_title(label, fontsize=10, pad=6)
        fit_rows.append({"scope": scope, "row_label": "All", "col_label": label, **filters, **fit})
    set_common_axes(axes, run_metrics)
    axes[0, 0].set_ylabel("Corrected payoff Gini", fontsize=10)
    for ax in axes.ravel():
        ax.set_xlabel("Elo variance", fontsize=9)
    fig.suptitle(title, fontsize=13, y=1.02)
    add_legend(fig, color_by)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    out_path = OUT_DIR / filename
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path, fit_rows


def apply_filters(frame: pd.DataFrame, filters: dict[str, object]) -> pd.DataFrame:
    out = frame
    for col, value in filters.items():
        out = out[out[col].eq(value)]
    return out


def add_legend(fig: plt.Figure, color_by: str) -> None:
    if color_by == "n_agents":
        handles = [
            Line2D([0], [0], marker="o", linestyle="", color=N_COLORS[n], label=f"N={n}", markersize=5.5, alpha=0.8)
            for n in N_ORDER
        ]
    elif color_by == "game_label":
        handles = [
            Line2D([0], [0], marker="o", linestyle="", color=GAME_COLORS[g], label=GAME_TITLES[g], markersize=5.5, alpha=0.8)
            for g in GAME_ORDER
        ]
    else:
        handles = []
    if handles:
        handles.append(Line2D([0], [0], color="#111111", lw=1.9, label="Fit"))
        fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=8.5)


def save_grid(
    run_metrics: pd.DataFrame,
    row_groups: list[tuple[str, dict[str, object]]],
    col_groups: list[tuple[str, dict[str, object]]],
    scope: str,
    title: str,
    filename: str,
    color_by: str = "all",
    figsize: tuple[float, float] | None = None,
) -> tuple[Path, list[dict[str, object]]]:
    if figsize is None:
        figsize = (3.25 * len(col_groups), 2.75 * len(row_groups))
    fig, axes = plt.subplots(len(row_groups), len(col_groups), figsize=figsize, sharex=True, sharey=True, squeeze=False)
    fit_rows: list[dict[str, object]] = []
    for row_idx, (row_label, row_filters) in enumerate(row_groups):
        for col_idx, (col_label, col_filters) in enumerate(col_groups):
            ax = axes[row_idx, col_idx]
            filters = {**row_filters, **col_filters}
            sub = apply_filters(run_metrics, filters)
            fit = draw_axis(ax, sub, color_by=color_by, compact_annotation=True)
            if row_idx == 0:
                ax.set_title(col_label, fontsize=9, pad=5)
            if col_idx == 0:
                ax.set_ylabel(f"{row_label}\nGini", fontsize=8.5)
            if row_idx == len(row_groups) - 1:
                ax.set_xlabel("Elo variance", fontsize=8.5)
            fit_rows.append({"scope": scope, "row_label": row_label, "col_label": col_label, **filters, **fit})
    set_common_axes(axes, run_metrics)
    fig.suptitle(title, fontsize=13, y=1.01)
    if color_by != "all":
        add_legend(fig, color_by)
        rect = (0, 0.06, 1, 1)
    else:
        rect = (0, 0.02, 1, 1)
    fig.tight_layout(rect=rect)
    out_path = OUT_DIR / filename
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path, fit_rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    run_metrics = load_run_metrics()

    outputs: list[Path] = []
    fit_rows: list[dict[str, object]] = []

    path, rows = save_overall(run_metrics)
    outputs.append(path)
    fit_rows.extend(rows)

    game_groups = [(GAME_TITLES[g], {"game_label": g}) for g in GAME_ORDER]
    competition_groups = [(COMPETITION_TITLES[c], {"competition_band": c}) for c in COMPETITION_ORDER]
    n_groups = [(f"N={n}", {"n_agents": n}) for n in N_ORDER]

    for args in [
        (
            game_groups,
            "by_game",
            "Heterogeneous payoff Gini vs Elo variance by game",
            "heterogeneous_payoff_gini_vs_elo_variance_by_game.png",
            "n_agents",
        ),
        (
            competition_groups,
            "by_competition",
            "Heterogeneous payoff Gini vs Elo variance by competition band",
            "heterogeneous_payoff_gini_vs_elo_variance_by_competition.png",
            "n_agents",
        ),
        (
            n_groups,
            "by_n",
            "Heterogeneous payoff Gini vs Elo variance by N",
            "heterogeneous_payoff_gini_vs_elo_variance_by_n.png",
            "game_label",
        ),
    ]:
        path, rows = save_one_row(run_metrics, *args)
        outputs.append(path)
        fit_rows.extend(rows)

    grid_specs = [
        (
            game_groups,
            n_groups,
            "by_game_n",
            "Heterogeneous payoff Gini vs Elo variance by game and N",
            "heterogeneous_payoff_gini_vs_elo_variance_by_game_n.png",
            "all",
            (16.5, 8.6),
        ),
        (
            game_groups,
            competition_groups,
            "by_game_competition",
            "Heterogeneous payoff Gini vs Elo variance by game and competition band",
            "heterogeneous_payoff_gini_vs_elo_variance_by_game_competition.png",
            "n_agents",
            (11.8, 8.7),
        ),
        (
            competition_groups,
            n_groups,
            "by_competition_n",
            "Heterogeneous payoff Gini vs Elo variance by competition band and N",
            "heterogeneous_payoff_gini_vs_elo_variance_by_competition_n.png",
            "all",
            (16.5, 8.6),
        ),
    ]
    for row_groups, col_groups, scope, title, filename, color_by, figsize in grid_specs:
        path, rows = save_grid(run_metrics, row_groups, col_groups, scope, title, filename, color_by, figsize)
        outputs.append(path)
        fit_rows.extend(rows)

    for n, filters in n_groups:
        n_value = int(filters["n_agents"])
        path, rows = save_grid(
            apply_filters(run_metrics, filters),
            game_groups,
            competition_groups,
            f"by_game_competition_n{n_value}",
            f"Heterogeneous payoff Gini vs Elo variance by game and competition band for {n}",
            f"heterogeneous_payoff_gini_vs_elo_variance_by_game_competition_n{n_value}.png",
            "all",
            (11.8, 8.7),
        )
        outputs.append(path)
        fit_rows.extend(rows)

    run_metrics_path = OUT_DIR / "heterogeneous_payoff_gini_vs_elo_variance_run_metrics.csv"
    fit_summary_path = OUT_DIR / "heterogeneous_payoff_gini_vs_elo_variance_fit_summary.csv"
    run_metrics.to_csv(run_metrics_path, index=False)
    pd.DataFrame(fit_rows).to_csv(fit_summary_path, index=False)

    print(f"Wrote {run_metrics_path}")
    print(f"Wrote {fit_summary_path}")
    for path in outputs:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
