#!/usr/bin/env python3
"""Scatter max payoff against max roster Elo for heterogeneous runs."""

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
            max_roster_elo=("elo", "max"),
            mean_roster_elo=("elo", "mean"),
            elo_variance=("elo", lambda s: float(np.var(s.to_numpy(dtype=float), ddof=0))),
            max_payoff=("final_utility", "max"),
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
    data = frame[["max_roster_elo", "max_payoff"]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < 3 or data["max_roster_elo"].nunique() < 2:
        return {
            "n_runs": int(len(data)),
            "slope": math.nan,
            "intercept": math.nan,
            "pearson_r": math.nan,
            "r_squared": math.nan,
            "p_value": math.nan,
            "stderr": math.nan,
        }
    fit = stats.linregress(data["max_roster_elo"].to_numpy(dtype=float), data["max_payoff"].to_numpy(dtype=float))
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


def plot_scatter(run_metrics: pd.DataFrame, fit: dict[str, float]) -> Path:
    fig, ax = plt.subplots(figsize=(7.4, 5.3))
    for n_agents in N_ORDER:
        sub = run_metrics[run_metrics["n_agents"].eq(n_agents)]
        if sub.empty:
            continue
        ax.scatter(
            sub["max_roster_elo"],
            sub["max_payoff"],
            s=22,
            alpha=0.36,
            linewidths=0,
            color=N_COLORS[n_agents],
            label=f"N={n_agents}",
        )

    if math.isfinite(fit["slope"]):
        xs = np.linspace(float(run_metrics["max_roster_elo"].min()), float(run_metrics["max_roster_elo"].max()), 200)
        ys = fit["slope"] * xs + fit["intercept"]
        ax.plot(xs, ys, color="#111111", linewidth=2.1, label="Overall fit")
        ax.text(
            0.035,
            0.96,
            (
                f"slope/100={fit['slope'] * 100:+.2f}\n"
                f"r={fit['pearson_r']:+.2f}\n"
                f"{p_text(fit['p_value'])}, n={fit['n_runs']}"
            ),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "#BBBBBB", "alpha": 0.88, "pad": 4},
        )

    ax.set_xlabel("Max roster Elo", fontsize=11)
    ax.set_ylabel("Max payoff in run", fontsize=11)
    ax.set_title("Heterogeneous runs: max payoff vs max roster Elo", fontsize=13, pad=10)
    ax.grid(True, alpha=0.22, linewidth=0.6)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    handles = [
        Line2D([0], [0], marker="o", linestyle="", color=N_COLORS[n], label=f"N={n}", markersize=6, alpha=0.8)
        for n in N_ORDER
    ]
    handles.append(Line2D([0], [0], color="#111111", lw=2.1, label="Overall fit"))
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=9)
    fig.tight_layout()
    out_path = OUT_DIR / "heterogeneous_max_payoff_vs_max_elo_overall.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    run_metrics = load_run_metrics()
    fit = fit_line(run_metrics)
    out_path = plot_scatter(run_metrics, fit)

    run_metrics_path = OUT_DIR / "heterogeneous_max_payoff_vs_max_elo_run_metrics.csv"
    fit_path = OUT_DIR / "heterogeneous_max_payoff_vs_max_elo_fit_summary.csv"
    run_metrics.to_csv(run_metrics_path, index=False)
    pd.DataFrame([fit]).to_csv(fit_path, index=False)

    print(f"Wrote {out_path}")
    print(f"Wrote {run_metrics_path}")
    print(f"Wrote {fit_path}")
    print(
        "Fit: "
        f"n={fit['n_runs']}, slope={fit['slope']:.6f}, "
        f"slope/100={fit['slope'] * 100:.4f}, "
        f"r={fit['pearson_r']:.4f}, R^2={fit['r_squared']:.4f}, p={fit['p_value']:.6g}"
    )


if __name__ == "__main__":
    main()
