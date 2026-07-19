#!/usr/bin/env python3
"""Scatter average payoff against within-roster Elo spread for heterogeneous runs."""

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
            elo_variance=("elo", lambda s: float(np.var(s.to_numpy(dtype=float), ddof=0))),
            elo_std=("elo", lambda s: float(np.std(s.to_numpy(dtype=float), ddof=0))),
            mean_roster_elo=("elo", "mean"),
            max_roster_elo=("elo", "max"),
            average_payoff=("final_utility", "mean"),
            total_payoff=("final_utility", "sum"),
            payoff_variance=("final_utility", lambda s: float(np.var(s.to_numpy(dtype=float), ddof=0))),
            n_agents_observed=("final_utility", "count"),
            model_count=("model", "nunique"),
        )
        .reset_index()
        .sort_values(["n_agents", "game_label", "config_id"])
    )
    return run_metrics


def fit_line(frame: pd.DataFrame, x_col: str) -> dict[str, float]:
    data = frame[[x_col, "average_payoff"]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < 3 or data[x_col].nunique() < 2:
        return {
            "n_runs": int(len(data)),
            "slope": math.nan,
            "intercept": math.nan,
            "pearson_r": math.nan,
            "r_squared": math.nan,
            "p_value": math.nan,
            "stderr": math.nan,
        }
    fit = stats.linregress(data[x_col].to_numpy(dtype=float), data["average_payoff"].to_numpy(dtype=float))
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


def plot_scatter(
    run_metrics: pd.DataFrame,
    fit: dict[str, float],
    x_col: str,
    x_label: str,
    title: str,
    filename: str,
) -> Path:
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    for n_agents in N_ORDER:
        sub = run_metrics[run_metrics["n_agents"].eq(n_agents)]
        if sub.empty:
            continue
        ax.scatter(
            sub[x_col],
            sub["average_payoff"],
            s=28,
            alpha=0.45,
            linewidths=0,
            color=N_COLORS[n_agents],
            label=f"N={n_agents}",
        )

    if math.isfinite(fit["slope"]):
        xs = np.linspace(float(run_metrics[x_col].min()), float(run_metrics[x_col].max()), 200)
        ys = fit["slope"] * xs + fit["intercept"]
        ax.plot(xs, ys, color="#111111", linewidth=2.2, label="Overall fit")
        ax.text(
            0.03,
            0.04,
            (
                f"slope={fit['slope']:.4f}\n"
                f"r={fit['pearson_r']:.2f}, R^2={fit['r_squared']:.2f}\n"
                f"{p_text(fit['p_value'])}, n={fit['n_runs']}"
            ),
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "#BBBBBB", "alpha": 0.88, "pad": 4},
        )

    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel("Average payoff", fontsize=11)
    ax.set_title(title, fontsize=13, pad=10)
    ax.grid(True, alpha=0.22, linewidth=0.6)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    handles = [
        Line2D([0], [0], marker="o", linestyle="", color=N_COLORS[n], label=f"N={n}", markersize=6, alpha=0.8)
        for n in N_ORDER
    ]
    handles.append(Line2D([0], [0], color="#111111", lw=2.2, label="Overall fit"))
    ax.legend(handles=handles, frameon=False, fontsize=9, loc="upper right")
    fig.tight_layout()
    out_path = OUT_DIR / filename
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    run_metrics = load_run_metrics()
    fit = fit_line(run_metrics, "elo_variance")
    out_path = plot_scatter(
        run_metrics,
        fit,
        "elo_variance",
        "Within-roster Elo variance",
        "Heterogeneous runs: average payoff vs Elo variance",
        "heterogeneous_average_payoff_vs_elo_variance.png",
    )
    std_fit = fit_line(run_metrics, "elo_std")
    std_out_path = plot_scatter(
        run_metrics,
        std_fit,
        "elo_std",
        "Within-roster Elo standard deviation",
        "Heterogeneous runs: average payoff vs Elo standard deviation",
        "heterogeneous_average_payoff_vs_elo_std.png",
    )

    run_metrics_path = OUT_DIR / "heterogeneous_average_payoff_vs_elo_variance_run_metrics.csv"
    fit_path = OUT_DIR / "heterogeneous_average_payoff_vs_elo_variance_fit_summary.csv"
    std_fit_path = OUT_DIR / "heterogeneous_average_payoff_vs_elo_std_fit_summary.csv"
    run_metrics.to_csv(run_metrics_path, index=False)
    pd.DataFrame([fit]).to_csv(fit_path, index=False)
    pd.DataFrame([std_fit]).to_csv(std_fit_path, index=False)

    print(f"Wrote {out_path}")
    print(f"Wrote {std_out_path}")
    print(f"Wrote {run_metrics_path}")
    print(f"Wrote {fit_path}")
    print(f"Wrote {std_fit_path}")
    print(
        "Fit: "
        f"n={fit['n_runs']}, slope={fit['slope']:.6f}, "
        f"r={fit['pearson_r']:.4f}, R^2={fit['r_squared']:.4f}, p={fit['p_value']:.6g}"
    )
    print(
        "Std fit: "
        f"n={std_fit['n_runs']}, slope={std_fit['slope']:.6f}, "
        f"r={std_fit['pearson_r']:.4f}, R^2={std_fit['r_squared']:.4f}, p={std_fit['p_value']:.6g}"
    )


if __name__ == "__main__":
    main()
