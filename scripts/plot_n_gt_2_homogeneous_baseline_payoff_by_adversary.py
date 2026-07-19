#!/usr/bin/env python3
"""Recreate homogeneous-adversary Figure 8 and plot baseline payoff vs N.

The input is the prepared homogeneous run table from the N=2 + N>2 analysis.
Both plots aggregate homogeneous-adversary runs over competition settings,
random seeds, and whether the focal adversary starts first or last.
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
RUN_TABLE = (
    PROJECT_ROOT
    / "experiments/results/n2_plus_multiagent_comparison_analysis_20260505"
    / "tables_multiagent/homogeneous_runs_fresh.csv"
)
OVERLEAF_OUT = PROJECT_ROOT / "overleaf/neurips/graphics/n_gt_2_report"
ITERATION_OUT = (
    PROJECT_ROOT
    / "experiments/results/figure_iteration_20260626/homogeneous_baseline_payoff"
)

GAME_ORDER = ["game1", "game2", "game3"]
GAME_TITLES = {
    "game1": "Game 1: Item Allocation",
    "game2": "Game 2: Diplomacy",
    "game3": "Game 3: Co-funding",
}
N_ORDER = [2, 4, 6, 8, 10]
N_COLORS = {
    2: "#1f77b4",
    4: "#d62728",
    6: "#2ca02c",
    8: "#9467bd",
    10: "#ff7f0e",
}
MODEL_LABELS = {
    "amazon-nova-micro-v1.0": "Nova Micro",
    "gpt-4o-mini-2024-07-18": "GPT-4o mini",
    "claude-sonnet-4-20250514": "Sonnet 4",
    "gemini-2.5-pro": "Gemini 2.5 Pro",
    "gpt-5.4-high": "GPT-5.4 High",
}


def sem(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if len(clean) <= 1:
        return 0.0
    return float(clean.std(ddof=1) / math.sqrt(len(clean)))


def linear_fit(x: pd.Series, y: pd.Series) -> tuple[float, float, float]:
    data = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < 2 or data["x"].nunique() < 2:
        return math.nan, math.nan, math.nan
    x_arr = data["x"].to_numpy(dtype=float)
    y_arr = data["y"].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x_arr, y_arr, 1)
    pred = slope * x_arr + intercept
    ss_res = float(np.sum((y_arr - pred) ** 2))
    ss_tot = float(np.sum((y_arr - np.mean(y_arr)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else math.nan
    return float(slope), float(intercept), float(r_squared)


def load_homogeneous_adversary_runs() -> pd.DataFrame:
    runs = pd.read_csv(RUN_TABLE)
    runs = runs[runs["experiment_family"].eq("homogeneous_adversary")].copy()
    for col in ["n_agents", "adversary_elo", "adversary_utility", "baseline_mean_utility"]:
        runs[col] = pd.to_numeric(runs[col], errors="coerce")
    runs = runs.dropna(
        subset=["game_label", "n_agents", "adversary_model", "adversary_elo", "adversary_utility", "baseline_mean_utility"]
    )
    runs["n_agents"] = runs["n_agents"].astype(int)
    runs["adversary_elo"] = runs["adversary_elo"].astype(int)
    runs["adversary_label"] = runs["adversary_model"].map(MODEL_LABELS).fillna(runs["adversary_model"])
    return runs


def aggregate_runs(runs: pd.DataFrame) -> pd.DataFrame:
    return (
        runs.groupby(["game_label", "n_agents", "adversary_model", "adversary_label", "adversary_elo"], dropna=False)
        .agg(
            run_count=("config_id", "count"),
            adversary_utility=("adversary_utility", "mean"),
            adversary_utility_sem=("adversary_utility", sem),
            baseline_mean_utility=("baseline_mean_utility", "mean"),
            baseline_mean_utility_sem=("baseline_mean_utility", sem),
        )
        .reset_index()
        .sort_values(["game_label", "n_agents", "adversary_elo"])
    )


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    ITERATION_OUT.mkdir(parents=True, exist_ok=True)
    (ITERATION_OUT / path.name).write_bytes(path.read_bytes())


def recreate_figure8(agg: pd.DataFrame) -> tuple[Path, pd.DataFrame]:
    slope_rows: list[dict[str, object]] = []
    fig, axes = plt.subplots(1, 3, figsize=(25.5, 8.35), sharey=False)

    for ax, game in zip(axes, GAME_ORDER):
        game_df = agg[agg["game_label"].eq(game)]
        for n in N_ORDER:
            sub = game_df[game_df["n_agents"].eq(n)].sort_values("adversary_elo")
            if sub.empty:
                continue
            color = N_COLORS[n]
            ax.errorbar(
                sub["adversary_elo"],
                sub["adversary_utility"],
                yerr=sub["adversary_utility_sem"].fillna(0.0),
                fmt="o-",
                color=color,
                ecolor=color,
                elinewidth=0.9,
                capsize=2.0,
                lw=1.3,
                ms=4.7,
                alpha=0.17,
            )
            slope, intercept, r2 = linear_fit(sub["adversary_elo"], sub["adversary_utility"])
            if math.isfinite(slope):
                xs = np.linspace(float(sub["adversary_elo"].min()), float(sub["adversary_elo"].max()), 150)
                ax.plot(xs, slope * xs + intercept, color=color, linestyle="--", lw=2.7, alpha=0.92)
            slope_rows.append(
                {
                    "game_label": game,
                    "n_agents": n,
                    "slope_per_elo": slope,
                    "slope_per_100_elo": slope * 100.0 if math.isfinite(slope) else math.nan,
                    "r_squared": r2,
                    "n_points": len(sub),
                }
            )

        ax.set_title(GAME_TITLES[game], fontsize=25, pad=11)
        ax.set_xlabel("Adversary Arena Elo", fontsize=20, labelpad=12)
        ax.set_ylabel("Adversary Payoff", fontsize=22, labelpad=12)
        ax.tick_params(axis="both", labelsize=16)
        ax.grid(True, alpha=0.18, linewidth=0.7)

    legend_handles = [
        Line2D([0], [0], color=N_COLORS[n], linestyle="--", lw=2.7, label=f"N={n}")
        for n in N_ORDER
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=len(N_ORDER),
        fontsize=17,
        frameon=True,
        handlelength=2.4,
        columnspacing=2.1,
    )
    fig.tight_layout(rect=(0, 0.12, 1, 1))

    out_path = OVERLEAF_OUT / "hom_adversary_payoff_vs_elo_by_n_recreated.png"
    save_figure(fig, out_path)
    return out_path, pd.DataFrame(slope_rows)


def plot_baseline_payoff_vs_n(agg: pd.DataFrame) -> Path:
    models = (
        agg[["adversary_model", "adversary_label", "adversary_elo"]]
        .drop_duplicates()
        .sort_values("adversary_elo")
        .reset_index(drop=True)
    )
    cmap = plt.cm.Blues(np.linspace(0.36, 0.92, len(models)))
    color_map = dict(zip(models["adversary_model"], cmap))

    fig, axes = plt.subplots(1, 3, figsize=(18.5, 5.2), sharey=False)
    for ax, game in zip(axes, GAME_ORDER):
        game_df = agg[agg["game_label"].eq(game)]
        for _, model_row in models.iterrows():
            model = model_row["adversary_model"]
            sub = game_df[game_df["adversary_model"].eq(model)].sort_values("n_agents")
            if sub.empty:
                continue
            label = f"{int(model_row['adversary_elo'])} ({model_row['adversary_label']})"
            color = color_map[model]
            ax.errorbar(
                sub["n_agents"],
                sub["baseline_mean_utility"],
                yerr=sub["baseline_mean_utility_sem"].fillna(0.0),
                fmt="o-",
                color=color,
                ecolor=color,
                elinewidth=0.95,
                capsize=2.2,
                lw=2.0,
                ms=4.8,
                alpha=0.92,
                label=label,
            )
        ax.set_title(GAME_TITLES[game], fontsize=17, pad=9)
        ax.set_xlabel("Number of agents (N)", fontsize=13, labelpad=9)
        ax.set_ylabel("Baseline-agent mean payoff", fontsize=13, labelpad=9)
        ax.set_xticks(N_ORDER)
        ax.tick_params(axis="both", labelsize=10)
        ax.grid(True, alpha=0.22, linewidth=0.6)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=3,
        fontsize=10,
        frameon=False,
        title="Adversary Elo",
        title_fontsize=10.5,
    )
    fig.suptitle("Baseline payoff against stronger homogeneous adversaries as N increases", fontsize=18, y=1.03)
    fig.tight_layout(rect=(0, 0.12, 1, 0.98), w_pad=1.3)

    out_path = OVERLEAF_OUT / "hom_baseline_payoff_vs_n_by_adversary_elo.png"
    save_figure(fig, out_path)
    return out_path


def plot_baseline_payoff_vs_elo(agg: pd.DataFrame) -> tuple[Path, pd.DataFrame]:
    slope_rows: list[dict[str, object]] = []
    fig, axes = plt.subplots(1, 3, figsize=(25.5, 8.35), sharey=False)

    for ax, game in zip(axes, GAME_ORDER):
        game_df = agg[agg["game_label"].eq(game)]
        for n in N_ORDER:
            sub = game_df[game_df["n_agents"].eq(n)].sort_values("adversary_elo")
            if sub.empty:
                continue
            color = N_COLORS[n]
            ax.errorbar(
                sub["adversary_elo"],
                sub["baseline_mean_utility"],
                yerr=sub["baseline_mean_utility_sem"].fillna(0.0),
                fmt="o-",
                color=color,
                ecolor=color,
                elinewidth=0.9,
                capsize=2.0,
                lw=1.3,
                ms=4.7,
                alpha=0.22,
            )
            slope, intercept, r2 = linear_fit(sub["adversary_elo"], sub["baseline_mean_utility"])
            if math.isfinite(slope):
                xs = np.linspace(float(sub["adversary_elo"].min()), float(sub["adversary_elo"].max()), 150)
                ax.plot(xs, slope * xs + intercept, color=color, linestyle="--", lw=2.7, alpha=0.92)
            slope_rows.append(
                {
                    "game_label": game,
                    "n_agents": n,
                    "slope_per_elo": slope,
                    "slope_per_100_elo": slope * 100.0 if math.isfinite(slope) else math.nan,
                    "r_squared": r2,
                    "n_points": len(sub),
                }
            )

        ax.set_title(GAME_TITLES[game], fontsize=25, pad=11)
        ax.set_xlabel("Adversary Arena Elo", fontsize=20, labelpad=12)
        ax.set_ylabel("Baseline-agent mean payoff", fontsize=22, labelpad=12)
        ax.tick_params(axis="both", labelsize=16)
        ax.grid(True, alpha=0.18, linewidth=0.7)

    legend_handles = [
        Line2D([0], [0], color=N_COLORS[n], linestyle="--", lw=2.7, label=f"N={n}")
        for n in N_ORDER
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=len(N_ORDER),
        fontsize=17,
        frameon=True,
        handlelength=2.4,
        columnspacing=2.1,
    )
    fig.suptitle("Baseline payoff vs adversary capability in homogeneous-adversary groups", fontsize=25, y=1.02)
    fig.tight_layout(rect=(0, 0.12, 1, 0.98))

    out_path = OVERLEAF_OUT / "hom_baseline_payoff_vs_adversary_elo_by_n.png"
    save_figure(fig, out_path)
    return out_path, pd.DataFrame(slope_rows)


def plot_baseline_payoff_vs_elo_pooled_n(agg: pd.DataFrame) -> tuple[Path, pd.DataFrame]:
    pooled = (
        agg.groupby(["game_label", "adversary_model", "adversary_label", "adversary_elo"], dropna=False)
        .agg(
            cell_count=("baseline_mean_utility", "count"),
            baseline_mean_utility=("baseline_mean_utility", "mean"),
            baseline_mean_utility_sem=("baseline_mean_utility", sem),
        )
        .reset_index()
        .sort_values(["game_label", "adversary_elo"])
    )
    slope_rows: list[dict[str, object]] = []
    fig, axes = plt.subplots(1, 3, figsize=(18.5, 5.2), sharey=False)

    for ax, game in zip(axes, GAME_ORDER):
        sub = pooled[pooled["game_label"].eq(game)].sort_values("adversary_elo")
        ax.errorbar(
            sub["adversary_elo"],
            sub["baseline_mean_utility"],
            yerr=sub["baseline_mean_utility_sem"].fillna(0.0),
            fmt="o",
            color="#1f77b4",
            ecolor="#1f77b4",
            elinewidth=1.1,
            capsize=2.5,
            ms=5.2,
            alpha=0.72,
        )
        slope, intercept, r2 = linear_fit(sub["adversary_elo"], sub["baseline_mean_utility"])
        if math.isfinite(slope):
            xs = np.linspace(float(sub["adversary_elo"].min()), float(sub["adversary_elo"].max()), 150)
            ax.plot(xs, slope * xs + intercept, color="#0b3d75", linestyle="--", lw=2.5, alpha=0.95)
        slope_rows.append(
            {
                "game_label": game,
                "slope_per_elo": slope,
                "slope_per_100_elo": slope * 100.0 if math.isfinite(slope) else math.nan,
                "r_squared": r2,
                "n_points": len(sub),
            }
        )
        ax.set_title(GAME_TITLES[game], fontsize=17, pad=9)
        ax.set_xlabel("Adversary Arena Elo", fontsize=13, labelpad=9)
        ax.set_ylabel("Baseline-agent mean payoff", fontsize=13, labelpad=9)
        ax.tick_params(axis="both", labelsize=10)
        ax.grid(True, alpha=0.22, linewidth=0.6)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    fig.suptitle("Baseline payoff vs adversary capability, averaged over N", fontsize=18, y=1.03)
    fig.tight_layout(rect=(0, 0, 1, 0.98), w_pad=1.3)

    out_path = OVERLEAF_OUT / "hom_baseline_payoff_vs_adversary_elo_pooled_n.png"
    save_figure(fig, out_path)
    return out_path, pd.DataFrame(slope_rows)


def main() -> None:
    OVERLEAF_OUT.mkdir(parents=True, exist_ok=True)
    ITERATION_OUT.mkdir(parents=True, exist_ok=True)

    runs = load_homogeneous_adversary_runs()
    agg = aggregate_runs(runs)
    figure8_path, slopes = recreate_figure8(agg)
    baseline_path = plot_baseline_payoff_vs_n(agg)
    baseline_elo_path, baseline_elo_slopes = plot_baseline_payoff_vs_elo(agg)
    baseline_elo_pooled_path, baseline_elo_pooled_slopes = plot_baseline_payoff_vs_elo_pooled_n(agg)

    for out_dir in [OVERLEAF_OUT, ITERATION_OUT]:
        agg.to_csv(out_dir / "hom_adversary_and_baseline_payoff_by_n_elo.csv", index=False)
        slopes.to_csv(out_dir / "hom_adversary_payoff_vs_elo_by_n_recreated_slopes.csv", index=False)
        baseline_elo_slopes.to_csv(out_dir / "hom_baseline_payoff_vs_adversary_elo_by_n_slopes.csv", index=False)
        baseline_elo_pooled_slopes.to_csv(
            out_dir / "hom_baseline_payoff_vs_adversary_elo_pooled_n_slopes.csv", index=False
        )

    print(f"Wrote {figure8_path}")
    print(f"Wrote {baseline_path}")
    print(f"Wrote {baseline_elo_path}")
    print(f"Wrote {baseline_elo_pooled_path}")
    print(f"Wrote summaries to {OVERLEAF_OUT} and {ITERATION_OUT}")


if __name__ == "__main__":
    main()
