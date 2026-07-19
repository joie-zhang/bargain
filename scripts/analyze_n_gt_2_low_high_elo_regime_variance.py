#!/usr/bin/env python3
"""Compare payoff variance in low-Elo and high-Elo heterogeneous rosters.

Low-Elo regime: every model in the run is in the bottom 60% of the heterogeneous
model roster by Arena Elo. High-Elo regime: every model is in the top 60%.
Because 60% + 60% overlaps, the script reports both inclusive and exclusive
comparisons.
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
    / "experiments/results/figure_iteration_20260626/low_high_elo_regime_variance"
)


def population_variance(values: pd.Series) -> float:
    arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if arr.size == 0:
        return math.nan
    return float(np.var(arr, ddof=0))


def sem(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) <= 1:
        return 0.0
    return float(clean.std(ddof=1) / math.sqrt(len(clean)))


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    agents = pd.read_csv(AGENT_TABLE)
    agents = agents[agents["experiment_family"].eq("heterogeneous_random")].copy()
    for col in ["n_agents", "elo", "final_utility", "competition_ci"]:
        agents[col] = pd.to_numeric(agents[col], errors="coerce")
    agents = agents.dropna(subset=["run_key", "model", "model_short", "elo", "final_utility", "n_agents", "game_label"])
    agents["n_agents"] = agents["n_agents"].astype(int)
    agents["elo"] = agents["elo"].astype(int)

    models = (
        agents[["model", "model_short", "elo"]]
        .drop_duplicates()
        .sort_values(["elo", "model"])
        .reset_index(drop=True)
    )
    models["rank_1based"] = models.index + 1
    cut_count = math.ceil(0.60 * len(models))
    models["bottom60_model"] = models["rank_1based"].le(cut_count)
    models["top60_model"] = models["rank_1based"].ge(len(models) - cut_count + 1)

    bottom_models = set(models.loc[models["bottom60_model"], "model"])
    top_models = set(models.loc[models["top60_model"], "model"])

    run_metrics = (
        agents.groupby(["run_key", "game_label", "n_agents", "competition_ci", "competition_label_ci"], dropna=False)
        .agg(
            models=("model", lambda values: tuple(sorted(set(values)))),
            model_count=("model", "nunique"),
            mean_roster_elo=("elo", "mean"),
            elo_variance=("elo", population_variance),
            payoff_variance=("final_utility", population_variance),
            mean_payoff=("final_utility", "mean"),
        )
        .reset_index()
    )
    run_metrics["all_bottom60"] = run_metrics["models"].map(lambda values: all(model in bottom_models for model in values))
    run_metrics["all_top60"] = run_metrics["models"].map(lambda values: all(model in top_models for model in values))
    run_metrics["regime_inclusive"] = np.select(
        [
            run_metrics["all_bottom60"] & run_metrics["all_top60"],
            run_metrics["all_bottom60"],
            run_metrics["all_top60"],
        ],
        ["both", "bottom60", "top60"],
        default="neither",
    )
    run_metrics["regime_exclusive"] = np.select(
        [
            run_metrics["all_bottom60"] & ~run_metrics["all_top60"],
            run_metrics["all_top60"] & ~run_metrics["all_bottom60"],
        ],
        ["bottom60", "top60"],
        default="excluded_overlap_or_mixed",
    )
    return models, run_metrics


def make_inclusive_long(values: pd.DataFrame) -> pd.DataFrame:
    frames = [
        values[values["all_bottom60"]].assign(regime="bottom60"),
        values[values["all_top60"]].assign(regime="top60"),
    ]
    return pd.concat(frames, ignore_index=True, sort=False)


def make_exclusive_long(values: pd.DataFrame) -> pd.DataFrame:
    return values[values["regime_exclusive"].isin(["bottom60", "top60"])].rename(
        columns={"regime_exclusive": "regime"}
    )


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby("regime", dropna=False)
        .agg(
            run_count=("run_key", "count"),
            mean_roster_elo=("mean_roster_elo", "mean"),
            mean_roster_elo_median=("mean_roster_elo", "median"),
            elo_variance_mean=("elo_variance", "mean"),
            elo_variance_median=("elo_variance", "median"),
            elo_variance_sem=("elo_variance", sem),
            payoff_variance_mean=("payoff_variance", "mean"),
            payoff_variance_median=("payoff_variance", "median"),
            payoff_variance_sem=("payoff_variance", sem),
            mean_payoff=("mean_payoff", "mean"),
        )
        .reset_index()
    )


def cell_balanced_summary(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cell = (
        frame.groupby(["game_label", "n_agents", "regime"], dropna=False)
        .agg(
            run_count=("run_key", "count"),
            mean_roster_elo=("mean_roster_elo", "mean"),
            elo_variance_mean=("elo_variance", "mean"),
            payoff_variance_mean=("payoff_variance", "mean"),
            mean_payoff=("mean_payoff", "mean"),
        )
        .reset_index()
    )
    balanced = (
        cell.groupby("regime", dropna=False)
        .agg(
            cells=("game_label", "count"),
            run_count=("run_count", "sum"),
            mean_roster_elo=("mean_roster_elo", "mean"),
            elo_variance_mean=("elo_variance_mean", "mean"),
            payoff_variance_mean=("payoff_variance_mean", "mean"),
            mean_payoff=("mean_payoff", "mean"),
        )
        .reset_index()
    )
    return cell, balanced


def add_delta_rows(summary: pd.DataFrame, label: str) -> pd.DataFrame:
    base = summary.set_index("regime")
    rows = []
    if {"bottom60", "top60"}.issubset(base.index):
        bottom = base.loc["bottom60"]
        top = base.loc["top60"]
        rows.append(
            {
                "comparison": label,
                "bottom60_runs": int(bottom["run_count"]),
                "top60_runs": int(top["run_count"]),
                "bottom60_mean_elo_variance": bottom["elo_variance_mean"],
                "top60_mean_elo_variance": top["elo_variance_mean"],
                "elo_variance_diff_top_minus_bottom": top["elo_variance_mean"] - bottom["elo_variance_mean"],
                "elo_variance_pct_diff_vs_bottom": 100.0
                * (top["elo_variance_mean"] - bottom["elo_variance_mean"])
                / bottom["elo_variance_mean"],
                "bottom60_mean_payoff_variance": bottom["payoff_variance_mean"],
                "top60_mean_payoff_variance": top["payoff_variance_mean"],
                "payoff_variance_diff_top_minus_bottom": top["payoff_variance_mean"]
                - bottom["payoff_variance_mean"],
                "payoff_variance_pct_diff_vs_bottom": 100.0
                * (top["payoff_variance_mean"] - bottom["payoff_variance_mean"])
                / bottom["payoff_variance_mean"],
            }
        )
    return pd.DataFrame(rows)


def plot_summary(run_metrics: pd.DataFrame, exclusive_summary: pd.DataFrame) -> Path:
    frame = run_metrics[run_metrics["regime_exclusive"].isin(["bottom60", "top60"])].copy()
    order = ["bottom60", "top60"]
    labels = ["All bottom 60%", "All top 60%"]
    colors = {"bottom60": "#4E79A7", "top60": "#E15759"}
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))

    for ax, metric, title, ylabel in [
        (axes[0], "elo_variance", "Within-roster Elo variance", "Elo variance"),
        (axes[1], "payoff_variance", "Within-run payoff variance", "Payoff variance"),
    ]:
        for x, regime in enumerate(order):
            sub = frame[frame["regime_exclusive"].eq(regime)]
            jitter = np.linspace(-0.12, 0.12, len(sub)) if len(sub) else []
            ax.scatter(
                np.full(len(sub), x) + jitter,
                sub[metric],
                s=8,
                color=colors[regime],
                alpha=0.20,
                linewidths=0,
            )
            row = exclusive_summary[exclusive_summary["regime"].eq(regime)].iloc[0]
            ax.errorbar(
                [x],
                [row[f"{metric}_mean"]],
                yerr=[row[f"{metric}_sem"]],
                fmt="o",
                color="#111111",
                ecolor="#111111",
                ms=6,
                capsize=4,
                lw=1.2,
            )
        ax.set_title(title, fontsize=12, pad=8)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, axis="y", alpha=0.22, linewidth=0.6)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    fig.suptitle("Low- vs high-Elo heterogeneous rosters", fontsize=14, y=1.03)
    fig.tight_layout()
    out_path = OVERLEAF_OUT / "heterogeneous_low_high_elo_regime_variance_comparison.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_outputs(outputs: list[tuple[pd.DataFrame, str]], plot_path: Path) -> None:
    for out_dir in [OVERLEAF_OUT, ITERATION_OUT]:
        out_dir.mkdir(parents=True, exist_ok=True)
        for frame, filename in outputs:
            frame.to_csv(out_dir / filename, index=False)
    (ITERATION_OUT / plot_path.name).write_bytes(plot_path.read_bytes())


def main() -> None:
    OVERLEAF_OUT.mkdir(parents=True, exist_ok=True)
    ITERATION_OUT.mkdir(parents=True, exist_ok=True)

    models, run_metrics = load_data()
    inclusive_long = make_inclusive_long(run_metrics)
    exclusive_long = make_exclusive_long(run_metrics)
    inclusive_summary = summarize(inclusive_long)
    exclusive_summary = summarize(exclusive_long)
    inclusive_cell, inclusive_balanced = cell_balanced_summary(inclusive_long)
    exclusive_cell, exclusive_balanced = cell_balanced_summary(exclusive_long)
    delta_summary = pd.concat(
        [
            add_delta_rows(inclusive_summary, "inclusive_flags"),
            add_delta_rows(exclusive_summary, "exclusive_no_overlap"),
            add_delta_rows(inclusive_balanced, "inclusive_cell_balanced"),
            add_delta_rows(exclusive_balanced, "exclusive_cell_balanced"),
        ],
        ignore_index=True,
    )
    plot_path = plot_summary(run_metrics, exclusive_summary)

    outputs = [
        (models, "heterogeneous_low_high_elo_regime_model_roster.csv"),
        (run_metrics.drop(columns=["models"]), "heterogeneous_low_high_elo_regime_run_metrics.csv"),
        (inclusive_long.drop(columns=["models"]), "heterogeneous_low_high_elo_regime_inclusive_long_run_metrics.csv"),
        (exclusive_long.drop(columns=["models"]), "heterogeneous_low_high_elo_regime_exclusive_long_run_metrics.csv"),
        (inclusive_summary, "heterogeneous_low_high_elo_regime_inclusive_summary.csv"),
        (exclusive_summary, "heterogeneous_low_high_elo_regime_exclusive_summary.csv"),
        (inclusive_cell, "heterogeneous_low_high_elo_regime_inclusive_cell_summary.csv"),
        (exclusive_cell, "heterogeneous_low_high_elo_regime_exclusive_cell_summary.csv"),
        (inclusive_balanced, "heterogeneous_low_high_elo_regime_inclusive_cell_balanced_summary.csv"),
        (exclusive_balanced, "heterogeneous_low_high_elo_regime_exclusive_cell_balanced_summary.csv"),
        (delta_summary, "heterogeneous_low_high_elo_regime_delta_summary.csv"),
    ]
    save_outputs(outputs, plot_path)

    print(f"Wrote {plot_path}")
    print("\nExclusive no-overlap summary:")
    print(exclusive_summary.to_string(index=False))
    print("\nCell-balanced exclusive summary:")
    print(exclusive_balanced.to_string(index=False))
    print("\nDelta summary:")
    print(delta_summary.to_string(index=False))


if __name__ == "__main__":
    main()
