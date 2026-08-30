#!/usr/bin/env python3
"""Plot ceiling-normalized role payoffs for both matched 100-run setups."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


ANALYSIS_DIR = Path(
    "/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/"
    "game1_gpt54_team_coordination_20260809_055844/analysis"
)
INPUT = ANALYSIS_DIR / "ceiling_normalization/condition_normalized_outcomes.csv"
OUTPUT_DIR = ANALYSIS_DIR / "ceiling_normalization/role_plots"
NS = (2, 4, 6, 8, 10)
SETUPS = (
    ("team", "Coordinated Nano-team treatment"),
    ("control", "Homogeneous-adversary control"),
)
COLORS = {"adversary": "#C83E36", "baseline": "#2463D4", "gap": "#7440E8"}


def summarize(values: pd.Series) -> dict[str, float | int]:
    clean = values.astype(float).dropna()
    mean = float(clean.mean())
    sd = float(clean.std(ddof=1))
    sem = sd / np.sqrt(len(clean))
    half = float(stats.t.ppf(0.975, len(clean) - 1) * sem)
    return {
        "runs": len(clean),
        "mean": mean,
        "ci95_low": mean - half,
        "ci95_high": mean + half,
    }


def add_points_and_mean(
    axis: plt.Axes,
    rows: pd.DataFrame,
    metric: str,
    *,
    color: str,
    marker: str,
    label: str | None,
    rng: np.random.Generator,
    jitter_offset: float = 0.0,
) -> None:
    means: list[float] = []
    lows: list[float] = []
    highs: list[float] = []
    for n_agents in NS:
        values = rows.loc[rows["n_agents"] == n_agents, metric].astype(float)
        jitter = rng.uniform(-0.10, 0.10, len(values)) + jitter_offset
        axis.scatter(
            n_agents + jitter,
            values,
            s=22,
            color=color,
            alpha=0.18,
            linewidths=0,
        )
        result = summarize(values)
        means.append(float(result["mean"]))
        lows.append(float(result["ci95_low"]))
        highs.append(float(result["ci95_high"]))
    means_array = np.asarray(means)
    axis.errorbar(
        NS,
        means_array,
        yerr=np.vstack([means_array - np.asarray(lows), np.asarray(highs) - means_array]),
        color=color,
        marker=marker,
        linewidth=2.2,
        capsize=4,
        label=label,
    )


def format_axis(axis: plt.Axes) -> None:
    axis.set_xticks(NS, [f"{n}\n({n-1})" for n in NS])
    axis.set_xlabel("Total agents N (Nano agents = N−1)")
    axis.grid(axis="y", alpha=0.22)


def plot_role_panel(axis: plt.Axes, rows: pd.DataFrame, title: str, seed: int) -> None:
    rng = np.random.default_rng(seed)
    add_points_and_mean(
        axis,
        rows,
        "raw_adversary_ceiling_multiple",
        color=COLORS["adversary"],
        marker="o",
        label="GPT-5.4 adversary",
        rng=rng,
        jitter_offset=-0.04,
    )
    add_points_and_mean(
        axis,
        rows,
        "raw_baseline_mean_ceiling_multiple",
        color=COLORS["baseline"],
        marker="s",
        label="Average GPT-5 Nano",
        rng=rng,
        jitter_offset=0.04,
    )
    axis.axhline(1, color="black", linewidth=1, linestyle="--", alpha=0.65)
    axis.set_title(title)
    axis.set_ylabel("Raw payoff / maximum feasible average payoff")
    axis.legend(frameon=False)
    format_axis(axis)


def plot_gap_panel(axis: plt.Axes, rows: pd.DataFrame, title: str, seed: int) -> None:
    add_points_and_mean(
        axis,
        rows,
        "raw_adversary_minus_baseline_normalized_gap",
        color=COLORS["gap"],
        marker="D",
        label=None,
        rng=np.random.default_rng(seed),
    )
    axis.axhline(0, color="black", linewidth=1, linestyle="--", alpha=0.7)
    axis.set_title(title)
    axis.set_ylabel("Normalized GPT-5.4 − Nano mean gap")
    format_axis(axis)


def main() -> None:
    all_rows = pd.read_csv(INPUT)
    selected = all_rows[all_rows["condition"].isin(["team", "control"])].copy()
    selected["raw_adversary_minus_baseline_normalized_gap"] = -selected[
        "raw_ceiling_normalized_gap"
    ]
    counts = selected.groupby(["condition", "n_agents"]).size()
    if len(selected) != 200 or not (counts == 20).all():
        raise RuntimeError(f"Expected 20 runs in each setup × N cell, found:\n{counts}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, float | int | str]] = []
    for condition, setup_title in SETUPS:
        rows = selected[selected["condition"] == condition]
        for n_agents in NS:
            cell = rows[rows["n_agents"] == n_agents]
            for metric in (
                "raw_adversary_ceiling_multiple",
                "raw_baseline_mean_ceiling_multiple",
                "raw_adversary_minus_baseline_normalized_gap",
            ):
                summary_rows.append(
                    {
                        "setup": condition,
                        "setup_label": setup_title,
                        "n_agents": n_agents,
                        "team_size": n_agents - 1,
                        "metric": metric,
                        **summarize(cell[metric]),
                    }
                )
    pd.DataFrame(summary_rows).to_csv(
        OUTPUT_DIR / "ceiling_normalized_role_summary.csv", index=False
    )

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for row_index, (condition, setup_title) in enumerate(SETUPS):
        rows = selected[selected["condition"] == condition]
        plot_role_panel(
            axes[row_index, 0], rows, f"{setup_title}: role payoffs", 20260816 + row_index
        )
        plot_gap_panel(
            axes[row_index, 1], rows, f"{setup_title}: role gap", 20260826 + row_index
        )
    fig.suptitle(
        "Game 1 payoff relative to each run's maximum feasible average\n"
        "Means and 95% t intervals across 20 runs per N",
        fontsize=15,
    )
    fig.savefig(OUTPUT_DIR / "ceiling_normalized_roles_both_setups.png", dpi=220)
    fig.savefig(OUTPUT_DIR / "ceiling_normalized_roles_both_setups.pdf")
    plt.close(fig)

    for index, (condition, setup_title) in enumerate(SETUPS):
        rows = selected[selected["condition"] == condition]

        fig, axis = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)
        plot_role_panel(axis, rows, setup_title, 20260836 + index)
        fig.savefig(OUTPUT_DIR / f"{condition}_normalized_role_payoffs.png", dpi=220)
        fig.savefig(OUTPUT_DIR / f"{condition}_normalized_role_payoffs.pdf")
        plt.close(fig)

        fig, axis = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)
        plot_gap_panel(axis, rows, setup_title, 20260846 + index)
        fig.savefig(OUTPUT_DIR / f"{condition}_normalized_role_gap.png", dpi=220)
        fig.savefig(OUTPUT_DIR / f"{condition}_normalized_role_gap.pdf")
        plt.close(fig)

    print(pd.DataFrame(summary_rows).to_string(index=False))


if __name__ == "__main__":
    main()
