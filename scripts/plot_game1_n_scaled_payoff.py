#!/usr/bin/env python3
"""Plot the requested N/2-scaled Game 1 payoffs and per-agent gap."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(
    "/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/"
    "game1_gpt54_team_coordination_20260809_055844/analysis"
)
INPUT = ROOT / "run_level_outcomes.csv"
OUTPUT_DIR = ROOT / "n_scaled_payoff"
NS = (2, 4, 6, 8, 10)


def summarize(values: pd.Series) -> tuple[float, float, float]:
    clean = values.astype(float).dropna()
    mean = float(clean.mean())
    sem = float(clean.sem())
    half = float(stats.t.ppf(0.975, len(clean) - 1) * sem)
    return mean, mean - half, mean + half


def main() -> None:
    rows = pd.read_csv(INPUT)
    rows = rows[rows["condition"] == "team"].copy()
    if len(rows) != 100:
        raise RuntimeError(f"Expected 100 coordinated-team rows, found {len(rows)}")

    rows["n_over_2_scale"] = rows["n_agents"] / 2.0
    rows["scaled_adversary_payoff"] = (
        rows["adversary_payoff"] * rows["n_over_2_scale"]
    )
    rows["scaled_nano_mean_payoff"] = (
        rows["baseline_mean_payoff"] * rows["n_over_2_scale"]
    )
    rows["scaled_per_agent_gap"] = (
        rows["per_agent_gap"] * rows["n_over_2_scale"]
    )

    summary_rows: list[dict[str, float | int]] = []
    for n_agents in NS:
        group = rows[rows["n_agents"] == n_agents]
        record: dict[str, float | int] = {
            "n_agents": n_agents,
            "team_size": n_agents - 1,
            "scale_factor_n_over_2": n_agents / 2.0,
            "runs": len(group),
        }
        for metric in (
            "scaled_adversary_payoff",
            "scaled_nano_mean_payoff",
            "scaled_per_agent_gap",
        ):
            mean, low, high = summarize(group[metric])
            record[f"{metric}_mean"] = mean
            record[f"{metric}_ci95_low"] = low
            record[f"{metric}_ci95_high"] = high
        summary_rows.append(record)
    summary = pd.DataFrame(summary_rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows.to_csv(OUTPUT_DIR / "n_over_2_scaled_run_outcomes.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "n_over_2_scaled_summary_by_n.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    series = (
        ("scaled_adversary_payoff", "GPT-5.4 adversary", "#C83E36", "o"),
        ("scaled_nano_mean_payoff", "Average GPT-5 Nano", "#2463D4", "s"),
    )
    for metric, label, color, marker in series:
        means = summary[f"{metric}_mean"].to_numpy()
        low = summary[f"{metric}_ci95_low"].to_numpy()
        high = summary[f"{metric}_ci95_high"].to_numpy()
        axes[0].errorbar(
            NS,
            means,
            yerr=np.vstack([means - low, high - means]),
            color=color,
            marker=marker,
            linewidth=2.2,
            capsize=4,
            label=label,
        )
    axes[0].set_title("Payoff multiplied by N/2")
    axes[0].set_ylabel("N/2-scaled discounted payoff")
    axes[0].legend(frameon=False)

    metric = "scaled_per_agent_gap"
    means = summary[f"{metric}_mean"].to_numpy()
    low = summary[f"{metric}_ci95_low"].to_numpy()
    high = summary[f"{metric}_ci95_high"].to_numpy()
    axes[1].errorbar(
        NS,
        means,
        yerr=np.vstack([means - low, high - means]),
        color="#7440E8",
        marker="D",
        linewidth=2.2,
        capsize=4,
    )
    axes[1].axhline(0, color="black", linewidth=1, linestyle="--", alpha=0.7)
    axes[1].set_title("Scaled Nano minus adversary gap")
    axes[1].set_ylabel("N/2 × (Nano mean payoff − GPT-5.4 payoff)")

    labels = [f"{n}\n(×{n/2:g})" for n in NS]
    for axis in axes:
        axis.set_xticks(NS, labels)
        axis.set_xlabel("Total agents N (applied scale factor)")
        axis.grid(axis="y", alpha=0.22)
    fig.suptitle(
        "Requested group-size scaling of coordinated Game 1 outcomes\n"
        "Means and 95% t intervals across 20 runs per N",
        fontsize=15,
    )
    fig.savefig(OUTPUT_DIR / "n_over_2_scaled_team_vs_adversary.png", dpi=220)
    fig.savefig(OUTPUT_DIR / "n_over_2_scaled_team_vs_adversary.pdf")
    plt.close(fig)

    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
