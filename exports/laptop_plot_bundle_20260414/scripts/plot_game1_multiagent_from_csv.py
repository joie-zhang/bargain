#!/usr/bin/env python3
"""Replot recent n>2 Game 1 multi-agent figure(s) from summary_by_model.csv."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("Agg")

TITLE_BY_PROPOSAL = {
    "proposal1_invasion": "Proposal 1: Invasion Into Nano Population",
    "proposal2_mixed_ecology": "Proposal 2: Matched Mixed Ecology",
}
OUT_BY_PROPOSAL = {
    "proposal1_invasion": "proposal1_advantage_vs_elo.png",
    "proposal2_mixed_ecology": "proposal2_advantage_vs_elo.png",
}


def competition_palette() -> dict[float, str]:
    return {0.0: "#0f766e", 0.5: "#2563eb", 0.9: "#d97706", 1.0: "#b91c1c"}


def plot_proposal(df: pd.DataFrame, proposal: str, output_path: Path) -> None:
    subset = df[df["proposal_family"] == proposal].copy()
    if subset.empty:
        return

    palette = competition_palette()
    n_values = sorted(subset["n_agents"].unique().tolist())
    fig, axes = plt.subplots(1, len(n_values), figsize=(8 * len(n_values), 5), squeeze=False)

    for ax, n_agents in zip(axes[0], n_values):
        ndf = subset[subset["n_agents"] == n_agents].sort_values("elo")
        for comp in sorted(ndf["competition_level"].unique().tolist()):
            cdf = ndf[ndf["competition_level"] == comp].sort_values("elo")
            ax.plot(
                cdf["elo"],
                cdf["mean_utility_advantage_vs_nano"],
                marker="o",
                linewidth=2,
                color=palette.get(float(comp), "#475569"),
                label=f"comp={comp:g}",
            )
            for _, row in cdf.iterrows():
                ax.annotate(str(row["short_name"]), (float(row["elo"]), float(row["mean_utility_advantage_vs_nano"])), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)

        ax.axhline(0.0, color="#94a3b8", linewidth=1, linestyle="--")
        ax.set_title(f"n = {int(n_agents)}")
        ax.set_xlabel("Chatbot Arena Elo")
        ax.set_ylabel("Mean Utility Advantage vs Nano Control")
        ax.grid(alpha=0.25)
        ax.legend()

    fig.suptitle(TITLE_BY_PROPOSAL.get(proposal, proposal))
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.summary_csv)
    for proposal in ["proposal1_invasion", "proposal2_mixed_ecology"]:
        plot_proposal(df, proposal, args.output_dir / OUT_BY_PROPOSAL[proposal])


if __name__ == "__main__":
    main()
