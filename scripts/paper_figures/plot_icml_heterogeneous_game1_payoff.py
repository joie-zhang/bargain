#!/usr/bin/env python3
"""Render the ICML main-text heterogeneous Game 1 payoff panel."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_n2_plus_multiagent_comparison import (  # noqa: E402
    N_COLORS,
    N_ORDER,
    add_fit_line,
    aggregate_het_agents,
)


DATA_PATH = (
    PROJECT_ROOT
    / "experiments/results/n2_plus_multiagent_comparison_analysis_20260505"
    / "tables_multiagent/heterogeneous_agents_fresh.csv"
)
MANIFEST_PATH = PROJECT_ROOT / "docs/reproducibility/paper_experiment_data_manifest.csv"
OUT_PATH = (
    PROJECT_ROOT
    / "overleaf/icml_aiwild_template/graphics/qualitative_ttc"
    / "heterogenous_game1_payoff_singlecolumn.png"
)


def normalize_path(value: str) -> str:
    path = Path(value)
    if not path.is_absolute():
        return str(path)
    return str(path.resolve().relative_to(PROJECT_ROOT))


def load_validated_agents() -> pd.DataFrame:
    agents = pd.read_csv(DATA_PATH)
    agents = agents[agents["experiment_family"].eq("heterogeneous_random")].copy()
    manifest = pd.read_csv(MANIFEST_PATH)
    expected_paths = set(
        manifest.loc[
            manifest["experiment_family"].eq("heterogeneous_random"), "result_path"
        ].astype(str)
    )
    actual_paths = set(agents["result_path"].astype(str).map(normalize_path))
    if len(expected_paths) != 1300 or actual_paths != expected_paths:
        raise RuntimeError(
            "The heterogeneous agent table does not match the 1,300 canonical result paths"
        )
    if agents["model"].astype(str).str.contains("phi", case=False).any():
        raise RuntimeError("Phi rows remain in the heterogeneous agent table")
    if agents["result_path"].nunique() != 1300:
        raise RuntimeError("Expected 1,300 unique heterogeneous runs")
    return agents


def main() -> None:
    agents = load_validated_agents()
    aggregate = aggregate_het_agents(agents, by_competition=False)
    game = aggregate[aggregate["game_label"].eq("game1")].copy()
    if set(pd.to_numeric(game["n_agents"]).astype(int)) != set(N_ORDER):
        raise RuntimeError("The Game 1 panel is missing one or more group sizes")

    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(figsize=(5.6, 4.35))
    for n_agents in N_ORDER:
        rows = game[game["n_agents"].eq(n_agents)].sort_values("elo")
        if rows.empty:
            raise RuntimeError(f"No Game 1 rows for N={n_agents}")
        color = N_COLORS[n_agents]
        ax.plot(
            rows["elo"],
            rows["final_utility"],
            linestyle="none",
            marker="o",
            markersize=7.0,
            color=color,
            label=f"n={n_agents}",
            alpha=0.72,
        )
        add_fit_line(ax, rows, "elo", "final_utility", color, linewidth=2.5, alpha=0.95)

    ax.set_xlabel("Arena Elo", fontsize=20, labelpad=7)
    ax.set_ylabel("Mean model payoff", fontsize=20, labelpad=8)
    ax.tick_params(axis="both", labelsize=16, width=1.45, length=5.0)
    ax.grid(True, color="#D1D5DB", alpha=0.52, linewidth=0.75)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.45)
    ax.spines["bottom"].set_linewidth(1.45)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        fontsize=15,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=len(N_ORDER),
        handletextpad=0.25,
        columnspacing=0.75,
    )
    fig.subplots_adjust(left=0.17, right=0.985, bottom=0.27, top=0.985)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_options = {"bbox_inches": "tight", "pad_inches": 0.02, "facecolor": "white"}
    fig.savefig(OUT_PATH, dpi=320, **save_options)
    fig.savefig(OUT_PATH.with_suffix(".pdf"), **save_options)
    plt.close(fig)
    print(OUT_PATH)


if __name__ == "__main__":
    main()
