#!/usr/bin/env python3
"""Render the cleaned single-row ICML appendix Figure 17.

The input is the raw-derived primary-run table produced by
``scripts/analyze_n2_baseline_comparison.py``. This renderer intentionally
shows only the Llama-3.3-70B baseline payoff split by exact competition level.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    PROJECT_ROOT
    / "experiments/results/n2_baseline_comparison_analysis_20260505/primary_runs_with_metrics.csv"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "experiments/results/n2_baseline_comparison_analysis_20260505/llama33/"
    / "04_baseline_payoff_by_competition.png"
)

GAME_ORDER = ("game1", "game2", "game3")
GAME_TITLES = {
    "game1": "Game 1: Item Allocation",
    "game2": "Game 2: Diplomatic Treaty",
    "game3": "Game 3: Co-Funding",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def aggregate_llama_baseline(input_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(input_path)
    required = {
        "baseline_key",
        "game_id",
        "competition_value",
        "competition_label",
        "adversary_model",
        "adversary_elo",
        "baseline_utility",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns in {input_path}: {missing}")

    frame = frame[frame["baseline_key"].eq("llama33")].copy()
    if len(frame) != 500:
        raise ValueError(f"Expected 500 Llama-baseline primary runs, found {len(frame)}")

    aggregate = (
        frame.groupby(
            [
                "game_id",
                "competition_value",
                "competition_label",
                "adversary_model",
                "adversary_elo",
            ],
            as_index=False,
        )
        .agg(mean_baseline_payoff=("baseline_utility", "mean"), n=("baseline_utility", "size"))
        .sort_values(["game_id", "competition_value", "adversary_elo"])
    )
    return aggregate


def render(aggregate: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.55), sharey=False)

    for panel_index, (ax, game_id) in enumerate(zip(axes, GAME_ORDER, strict=True)):
        game = aggregate[aggregate["game_id"].eq(game_id)]
        competition_values = sorted(game["competition_value"].dropna().unique())
        colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(competition_values)))

        for color, competition_value in zip(colors, competition_values, strict=True):
            series = game[game["competition_value"].eq(competition_value)].sort_values(
                "adversary_elo"
            )
            label = str(series["competition_label"].iloc[0])

            # Large, muted observations without error bars or connecting lines.
            ax.scatter(
                series["adversary_elo"],
                series["mean_baseline_payoff"],
                s=62,
                color=color,
                alpha=0.34,
                edgecolors="none",
                zorder=2,
            )

            # Prominent trend line. Slopes remain part of the computation but
            # are not printed in the legend.
            x = series["adversary_elo"].to_numpy(dtype=float)
            y = series["mean_baseline_payoff"].to_numpy(dtype=float)
            if len(series) >= 2 and np.unique(x).size >= 2:
                slope, intercept = np.polyfit(x, y, deg=1)
                trend_x = np.linspace(float(x.min()), float(x.max()), 200)
                ax.plot(
                    trend_x,
                    slope * trend_x + intercept,
                    color=color,
                    linestyle="--",
                    linewidth=2.8,
                    alpha=0.98,
                    label=label,
                    zorder=3,
                )

        ax.set_title(GAME_TITLES[game_id], fontsize=18, fontweight="normal", pad=10)
        ax.set_xlabel("Adversary Elo", fontsize=17, fontweight="normal", labelpad=7)
        if panel_index == 0:
            ax.set_ylabel("Mean Baseline Payoff", fontsize=17, fontweight="normal", labelpad=8)
        else:
            ax.set_ylabel("")
        ax.tick_params(axis="both", which="major", labelsize=12.5, width=1.0)
        ax.grid(alpha=0.18, linewidth=0.8)
        ax.margins(x=0.035)

        legend_kwargs = {
            "title": "Competition",
            "fontsize": 12.0,
            "title_fontsize": 13.0,
            "frameon": True,
            "framealpha": 0.94,
            "handlelength": 2.2,
            "borderpad": 0.45,
            "labelspacing": 0.35,
        }
        if game_id == "game1":
            legend = ax.legend(loc="lower left", ncol=2, **legend_kwargs)
        elif game_id == "game2":
            legend = ax.legend(loc="lower left", ncol=1, **legend_kwargs)
        else:
            legend = ax.legend(loc="upper right", ncol=1, **legend_kwargs)
        plt.setp(legend.get_title(), fontweight="normal")
        plt.setp(legend.get_texts(), fontweight="normal")

    # Deliberately no figure-level title: the paper caption supplies the context.
    fig.tight_layout(pad=0.7, w_pad=1.1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    aggregate = aggregate_llama_baseline(input_path)
    render(aggregate, output_path)
    print(f"Wrote {output_path}")
    print(f"Aggregated rows: {len(aggregate)} from 500 primary runs")


if __name__ == "__main__":
    main()
