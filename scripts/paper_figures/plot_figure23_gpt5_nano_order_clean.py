#!/usr/bin/env python3
"""Render the cleaned single-row ICML appendix Figure 23.

The input is the raw-derived primary-run table produced by
``scripts/analyze_n2_baseline_comparison.py``. This renderer keeps only the
GPT-5-nano baseline and compares adversary payoff when the adversary moves
first versus when the baseline moves first.
"""

from __future__ import annotations

import argparse
import shutil
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
    / "experiments/results/n2_baseline_comparison_analysis_20260505/gpt5_nano/"
    / "05_adversary_payoff_by_order_clean.png"
)
PAPER_OUTPUT = (
    PROJECT_ROOT
    / "overleaf/icml_aiwild_template/graphics/appendix/"
    / "bilateral_order_diagnostics_gpt5_nano_order_1x3.png"
)

GAME_ORDER = ("game1", "game2", "game3")
GAME_TITLES = {
    "game1": "Game 1: Item Allocation",
    "game2": "Game 2: Diplomatic Treaty",
    "game3": "Game 3: Co-funding",
}
ORDER_STYLES = {
    "adversary_first": ("Adversary first", "#b45309"),
    "baseline_first": ("Baseline first", "#2563eb"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--paper-output",
        type=Path,
        default=PAPER_OUTPUT,
        help="Paper asset path that receives a synchronized copy of the rendered PNG.",
    )
    return parser.parse_args()


def aggregate_gpt5_nano_order(input_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(input_path)
    required = {
        "baseline_key",
        "game_id",
        "conceptual_order",
        "adversary_model",
        "adversary_elo",
        "adversary_utility",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns in {input_path}: {missing}")

    frame = frame[frame["baseline_key"].eq("gpt5_nano")].copy()
    if len(frame) != 1500:
        raise ValueError(f"Expected 1,500 GPT-5-nano primary runs, found {len(frame)}")
    observed_orders = set(frame["conceptual_order"].dropna().unique())
    if observed_orders != set(ORDER_STYLES):
        raise ValueError(f"Unexpected model-order labels: {sorted(observed_orders)}")

    aggregate = (
        frame.groupby(
            ["game_id", "conceptual_order", "adversary_model", "adversary_elo"],
            as_index=False,
        )
        .agg(
            mean_adversary_payoff=("adversary_utility", "mean"),
            n=("adversary_utility", "size"),
        )
        .sort_values(["game_id", "conceptual_order", "adversary_elo"])
    )
    return aggregate


def render(aggregate: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.45), sharey=False)

    for panel_index, (ax, game_id) in enumerate(zip(axes, GAME_ORDER, strict=True)):
        game = aggregate[aggregate["game_id"].eq(game_id)]

        for order, (label, color) in ORDER_STYLES.items():
            series = game[game["conceptual_order"].eq(order)].sort_values("adversary_elo")
            x = series["adversary_elo"].to_numpy(dtype=float)
            y = series["mean_adversary_payoff"].to_numpy(dtype=float)

            # Muted model-level means, with no error bars and no lines joining dots.
            ax.scatter(
                x,
                y,
                s=66,
                color=color,
                alpha=0.36,
                edgecolors="none",
                zorder=2,
            )

            # A prominent linear trend; slope values are intentionally omitted
            # from the legend and all visible annotations.
            if len(series) >= 2 and np.unique(x).size >= 2:
                slope, intercept = np.polyfit(x, y, deg=1)
                trend_x = np.linspace(float(x.min()), float(x.max()), 200)
                ax.plot(
                    trend_x,
                    slope * trend_x + intercept,
                    color=color,
                    linestyle="--",
                    linewidth=3.0,
                    alpha=0.98,
                    label=label,
                    zorder=3,
                )

        ax.set_title(GAME_TITLES[game_id], fontsize=16, fontweight="normal", pad=10)
        ax.set_xlabel("Adversary Elo", fontsize=15, fontweight="normal", labelpad=7)
        if panel_index == 0:
            ax.set_ylabel(
                "Mean Adversary Payoff",
                fontsize=15,
                fontweight="normal",
                labelpad=8,
            )
        else:
            ax.set_ylabel("")
        ax.tick_params(axis="both", which="major", labelsize=12.5, width=1.0)
        ax.grid(alpha=0.18, linewidth=0.8)
        ax.margins(x=0.035)

        legend = ax.legend(
            title="Order",
            fontsize=12.5,
            title_fontsize=13.0,
            frameon=True,
            framealpha=0.94,
            handlelength=2.25,
            borderpad=0.5,
            labelspacing=0.4,
            loc="best",
        )
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
    paper_output = args.paper_output.resolve()
    aggregate = aggregate_gpt5_nano_order(input_path)
    render(aggregate, output_path)
    paper_output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(output_path, paper_output)
    print(f"Wrote {output_path}")
    print(f"Wrote {paper_output}")
    print(f"Aggregated rows: {len(aggregate)} from 1,500 primary runs")


if __name__ == "__main__":
    main()
