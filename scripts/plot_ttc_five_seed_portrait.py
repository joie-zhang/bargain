#!/usr/bin/env python3
"""Render the five-seed TTC target-payoff plot in a tight portrait layout."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ANALYSIS_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "results"
    / "ttc_native_scaling_seed1024_20260725_211500"
    / "analysis"
    / "seeds42_984_526_423_1024"
)
DEFAULT_OUTPUT_NAME = "target_payoff_across_seed_mean_1x3_tall_black_ylim.png"

FAMILY_ORDER = ["gpt-5", "claude-sonnet-4-6", "gemini-3-flash"]
FAMILY_LABELS = {
    "gpt-5": "GPT-5",
    "claude-sonnet-4-6": "Claude Sonnet 4.6",
    "gemini-3-flash": "Gemini 3 Flash",
}
SEEDS = [42, 984, 526, 423, 1024]
SEED_COLORS = {
    42: "#64748b",
    984: "#2563eb",
    526: "#dc2626",
    423: "#059669",
    1024: "#9333ea",
}
SEED_MARKERS = {42: "s", 984: "o", 526: "^", 423: "D", 1024: "P"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=DEFAULT_ANALYSIS_DIR,
        help="Directory containing the five-seed summary CSVs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=f"Output PNG path (default: ANALYSIS_DIR/{DEFAULT_OUTPUT_NAME}).",
    )
    parser.add_argument(
        "--padding-fraction",
        type=float,
        default=0.045,
        help="Fractional y-margin beyond each panel's visible min/max.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    analysis_dir = args.analysis_dir.resolve()
    output = (
        args.output.resolve()
        if args.output
        else analysis_dir / DEFAULT_OUTPUT_NAME
    )
    if args.padding_fraction < 0:
        raise ValueError("--padding-fraction must be nonnegative")

    by_seed = pd.read_csv(
        analysis_dir / "family_effort_by_seed_all_five.csv"
    )
    seed_ci = pd.read_csv(
        analysis_dir / "family_effort_across_seed_ci95_all_five.csv"
    )

    # Keep the three families side-by-side, but make each panel deliberately
    # narrow and tall so within-family variation is visually legible.
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 18.0), sharex=False)
    for panel_index, (ax, family) in enumerate(zip(axes, FAMILY_ORDER)):
        family_data = by_seed[by_seed["family"].eq(family)]

        for seed in SEEDS:
            subset = family_data[family_data["seed"].eq(seed)].sort_values(
                "level_index"
            )
            if len(subset) != 4:
                raise RuntimeError(
                    f"Expected four effort levels for {family}/seed {seed}, "
                    f"found {len(subset)}"
                )
            x = subset["level_index"].to_numpy(dtype=float)
            y = subset["target_utility_mean"].to_numpy(dtype=float)
            ax.plot(
                x,
                y,
                marker=SEED_MARKERS[seed],
                markersize=5.0,
                linewidth=1.35,
                alpha=0.38,
                color=SEED_COLORS[seed],
                label=f"Seed {seed}",
            )

        summary = seed_ci[seed_ci["family"].eq(family)].sort_values(
            "level_index"
        )
        if len(summary) != 4:
            raise RuntimeError(
                f"Expected four CI rows for {family}, found {len(summary)}"
            )
        x = summary["level_index"].to_numpy(dtype=float)
        mean = summary["target_utility_mean"].to_numpy(dtype=float)
        ax.plot(
            x,
            mean,
            marker="o",
            markersize=7.5,
            linewidth=3.0,
            color="#111827",
            label="Across-seed mean",
            zorder=10,
        )

        # Deliberately frame each panel around the black across-seed mean.
        # Colored individual-seed trajectories remain visible where they fall
        # inside this range and may be clipped outside it.
        data_min = float(np.min(mean))
        data_max = float(np.max(mean))
        data_span = data_max - data_min
        margin = max(data_span * args.padding_fraction, 0.30)
        ax.set_ylim(data_min - margin, data_max + margin)

        labels = summary.set_index("level_index")["level"]
        ax.set_xticks(labels.index.astype(float))
        ax.set_xticklabels(labels.astype(str))
        ax.set_title(
            f"({chr(97 + panel_index)}) {FAMILY_LABELS[family]}",
            pad=12,
            fontsize=15,
        )
        ax.set_xlabel("Requested reasoning effort")
        ax.set_ylabel("Mean target payoff")
        ax.grid(axis="y", alpha=0.28)

    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.018),
    )
    fig.suptitle(
        "Test-time compute across five seeds\n"
        "Individual seeds and across-seed mean",
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0.065, 1, 0.975], w_pad=2.6)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    print(output)
    print(output.with_suffix(".pdf"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
