#!/usr/bin/env python3
"""Create one clean average target-utility plot for the TTC seed study."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


FAMILIES = [
    ("gpt-5", "GPT-5", "#2563eb"),
    ("claude-sonnet-4-6", "Claude Sonnet 4.6", "#7c3aed"),
    ("gemini-3-flash", "Gemini 3 Flash", "#059669"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ci_csv", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--output-stem",
        default="target_utility_average_10seeds_partial",
        help="Output filename without an extension.",
    )
    parser.add_argument("--y-min", type=float)
    parser.add_argument("--y-max", type=float)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data = pd.read_csv(args.ci_csv)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), sharey=True)
    lower_values = []
    upper_values = []

    for ax, (family, label, color) in zip(axes, FAMILIES):
        subset = data[data["family"].eq(family)].sort_values("level_index")
        x = subset["level_index"].to_numpy(dtype=float)
        mean = subset["target_utility_mean"].to_numpy(dtype=float)
        low = subset["target_utility_seed_ci95_low"].to_numpy(dtype=float)
        high = subset["target_utility_seed_ci95_high"].to_numpy(dtype=float)

        lower_values.extend(low.tolist())
        upper_values.extend(high.tolist())
        ax.fill_between(x, low, high, color=color, alpha=0.16, linewidth=0)
        ax.plot(
            x,
            mean,
            color=color,
            marker="o",
            markersize=7,
            linewidth=2.8,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(subset["level"].astype(str), fontsize=11)
        ax.set_title(label, fontsize=15, pad=12)
        ax.set_xlabel("Reasoning effort", fontsize=12)
        ax.grid(axis="y", color="#d1d5db", alpha=0.6, linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        incomplete = subset[subset["seed_count"].lt(10)]
        for _, point in incomplete.iterrows():
            ax.annotate(
                f"n={int(point['seed_count'])}",
                (
                    float(point["level_index"]),
                    float(point["target_utility_seed_ci95_high"]),
                ),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=9,
                color="#4b5563",
            )

    if (args.y_min is None) != (args.y_max is None):
        raise ValueError("--y-min and --y-max must be provided together")
    if args.y_min is not None and args.y_max is not None:
        axes[0].set_ylim(args.y_min, args.y_max)
    else:
        y_min = min(lower_values)
        y_max = max(upper_values)
        padding = max(1.0, 0.10 * (y_max - y_min))
        axes[0].set_ylim(y_min - padding, y_max + padding)
    axes[0].set_ylabel("Mean target utility", fontsize=12)

    fig.suptitle(
        "Average target utility across random seeds",
        fontsize=18,
        fontweight="semibold",
        y=1.02,
    )
    fig.text(
        0.5,
        -0.01,
        "Line = across-seed mean; shaded region = 95% confidence interval. "
        "Claude max uses 8 available seeds; all other points use 10.",
        ha="center",
        fontsize=10,
        color="#4b5563",
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])

    stem = output_dir / args.output_stem
    fig.savefig(stem.with_suffix(".png"), dpi=500, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)

    print(stem)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
