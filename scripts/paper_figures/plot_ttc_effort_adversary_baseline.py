#!/usr/bin/env python3
"""
Generate the NeurIPS TTC effort figure with adversary and baseline utilities.

This is the saved generator for:
    overleaf/neurips/graphics/ttc_full/
    overall_by_effort_adversary_baseline_combined.png

The plot summarizes the native test-time-compute scaling run by provider
family and requested reasoning effort. Solid lines show the adversary utility;
dotted lines show the fixed GPT-5-nano baseline utility. Points are means over
matched runs and error bars are SEM.

Usage:
    python scripts/paper_figures/plot_ttc_effort_adversary_baseline.py
    python scripts/paper_figures/plot_ttc_effort_adversary_baseline.py --output /tmp/ttc.png
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_CSV = (
    PROJECT_ROOT
    / "experiments"
    / "results"
    / "ttc_native_scaling_20260502_212943"
    / "monitoring"
    / "partial_results_latest.csv"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "overleaf"
    / "neurips"
    / "graphics"
    / "ttc_full"
    / "overall_by_effort_adversary_baseline_combined.png"
)

FAMILY_ORDER = ["gpt-5", "claude-sonnet-4-6", "gemini-3-flash"]
FAMILY_TITLES = {
    "gpt-5": "GPT-5",
    "claude-sonnet-4-6": "Claude Sonnet 4.6",
    "gemini-3-flash": "Gemini 3 Flash",
}
ROLE_STYLES = {
    "adversary": {
        "label": "Adversary utility",
        "color": "#1f5aa6",
        "linestyle": "-",
        "marker": "o",
        "xoffset": -0.035,
    },
    "baseline": {
        "label": "Baseline utility",
        "color": "#c55a11",
        "linestyle": (0, (1.2, 2.0)),
        "marker": "s",
        "xoffset": 0.035,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-csv",
        type=Path,
        default=DEFAULT_RESULTS_CSV,
        help="TTC partial-results CSV to summarize.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="PNG output path.",
    )
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def load_summary(results_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(results_csv)
    required = {
        "config_id",
        "family",
        "level_index",
        "level",
        "target_utility",
        "baseline_utility",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        missing_cols = ", ".join(missing)
        raise ValueError(f"{results_csv} is missing required columns: {missing_cols}")

    for col in ["level_index", "target_utility", "baseline_utility"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    summary = (
        df.groupby(["family", "level_index", "level"], dropna=False)
        .agg(
            n=("config_id", "count"),
            adversary_mean=("target_utility", "mean"),
            adversary_std=("target_utility", "std"),
            baseline_mean=("baseline_utility", "mean"),
            baseline_std=("baseline_utility", "std"),
        )
        .reset_index()
    )
    for role in ["adversary", "baseline"]:
        summary[f"{role}_sem"] = summary[f"{role}_std"] / summary["n"].pow(0.5)
        summary[f"{role}_sem"] = summary[f"{role}_sem"].fillna(0.0)
    return summary


def draw_plot(summary: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.64), sharey=True)
    all_vals: list[float] = []

    for ax, family in zip(axes, FAMILY_ORDER):
        fam = summary[summary["family"] == family].sort_values(["level_index", "level"])
        if fam.empty:
            ax.set_title(FAMILY_TITLES.get(family, family), fontsize=11, pad=7)
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue

        xs = fam["level_index"].astype(float).to_numpy()
        labels = fam["level"].astype(str).tolist()
        for role, style in ROLE_STYLES.items():
            ys = fam[f"{role}_mean"].astype(float).to_numpy()
            yerr = fam[f"{role}_sem"].astype(float).to_numpy()
            all_vals.extend((ys - yerr).tolist())
            all_vals.extend((ys + yerr).tolist())
            ax.errorbar(
                xs + style["xoffset"],
                ys,
                yerr=yerr,
                fmt=style["marker"],
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=2.0,
                markersize=5.5,
                elinewidth=1.0,
                capsize=3.0,
                capthick=1.0,
                alpha=0.95,
                label=style["label"],
            )

        ax.set_title(FAMILY_TITLES.get(family, family), fontsize=11, pad=7)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_xlabel("Requested reasoning effort", fontsize=10)
        ax.grid(axis="y", alpha=0.28, linewidth=0.8)
        ax.grid(axis="x", alpha=0.10, linewidth=0.6)
        ax.set_xlim(xs.min() - 0.35, xs.max() + 0.35)
        ax.text(
            0.02,
            0.04,
            "n=18 per point",
            transform=ax.transAxes,
            fontsize=8,
            color="#555555",
        )

    axes[0].set_ylabel("Mean discounted utility", fontsize=10)
    finite_vals = [v for v in all_vals if math.isfinite(float(v))]
    if finite_vals:
        lo, hi = min(finite_vals), max(finite_vals)
        pad = max((hi - lo) * 0.12, 2.0)
        axes[0].set_ylim(max(0, lo - pad), hi + pad)

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            linewidth=2.0,
            markersize=5.5,
            label=style["label"],
        )
        for style in ROLE_STYLES.values()
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0.08, 1, 0.97])
    return fig


def main() -> None:
    args = parse_args()
    summary = load_summary(args.results_csv)
    fig = draw_plot(summary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
