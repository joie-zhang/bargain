#!/usr/bin/env python3
"""Bridge plots from TTC settings to transcript-language mechanisms."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "analysis/neurips_revision_20260504/ttc_objective_shift"
GRAPHICS_DIR = PROJECT_ROOT / "overleaf/neurips/graphics"

EFFORT_SUMMARY = OUT_DIR / "ttc_objective_shift_effort_summary.csv"
DELTA_CSV = OUT_DIR / "ttc_objective_shift_weak_strong_deltas.csv"

FAMILY_ORDER = ["gpt-5", "claude-sonnet-4-6", "gemini-3-flash"]
FAMILY_LABELS = {
    "gpt-5": "GPT-5",
    "claude-sonnet-4-6": "Claude Sonnet 4.6",
    "gemini-3-flash": "Gemini 3 Flash",
}
FAMILY_COLORS = {
    "gpt-5": "#2563eb",
    "claude-sonnet-4-6": "#dc2626",
    "gemini-3-flash": "#16a34a",
}
EFFORT_LABELS = {
    "minimal": "Minimal",
    "low": "Low",
    "medium": "Medium",
    "high": "High",
    "max": "Max",
}


def style_axis(ax: plt.Axes) -> None:
    ax.grid(True, color="#d1d5db", alpha=0.42, linewidth=0.85)
    ax.tick_params(axis="both", labelsize=10.5)


def plot_requested_effort_bridge(summary: pd.DataFrame) -> Path:
    specs = [
        (
            "passable_minus_self_per_1k_mean",
            "Passable minus self-interest language / 1k words",
            "Objective-shift language",
        ),
        (
            "refusal_language_per_1k_mean",
            "Refusal/infeasibility language / 1k words",
            "Refusal language",
        ),
        (
            "payoff_gini_corrected_mean",
            "Corrected payoff Gini",
            "Inequality outcome",
        ),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.4))

    for ax, (metric, ylabel, title) in zip(axes, specs, strict=True):
        for family in FAMILY_ORDER:
            sub = summary[summary["family"].eq(family)].sort_values("level_index")
            if sub.empty:
                continue
            ax.plot(
                sub["level_index"],
                sub[metric],
                marker="o",
                linewidth=2.2,
                markersize=7,
                color=FAMILY_COLORS[family],
                label=FAMILY_LABELS[family],
            )
            for _, row in sub.iterrows():
                ax.annotate(
                    EFFORT_LABELS[str(row["level"])],
                    (row["level_index"], row[metric]),
                    textcoords="offset points",
                    xytext=(0, 8),
                    ha="center",
                    fontsize=7.6,
                    color=FAMILY_COLORS[family],
                )
        if metric == "passable_minus_self_per_1k_mean":
            ax.axhline(0, color="#94a3b8", linewidth=0.9)
        ax.set_title(title, fontsize=13.5)
        ax.set_xlabel("Requested reasoning effort order")
        ax.set_ylabel(ylabel)
        style_axis(ax)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=True, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle("Does requested TTC effort change the transcript mechanism?", fontsize=17, y=1.02)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    out = GRAPHICS_DIR / "ttc_requested_effort_to_language_bridge.png"
    fig.savefig(out, dpi=240, bbox_inches="tight", pad_inches=0.14)
    plt.close(fig)
    return out


def plot_observed_tokens_bridge(summary: pd.DataFrame) -> Path:
    specs = [
        (
            "passable_minus_self_per_1k_mean",
            "Passable minus self-interest language / 1k words",
            "Objective-shift language",
        ),
        (
            "refusal_language_per_1k_mean",
            "Refusal/infeasibility language / 1k words",
            "Refusal language",
        ),
        (
            "payoff_gini_corrected_mean",
            "Corrected payoff Gini",
            "Inequality outcome",
        ),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.4))

    for ax, (metric, ylabel, title) in zip(axes, specs, strict=True):
        for family in FAMILY_ORDER:
            sub = summary[summary["family"].eq(family)].sort_values("level_index")
            if sub.empty:
                continue
            ax.plot(
                sub["target_tokens_mean"],
                sub[metric],
                marker="o",
                linewidth=2.2,
                markersize=7,
                color=FAMILY_COLORS[family],
                label=FAMILY_LABELS[family],
            )
            for _, row in sub.iterrows():
                ax.annotate(
                    EFFORT_LABELS[str(row["level"])],
                    (row["target_tokens_mean"], row[metric]),
                    textcoords="offset points",
                    xytext=(0, 8),
                    ha="center",
                    fontsize=7.6,
                    color=FAMILY_COLORS[family],
                )
        if metric == "passable_minus_self_per_1k_mean":
            ax.axhline(0, color="#94a3b8", linewidth=0.9)
        ax.set_title(title, fontsize=13.5)
        ax.set_xlabel("Mean observed target tokens/call")
        ax.set_ylabel(ylabel)
        style_axis(ax)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=True, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle("Observed target tokens/call versus transcript mechanism", fontsize=17, y=1.02)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    out = GRAPHICS_DIR / "ttc_observed_tokens_to_language_bridge.png"
    fig.savefig(out, dpi=240, bbox_inches="tight", pad_inches=0.14)
    plt.close(fig)
    return out


def plot_weak_strong_language_delta(deltas: pd.DataFrame) -> Path:
    specs = [
        (
            "delta_passable_minus_self_per_1k",
            "Delta passable minus self-interest / 1k words",
            "Objective shift induced by stronger TTC",
        ),
        (
            "delta_refusal_language_per_1k",
            "Delta refusal/infeasibility language / 1k words",
            "Refusal language induced by stronger TTC",
        ),
        (
            "delta_payoff_gini_corrected",
            "Delta corrected payoff Gini",
            "Inequality change",
        ),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.4))
    x_positions = np.arange(len(FAMILY_ORDER))

    for ax, (metric, ylabel, title) in zip(axes, specs, strict=True):
        for i, family in enumerate(FAMILY_ORDER):
            sub = deltas[deltas["family"].eq(family)]
            if sub.empty:
                continue
            # Deterministic jitter keeps overlapping cells visible.
            jitter = np.linspace(-0.16, 0.16, len(sub)) if len(sub) > 1 else np.array([0.0])
            ax.scatter(
                np.full(len(sub), i) + jitter,
                sub[metric],
                s=36,
                color=FAMILY_COLORS[family],
                alpha=0.58,
                edgecolor="white",
                linewidth=0.45,
            )
            mean = float(sub[metric].mean())
            ax.plot([i - 0.23, i + 0.23], [mean, mean], color="#111827", linewidth=2.4)
            ax.text(i, mean, f" {mean:+.2f}", va="center", ha="left", fontsize=8.5, color="#111827")
        ax.axhline(0, color="#94a3b8", linewidth=0.95)
        ax.set_xticks(x_positions, [FAMILY_LABELS[f] for f in FAMILY_ORDER], rotation=18, ha="right")
        ax.set_title(title, fontsize=13.5)
        ax.set_ylabel(ylabel)
        style_axis(ax)

    fig.suptitle("Matched weak-to-strong TTC changes in language mechanisms", fontsize=17, y=1.02)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = GRAPHICS_DIR / "ttc_weak_strong_language_delta_bridge.png"
    fig.savefig(out, dpi=240, bbox_inches="tight", pad_inches=0.14)
    plt.close(fig)
    return out


def write_summary_tables(summary: pd.DataFrame, deltas: pd.DataFrame) -> None:
    cols = [
        "family",
        "level",
        "target_tokens_mean",
        "passable_minus_self_per_1k_mean",
        "refusal_language_per_1k_mean",
        "payoff_gini_corrected_mean",
        "target_utility_mean",
        "consensus_rate_mean",
        "mean_round_mean",
    ]
    summary[cols].to_csv(OUT_DIR / "ttc_compute_to_language_bridge_effort_summary.csv", index=False)

    delta_cols = [
        "family",
        "delta_tokens",
        "delta_passable_minus_self_per_1k",
        "delta_refusal_language_per_1k",
        "delta_payoff_gini_corrected",
        "delta_target_utility",
        "delta_absolute_payoff_gap",
    ]
    delta_summary = deltas.groupby("family", as_index=False)[delta_cols[1:]].mean()
    delta_summary.to_csv(OUT_DIR / "ttc_compute_to_language_bridge_delta_summary.csv", index=False)


def main() -> None:
    summary = pd.read_csv(EFFORT_SUMMARY)
    deltas = pd.read_csv(DELTA_CSV)
    write_summary_tables(summary, deltas)
    for path in [
        plot_requested_effort_bridge(summary),
        plot_observed_tokens_bridge(summary),
        plot_weak_strong_language_delta(deltas),
    ]:
        print(path)
    print(OUT_DIR / "ttc_compute_to_language_bridge_effort_summary.csv")
    print(OUT_DIR / "ttc_compute_to_language_bridge_delta_summary.csv")


if __name__ == "__main__":
    main()
