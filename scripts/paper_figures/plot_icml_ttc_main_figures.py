#!/usr/bin/env python3
"""Regenerate ICML TTC main-text figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ICML_GRAPHICS = PROJECT_ROOT / "overleaf/icml_aiwild_template/graphics"
TTC_SUMMARY_CSV = PROJECT_ROOT / "analysis/neurips_revision_20260504/ttc_game_averaged_by_effort.csv"
TTC_INTENSITY_CSV = (
    PROJECT_ROOT
    / "analysis/ttc_group_intensity_turn_dedup_verification_20260701/ttc_group_intensity_turn_dedup_summary.csv"
)

FAMILY_ORDER = ["gpt-5", "claude-sonnet-4-6", "gemini-3-flash"]
FAMILY_LABELS = {
    "gpt-5": "GPT-5",
    "claude-sonnet-4-6": "Claude Sonnet 4.6",
    "gemini-3-flash": "Gemini 3 Flash",
}
EFFORT_ORDER = ["minimal", "low", "medium", "high", "max"]
EFFORT_LABELS = {
    "minimal": "Minimal",
    "low": "Low",
    "medium": "Medium",
    "high": "High",
    "max": "Max",
}
EFFORT_SHORT_LABELS = {
    "minimal": "min",
    "low": "low",
    "medium": "med",
    "high": "high",
    "max": "max",
}
EFFORT_COLORS = {
    "minimal": "#64748b",
    "low": "#2563eb",
    "medium": "#0f766e",
    "high": "#f97316",
    "max": "#7c3aed",
}
FAMILY_COLORS = {
    "gemini-3-flash": "#1f77b4",
    "gpt-5": "#d62728",
}
FOCUS_FAMILIES = ["gemini-3-flash", "gpt-5"]
FOCUS_GROUPS = [
    "emotional persuasion",
    "trade/compromise",
    "logical persuasion",
    "pressure",
    "self-interest/exploitation",
    "formalization",
]
GROUP_DISPLAY = {
    "emotional persuasion": "Emotional Persuasion",
    "trade/compromise": "Trade/Compromise",
    "logical persuasion": "Logical Persuasion",
    "pressure": "Pressure",
    "self-interest/exploitation": "Self-Interest/Exploitation",
    "formalization": "Formalization",
}


def plot_ttc_payoff_bars() -> Path:
    summary = pd.read_csv(TTC_SUMMARY_CSV)
    summary = summary[summary["family"].isin(FAMILY_ORDER)].copy()
    summary["level"] = pd.Categorical(summary["level"], EFFORT_ORDER, ordered=True)
    summary = summary.sort_values(["family", "level"])

    fig, axes = plt.subplots(1, 3, figsize=(14.6, 4.9), sharey=True)
    fig.patch.set_facecolor("white")

    for ax, family in zip(axes, FAMILY_ORDER, strict=True):
        family_df = summary[summary["family"].eq(family)].sort_values("level")
        x = np.arange(len(family_df))
        colors = [EFFORT_COLORS[str(level)] for level in family_df["level"]]
        ax.bar(
            x,
            family_df["target_utility_mean"],
            yerr=family_df["target_utility_sem"],
            color=colors,
            edgecolor="white",
            linewidth=1.0,
            error_kw={"elinewidth": 1.45, "capsize": 4.0, "capthick": 1.45, "ecolor": "#334155"},
            width=0.72,
            zorder=3,
        )
        ax.set_title(FAMILY_LABELS[family], fontsize=24, pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels([EFFORT_LABELS[str(level)] for level in family_df["level"]], rotation=0)
        ax.tick_params(axis="both", labelsize=14.5)
        ax.grid(True, axis="y", color="#d1d5db", alpha=0.45, linewidth=0.85, zorder=0)
        ax.set_ylim(48, 82)

    axes[0].set_ylabel("Mean Target Payoff", fontsize=20, labelpad=10)
    fig.supxlabel("Requested Reasoning Effort", fontsize=20, y=0.12)
    handles = [
        Line2D([0], [0], color=EFFORT_COLORS[level], marker="s", linestyle="", markersize=11, label=EFFORT_LABELS[level])
        for level in EFFORT_ORDER
    ]
    legend = fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.045),
        ncol=len(handles),
        title="Reasoning Effort",
        title_fontsize=13.5,
        fontsize=13.2,
        frameon=True,
        facecolor="white",
        framealpha=0.94,
        columnspacing=1.0,
        handletextpad=0.45,
        borderpad=0.45,
    )
    legend.get_frame().set_edgecolor("#d1d5db")
    fig.subplots_adjust(left=0.08, right=0.995, top=0.84, bottom=0.30, wspace=0.14)

    out_path = ICML_GRAPHICS / "ttc_game_averaged_target_payoff_vs_compute.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=260, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    return out_path


def plot_ttc_intensity_grid() -> tuple[Path, Path]:
    summary = pd.read_csv(TTC_INTENSITY_CSV)
    summary = summary[
        summary["family"].isin(FOCUS_FAMILIES)
        & summary["category"].isin(FOCUS_GROUPS)
        & summary["level"].isin(["minimal", "low", "medium", "high"])
    ].copy()
    summary["level"] = pd.Categorical(summary["level"], ["minimal", "low", "medium", "high"], ordered=True)
    summary["category"] = pd.Categorical(summary["category"], FOCUS_GROUPS, ordered=True)
    summary = summary.sort_values(["category", "family", "level"])

    fig, axes = plt.subplots(3, 2, figsize=(6.6, 8.65), sharex=True)
    axes = axes.ravel()

    for ax, group in zip(axes, FOCUS_GROUPS, strict=True):
        group_df = summary[summary["category"].eq(group)]
        for family in FOCUS_FAMILIES:
            sub = group_df[group_df["family"].eq(family)].sort_values("level")
            x = np.arange(len(sub))
            ax.plot(
                x,
                sub["unique_turn_events_per_rollout"],
                marker="o",
                linewidth=2.25,
                markersize=6.8,
                color=FAMILY_COLORS[family],
                label=FAMILY_LABELS[family],
            )
        ax.set_title(GROUP_DISPLAY[group], fontsize=17, pad=4)
        ax.set_xticks(np.arange(4))
        ax.set_xticklabels([EFFORT_SHORT_LABELS[level] for level in ["minimal", "low", "medium", "high"]], fontsize=13)
        ax.tick_params(axis="y", labelsize=13)
        ax.grid(True, color="#d1d5db", alpha=0.48, linewidth=0.8)

    fig.supylabel("Average Occurrences", fontsize=23, x=0.02)
    fig.supxlabel("Reasoning Effort", fontsize=22, y=0.055)
    handles = [
        Line2D([0], [0], color=FAMILY_COLORS[family], marker="o", linewidth=2.25, markersize=6.8, label=FAMILY_LABELS[family])
        for family in FOCUS_FAMILIES
    ]
    legend = fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.012),
        ncol=2,
        fontsize=14,
        frameon=True,
        facecolor="white",
        framealpha=0.95,
        borderpad=0.35,
    )
    legend.get_frame().set_edgecolor("#d1d5db")
    fig.tight_layout(rect=[0.06, 0.09, 1.0, 1.0], h_pad=1.0, w_pad=1.0)

    out_png = ICML_GRAPHICS / "qualitative_ttc/ttc_group_intensity_singlecolumn_3x2.png"
    out_pdf = ICML_GRAPHICS / "qualitative_ttc/ttc_group_intensity_singlecolumn_3x2.pdf"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return out_png, out_pdf


def plot_ttc_intensity_grid_fullwidth_2x3_compact() -> tuple[Path, Path]:
    summary = pd.read_csv(TTC_INTENSITY_CSV)
    summary = summary[
        summary["family"].isin(FOCUS_FAMILIES)
        & summary["category"].isin(FOCUS_GROUPS)
        & summary["level"].isin(["minimal", "low", "medium", "high"])
    ].copy()
    summary["level"] = pd.Categorical(summary["level"], ["minimal", "low", "medium", "high"], ordered=True)
    summary["category"] = pd.Categorical(summary["category"], FOCUS_GROUPS, ordered=True)
    summary = summary.sort_values(["category", "family", "level"])

    fig, axes = plt.subplots(2, 3, figsize=(11.1, 5.9), sharex=True)
    axes = axes.ravel()

    for ax, group in zip(axes, FOCUS_GROUPS, strict=True):
        group_df = summary[summary["category"].eq(group)]
        for family in FOCUS_FAMILIES:
            sub = group_df[group_df["family"].eq(family)].sort_values("level")
            x = np.arange(len(sub))
            ax.plot(
                x,
                sub["unique_turn_events_per_rollout"],
                marker="o",
                linewidth=2.1,
                markersize=5.7,
                color=FAMILY_COLORS[family],
                label=FAMILY_LABELS[family],
            )
        ax.set_title(GROUP_DISPLAY[group], fontsize=13.6, pad=4)
        ax.set_xticks(np.arange(4))
        ax.set_xticklabels([EFFORT_SHORT_LABELS[level] for level in ["minimal", "low", "medium", "high"]], fontsize=10.6)
        ax.tick_params(axis="y", labelsize=10.6)
        ax.grid(True, color="#d1d5db", alpha=0.48, linewidth=0.75)

    fig.supylabel("Average Occurrences", fontsize=15.5, x=0.01)
    fig.supxlabel("Reasoning Effort", fontsize=15.5, y=0.035)
    handles = [
        Line2D([0], [0], color=FAMILY_COLORS[family], marker="o", linewidth=2.1, markersize=5.7, label=FAMILY_LABELS[family])
        for family in FOCUS_FAMILIES
    ]
    legend = fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.018),
        ncol=2,
        fontsize=10.8,
        frameon=True,
        facecolor="white",
        framealpha=0.95,
        borderpad=0.32,
    )
    legend.get_frame().set_edgecolor("#d1d5db")
    fig.tight_layout(rect=[0.045, 0.085, 1.0, 1.0], h_pad=0.78, w_pad=0.86)

    out_png = ICML_GRAPHICS / "qualitative_ttc/ttc_group_intensity_fullwidth_2x3_compact.png"
    out_pdf = ICML_GRAPHICS / "qualitative_ttc/ttc_group_intensity_fullwidth_2x3_compact.pdf"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160, bbox_inches="tight", pad_inches=0.06)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    return out_png, out_pdf


def main() -> None:
    print(plot_ttc_payoff_bars())
    print(*plot_ttc_intensity_grid(), sep="\n")
    print(*plot_ttc_intensity_grid_fullwidth_2x3_compact(), sep="\n")


if __name__ == "__main__":
    main()
