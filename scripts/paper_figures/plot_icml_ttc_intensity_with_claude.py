#!/usr/bin/env python3
"""Render the TTC qualitative-intensity grid with Claude included."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SUMMARY_CSV = (
    PROJECT_ROOT
    / "analysis/ttc_group_intensity_turn_dedup_verification_20260701"
    / "ttc_group_intensity_turn_dedup_summary.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "overleaf/icml_aiwild_template/graphics/qualitative_ttc"

FAMILY_ORDER = ["gemini-3-flash", "gpt-5", "claude-sonnet-4-6"]
FAMILY_LABELS = {
    "gemini-3-flash": "Gemini 3 Flash",
    "gpt-5": "GPT-5",
    "claude-sonnet-4-6": "Claude Sonnet 4.6",
}
FAMILY_COLORS = {
    "gemini-3-flash": "#1f77b4",
    "gpt-5": "#d62728",
    "claude-sonnet-4-6": "#2ca02c",
}
FAMILY_MARKERS = {
    "gemini-3-flash": "o",
    "gpt-5": "o",
    "claude-sonnet-4-6": "D",
}
FAMILY_LEVELS = {
    "gemini-3-flash": ["minimal", "low", "medium", "high"],
    "gpt-5": ["minimal", "low", "medium", "high"],
    "claude-sonnet-4-6": ["low", "medium", "high", "max"],
}
LEVEL_INDEX = {"minimal": 0, "low": 1, "medium": 2, "high": 3, "max": 4}
LEVEL_SHORT_LABELS = ["min", "low", "med", "high", "max"]

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


def load_and_validate_summary() -> pd.DataFrame:
    summary = pd.read_csv(SUMMARY_CSV)
    focus = summary[
        summary["family"].isin(FAMILY_ORDER)
        & summary["category"].isin(FOCUS_GROUPS)
    ].copy()

    expected_rows = len(FAMILY_ORDER) * len(FOCUS_GROUPS) * 4
    if len(focus) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} displayed family-category-effort cells, "
            f"found {len(focus)}"
        )
    if set(pd.to_numeric(focus["rollout_count"])) != {18}:
        raise RuntimeError("Every family-effort cell must contain exactly 18 rollouts")
    if focus["unique_turn_events_per_rollout"].isna().any():
        raise RuntimeError("Displayed qualitative-intensity values contain missing data")

    for family, expected_levels in FAMILY_LEVELS.items():
        actual_levels = (
            focus.loc[focus["family"].eq(family), "level"]
            .drop_duplicates()
            .sort_values(key=lambda levels: levels.map(LEVEL_INDEX))
            .tolist()
        )
        if actual_levels != expected_levels:
            raise RuntimeError(
                f"Unexpected effort levels for {family}: "
                f"expected {expected_levels}, found {actual_levels}"
            )

    return focus


def plot_with_claude(summary: pd.DataFrame) -> tuple[Path, Path]:
    fig, axes = plt.subplots(3, 2, figsize=(6.6, 8.65), sharex=True)
    axes = axes.ravel()

    for ax, group in zip(axes, FOCUS_GROUPS, strict=True):
        group_df = summary[summary["category"].eq(group)]
        for family in FAMILY_ORDER:
            sub = group_df[group_df["family"].eq(family)].copy()
            sub["_x"] = sub["level"].map(LEVEL_INDEX)
            sub = sub.sort_values("_x")
            ax.plot(
                sub["_x"],
                sub["unique_turn_events_per_rollout"],
                marker=FAMILY_MARKERS[family],
                linewidth=2.15,
                markersize=6.3,
                color=FAMILY_COLORS[family],
                label=FAMILY_LABELS[family],
            )

        ax.set_title(GROUP_DISPLAY[group], fontsize=17, pad=4)
        ax.set_xticks(range(5))
        ax.set_xticklabels(LEVEL_SHORT_LABELS, fontsize=12)
        ax.tick_params(axis="y", labelsize=13)
        ax.grid(True, color="#d1d5db", alpha=0.48, linewidth=0.8)

    fig.supylabel("Average Occurrences", fontsize=23, x=0.02)
    fig.supxlabel("Requested Reasoning Effort", fontsize=21, y=0.06)
    handles = [
        Line2D(
            [0],
            [0],
            color=FAMILY_COLORS[family],
            marker=FAMILY_MARKERS[family],
            linewidth=2.15,
            markersize=6.3,
            label=FAMILY_LABELS[family],
        )
        for family in FAMILY_ORDER
    ]
    legend = fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.006),
        ncol=3,
        fontsize=10.6,
        frameon=True,
        facecolor="white",
        framealpha=0.95,
        columnspacing=0.8,
        handlelength=2.2,
        handletextpad=0.45,
        borderpad=0.35,
    )
    legend.get_frame().set_edgecolor("#d1d5db")
    fig.tight_layout(rect=[0.06, 0.105, 1.0, 1.0], h_pad=1.0, w_pad=1.0)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_png = OUTPUT_DIR / "ttc_group_intensity_singlecolumn_3x2_with_claude.png"
    out_pdf = OUTPUT_DIR / "ttc_group_intensity_singlecolumn_3x2_with_claude.pdf"
    fig.savefig(out_png, dpi=320, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return out_png, out_pdf


def main() -> None:
    summary = load_and_validate_summary()
    for path in plot_with_claude(summary):
        print(path)


if __name__ == "__main__":
    main()
