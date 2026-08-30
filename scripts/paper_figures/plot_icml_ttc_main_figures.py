#!/usr/bin/env python3
"""Regenerate ICML TTC main-text figures."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ICML_GRAPHICS = PROJECT_ROOT / "overleaf/icml_aiwild_template/graphics"
TTC_SUMMARY_CSV = (
    PROJECT_ROOT
    / "analysis/ttc_complete_family_seed_panels_20260810/family_effort_complete_seed_ci95.csv"
)
TTC_PAYOFF_SEED_COUNTS = {
    "gpt-5": 10,
    "claude-sonnet-4-6": 10,
    "gemini-3-flash": 10,
}
TTC_INTENSITY_CSV = (
    PROJECT_ROOT
    / "overleaf/icml_aiwild_template/graphics/qualitative_ttc/ttc_selected23_summary.csv"
)
PROVENANCE_PATH = ICML_GRAPHICS / "ttc_main_figures_provenance.json"

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
    "claude-sonnet-4-6": "#2ca02c",
}
FAMILY_MARKERS = {
    "gemini-3-flash": "o",
    "gpt-5": "o",
    "claude-sonnet-4-6": "o",
}
FOCUS_FAMILIES = ["gemini-3-flash", "gpt-5", "claude-sonnet-4-6"]
INTENSITY_LEVELS = {
    "gemini-3-flash": ["minimal", "low", "medium", "high"],
    "gpt-5": ["minimal", "low", "medium", "high"],
    "claude-sonnet-4-6": ["low", "medium", "high", "max"],
}
INTENSITY_X_LABELS = ["min/low", "low/med", "med/high", "high/max"]
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

EXPECTED_FAMILY_LEVELS = {
    "gpt-5": {"minimal", "low", "medium", "high"},
    "claude-sonnet-4-6": {"low", "medium", "high", "max"},
    "gemini-3-flash": {"minimal", "low", "medium", "high"},
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_payoff_summary(summary: pd.DataFrame) -> None:
    if len(summary) != 12:
        raise RuntimeError(f"Expected 12 TTC family-effort rows, found {len(summary)}")
    actual = {
        family: set(group["level"].astype(str))
        for family, group in summary.groupby("family")
    }
    if actual != EXPECTED_FAMILY_LEVELS:
        raise RuntimeError(f"Unexpected TTC family-effort grid: {actual}")
    seed_counts = {
        family: set(group["seed_count"].astype(int))
        for family, group in summary.groupby("family")
    }
    expected_counts = {
        family: {count} for family, count in TTC_PAYOFF_SEED_COUNTS.items()
    }
    if seed_counts != expected_counts:
        raise RuntimeError(f"Unexpected complete-seed counts: {seed_counts}")
    if set(summary["game_cell_count_min"].astype(int)) != {9} or set(
        summary["game_cell_count_max"].astype(int)
    ) != {9}:
        raise RuntimeError("Each TTC seed estimate must average nine game cells")
    required = [
        "target_utility_mean",
        "target_utility_seed_ci95_low",
        "target_utility_seed_ci95_high",
    ]
    if summary[required].isna().any().any():
        raise RuntimeError("The TTC payoff summary contains missing values")


def validate_intensity_summary(summary: pd.DataFrame) -> None:
    focus = summary[
        summary["family"].isin(FOCUS_FAMILIES)
        & summary["category"].isin(FOCUS_GROUPS)
    ].copy()
    expected_rows = len(FOCUS_FAMILIES) * len(FOCUS_GROUPS) * 4
    if len(focus) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} TTC intensity cells, found {len(focus)}")
    if set(pd.to_numeric(focus["rollout_count"])) != {18}:
        raise RuntimeError("Each displayed TTC family-effort cell must contain 18 rollouts")
    value_columns = [
        "mean_turn_deduplicated_category_events",
        "sem_across_rollouts",
    ]
    if focus[value_columns].isna().any().any():
        raise RuntimeError("The TTC intensity summary contains missing displayed values")
    if set(focus["tag_universe"].astype(str)) != {"selected 23 paper tags"}:
        raise RuntimeError("The TTC intensity summary is not the selected-23 analysis")
    for family, expected_levels in INTENSITY_LEVELS.items():
        actual_levels = (
            focus.loc[focus["family"].eq(family)]
            .sort_values("level_position")["level"]
            .drop_duplicates()
            .tolist()
        )
        if actual_levels != expected_levels:
            raise RuntimeError(
                f"Unexpected intensity levels for {family}: {actual_levels}"
            )


def plot_ttc_payoff_bars() -> Path:
    summary = pd.read_csv(TTC_SUMMARY_CSV)
    summary = summary[summary["family"].isin(FAMILY_ORDER)].copy()
    validate_payoff_summary(summary)
    summary["level"] = pd.Categorical(summary["level"], EFFORT_ORDER, ordered=True)
    summary = summary.sort_values(["family", "level"])

    fig, axes = plt.subplots(1, 3, figsize=(14.6, 4.9), sharey=True)
    fig.patch.set_facecolor("white")

    for ax, family in zip(axes, FAMILY_ORDER, strict=True):
        family_df = summary[summary["family"].eq(family)].sort_values("level")
        means = family_df["target_utility_mean"].to_numpy(dtype=float)
        yerr = np.vstack(
            [
                means
                - family_df["target_utility_seed_ci95_low"].to_numpy(dtype=float),
                family_df["target_utility_seed_ci95_high"].to_numpy(dtype=float)
                - means,
            ]
        )
        x = np.arange(len(family_df))
        colors = [EFFORT_COLORS[str(level)] for level in family_df["level"]]
        ax.bar(
            x,
            means,
            yerr=yerr,
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

    axes[0].set_ylabel(
        "Mean Target Payoff\n(no consensus = 0)", fontsize=20, labelpad=10
    )
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
    plt.rcParams.update({"font.family": "DejaVu Sans", "lines.solid_capstyle": "round"})
    summary = pd.read_csv(TTC_INTENSITY_CSV)
    validate_intensity_summary(summary)
    summary = summary[
        summary["family"].isin(FOCUS_FAMILIES)
        & summary["category"].isin(FOCUS_GROUPS)
    ].copy()
    summary["category"] = pd.Categorical(summary["category"], FOCUS_GROUPS, ordered=True)
    summary = summary.sort_values(["category", "family", "level_position"])

    fig, axes = plt.subplots(3, 2, figsize=(6.9, 7.75), sharex=True)
    axes = axes.ravel()

    for ax, group in zip(axes, FOCUS_GROUPS, strict=True):
        group_df = summary[summary["category"].eq(group)]
        for family in FOCUS_FAMILIES:
            sub = group_df[group_df["family"].eq(family)].sort_values("level_position")
            ax.errorbar(
                sub["level_position"],
                sub["mean_turn_deduplicated_category_events"],
                yerr=sub["sem_across_rollouts"],
                color=FAMILY_COLORS[family],
                marker=FAMILY_MARKERS[family],
                markersize=4.6,
                linewidth=1.8,
                elinewidth=1.0,
                capsize=2.5,
                capthick=1.0,
                label=FAMILY_LABELS[family],
            )
        ax.set_title(GROUP_DISPLAY[group], fontsize=11.5, pad=1.5)
        ax.set_xticks(np.arange(1, 5))
        ax.set_xticklabels(INTENSITY_X_LABELS, fontsize=8)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(True, color="#d1d5db", alpha=0.48, linewidth=0.7)

    fig.supylabel("Average Occurrences", fontsize=12.5, x=0.038)
    fig.supxlabel("Requested Effort (GPT-5/Gemini / Claude)", fontsize=11.5, y=0.068)
    handles = [
        Line2D(
            [0],
            [0],
            color=FAMILY_COLORS[family],
            marker=FAMILY_MARKERS[family],
            linewidth=1.8,
            markersize=4.6,
            label=FAMILY_LABELS[family],
        )
        for family in FOCUS_FAMILIES
    ]
    legend = fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        ncol=len(FOCUS_FAMILIES),
        fontsize=8.5,
        frameon=True,
        facecolor="white",
        framealpha=0.95,
        borderpad=0.35,
    )
    legend.get_frame().set_edgecolor("#d1d5db")
    fig.tight_layout(rect=[0.055, 0.082, 1.0, 1.0], h_pad=0.65, w_pad=0.8)

    out_png = ICML_GRAPHICS / "qualitative_ttc/ttc_group_intensity_singlecolumn_3x2.png"
    out_pdf = ICML_GRAPHICS / "qualitative_ttc/ttc_group_intensity_singlecolumn_3x2.pdf"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220, bbox_inches="tight", pad_inches=0.06)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    return out_png, out_pdf


def plot_ttc_intensity_grid_fullwidth_2x3_compact() -> tuple[Path, Path]:
    summary = pd.read_csv(TTC_INTENSITY_CSV)
    validate_intensity_summary(summary)
    summary = summary[
        summary["family"].isin(FOCUS_FAMILIES)
        & summary["category"].isin(FOCUS_GROUPS)
    ].copy()
    summary["category"] = pd.Categorical(summary["category"], FOCUS_GROUPS, ordered=True)
    summary = summary.sort_values(["category", "family", "level_position"])

    fig, axes = plt.subplots(2, 3, figsize=(11.1, 5.9), sharex=True)
    axes = axes.ravel()

    for ax, group in zip(axes, FOCUS_GROUPS, strict=True):
        group_df = summary[summary["category"].eq(group)]
        for family in FOCUS_FAMILIES:
            sub = group_df[group_df["family"].eq(family)].sort_values("level_position")
            ax.errorbar(
                sub["level_position"],
                sub["mean_turn_deduplicated_category_events"],
                yerr=sub["sem_across_rollouts"],
                marker=FAMILY_MARKERS[family],
                linewidth=2.1,
                markersize=5.7,
                elinewidth=0.9,
                capsize=2.3,
                color=FAMILY_COLORS[family],
                label=FAMILY_LABELS[family],
            )
        ax.set_title(GROUP_DISPLAY[group], fontsize=13.6, pad=4)
        ax.set_xticks(np.arange(1, 5))
        ax.set_xticklabels(INTENSITY_X_LABELS, fontsize=10.6)
        ax.tick_params(axis="y", labelsize=10.6)
        ax.grid(True, color="#d1d5db", alpha=0.48, linewidth=0.75)

    fig.supylabel("Average Occurrences", fontsize=15.5, x=0.01)
    fig.supxlabel("Reasoning Effort", fontsize=15.5, y=0.035)
    handles = [
        Line2D([0], [0], color=FAMILY_COLORS[family], marker=FAMILY_MARKERS[family], linewidth=2.1, markersize=5.7, label=FAMILY_LABELS[family])
        for family in FOCUS_FAMILIES
    ]
    legend = fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.018),
        ncol=len(FOCUS_FAMILIES),
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


def write_provenance(intensity_png: Path, intensity_pdf: Path) -> None:
    PROVENANCE_PATH.write_text(
        json.dumps(
            {
                "producer": str(Path(__file__).resolve().relative_to(PROJECT_ROOT)),
                "payoff_summary": str(TTC_SUMMARY_CSV.relative_to(PROJECT_ROOT)),
                "payoff_summary_sha256": sha256(TTC_SUMMARY_CSV),
                "payoff_terminal_result_artifact_count": 2160,
                "payoff_standard_cap_terminal_result_count": 2159,
                "payoff_retained_comparable_panel_game_count": 2160,
                "payoff_family_seed_counts": TTC_PAYOFF_SEED_COUNTS,
                "payoff_protocol_exception": {
                    "seed": 612,
                    "config_id": 142,
                    "family": "claude-sonnet-4-6",
                    "recovery_max_tokens_per_phase": 65536,
                    "analysis_treatment": "include the complete seed-family panel",
                },
                "payoff_uncertainty": {
                    "unit": "complete family-seed estimate",
                    "method": "two-sided Student-t 95% CI",
                    "degrees_of_freedom_by_family": {
                        "gpt-5": 9,
                        "claude-sonnet-4-6": 9,
                        "gemini-3-flash": 9,
                    },
                },
                "payoff_estimand": "unconditional expected target payoff; no consensus = 0",
                "intensity_summary": str(TTC_INTENSITY_CSV.relative_to(PROJECT_ROOT)),
                "intensity_summary_sha256": sha256(TTC_INTENSITY_CSV),
                "intensity_tag_universe": "selected_23_non_coalition_labels",
                "intensity_seed": 42,
                "intensity_run_count": 216,
                "displayed_intensity_families": FOCUS_FAMILIES,
                "displayed_rollouts_per_family_effort": 18,
                "intensity_png_sha256": sha256(intensity_png),
                "intensity_pdf_sha256": sha256(intensity_pdf),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--payoff-only",
        action="store_true",
        help=(
            "Regenerate only the multi-seed payoff figure and provenance; "
            "leave the separate seed-42 qualitative assets unchanged."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payoff_path = plot_ttc_payoff_bars()
    intensity_png = ICML_GRAPHICS / "qualitative_ttc/ttc_group_intensity_singlecolumn_3x2.png"
    intensity_pdf = ICML_GRAPHICS / "qualitative_ttc/ttc_group_intensity_singlecolumn_3x2.pdf"
    print(payoff_path)
    if not args.payoff_only:
        intensity_png, intensity_pdf = plot_ttc_intensity_grid()
        fullwidth_png, fullwidth_pdf = plot_ttc_intensity_grid_fullwidth_2x3_compact()
        print(intensity_png, intensity_pdf, sep="\n")
        print(fullwidth_png, fullwidth_pdf, sep="\n")
    elif not intensity_png.is_file() or not intensity_pdf.is_file():
        raise FileNotFoundError(
            "The payoff-only mode requires the retained seed-42 qualitative assets."
        )
    write_provenance(intensity_png, intensity_pdf)


if __name__ == "__main__":
    main()
