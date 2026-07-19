#!/usr/bin/env python3
"""Plot turn-deduplicated TTC group intensity for GPT-5 and Gemini."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TAG_ROOT = PROJECT_ROOT / "analysis/ttc_llm_strategic_tag_adjudication_20260629"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "overleaf/neurips/graphics/qualitative_ttc"
DEFAULT_VERIFY_DIR = PROJECT_ROOT / "analysis/ttc_group_intensity_turn_dedup_verification_20260701"

FAMILY_LABELS = {
    "gpt-5": "GPT-5",
    "gemini-3-flash": "Gemini 3 Flash",
}
COMBINED_COLORS = {
    "gemini-3-flash": "#1f77b4",
    "gpt-5": "#d62728",
}
SINGLE_COLORS = {
    "gemini-3-flash": "#1f77b4",
    "gpt-5": "#1f77b4",
}
LEVEL_ORDER = {"minimal": 0, "low": 1, "medium": 2, "high": 3, "max": 4}
FOCUS_FAMILIES = ["gemini-3-flash", "gpt-5"]
FOCUS_GROUPS = [
    "emotional persuasion",
    "trade/compromise",
    "logical persuasion",
    "pressure",
    "self-interest/exploitation",
    "formalization",
]


def read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_turn_dedup_summary(tag_root: Path) -> pd.DataFrame:
    events = pd.DataFrame(read_jsonl(tag_root / "ttc_llm_event_tags.jsonl"))
    manifest = pd.DataFrame(read_jsonl(tag_root / "all_ttc_rollouts_manifest.jsonl"))
    codebook = pd.DataFrame(json.loads((tag_root / "llm_tag_codebook.json").read_text(encoding="utf-8")))

    events = events.merge(codebook[["tag_code", "category"]], on="tag_code", how="left")
    events["level_index"] = events["level"].map(LEVEL_ORDER).fillna(events["level_index"]).astype(int)
    manifest["level_index"] = manifest["level"].map(LEVEL_ORDER).fillna(manifest["level_index"]).astype(int)

    denoms = (
        manifest.groupby(["family", "level", "level_index"], as_index=False)
        .agg(rollout_count=("config_id", "nunique"))
        .sort_values(["family", "level_index"])
    )

    target = events[events["speaker_role"].eq("target")].copy()
    turn_keys = ["config_id", "category", "round", "phase", "discussion_turn"]
    target_unique_turns = target.drop_duplicates(turn_keys)
    counts = (
        target_unique_turns.groupby(["category", "family", "level", "level_index"], as_index=False)
        .size()
        .rename(columns={"size": "unique_turn_event_count"})
    )

    categories = codebook[["category"]].dropna().drop_duplicates()
    grid = categories.merge(denoms, how="cross")
    summary = grid.merge(counts, on=["category", "family", "level", "level_index"], how="left")
    summary["unique_turn_event_count"] = summary["unique_turn_event_count"].fillna(0).astype(int)
    summary["unique_turn_events_per_rollout"] = summary["unique_turn_event_count"] / summary["rollout_count"]
    return summary.sort_values(["family", "category", "level_index"])


def style_axis(ax: plt.Axes) -> None:
    ax.grid(True, color="#d1d5db", alpha=0.45, linewidth=0.8)
    ax.tick_params(axis="both", labelsize=17)


def plot_single_family(summary: pd.DataFrame, family: str, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), sharex=True)
    axes = axes.ravel()
    color = SINGLE_COLORS[family]
    sub_family = summary[summary["family"].eq(family)]

    for ax, group in zip(axes, FOCUS_GROUPS, strict=True):
        sub = sub_family[sub_family["category"].eq(group)].sort_values("level_index")
        ax.plot(
            sub["level_index"],
            sub["unique_turn_events_per_rollout"],
            marker="o",
            linewidth=2.0,
            markersize=7.0,
            color=color,
        )
        ax.set_title(group, fontsize=21, pad=5)
        ax.set_xticks([0, 1, 2, 3], ["minimal", "low", "medium", "high"])
        style_axis(ax)

    fig.suptitle(
        f"{FAMILY_LABELS[family]}: group intensity vs. TTC effort (turn-deduplicated)",
        fontsize=26,
        y=0.985,
    )
    fig.supylabel("Mean unique-turn events per rollout", fontsize=25, x=0.025)
    fig.supxlabel("Requested reasoning effort", fontsize=25, y=0.035)
    fig.tight_layout(rect=[0.045, 0.06, 1.0, 0.93])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_combined(summary: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), sharex=True)
    axes = axes.ravel()

    for ax, group in zip(axes, FOCUS_GROUPS, strict=True):
        for family in FOCUS_FAMILIES:
            sub = summary[
                summary["family"].eq(family)
                & summary["category"].eq(group)
                & summary["level"].isin(["minimal", "low", "medium", "high"])
            ].sort_values("level_index")
            ax.plot(
                sub["level_index"],
                sub["unique_turn_events_per_rollout"],
                marker="o",
                linewidth=2.0,
                markersize=6.7,
                color=COMBINED_COLORS[family],
                label=FAMILY_LABELS[family],
            )
        ax.set_title(group, fontsize=21, pad=5)
        ax.set_xticks([0, 1, 2, 3], ["minimal", "low", "medium", "high"])
        style_axis(ax)

    handles = [
        plt.Line2D([0], [0], color=COMBINED_COLORS[f], marker="o", linewidth=2.0, label=FAMILY_LABELS[f])
        for f in FOCUS_FAMILIES
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=True, bbox_to_anchor=(0.5, 0.935), fontsize=14)
    fig.suptitle("Group intensity vs. TTC effort", fontsize=26, y=0.985)
    fig.supylabel("Mean unique-turn events per rollout", fontsize=25, x=0.025)
    fig.supxlabel("Requested reasoning effort", fontsize=25, y=0.035)
    fig.tight_layout(rect=[0.045, 0.06, 1.0, 0.89])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag-root", type=Path, default=TAG_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--verify-dir", type=Path, default=DEFAULT_VERIFY_DIR)
    args = parser.parse_args()

    summary = load_turn_dedup_summary(args.tag_root)
    args.verify_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = args.verify_dir / "ttc_group_intensity_turn_dedup_summary.csv"
    summary.to_csv(summary_path, index=False)

    verify_paths = {
        "gemini-3-flash": args.verify_dir / "ttc_gemini_group_intensity_recreated.png",
        "gpt-5": args.verify_dir / "ttc_gpt5_group_intensity_recreated.png",
    }
    for family, path in verify_paths.items():
        plot_single_family(summary, family, path)

    combined_path = args.output_dir / "ttc_gemini_gpt5_group_intensity_combined.png"
    plot_combined(summary, combined_path)

    print(summary_path)
    for path in verify_paths.values():
        print(path)
    print(combined_path)


if __name__ == "__main__":
    main()
