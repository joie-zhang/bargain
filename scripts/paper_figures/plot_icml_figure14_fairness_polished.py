#!/usr/bin/env python3
"""Render the polished three-row ICML AIWILD Appendix Figure 14.

The paper constructs Figure 14 from three full-width PNGs.  This renderer uses
the retained primary-protocol run table, preserves the existing data and color
semantics, and applies the appendix-specific typography/layout requested for
the final paper.  It also writes an asset-only vertical composite for review.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_CSV = (
    REPO_ROOT
    / "experiments/results/n2_baseline_comparison_analysis_20260505"
    / "primary_runs_with_metrics.csv"
)
OUTPUT_DIR = (
    REPO_ROOT
    / "overleaf/icml_aiwild_template/graphics/n2_gpt5_nano"
)

OVERALL_OUTPUT = OUTPUT_DIR / "11_fairness_distance_overall.png"
COMPETITION_OUTPUT = OUTPUT_DIR / "12_fairness_distance_by_competition.png"
ROLE_OUTPUT = OUTPUT_DIR / "13_fairness_excess_by_role_overall.png"
COMPOSITE_OUTPUT = OUTPUT_DIR / "figure14_fairness_benchmark_relative_extraction_revised.png"

GAME_ORDER = ("game1", "game2", "game3")
GAME_TITLES = {
    "game1": "Game 1: Item Allocation",
    "game2": "Game 2: Diplomatic Treaty",
    "game3": "Game 3: Co-Funding",
}

OVERALL_POINT_COLOR = "#be123c"
FIT_COLOR = "#111827"
ADVERSARY_COLOR = "#b45309"
BASELINE_COLOR = "#2563eb"

TITLE_SIZE = 20
Y_LABEL_SIZE = 17
X_LABEL_SIZE = 15
TICK_SIZE = 12.5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sem(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if len(clean) <= 1:
        return 0.0
    return float(clean.std(ddof=1) / np.sqrt(len(clean)))


def aggregate(df: pd.DataFrame, groups: list[str], metric: str) -> pd.DataFrame:
    return (
        df.groupby(groups, as_index=False, dropna=False)
        .agg(n=(metric, "size"), mean=(metric, "mean"), sem=(metric, sem))
        .replace([np.inf, -np.inf], np.nan)
    )


def trend(ax: plt.Axes, df: pd.DataFrame, color: str, *, linewidth: float = 3.0) -> float:
    clean = df[["adversary_elo", "mean"]].dropna()
    if len(clean) < 2 or clean["adversary_elo"].nunique() < 2:
        return float("nan")
    slope, intercept = np.polyfit(clean["adversary_elo"], clean["mean"], 1)
    xs = np.linspace(float(clean["adversary_elo"].min()), float(clean["adversary_elo"].max()), 200)
    ax.plot(
        xs,
        slope * xs + intercept,
        color=color,
        linestyle="--",
        linewidth=linewidth,
        alpha=0.96,
        zorder=2,
    )
    return float(slope * 100.0)


def style_panel(
    ax: plt.Axes,
    game_id: str,
    *,
    column: int,
    ylabel: str,
    show_x: bool,
    show_title: bool,
) -> None:
    ax.set_title(GAME_TITLES[game_id] if show_title else "", fontsize=TITLE_SIZE, pad=13)
    ax.set_ylabel(ylabel if column == 0 else "", fontsize=Y_LABEL_SIZE, labelpad=10)
    if show_x:
        ax.set_xlabel("Adversary Chatbot Arena Elo", fontsize=X_LABEL_SIZE, labelpad=8)
        ax.tick_params(axis="x", labelsize=TICK_SIZE)
    else:
        ax.set_xlabel("")
        ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax.tick_params(axis="y", labelsize=TICK_SIZE)
    ax.grid(alpha=0.20, linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)


def render_overall(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(19.0, 4.65), sharey=False)
    for column, (ax, game_id) in enumerate(zip(axes, GAME_ORDER, strict=True)):
        game = df[df["game_id"].eq(game_id)]
        points = aggregate(
            game,
            ["adversary_model", "adversary_elo"],
            "fairness_distance",
        ).sort_values("adversary_elo")
        defined = points.dropna(subset=["adversary_elo", "mean"])
        ax.scatter(
            defined["adversary_elo"],
            defined["mean"],
            s=64,
            color=OVERALL_POINT_COLOR,
            alpha=0.88,
            edgecolors="none",
            zorder=3,
        )
        trend(ax, defined, FIT_COLOR, linewidth=3.6)
        style_panel(
            ax,
            game_id,
            column=column,
            ylabel="Mean Fairness Distance",
            show_x=False,
            show_title=True,
        )
    fig.subplots_adjust(left=0.070, right=0.995, bottom=0.070, top=0.875, wspace=0.18)
    fig.savefig(OVERALL_OUTPUT, dpi=220, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def render_by_competition(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(19.0, 5.35), sharey=False)
    for column, (ax, game_id) in enumerate(zip(axes, GAME_ORDER, strict=True)):
        game = df[df["game_id"].eq(game_id)]
        points = aggregate(
            game,
            [
                "competition_value",
                "competition_label",
                "adversary_model",
                "adversary_elo",
            ],
            "fairness_distance",
        ).sort_values(["competition_value", "adversary_elo"])
        competition_values = sorted(points["competition_value"].dropna().unique())
        colors = plt.cm.viridis(np.linspace(0.05, 0.95, max(len(competition_values), 1)))
        handles: list[Line2D] = []
        for color, competition_value in zip(colors, competition_values, strict=False):
            group = points[points["competition_value"].eq(competition_value)].sort_values("adversary_elo")
            defined = group.dropna(subset=["adversary_elo", "mean"])
            missing = group[group["mean"].isna()]
            trend(ax, defined, color, linewidth=3.2)
            ax.scatter(
                defined["adversary_elo"],
                defined["mean"],
                s=48,
                color=color,
                alpha=0.56,
                edgecolors="none",
                zorder=3,
            )
            if not missing.empty:
                ymin, ymax = ax.get_ylim()
                marker_y = ymin + 0.03 * (ymax - ymin)
                ax.scatter(
                    missing["adversary_elo"],
                    np.full(len(missing), marker_y),
                    marker="x",
                    s=48,
                    linewidths=1.2,
                    color=color,
                    alpha=0.60,
                    zorder=4,
                )
            label = str(group["competition_label"].iloc[0])
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color=color,
                    linestyle="--",
                    linewidth=3.2,
                    marker="o",
                    markersize=6.0,
                    markerfacecolor=color,
                    markeredgewidth=0,
                    alpha=0.90,
                    label=label,
                )
            )
        style_panel(
            ax,
            game_id,
            column=column,
            ylabel="Mean Fairness Distance",
            show_x=False,
            show_title=False,
        )
        ax.set_ylim(bottom=0)
        legend = ax.legend(
            handles=handles,
            title="Competition",
            fontsize=12.2,
            title_fontsize=13.2,
            frameon=True,
            loc="best",
            borderpad=0.55,
            labelspacing=0.35,
            handlelength=2.2,
        )
    fig.subplots_adjust(left=0.070, right=0.995, bottom=0.060, top=0.885, wspace=0.18)
    fig.savefig(COMPETITION_OUTPUT, dpi=220, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def render_by_role(df: pd.DataFrame) -> None:
    historical = df.copy()
    no_project = historical["game_id"].eq("game3") & historical[
        "actual_funded_project_count"
    ].fillna(0).eq(0)
    historical.loc[
        no_project,
        ["adversary_fairness_excess", "baseline_fairness_excess"],
    ] = np.nan

    metrics = (
        ("adversary_fairness_excess", "Adversary", ADVERSARY_COLOR),
        ("baseline_fairness_excess", "Baseline", BASELINE_COLOR),
    )
    fig, axes = plt.subplots(1, 3, figsize=(19.0, 5.15), sharey=False)
    for column, (ax, game_id) in enumerate(zip(axes, GAME_ORDER, strict=True)):
        game = historical[historical["game_id"].eq(game_id)]
        handles: list[Line2D] = []
        for metric, label, color in metrics:
            points = aggregate(
                game,
                ["adversary_model", "adversary_elo"],
                metric,
            ).sort_values("adversary_elo")
            defined = points.dropna(subset=["adversary_elo", "mean"])
            trend(ax, defined, color, linewidth=3.4)
            ax.scatter(
                defined["adversary_elo"],
                defined["mean"],
                s=52,
                color=color,
                alpha=0.58,
                edgecolors="none",
                zorder=3,
            )
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color=color,
                    linestyle="--",
                    linewidth=3.4,
                    marker="o",
                    markersize=6.2,
                    markerfacecolor=color,
                    markeredgewidth=0,
                    alpha=0.92,
                    label=label,
                )
            )
        ax.axhline(0, color="#111827", linewidth=1.1, alpha=0.70, zorder=1)
        style_panel(
            ax,
            game_id,
            column=column,
            ylabel="Utility Above Fair Benchmark",
            show_x=True,
            show_title=False,
        )
        legend = ax.legend(
            handles=handles,
            title="Role",
            fontsize=13.5,
            title_fontsize=14.5,
            frameon=True,
            loc="best",
            borderpad=0.65,
            labelspacing=0.45,
            handlelength=2.3,
        )
    fig.subplots_adjust(left=0.070, right=0.995, bottom=0.185, top=0.880, wspace=0.18)
    fig.savefig(ROLE_OUTPUT, dpi=220, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def write_vertical_composite(paths: tuple[Path, ...], output: Path) -> None:
    images = [Image.open(path).convert("RGB") for path in paths]
    width = max(image.width for image in images)
    gap = max(24, int(width * 0.008))
    scaled: list[Image.Image] = []
    for image in images:
        if image.width == width:
            scaled.append(image)
            continue
        height = round(image.height * width / image.width)
        scaled.append(image.resize((width, height), Image.Resampling.LANCZOS))
    height = sum(image.height for image in scaled) + gap * (len(scaled) - 1)
    canvas = Image.new("RGB", (width, height), "white")
    y = 0
    for image in scaled:
        canvas.paste(image, ((width - image.width) // 2, y))
        y += image.height + gap
    canvas.save(output, dpi=(220, 220), optimize=True)


def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(INPUT_CSV)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(INPUT_CSV)
    rows = rows[rows["baseline_key"].eq("gpt5_nano")].copy()
    expected = {"game1": 420, "game2": 540, "game3": 540}
    actual = {str(key): int(value) for key, value in rows.groupby("game_id").size().items()}
    if actual != expected:
        raise RuntimeError(f"Unexpected Figure 14 input counts: {actual}; expected {expected}")

    render_overall(rows)
    render_by_competition(rows)
    render_by_role(rows)
    write_vertical_composite(
        (OVERALL_OUTPUT, COMPETITION_OUTPUT, ROLE_OUTPUT),
        COMPOSITE_OUTPUT,
    )

    print(f"Input: {INPUT_CSV}")
    print(f"Rows: {len(rows)} ({actual})")
    for path in (OVERALL_OUTPUT, COMPETITION_OUTPUT, ROLE_OUTPUT, COMPOSITE_OUTPUT):
        with Image.open(path) as image:
            print(f"Wrote: {path} | {image.width}x{image.height} | sha256={sha256(path)}")


if __name__ == "__main__":
    main()
