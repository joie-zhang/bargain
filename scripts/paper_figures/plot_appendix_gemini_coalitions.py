#!/usr/bin/env python3
"""Render appendix figures for payoff variance and Gemini coalition case studies."""

from __future__ import annotations

import argparse
import math
import textwrap
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = ROOT / "analysis" / "figure_recreation_20260816"
PAPER_DIRS = (
    ROOT
    / "overleaf"
    / "NExT_Game_2026_style_new"
    / "graphics"
    / "appendix_multiagent",
    ROOT
    / "overleaf"
    / "icml_aiwild_template"
    / "graphics"
    / "appendix_multiagent",
)

GAME_COLORS = {"game1": "#2B7BBA", "game2": "#E63946", "game3": "#2CA02C"}
GAME_LABELS = {"game1": "Game 1", "game2": "Game 2", "game3": "Game 3"}
HETEROGENEOUS_COLOR = "#4B5563"
MONOCULTURE_COLOR = "#287D8E"
LABEL_FONTSIZE = 21
TICK_FONTSIZE = 17
LEFT_LABEL_FONTSIZE = 23
LEFT_TICK_FONTSIZE = 19
LEFT_CATEGORY_FONTSIZE = 22
LEGEND_FONTSIZE = 18
ANNOTATION_FONTSIZE = 19
MARKER_SIZE = 9
STROKE_WIDTH = 1.5

CASES = [
    {
        "panel": "(a)",
        "config": "Config 0107",
        "rule": "3 of 4 votes",
        "payoffs": "Payoffs  [83, 72, 81, 20]",
        "planner": "Agent 1",
        "plan": (
            "A supermajority only requires 3 votes. I can secure a guaranteed "
            "supermajority by forming an alliance with Agent_2 and Agent_3 ... "
            "leaving Agent_4 with just the Clock."
        ),
        "excluded": "Agent 4",
        "reply": (
            "This proposal only gives me the Clock, resulting in a very low "
            "utility of 20."
        ),
        "result": "Passed 3–1",
    },
    {
        "panel": "(b)",
        "config": "Config 0108",
        "rule": "3 of 4 votes",
        "payoffs": "Payoffs  [49, 73, 55, 0]",
        "planner": "Agent 2",
        "plan": (
            "My best path forward is to form a 3-agent voting bloc with Agent_1 "
            "and Agent_3, effectively freezing out Agent_4."
        ),
        "excluded": "Agent 4",
        "reply": (
            "This proposal gives me absolutely nothing, resulting in zero utility."
        ),
        "result": "Passed 3–1",
    },
    {
        "panel": "(c)",
        "config": "Config 0117",
        "rule": "6 of 8 votes",
        "payoffs": "Payoffs  [51.3, 0, 69.3, 59.4, 50.4, 63.9, 0, 52.2]",
        "planner": "Agent 8",
        "plan": (
            "I present the Perfect Alliance of 6. By excluding them, we have "
            "exactly 6 votes where NO ONE in the Core 5 has to compromise."
        ),
        "excluded": "Agents 2 and 7",
        "reply": (
            "Agent 2: This proposal gives me absolutely zero items, resulting in "
            "a utility of 0. I cannot accept it.\nAgent 7: This proposal gives me "
            "zero items, resulting in a utility of 0."
        ),
        "result": "Passed 6–2",
    },
]


def sem(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="raise").dropna()
    return float(clean.std(ddof=1) / math.sqrt(len(clean)))


def save_figure(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")


def style_axis(ax: plt.Axes, *, tick_size: int = TICK_FONTSIZE) -> None:
    ax.grid(axis="y", color="#D1D5DB", linewidth=0.75, alpha=0.52)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(STROKE_WIDTH)
    ax.spines["left"].set_linewidth(STROKE_WIDTH)
    ax.tick_params(
        axis="both",
        labelsize=tick_size,
        length=5,
        width=STROKE_WIDTH,
    )


def draw_variance_figure() -> None:
    aggregate = pd.read_csv(
        ANALYSIS_DIR / "figure_08_payoff_variance_aggregate.csv"
    )
    models = pd.read_csv(
        ANALYSIS_DIR / "figure_08_payoff_variance_model_values.csv"
    )

    heterogeneous_mean = float(aggregate.iloc[0]["payoff_variance_mean"])
    heterogeneous_sem = float(aggregate.iloc[0]["payoff_variance_sem"])

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14.2, 6.0),
        gridspec_kw={"width_ratios": [0.95, 1.60], "wspace": 0.34},
    )
    left, right = axes
    bar_colors = [HETEROGENEOUS_COLOR, MONOCULTURE_COLOR]
    error_colors = ["#374151", "#155E75"]
    for x, mean, error, color, error_color in zip(
        [0, 1],
        aggregate["payoff_variance_mean"],
        aggregate["payoff_variance_sem"],
        bar_colors,
        error_colors,
    ):
        left.bar(
            x,
            mean,
            color=color,
            width=0.64,
            edgecolor="white",
            linewidth=1.2,
            alpha=0.9,
            zorder=2,
        )
        left.errorbar(
            x,
            mean,
            yerr=error,
            fmt="none",
            ecolor=error_color,
            elinewidth=STROKE_WIDTH,
            capsize=5,
            capthick=STROKE_WIDTH,
            zorder=3,
        )
    left.set_xticks([0, 1])
    left.set_xticklabels(
        ["Heterogeneous\nruns", "Homogeneous\nruns"],
        fontsize=LEFT_CATEGORY_FONTSIZE,
    )
    left.set_ylabel(
        "Mean payoff variance",
        fontsize=LEFT_LABEL_FONTSIZE,
        labelpad=8,
    )
    left.set_ylim(
        0,
        1.18
        * float(
            (aggregate["payoff_variance_mean"] + aggregate["payoff_variance_sem"]).max()
        ),
    )
    style_axis(left, tick_size=LEFT_TICK_FONTSIZE)

    for game in ["game1", "game2", "game3"]:
        group = models[models["game_label"].eq(game)]
        right.errorbar(
            group["model_elo"],
            group["payoff_variance_mean"],
            yerr=group["payoff_variance_sem"],
            fmt="o",
            markersize=MARKER_SIZE,
            capsize=4,
            elinewidth=STROKE_WIDTH,
            capthick=STROKE_WIDTH,
            color=GAME_COLORS[game],
            alpha=0.78,
            label=GAME_LABELS[game],
            zorder=3,
        )

    right.axhspan(
        max(0, heterogeneous_mean - heterogeneous_sem),
        heterogeneous_mean + heterogeneous_sem,
        color=HETEROGENEOUS_COLOR,
        alpha=0.14,
        zorder=1,
    )
    right.axhline(
        heterogeneous_mean,
        color=HETEROGENEOUS_COLOR,
        linestyle="--",
        linewidth=STROKE_WIDTH,
        label="Heterogeneous mean",
        zorder=2,
    )
    right.set_xlim(1210, 1520)
    right.set_ylim(
        0,
        1.08
        * float((models["payoff_variance_mean"] + models["payoff_variance_sem"]).max()),
    )
    right.set_xlabel(
        "Homogeneous model's Arena Elo",
        fontsize=LABEL_FONTSIZE,
        labelpad=6,
    )
    right.set_ylabel(
        "Mean payoff variance",
        fontsize=LABEL_FONTSIZE,
        labelpad=8,
    )
    gemini = models[models["model"].eq("gemini-3.1-pro")].iloc[0]
    right.annotate(
        "Gemini 3.1 Pro",
        xy=(gemini["model_elo"], gemini["payoff_variance_mean"]),
        xytext=(-10, 8),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=ANNOTATION_FONTSIZE,
        color=GAME_COLORS["game1"],
        path_effects=[
            path_effects.withStroke(
                linewidth=0.35,
                foreground=GAME_COLORS["game1"],
            )
        ],
    )
    handles, labels = right.get_legend_handles_labels()
    order = [
        labels.index("Heterogeneous mean"),
        labels.index(GAME_LABELS["game1"]),
        labels.index(GAME_LABELS["game2"]),
        labels.index(GAME_LABELS["game3"]),
    ]
    right.legend(
        [handles[index] for index in order],
        [labels[index] for index in order],
        frameon=False,
        ncol=1,
        loc="upper left",
        fontsize=LEGEND_FONTSIZE,
        handlelength=2.0,
        columnspacing=1.2,
        handletextpad=0.55,
    )
    style_axis(right)

    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.19, top=0.97)
    for paper_dir in PAPER_DIRS:
        save_figure(
            fig,
            paper_dir / "payoff_variance_homogeneous_runs",
        )
    plt.close(fig)


def add_box(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    facecolor: str,
    edgecolor: str,
    radius: float = 0.03,
) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle=f"round,pad=0.012,rounding_size={radius}",
            linewidth=1.2,
            facecolor=facecolor,
            edgecolor=edgecolor,
        )
    )


def draw_quote_figure_v1() -> None:
    """Render the initial three-column design for visual review."""
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 6.2))
    fig.patch.set_facecolor("white")

    for ax, case in zip(axes, CASES, strict=True):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        add_box(ax, 0.02, 0.02, 0.96, 0.95, facecolor="#F7F8FA", edgecolor="#D9DEE7")
        ax.text(
            0.07,
            0.92,
            f"{case['panel']} {case['config']}",
            fontsize=13,
            fontweight="bold",
            color="#17324D",
            va="top",
        )
        ax.text(0.07, 0.865, case["rule"], fontsize=10, color="#52606D", va="top")

        add_box(ax, 0.07, 0.52, 0.86, 0.29, facecolor="#E8F1FB", edgecolor="#8CB3D9")
        ax.text(0.10, 0.77, case["planner"], fontsize=10.5, fontweight="bold", color="#255C8D")
        ax.text(
            0.10,
            0.725,
            textwrap.fill(f'“{case["plan"]}”', width=42),
            fontsize=9.2,
            color="#1F2933",
            va="top",
            linespacing=1.25,
        )

        add_box(ax, 0.07, 0.20, 0.86, 0.24, facecolor="#FDEBEC", edgecolor="#DFA0A4")
        ax.text(0.10, 0.40, case["excluded"], fontsize=10.5, fontweight="bold", color="#A63D45")
        ax.text(
            0.10,
            0.355,
            textwrap.fill(f'“{case["reply"]}”', width=42),
            fontsize=9.2,
            color="#1F2933",
            va="top",
            linespacing=1.25,
        )

        ax.text(0.07, 0.12, case["payoffs"], fontsize=9.1, color="#52606D")
        ax.text(
            0.93,
            0.07,
            case["result"],
            fontsize=10,
            fontweight="bold",
            color="#1F7A4D",
            ha="right",
        )

    fig.suptitle(
        "Gemini 3.1 Pro forms minimum-winning coalitions",
        fontsize=16,
        fontweight="bold",
        color="#17324D",
        y=0.99,
    )
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.03, top=0.91, wspace=0.04)
    save_figure(fig, ANALYSIS_DIR / "gemini_minimum_coalitions_quotes_v1")
    plt.close(fig)


def draw_quote_figure_v2() -> None:
    """Render the revised full-width-row design used in the paper."""
    fig, ax = plt.subplots(figsize=(7.2, 8.35))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.968,
        "Gemini 3.1 Pro forms minimum-winning coalitions",
        fontsize=15,
        fontweight="bold",
        color="#17324D",
        ha="center",
        va="top",
    )
    ax.text(
        0.5,
        0.927,
        "Agents count the required votes, protect the coalition, and exclude the rest.",
        fontsize=9.5,
        color="#52606D",
        ha="center",
        va="top",
    )

    row_tops = [0.885, 0.615, 0.345]
    for case, top in zip(CASES, row_tops, strict=True):
        bottom = top - 0.225
        add_box(
            ax,
            0.018,
            bottom,
            0.964,
            0.215,
            facecolor="#F7F8FA",
            edgecolor="#D9DEE7",
            radius=0.018,
        )

        ax.text(
            0.045,
            top - 0.026,
            f"{case['panel']} {case['config']}",
            fontsize=10.5,
            fontweight="bold",
            color="#17324D",
            va="top",
        )
        ax.text(
            0.045,
            top - 0.073,
            case["rule"],
            fontsize=8.8,
            color="#52606D",
            va="top",
        )
        payoff_text = case["payoffs"]
        if case["config"] == "Config 0117":
            payoff_text = (
                "Payoffs\n[51.3, 0, 69.3, 59.4,\n50.4, 63.9, 0, 52.2]"
            )
        ax.text(
            0.045,
            top - 0.115,
            payoff_text,
            fontsize=7.2,
            color="#52606D",
            va="top",
            linespacing=1.2,
        )
        add_box(
            ax,
            0.045,
            bottom + 0.025,
            0.14,
            0.034,
            facecolor="#E9F7EF",
            edgecolor="#9CCFAF",
            radius=0.012,
        )
        ax.text(
            0.115,
            bottom + 0.042,
            case["result"],
            fontsize=8.2,
            fontweight="bold",
            color="#1F7A4D",
            ha="center",
            va="center",
        )

        add_box(
            ax,
            0.22,
            bottom + 0.025,
            0.45,
            0.155,
            facecolor="#E8F1FB",
            edgecolor="#8CB3D9",
            radius=0.018,
        )
        ax.text(
            0.245,
            top - 0.052,
            f"Coalition planning · {case['planner']}",
            fontsize=8.8,
            fontweight="bold",
            color="#255C8D",
            va="top",
        )
        ax.text(
            0.245,
            top - 0.087,
            textwrap.fill(f'“{case["plan"]}”', width=45),
            fontsize=7.75,
            color="#1F2933",
            va="top",
            linespacing=1.23,
        )

        ax.add_patch(
            FancyArrowPatch(
                (0.678, bottom + 0.112),
                (0.708, bottom + 0.112),
                arrowstyle="-|>",
                mutation_scale=10,
                linewidth=1.2,
                color="#9AA5B1",
            )
        )

        add_box(
            ax,
            0.72,
            bottom + 0.025,
            0.24,
            0.155,
            facecolor="#FDEBEC",
            edgecolor="#DFA0A4",
            radius=0.018,
        )
        ax.text(
            0.74,
            top - 0.052,
            (
                "Excluded votes"
                if case["config"] == "Config 0117"
                else f"Excluded vote · {case['excluded']}"
            ),
            fontsize=8.0,
            fontweight="bold",
            color="#A63D45",
            va="top",
        )
        if case["config"] == "Config 0117":
            reply = (
                "Agent 2: “This proposal gives me absolutely zero items, resulting "
                "in a utility of 0. I cannot accept it.”\n\nAgent 7: “This proposal "
                "gives me zero items, resulting in a utility of 0.”"
            )
            reply_size = 6.45
            reply_width = 31
        else:
            reply = f'“{case["reply"]}”'
            reply_size = 7.35
            reply_width = 31
        wrapped_reply = "\n".join(
            textwrap.fill(part, width=reply_width) for part in reply.split("\n")
        )
        ax.text(
            0.74,
            top - 0.087,
            wrapped_reply,
            fontsize=reply_size,
            color="#1F2933",
            va="top",
            linespacing=1.18,
        )

    add_box(
        ax,
        0.08,
        0.018,
        0.84,
        0.052,
        facecolor="#FFF7E6",
        edgecolor="#E7C77A",
        radius=0.014,
    )
    ax.text(
        0.5,
        0.044,
        "Observed strategy: satisfy exactly enough voters, then ignore outsider welfare.",
        fontsize=9.2,
        fontweight="bold",
        color="#7A5A13",
        ha="center",
        va="center",
    )

    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)
    for paper_dir in PAPER_DIRS:
        save_figure(fig, paper_dir / "gemini_minimum_winning_coalitions")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--render",
        choices=["variance", "quotes-v1", "quotes-v2", "all-v1", "all"],
        default="all",
    )
    args = parser.parse_args()

    if args.render in {"variance", "all-v1", "all"}:
        draw_variance_figure()
    if args.render in {"quotes-v1", "all-v1"}:
        draw_quote_figure_v1()
    if args.render in {"quotes-v2", "all"}:
        draw_quote_figure_v2()


if __name__ == "__main__":
    main()
