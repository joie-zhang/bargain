#!/usr/bin/env python3
"""
=============================================================================
Combined 3-panel Utility-vs-Elo Figure (Games 1, 2, 3)
=============================================================================

Reads the per-game utility-vs-elo summary CSVs and renders a single
1x3 figure: one panel per game with a scatter of (elo, avg_utility),
short-model annotations, and a dotted linear best-fit trendline.
Subplot titles are simply "Game 1", "Game 2", "Game 3".
Styling matches the standalone Game 1 plot. The three panels share a
uniform x-axis (Elo) and y-axis (Mean Adversary Utility) range.

Usage:
    python scripts/plot_combined_utility_vs_elo.py
    python scripts/plot_combined_utility_vs_elo.py --output Figures/cross_game/foo.png

What it creates:
    Figures/cross_game/
    └── utility_vs_elo_combined_3panel.png

Inputs (defaults — override via flags):
    --game1-csv  Figures/game_1/average_utility_vs_elo.csv
    --game2-csv  Figures/game_2/utility_vs_elo_overall.csv
    --game3-csv  Figures/game_3/utility_vs_elo_all_models.csv

Dependencies:
    matplotlib, numpy
    strong_models_experiment.analysis.active_model_roster.short_model_name
=============================================================================
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from strong_models_experiment.analysis.active_model_roster import short_model_name


SCATTER_COLOR = "#2563eb"
TREND_COLOR = "#1d4ed8"

PANEL_TITLE_SIZE = 20
AXIS_LABEL_SIZE = 15
TICK_SIZE = 13
ANNOT_SIZE = 11

X_PADDING = 25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--game1-csv",
        type=Path,
        default=PROJECT_ROOT / "Figures/game_1/average_utility_vs_elo.csv",
    )
    parser.add_argument(
        "--game2-csv",
        type=Path,
        default=PROJECT_ROOT / "Figures/game_2/utility_vs_elo_overall.csv",
    )
    parser.add_argument(
        "--game3-csv",
        type=Path,
        default=PROJECT_ROOT / "Figures/game_3/utility_vs_elo_all_models.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "Figures/cross_game/utility_vs_elo_combined_3panel.png",
    )
    return parser.parse_args()


def load_rows(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def extract_xy_labels(rows: List[Dict[str, str]]) -> Tuple[List[float], List[float], List[str]]:
    xs: List[float] = []
    ys: List[float] = []
    labels: List[str] = []
    for row in rows:
        xs.append(float(row["elo"]))
        ys.append(float(row["avg_utility"]))
        label = row.get("model_short") or short_model_name(row["model"])
        labels.append(label)
    order = sorted(range(len(xs)), key=lambda idx: xs[idx])
    return [xs[i] for i in order], [ys[i] for i in order], [labels[i] for i in order]


def best_fit_line(xs: List[float], ys: List[float]) -> Tuple[np.ndarray, np.ndarray]:
    if len(xs) < 2:
        return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
    slope, intercept = np.polyfit(xs, ys, deg=1)
    x_min, x_max = min(xs), max(xs)
    if x_min == x_max:
        return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
    line_x = np.linspace(x_min, x_max, 200)
    return line_x, slope * line_x + intercept


def draw_panel(ax, title: str, xs: List[float], ys: List[float], labels: List[str]) -> None:
    line_x, line_y = best_fit_line(xs, ys)
    ax.plot(
        line_x,
        line_y,
        color=TREND_COLOR,
        linewidth=2.2,
        alpha=0.9,
        linestyle=":",
        zorder=2,
    )

    for x, y, label in zip(xs, ys, labels):
        ax.scatter(x, y, s=110, color=SCATTER_COLOR, alpha=0.9, zorder=3)
        ax.annotate(
            label,
            (x, y),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=ANNOT_SIZE,
        )

    ax.set_title(title, fontsize=PANEL_TITLE_SIZE, fontweight="bold")
    ax.set_xlabel("Chatbot Arena Elo (adversary model)", fontsize=AXIS_LABEL_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_SIZE)
    ax.grid(True, alpha=0.25)
    ax.set_axisbelow(True)


def main() -> None:
    args = parse_args()

    panels = [
        ("Game 1", args.game1_csv),
        ("Game 2", args.game2_csv),
        ("Game 3", args.game3_csv),
    ]

    panel_data = []
    for title, csv_path in panels:
        rows = load_rows(csv_path)
        xs, ys, labels = extract_xy_labels(rows)
        panel_data.append((title, xs, ys, labels))

    all_xs = [x for _, xs, _, _ in panel_data for x in xs]
    x_lim = (min(all_xs) - X_PADDING, max(all_xs) + X_PADDING)

    fig, axes = plt.subplots(1, 3, figsize=(24, 8), sharey=False)

    for ax, (title, xs, ys, labels) in zip(axes, panel_data):
        draw_panel(ax, title, xs, ys, labels)
        ax.set_xlim(x_lim)
        ax.set_ylabel("Mean Adversary Utility", fontsize=AXIS_LABEL_SIZE)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote plot: {args.output}")


if __name__ == "__main__":
    main()
