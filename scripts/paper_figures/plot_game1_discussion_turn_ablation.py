#!/usr/bin/env python3
"""Render the matched one-turn vs. two-turn Game 1 protocol comparison.

This is a Figure 2a-style adaptation of ``render_figure2_large_fonts.py``.
Each point is an adversary-model mean over the seven competition levels and
both model orders; error bars are SEM over those 14 runs. Dashed/dotted lines
are linear fits through the 30 model-level means in each protocol arm.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator


ROOT = Path(__file__).resolve().parents[2]
INPUT_CSV = ROOT / "experiments/results/n2_baseline_comparison_analysis_20260505/all_runs_with_metrics.csv"
OUTPUT_DIR = ROOT / "overleaf/icml_aiwild_template/graphics/n2_gpt5_nano"
OUTPUT_PNG = OUTPUT_DIR / "game1_discussion_turn_ablation.png"
OUTPUT_SUMMARY = OUTPUT_DIR / "game1_discussion_turn_ablation_summary.csv"
PROVENANCE_PATH = OUTPUT_PNG.with_name("game1_discussion_turn_ablation_provenance.json")

TURN_ORDER = [1, 2]
TURN_STYLES = {
    1: {
        "label": "One discussion turn",
        "color": "#6b7280",
        "marker": "o",
        "linestyle": ":",
    },
    2: {
        "label": "Two discussion turns",
        "color": "#2b7bba",
        "marker": "s",
        "linestyle": "--",
    },
}
EXPECTED_RUN_COUNTS = {1: 420, 2: 420}
EXPECTED_MODELS = 30
EXPECTED_COMPETITION_LEVELS = {0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0}
DESIGN_COLUMNS = ["adversary_model", "adversary_elo", "competition_value", "model_order"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sem(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if len(clean) <= 1:
        return 0.0
    return float(clean.std(ddof=1) / np.sqrt(len(clean)))


def fit_line(x: pd.Series, y: pd.Series) -> tuple[np.ndarray, np.ndarray, float]:
    x_arr = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    y_arr = pd.to_numeric(y, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    if mask.sum() < 2:
        return np.array([]), np.array([]), float("nan")
    slope, intercept = np.polyfit(x_arr[mask], y_arr[mask], deg=1)
    xs = np.linspace(float(x_arr[mask].min()), float(x_arr[mask].max()), 100)
    return xs, slope * xs + intercept, float(slope)


def select_rows(df: pd.DataFrame) -> pd.DataFrame:
    selected = df[
        df["baseline_key"].eq("gpt5_nano")
        & df["game_id"].eq("game1")
        & df["discussion_turns"].isin(TURN_ORDER)
    ].copy()
    selected = selected.replace([np.inf, -np.inf], np.nan)
    required = [
        "experiment_id",
        "result_path",
        "discussion_turns",
        *DESIGN_COLUMNS,
        "adversary_utility",
    ]
    return selected.dropna(subset=required)


def validate_input(df: pd.DataFrame) -> None:
    counts = {int(turns): int(count) for turns, count in df.groupby("discussion_turns").size().items()}
    if counts != EXPECTED_RUN_COUNTS:
        raise RuntimeError(f"Expected Game 1 run counts {EXPECTED_RUN_COUNTS}, found {counts}")
    if len(df) != 840 or df["result_path"].nunique() != 840:
        raise RuntimeError("The discussion-turn comparison requires 840 unique result files")
    if df["adversary_model"].astype(str).str.contains("phi", case=False).any():
        raise RuntimeError("Phi rows remain in the discussion-turn comparison")

    for turns in TURN_ORDER:
        arm = df[df["discussion_turns"].eq(turns)]
        if arm["adversary_model"].nunique() != EXPECTED_MODELS:
            raise RuntimeError(
                f"Expected {EXPECTED_MODELS} adversary models for {turns} turn(s), "
                f"found {arm['adversary_model'].nunique()}"
            )
        competition_levels = set(pd.to_numeric(arm["competition_value"]))
        if competition_levels != EXPECTED_COMPETITION_LEVELS:
            raise RuntimeError(
                f"Unexpected competition levels for {turns} turn(s): {sorted(competition_levels)}"
            )
        if set(arm["model_order"].astype(str)) != {"weak_first", "strong_first"}:
            raise RuntimeError(f"Both model orders are required for {turns} turn(s)")
        if arm.duplicated(DESIGN_COLUMNS).any():
            raise RuntimeError(f"Duplicate design cells found for {turns} turn(s)")

    one_turn_cells = {
        tuple(row) for row in df[df["discussion_turns"].eq(1)][DESIGN_COLUMNS].itertuples(index=False, name=None)
    }
    two_turn_cells = {
        tuple(row) for row in df[df["discussion_turns"].eq(2)][DESIGN_COLUMNS].itertuples(index=False, name=None)
    }
    if one_turn_cells != two_turn_cells:
        raise RuntimeError("The one-turn and two-turn arms do not have identical design support")

    missing = [path for path in df["result_path"].astype(str) if not (ROOT / path).is_file()]
    if missing:
        raise RuntimeError(f"Missing {len(missing)} result files; first: {missing[0]}")


def model_means(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["discussion_turns", "adversary_model", "adversary_elo"], as_index=False)
        .agg(mean=("adversary_utility", "mean"), err=("adversary_utility", sem), runs=("experiment_id", "size"))
        .sort_values(["discussion_turns", "adversary_elo", "adversary_model"])
    )


def draw_plot(ax: plt.Axes, means: pd.DataFrame) -> dict[int, float]:
    slopes: dict[int, float] = {}
    for turns in TURN_ORDER:
        style = TURN_STYLES[turns]
        arm = means[means["discussion_turns"].eq(turns)]
        ax.errorbar(
            arm["adversary_elo"],
            arm["mean"],
            yerr=arm["err"],
            fmt=style["marker"],
            markersize=4.7,
            markerfacecolor="white",
            markeredgewidth=1.05,
            color=style["color"],
            ecolor=style["color"],
            elinewidth=0.9,
            capsize=2.2,
            alpha=0.72,
            zorder=3,
        )
        xs, ys, slope = fit_line(arm["adversary_elo"], arm["mean"])
        slopes[turns] = slope
        ax.plot(
            xs,
            ys,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=2.8,
            zorder=4,
        )

    ax.set_xlabel("Adversary Elo", fontsize=18, labelpad=6)
    ax.set_ylabel("Adversary payoff", fontsize=18, labelpad=8)
    ax.tick_params(axis="both", labelsize=14, length=4.5, width=0.8)
    ax.grid(True, color="#d1d5db", alpha=0.52, linewidth=0.75)
    ax.xaxis.set_major_locator(MultipleLocator(100))
    ax.set_xlim(1088, 1512)
    ax.set_ylim(-10, 102)
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)
    return slopes


def write_summary(df: pd.DataFrame, means: pd.DataFrame, slopes: dict[int, float]) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for turns in TURN_ORDER:
        arm = df[df["discussion_turns"].eq(turns)]
        arm_means = means[means["discussion_turns"].eq(turns)]
        rows.append(
            {
                "discussion_turns": turns,
                "label": TURN_STYLES[turns]["label"],
                "completed_runs": len(arm),
                "adversary_models": arm["adversary_model"].nunique(),
                "runs_per_model": int(arm.groupby("adversary_model").size().iloc[0]),
                "mean_adversary_payoff": arm["adversary_utility"].mean(),
                "sem_across_model_means": sem(arm_means["mean"]),
                "payoff_slope_per_100_elo": 100.0 * slopes[turns],
            }
        )
    summary = pd.DataFrame(rows)
    OUTPUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_SUMMARY, index=False)
    return summary


def main() -> None:
    df = select_rows(pd.read_csv(INPUT_CSV))
    validate_input(df)
    means = model_means(df)
    if not means["runs"].eq(14).all():
        raise RuntimeError("Every plotted model mean must aggregate exactly 14 runs")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.unicode_minus": False,
            "mathtext.fontset": "dejavusans",
        }
    )
    fig = plt.figure(figsize=(6.8, 5.68), dpi=300)
    ax = fig.add_axes([0.14, 0.26, 0.83, 0.70])
    slopes = draw_plot(ax, means)
    handles = [
        Line2D(
            [0],
            [0],
            color=TURN_STYLES[turns]["color"],
            marker=TURN_STYLES[turns]["marker"],
            markerfacecolor="white",
            linestyle=TURN_STYLES[turns]["linestyle"],
            linewidth=2.8,
            label=TURN_STYLES[turns]["label"],
        )
        for turns in TURN_ORDER
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.55, 0.065),
        ncol=2,
        fontsize=13.5,
        frameon=False,
        handlelength=1.65,
        columnspacing=1.4,
        handletextpad=0.5,
    )

    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=300, facecolor="white")
    plt.close(fig)
    summary = write_summary(df, means, slopes)
    PROVENANCE_PATH.write_text(
        json.dumps(
            {
                "producer": str(Path(__file__).resolve().relative_to(ROOT)),
                "adapted_from": "scripts/paper_figures/render_figure2_large_fonts.py",
                "input": str(INPUT_CSV.relative_to(ROOT)),
                "input_sha256": sha256(INPUT_CSV),
                "filters": {
                    "baseline_key": "gpt5_nano",
                    "game_id": "game1",
                    "discussion_turns": TURN_ORDER,
                },
                "run_counts": EXPECTED_RUN_COUNTS,
                "adversary_model_count_per_arm": EXPECTED_MODELS,
                "competition_levels": sorted(EXPECTED_COMPETITION_LEVELS),
                "model_orders": ["weak_first", "strong_first"],
                "matched_design": True,
                "aggregation": "adversary-model means over 7 competition levels x 2 model orders",
                "error_bars": "SEM over 14 runs within each adversary-model mean",
                "fit": "OLS linear fit through the 30 model-level means in each arm",
                "output": str(OUTPUT_PNG.relative_to(ROOT)),
                "output_sha256": sha256(OUTPUT_PNG),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(OUTPUT_PNG)
    print(OUTPUT_SUMMARY)
    print(PROVENANCE_PATH)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
