#!/usr/bin/env python3
"""Render the paper's Appendix Figure 32 as one clean 3x3 PNG."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUMMARY = (
    PROJECT_ROOT
    / "experiments/results/n2_plus_multiagent_comparison_analysis_20260505"
    / "tables_multiagent/multiagent_fairness_by_n_summary.csv"
)
DEFAULT_HOMOGENEOUS_RUNS = (
    PROJECT_ROOT
    / "experiments/results/n2_plus_multiagent_comparison_analysis_20260505"
    / "tables_multiagent/homogeneous_runs_fresh.csv"
)
DEFAULT_HETEROGENEOUS_RUNS = (
    PROJECT_ROOT
    / "experiments/results/n2_plus_multiagent_comparison_analysis_20260505"
    / "tables_multiagent/heterogeneous_runs_fresh.csv"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "overleaf/icml_aiwild_template/graphics/n_gt_2_report"
    / "multiagent_fairness_inequality_efficiency_3x3.png"
)
DEFAULT_DATA_OUTPUT = (
    PROJECT_ROOT
    / "experiments/results/n2_plus_multiagent_comparison_analysis_20260505"
    / "plots_multiagent/multiagent_fairness_payoff_variance_efficiency_3x3.csv"
)

GAME_ORDER = ("game1", "game2", "game3")
GAME_TITLES = {
    "game1": "Game 1: Item Allocation",
    "game2": "Game 2: Diplomatic Treaty",
    "game3": "Game 3: Co-Funding",
}
N_ORDER = (2, 4, 6, 8, 10)
FAMILY_ORDER = (
    "homogeneous_control",
    "homogeneous_adversary",
    "heterogeneous_random",
)
FAMILY_LABELS = {
    "homogeneous_control": "Homogeneous control",
    "homogeneous_adversary": "Homogeneous adversary",
    "heterogeneous_random": "Heterogeneous",
}
FAMILY_COLORS = {
    "homogeneous_control": "#1f77b4",
    "homogeneous_adversary": "#d62728",
    "heterogeneous_random": "#2ca02c",
}
METRICS = (
    ("fairness_distance", "NBS/Lindahl Distance"),
    ("payoff_variance", "Payoff Variance"),
    ("sw_efficiency", "Social-Welfare Efficiency"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--homogeneous-runs", type=Path, default=DEFAULT_HOMOGENEOUS_RUNS)
    parser.add_argument("--heterogeneous-runs", type=Path, default=DEFAULT_HETEROGENEOUS_RUNS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--data-output", type=Path, default=DEFAULT_DATA_OUTPUT)
    return parser.parse_args()


def sem(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if len(clean) <= 1:
        return 0.0
    return float(clean.std(ddof=1) / np.sqrt(len(clean)))


def add_payoff_variance(
    summary: pd.DataFrame,
    homogeneous_path: Path,
    heterogeneous_path: Path,
) -> pd.DataFrame:
    runs = pd.concat(
        [pd.read_csv(homogeneous_path), pd.read_csv(heterogeneous_path)],
        ignore_index=True,
    )
    required = {"game_label", "n_agents", "experiment_family", "utility_variance", "result_path"}
    missing = sorted(required.difference(runs.columns))
    if missing:
        raise ValueError(f"run data is missing columns: {missing}")
    if len(runs) != 2730 or runs["result_path"].nunique() != 2730:
        raise ValueError("expected 2,730 unique multi-agent runs")

    variance = (
        runs.groupby(["game_label", "n_agents", "experiment_family"], as_index=False)
        .agg(
            payoff_variance=("utility_variance", "mean"),
            payoff_variance_count=("utility_variance", "count"),
            payoff_variance_sem=("utility_variance", sem),
        )
        .rename(columns={"game_label": "game_id"})
    )
    if len(variance) != 45:
        raise ValueError(f"expected 45 payoff-variance cells, found {len(variance)}")

    merged = summary.merge(
        variance,
        on=["game_id", "n_agents", "experiment_family"],
        how="left",
        validate="one_to_one",
    )
    if merged["payoff_variance"].isna().any():
        raise ValueError("payoff variance is missing for one or more plotting cells")
    if not merged["payoff_variance_count"].eq(merged["run_count"]).all():
        raise ValueError("payoff-variance counts do not match the source summary")
    return merged


def validate_summary(summary: pd.DataFrame) -> None:
    required = {
        "game_id",
        "n_agents",
        "experiment_family",
        *(metric for metric, _ in METRICS),
        *(f"{metric}_sem" for metric, _ in METRICS),
    }
    missing = sorted(required.difference(summary.columns))
    if missing:
        raise ValueError(f"summary is missing columns: {missing}")

    expected = {
        (game, n_agents, family)
        for game in GAME_ORDER
        for n_agents in N_ORDER
        for family in FAMILY_ORDER
    }
    observed = set(
        summary[["game_id", "n_agents", "experiment_family"]]
        .itertuples(index=False, name=None)
    )
    missing_cells = sorted(expected.difference(observed))
    if missing_cells:
        raise ValueError(f"summary is missing {len(missing_cells)} plotting cells")


def render(summary: pd.DataFrame, output: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.weight": "normal",
            "axes.titleweight": "normal",
            "axes.labelweight": "normal",
            "figure.titleweight": "normal",
        }
    )
    fig, axes = plt.subplots(3, 3, figsize=(18.0, 14.5), sharex="col")

    for row, (metric, y_label) in enumerate(METRICS):
        for column, game in enumerate(GAME_ORDER):
            ax = axes[row, column]
            game_df = summary.loc[summary["game_id"].eq(game)]

            for family in FAMILY_ORDER:
                sub = game_df.loc[
                    game_df["experiment_family"].eq(family)
                ].sort_values("n_agents")
                y_error = (
                    pd.to_numeric(sub[f"{metric}_sem"], errors="coerce")
                    .fillna(0.0)
                    .to_numpy(dtype=float)
                )
                ax.errorbar(
                    sub["n_agents"],
                    sub[metric],
                    yerr=np.maximum(y_error, 0.0),
                    color=FAMILY_COLORS[family],
                    marker="o",
                    markersize=8.2,
                    markeredgewidth=0.0,
                    linewidth=2.4,
                    capsize=4.0,
                    capthick=1.35,
                    elinewidth=1.35,
                    alpha=0.92,
                    label=FAMILY_LABELS[family],
                )

            if row == 0:
                ax.set_title(
                    GAME_TITLES[game],
                    fontsize=27,
                    fontweight="normal",
                    pad=13,
                )
            else:
                ax.set_title("")

            if row == len(METRICS) - 1:
                ax.set_xlabel("N agents", fontsize=25, fontweight="normal", labelpad=9)
            else:
                ax.set_xlabel("")

            if column == 0:
                ax.set_ylabel(y_label, fontsize=25, fontweight="normal", labelpad=8)
            else:
                ax.set_ylabel("")

            ax.set_xticks(N_ORDER)
            ax.tick_params(
                axis="x",
                labelbottom=row == len(METRICS) - 1,
                labelsize=20,
                width=1.2,
                length=5.0,
            )
            ax.tick_params(
                axis="y",
                labelleft=column == 0,
                labelsize=20,
                width=1.2,
                length=5.0,
            )
            ax.grid(True, alpha=0.22, linewidth=0.9)
            ax.margins(x=0.07)
            for spine in ax.spines.values():
                spine.set_linewidth(1.15)
                spine.set_color("#555555")

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=FAMILY_COLORS[family],
            marker="o",
            markersize=8.2,
            markeredgewidth=0.0,
            linewidth=2.4,
            label=FAMILY_LABELS[family],
        )
        for family in FAMILY_ORDER
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.004),
        ncol=3,
        fontsize=23,
        frameon=False,
        handlelength=3.0,
        handletextpad=0.8,
        columnspacing=2.2,
    )
    fig.subplots_adjust(
        left=0.095,
        right=0.995,
        top=0.972,
        bottom=0.135,
        hspace=0.105,
        wspace=0.085,
    )

    # The paper request explicitly forbids bold text anywhere in the plot.
    fig.canvas.draw()
    bold_text = [
        text.get_text()
        for text in fig.findobj(match=matplotlib.text.Text)
        if text.get_text() and str(text.get_fontweight()).lower() not in {"normal", "400"}
    ]
    if bold_text:
        raise RuntimeError(f"unexpected bold text in figure: {bold_text}")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, facecolor="white", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    summary = pd.read_csv(args.summary)
    summary = add_payoff_variance(
        summary,
        args.homogeneous_runs,
        args.heterogeneous_runs,
    )
    validate_summary(summary)
    render(summary, args.output)
    args.data_output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.data_output, index=False)
    print(f"wrote {args.output.resolve()} from {len(summary)} summary cells")


if __name__ == "__main__":
    main()
