#!/usr/bin/env python3
"""Render the compact main-text figure for the Game 1 coordination stress test."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_RESULTS_ROOT = (
    REPO_ROOT
    / "experiments/results/game1_gpt54_binding_team_v3_20260816_093310"
)

from scripts import analyze_game1_gpt54_team_coordination as cohort_analysis  # noqa: E402
from scripts.analyze_game1_gpt54_binding_team_dynamics import (  # noqa: E402
    allocation_metrics,
)


N_VALUES = (2, 4, 6, 8, 10)
TEAM_SIZES = tuple(n_agents - 1 for n_agents in N_VALUES)
PERCENT_SCALE = 100.0
GAP_Y_LIMITS = (-65.0, 20.0)
LABEL_FONTSIZE = 18
TICK_FONTSIZE = 15
MARKER_SIZE = 10
LINE_WIDTH = 2.2
ERRORBAR_LINE_WIDTH = 1.5
ERRORBAR_CAP_SIZE = 4.5
AXIS_LINE_WIDTH = ERRORBAR_LINE_WIDTH
TICK_LENGTH = 5.0
SERIES_ALPHA = 0.72
GRID_COLOR = "#D1D5DB"
GRID_ALPHA = 0.52
GRID_LINE_WIDTH = 0.75


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help="Experiment directory containing configs/ and runs/.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def mean_and_sem(values: list[float]) -> tuple[float, float]:
    if len(values) < 2:
        raise ValueError(f"Expected at least two values, found {len(values)}")
    return statistics.mean(values), statistics.stdev(values) / math.sqrt(len(values))


def load_cohort(
    results_root: Path,
) -> tuple[list[dict[str, float]], list[dict[str, float]], dict[str, Any]]:
    results_root = results_root.resolve()
    run_rows, pair_rows, validation = cohort_analysis.load_rows(
        results_root,
        allow_partial=False,
    )
    treatment_rows = [row for row in run_rows if row["condition"] == "team"]
    if len(treatment_rows) != 100 or len(pair_rows) != 100:
        raise ValueError(
            f"Expected 100 treatment rows and pairs, found "
            f"{len(treatment_rows)} rows and {len(pair_rows)} pairs"
        )
    for key in (
        "control_hash_mismatches",
        "factor_mismatches",
        "missing_treatment_results",
        "preference_mismatches",
        "treatment_model_route_mismatches",
    ):
        if validation.get(key):
            raise ValueError(f"Validation failed for {key}: {validation[key]}")

    estimates: list[dict[str, float]] = []
    for n_agents in N_VALUES:
        selected = [
            row for row in treatment_rows if int(row["n_agents"]) == n_agents
        ]
        if len(selected) != 20:
            raise ValueError(f"Expected 20 treatment rows at N={n_agents}")
        gaps = [float(row["adversary_minus_nano_gap"]) for row in selected]
        mean, sem = mean_and_sem(gaps)
        estimates.append({"mean": mean, "sem": sem})

    exact_by_n = {n_agents: 0 for n_agents in N_VALUES}
    runs_by_n = {n_agents: 0 for n_agents in N_VALUES}
    config_paths = sorted((results_root / "configs").glob("config_*.json"))
    if len(config_paths) != 100:
        raise ValueError(f"Expected 100 configs, found {len(config_paths)}")
    for config_path in config_paths:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        n_agents = int(config["n_agents"])
        result_path = Path(config["output_dir"]) / "experiment_results.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        metrics = allocation_metrics(result, config["agent_role_map"])
        exact = math.isclose(
            float(metrics["team_discounted_efficiency"]),
            1.0,
            abs_tol=1e-9,
        )
        runs_by_n[n_agents] += 1
        exact_by_n[n_agents] += int(exact)

    optimality: list[dict[str, float]] = []
    for n_agents in N_VALUES:
        runs = runs_by_n[n_agents]
        if runs != 20:
            raise ValueError(f"Expected 20 raw results at N={n_agents}, found {runs}")
        proportion = exact_by_n[n_agents] / runs
        optimality.append({
            "team_size": float(n_agents - 1),
            "percent": PERCENT_SCALE * proportion,
            "sem": PERCENT_SCALE
            * math.sqrt(proportion * (1.0 - proportion) / (runs - 1.0)),
        })

    provenance = {
        "results_root": str(results_root),
        "source": "100 saved treatment experiment_results.json files",
        "gap_definition": "GPT-5.4 payoff minus mean per-agent Nano payoff",
        "optimality_definition": (
            "discounted Nano-team payoff equals the maximum available in round 1"
        ),
        "n_values": list(N_VALUES),
        "runs_per_n": [runs_by_n[n_agents] for n_agents in N_VALUES],
        "gap_mean": [row["mean"] for row in estimates],
        "gap_sem": [row["sem"] for row in estimates],
        "exact_round1_ceiling_runs": [
            exact_by_n[n_agents] for n_agents in N_VALUES
        ],
        "validation_loaded_pairs": validation.get("loaded_pairs"),
    }
    return estimates, optimality, provenance


def draw_panel(
    ax: plt.Axes,
    estimates: list[dict[str, float]],
    *,
    color: str,
    ylabel: str,
    marker: str,
) -> None:
    means = [row["mean"] for row in estimates]
    sems = [row["sem"] for row in estimates]
    ax.errorbar(
        TEAM_SIZES,
        means,
        yerr=sems,
        fmt="none",
        ecolor=color,
        elinewidth=ERRORBAR_LINE_WIDTH,
        capsize=ERRORBAR_CAP_SIZE,
        capthick=ERRORBAR_LINE_WIDTH,
        alpha=0.88,
        zorder=2,
    )
    ax.plot(
        TEAM_SIZES,
        means,
        color=color,
        marker=marker,
        markersize=MARKER_SIZE,
        linewidth=LINE_WIDTH,
        alpha=SERIES_ALPHA,
        zorder=3,
    )
    ax.axhline(0, color="#475569", linewidth=1.2, linestyle="--", alpha=0.8)
    ax.set_ylabel(ylabel, fontsize=LABEL_FONTSIZE, labelpad=8)
    ax.set_xticks(TEAM_SIZES)
    ax.set_xlim(0.45, 9.55)
    ax.tick_params(
        axis="both",
        labelsize=TICK_FONTSIZE,
        length=TICK_LENGTH,
        width=AXIS_LINE_WIDTH,
    )
    ax.grid(
        axis="y",
        color=GRID_COLOR,
        linewidth=GRID_LINE_WIDTH,
        alpha=GRID_ALPHA,
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["bottom", "left"]].set_linewidth(AXIS_LINE_WIDTH)


def draw_optimality_panel(
    ax: plt.Axes,
    optimality: list[dict[str, float]],
) -> None:
    team_sizes = [row["team_size"] for row in optimality]
    percentages = [row["percent"] for row in optimality]
    sems = [row["sem"] for row in optimality]
    ax.errorbar(
        team_sizes,
        percentages,
        yerr=sems,
        fmt="none",
        ecolor="#6D28D9",
        elinewidth=ERRORBAR_LINE_WIDTH,
        capsize=ERRORBAR_CAP_SIZE,
        capthick=ERRORBAR_LINE_WIDTH,
        alpha=0.88,
        zorder=2,
    )
    ax.plot(
        team_sizes,
        percentages,
        color="#7C3AED",
        marker="D",
        markersize=MARKER_SIZE,
        linewidth=LINE_WIDTH,
        alpha=SERIES_ALPHA,
        zorder=3,
    )
    ax.set_ylabel(
        "% of games where outcome\nwas optimal for Nano team",
        fontsize=LABEL_FONTSIZE,
        labelpad=8,
    )
    ax.set_xlabel(
        "Number of GPT-5-Nano agents",
        fontsize=LABEL_FONTSIZE,
        labelpad=6,
    )
    ax.set_xticks(team_sizes)
    ax.set_xlim(0.45, 9.55)
    ax.set_ylim(0, 112)
    ax.set_yticks(range(0, 101, 20))
    ax.tick_params(
        axis="both",
        labelsize=TICK_FONTSIZE,
        length=TICK_LENGTH,
        width=AXIS_LINE_WIDTH,
    )
    ax.grid(
        axis="y",
        color=GRID_COLOR,
        linewidth=GRID_LINE_WIDTH,
        alpha=GRID_ALPHA,
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["bottom", "left"]].set_linewidth(AXIS_LINE_WIDTH)


def main() -> None:
    args = parse_args()
    coordinated, optimality, provenance = load_cohort(args.results_root)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.unicode_minus": True,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.65))
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.19, top=0.97, wspace=0.38)
    draw_panel(
        axes[0],
        coordinated,
        color="#2563EB",
        marker="o",
        ylabel="GPT-5.4 Payoff -\nMean Nano Payoff",
    )
    draw_optimality_panel(axes[1], optimality)
    axes[0].set_ylim(*GAP_Y_LIMITS)
    axes[0].set_xlabel(
        "Number of GPT-5-Nano agents",
        fontsize=LABEL_FONTSIZE,
        labelpad=6,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(fig)
    provenance_path = args.output.with_name(
        args.output.stem + "_provenance.json"
    )
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
