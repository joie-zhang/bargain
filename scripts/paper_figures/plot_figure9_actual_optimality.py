#!/usr/bin/env python3
"""Plot observed exact Nano-team optimality in both Figure 9 conditions."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
import sys

if __package__:
    from .plot_figure9_absolute_payoffs import (
        COLORS, CONDITIONS, LABELS, NS, ROOT, add_input_arguments, digest, load_cohort, require,
    )
else:
    from plot_figure9_absolute_payoffs import (
        COLORS, CONDITIONS, LABELS, NS, ROOT, add_input_arguments, digest, load_cohort, require,
    )
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    """Use the same 95% Wilson interval as the original optimality panel."""
    require(0 <= successes <= total and total > 0, "Invalid success count")
    z = 1.96
    proportion = successes / total
    denominator = 1 + z**2 / total
    center = (proportion + z**2 / (2 * total)) / denominator
    half_width = z * np.sqrt(
        (proportion * (1 - proportion) + z**2 / (4 * total)) / total
    ) / denominator
    return 100 * (center - half_width), 100 * (center + half_width)


def summarize_optimality(frame: pd.DataFrame) -> pd.DataFrame:
    """Count observed exact optima independently in each condition."""
    require(set(frame.condition) == set(CONDITIONS), "Expected both coordination conditions")
    require(set(frame.n_agents) == set(NS), "Unexpected group sizes")
    require(frame.nano_exact_maximum.notna().all() and
            pd.api.types.is_bool_dtype(frame.nano_exact_maximum), "Optimality values must be booleans")
    rows = []
    for condition in CONDITIONS:
        for n in NS:
            cell = frame[frame.condition.eq(condition) & frame.n_agents.eq(n)]
            require(len(cell) == 20, f"Expected 20 runs: {condition}, n={n}")
            successes = int(cell.nano_exact_maximum.sum())
            low, high = wilson_interval(successes, len(cell))
            rows.append({
                "condition": condition, "n_agents": n, "nano_agents": n - 1,
                "optimal_runs": successes, "runs": len(cell),
                "optimal_percent": 100 * successes / len(cell),
                "ci95_low": low, "ci95_high": high,
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_input_arguments(parser)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    require(not out.is_relative_to(ROOT / "overleaf"), "Use an output directory outside the paper folder")

    frame, audit = load_cohort(args.results_root, data_root=args.data_root)
    summary = summarize_optimality(frame)
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "run_level.csv", index=False)
    summary.to_csv(out / "optimality_summary.csv", index=False)
    breakdown = frame.groupby(
        ["condition", "n_agents", "competition_level"], as_index=False,
    ).agg(optimal_runs=("nano_exact_maximum", "sum"), runs=("config_id", "size"))
    breakdown["optimal_percent"] = 100 * breakdown.optimal_runs / breakdown.runs
    breakdown.to_csv(out / "optimality_by_competition.csv", index=False)

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 14,
        "axes.labelsize": 18, "xtick.labelsize": 14, "ytick.labelsize": 14,
        "axes.linewidth": 1.3, "pdf.fonttype": 42,
    })
    figure, axis = plt.subplots(figsize=(6.8, 4.8))
    for condition in CONDITIONS:
        data = summary[summary.condition.eq(condition)].sort_values("n_agents")
        mean = data.optimal_percent.to_numpy()
        axis.errorbar(
            data.n_agents, mean,
            yerr=np.vstack([mean - data.ci95_low, data.ci95_high - mean]),
            color=COLORS[condition], ecolor=COLORS[condition],
            marker="s" if condition == "coordinated" else "o",
            linewidth=2.1, elinewidth=1.3, capsize=3.2,
            markersize=7, alpha=.84, label=LABELS[condition],
        )
    axis.set(
        xlabel="Total number of agents, $n$",
        ylabel="Runs attaining Nano team\nmaximum payoff (%)",
        xticks=NS, yticks=np.arange(0, 101, 20),
        xlim=(1.55, 10.45), ylim=(-2, 105),
    )
    axis.grid(True, color="#D1D5DB", alpha=.52, linewidth=.75)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(width=1.3, length=4.5)
    axis.legend(
        loc="lower center", bbox_to_anchor=(.5, 1.01),
        frameon=False, fontsize=13, handlelength=2.3,
    )
    figure.subplots_adjust(left=.17, right=.99, bottom=.16, top=.82)
    for suffix in ["pdf", "png"]:
        figure.savefig(
            out / f"nano_actual_optimality.{suffix}", dpi=320,
            bbox_inches="tight", pad_inches=.035,
        )
    plt.close(figure)

    caption = (
        "Observed Nano-team optimality with and without coordination in Game 1. "
        "The horizontal axis is total agents n: one GPT-5.4 High adversary model "
        "and n-1 GPT-5-nano agents. The vertical axis is the percentage of 20 "
        "runs per condition and n whose total discounted Nano payoff equals "
        "the maximum possible round-one Nano payoff (absolute tolerance 1e-8). "
        "The maximum assigns each item to the Nano agent who values it most. "
        "Both conditions use this same outcome criterion, whether or not private "
        "team coordination was available. This is the frequency of reaching the "
        "maximum, not the mean percentage of maximum payoff attained. "
        "Error bars are pointwise 95% Wilson intervals, matching the original "
        "optimality panel's method; these summarize the pooled design descriptively. "
        "Each point pools five competition levels, two adversary positions, "
        "and two seeds. At n=2 neither condition adds team coordination; "
        "the two conditions are separate historical and later runs. "
        "Historical controls use mixed model routes, while the coordinated arm "
        "uses direct OpenAI. Increasing n also changes the number of items. "
        "The older optimality renderer assigned zero to the no-coordination "
        "curve; this plot computes both curves from observed outcomes. "
        "It does not replace the current paper's continuous payoff-percentage figure.\n"
    )
    (out / "caption.txt").write_text(caption)
    script = Path(__file__).resolve()
    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join([sys.executable, str(script), *sys.argv[1:]]),
        "repository_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
        ).strip(),
        "script_sha256": digest(script),
        "cohort_loader_sha256": digest(script.with_name("plot_figure9_absolute_payoffs.py")),
        "results_root": audit["results_root"],
        "relocated_data_root": audit["relocated_data_root"],
        "input_sha256": audit["input_sha256"],
        "validation": audit["checks"] + " Both optimality curves are computed from the verified outcomes.",
        "observed_adversary_routes": audit["observed_adversary_routes"],
        "definition_and_methods": caption,
        "outputs_sha256": {
            name: digest(out / name) for name in [
                "run_level.csv", "optimality_summary.csv", "optimality_by_competition.csv",
                "nano_actual_optimality.pdf", "nano_actual_optimality.png", "caption.txt",
            ]
        },
    }
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(summary.to_string(index=False))
    print(f"Saved plot to {out / 'nano_actual_optimality.pdf'}")


if __name__ == "__main__":
    main()
