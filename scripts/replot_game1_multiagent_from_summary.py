#!/usr/bin/env python3
"""Replot Game 1 multi-agent advantage-vs-Elo figures from summary_by_model.csv."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import List

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from strong_models_experiment.analysis.game1_multiagent_smoke import (
    PROPOSAL_LABELS,
    identify_gemini_low_pair_rows,
    pair_count_column,
    plot_advantage_vs_elo,
)


DEFAULT_SUMMARY_CSV = Path("experiments/results/game1_multiagent_full_latest/analysis/summary_by_model.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replot Proposal 1/2 advantage-vs-Elo using summary_by_model.csv, with run-count labels "
            "and optional Gemini low-sample filtering."
        )
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=DEFAULT_SUMMARY_CSV,
        help="Path to summary_by_model.csv (default: game1_multiagent_full_latest analysis output).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for generated plots/tables (default: summary csv parent).",
    )
    parser.add_argument(
        "--proposal",
        choices=["proposal1_invasion", "proposal2_mixed_ecology", "all"],
        default="proposal1_invasion",
        help="Which proposal family to plot.",
    )
    parser.add_argument(
        "--gemini-min-pairs",
        type=int,
        default=None,
        help="If set, also emit a filtered plot dropping Gemini points with pair count below this threshold.",
    )
    parser.add_argument(
        "--annotate-runs",
        dest="annotate_runs",
        action="store_true",
        help="Append run count as '(r=...)' to each point label.",
    )
    parser.add_argument(
        "--no-annotate-runs",
        dest="annotate_runs",
        action="store_false",
        help="Disable run-count point-label annotation.",
    )
    parser.set_defaults(annotate_runs=True)
    return parser.parse_args()


def resolve_proposals(raw: str) -> List[str]:
    if raw == "all":
        return ["proposal1_invasion", "proposal2_mixed_ecology"]
    return [raw]


def main() -> int:
    args = parse_args()
    summary_csv = args.summary_csv.resolve()
    if not summary_csv.exists():
        raise FileNotFoundError(f"Missing summary csv: {summary_csv}")

    output_dir = args.output_dir.resolve() if args.output_dir else summary_csv.parent.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(summary_csv)
    proposals = resolve_proposals(args.proposal)

    count_col = pair_count_column(summary)
    if count_col is None and args.annotate_runs:
        print("warning: no pair-count column found; run-count annotations disabled.")
        args.annotate_runs = False

    for proposal in proposals:
        proposal_df = summary[summary["proposal_family"] == proposal].copy()
        if proposal_df.empty:
            print(f"skipping_empty_proposal: {proposal}")
            continue

        title = PROPOSAL_LABELS.get(proposal, proposal)
        stem = f"{proposal}_advantage_vs_elo"
        run_label_suffix = "with_run_counts" if args.annotate_runs else "default_labels"

        full_plot = output_dir / f"{stem}_{run_label_suffix}.png"
        plot_advantage_vs_elo(
            proposal_df,
            full_plot,
            title,
            annotate_pair_counts=args.annotate_runs,
        )
        print(f"wrote_plot: {full_plot}")

        if args.gemini_min_pairs is None:
            continue

        filtered_plot = output_dir / (
            f"{stem}_{run_label_suffix}_gemini_min_pairs_{int(args.gemini_min_pairs)}.png"
        )
        plot_advantage_vs_elo(
            proposal_df,
            filtered_plot,
            title,
            annotate_pair_counts=args.annotate_runs,
            gemini_min_pairs=args.gemini_min_pairs,
        )
        print(f"wrote_filtered_plot: {filtered_plot}")

        flagged = identify_gemini_low_pair_rows(proposal_df, args.gemini_min_pairs)
        if flagged.empty:
            print(
                f"flagged_gemini_low_pairs: 0 (threshold={int(args.gemini_min_pairs)}) for {proposal}"
            )
            continue

        key_cols = [
            "proposal_family",
            "short_name",
            "focal_model",
            "elo",
            "n_agents",
            "competition_level",
            "mean_utility_advantage_vs_nano",
        ]
        if count_col is not None and count_col in flagged.columns:
            key_cols.insert(6, count_col)
        key_cols = [col for col in key_cols if col in flagged.columns]
        flagged_table = flagged[key_cols].copy().sort_values(
            ["n_agents", "competition_level", "elo"]
        )

        flagged_csv = output_dir / f"{proposal}_gemini_points_below_{int(args.gemini_min_pairs)}_pairs.csv"
        flagged_table.to_csv(flagged_csv, index=False)
        print(f"flagged_gemini_low_pairs: {len(flagged_table)}")
        print(f"wrote_flagged_csv: {flagged_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
