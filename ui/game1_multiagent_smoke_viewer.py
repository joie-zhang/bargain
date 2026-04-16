"""
Streamlit viewer for Game 1 multi-agent smoke batches.

Usage:
    streamlit run ui/game1_multiagent_smoke_viewer.py
    streamlit run ui/game1_multiagent_smoke_viewer.py -- --results-root <dir>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from strong_models_experiment.analysis.game1_multiagent_smoke import (  # noqa: E402
    PROPOSAL_LABELS,
    aggregate_paired_rows,
    collect_condition_rows,
    collect_paired_rows,
    completion_summary,
    completion_table_rows,
    load_manifest,
    resolve_results_root,
    write_analysis_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--results-root", type=str, default=None)
    args, _ = parser.parse_known_args(sys.argv[1:])
    return args


def sidebar_filters(summary: pd.DataFrame) -> tuple[Optional[str], Optional[int], Optional[float]]:
    proposal = st.sidebar.selectbox(
        "Proposal",
        options=[None] + sorted(summary["proposal_family"].unique().tolist()),
        format_func=lambda value: "All" if value is None else PROPOSAL_LABELS.get(value, value),
    )
    n_agents = st.sidebar.selectbox(
        "n",
        options=[None] + sorted(summary["n_agents"].unique().tolist()),
        format_func=lambda value: "All" if value is None else str(value),
    )
    competition = st.sidebar.selectbox(
        "Competition",
        options=[None] + sorted(summary["competition_level"].unique().tolist()),
        format_func=lambda value: "All" if value is None else f"{value:g}",
    )
    return proposal, n_agents, competition


def apply_filters(frame: pd.DataFrame, proposal: Optional[str], n_agents: Optional[int], competition: Optional[float]) -> pd.DataFrame:
    filtered = frame.copy()
    if proposal is not None:
        filtered = filtered[filtered["proposal_family"] == proposal]
    if n_agents is not None:
        filtered = filtered[filtered["n_agents"] == n_agents]
    if competition is not None:
        filtered = filtered[filtered["competition_level"] == competition]
    return filtered.reset_index(drop=True)


def main() -> None:
    args = parse_args()
    results_root = resolve_results_root(args.results_root)

    st.set_page_config(
        page_title="Game 1 Multi-Agent Smoke Viewer",
        page_icon="S",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.sidebar.title("Game 1 Smoke")
    st.sidebar.code(str(results_root))

    if st.sidebar.button("Refresh Analysis", use_container_width=True):
        write_analysis_outputs(results_root)

    manifest = load_manifest(results_root)
    conditions = collect_condition_rows(results_root)
    paired = collect_paired_rows(results_root)
    summary = aggregate_paired_rows(paired)
    completion = completion_summary(conditions)

    st.title("Game 1 Multi-Agent Smoke")
    st.caption(f"`{results_root}`")

    top = st.columns(4)
    top[0].metric("Total Conditions", completion["total"])
    top[1].metric("Successful", completion["successful"])
    top[2].metric("In Progress", completion["in_progress"])
    top[3].metric("Not Started", completion["not_started"])

    manifest_rows = [
        {"name": "baseline_model", "value": manifest["baseline_model"]},
        {"name": "focal_models", "value": ", ".join(manifest["focal_models"])},
        {"name": "n_values", "value": ", ".join(str(v) for v in manifest["n_values"])},
        {"name": "competition_levels", "value": ", ".join(str(v) for v in manifest["competition_levels"])},
        {"name": "proposal1_reps", "value": manifest["proposal1_reps"]},
        {"name": "proposal2_fields", "value": manifest["proposal2_fields"]},
        {"name": "discussion_turns", "value": manifest["discussion_turns"]},
        {"name": "num_items", "value": manifest["num_items"]},
        {"name": "max_rounds", "value": manifest["max_rounds"]},
    ]
    st.subheader("Batch Definition")
    st.dataframe(pd.DataFrame(manifest_rows), hide_index=True, use_container_width=True)

    st.subheader("Completion By Slice")
    completion_rows = completion_table_rows(conditions)
    if completion_rows:
        st.dataframe(pd.DataFrame(completion_rows), hide_index=True, use_container_width=True)
    else:
        st.info("No configs found.")

    if summary.empty:
        st.warning("No completed paired results yet.")
        return

    proposal_filter, n_filter, comp_filter = sidebar_filters(summary)
    filtered_summary = apply_filters(summary, proposal_filter, n_filter, comp_filter)
    filtered_pairs = apply_filters(paired, proposal_filter, n_filter, comp_filter)

    st.subheader("Aggregated Summary")
    st.dataframe(filtered_summary, hide_index=True, use_container_width=True)

    st.subheader("Paired Runs")
    st.dataframe(filtered_pairs, hide_index=True, use_container_width=True)

    analysis_dir = results_root / "analysis"
    proposal1_plot = analysis_dir / "proposal1_advantage_vs_elo.png"
    proposal2_plot = analysis_dir / "proposal2_advantage_vs_elo.png"

    st.subheader("Saved Plots")
    if proposal1_plot.exists():
        st.image(str(proposal1_plot), caption=PROPOSAL_LABELS["proposal1_invasion"], use_container_width=True)
    else:
        st.info("Run analysis to generate the Proposal 1 plot.")

    if proposal2_plot.exists():
        st.image(str(proposal2_plot), caption=PROPOSAL_LABELS["proposal2_mixed_ecology"], use_container_width=True)
    else:
        st.info("Run analysis to generate the Proposal 2 plot.")


if __name__ == "__main__":
    main()
