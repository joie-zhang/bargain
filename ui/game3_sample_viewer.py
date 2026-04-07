"""
Dedicated Streamlit viewer for a single Game 3 co-funding run.

Usage:
    streamlit run ui/game3_sample_viewer.py -- --results-dir <results_dir>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ui.experiment_viewer import (  # noqa: E402
    GAME_TYPES,
    load_all_interactions,
    load_experiment_results,
    render_agent_comparison_tab,
    render_analytics_tab,
    render_metrics_overview,
    render_raw_data_tab,
    render_sidebar_agent_info,
    render_timeline_tab,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--results-dir", type=str, required=True)
    args, _ = parser.parse_known_args(sys.argv[1:])
    return args


def resolve_results_dir(raw_value: str) -> Path:
    candidate = Path(raw_value)
    if candidate.is_absolute():
        return candidate.resolve()
    return (Path.cwd() / candidate).resolve()


def compute_competition_index(config: dict) -> Optional[float]:
    if config.get("game_type") != "co_funding":
        return None
    alpha = config.get("alpha")
    sigma = config.get("sigma")
    if alpha is None or sigma is None:
        return None
    return (1.0 - float(alpha)) * (1.0 - float(sigma))


def main() -> None:
    args = parse_args()
    results_dir = resolve_results_dir(args.results_dir)

    st.set_page_config(
        page_title="Game 3 Sample Viewer",
        page_icon="S",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    results = load_experiment_results(str(results_dir))
    if results is None:
        st.error(f"Could not load experiment results from: `{results_dir}`")
        return

    interactions = load_all_interactions(str(results_dir))
    config = results.get("config", {})
    game_type = config.get("game_type", "co_funding")
    experiment_id = results_dir.name

    st.sidebar.title("Game 3 Sample")
    st.sidebar.markdown("---")
    show_prompts = st.sidebar.checkbox("Show prompts", value=False, key="sidebar_show_prompts")
    show_private = st.sidebar.checkbox(
        "Show private thinking", value=True, key="sidebar_show_private"
    )
    render_sidebar_agent_info(results, game_type)

    game_label = GAME_TYPES.get(game_type, game_type)
    st.title(game_label)
    st.caption(f"`{results_dir}`")

    if game_type == "co_funding":
        alpha = config.get("alpha")
        sigma = config.get("sigma")
        competition_index = compute_competition_index(config)
        gamma = config.get("cofunding_time_discount")
        st.markdown(
            " | ".join(
                [
                    f"**alpha** = {alpha}",
                    f"**sigma** = {sigma}",
                    f"**competition_index** = {competition_index:.3f}" if competition_index is not None else "",
                    f"**gamma** = {gamma}" if gamma is not None else "",
                ]
            ).strip(" |")
        )

    render_metrics_overview(results, game_type)
    st.markdown("---")

    tab_labels = ["Timeline", "Agent Comparison", "Analytics", "Raw Data"]
    active_tab = st.radio(
        "View",
        tab_labels,
        horizontal=True,
        label_visibility="collapsed",
        key="main_tab_selector",
    )
    st.markdown("---")

    if active_tab == "Timeline":
        render_timeline_tab(
            results, game_type, show_prompts, show_private, experiment_id, interactions
        )
    elif active_tab == "Agent Comparison":
        render_agent_comparison_tab(
            results, interactions, game_type, show_prompts, experiment_id
        )
    elif active_tab == "Analytics":
        render_analytics_tab(results, game_type, experiment_id, interactions)
    elif active_tab == "Raw Data":
        render_raw_data_tab(results, interactions, experiment_id)


if __name__ == "__main__":
    main()
