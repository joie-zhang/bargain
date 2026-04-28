"""
Unified Streamlit viewer for single-sample runs across all three games.

Usage:
    streamlit run ui/multi_game_sample_viewer.py
    streamlit run ui/multi_game_sample_viewer.py -- --results-dir <dir1> --results-dir <dir2>
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = PROJECT_ROOT / "experiments" / "results"
sys.path.insert(0, str(PROJECT_ROOT))

import ui.experiment_viewer as exp_viewer  # noqa: E402
import ui.game1_sample_viewer as game1_viewer  # noqa: E402


GAME_LABELS = {
    "item_allocation": "Game 1: Item Allocation",
    "diplomacy": "Game 2: Diplomatic Treaty",
    "co_funding": "Game 3: Co-Funding",
}

GAME_ORDER = {
    "item_allocation": 0,
    "diplomacy": 1,
    "co_funding": 2,
}

AGENT_PALETTE = [
    "#2563eb",
    "#dc2626",
    "#059669",
    "#7c3aed",
    "#ea580c",
    "#0891b2",
    "#65a30d",
    "#db2777",
    "#4338ca",
    "#0f766e",
]


@dataclass(frozen=True)
class SampleRun:
    path: Path
    game_type: str
    label: str
    agents: List[str]
    models: List[str]
    consensus: bool
    final_round: Optional[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--results-dir", action="append", default=[])
    parser.add_argument("--results-root", action="append", default=[])
    args, _ = parser.parse_known_args(sys.argv[1:])
    return args


def resolve_results_dir(raw_value: str) -> Path:
    candidate = Path(raw_value)
    if candidate.is_absolute():
        return candidate.resolve()
    return (Path.cwd() / candidate).resolve()


def _latest_matching_dir(pattern: str) -> Optional[Path]:
    matches = sorted(RESULTS_ROOT.glob(pattern))
    return matches[-1].resolve() if matches else None


def default_results_dirs() -> List[Path]:
    defaults = [
        _latest_matching_dir("codex_n3_smoke_item_*"),
        _latest_matching_dir("codex_n3_smoke_diplomacy_*"),
        _latest_matching_dir("codex_n3_smoke_cofunding_*"),
    ]
    return [path for path in defaults if path is not None]


def discover_result_dirs(candidate: Path) -> List[Path]:
    if (candidate / "experiment_results.json").exists():
        return [candidate]

    runs_dir = candidate / "runs"
    if runs_dir.exists():
        return sorted(path.parent for path in runs_dir.glob("*/experiment_results.json"))

    return sorted(path.parent for path in candidate.glob("*/experiment_results.json"))


def _load_results_json(results_dir: Path) -> Dict:
    results_path = results_dir / "experiment_results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"Missing experiment_results.json in {results_dir}")
    return json.loads(results_path.read_text(encoding="utf-8"))


def _build_sample_run(results_dir: Path) -> SampleRun:
    results = _load_results_json(results_dir)
    config = results.get("config", {})
    game_type = str(config.get("game_type") or "unknown")
    perf = results.get("agent_performance", {})
    agents = list(config.get("agents") or sorted(perf.keys()))
    models = [perf.get(agent, {}).get("model", "?") for agent in agents]
    game_label = GAME_LABELS.get(game_type, game_type)
    label = f"{game_label} | {results_dir.name}"
    return SampleRun(
        path=results_dir,
        game_type=game_type,
        label=label,
        agents=agents,
        models=models,
        consensus=bool(results.get("consensus_reached", False)),
        final_round=results.get("final_round"),
    )


def discover_runs(raw_results_dirs: List[str], raw_results_roots: List[str]) -> List[SampleRun]:
    if raw_results_dirs or raw_results_roots:
        candidate_dirs: List[Path] = []
        for raw_dir in raw_results_dirs:
            candidate_dirs.extend(discover_result_dirs(resolve_results_dir(raw_dir)))
        for raw_root in raw_results_roots:
            candidate_dirs.extend(discover_result_dirs(resolve_results_dir(raw_root)))
    else:
        candidate_dirs = default_results_dirs()

    runs: List[SampleRun] = []
    seen: set[Path] = set()
    for results_dir in candidate_dirs:
        if results_dir in seen:
            continue
        seen.add(results_dir)
        if not results_dir.exists():
            continue
        try:
            runs.append(_build_sample_run(results_dir))
        except (FileNotFoundError, json.JSONDecodeError):
            continue
    runs.sort(key=lambda run: (GAME_ORDER.get(run.game_type, 99), run.path.name))
    return runs


def ensure_multiagent_colors(max_agents: int = 12) -> None:
    for index in range(max_agents):
        exp_viewer.AGENT_COLORS.setdefault(
            f"Agent_{index + 1}",
            AGENT_PALETTE[index % len(AGENT_PALETTE)],
        )


def render_sidebar_run_info(run: SampleRun) -> None:
    st.sidebar.markdown("**Selected Run**")
    st.sidebar.code(str(run.path))
    st.sidebar.markdown("**Agents**")
    for agent, model in zip(run.agents, run.models):
        st.sidebar.write(f"- {agent}: {model}")
    st.sidebar.markdown("**Outcome**")
    st.sidebar.write(f"- consensus: {'yes' if run.consensus else 'no'}")
    st.sidebar.write(f"- final_round: {run.final_round}")


def utility_vs_elo_frame(results: Dict) -> pd.DataFrame:
    config = results.get("config", {})
    final_utilities = results.get("final_utilities") or {}
    agent_elo_map = config.get("agent_elo_map") or {}
    agent_model_map = config.get("agent_model_map") or {}
    agent_role_map = config.get("agent_role_map") or {}
    performance = results.get("agent_performance") or {}
    agents = config.get("agents") or sorted(final_utilities.keys(), key=game1_viewer.agent_sort_key)

    rows = []
    for agent in agents:
        if agent not in final_utilities or agent not in agent_elo_map:
            continue
        try:
            utility = float(final_utilities[agent])
            elo = float(agent_elo_map[agent])
        except (TypeError, ValueError):
            continue
        rows.append({
            "Agent": agent,
            "Model": agent_model_map.get(agent) or performance.get(agent, {}).get("model", "?"),
            "Role": agent_role_map.get(agent, "agent"),
            "Elo": elo,
            "Final Utility": utility,
        })
    return pd.DataFrame(rows)


def render_utility_vs_elo(results: Dict, key_prefix: str) -> None:
    frame = utility_vs_elo_frame(results)
    if frame.empty:
        st.info("No agent Elo metadata was found for this run.")
        return

    st.markdown("#### Final Utility vs Elo")
    if exp_viewer.PLOTLY_AVAILABLE:
        fig = exp_viewer.px.scatter(
            frame,
            x="Elo",
            y="Final Utility",
            color="Model",
            symbol="Role",
            text="Agent",
            hover_data=["Agent", "Model", "Role", "Elo", "Final Utility"],
            title=None,
        )
        fig.update_traces(textposition="top center", marker=dict(size=12, opacity=0.85))
        if len(frame) >= 2 and frame["Elo"].nunique() >= 2:
            x_values = frame["Elo"].astype(float).to_numpy()
            y_values = frame["Final Utility"].astype(float).to_numpy()
            slope, intercept = np.polyfit(x_values, y_values, 1)
            x_fit = np.array([x_values.min(), x_values.max()])
            y_fit = slope * x_fit + intercept
            fig.add_trace(
                exp_viewer.go.Scatter(
                    x=x_fit,
                    y=y_fit,
                    mode="lines",
                    name="Linear fit",
                    line=dict(color="rgba(31, 41, 55, 0.45)", dash="dot", width=2),
                    hovertemplate=(
                        "Linear fit<br>"
                        "Elo=%{x:.0f}<br>"
                        "Fitted utility=%{y:.2f}<extra></extra>"
                    ),
                )
            )
        fig.update_layout(
            height=420,
            margin=dict(t=20, b=40, l=50, r=20),
            xaxis_title="Model Elo",
            yaxis_title="Final Utility",
            legend_title_text="Model",
        )
        st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_utility_vs_elo")
    else:
        st.scatter_chart(frame, x="Elo", y="Final Utility")

    st.dataframe(
        frame.sort_values(["Elo", "Agent"]).reset_index(drop=True),
        hide_index=True,
        use_container_width=True,
    )


def render_item_allocation_view(results_dir: Path, show_prompts: bool, show_raw: bool) -> None:
    payload = game1_viewer.load_sample(str(results_dir))
    interactions = payload["interactions"]
    results = payload["results"]
    config = results["config"]

    agent_models = game1_viewer.extract_agent_models(interactions)
    grouped_entries = game1_viewer.collect_entries_by_round(interactions)
    grouped_logs = game1_viewer.conversation_logs_by_round(results)
    item_lookup = game1_viewer.build_item_lookup(config)
    setup_entries = grouped_entries.get(0, [])
    preferences_by_agent = game1_viewer.extract_preferences_from_setup(setup_entries)
    token_summary = game1_viewer.aggregate_token_usage(interactions)

    st.title(GAME_LABELS["item_allocation"])
    st.caption(f"`{results_dir}`")

    metric_columns = st.columns(6)
    metric_columns[0].metric(
        "Outcome",
        "Consensus" if results.get("consensus_reached") else "No Consensus",
    )
    metric_columns[1].metric("Final Round", str(results.get("final_round")))
    metric_columns[2].metric("Agents", str(config.get("n_agents")))
    metric_columns[3].metric("Items", str(config.get("m_items")))
    metric_columns[4].metric("Competition", str(config.get("competition_level")))
    metric_columns[5].metric("Seed", str(config.get("random_seed")))

    st.subheader("Final Outcome")
    render_utility_vs_elo(results, f"game1_{results_dir.name}")

    hyperparameter_rows = [
        {
            "name": "models",
            "value": ", ".join(
                agent_models[agent_id]
                for agent_id in sorted(agent_models, key=game1_viewer.agent_sort_key)
            ),
        },
        {"name": "game_type", "value": config.get("game_type")},
        {"name": "n_agents", "value": config.get("n_agents")},
        {"name": "m_items", "value": config.get("m_items")},
        {"name": "t_rounds", "value": config.get("t_rounds")},
        {"name": "competition_level", "value": config.get("competition_level")},
        {"name": "gamma_discount", "value": config.get("gamma_discount")},
        {"name": "discussion_turns", "value": config.get("discussion_turns")},
        {"name": "disable_discussion", "value": config.get("disable_discussion")},
        {"name": "disable_thinking", "value": config.get("disable_thinking")},
        {"name": "disable_reflection", "value": config.get("disable_reflection")},
    ]
    st.subheader("Configuration")
    st.dataframe(pd.DataFrame(hyperparameter_rows), hide_index=True, use_container_width=True)

    game1_viewer.render_setup_section(
        setup_entries,
        agent_models,
        preferences_by_agent,
        show_prompts,
    )

    st.subheader("Outcome Summary")
    if results.get("consensus_reached"):
        st.success("A proposal reached the configured acceptance threshold.")
        st.json(results.get("final_allocation", {}))
        st.json(results.get("final_utilities", {}))
    else:
        st.warning("No proposal reached the configured acceptance threshold in this sample.")

    st.subheader("Full Rollout")
    round_numbers = [round_num for round_num in grouped_entries.keys() if round_num > 0]
    for round_num in round_numbers:
        game1_viewer.render_round_section(
            round_num,
            grouped_entries.get(round_num, []),
            grouped_logs.get(round_num, []),
            preferences_by_agent,
            item_lookup,
            float(config.get("gamma_discount", 0.9)),
            show_prompts,
        )

    st.subheader("Token Usage")
    if token_summary.empty:
        st.info("No token usage metadata found.")
    else:
        st.dataframe(token_summary, hide_index=True, use_container_width=True)

    if show_raw:
        st.subheader("Raw JSON")
        raw_columns = st.columns(2)
        with raw_columns[0]:
            st.markdown("**Experiment Results**")
            st.json(results)
        with raw_columns[1]:
            st.markdown("**All Interactions**")
            st.json(interactions)


def render_generic_view(
    results_dir: Path,
    game_type: str,
    show_prompts: bool,
    show_private: bool,
) -> None:
    results = exp_viewer.load_experiment_results(str(results_dir))
    interactions = exp_viewer.load_all_interactions(str(results_dir))

    if results is None:
        st.error(f"Could not load experiment results from `{results_dir}`")
        return

    config = results.get("config", {})
    st.title(GAME_LABELS.get(game_type, game_type))
    st.caption(f"`{results_dir}`")

    if game_type == "diplomacy":
        rho = config.get("rho")
        theta = config.get("theta")
        st.markdown(f"**rho** = {rho} | **theta** = {theta}")
    elif game_type == "co_funding":
        alpha = config.get("alpha")
        sigma = config.get("sigma")
        st.markdown(f"**alpha** = {alpha} | **sigma** = {sigma}")

    exp_viewer.render_metrics_overview(results, game_type)
    render_utility_vs_elo(results, f"{game_type}_{results_dir.name}")
    st.markdown("---")

    tab_labels = ["Timeline", "Agent Comparison", "Analytics", "Raw Data"]
    active_tab = st.radio(
        "View",
        tab_labels,
        horizontal=True,
        label_visibility="collapsed",
        key=f"view_{results_dir.name}",
    )
    st.markdown("---")

    if active_tab == "Timeline":
        exp_viewer.render_timeline_tab(
            results,
            game_type,
            show_prompts,
            show_private,
            results_dir.name,
            interactions,
        )
    elif active_tab == "Agent Comparison":
        exp_viewer.render_agent_comparison_tab(
            results,
            interactions,
            game_type,
            show_prompts,
            results_dir.name,
        )
    elif active_tab == "Analytics":
        exp_viewer.render_analytics_tab(results, game_type, results_dir.name, interactions)
    elif active_tab == "Raw Data":
        exp_viewer.render_raw_data_tab(results, interactions, results_dir.name)


def main() -> None:
    st.set_page_config(
        page_title="Multi-Game Sample Viewer",
        page_icon="S",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    args = parse_args()
    ensure_multiagent_colors()
    runs = discover_runs(args.results_dir, args.results_root)

    if not runs:
        st.error(
            "No sample result directories were found. "
            "Pass one or more `--results-dir` arguments or create `codex_n3_smoke_*` runs."
        )
        return

    st.sidebar.title("Multi-Game Samples")
    st.sidebar.metric("Loaded Runs", len(runs))
    selected_label = st.sidebar.selectbox(
        "Run",
        options=[run.label for run in runs],
    )
    selected_run = next(run for run in runs if run.label == selected_label)

    show_prompts = st.sidebar.checkbox("Show prompts", value=False)
    show_private = st.sidebar.checkbox("Show private thinking", value=True)
    show_raw = st.sidebar.checkbox("Show raw JSON", value=False)

    render_sidebar_run_info(selected_run)

    if selected_run.game_type == "item_allocation":
        render_item_allocation_view(selected_run.path, show_prompts, show_raw)
    elif selected_run.game_type in {"diplomacy", "co_funding"}:
        render_generic_view(
            selected_run.path,
            selected_run.game_type,
            show_prompts,
            show_private,
        )
    else:
        st.error(f"Unsupported game type: {selected_run.game_type}")


if __name__ == "__main__":
    main()
