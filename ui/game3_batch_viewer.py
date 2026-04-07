"""
Dedicated Streamlit viewer for Game 3 co-funding model-scale batches.

Usage:
    streamlit run ui/game3_batch_viewer.py -- --results-dir <results_dir_or_configs_dir>
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = PROJECT_ROOT / "experiments" / "results"
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


MODEL_INFO: Dict[str, Dict[str, Any]] = {
    "claude-opus-4-6": {"short": "Opus 4.6", "elo": 1475},
    "gpt-5-nano": {"short": "GPT-5-nano", "elo": 1338},
    "gpt-3.5-turbo-0125": {"short": "GPT-3.5", "elo": 1225},
    "grok-4": {"short": "Grok-4", "elo": 1409},
    "o3-mini-high": {"short": "o3-mini-high", "elo": 1364},
}
MODEL_ALIASES: Dict[str, str] = {
    "grok-4-0709": "grok-4",
}
PLOT_TEMPLATE = "plotly_white"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--results-dir", type=str, default=None)
    args, _ = parser.parse_known_args(sys.argv[1:])
    return args


def competition_index(alpha: float, sigma: float) -> float:
    return (1.0 - float(alpha)) * (1.0 - float(sigma))


def _resolve_candidate(path_str: str) -> Path:
    candidate = Path(path_str)
    if candidate.is_absolute():
        return candidate.resolve()
    return (Path.cwd() / candidate).resolve()


def latest_cofunding_dir() -> Optional[Path]:
    preferred = sorted(
        path
        for path in RESULTS_ROOT.glob("cofunding_*_reference_slate")
        if path.is_dir()
    )
    if preferred:
        return preferred[-1]

    generic = sorted(
        path
        for path in RESULTS_ROOT.glob("cofunding_*")
        if path.is_dir() and path.name != "cofunding_latest"
    )
    return generic[-1] if generic else None


def resolve_results_root(raw_value: Optional[str]) -> Path:
    if raw_value is None:
        latest = latest_cofunding_dir()
        if latest is None:
            raise FileNotFoundError(f"No cofunding_* directories found under {RESULTS_ROOT}")
        return latest.resolve()

    candidate = _resolve_candidate(raw_value)
    if candidate.name == "configs" and (candidate / "experiment_index.csv").exists():
        return candidate.parent.resolve()
    if (candidate / "configs" / "experiment_index.csv").exists():
        return candidate.resolve()
    raise FileNotFoundError(
        f"Expected a cofunding results root or configs directory, got: {candidate}"
    )


def canonical_model_name(model_name: str) -> str:
    return MODEL_ALIASES.get(model_name, model_name)


def model_short_name(model_name: str) -> str:
    canonical = canonical_model_name(model_name)
    return MODEL_INFO.get(canonical, {}).get("short", canonical)


def model_elo(model_name: str) -> Optional[float]:
    canonical = canonical_model_name(model_name)
    return MODEL_INFO.get(canonical, {}).get("elo")


def _resolve_output_dir(raw_output_dir: str) -> Path:
    candidate = Path(raw_output_dir)
    if candidate.is_absolute():
        return candidate.resolve()
    return (PROJECT_ROOT / candidate).resolve()


def _result_file(output_dir: Path) -> Optional[Path]:
    for candidate in [
        output_dir / "run_1_experiment_results.json",
        output_dir / "experiment_results.json",
    ]:
        if candidate.exists():
            return candidate
    return None


def _interactions_file(output_dir: Path) -> Optional[Path]:
    for candidate in [
        output_dir / "run_1_all_interactions.json",
        output_dir / "all_interactions.json",
    ]:
        if candidate.exists():
            return candidate
    return None


def _load_partial_interactions(interactions_path: str) -> List[Dict[str, Any]]:
    path = Path(interactions_path)
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    return []


def _infer_utility_by_model(
    model1: str,
    model2: str,
    final_utilities: Dict[str, Any],
    agent_performance: Dict[str, Any],
) -> Dict[str, float]:
    utility_by_model: Dict[str, float] = {}
    unknown_agents: List[str] = []

    for agent_id, payload in agent_performance.items():
        if agent_id not in final_utilities:
            continue
        raw_model = payload.get("model")
        if raw_model and raw_model != "unknown":
            utility_by_model[canonical_model_name(raw_model)] = float(final_utilities[agent_id])
        else:
            unknown_agents.append(agent_id)

    remaining_models = [model for model in (model1, model2) if model not in utility_by_model]
    if len(unknown_agents) == len(remaining_models):
        for agent_id, model_name in zip(sorted(unknown_agents), remaining_models):
            utility_by_model[model_name] = float(final_utilities[agent_id])

    return utility_by_model


def _sort_maybe_numeric(values: List[Any]) -> List[Any]:
    try:
        return sorted(values)
    except TypeError:
        return sorted(values, key=lambda value: str(value))


def _funded_project_count(final_allocation: Any) -> int:
    if isinstance(final_allocation, list):
        return len(final_allocation)
    return 0


@st.cache_data(show_spinner=False)
def load_batch_data(results_root: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    root = Path(results_root)
    index_path = root / "configs" / "experiment_index.csv"
    if not index_path.exists():
        raise FileNotFoundError(f"No experiment index found at {index_path}")

    index_rows: List[Dict[str, Any]] = []
    experiment_rows: List[Dict[str, Any]] = []
    long_rows: List[Dict[str, Any]] = []

    with index_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            config_file = root / "configs" / row["config_file"]
            config_payload = json.loads(config_file.read_text(encoding="utf-8"))

            model1 = row["model1"]
            model2 = row["model2"]
            alpha = float(config_payload["alpha"])
            sigma = float(config_payload["sigma"])
            ci = competition_index(alpha, sigma)
            output_dir = _resolve_output_dir(config_payload["output_dir"])
            result_path = _result_file(output_dir)
            interactions_path = _interactions_file(output_dir)
            if result_path is not None:
                status = "completed"
            elif interactions_path is not None:
                status = "in_progress"
            else:
                status = "pending"

            index_row: Dict[str, Any] = {
                "experiment_id": int(row["experiment_id"]),
                "experiment_type": row["experiment_type"],
                "model1": model1,
                "model2": model2,
                "pair_label": f"{model1} vs {model2}",
                "model_order": row["model_order"],
                "run_number": int(row["run_number"]),
                "seed": int(row["seed"]),
                "alpha": alpha,
                "sigma": sigma,
                "competition_index": ci,
                "ci_label": f"CI={ci:.2f}",
                "discussion_turns": int(config_payload.get("discussion_turns", 0)),
                "max_rounds": int(config_payload.get("max_rounds", 0)),
                "output_dir": str(output_dir),
                "config_file": row["config_file"],
                "result_path": str(result_path) if result_path is not None else "",
                "interactions_path": str(interactions_path) if interactions_path is not None else "",
                "status": status,
            }
            index_row["selection_label"] = (
                f"#{index_row['experiment_id']:04d} | {model_short_name(model2)} | "
                f"{index_row['model_order']} | alpha={alpha:.1f} | sigma={sigma:.1f} | "
                f"CI={ci:.2f} | {status}"
            )
            index_rows.append(index_row)

            if result_path is None:
                continue

            result_payload = json.loads(result_path.read_text(encoding="utf-8"))
            final_utilities = result_payload.get("final_utilities", {})
            agent_performance = result_payload.get("agent_performance", {})
            utility_by_model = _infer_utility_by_model(
                model1=model1,
                model2=model2,
                final_utilities=final_utilities,
                agent_performance=agent_performance,
            )

            baseline_model = model1
            adversary_model = model2
            social_welfare = sum(float(value) for value in final_utilities.values())
            funded_count = _funded_project_count(result_payload.get("final_allocation"))

            experiment_rows.append(
                {
                    "experiment_id": int(row["experiment_id"]),
                    "pair_label": f"{model1} vs {model2}",
                    "baseline_model": baseline_model,
                    "adversary_model": adversary_model,
                    "baseline_short": model_short_name(baseline_model),
                    "adversary_short": model_short_name(adversary_model),
                    "baseline_elo": model_elo(baseline_model),
                    "adversary_elo": model_elo(adversary_model),
                    "model_order": row["model_order"],
                    "alpha": alpha,
                    "sigma": sigma,
                    "competition_index": ci,
                    "ci_label": f"CI={ci:.2f}",
                    "baseline_utility": utility_by_model.get(baseline_model),
                    "adversary_utility": utility_by_model.get(adversary_model),
                    "social_welfare": social_welfare,
                    "funded_project_count": funded_count,
                    "consensus_reached": bool(result_payload.get("consensus_reached", False)),
                    "final_round": int(result_payload.get("final_round", -1) or -1),
                    "output_dir": str(output_dir),
                }
            )

            for model_name, utility in utility_by_model.items():
                long_rows.append(
                    {
                        "experiment_id": int(row["experiment_id"]),
                        "pair_label": f"{model1} vs {model2}",
                        "model": model_name,
                        "model_short": model_short_name(model_name),
                        "role": "baseline" if model_name == baseline_model else "adversary",
                        "utility": utility,
                        "elo": model_elo(model_name),
                        "model_order": row["model_order"],
                        "alpha": alpha,
                        "sigma": sigma,
                        "competition_index": ci,
                        "ci_label": f"CI={ci:.2f}",
                        "adversary_model": adversary_model,
                        "adversary_short": model_short_name(adversary_model),
                    }
                )

    index_df = pd.DataFrame(index_rows).sort_values(
        ["model2", "model_order", "alpha", "sigma", "experiment_id"]
    )
    experiment_df = pd.DataFrame(experiment_rows)
    long_df = pd.DataFrame(long_rows)
    return index_df.reset_index(drop=True), experiment_df, long_df


def sidebar_select_filter(label: str, values: List[Any], format_func=None) -> Any:
    options = [None] + list(values)

    def _format(value: Any) -> str:
        if value is None:
            return "Any"
        if format_func is None:
            return str(value)
        return str(format_func(value))

    return st.sidebar.selectbox(label, options=options, format_func=_format)


def filter_index(index_df: pd.DataFrame) -> pd.DataFrame:
    status_options = _sort_maybe_numeric(index_df["status"].unique().tolist())
    model_options = _sort_maybe_numeric(index_df["model2"].unique().tolist())
    order_options = _sort_maybe_numeric(index_df["model_order"].unique().tolist())
    alpha_options = _sort_maybe_numeric(index_df["alpha"].unique().tolist())
    sigma_options = _sort_maybe_numeric(index_df["sigma"].unique().tolist())
    ci_options = _sort_maybe_numeric(index_df["competition_index"].unique().tolist())

    filters = {
        "status": sidebar_select_filter("Status", status_options),
        "model2": sidebar_select_filter("Adversary", model_options, model_short_name),
        "model_order": sidebar_select_filter("Model Order", order_options),
        "alpha": sidebar_select_filter(
            "alpha",
            alpha_options,
            format_func=lambda value: f"{float(value):.1f}",
        ),
        "sigma": sidebar_select_filter(
            "sigma",
            sigma_options,
            format_func=lambda value: f"{float(value):.1f}",
        ),
        "competition_index": sidebar_select_filter(
            "Competition Index",
            ci_options,
            format_func=lambda value: f"{float(value):.2f}",
        ),
    }

    filtered = index_df.copy()
    for column, selected in filters.items():
        if selected is not None:
            filtered = filtered[filtered[column] == selected]
    return filtered.reset_index(drop=True)


def coverage_table(index_df: pd.DataFrame) -> pd.DataFrame:
    table = (
        index_df.groupby(["model2", "model_order", "status"])
        .size()
        .reset_index(name="count")
        .pivot(index=["model2", "model_order"], columns="status", values="count")
        .fillna(0)
        .reset_index()
    )
    if "completed" not in table.columns:
        table["completed"] = 0
    if "in_progress" not in table.columns:
        table["in_progress"] = 0
    if "pending" not in table.columns:
        table["pending"] = 0
    table["adversary"] = table["model2"].map(model_short_name)
    return table[["adversary", "model_order", "completed", "in_progress", "pending"]].sort_values(
        ["adversary", "model_order"]
    )


def render_overview_metrics(index_df: pd.DataFrame, experiment_df: pd.DataFrame) -> None:
    completed = int((index_df["status"] == "completed").sum())
    in_progress = int((index_df["status"] == "in_progress").sum())
    pending = int((index_df["status"] != "completed").sum())
    avg_welfare = float(experiment_df["social_welfare"].mean()) if not experiment_df.empty else 0.0
    avg_funded = (
        float(experiment_df["funded_project_count"].mean()) if not experiment_df.empty else 0.0
    )

    columns = st.columns(5)
    columns[0].metric("Total Configs", str(len(index_df)))
    columns[1].metric("Completed", str(completed))
    columns[2].metric("In Progress", str(in_progress))
    columns[3].metric("Pending", str(pending - in_progress))
    columns[4].metric("Mean Welfare", f"{avg_welfare:.2f}")
    st.caption(f"Mean funded projects across completed runs: {avg_funded:.2f}")

    st.caption("Game 3 competition proxy: `CI3 = (1 - alpha) * (1 - sigma)`.")
    st.dataframe(coverage_table(index_df), hide_index=True, use_container_width=True)


def _configure_figure_layout(fig: go.Figure, x_title: str, y_title: str) -> go.Figure:
    fig.update_layout(
        template=PLOT_TEMPLATE,
        margin=dict(l=20, r=20, t=60, b=20),
        legend_title_text="",
        xaxis_title=x_title,
        yaxis_title=y_title,
        height=460,
    )
    return fig


def render_elo_plots(long_df: pd.DataFrame) -> None:
    st.subheader("Utility vs Elo")
    if long_df.empty:
        st.info("No completed run data available yet.")
        return

    overall = (
        long_df.groupby(["model", "model_short", "role", "elo"], as_index=False)
        .agg(avg_utility=("utility", "mean"), num_runs=("utility", "size"))
        .sort_values("elo")
    )
    overall_fig = px.scatter(
        overall,
        x="elo",
        y="avg_utility",
        color="role",
        text="model_short",
        hover_data={"model": True, "num_runs": True, "elo": True, "avg_utility": ":.2f"},
        title="Average utility by model",
    )
    overall_fig.update_traces(
        textposition="top center",
        marker=dict(size=13, line=dict(width=1, color="#1f2937")),
    )
    _configure_figure_layout(overall_fig, "Model Elo", "Average Utility")
    st.plotly_chart(overall_fig, use_container_width=True)

    by_ci = (
        long_df.groupby(["competition_index", "ci_label", "model_short", "elo"], as_index=False)
        .agg(avg_utility=("utility", "mean"))
        .sort_values(["competition_index", "elo", "model_short"])
    )
    ci_fig = px.line(
        by_ci,
        x="elo",
        y="avg_utility",
        color="ci_label",
        markers=True,
        hover_data={"model_short": True, "competition_index": ":.2f", "avg_utility": ":.2f"},
        custom_data=["model_short"],
        title="Average utility by model, with one line per competition index",
    )
    ci_fig.update_traces(
        mode="lines+markers",
        hovertemplate="%{customdata[0]}<br>Elo=%{x}<br>Utility=%{y:.2f}<extra>%{fullData.name}</extra>",
    )
    _configure_figure_layout(ci_fig, "Model Elo", "Average Utility")
    st.plotly_chart(ci_fig, use_container_width=True)


def render_parameter_plots(long_df: pd.DataFrame) -> None:
    st.subheader("Utility vs alpha, sigma, and competition index")
    if long_df.empty:
        st.info("No completed run data available yet.")
        return

    alpha_df = (
        long_df.groupby(["alpha", "model_short"], as_index=False)
        .agg(avg_utility=("utility", "mean"))
        .sort_values(["model_short", "alpha"])
    )
    alpha_fig = px.line(
        alpha_df,
        x="alpha",
        y="avg_utility",
        color="model_short",
        markers=True,
        title="Average utility vs alpha",
    )
    _configure_figure_layout(alpha_fig, "alpha", "Average Utility")

    sigma_df = (
        long_df.groupby(["sigma", "model_short"], as_index=False)
        .agg(avg_utility=("utility", "mean"))
        .sort_values(["model_short", "sigma"])
    )
    sigma_fig = px.line(
        sigma_df,
        x="sigma",
        y="avg_utility",
        color="model_short",
        markers=True,
        title="Average utility vs sigma",
    )
    _configure_figure_layout(sigma_fig, "sigma", "Average Utility")

    ci_df = (
        long_df.groupby(["competition_index", "model_short"], as_index=False)
        .agg(avg_utility=("utility", "mean"))
        .sort_values(["model_short", "competition_index"])
    )
    ci_fig = px.line(
        ci_df,
        x="competition_index",
        y="avg_utility",
        color="model_short",
        markers=True,
        title="Average utility vs scalar competition index",
    )
    _configure_figure_layout(ci_fig, "Competition Index", "Average Utility")

    top_row = st.columns(2)
    with top_row[0]:
        st.plotly_chart(alpha_fig, use_container_width=True)
    with top_row[1]:
        st.plotly_chart(sigma_fig, use_container_width=True)
    st.plotly_chart(ci_fig, use_container_width=True)


def _heatmap_figure(
    data: pd.DataFrame,
    value_col: str,
    title: str,
    z_title: str,
    color_scale: str = "RdBu",
) -> go.Figure:
    pivot = (
        data.pivot(index="alpha", columns="sigma", values=value_col)
        .sort_index(ascending=True)
        .sort_index(axis=1, ascending=True)
    )
    fig = go.Figure(
        data=go.Heatmap(
            x=[f"{float(val):.1f}" for val in pivot.columns.tolist()],
            y=[f"{float(val):.1f}" for val in pivot.index.tolist()],
            z=pivot.values,
            colorscale=color_scale,
            colorbar=dict(title=z_title),
            hovertemplate="alpha=%{y}<br>sigma=%{x}<br>value=%{z:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        template=PLOT_TEMPLATE,
        title=title,
        xaxis_title="sigma",
        yaxis_title="alpha",
        height=420,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def render_heatmaps(experiment_df: pd.DataFrame, selected_row: pd.Series) -> None:
    st.subheader("alpha x sigma views")
    if experiment_df.empty:
        st.info("No completed run data available yet.")
        return

    selected_adversary = selected_row["model2"]
    selected_pair_df = experiment_df[experiment_df["adversary_model"] == selected_adversary]
    if selected_pair_df.empty:
        st.info("No completed runs are available yet for the selected adversary model.")
        return

    welfare_df = (
        experiment_df.groupby(["alpha", "sigma"], as_index=False)
        .agg(social_welfare=("social_welfare", "mean"))
    )
    funded_df = (
        selected_pair_df.groupby(["alpha", "sigma"], as_index=False)
        .agg(funded_project_count=("funded_project_count", "mean"))
    )
    pair_df = (
        selected_pair_df.groupby(["alpha", "sigma"], as_index=False)
        .agg(
            baseline_utility=("baseline_utility", "mean"),
            adversary_utility=("adversary_utility", "mean"),
        )
    )

    columns = st.columns(2)
    with columns[0]:
        st.plotly_chart(
            _heatmap_figure(
                welfare_df,
                value_col="social_welfare",
                title="Mean social welfare across all completed runs",
                z_title="Welfare",
                color_scale="Viridis",
            ),
            use_container_width=True,
        )
    with columns[1]:
        st.plotly_chart(
            _heatmap_figure(
                funded_df,
                value_col="funded_project_count",
                title=f"Mean funded projects vs alpha x sigma for {model_short_name(selected_adversary)}",
                z_title="Projects",
                color_scale="YlGnBu",
            ),
            use_container_width=True,
        )

    bottom = st.columns(2)
    with bottom[0]:
        st.plotly_chart(
            _heatmap_figure(
                pair_df,
                value_col="baseline_utility",
                title=f"{model_short_name(selected_row['model1'])} utility vs alpha x sigma",
                z_title="Utility",
            ),
            use_container_width=True,
        )
    with bottom[1]:
        st.plotly_chart(
            _heatmap_figure(
                pair_df,
                value_col="adversary_utility",
                title=f"{model_short_name(selected_adversary)} utility vs alpha x sigma",
                z_title="Utility",
            ),
            use_container_width=True,
        )


def render_selected_config_table(selected_row: pd.Series) -> None:
    rows = [
        {"name": "experiment_id", "value": str(int(selected_row["experiment_id"]))},
        {"name": "pair", "value": str(selected_row["pair_label"])},
        {"name": "model_order", "value": str(selected_row["model_order"])},
        {"name": "alpha", "value": f"{float(selected_row['alpha']):.1f}"},
        {"name": "sigma", "value": f"{float(selected_row['sigma']):.1f}"},
        {"name": "competition_index", "value": f"{float(selected_row['competition_index']):.2f}"},
        {"name": "max_rounds", "value": str(int(selected_row["max_rounds"]))},
        {"name": "discussion_turns", "value": str(int(selected_row["discussion_turns"]))},
        {"name": "seed", "value": str(int(selected_row["seed"]))},
        {"name": "status", "value": str(selected_row["status"])},
        {"name": "config_file", "value": str(selected_row["config_file"])},
        {"name": "output_dir", "value": str(selected_row["output_dir"])},
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def render_partial_run(selected_row: pd.Series, show_prompts: bool, show_private: bool) -> None:
    interactions = _load_partial_interactions(str(selected_row["interactions_path"]))
    st.info(
        "This config is still running. Showing the partial exchange history saved so far."
    )
    if not interactions:
        st.warning("No interaction payload is available yet for this in-progress config.")
        return

    current_round = max(int(entry.get("round", 0) or 0) for entry in interactions)
    last_phase = interactions[-1].get("phase", "unknown")
    agents_seen = sorted({str(entry.get("agent_id", "unknown")) for entry in interactions})

    columns = st.columns(3)
    columns[0].metric("Saved Exchanges", str(len(interactions)))
    columns[1].metric("Current Round", str(current_round))
    columns[2].metric("Last Phase", str(last_phase))
    st.caption(f"Agents seen so far: {', '.join(agents_seen)}")

    preview_rows = []
    for entry in interactions:
        prompt = entry.get("prompt", "")
        response = entry.get("response", "")
        preview_rows.append(
            {
                "round": entry.get("round"),
                "phase": entry.get("phase"),
                "agent": entry.get("agent_id"),
                "model": entry.get("model_name"),
                "prompt_preview": prompt[:220] if show_prompts else "",
                "response_preview": response[:320],
            }
        )

    visible_columns = ["round", "phase", "agent", "model", "response_preview"]
    if show_prompts:
        visible_columns.append("prompt_preview")
    st.dataframe(
        pd.DataFrame(preview_rows)[visible_columns],
        hide_index=True,
        use_container_width=True,
    )

    with st.expander("Latest raw interaction payload", expanded=False):
        st.json(interactions[-1])


def render_selected_run(selected_row: pd.Series, show_prompts: bool, show_private: bool) -> None:
    st.subheader("Selected Config")
    render_selected_config_table(selected_row)

    if selected_row["status"] == "in_progress":
        render_partial_run(selected_row, show_prompts, show_private)
        return

    if selected_row["status"] != "completed":
        st.warning(
            "This config does not have result artifacts yet. "
            "Pick a completed config from the sidebar to inspect the full rollout."
        )
        return

    results_dir = selected_row["output_dir"]
    results = load_experiment_results(results_dir)
    if results is None:
        st.error(f"Could not load experiment results from: `{results_dir}`")
        return

    interactions = load_all_interactions(results_dir)
    config = results.get("config", {})
    game_type = config.get("game_type", "co_funding")
    experiment_id = f"game3_batch_{selected_row['experiment_id']}"

    render_metrics_overview(results, game_type)
    st.markdown("---")

    tab_labels = ["Timeline", "Agent Comparison", "Analytics", "Raw Data"]
    active_tab = st.radio(
        "Selected run view",
        tab_labels,
        horizontal=True,
        label_visibility="collapsed",
        key=f"selected_run_tab_{selected_row['experiment_id']}",
    )
    st.markdown("---")

    if active_tab == "Timeline":
        render_timeline_tab(
            results,
            game_type,
            show_prompts,
            show_private,
            experiment_id,
            interactions,
        )
    elif active_tab == "Agent Comparison":
        render_agent_comparison_tab(
            results,
            interactions,
            game_type,
            show_prompts,
            experiment_id,
        )
    elif active_tab == "Analytics":
        render_analytics_tab(results, game_type, experiment_id, interactions)
    elif active_tab == "Raw Data":
        render_raw_data_tab(results, interactions, experiment_id)


def main() -> None:
    args = parse_args()
    results_root = resolve_results_root(args.results_dir)

    st.set_page_config(
        page_title="Game 3 Batch Viewer",
        page_icon="S",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    index_df, experiment_df, long_df = load_batch_data(str(results_root))
    if index_df.empty:
        st.error("No configs found under the selected Game 3 batch.")
        st.stop()

    st.sidebar.title("Game 3 Batch")
    st.sidebar.markdown("---")
    st.sidebar.caption(str(results_root))
    if st.sidebar.button("Refresh batch data", use_container_width=True):
        load_batch_data.clear()
        st.rerun()
    show_prompts = st.sidebar.checkbox("Show prompts", value=False, key="batch_show_prompts")
    show_private = st.sidebar.checkbox(
        "Show private thinking", value=True, key="batch_show_private"
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Filter Configs**")

    filtered_index = filter_index(index_df)
    st.sidebar.caption(f"{len(filtered_index)} matching configs")
    if filtered_index.empty:
        st.warning("No configs match the current sidebar filters.")
        st.stop()

    selection_options = filtered_index["selection_label"].tolist()
    if st.session_state.get("game3_batch_selected_config") not in selection_options:
        st.session_state["game3_batch_selected_config"] = selection_options[0]

    selected_label = st.sidebar.selectbox(
        "Matching Config",
        options=selection_options,
        key="game3_batch_selected_config",
    )
    selected_row = filtered_index.loc[
        filtered_index["selection_label"] == selected_label
    ].iloc[0]

    if selected_row["status"] == "completed":
        selected_results = load_experiment_results(selected_row["output_dir"])
        if selected_results is not None:
            render_sidebar_agent_info(
                selected_results,
                selected_results.get("config", {}).get("game_type", "co_funding"),
            )

    st.title(GAME_TYPES.get("co_funding", "Game 3 Co-Funding"))
    st.caption(
        f"Batch root: `{results_root}` | Selected config: `{selected_row['selection_label']}`"
    )

    active_main_view = st.radio(
        "Game 3 batch view",
        ["Selected Run", "Batch Summary"],
        horizontal=True,
        label_visibility="collapsed",
        key="game3_batch_main_view",
    )
    st.markdown("---")

    if active_main_view == "Batch Summary":
        render_overview_metrics(index_df, experiment_df)
        render_elo_plots(long_df)
        render_parameter_plots(long_df)
        render_heatmaps(experiment_df, selected_row)
    else:
        render_selected_run(selected_row, show_prompts, show_private)


if __name__ == "__main__":
    main()
