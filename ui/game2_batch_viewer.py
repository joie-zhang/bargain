"""
Dedicated Streamlit viewer for Game 2 diplomacy model-scale batches.

Usage:
    streamlit run ui/game2_batch_viewer.py -- --results-dir <results_dir_or_configs_dir_or_run_dir>
"""

from __future__ import annotations

import argparse
import csv
import json
import re
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
    "gemini-3-flash": {"short": "Gemini 3 Flash", "elo": 1472},
    "grok-4": {"short": "Grok-4", "elo": 1409},
    "gpt-5-nano": {"short": "GPT-5-nano", "elo": 1338},
    "gpt-3.5-turbo-0125": {"short": "GPT-3.5", "elo": 1225},
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


def competition_index(rho: float, theta: float) -> float:
    return float(theta) * (1.0 - float(rho)) / 2.0


def _resolve_candidate(path_str: str) -> Path:
    candidate = Path(path_str).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (Path.cwd() / candidate).resolve()


def latest_diplomacy_dir() -> Optional[Path]:
    candidates = sorted(
        path
        for path in RESULTS_ROOT.glob("diplomacy_*")
        if path.is_dir() and path.name != "diplomacy_latest"
    )
    return candidates[-1] if candidates else None


def default_results_root() -> Path:
    latest = latest_diplomacy_dir()
    if latest is None:
        raise FileNotFoundError(f"No diplomacy_* directories found under {RESULTS_ROOT}")
    return latest.resolve()


def resolve_results_root(raw_value: Optional[str]) -> Path:
    if raw_value is None:
        return default_results_root()

    candidate = _resolve_candidate(raw_value)
    if candidate.is_file():
        candidate = candidate.parent.resolve()
    if candidate.name == "configs" and (candidate / "experiment_index.csv").exists():
        return candidate.parent.resolve()
    if (candidate / "configs" / "experiment_index.csv").exists():
        return candidate.resolve()
    if candidate.is_dir() and _result_file(candidate) is not None:
        return candidate.resolve()
    raise FileNotFoundError(
        "Expected a Game 2 batch root, a configs directory, or a single run folder with "
        f"`experiment_results.json`, got: {candidate}"
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


def _results_source_mode(root: Path) -> str:
    if (root / "configs" / "experiment_index.csv").exists():
        return "batch"
    if _result_file(root) is not None:
        return "single_run"
    raise FileNotFoundError(f"Could not determine how to read results from: {root}")


def _default_source_input(raw_value: Optional[str]) -> str:
    if raw_value is None:
        return str(default_results_root())
    try:
        return str(resolve_results_root(raw_value))
    except FileNotFoundError:
        return str(_resolve_candidate(raw_value))


def _parse_model_pair_from_name(path: Path) -> Tuple[Optional[str], Optional[str]]:
    match = re.match(r"(.+?)_vs_(.+?)_config", path.name)
    if not match:
        return None, None
    return canonical_model_name(match.group(1)), canonical_model_name(match.group(2))


def _infer_models_from_payload(path: Path, result_payload: Dict[str, Any]) -> Tuple[str, str]:
    parsed_model1, parsed_model2 = _parse_model_pair_from_name(path)
    if parsed_model1 and parsed_model2:
        return parsed_model1, parsed_model2

    agent_performance = result_payload.get("agent_performance", {})
    models: List[str] = []
    for agent_id in sorted(agent_performance):
        raw_model = agent_performance.get(agent_id, {}).get("model", "unknown")
        models.append(canonical_model_name(str(raw_model)))

    if len(models) >= 2:
        return models[0], models[1]
    if len(models) == 1:
        return models[0], "unknown"
    return "unknown", "unknown"


def _infer_run_number(path: Path) -> int:
    match = re.search(r"(?:^|[_-])run[_-]?(\d+)(?:$|[_-])", path.name)
    if match:
        return int(match.group(1))
    return 1


def _coerce_optional_int(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    return int(value)


def _extract_single_run_frames(root: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    result_path = _result_file(root)
    if result_path is None:
        raise FileNotFoundError(f"No experiment results found under {root}")

    interactions_path = _interactions_file(root)
    result_payload = json.loads(result_path.read_text(encoding="utf-8"))
    config_payload = result_payload.get("config", {})

    model1, model2 = _infer_models_from_payload(root, result_payload)
    rho = float(config_payload.get("rho", 0.0))
    theta = float(config_payload.get("theta", 0.0))
    ci = competition_index(rho, theta)
    run_number = _infer_run_number(root)
    model_order = str(config_payload.get("model_order") or config_payload.get("actual_order") or "unknown")
    seed = _coerce_optional_int(config_payload.get("random_seed"))
    source_experiment_id = str(
        config_payload.get("experiment_id") or result_payload.get("experiment_id") or root.name
    )

    index_row: Dict[str, Any] = {
        "experiment_id": 1,
        "source_experiment_id": source_experiment_id,
        "experiment_type": "single_run",
        "model1": model1,
        "model2": model2,
        "pair_label": f"{model1} vs {model2}",
        "model_order": model_order,
        "run_number": run_number,
        "seed": seed,
        "rho": rho,
        "theta": theta,
        "competition_index": ci,
        "ci_label": f"CI={ci:.2f}",
        "discussion_turns": int(config_payload.get("discussion_turns", 0)),
        "output_dir": str(root.resolve()),
        "config_file": "",
        "result_path": str(result_path),
        "interactions_path": str(interactions_path) if interactions_path is not None else "",
        "status": "completed",
    }
    index_row["selection_label"] = (
        f"#0001 | {model_short_name(model2)} | {model_order} | "
        f"rho={rho:.1f} | theta={theta:.1f} | CI={ci:.2f} | completed"
    )

    final_utilities = result_payload.get("final_utilities", {})
    agent_performance = result_payload.get("agent_performance", {})
    utility_by_model = _infer_utility_by_model(
        model1=model1,
        model2=model2,
        final_utilities=final_utilities,
        agent_performance=agent_performance,
    )
    baseline_utility = utility_by_model.get(model1)
    adversary_utility = utility_by_model.get(model2)
    social_welfare = sum(float(value) for value in final_utilities.values())

    experiment_row = {
        "experiment_id": 1,
        "pair_label": f"{model1} vs {model2}",
        "baseline_model": model1,
        "adversary_model": model2,
        "baseline_short": model_short_name(model1),
        "adversary_short": model_short_name(model2),
        "baseline_elo": model_elo(model1),
        "adversary_elo": model_elo(model2),
        "model_order": model_order,
        "rho": rho,
        "theta": theta,
        "competition_index": ci,
        "ci_label": f"CI={ci:.2f}",
        "baseline_utility": baseline_utility,
        "adversary_utility": adversary_utility,
        "social_welfare": social_welfare,
        "consensus_reached": bool(result_payload.get("consensus_reached", False)),
        "final_round": int(result_payload.get("final_round", -1) or -1),
        "output_dir": str(root.resolve()),
    }

    long_rows: List[Dict[str, Any]] = []
    for model_name, utility in utility_by_model.items():
        long_rows.append(
            {
                "experiment_id": 1,
                "pair_label": f"{model1} vs {model2}",
                "model": model_name,
                "model_short": model_short_name(model_name),
                "role": "baseline" if model_name == model1 else "adversary",
                "utility": utility,
                "elo": model_elo(model_name),
                "model_order": model_order,
                "rho": rho,
                "theta": theta,
                "competition_index": ci,
                "ci_label": f"CI={ci:.2f}",
                "adversary_model": model2,
                "adversary_short": model_short_name(model2),
            }
        )

    index_df = pd.DataFrame([index_row])
    experiment_df = pd.DataFrame([experiment_row])
    long_df = pd.DataFrame(long_rows)
    return index_df, experiment_df, long_df

@st.cache_data(show_spinner=False)
def load_batch_data(results_root: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    root = Path(results_root)
    mode = _results_source_mode(root)
    if mode == "single_run":
        return _extract_single_run_frames(root)

    return _load_batch_frames(root)


def _load_batch_frames(root: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    index_path = root / "configs" / "experiment_index.csv"
    index_rows: List[Dict[str, Any]] = []
    experiment_rows: List[Dict[str, Any]] = []
    long_rows: List[Dict[str, Any]] = []

    with open(index_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            config_file = root / "configs" / row["config_file"]
            config_payload = json.loads(config_file.read_text(encoding="utf-8"))

            model1 = row["model1"]
            model2 = row["model2"]
            rho = float(row["rho"])
            theta = float(row["theta"])
            ci = competition_index(rho, theta)
            output_dir = _resolve_output_dir(config_payload["output_dir"])
            result_path = _result_file(output_dir)
            interactions_path = _interactions_file(output_dir)
            status = "completed" if result_path is not None else "pending"

            index_row: Dict[str, Any] = {
                "experiment_id": int(row["experiment_id"]),
                "experiment_type": row["experiment_type"],
                "model1": model1,
                "model2": model2,
                "pair_label": f"{model1} vs {model2}",
                "model_order": row["model_order"],
                "run_number": int(row["run_number"]),
                "seed": int(row["seed"]),
                "rho": rho,
                "theta": theta,
                "competition_index": ci,
                "ci_label": f"CI={ci:.2f}",
                "discussion_turns": int(config_payload.get("discussion_turns", 0)),
                "output_dir": str(output_dir),
                "config_file": row["config_file"],
                "result_path": str(result_path) if result_path is not None else "",
                "interactions_path": str(interactions_path) if interactions_path is not None else "",
                "status": status,
            }
            index_row["selection_label"] = (
                f"#{index_row['experiment_id']:04d} | {model_short_name(model2)} | "
                f"{index_row['model_order']} | rho={rho:.1f} | theta={theta:.1f} | "
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
            baseline_utility = utility_by_model.get(baseline_model)
            adversary_utility = utility_by_model.get(adversary_model)
            social_welfare = sum(float(value) for value in final_utilities.values())

            experiment_row = {
                "experiment_id": int(row["experiment_id"]),
                "pair_label": f"{model1} vs {model2}",
                "baseline_model": baseline_model,
                "adversary_model": adversary_model,
                "baseline_short": model_short_name(baseline_model),
                "adversary_short": model_short_name(adversary_model),
                "baseline_elo": model_elo(baseline_model),
                "adversary_elo": model_elo(adversary_model),
                "model_order": row["model_order"],
                "rho": rho,
                "theta": theta,
                "competition_index": ci,
                "ci_label": f"CI={ci:.2f}",
                "baseline_utility": baseline_utility,
                "adversary_utility": adversary_utility,
                "social_welfare": social_welfare,
                "consensus_reached": bool(result_payload.get("consensus_reached", False)),
                "final_round": int(result_payload.get("final_round", -1) or -1),
                "output_dir": str(output_dir),
            }
            experiment_rows.append(experiment_row)

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
                        "rho": rho,
                        "theta": theta,
                        "competition_index": ci,
                        "ci_label": f"CI={ci:.2f}",
                        "adversary_model": adversary_model,
                        "adversary_short": model_short_name(adversary_model),
                    }
                )

    index_df = pd.DataFrame(index_rows).sort_values(
        ["model2", "model_order", "rho", "theta", "experiment_id"]
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
        formatted = format_func(value)
        return str(formatted)

    return st.sidebar.selectbox(
        label,
        options=options,
        format_func=_format,
    )


def filter_index(index_df: pd.DataFrame) -> pd.DataFrame:
    status_options = _sort_maybe_numeric(index_df["status"].unique().tolist())
    model_options = _sort_maybe_numeric(index_df["model2"].unique().tolist())
    order_options = _sort_maybe_numeric(index_df["model_order"].unique().tolist())
    rho_options = _sort_maybe_numeric(index_df["rho"].unique().tolist())
    theta_options = _sort_maybe_numeric(index_df["theta"].unique().tolist())
    ci_options = _sort_maybe_numeric(index_df["competition_index"].unique().tolist())

    filters = {
        "status": sidebar_select_filter("Status", status_options),
        "model2": sidebar_select_filter("Adversary", model_options, model_short_name),
        "model_order": sidebar_select_filter("Model Order", order_options),
        "rho": sidebar_select_filter(
            "rho",
            rho_options,
            format_func=lambda value: "Any" if value is None else f"{float(value):.1f}",
        ),
        "theta": sidebar_select_filter(
            "theta",
            theta_options,
            format_func=lambda value: "Any" if value is None else f"{float(value):.1f}",
        ),
        "competition_index": sidebar_select_filter(
            "Competition Index",
            ci_options,
            format_func=lambda value: "Any" if value is None else f"{float(value):.2f}",
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
    if "pending" not in table.columns:
        table["pending"] = 0
    table["adversary"] = table["model2"].map(model_short_name)
    return table[["adversary", "model_order", "completed", "pending"]].sort_values(
        ["adversary", "model_order"]
    )


def render_overview_metrics(index_df: pd.DataFrame, experiment_df: pd.DataFrame) -> None:
    completed = int((index_df["status"] == "completed").sum())
    pending = int((index_df["status"] != "completed").sum())
    avg_welfare = (
        float(experiment_df["social_welfare"].mean()) if not experiment_df.empty else 0.0
    )

    columns = st.columns(4)
    columns[0].metric("Total Runs", str(len(index_df)))
    columns[1].metric("Completed", str(completed))
    columns[2].metric("Pending", str(pending))
    columns[3].metric("Mean Social Welfare", f"{avg_welfare:.2f}")

    st.caption(
        "The scalar competition proxy shown in this viewer is "
        "`theta * (1 - rho) / 2`."
    )
    st.dataframe(coverage_table(index_df), hide_index=True, use_container_width=True)


def _format_config_value(value: Any, *, numeric_decimals: Optional[int] = None) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "N/A"
    if numeric_decimals is not None:
        return f"{float(value):.{numeric_decimals}f}"
    return str(value)


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
    overall_fig.update_traces(textposition="top center", marker=dict(size=13, line=dict(width=1, color="#1f2937")))
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
    st.subheader("Utility vs theta, rho, and the scalar competition proxy")

    if long_df.empty:
        st.info("No completed run data available yet.")
        return

    theta_df = (
        long_df.groupby(["theta", "model_short"], as_index=False)
        .agg(avg_utility=("utility", "mean"))
        .sort_values(["model_short", "theta"])
    )
    theta_fig = px.line(
        theta_df,
        x="theta",
        y="avg_utility",
        color="model_short",
        markers=True,
        title="Average utility vs theta",
    )
    _configure_figure_layout(theta_fig, "theta", "Average Utility")

    rho_df = (
        long_df.groupby(["rho", "model_short"], as_index=False)
        .agg(avg_utility=("utility", "mean"))
        .sort_values(["model_short", "rho"])
    )
    rho_fig = px.line(
        rho_df,
        x="rho",
        y="avg_utility",
        color="model_short",
        markers=True,
        title="Average utility vs rho",
    )
    _configure_figure_layout(rho_fig, "rho", "Average Utility")

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
        st.plotly_chart(theta_fig, use_container_width=True)
    with top_row[1]:
        st.plotly_chart(rho_fig, use_container_width=True)

    st.plotly_chart(ci_fig, use_container_width=True)


def _heatmap_figure(
    data: pd.DataFrame,
    value_col: str,
    title: str,
    z_title: str,
    color_scale: str = "RdBu",
) -> go.Figure:
    pivot = (
        data.pivot(index="rho", columns="theta", values=value_col)
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
            hovertemplate="rho=%{y}<br>theta=%{x}<br>value=%{z:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        template=PLOT_TEMPLATE,
        title=title,
        xaxis_title="theta",
        yaxis_title="rho",
        height=420,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def render_heatmaps(experiment_df: pd.DataFrame, selected_row: pd.Series) -> None:
    st.subheader("rho × theta views")

    if experiment_df.empty:
        st.info("No completed run data available yet.")
        return

    selected_adversary = selected_row["model2"]
    selected_pair_df = experiment_df[experiment_df["adversary_model"] == selected_adversary]
    if selected_pair_df.empty:
        st.info("No completed runs are available yet for the selected adversary model.")
        return

    welfare_df = (
        experiment_df.groupby(["rho", "theta"], as_index=False)
        .agg(social_welfare=("social_welfare", "mean"))
    )
    pair_df = (
        selected_pair_df.groupby(["rho", "theta"], as_index=False)
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
                pair_df,
                value_col="baseline_utility",
                title=f"{model_short_name(selected_row['model1'])} utility vs rho × theta for {model_short_name(selected_adversary)}",
                z_title="Utility",
            ),
            use_container_width=True,
        )

    st.plotly_chart(
        _heatmap_figure(
            pair_df,
            value_col="adversary_utility",
            title=f"{model_short_name(selected_adversary)} utility vs rho × theta",
            z_title="Utility",
        ),
        use_container_width=True,
    )


def render_selected_config_table(selected_row: pd.Series) -> None:
    rows = [
        {"name": "experiment_id", "value": str(int(selected_row["experiment_id"]))},
        {"name": "source_experiment_id", "value": _format_config_value(selected_row.get("source_experiment_id"))},
        {"name": "pair", "value": str(selected_row["pair_label"])},
        {"name": "model_order", "value": str(selected_row["model_order"])},
        {"name": "rho", "value": _format_config_value(selected_row["rho"], numeric_decimals=1)},
        {"name": "theta", "value": _format_config_value(selected_row["theta"], numeric_decimals=1)},
        {
            "name": "competition_index",
            "value": _format_config_value(selected_row["competition_index"], numeric_decimals=2),
        },
        {"name": "discussion_turns", "value": _format_config_value(selected_row["discussion_turns"])},
        {"name": "seed", "value": _format_config_value(selected_row.get("seed"))},
        {"name": "status", "value": str(selected_row["status"])},
        {"name": "config_file", "value": _format_config_value(selected_row.get("config_file"))},
        {"name": "output_dir", "value": str(selected_row["output_dir"])},
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def render_selected_run(selected_row: pd.Series, show_prompts: bool, show_private: bool) -> None:
    st.subheader("Selected Run")
    render_selected_config_table(selected_row)

    if selected_row["status"] != "completed":
        st.warning(
            "This run does not have result artifacts yet. "
            "Pick a completed run from the sidebar to inspect the full rollout."
        )
        return

    results_dir = selected_row["output_dir"]
    results = load_experiment_results(results_dir)
    if results is None:
        st.error(f"Could not load experiment results from: `{results_dir}`")
        return

    interactions = load_all_interactions(results_dir)
    config = results.get("config", {})
    game_type = config.get("game_type", "diplomacy")
    experiment_id = f"game2_batch_{selected_row['experiment_id']}"

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

    st.set_page_config(
        page_title="Game 2 Viewer",
        page_icon="S",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title(GAME_TYPES.get("diplomacy", "Game 2 Diplomacy"))
    if "game2_source_path" not in st.session_state:
        st.session_state["game2_source_path"] = _default_source_input(args.results_dir)
    st.text_input(
        "Results folder",
        key="game2_source_path",
        help=(
            "Paste a Game 2 batch root, its `configs/` directory, or a single completed run "
            "folder with `experiment_results.json`."
        ),
    )

    raw_source = str(st.session_state.get("game2_source_path", "")).strip()
    if not raw_source:
        st.error("Enter a Game 2 results folder to continue.")
        st.stop()

    try:
        results_root = resolve_results_root(raw_source)
        source_mode = _results_source_mode(results_root)
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    index_df, experiment_df, long_df = load_batch_data(str(results_root))
    if index_df.empty:
        st.error("No Game 2 runs were found under the selected source path.")
        st.stop()

    st.sidebar.title("Game 2 Viewer")
    st.sidebar.markdown("---")
    st.sidebar.caption(f"Source mode: {source_mode.replace('_', ' ')}")
    st.sidebar.caption(str(results_root))
    if st.sidebar.button("Refresh data", use_container_width=True):
        load_batch_data.clear()
        load_experiment_results.clear()
        load_all_interactions.clear()
        st.rerun()
    show_prompts = st.sidebar.checkbox("Show prompts", value=False, key="batch_show_prompts")
    show_private = st.sidebar.checkbox(
        "Show private thinking", value=True, key="batch_show_private"
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Filter Runs**")

    filtered_index = filter_index(index_df)
    st.sidebar.caption(f"{len(filtered_index)} matching runs")
    if filtered_index.empty:
        st.warning("No runs match the current sidebar filters.")
        st.stop()

    selection_options = filtered_index["selection_label"].tolist()
    if st.session_state.get("batch_selected_config") not in selection_options:
        st.session_state["batch_selected_config"] = selection_options[0]

    selected_label = st.sidebar.selectbox(
        "Matching Run",
        options=selection_options,
        key="batch_selected_config",
    )
    selected_row = filtered_index.loc[
        filtered_index["selection_label"] == selected_label
    ].iloc[0]

    if selected_row["status"] == "completed":
        selected_results = load_experiment_results(selected_row["output_dir"])
        if selected_results is not None:
            render_sidebar_agent_info(selected_results, selected_results.get("config", {}).get("game_type", "diplomacy"))

    st.caption(
        f"Results source: `{results_root}` | Selected run: `{selected_row['selection_label']}`"
    )

    active_main_view = st.radio(
        "Game 2 view",
        ["Selected Run", "Summary"],
        horizontal=True,
        label_visibility="collapsed",
        key="game2_batch_main_view",
    )
    st.markdown("---")

    if active_main_view == "Summary":
        render_overview_metrics(index_df, experiment_df)
        render_elo_plots(long_df)
        render_parameter_plots(long_df)
        render_heatmaps(experiment_df, selected_row)
    else:
        render_selected_run(selected_row, show_prompts, show_private)


if __name__ == "__main__":
    main()
