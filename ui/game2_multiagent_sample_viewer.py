"""
Streamlit viewer for the five Game 2 multi-agent sample tests.

Usage:
    streamlit run ui/game2_multiagent_sample_viewer.py -- --results-dir <game2_samples_dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--results-dir", type=str, default=None)
    args, _ = parser.parse_known_args(sys.argv[1:])
    return args


def latest_samples_dir() -> Optional[Path]:
    candidates = sorted(
        path for path in RESULTS_ROOT.glob("game2_samples_*") if path.is_dir()
    )
    return candidates[-1] if candidates else None


def resolve_results_dir(raw_value: Optional[str]) -> Path:
    if raw_value:
        candidate = Path(raw_value).expanduser()
        if not candidate.is_absolute():
            candidate = (Path.cwd() / candidate)
        return candidate.resolve()

    latest = latest_samples_dir()
    if latest is None:
        raise FileNotFoundError(f"No game2_samples_* directories found under {RESULTS_ROOT}")
    return latest.resolve()


def result_file(output_dir: Path) -> Optional[Path]:
    for name in ("run_1_experiment_results.json", "experiment_results.json"):
        candidate = output_dir / name
        if candidate.exists():
            return candidate
    return None


def load_manifest(results_dir: Path) -> Dict[str, Any]:
    manifest_path = results_dir / "manifest.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    return {"samples": []}


def sample_rows(results_dir: Path, manifest: Dict[str, Any]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for sample in manifest.get("samples", []):
        output_dir = Path(sample["output_dir"])
        if not output_dir.is_absolute():
            output_dir = (PROJECT_ROOT / output_dir).resolve()
        res_file = result_file(output_dir)
        status = "completed" if res_file else "pending"
        consensus = None
        final_round = None
        rho_empirical = None
        theta_empirical = None
        if res_file:
            payload = json.loads(res_file.read_text(encoding="utf-8"))
            consensus = payload.get("consensus_reached")
            final_round = payload.get("final_round")
            diag = payload.get("config", {}).get("game2_parameter_diagnostics", {})
            rho_empirical = diag.get("empirical_rho_spearman", {}).get("mean")
            theta_empirical = diag.get("empirical_theta_cosine", {}).get("mean")

        rows.append({
            "id": int(sample["experiment_id"]),
            "sample": sample["sample_name"],
            "type": sample["experiment_type"],
            "status": status,
            "adversary": sample.get("adversary_model") or "none",
            "position": sample.get("adversary_position") or "random",
            "n_agents": sample["n_agents"],
            "n_issues": sample["n_issues"],
            "rho": sample["rho"],
            "theta": sample["theta"],
            "empirical_rho_spearman_mean": rho_empirical,
            "empirical_theta_cosine_mean": theta_empirical,
            "consensus": consensus,
            "final_round": final_round,
            "output_dir": str(output_dir),
        })
    return pd.DataFrame(rows).sort_values("id") if rows else pd.DataFrame()


def render_sample_metadata(results: Dict[str, Any]) -> None:
    config = results.get("config", {})
    metadata = config.get("sample_metadata", {})
    diagnostics = config.get("game2_parameter_diagnostics", {})

    st.subheader("Sample Metadata")
    cols = st.columns(4)
    cols[0].metric("N", str(metadata.get("n_agents", config.get("n_agents", "?"))))
    cols[1].metric("Issues", str(metadata.get("n_issues", config.get("n_issues", "?"))))
    cols[2].metric("Commanded rho", f"{float(config.get('rho', 0.0)):.4f}")
    cols[3].metric("Commanded theta", f"{float(config.get('theta', 0.0)):.4f}")

    diag_rows = []
    for name, payload in diagnostics.items():
        if name == "commanded" or not isinstance(payload, dict):
            continue
        diag_rows.append({
            "metric": name,
            "mean": payload.get("mean"),
            "std": payload.get("std"),
            "min": payload.get("min"),
            "p50": payload.get("p50"),
            "max": payload.get("max"),
            "count": payload.get("count"),
        })
    if diag_rows:
        st.dataframe(pd.DataFrame(diag_rows), hide_index=True, use_container_width=True)

    agent_map = metadata.get("agent_model_map") or config.get("agent_model_map", {})
    elo_map = metadata.get("agent_elo_map") or config.get("agent_elo_map", {})
    if agent_map:
        st.markdown("**Agent Model Map**")
        st.dataframe(
            pd.DataFrame(
                [{"agent": agent, "model": model, "elo": elo_map.get(agent)} for agent, model in agent_map.items()]
            ),
            hide_index=True,
            use_container_width=True,
        )


def main() -> None:
    args = parse_args()
    results_dir = resolve_results_dir(args.results_dir)
    manifest = load_manifest(results_dir)
    rows = sample_rows(results_dir, manifest)

    st.set_page_config(
        page_title="Game 2 Multi-Agent Samples",
        page_icon="S",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("Game 2 Multi-Agent Samples")
    st.caption(f"`{results_dir}`")

    if rows.empty:
        st.error("No samples found in manifest.")
        return

    st.sidebar.title("Samples")
    st.sidebar.caption(str(results_dir))
    show_prompts = st.sidebar.checkbox("Show prompts", value=False)
    show_private = st.sidebar.checkbox("Show private thinking", value=True)
    labels = [
        f"#{int(row['id']):02d} | {row['sample']} | {row['status']}"
        for _, row in rows.iterrows()
    ]
    selected_label = st.sidebar.selectbox("Sample", labels)
    selected_id = int(selected_label.split("|", 1)[0].strip().lstrip("#"))
    selected = rows[rows["id"] == selected_id].iloc[0]

    st.subheader("Run Set")
    st.dataframe(rows.drop(columns=["output_dir"]), hide_index=True, use_container_width=True)

    if selected["status"] != "completed":
        st.warning("Selected sample has no completed result yet.")
        return

    results = load_experiment_results(selected["output_dir"])
    if results is None:
        st.error(f"Could not load results from `{selected['output_dir']}`")
        return
    interactions = load_all_interactions(selected["output_dir"])
    game_type = results.get("config", {}).get("game_type", "diplomacy")
    render_sidebar_agent_info(results, game_type)

    st.markdown("---")
    render_sample_metadata(results)
    st.markdown("---")
    render_metrics_overview(results, game_type)
    st.markdown("---")

    tab = st.radio(
        "View",
        ["Timeline", "Agent Comparison", "Analytics", "Raw Data"],
        horizontal=True,
        label_visibility="collapsed",
    )
    st.markdown("---")
    experiment_id = f"game2_sample_{selected_id}"
    if tab == "Timeline":
        render_timeline_tab(results, game_type, show_prompts, show_private, experiment_id, interactions)
    elif tab == "Agent Comparison":
        render_agent_comparison_tab(results, interactions, game_type, show_prompts, experiment_id)
    elif tab == "Analytics":
        render_analytics_tab(results, game_type, experiment_id, interactions)
    else:
        render_raw_data_tab(results, interactions, experiment_id)


if __name__ == "__main__":
    main()
