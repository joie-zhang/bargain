"""
Dedicated Streamlit viewer for a single Game 3 co-funding run.

Usage:
    streamlit run ui/game3_sample_viewer.py -- --results-dir <results_dir>
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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


def normalize_target_dirs(raw_value: str) -> Tuple[Path, Path]:
    candidate = resolve_results_dir(raw_value)
    if candidate.name == "agent_interactions":
        return candidate.parent.resolve(), candidate.resolve()
    return candidate.resolve(), (candidate / "agent_interactions").resolve()


def canonical_phase(raw_phase: str) -> str:
    phase = raw_phase.lower().strip()
    if "discussion" in phase:
        return "discussion"
    if phase.startswith("private_thinking"):
        return "private_thinking"
    if phase.startswith("proposal"):
        return "proposal"
    if phase.startswith("voting"):
        return "voting"
    if phase.startswith("reflection"):
        return "reflection"
    if phase.startswith("pledge"):
        return "pledge_submission"
    if phase.startswith("aggregate_funding"):
        return "aggregate_funding"
    if "commit_vote" in phase and "summary" not in phase:
        return "cofunding_commit_vote"
    if "commit_vote" in phase and "summary" in phase:
        return "cofunding_commit_vote_summary"
    if "feedback" in phase:
        return "feedback"
    return phase


def _parse_response_payload(response: Any) -> Dict[str, Any]:
    if isinstance(response, dict):
        return response
    if not isinstance(response, str):
        return {}
    try:
        parsed = json.loads(response)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def load_agent_interaction_payloads(agent_dir: Path) -> Dict[str, Dict[str, Any]]:
    payloads: Dict[str, Dict[str, Any]] = {}
    if not agent_dir.exists():
        return payloads

    for path in sorted(agent_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            mtime = path.stat().st_mtime
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue

        agent_id = str(payload.get("agent_id") or path.stem)
        interactions = payload.get("interactions", [])
        total_interactions = int(payload.get("total_interactions") or len(interactions))
        candidate = {
            "path": str(path),
            "payload": payload,
            "total_interactions": total_interactions,
            "mtime": mtime,
        }
        existing = payloads.get(agent_id)
        if existing is None or (
            candidate["total_interactions"],
            candidate["mtime"],
        ) >= (
            existing["total_interactions"],
            existing["mtime"],
        ):
            payloads[agent_id] = candidate

    return payloads


def combine_agent_interactions(agent_payloads: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    combined: List[Dict[str, Any]] = []
    seen = set()

    for payload_info in agent_payloads.values():
        interactions = payload_info["payload"].get("interactions", [])
        if not isinstance(interactions, list):
            continue
        for entry in interactions:
            if not isinstance(entry, dict):
                continue
            identity = (
                entry.get("timestamp"),
                entry.get("agent_id"),
                entry.get("phase"),
                entry.get("round"),
                entry.get("response"),
            )
            if identity in seen:
                continue
            seen.add(identity)
            combined.append(entry)

    combined.sort(
        key=lambda item: (
            float(item.get("timestamp", 0) or 0),
            int(item.get("round", 0) or 0),
            str(item.get("agent_id", "")),
        )
    )
    return combined


def parse_path_float(run_dir: Path, key: str) -> Optional[float]:
    match = re.search(rf"{key}_(n?\d+)_(\d+)", str(run_dir))
    if not match:
        return None
    integer = match.group(1)
    negative = integer.startswith("n")
    if negative:
        integer = integer[1:]
    value = float(f"{integer}.{match.group(2)}")
    return -value if negative else value


def parse_setup_metadata(
    interactions: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, List[float]], Dict[str, float], Optional[float], Optional[int]]:
    items: List[Dict[str, Any]] = []
    item_index: Dict[str, int] = {}
    preferences: Dict[str, List[float]] = {}
    budgets: Dict[str, float] = {}
    gamma_value: Optional[float] = None
    max_rounds: Optional[int] = None

    project_pattern = re.compile(
        r"^\s*-\s+(?P<name>.+?):\s*cost\s*=\s*(?P<cost>[-\d.]+)\s*$",
        re.MULTILINE,
    )
    valuation_pattern = re.compile(
        r"^\s*(?:- )?(?P<name>.+?)\s+\(cost:\s*(?P<cost>[-\d.]+)\):\s*Your valuation =\s*(?P<value>[-\d.]+)",
        re.MULTILINE,
    )

    def ensure_item(name: str, cost: float) -> int:
        existing = item_index.get(name)
        if existing is not None:
            return existing
        item_index[name] = len(items)
        items.append({"name": name, "cost": float(cost)})
        for values in preferences.values():
            values.append(0.0)
        return item_index[name]

    for entry in interactions:
        prompt = str(entry.get("prompt") or "")
        if not prompt:
            continue

        if gamma_value is None:
            gamma_match = re.search(r"gamma\s*=\s*([-\d.]+)", prompt)
            if gamma_match:
                gamma_value = float(gamma_match.group(1))

        if max_rounds is None:
            rounds_match = re.search(
                r"game lasts up to\s+(\d+)\s+rounds",
                prompt,
                re.IGNORECASE,
            )
            if rounds_match:
                max_rounds = int(rounds_match.group(1))

        for match in project_pattern.finditer(prompt):
            ensure_item(match.group("name").strip(), float(match.group("cost")))

        agent_id = str(entry.get("agent_id") or "")
        budget_match = re.search(r"\*\*YOUR(?: PRIVATE)? BUDGET:\*\*\s*([-\d.]+)", prompt)
        if agent_id and budget_match:
            budgets[agent_id] = float(budget_match.group(1))

        valuation_matches = list(valuation_pattern.finditer(prompt))
        if agent_id and valuation_matches:
            for match in valuation_matches:
                ensure_item(match.group("name").strip(), float(match.group("cost")))

            values = preferences.setdefault(agent_id, [0.0] * len(items))
            if len(values) < len(items):
                values.extend([0.0] * (len(items) - len(values)))

            for match in valuation_matches:
                idx = item_index[match.group("name").strip()]
                values[idx] = float(match.group("value"))

    return items, preferences, budgets, gamma_value, max_rounds


def build_partial_results(run_dir: Path, interactions: List[Dict[str, Any]]) -> Dict[str, Any]:
    items, preferences, budgets, gamma_value, max_rounds = parse_setup_metadata(interactions)
    agents = sorted(
        {
            str(entry.get("agent_id"))
            for entry in interactions
            if entry.get("agent_id")
        }
    )
    model_by_agent: Dict[str, str] = {}
    for entry in interactions:
        agent_id = entry.get("agent_id")
        model_name = entry.get("model_name")
        if agent_id and model_name and agent_id not in model_by_agent:
            model_by_agent[str(agent_id)] = str(model_name)

    logs: List[Dict[str, Any]] = []
    for entry in interactions:
        phase = canonical_phase(str(entry.get("phase", "")))
        response = entry.get("response", "")
        log_entry: Dict[str, Any] = {
            "phase": phase,
            "round": int(entry.get("round", 0) or 0),
            "from": entry.get("agent_id", "system"),
            "content": response,
            "timestamp": entry.get("timestamp"),
            "prompt": entry.get("prompt", ""),
        }
        parsed = _parse_response_payload(response)
        if phase in {"proposal", "pledge_submission"} and parsed:
            field = "proposal" if phase == "proposal" else "pledge"
            log_entry[field] = parsed
        if phase == "cofunding_commit_vote" and parsed:
            log_entry["parsed_vote"] = parsed
        logs.append(log_entry)

    logs.sort(
        key=lambda item: (
            int(item.get("round", 0) or 0),
            float(item.get("timestamp", 0) or 0),
            str(item.get("from", "")),
        )
    )

    config: Dict[str, Any] = {
        "game_type": "co_funding",
        "agents": agents,
        "items": items,
        "m_projects": len(items),
        "agent_budgets": budgets,
    }
    alpha = parse_path_float(run_dir, "alpha")
    sigma = parse_path_float(run_dir, "sigma")
    if alpha is not None:
        config["alpha"] = alpha
    if sigma is not None:
        config["sigma"] = sigma
    if gamma_value is not None:
        config["cofunding_time_discount"] = gamma_value
    if max_rounds is not None:
        config["max_rounds"] = max_rounds

    final_round = max((int(entry.get("round", 0) or 0) for entry in interactions), default=0)
    return {
        "config": config,
        "conversation_logs": logs,
        "agent_preferences": preferences,
        "agent_performance": {
            agent: {"model": model_by_agent.get(agent, "unknown")}
            for agent in agents
        },
        "final_utilities": {},
        "consensus_reached": False,
        "final_round": final_round,
        "strategic_behaviors": {},
        "exploitation_detected": False,
        "final_allocation": [],
        "partial_interactions_only": True,
    }


def render_partial_overview(
    results: Dict[str, Any],
    interactions: List[Dict[str, Any]],
    agent_payloads: Dict[str, Dict[str, Any]],
) -> None:
    agents = results.get("config", {}).get("agents", [])
    current_round = max((int(entry.get("round", 0) or 0) for entry in interactions), default=0)
    last_phase_raw = str(interactions[-1].get("phase", "unknown")) if interactions else "unknown"
    last_phase = canonical_phase(last_phase_raw)

    st.info(
        "Interactions-only mode: no `experiment_results.json` was found for this run, "
        "so the UI is rendering from `run_1_all_interactions.json` and the per-agent files."
    )

    columns = st.columns(5)
    columns[0].metric("Saved Exchanges", str(len(interactions)))
    columns[1].metric("Agents", str(len(agents)))
    columns[2].metric("Rounds Seen", str(current_round))
    columns[3].metric("Last Phase", last_phase)
    columns[4].metric("Agent Files", str(len(agent_payloads)))


def render_partial_raw_data(
    results: Dict[str, Any],
    interactions: List[Dict[str, Any]],
    agent_payloads: Dict[str, Dict[str, Any]],
    experiment_id: str,
) -> None:
    options = [
        "Synthetic Results",
        "All Interactions",
        "Agent Interaction Files",
        "Config",
        "Preferences",
    ]
    choice = st.selectbox(
        "Select data",
        options,
        key=f"partial_raw_data_select_{experiment_id}",
    )

    if choice == "Synthetic Results":
        st.json({k: v for k, v in results.items() if k != "conversation_logs"})
    elif choice == "All Interactions":
        st.caption(f"{len(interactions)} total entries")
        st.json(interactions)
    elif choice == "Agent Interaction Files":
        if not agent_payloads:
            st.warning("No per-agent interaction files were found.")
            return
        selected_agent = st.selectbox(
            "Agent file",
            sorted(agent_payloads.keys()),
            key=f"agent_payload_select_{experiment_id}",
        )
        payload_info = agent_payloads[selected_agent]
        st.caption(payload_info["path"])
        st.json(payload_info["payload"])
    elif choice == "Config":
        st.json(results.get("config", {}))
    elif choice == "Preferences":
        st.json(results.get("agent_preferences", {}))


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
    results_dir, agent_dir = normalize_target_dirs(args.results_dir)

    st.set_page_config(
        page_title="Game 3 Sample Viewer",
        page_icon="S",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    results = load_experiment_results(str(results_dir))
    interactions = load_all_interactions(str(results_dir))
    agent_payloads = load_agent_interaction_payloads(agent_dir)
    if not interactions:
        interactions = combine_agent_interactions(agent_payloads)

    partial_mode = results is None and bool(interactions)
    if partial_mode:
        results = build_partial_results(results_dir, interactions)
    elif results is None:
        st.error(f"Could not load experiment results or interactions from: `{results_dir}`")
        return

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
    caption_path = agent_dir if agent_dir.exists() else results_dir
    st.caption(f"`{caption_path}`")

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
                    (
                        f"**competition_index** = {competition_index:.3f}"
                        if competition_index is not None else ""
                    ),
                    f"**gamma** = {gamma}" if gamma is not None else "",
                ]
            ).strip(" |")
        )

    if partial_mode:
        render_partial_overview(results, interactions, agent_payloads)
    else:
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
        if partial_mode:
            render_partial_raw_data(results, interactions, agent_payloads, experiment_id)
        else:
            render_raw_data_tab(results, interactions, experiment_id)


if __name__ == "__main__":
    main()
