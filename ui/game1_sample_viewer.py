"""
Dedicated Streamlit viewer for Game 1 runs.

Supports:
- a single sample/results directory
- a scaling-experiment root with sidebar filtering across many runs

Usage:
    streamlit run ui/game1_sample_viewer.py -- --results-dir <results_dir>
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st


RESULTS_ROOT = Path(__file__).resolve().parent.parent / "experiments" / "results"
PROJECT_ROOT = Path(__file__).resolve().parent.parent

PHASE_SEQUENCE = [
    "game_setup",
    "discussion",
    "private_thinking",
    "proposal",
    "proposal_enumeration",
    "voting",
    "vote_tabulation",
    "reflection",
]

PHASE_LABELS = {
    "game_setup": "Combined Setup",
    "discussion": "Discussion",
    "private_thinking": "Private Thinking",
    "proposal": "Formal Proposals",
    "proposal_enumeration": "Proposal Enumeration",
    "voting": "Batch Voting",
    "vote_tabulation": "Vote Tabulation",
    "reflection": "Reflection",
}

PHASE_COLORS = {
    "game_setup": "#4f46e5",
    "discussion": "#2563eb",
    "private_thinking": "#475569",
    "proposal": "#059669",
    "proposal_enumeration": "#0f766e",
    "voting": "#d97706",
    "vote_tabulation": "#dc2626",
    "reflection": "#be185d",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--results-dir", type=str, default=None)
    args, _ = parser.parse_known_args(sys.argv[1:])
    return args


def normalize_phase(phase: str) -> str:
    phase_lower = phase.lower()
    if phase_lower.startswith("discussion"):
        return "discussion"
    if phase_lower.startswith("private_thinking"):
        return "private_thinking"
    if phase_lower.startswith("proposal_round"):
        return "proposal"
    if phase_lower.startswith("voting_round"):
        return "voting"
    if phase_lower.startswith("reflection_round"):
        return "reflection"
    return phase_lower


def extract_discussion_turn(phase: str) -> Optional[int]:
    match = re.search(r"_turn_(\d+)$", phase or "")
    if match:
        return int(match.group(1))
    return None


def latest_sample_dir() -> Optional[Path]:
    candidates = sorted(RESULTS_ROOT.glob("game1_sample_*"))
    return candidates[-1] if candidates else None


def is_scaling_experiment_root(path: Path) -> bool:
    return (path / "configs" / "experiment_index.csv").exists()


def resolve_results_dir(raw_value: Optional[str]) -> Path:
    if raw_value:
        candidate = Path(raw_value)
        if not candidate.is_absolute():
            candidate = (Path.cwd() / candidate).resolve()
        return candidate

    latest = latest_sample_dir()
    if latest is None:
        raise FileNotFoundError(
            f"No game1_sample_* directories found under {RESULTS_ROOT}"
        )
    return latest.resolve()


def _load_first_existing(paths: List[Path]) -> Any:
    for path in paths:
        if path.exists():
            with open(path, "r", encoding="utf-8") as handle:
                return json.load(handle)
    raise FileNotFoundError(f"Could not find any of: {paths}")


@st.cache_data(show_spinner=False)
def load_sample(results_dir: str) -> Dict[str, Any]:
    base = Path(results_dir)
    interactions = _load_first_existing(
        [
            base / "run_1_all_interactions.json",
            base / "all_interactions.json",
        ]
    )
    results = _load_first_existing(
        [
            base / "run_1_experiment_results.json",
            base / "experiment_results.json",
        ]
    )
    return {
        "base": str(base),
        "interactions": interactions,
        "results": results,
    }


def _resolve_output_dir(raw_output_dir: str) -> Path:
    candidate = Path(raw_output_dir)
    if candidate.is_absolute():
        return candidate.resolve()
    return (PROJECT_ROOT / candidate).resolve()


def _artifacts_exist(results_dir: Path) -> bool:
    return any(
        path.exists()
        for path in [
            results_dir / "run_1_all_interactions.json",
            results_dir / "all_interactions.json",
        ]
    ) and any(
        path.exists()
        for path in [
            results_dir / "run_1_experiment_results.json",
            results_dir / "experiment_results.json",
        ]
    )


@st.cache_data(show_spinner=False)
def load_scaling_index(root_dir: str) -> pd.DataFrame:
    root = Path(root_dir)
    index_path = root / "configs" / "experiment_index.csv"
    rows: List[Dict[str, Any]] = []

    with open(index_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            config_file = root / "configs" / row["config_file"]
            config_payload = json.loads(config_file.read_text(encoding="utf-8"))
            output_dir = _resolve_output_dir(config_payload["output_dir"])
            if not _artifacts_exist(output_dir):
                continue

            row_payload: Dict[str, Any] = {
                "experiment_id": int(config_payload["experiment_id"]),
                "weak_model": config_payload["weak_model"],
                "strong_model": config_payload["strong_model"],
                "model_order": config_payload["model_order"],
                "competition_level": float(config_payload["competition_level"]),
                "discussion_turns": int(config_payload["discussion_turns"]),
                "run_number": int(config_payload["run_number"]),
                "random_seed": int(config_payload["random_seed"]),
                "num_items": int(config_payload["num_items"]),
                "max_rounds": int(config_payload["max_rounds"]),
                "output_dir": str(output_dir),
                "config_file": row["config_file"],
            }
            row_payload["pair_label"] = (
                f"{row_payload['weak_model']} vs {row_payload['strong_model']}"
            )
            row_payload["selection_label"] = (
                f"{row_payload['pair_label']} | "
                f"comp={row_payload['competition_level']:.1f} | "
                f"order={row_payload['model_order']} | "
                f"turns={row_payload['discussion_turns']} | "
                f"items={row_payload['num_items']} | "
                f"rounds={row_payload['max_rounds']} | "
                f"seed={row_payload['random_seed']} | "
                f"run={row_payload['run_number']}"
            )
            rows.append(row_payload)

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values(
        [
            "weak_model",
            "strong_model",
            "competition_level",
            "model_order",
            "discussion_turns",
            "run_number",
        ]
    ).reset_index(drop=True)


def sidebar_select_filter(
    label: str,
    values: List[Any],
    format_func=None,
) -> Any:
    options = [None] + list(values)
    return st.sidebar.selectbox(
        label,
        options=options,
        format_func=(
            format_func
            if format_func is not None
            else lambda value: "Any" if value is None else str(value)
        ),
    )


def filter_experiment_index(index_frame: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    filters = {
        "weak_model": sidebar_select_filter(
            "Weak Model",
            sorted(index_frame["weak_model"].unique().tolist()),
        ),
        "strong_model": sidebar_select_filter(
            "Strong Model",
            sorted(index_frame["strong_model"].unique().tolist()),
        ),
        "model_order": sidebar_select_filter(
            "Model Order",
            sorted(index_frame["model_order"].unique().tolist()),
        ),
        "competition_level": sidebar_select_filter(
            "Competition Level",
            sorted(index_frame["competition_level"].unique().tolist()),
            format_func=lambda value: "Any" if value is None else f"{float(value):.1f}",
        ),
        "discussion_turns": sidebar_select_filter(
            "Discussion Turns",
            sorted(index_frame["discussion_turns"].unique().tolist()),
        ),
        "num_items": sidebar_select_filter(
            "Num Items",
            sorted(index_frame["num_items"].unique().tolist()),
        ),
        "max_rounds": sidebar_select_filter(
            "Max Rounds",
            sorted(index_frame["max_rounds"].unique().tolist()),
        ),
        "random_seed": sidebar_select_filter(
            "Random Seed",
            sorted(index_frame["random_seed"].unique().tolist()),
        ),
    }

    filtered = index_frame.copy()
    for column, selected_value in filters.items():
        if selected_value is not None:
            filtered = filtered[filtered[column] == selected_value]

    return filtered.reset_index(drop=True), filters


def render_prompt_block(title: str, prompt: str) -> None:
    with st.expander(title, expanded=False):
        st.code(prompt)


def agent_sort_key(agent_id: str) -> tuple[int, str]:
    match = re.search(r"(\d+)$", agent_id)
    if match:
        return (0, int(match.group(1)))
    return (1, agent_id)


def extract_agent_models(interactions: List[Dict[str, Any]]) -> Dict[str, str]:
    models: Dict[str, str] = {}
    for entry in interactions:
        agent_id = entry.get("agent_id")
        model_name = entry.get("model_name")
        if agent_id and model_name and agent_id not in models:
            models[agent_id] = model_name
    return models


def parse_setup_preferences(prompt: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    pattern = re.compile(
        r"^\s*(\d+):\s*(.+?)\s*->\s*([0-9]+(?:\.[0-9]+)?)(?:\s*\((.+)\))?\s*$"
    )
    for line in prompt.splitlines():
        match = pattern.match(line)
        if not match:
            continue
        rows.append(
            {
                "item_index": int(match.group(1)),
                "item_name": match.group(2).strip(),
                "value": float(match.group(3)),
                "priority": match.group(4).strip() if match.group(4) else "",
            }
        )
    return rows


def extract_preferences_from_setup(
    setup_entries: List[Dict[str, Any]]
) -> Dict[str, List[Dict[str, Any]]]:
    parsed: Dict[str, List[Dict[str, Any]]] = {}
    for entry in setup_entries:
        parsed[entry["agent_id"]] = parse_setup_preferences(entry.get("prompt", ""))
    return parsed


def parse_json_response(text: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return None


def build_item_lookup(config: Dict[str, Any]) -> Dict[int, str]:
    items = config.get("items", [])
    lookup: Dict[int, str] = {}
    for idx, item in enumerate(items):
        if isinstance(item, dict):
            lookup[idx] = str(item.get("name", f"Item {idx}"))
        else:
            lookup[idx] = str(item)
    return lookup


def allocation_to_text(
    allocation: Dict[str, List[int]], item_lookup: Dict[int, str]
) -> str:
    segments = []
    for agent_id in sorted(allocation.keys(), key=agent_sort_key):
        item_ids = allocation.get(agent_id, [])
        if not item_ids:
            segments.append(f"{agent_id}: none")
            continue
        labels = [f"{idx}:{item_lookup.get(idx, f'Item {idx}')}" for idx in item_ids]
        segments.append(f"{agent_id}: {', '.join(labels)}")
    return " | ".join(segments)


def compute_utilities_for_allocation(
    allocation: Dict[str, List[int]],
    preferences_by_agent: Dict[str, List[Dict[str, Any]]],
    round_num: int,
    gamma_discount: float,
) -> Dict[str, float]:
    utilities: Dict[str, float] = {}
    discount = gamma_discount ** max(round_num - 1, 0)
    for agent_id, pref_rows in preferences_by_agent.items():
        values = {row["item_index"]: row["value"] for row in pref_rows}
        utilities[agent_id] = round(
            sum(values.get(item_idx, 0.0) for item_idx in allocation.get(agent_id, []))
            * discount,
            2,
        )
    return utilities


def collect_entries_by_round(
    interactions: List[Dict[str, Any]]
) -> Dict[int, List[Dict[str, Any]]]:
    grouped: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for entry in interactions:
        round_num = int(entry.get("round") or 0)
        grouped[round_num].append(entry)
    return dict(sorted(grouped.items()))


def conversation_logs_by_round(
    results: Dict[str, Any]
) -> Dict[int, List[Dict[str, Any]]]:
    grouped: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for entry in results.get("conversation_logs", []):
        round_num = int(entry.get("round") or 0)
        grouped[round_num].append(entry)
    return dict(sorted(grouped.items()))


def aggregate_token_usage(interactions: List[Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for entry in interactions:
        usage = entry.get("token_usage") or {}
        rows.append(
            {
                "agent_id": entry.get("agent_id", "unknown"),
                "phase": normalize_phase(entry.get("phase", "unknown")),
                "input_tokens": usage.get("input_tokens", 0) or 0,
                "output_tokens": usage.get("output_tokens", 0) or 0,
                "reasoning_tokens": usage.get("reasoning_tokens", 0) or 0,
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return (
        frame.groupby(["agent_id", "phase"], as_index=False)[
            ["input_tokens", "output_tokens", "reasoning_tokens"]
        ]
        .sum()
        .sort_values(["agent_id", "phase"])
    )


def render_phase_chip(label: str, phase: str) -> str:
    color = PHASE_COLORS.get(phase, "#6b7280")
    return (
        f"<span style='display:inline-block;padding:4px 10px;margin:2px;"
        f"border-radius:999px;background:{color};color:white;font-size:0.82rem;'>"
        f"{label}</span>"
    )


def render_text_card(
    title: str,
    body: str,
    phase: str,
    show_prompt: bool = False,
    prompt: Optional[str] = None,
    metadata: Optional[str] = None,
) -> None:
    color = PHASE_COLORS.get(phase, "#6b7280")
    if show_prompt and prompt:
        render_prompt_block(f"Prompt for {title}", prompt)
    st.markdown(
        (
            "<div style='border-left:4px solid {color};padding:14px 16px;"
            "background:#f8fafc;border-radius:10px;margin:10px 0;'>"
            "<div style='font-weight:700;margin-bottom:8px;'>{title}</div>"
            "{metadata_html}"
            "<div style='white-space:pre-wrap;line-height:1.45;'>{body}</div>"
            "</div>"
        ).format(
            color=color,
            title=html.escape(title),
            metadata_html=(
                f"<div style='color:#475569;font-size:0.9rem;margin-bottom:8px;'>{html.escape(metadata)}</div>"
                if metadata
                else ""
            ),
            body=html.escape(body),
        ),
        unsafe_allow_html=True,
    )


def render_setup_section(
    setup_entries: List[Dict[str, Any]],
    agent_models: Dict[str, str],
    preferences_by_agent: Dict[str, List[Dict[str, Any]]],
    show_prompts: bool,
) -> None:
    st.subheader("Setup And Sampled Preferences")
    columns = st.columns(len(setup_entries) or 1)
    for column, entry in zip(columns, sorted(setup_entries, key=lambda item: agent_sort_key(item["agent_id"]))):
        agent_id = entry["agent_id"]
        with column:
            st.markdown(f"**{agent_id}**")
            st.caption(agent_models.get(agent_id, "unknown model"))
            pref_rows = preferences_by_agent.get(agent_id, [])
            if pref_rows:
                frame = pd.DataFrame(pref_rows)
                frame = frame.rename(
                    columns={
                        "item_index": "idx",
                        "item_name": "item",
                        "value": "value",
                        "priority": "priority",
                    }
                )
                st.dataframe(frame, hide_index=True, use_container_width=True)
            render_text_card(
                title=f"{agent_id} setup response",
                body=entry.get("response", ""),
                phase="game_setup",
                show_prompt=show_prompts,
                prompt=entry.get("prompt"),
            )


def render_thinking_section(
    thinking_entries: List[Dict[str, Any]],
    show_prompts: bool,
) -> None:
    st.markdown("**Private Thinking**")
    columns = st.columns(len(thinking_entries) or 1)
    for column, entry in zip(columns, sorted(thinking_entries, key=lambda item: agent_sort_key(item["agent_id"]))):
        payload = parse_json_response(entry.get("response", "")) or {}
        with column:
            st.markdown(f"**{entry['agent_id']}**")
            if show_prompts and entry.get("prompt"):
                render_prompt_block(f"Thinking prompt: {entry['agent_id']}", entry["prompt"])
            st.write("Strategy:", payload.get("strategy", ""))
            priorities = payload.get("key_priorities") or payload.get("target_items") or []
            concessions = payload.get("potential_concessions") or payload.get("anticipated_resistance") or []
            if priorities:
                st.write("Key priorities:")
                for item in priorities:
                    st.write(f"- {item}")
            if concessions:
                st.write("Potential concessions:")
                for item in concessions:
                    st.write(f"- {item}")
            with st.expander(f"Full reasoning: {entry['agent_id']}", expanded=False):
                st.write(payload.get("reasoning", entry.get("response", "")))


def render_proposal_section(
    proposal_entries: List[Dict[str, Any]],
    preferences_by_agent: Dict[str, List[Dict[str, Any]]],
    item_lookup: Dict[int, str],
    gamma_discount: float,
    show_prompts: bool,
) -> None:
    st.markdown("**Formal Proposals**")
    if show_prompts:
        for entry in proposal_entries:
            render_prompt_block(f"Proposal prompt: {entry['agent_id']}", entry.get("prompt", ""))
    rows: List[Dict[str, Any]] = []
    for index, entry in enumerate(proposal_entries, start=1):
        payload = parse_json_response(entry.get("response", "")) or {}
        allocation = payload.get("allocation", {})
        utilities = compute_utilities_for_allocation(
            allocation,
            preferences_by_agent,
            int(entry.get("round") or 1),
            gamma_discount,
        )
        row: Dict[str, Any] = {
            "proposal_number": index,
            "proposed_by": payload.get("proposed_by", entry.get("agent_id")),
            "allocation": allocation_to_text(allocation, item_lookup),
            "reasoning": payload.get("reasoning", ""),
        }
        for agent_id in sorted(preferences_by_agent.keys(), key=agent_sort_key):
            row[f"{agent_id}_utility"] = utilities.get(agent_id, 0.0)
        rows.append(row)
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def render_voting_section(
    voting_entries: List[Dict[str, Any]],
    show_prompts: bool,
) -> None:
    st.markdown("**Batch Voting**")
    if show_prompts:
        seen_prompts = set()
        for entry in voting_entries:
            key = (entry.get("agent_id"), entry.get("prompt"))
            if key in seen_prompts:
                continue
            seen_prompts.add(key)
            render_prompt_block(f"Voting prompt: {entry['agent_id']}", entry.get("prompt", ""))
    rows: List[Dict[str, Any]] = []
    grouped: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for entry in voting_entries:
        payload = parse_json_response(entry.get("response", "")) or {}
        proposal_number = int(payload.get("proposal_number", 0) or 0)
        grouped[proposal_number].append(payload)
        rows.append(
            {
                "proposal_number": proposal_number,
                "voter": payload.get("voter"),
                "proposal_by": payload.get("proposal_by"),
                "vote": payload.get("vote_decision"),
                "reasoning": payload.get("reasoning", ""),
            }
        )
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    if grouped:
        summary_rows = []
        for proposal_number in sorted(grouped):
            votes = grouped[proposal_number]
            summary_rows.append(
                {
                    "proposal_number": proposal_number,
                    "accepts": sum(1 for vote in votes if vote.get("vote_decision") == "accept"),
                    "rejects": sum(1 for vote in votes if vote.get("vote_decision") == "reject"),
                }
            )
        st.caption("Per-proposal vote totals")
        st.dataframe(pd.DataFrame(summary_rows), hide_index=True, use_container_width=True)


def render_reflection_section(
    reflection_entries: List[Dict[str, Any]],
    show_prompts: bool,
) -> None:
    st.markdown("**Reflection**")
    columns = st.columns(len(reflection_entries) or 1)
    for column, entry in zip(columns, sorted(reflection_entries, key=lambda item: agent_sort_key(item["agent_id"]))):
        with column:
            render_text_card(
                title=entry["agent_id"],
                body=entry.get("response", ""),
                phase="reflection",
                show_prompt=show_prompts,
                prompt=entry.get("prompt"),
            )


def render_round_section(
    round_num: int,
    round_entries: List[Dict[str, Any]],
    round_logs: List[Dict[str, Any]],
    preferences_by_agent: Dict[str, List[Dict[str, Any]]],
    item_lookup: Dict[int, str],
    gamma_discount: float,
    show_prompts: bool,
) -> None:
    setup_logs = [log for log in round_logs if log.get("phase") == "proposal_enumeration"]
    tabulation_logs = [log for log in round_logs if log.get("phase") == "vote_tabulation"]
    discussion_entries = [entry for entry in round_entries if normalize_phase(entry["phase"]) == "discussion"]
    thinking_entries = [entry for entry in round_entries if normalize_phase(entry["phase"]) == "private_thinking"]
    proposal_entries = [entry for entry in round_entries if normalize_phase(entry["phase"]) == "proposal"]
    voting_entries = [entry for entry in round_entries if normalize_phase(entry["phase"]) == "voting"]
    reflection_entries = [entry for entry in round_entries if normalize_phase(entry["phase"]) == "reflection"]

    with st.expander(f"Round {round_num}", expanded=True):
        active_phases = []
        for phase in PHASE_SEQUENCE[1:]:
            has_entries = any(normalize_phase(entry["phase"]) == phase for entry in round_entries)
            has_logs = any(log.get("phase") == phase for log in round_logs)
            if has_entries or has_logs:
                active_phases.append(render_phase_chip(PHASE_LABELS[phase], phase))
        st.markdown("".join(active_phases), unsafe_allow_html=True)

        if discussion_entries:
            st.markdown("**Discussion**")
            for entry in discussion_entries:
                discussion_turn = extract_discussion_turn(entry.get("phase", ""))
                metadata = (
                    f"Turn {discussion_turn}"
                    if discussion_turn is not None
                    else None
                )
                render_text_card(
                    title=entry["agent_id"],
                    body=entry.get("response", ""),
                    phase="discussion",
                    show_prompt=show_prompts,
                    prompt=entry.get("prompt"),
                    metadata=metadata,
                )

        if thinking_entries:
            render_thinking_section(thinking_entries, show_prompts)

        if proposal_entries:
            render_proposal_section(
                proposal_entries,
                preferences_by_agent,
                item_lookup,
                gamma_discount,
                show_prompts,
            )

        if setup_logs:
            st.markdown("**Proposal Enumeration**")
            for log in setup_logs:
                st.code(log.get("content", ""))

        if voting_entries:
            render_voting_section(voting_entries, show_prompts)

        if tabulation_logs:
            st.markdown("**Vote Tabulation**")
            for log in tabulation_logs:
                st.code(log.get("content", ""))

        if reflection_entries:
            render_reflection_section(reflection_entries, show_prompts)


def main() -> None:
    st.set_page_config(page_title="Game 1 Viewer", layout="wide")

    args = parse_args()
    results_dir = resolve_results_dir(args.results_dir)
    scaling_mode = is_scaling_experiment_root(results_dir)

    selected_experiment_config: Optional[Dict[str, Any]] = None
    if scaling_mode:
        st.title("Game 1 Experiment Browser")
        index_frame = load_scaling_index(str(results_dir))
        if index_frame.empty:
            st.error("No completed experiments were found under this scaling-experiment root.")
            st.stop()

        st.sidebar.markdown("**Scaling Experiment Root**")
        st.sidebar.code(str(results_dir))
        st.sidebar.markdown("**Filter Experiments**")

        filtered_index, active_filters = filter_experiment_index(index_frame)
        st.sidebar.caption(f"{len(filtered_index)} matching experiments")
        if filtered_index.empty:
            st.warning("No experiments match the current sidebar filter combination.")
            st.stop()

        selected_label = st.sidebar.selectbox(
            "Matching Experiment",
            options=filtered_index["selection_label"].tolist(),
        )
        selected_row = filtered_index.loc[
            filtered_index["selection_label"] == selected_label
        ].iloc[0]
        results_dir = Path(selected_row["output_dir"]).resolve()
        selected_experiment_config = selected_row.to_dict()
    else:
        st.title("Game 1 Sample Viewer")

    payload = load_sample(str(results_dir))
    interactions = payload["interactions"]
    results = payload["results"]
    config = results["config"]

    agent_models = extract_agent_models(interactions)
    grouped_entries = collect_entries_by_round(interactions)
    grouped_logs = conversation_logs_by_round(results)
    item_lookup = build_item_lookup(config)

    setup_entries = grouped_entries.get(0, [])
    preferences_by_agent = extract_preferences_from_setup(setup_entries)
    token_summary = aggregate_token_usage(interactions)

    show_prompts = st.sidebar.checkbox("Show full prompts", value=False)
    show_raw = st.sidebar.checkbox("Show raw JSON", value=False)

    st.sidebar.markdown("**Selected Results Directory**")
    st.sidebar.code(str(results_dir))
    st.sidebar.markdown("**Agent Models**")
    for agent_id in sorted(agent_models, key=agent_sort_key):
        st.sidebar.write(f"- {agent_id}: {agent_models[agent_id]}")

    headline = "No Consensus" if not results.get("consensus_reached") else "Consensus Reached"
    st.caption(
        f"{headline} in round {results.get('final_round')} | game_type={config.get('game_type')}"
    )

    metric_columns = st.columns(5)
    metric_columns[0].metric("Outcome", headline)
    metric_columns[1].metric("Final Round", str(results.get("final_round")))
    metric_columns[2].metric("Items", str(config.get("m_items")))
    metric_columns[3].metric("Competition", str(config.get("competition_level")))
    metric_columns[4].metric("Seed", str(config.get("random_seed")))

    st.subheader("Chosen Hyperparameters")
    hyperparameter_rows = [
        {"name": "models", "value": ", ".join(agent_models[agent_id] for agent_id in sorted(agent_models, key=agent_sort_key))},
        {"name": "game_type", "value": config.get("game_type")},
        {"name": "m_items", "value": config.get("m_items")},
        {"name": "t_rounds", "value": config.get("t_rounds")},
        {"name": "competition_level", "value": config.get("competition_level")},
        {"name": "gamma_discount", "value": config.get("gamma_discount")},
        {"name": "discussion_turns", "value": config.get("discussion_turns")},
        {"name": "model_order", "value": config.get("model_order")},
        {"name": "random_seed", "value": config.get("random_seed")},
        {"name": "disable_discussion", "value": config.get("disable_discussion")},
        {"name": "disable_thinking", "value": config.get("disable_thinking")},
        {"name": "disable_reflection", "value": config.get("disable_reflection")},
    ]
    if selected_experiment_config is not None:
        hyperparameter_rows = [
            {"name": "weak_model", "value": selected_experiment_config.get("weak_model")},
            {"name": "strong_model", "value": selected_experiment_config.get("strong_model")},
            {"name": "experiment_id", "value": selected_experiment_config.get("experiment_id")},
            {"name": "run_number", "value": selected_experiment_config.get("run_number")},
            {"name": "config_file", "value": selected_experiment_config.get("config_file")},
        ] + hyperparameter_rows
    hyperparameter_frame = pd.DataFrame(hyperparameter_rows)
    st.dataframe(hyperparameter_frame, hide_index=True, use_container_width=True)

    render_setup_section(setup_entries, agent_models, preferences_by_agent, show_prompts)

    st.subheader("Outcome Summary")
    if results.get("consensus_reached"):
        st.success("A proposal reached the configured acceptance threshold.")
        st.json(results.get("final_allocation", {}))
        st.json(results.get("final_utilities", {}))
    else:
        st.warning(
            "No proposal reached the configured acceptance threshold. Under the Game 1 rules, this sample ends with no deal."
        )

    round_numbers = [round_num for round_num in grouped_entries.keys() if round_num > 0]
    st.subheader("Full Rollout")
    for round_num in round_numbers:
        render_round_section(
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


if __name__ == "__main__":
    main()
