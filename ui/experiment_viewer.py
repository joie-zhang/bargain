#!/usr/bin/env python3
"""
=============================================================================
Experiment Trajectory Viewer for Diplomacy & Co-Funding Games
=============================================================================

Streamlit app to browse full conversation trajectories for Game 2 (Diplomatic
Treaty) and Game 3 (Co-Funding / Participatory Budgeting) experiments.

Usage:
    streamlit run ui/experiment_viewer.py

What it shows:
    - Timeline view: round-by-round conversation with phase-specific rendering
    - Agent comparison: side-by-side agent responses per round
    - Analytics: utility charts, voting heatmaps, funding progress
    - Raw data: JSON viewer + download

Dependencies:
    - streamlit, plotly, pandas
    - Local: ui/components.py (reuses styled_metric_card, render_message_bubble)

=============================================================================
"""

import streamlit as st
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from collections import defaultdict

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

import pandas as pd

# ---------------------------------------------------------------------------
# Import reusable components from the existing UI package
# ---------------------------------------------------------------------------
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ui.components import styled_metric_card, render_message_bubble

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

RESULTS_DIR = Path(__file__).resolve().parent.parent / "experiments" / "results"

GAME_TYPES = {
    "diplomacy": "Diplomatic Treaty (Game 2)",
    "co_funding": "Co-Funding / Participatory Budgeting (Game 3)",
}

PHASE_CONFIG = {
    # Setup phases (round 0, from all_interactions)
    "game_setup":             {"icon": "🎮", "color": "#6366f1", "label": "Game Setup"},
    "preference_assignment":  {"icon": "🔒", "color": "#8b5cf6", "label": "Preferences"},
    # Diplomacy phases
    "discussion":             {"icon": "💬", "color": "#3b82f6", "label": "Discussion"},
    "private_thinking":       {"icon": "🧠", "color": "#64748b", "label": "Private Thinking"},
    "proposal":               {"icon": "📝", "color": "#10b981", "label": "Proposal"},
    "proposal_enumeration":   {"icon": "📋", "color": "#14b8a6", "label": "Proposal Enumeration"},
    "voting":                 {"icon": "🗳️", "color": "#f59e0b", "label": "Voting"},
    "vote_tabulation":        {"icon": "📊", "color": "#ef4444", "label": "Vote Tabulation"},
    "reflection":             {"icon": "🔄", "color": "#ec4899", "label": "Reflection"},
    # Cofunding phases
    "pledge_submission":      {"icon": "💰", "color": "#10b981", "label": "Pledge Submission"},
    "feedback":               {"icon": "💡", "color": "#f59e0b", "label": "Feedback"},
    "aggregate_funding":      {"icon": "📊", "color": "#8b5cf6", "label": "Funding Status"},
}

AGENT_COLORS = {
    "Agent_Alpha": "#3b82f6",
    "Agent_Beta":  "#ef4444",
    "system":      "#64748b",
}


# ---------------------------------------------------------------------------
# DATA CLASSES
# ---------------------------------------------------------------------------

@dataclass
class ExperimentInfo:
    """Metadata for a discovered experiment."""
    game_type: str            # "diplomacy" or "co_funding"
    experiment_name: str      # top-level dir name, e.g. "diplomacy_20260222_004806"
    sub_experiment: str       # "model_scale", "ttc_scaling", or "flat"
    model_pair: str           # e.g. "gpt-5-nano_vs_o3-mini-high"
    model_order: str          # "weak_first" or "strong_first"
    params: str               # e.g. "rho_0_5_theta_0_2" or "alpha_0_5_sigma_0_7"
    budget_level: str         # only for ttc_scaling, e.g. "budget_1000"; "" otherwise
    folder_path: str          # absolute path to leaf directory with JSON files
    display_label: str = ""   # computed human-readable label

    def __post_init__(self):
        parts = [self.model_pair, self.model_order]
        if self.budget_level:
            parts.append(self.budget_level)
        parts.append(self.params)
        self.display_label = " / ".join(parts)


# ---------------------------------------------------------------------------
# EXPERIMENT DISCOVERY
# ---------------------------------------------------------------------------

def discover_experiments() -> List[ExperimentInfo]:
    """Walk the results directory and find all diplomacy/cofunding experiments."""
    experiments = []
    if not RESULTS_DIR.exists():
        return experiments

    for top_dir in sorted(RESULTS_DIR.iterdir()):
        if not top_dir.is_dir() or top_dir.is_symlink():
            continue
        name = top_dir.name

        # Determine game type from directory name
        if name.startswith("diplomacy") or "_diplo_" in name:
            game_type = "diplomacy"
        elif name.startswith("cofunding") or name.startswith("cofund") or "_cofund_" in name:
            game_type = "co_funding"
        else:
            continue

        # Check for flat experiment (JSON files directly in top_dir)
        if _has_result_files(top_dir):
            model_pair, model_order, params = _extract_flat_metadata(top_dir, name)
            experiments.append(ExperimentInfo(
                game_type=game_type,
                experiment_name=name,
                sub_experiment="flat",
                model_pair=model_pair,
                model_order=model_order,
                params=params,
                budget_level="",
                folder_path=str(top_dir),
            ))
            continue

        # Check for sub-experiment types: model_scale, ttc_scaling, or pair dirs directly
        sub_dirs = [d for d in top_dir.iterdir() if d.is_dir() and not d.is_symlink()]
        for sub_dir in sub_dirs:
            sub_name = sub_dir.name
            if sub_name in ("configs", "logs", "__pycache__"):
                continue

            if sub_name in ("model_scale", "ttc_scaling"):
                _discover_nested(experiments, game_type, name, sub_name, sub_dir)
            elif "_vs_" in sub_name:
                # Pair dir directly under experiment (cofunding_smoke_test pattern)
                _discover_from_pair_dir(experiments, game_type, name, "flat", sub_dir)

    return experiments


def _discover_nested(experiments, game_type, exp_name, sub_exp, sub_dir):
    """Discover experiments inside model_scale/ or ttc_scaling/ directories."""
    for pair_dir in sorted(sub_dir.iterdir()):
        if not pair_dir.is_dir() or pair_dir.is_symlink() or "_vs_" not in pair_dir.name:
            continue
        _discover_from_pair_dir(experiments, game_type, exp_name, sub_exp, pair_dir)


def _discover_from_pair_dir(experiments, game_type, exp_name, sub_exp, pair_dir):
    """From a model pair directory, discover order → params → experiments."""
    pair_name = pair_dir.name
    for order_dir in sorted(pair_dir.iterdir()):
        if not order_dir.is_dir() or order_dir.is_symlink():
            continue
        order_name = order_dir.name
        if order_name not in ("weak_first", "strong_first"):
            continue

        for param_or_budget_dir in sorted(order_dir.iterdir()):
            if not param_or_budget_dir.is_dir() or param_or_budget_dir.is_symlink():
                continue

            pname = param_or_budget_dir.name
            # ttc_scaling has an extra budget level
            if pname.startswith("budget_"):
                for param_dir in sorted(param_or_budget_dir.iterdir()):
                    if not param_dir.is_dir() or param_dir.is_symlink():
                        continue
                    if _has_result_files(param_dir):
                        experiments.append(ExperimentInfo(
                            game_type=game_type,
                            experiment_name=exp_name,
                            sub_experiment=sub_exp,
                            model_pair=pair_name,
                            model_order=order_name,
                            params=param_dir.name,
                            budget_level=pname,
                            folder_path=str(param_dir),
                        ))
            elif _has_result_files(param_or_budget_dir):
                experiments.append(ExperimentInfo(
                    game_type=game_type,
                    experiment_name=exp_name,
                    sub_experiment=sub_exp,
                    model_pair=pair_name,
                    model_order=order_name,
                    params=pname,
                    budget_level="",
                    folder_path=str(param_or_budget_dir),
                ))


def _extract_flat_metadata(top_dir: Path, name: str) -> Tuple[str, str, str]:
    """Extract model pair, order, and params from a flat experiment directory.

    Tries:
      1. Parse directory name for _vs_ pattern and parameter suffixes
      2. Load experiment_results.json to get model info from agent_performance
    Returns (model_pair, model_order, params).
    """
    model_pair = "(flat)"
    model_order = "(flat)"
    params = "(flat)"

    # Try to extract from directory name (e.g. "gpt-5-nano_vs_gpt-5-nano_config_..._rho0_0_theta0_5")
    vs_match = re.match(r"(.+?)_vs_(.+?)_config", name)
    if vs_match:
        model_pair = f"{vs_match.group(1)}_vs_{vs_match.group(2)}"

    # Extract params from name (rho, theta, alpha, sigma)
    param_parts = []
    for pat, key in [
        (r"_rho(n?\d+_\d+)", "rho"),
        (r"_theta(\d+_\d+)", "theta"),
        (r"_alpha(\d+_\d+)", "alpha"),
        (r"_sigma(\d+_\d+)", "sigma"),
    ]:
        m = re.search(pat, name)
        if m:
            param_parts.append(f"{key}_{m.group(1)}")
    if param_parts:
        params = "_".join(param_parts)

    # If directory name didn't have _vs_, try loading from experiment results
    if model_pair == "(flat)":
        for pattern in ["run_1_experiment_results.json", "experiment_results.json"]:
            f = top_dir / pattern
            if f.exists():
                try:
                    with open(f) as fh:
                        data = json.load(fh)
                    perf = data.get("agent_performance", {})
                    models = []
                    for agent in sorted(perf.keys()):
                        m = perf[agent].get("model", "unknown")
                        models.append(m)
                    if models:
                        model_pair = "_vs_".join(models)
                    # Extract order from config
                    config = data.get("config", {})
                    order = config.get("model_order") or config.get("actual_order", "")
                    if order:
                        model_order = order
                except (json.JSONDecodeError, OSError):
                    pass
                break

    return model_pair, model_order, params


def _has_result_files(directory: Path) -> bool:
    """Check if a directory contains experiment result JSON files."""
    for f in directory.iterdir():
        if f.is_file() and f.name.endswith(".json") and "experiment_results" in f.name:
            return True
    return False


# ---------------------------------------------------------------------------
# DATA LOADING (cached)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def load_experiment_results(folder_path: str) -> Optional[Dict]:
    """Load experiment_results.json, preferring run-specific file."""
    p = Path(folder_path)
    # Try run-specific first (dedup: if both exist, run-specific has same data)
    for pattern in ["run_1_experiment_results.json", "experiment_results.json"]:
        f = p / pattern
        if f.exists():
            with open(f) as fh:
                return json.load(fh)
    return None


@st.cache_data(ttl=300)
def load_all_interactions(folder_path: str) -> List[Dict]:
    """Load all_interactions.json for the agent comparison tab."""
    p = Path(folder_path)
    for pattern in ["run_1_all_interactions.json", "all_interactions.json"]:
        f = p / pattern
        if f.exists():
            with open(f) as fh:
                return json.load(fh)
    return []


# ---------------------------------------------------------------------------
# DATA PROCESSING
# ---------------------------------------------------------------------------

def classify_phase(entry: Dict) -> str:
    """Normalize a conversation log entry's phase to a canonical key."""
    phase = entry.get("phase", "unknown").lower().strip()
    # Handle discussion_round_X_turn_Y from all_interactions
    if "discussion" in phase:
        return "discussion"
    if phase in PHASE_CONFIG:
        return phase
    # Handle all_interactions naming: pledge_round_N → pledge_submission
    if phase.startswith("pledge"):
        return "pledge_submission"
    # Fuzzy matching
    for key in PHASE_CONFIG:
        if key in phase:
            return key
    return phase


def group_by_round(logs: List[Dict]) -> Dict[int, List[Dict]]:
    """Group conversation log entries by round number."""
    rounds = defaultdict(list)
    for entry in logs:
        r = entry.get("round", 0)
        rounds[r].append(entry)
    return dict(sorted(rounds.items()))


def get_item_names(config: Dict) -> List[str]:
    """Extract item/project names from config."""
    items = config.get("items", [])
    return [item.get("name", f"Item {i}") for i, item in enumerate(items)]


def get_item_costs(config: Dict) -> List[float]:
    """Extract project costs from config (cofunding only)."""
    items = config.get("items", [])
    return [item.get("cost", 0.0) for item in items]


def _get_agent_budgets(config: Dict, results: Dict, game_type: str,
                       item_costs: List[float]) -> Dict[str, float]:
    """Get agent budgets from config, with fallback computation for older results."""
    budgets = config.get("agent_budgets", {})
    if budgets:
        return budgets
    # Fallback: compute from sigma and costs (for results saved before the fix)
    if game_type == "co_funding" and item_costs:
        sigma = config.get("sigma", 0.5)
        agents = sorted(results.get("final_utilities", {}).keys())
        if not agents:
            agents = config.get("agents", [])
        total_budget = sigma * sum(item_costs)
        per_agent = total_budget / max(len(agents), 1)
        return {a: round(per_agent, 2) for a in agents}
    return {}


def _short_experiment_label(name: str) -> str:
    """Create a concise sidebar label from a long directory name.

    Examples:
        "gpt-5-nano_vs_gpt-5-nano_config_unknown_runs10_diplo_issues3_rho0_0_theta0_5"
        → "nano vs nano 3iss ρ0.0 θ0.5"

        "diplomacy_20260222_004806"
        → "diplomacy 02-22 00:48"

        "cofunding_smoke_test"
        → "cofunding_smoke_test"
    """
    import re as _re

    def _model_short(s):
        """Shorten model names: gpt-5-nano→nano, claude-3-5-sonnet→3.5-sonnet."""
        if s.startswith("claude-"):
            # claude-3-5-sonnet → 3.5-sonnet, claude-3-7-sonnet → 3.7-sonnet
            rest = s[len("claude-"):]
            parts = rest.split("-")
            if len(parts) >= 3:
                return f"{parts[0]}.{parts[1]}-{'-'.join(parts[2:])}"
            return rest
        if s.startswith("gpt-"):
            # gpt-5-nano → nano, gpt-4o → 4o
            parts = s.split("-")
            return parts[-1] if len(parts) >= 3 else s[4:]
        return s

    def _fmt_val(raw):
        if raw is None:
            return None
        neg = raw.startswith("n")
        if neg:
            raw = raw[1:]
        val = raw.replace("_", ".")
        return f"-{val}" if neg else val

    # Extract params directly with individual searches (more robust than one big regex)
    if "_vs_" in name and "_config" in name:
        # Extract model pair
        pair_m = _re.match(r"(.+?)_vs_(.+?)_config", name)
        if pair_m:
            m1 = _model_short(pair_m.group(1))
            m2 = _model_short(pair_m.group(2))
            parts = [f"{m1} vs {m2}"]

            # Diplomacy params
            iss_m = _re.search(r"_diplo_issues(\d+)", name)
            if iss_m:
                parts.append(f"{iss_m.group(1)}iss")
            rho_m = _re.search(r"_rho(n?\d+_\d+)", name)
            if rho_m:
                parts.append(f"\u03c1{_fmt_val(rho_m.group(1))}")
            theta_m = _re.search(r"_theta(\d+_\d+)", name)
            if theta_m:
                parts.append(f"\u03b8{_fmt_val(theta_m.group(1))}")

            # Co-funding params
            proj_m = _re.search(r"_cofund_proj(\d+)", name)
            if proj_m:
                parts.append(f"{proj_m.group(1)}proj")
            alpha_m = _re.search(r"_alpha(\d+_\d+)", name)
            if alpha_m:
                parts.append(f"\u03b1{_fmt_val(alpha_m.group(1))}")
            sigma_m = _re.search(r"_sigma(\d+_\d+)", name)
            if sigma_m:
                parts.append(f"\u03c3{_fmt_val(sigma_m.group(1))}")

            return " ".join(parts)

    # Pattern: "diplomacy_YYYYMMDD_HHMMSS"
    ts_m = _re.match(r"(diplomacy|cofunding)_(\d{8})_(\d{6})", name)
    if ts_m:
        d, t = ts_m.group(2), ts_m.group(3)
        return f"{ts_m.group(1)} {d[4:6]}-{d[6:]} {t[:2]}:{t[2:4]}"

    # Fallback: return as-is
    return name


def parse_params(params_str: str) -> Dict[str, float]:
    """Parse parameter string like 'rho_0_5_theta_0_2' into dict."""
    result = {}
    # Handle negative values: 'rho_n0_5' means rho = -0.5
    # Pattern: key_[n]X_Y where n prefix means negative
    parts = params_str.split("_")
    i = 0
    while i < len(parts):
        key = parts[i]
        if key in ("rho", "theta", "alpha", "sigma") and i + 2 < len(parts):
            # Check for negative prefix
            integer_part = parts[i + 1]
            negative = False
            if integer_part.startswith("n"):
                negative = True
                integer_part = integer_part[1:]
            decimal_part = parts[i + 2]
            try:
                val = float(f"{integer_part}.{decimal_part}")
                if negative:
                    val = -val
                result[key] = val
                i += 3
                continue
            except ValueError:
                pass
        i += 1
    return result


# ---------------------------------------------------------------------------
# PHASE RENDERING HELPERS
# ---------------------------------------------------------------------------

def render_phase_badge(phase: str) -> str:
    """Return HTML for a colored phase badge."""
    cfg = PHASE_CONFIG.get(phase, {"icon": "❓", "color": "#6b7280", "label": phase})
    return (
        f'<span style="background: {cfg["color"]}22; color: {cfg["color"]}; '
        f'padding: 2px 8px; border-radius: 12px; font-size: 12px; font-weight: 600;">'
        f'{cfg["icon"]} {cfg["label"]}</span>'
    )


def render_discussion_entry(entry: Dict):
    """Render a discussion message as a chat bubble."""
    speaker = entry.get("from", "system")
    content = entry.get("content", "")
    ts = entry.get("timestamp")
    render_message_bubble(speaker, content, "discussion", ts)


def render_diplomacy_proposal(entry: Dict, item_names: List[str], preferences: Dict):
    """Render a diplomacy proposal with allocation table or agreement vector."""
    proposal = entry.get("proposal", {})
    agreement = proposal.get("agreement")
    allocation = proposal.get("allocation", {})
    proposer = proposal.get("proposed_by", entry.get("from", "Unknown"))
    reasoning = proposal.get("reasoning", "")
    round_num = entry.get("round", 1)

    st.markdown(
        f'<div style="border-left: 4px solid #10b981; padding: 8px 12px; '
        f'background: #10b98111; border-radius: 4px; margin: 8px 0;">'
        f'<strong style="color: #059669;">📝 Proposal by {proposer}</strong></div>',
        unsafe_allow_html=True,
    )

    if agreement is not None:
        # Diplomacy agreement vector format: display issue-by-issue
        for i, val in enumerate(agreement):
            name = item_names[i] if i < len(item_names) else f"Issue {i}"
            bar_pct = val * 100
            st.markdown(
                f'<div style="margin: 4px 0;">'
                f'<div style="display: flex; justify-content: space-between; font-size: 13px;">'
                f'<span>{name}</span>'
                f'<span style="color: #6b7280;">{val:.3f}</span>'
                f'</div>'
                f'<div style="background: #e5e7eb; border-radius: 4px; height: 6px; overflow: hidden;">'
                f'<div style="background: #10b981; height: 100%; width: {bar_pct:.0f}%;"></div>'
                f'</div></div>',
                unsafe_allow_html=True,
            )
    elif allocation:
        # Legacy item allocation format
        cols = st.columns(len(allocation))
        for idx, (agent, item_indices) in enumerate(allocation.items()):
            with cols[idx]:
                color = AGENT_COLORS.get(agent, "#6b7280")
                names = [item_names[i] if i < len(item_names) else f"Item {i}" for i in item_indices]
                utility = 0.0
                if agent in preferences:
                    prefs = preferences[agent]
                    utility = sum(prefs[i] for i in item_indices if i < len(prefs))
                    utility *= 0.9 ** (round_num - 1)
                st.markdown(
                    f'<div style="background: {color}11; border: 1px solid {color}44; '
                    f'border-radius: 8px; padding: 10px;">'
                    f'<div style="font-weight: bold; color: {color};">{agent}</div>'
                    f'<div style="margin: 6px 0;">{", ".join(names) or "None"}</div>'
                    f'<div style="font-size: 12px; color: #6b7280;">Utility: {utility:.3f}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    if reasoning:
        with st.expander("Reasoning", expanded=False):
            st.markdown(reasoning)


def render_diplomacy_vote_tabulation(entry: Dict):
    """Render vote tabulation results."""
    content = entry.get("content", "")
    st.markdown(
        f'<div style="border-left: 4px solid #ef4444; padding: 8px 12px; '
        f'background: #ef444411; border-radius: 4px; margin: 8px 0;">'
        f'<strong style="color: #ef4444;">📊 Vote Tabulation</strong></div>',
        unsafe_allow_html=True,
    )
    # Content is already formatted nicely by the system
    st.code(content, language=None)


def render_diplomacy_proposal_enumeration(entry: Dict, item_names: List[str]):
    """Render the proposal enumeration phase."""
    proposals = entry.get("enumerated_proposals", [])
    if not proposals:
        st.info(entry.get("content", "No proposals enumerated."))
        return

    st.markdown(
        f'<div style="border-left: 4px solid #14b8a6; padding: 8px 12px; '
        f'background: #14b8a611; border-radius: 4px; margin: 8px 0;">'
        f'<strong style="color: #14b8a6;">📋 Proposals Under Vote</strong></div>',
        unsafe_allow_html=True,
    )
    for p in proposals:
        num = p.get("proposal_number", "?")
        proposer = p.get("proposer", "Unknown")
        agreement = p.get("agreement")
        alloc = p.get("allocation", {})

        if agreement is not None:
            # Diplomacy agreement vector
            issue_vals = ", ".join(
                f"{item_names[i] if i < len(item_names) else f'Issue {i}'}: {v:.3f}"
                for i, v in enumerate(agreement)
            )
            st.markdown(f"**Proposal #{num}** (by {proposer}): {issue_vals}")
        elif alloc:
            # Legacy item allocation
            alloc_str = ", ".join(
                f"**{agent}**: {[item_names[i] if i < len(item_names) else f'Item {i}' for i in items]}"
                for agent, items in alloc.items()
            )
            st.markdown(f"**Proposal #{num}** (by {proposer}): {alloc_str}")
        else:
            st.markdown(f"**Proposal #{num}** (by {proposer}): (no data)")


def render_cofunding_pledge(entry: Dict, item_names: List[str], item_costs: List[float],
                            agent_budgets: Dict):
    """Render a cofunding pledge submission with progress bars."""
    pledge = entry.get("pledge", {})
    contributions = pledge.get("contributions", [])
    proposer = pledge.get("proposed_by", entry.get("from", "Unknown"))
    reasoning = pledge.get("reasoning", "")

    color = AGENT_COLORS.get(proposer, "#6b7280")
    total_pledged = sum(contributions) if contributions else 0
    budget = agent_budgets.get(proposer, 0)

    st.markdown(
        f'<div style="border-left: 4px solid {color}; padding: 8px 12px; '
        f'background: {color}11; border-radius: 4px; margin: 8px 0;">'
        f'<strong style="color: {color};">💰 Pledge by {proposer}</strong> '
        f'<span style="font-size: 12px; color: #6b7280;">'
        f'(${total_pledged:.2f} / ${budget:.2f} budget)</span></div>',
        unsafe_allow_html=True,
    )

    if contributions:
        for i, amount in enumerate(contributions):
            name = item_names[i] if i < len(item_names) else f"Project {i}"
            cost = item_costs[i] if i < len(item_costs) else 1.0
            pct = (amount / cost * 100) if cost > 0 else 0
            bar_color = "#10b981" if amount > 0 else "#e5e7eb"
            st.markdown(
                f'<div style="margin: 4px 0;">'
                f'<div style="display: flex; justify-content: space-between; font-size: 13px;">'
                f'<span>{name}</span>'
                f'<span style="color: #6b7280;">${amount:.2f} / ${cost:.2f} ({pct:.0f}%)</span>'
                f'</div>'
                f'<div style="background: #e5e7eb; border-radius: 4px; height: 6px; overflow: hidden;">'
                f'<div style="background: {bar_color}; height: 100%; width: {min(pct, 100):.0f}%;"></div>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

    if reasoning:
        with st.expander("Reasoning", expanded=False):
            st.markdown(reasoning)


def render_cofunding_aggregate(entry: Dict, item_names: List[str], item_costs: List[float]):
    """Render aggregate funding status (system message showing total pledges)."""
    content = entry.get("content", "")
    st.markdown(
        f'<div style="border-left: 4px solid #8b5cf6; padding: 8px 12px; '
        f'background: #8b5cf611; border-radius: 4px; margin: 8px 0;">'
        f'<strong style="color: #8b5cf6;">📊 Aggregate Funding Status</strong></div>',
        unsafe_allow_html=True,
    )
    st.markdown(content)


def render_private_thinking(entry: Dict):
    """Render private thinking phase in a collapsed expander."""
    speaker = entry.get("from", "system")
    content = entry.get("content", "")
    color = AGENT_COLORS.get(speaker, "#64748b")
    with st.expander(f"🧠 Private Thinking — {speaker}", expanded=False):
        st.markdown(
            f'<div style="border-left: 3px solid {color}; padding: 8px 12px; '
            f'color: #475569; font-style: italic;">{content}</div>',
            unsafe_allow_html=True,
        )


def render_reflection(entry: Dict):
    """Render reflection phase."""
    speaker = entry.get("from", "system")
    content = entry.get("content", "")
    color = AGENT_COLORS.get(speaker, "#ec4899")
    with st.expander(f"🔄 Reflection — {speaker}", expanded=False):
        st.markdown(
            f'<div style="border-left: 3px solid {color}; padding: 8px 12px;">{content}</div>',
            unsafe_allow_html=True,
        )


def render_feedback(entry: Dict):
    """Render cofunding feedback phase."""
    speaker = entry.get("from", "system")
    content = entry.get("content", "")
    color = AGENT_COLORS.get(speaker, "#f59e0b")
    with st.expander(f"💡 Feedback — {speaker}", expanded=False):
        st.markdown(
            f'<div style="border-left: 3px solid {color}; padding: 8px 12px;">{content}</div>',
            unsafe_allow_html=True,
        )


def _render_setup_entry(entry: Dict, phase: str):
    """Render game_setup or preference_assignment phase entries."""
    speaker = entry.get("from", "system")
    content = entry.get("content", "")
    cfg = PHASE_CONFIG.get(phase, {"icon": "❓", "color": "#6b7280", "label": phase})
    color = AGENT_COLORS.get(speaker, cfg["color"])
    with st.expander(f'{cfg["icon"]} {cfg["label"]} — {speaker}', expanded=False):
        st.markdown(
            f'<div style="border-left: 3px solid {color}; padding: 8px 12px;">{content}</div>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# RESULTS SUMMARY PANEL
# ---------------------------------------------------------------------------

def _parse_diplomacy_weights(interactions: List[Dict]) -> Dict[str, List[float]]:
    """Extract importance weights from preference_assignment prompts in all_interactions.

    The prompt contains a section like:
        **YOUR IMPORTANCE WEIGHTS** (how much you care about each issue):
          Trade Policy: 0.049 (Low priority)
          Military Access: 0.005 (Negligible)
          ...
    """
    weights: Dict[str, List[float]] = {}
    for entry in interactions:
        if entry.get("phase") != "preference_assignment":
            continue
        agent = entry.get("agent_id", "")
        prompt = entry.get("prompt", "")
        # Find the weights section
        start = prompt.find("IMPORTANCE WEIGHTS")
        if start == -1:
            continue
        # Skip past the current header line to get to the data lines
        newline_after = prompt.find("\n", start)
        if newline_after == -1:
            continue
        rest = prompt[newline_after:]
        # Take text until the next bold header (**STRATEGIC or similar)
        next_header = rest.find("\n**")
        if next_header > 0:
            rest = rest[:next_header]
        # Parse "IssueName: 0.049 (priority)" lines
        parsed = re.findall(r":\s+([\d.]+)\s+\(", rest)
        if parsed:
            weights[agent] = [float(v) for v in parsed]
    return weights


def _parse_cofunding_budgets(interactions: List[Dict]) -> Dict[str, float]:
    """Extract per-agent budgets from preference_assignment prompts."""
    budgets: Dict[str, float] = {}
    for entry in interactions:
        if entry.get("phase") != "preference_assignment":
            continue
        agent = entry.get("agent_id", "")
        prompt = entry.get("prompt", "")
        m = re.search(r"\*\*YOUR BUDGET:\*\*\s+([\d.]+)", prompt)
        if m:
            budgets[agent] = float(m.group(1))
    return budgets


def render_results_summary(results: Dict, game_type: str, experiment_id: str,
                           interactions: Optional[List[Dict]] = None):
    """Render a graphical summary of agent preferences and final outcomes."""
    if not PLOTLY_AVAILABLE:
        return

    config = results.get("config", {})
    preferences = results.get("agent_preferences", {})
    final_utils = results.get("final_utilities", {})
    final_alloc = results.get("final_allocation", {})
    perf = results.get("agent_performance", {})
    item_names = get_item_names(config)
    agents = sorted(preferences.keys())

    if not agents or not item_names:
        return

    st.markdown("---")
    st.subheader("Results Summary")

    if game_type == "diplomacy":
        _render_diplomacy_summary(
            agents, item_names, preferences, final_utils, final_alloc,
            perf, config, interactions or [], experiment_id,
        )
    elif game_type == "co_funding":
        _render_cofunding_summary(
            agents, item_names, preferences, final_utils, final_alloc,
            perf, config, interactions or [], experiment_id,
        )


def _render_diplomacy_summary(agents, item_names, preferences, final_utils,
                               final_alloc, perf, config, interactions, experiment_id):
    """Side-by-side diplomacy results: positions, weights, and utilities."""
    weights = _parse_diplomacy_weights(interactions)
    gamma = config.get("gamma_discount", 0.9)
    colors = [AGENT_COLORS.get(a, "#6b7280") for a in agents]

    # --- Side-by-side: Ideal Positions ---
    st.markdown("#### Ideal Positions per Issue")
    cols = st.columns(len(agents))
    for idx, agent in enumerate(agents):
        with cols[idx]:
            model = perf.get(agent, {}).get("model", "?")
            positions = preferences.get(agent, [])
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=item_names[:len(positions)],
                y=positions,
                marker_color=colors[idx],
                text=[f"{v:.3f}" for v in positions],
                textposition="outside",
            ))
            fig.update_layout(
                title=dict(text=f"{agent} ({model})", font=dict(size=14)),
                yaxis=dict(range=[0, 1.1], title="Position"),
                height=300,
                margin=dict(t=40, b=30, l=40, r=20),
            )
            st.plotly_chart(fig, use_container_width=True,
                            key=f"summ_pos_{agent}_{experiment_id}")

    # --- Side-by-side: Importance Weights ---
    if weights:
        st.markdown("#### Importance Weights per Issue")
        cols = st.columns(len(agents))
        for idx, agent in enumerate(agents):
            with cols[idx]:
                w = weights.get(agent, [])
                if not w:
                    st.info("Weights not available")
                    continue
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=item_names[:len(w)],
                    y=w,
                    marker_color=colors[idx],
                    text=[f"{v:.3f}" for v in w],
                    textposition="outside",
                ))
                max_w = max(w) if w else 1.0
                fig.update_layout(
                    title=dict(text=f"{agent}", font=dict(size=14)),
                    yaxis=dict(range=[0, max_w * 1.3], title="Weight"),
                    height=300,
                    margin=dict(t=40, b=30, l=40, r=20),
                )
                st.plotly_chart(fig, use_container_width=True,
                                key=f"summ_wt_{agent}_{experiment_id}")

    # --- Final Utility Comparison ---
    st.markdown("#### Final Outcome")
    cols = st.columns(len(agents))
    for idx, agent in enumerate(agents):
        with cols[idx]:
            util = final_utils.get(agent, 0)
            model = perf.get(agent, {}).get("model", "?")
            color = colors[idx]
            # Show agreement vector values for diplomacy (shared agreement, not per-agent)
            if isinstance(final_alloc, dict):
                assigned = final_alloc.get(agent, [])
                assigned_names = [item_names[i] if i < len(item_names) else f"Issue {i}"
                                  for i in assigned] if isinstance(assigned, list) else []
                assigned_str = ", ".join(assigned_names) if assigned_names else "None"
            else:
                # Diplomacy: final_alloc is the shared agreement vector
                assigned_str = "Shared agreement"
            st.markdown(
                f'<div style="background: {color}11; border: 2px solid {color}; '
                f'border-radius: 12px; padding: 16px; text-align: center;">'
                f'<div style="font-size: 14px; color: {color}; font-weight: bold;">'
                f'{agent}</div>'
                f'<div style="font-size: 12px; color: #6b7280;">{model}</div>'
                f'<div style="font-size: 28px; font-weight: bold; margin: 8px 0;">'
                f'{util:.3f}</div>'
                f'<div style="font-size: 11px; color: #6b7280;">Final Utility</div>'
                f'<div style="margin-top: 8px; font-size: 12px;">'
                f'Assigned: {assigned_str}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )


def _render_cofunding_summary(agents, item_names, preferences, final_utils,
                               final_alloc, perf, config, interactions, experiment_id):
    """Side-by-side co-funding results: valuations, budgets, and utilities."""
    item_costs = get_item_costs(config)
    budgets = _parse_cofunding_budgets(interactions)
    colors = [AGENT_COLORS.get(a, "#6b7280") for a in agents]

    # --- Side-by-side: Project Valuations ---
    st.markdown("#### Project Valuations per Agent")
    cols = st.columns(len(agents))
    for idx, agent in enumerate(agents):
        with cols[idx]:
            model = perf.get(agent, {}).get("model", "?")
            vals = preferences.get(agent, [])
            budget = budgets.get(agent, config.get("agent_budgets", {}).get(agent, 0))
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=item_names[:len(vals)],
                y=vals,
                marker_color=colors[idx],
                text=[f"{v:.1f}" for v in vals],
                textposition="outside",
            ))
            max_v = max(vals) if vals else 1.0
            fig.update_layout(
                title=dict(text=f"{agent} ({model}) — Budget: ${budget:.2f}",
                           font=dict(size=13)),
                yaxis=dict(range=[0, max_v * 1.3], title="Valuation"),
                height=300,
                margin=dict(t=40, b=30, l=40, r=20),
            )
            st.plotly_chart(fig, use_container_width=True,
                            key=f"summ_val_{agent}_{experiment_id}")

    # --- Project Costs vs Funded Status ---
    funded_indices = final_alloc if isinstance(final_alloc, list) else []
    st.markdown("#### Project Costs & Funding Status")
    cost_colors = ["#10b981" if i in funded_indices else "#ef4444"
                   for i in range(len(item_costs))]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=item_names[:len(item_costs)],
        y=item_costs,
        marker_color=cost_colors,
        text=[f"${c:.0f}" + (" Funded" if i in funded_indices else " Not funded")
              for i, c in enumerate(item_costs)],
        textposition="outside",
    ))
    fig.update_layout(
        yaxis=dict(title="Cost ($)"),
        height=300,
        margin=dict(t=20, b=30, l=40, r=20),
    )
    st.plotly_chart(fig, use_container_width=True,
                    key=f"summ_costs_{experiment_id}")

    # --- Final Utility Comparison ---
    st.markdown("#### Final Outcome")
    cols = st.columns(len(agents))
    for idx, agent in enumerate(agents):
        with cols[idx]:
            util = final_utils.get(agent, 0)
            model = perf.get(agent, {}).get("model", "?")
            budget = budgets.get(agent, 0)
            color = colors[idx]
            st.markdown(
                f'<div style="background: {color}11; border: 2px solid {color}; '
                f'border-radius: 12px; padding: 16px; text-align: center;">'
                f'<div style="font-size: 14px; color: {color}; font-weight: bold;">'
                f'{agent}</div>'
                f'<div style="font-size: 12px; color: #6b7280;">{model}</div>'
                f'<div style="font-size: 28px; font-weight: bold; margin: 8px 0;">'
                f'{util:.3f}</div>'
                f'<div style="font-size: 11px; color: #6b7280;">Final Utility</div>'
                f'<div style="margin-top: 8px; font-size: 12px;">'
                f'Budget: ${budget:.2f}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# TAB RENDERERS
# ---------------------------------------------------------------------------

def render_metrics_overview(results: Dict, game_type: str):
    """Render top-row metric cards."""
    config = results.get("config", {})
    final_utils = results.get("final_utilities", {})
    consensus = results.get("consensus_reached", False)
    final_round = results.get("final_round", "?")
    behaviors = results.get("strategic_behaviors", {})
    exploitation = results.get("exploitation_detected", False)

    cols = st.columns(5)
    with cols[0]:
        styled_metric_card(
            "Consensus",
            "Reached" if consensus else "Failed",
            icon="✅" if consensus else "❌",
            color="#10b981" if consensus else "#ef4444",
        )
    with cols[1]:
        styled_metric_card("Final Round", str(final_round), icon="🔄", color="#3b82f6")
    with cols[2]:
        avg_util = sum(final_utils.values()) / max(len(final_utils), 1)
        styled_metric_card("Avg Utility", f"{avg_util:.3f}", icon="📈", color="#8b5cf6")
    with cols[3]:
        if game_type == "co_funding":
            funded = results.get("final_allocation", [])
            n_projects = config.get("m_projects", config.get("m_items", 5))
            styled_metric_card(
                "Projects Funded",
                f"{len(funded)}/{n_projects}",
                icon="🏗️",
                color="#10b981",
            )
        else:
            coop = behaviors.get("cooperation_signals", 0)
            styled_metric_card("Cooperation Signals", str(coop), icon="🤝", color="#10b981")
    with cols[4]:
        styled_metric_card(
            "Exploitation",
            "Yes" if exploitation else "No",
            icon="⚠️" if exploitation else "🛡️",
            color="#ef4444" if exploitation else "#10b981",
        )


def render_timeline_tab(results: Dict, game_type: str, show_prompts: bool,
                        show_private: bool, experiment_id: str,
                        interactions: Optional[List[Dict]] = None):
    """Render the timeline tab — round-by-round conversation."""
    config = results.get("config", {})
    logs = results.get("conversation_logs", [])
    preferences = results.get("agent_preferences", {})
    item_names = get_item_names(config)
    item_costs = get_item_costs(config)
    agent_budgets = _get_agent_budgets(config, results, game_type, item_costs)

    if not logs and not interactions:
        st.warning("No conversation logs found in this experiment.")
        return

    rounds = group_by_round(logs)

    # Build a prompt lookup from all_interactions keyed by (round, agent, phase_prefix)
    # Use a list to handle multiple entries per key (e.g. discussion turns)
    prompt_lookup: Dict[Tuple, List[str]] = defaultdict(list)
    if interactions:
        for e in interactions:
            r = e.get("round", -1)
            agent = e.get("agent_id", "")
            phase_raw = e.get("phase", "")
            prompt_text = e.get("prompt", "")
            if prompt_text:
                # Normalize phase for matching: "discussion_round_1_turn_1" → "discussion"
                phase_key = classify_phase({"phase": phase_raw})
                prompt_lookup[(r, agent, phase_key)].append(prompt_text)

    # Merge round 0 (setup phases) from all_interactions if available
    if interactions and 0 not in rounds:
        round_0_entries = [
            e for e in interactions if e.get("round", -1) == 0
        ]
        if round_0_entries:
            # Convert all_interactions format to conversation_log format
            converted = []
            for e in round_0_entries:
                converted.append({
                    "phase": e.get("phase", "unknown"),
                    "round": 0,
                    "from": e.get("agent_id", "system"),
                    "content": e.get("response", ""),
                    "prompt": e.get("prompt", ""),
                    "timestamp": e.get("timestamp"),
                })
            rounds[0] = converted

    # Inject prompts from all_interactions into conversation_log entries
    if show_prompts and prompt_lookup:
        for r, entries in rounds.items():
            for entry in entries:
                if not entry.get("prompt"):
                    agent = entry.get("from", "")
                    phase_key = classify_phase(entry)
                    key = (r, agent, phase_key)
                    prompts = prompt_lookup.get(key, [])
                    if prompts:
                        entry["prompt"] = prompts.pop(0)

    round_nums = sorted(rounds.keys())

    selected_rounds = st.multiselect(
        "Filter rounds",
        options=round_nums,
        default=round_nums,
        key=f"timeline_rounds_{experiment_id}",
    )

    for r in selected_rounds:
        entries = rounds.get(r, [])
        phase_counts = defaultdict(int)
        for e in entries:
            phase_counts[classify_phase(e)] += 1
        phase_summary = " ".join(
            f'{PHASE_CONFIG.get(p, {}).get("icon", "❓")}{c}'
            for p, c in phase_counts.items()
        )

        with st.expander(f"**Round {r}** — {phase_summary}", expanded=(r == round_nums[0])):
            for entry in entries:
                phase = classify_phase(entry)

                # Skip private thinking if user doesn't want it
                if phase == "private_thinking" and not show_private:
                    continue

                # Show prompt if requested and available
                if show_prompts and entry.get("prompt"):
                    with st.expander("📨 Prompt", expanded=False):
                        st.code(entry["prompt"][:3000], language=None)

                # Dispatch to game-specific renderers
                if phase == "game_setup":
                    _render_setup_entry(entry, phase)
                elif phase == "preference_assignment":
                    _render_setup_entry(entry, phase)
                elif phase == "discussion":
                    render_discussion_entry(entry)
                elif phase == "private_thinking":
                    render_private_thinking(entry)
                elif phase == "reflection":
                    render_reflection(entry)
                elif game_type == "diplomacy":
                    _render_diplomacy_phase(entry, phase, item_names, preferences)
                elif game_type == "co_funding":
                    _render_cofunding_phase(entry, phase, item_names, item_costs, agent_budgets)
                else:
                    # Generic fallback
                    _render_generic_entry(entry, phase)

    # Results summary at the bottom of the timeline
    render_results_summary(results, game_type, experiment_id, interactions)


def _render_diplomacy_phase(entry, phase, item_names, preferences):
    """Dispatch diplomacy-specific phase rendering."""
    if phase == "proposal":
        render_diplomacy_proposal(entry, item_names, preferences)
    elif phase == "proposal_enumeration":
        render_diplomacy_proposal_enumeration(entry, item_names)
    elif phase == "vote_tabulation":
        render_diplomacy_vote_tabulation(entry)
    elif phase == "voting":
        _render_voting_entry(entry)
    else:
        _render_generic_entry(entry, phase)


def _render_cofunding_phase(entry, phase, item_names, item_costs, agent_budgets):
    """Dispatch cofunding-specific phase rendering."""
    if phase == "pledge_submission":
        render_cofunding_pledge(entry, item_names, item_costs, agent_budgets)
    elif phase == "feedback":
        render_feedback(entry)
    elif phase == "aggregate_funding":
        render_cofunding_aggregate(entry, item_names, item_costs)
    else:
        _render_generic_entry(entry, phase)


def _render_voting_entry(entry: Dict):
    """Render an individual agent's vote."""
    voter = entry.get("from", "Unknown")
    content = entry.get("content", "")
    color = AGENT_COLORS.get(voter, "#6b7280")

    # Try to parse structured vote
    vote_decision = None
    try:
        parsed = json.loads(content) if isinstance(content, str) else content
        if isinstance(parsed, dict):
            vote_decision = parsed.get("vote_decision") or parsed.get("vote")
    except (json.JSONDecodeError, TypeError):
        pass

    if vote_decision:
        vote_color = "#10b981" if vote_decision == "accept" else "#ef4444"
        vote_icon = "✅" if vote_decision == "accept" else "❌"
        st.markdown(
            f'<div style="border-left: 4px solid {color}; padding: 6px 12px; margin: 4px 0;">'
            f'<strong style="color: {color};">{voter}</strong> voted '
            f'<span style="color: {vote_color}; font-weight: bold;">{vote_icon} {vote_decision}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        render_message_bubble(voter, content, "voting", entry.get("timestamp"))


def _render_generic_entry(entry: Dict, phase: str):
    """Fallback renderer for unrecognized phases."""
    speaker = entry.get("from", "system")
    content = entry.get("content", "")
    badge = render_phase_badge(phase)
    st.markdown(badge, unsafe_allow_html=True)
    render_message_bubble(speaker, content, phase, entry.get("timestamp"))


def render_agent_comparison_tab(results: Dict, interactions: List[Dict],
                                game_type: str, show_prompts: bool,
                                experiment_id: str):
    """Render side-by-side agent comparison for a selected round."""
    logs = results.get("conversation_logs", [])
    config = results.get("config", {})
    agents = config.get("agents", ["Agent_Alpha", "Agent_Beta"])

    # Use all_interactions if available (has prompts + token usage)
    source = interactions if interactions else logs
    source_label = "all_interactions.json" if interactions else "conversation_logs"

    if not source:
        st.warning("No interaction data available.")
        return

    # Get available rounds
    rounds_available = sorted(set(e.get("round", 0) for e in source))
    if not rounds_available:
        st.warning("No rounds found.")
        return

    selected_round = st.selectbox(
        "Select round",
        options=rounds_available,
        index=0,
        key=f"comparison_round_{experiment_id}",
    )

    st.caption(f"Data source: `{source_label}`")

    round_entries = [e for e in source if e.get("round") == selected_round]

    cols = st.columns(len(agents))
    for col_idx, agent in enumerate(agents):
        with cols[col_idx]:
            color = AGENT_COLORS.get(agent, "#6b7280")
            perf = results.get("agent_performance", {}).get(agent, {})
            model = perf.get("model", "unknown")
            st.markdown(
                f'<div style="text-align: center; padding: 8px; background: {color}11; '
                f'border-radius: 8px; border: 1px solid {color}44; margin-bottom: 12px;">'
                f'<strong style="color: {color}; font-size: 16px;">{agent}</strong><br>'
                f'<span style="font-size: 12px; color: #6b7280;">{model}</span></div>',
                unsafe_allow_html=True,
            )

            agent_entries = [e for e in round_entries
                            if e.get("agent_id", e.get("from", "")) == agent]

            if not agent_entries:
                st.info("No entries for this agent in this round.")
                continue

            for entry in agent_entries:
                phase = entry.get("phase", classify_phase(entry))
                st.markdown(render_phase_badge(phase), unsafe_allow_html=True)

                # Show prompt if requested
                if show_prompts and "prompt" in entry:
                    with st.expander("📨 Prompt", expanded=False):
                        st.code(entry["prompt"][:3000], language=None)

                # Show response
                response = entry.get("response", entry.get("content", ""))
                if response:
                    st.markdown(
                        f'<div style="border-left: 3px solid {color}; padding: 6px 10px; '
                        f'margin: 4px 0; font-size: 13px;">{response[:2000]}'
                        f'{"..." if len(response) > 2000 else ""}</div>',
                        unsafe_allow_html=True,
                    )

                # Show token usage if available
                tokens = entry.get("token_usage", {})
                if tokens:
                    total = tokens.get("total_tokens", 0)
                    reasoning = tokens.get("reasoning_tokens", 0)
                    token_str = f"Tokens: {total:,}"
                    if reasoning:
                        token_str += f" (reasoning: {reasoning:,})"
                    st.caption(token_str)

                st.markdown("---")


def render_analytics_tab(results: Dict, game_type: str, experiment_id: str,
                         interactions: Optional[List[Dict]] = None):
    """Render analytics charts — game-specific."""
    if not PLOTLY_AVAILABLE:
        st.warning("Plotly not available. Install with: `pip install plotly`")
        return

    if game_type == "diplomacy":
        _render_diplomacy_analytics(results, experiment_id, interactions)
    elif game_type == "co_funding":
        _render_cofunding_analytics(results, experiment_id)


def _render_diplomacy_analytics(results: Dict, experiment_id: str,
                                 interactions: Optional[List[Dict]] = None):
    """Diplomacy-specific analytics: utility evolution + voting heatmap."""
    logs = results.get("conversation_logs", [])
    preferences = results.get("agent_preferences", {})
    config = results.get("config", {})
    gamma = config.get("gamma_discount", 0.9)
    item_names = get_item_names(config)

    # --- Utility evolution chart ---
    st.subheader("Utility Evolution Across Rounds")
    utility_data = []

    # Get positions/weights for diplomacy utility calculation
    agent_positions = config.get("agent_positions", {})
    agent_weights = config.get("agent_weights", {})

    # Fallback: use agent_preferences as positions (ideal positions)
    # and equal weights if not stored in config
    if not agent_positions and preferences:
        agent_positions = preferences
    if not agent_weights and preferences:
        n_issues = len(next(iter(preferences.values()), []))
        if n_issues > 0:
            agent_weights = {
                agent: [1.0 / n_issues] * n_issues
                for agent in preferences
            }

    for entry in logs:
        if classify_phase(entry) != "proposal":
            continue
        proposal = entry.get("proposal", {})
        agreement = proposal.get("agreement")
        allocation = proposal.get("allocation", {})
        round_num = entry.get("round", 1)
        proposer = proposal.get("proposed_by", "?")

        if agreement is not None and agent_positions:
            # Diplomacy: utility = Σ w_k × (1 - |p_k - a_k|)
            for agent in preferences:
                positions = agent_positions.get(agent, [])
                weights = agent_weights.get(agent, [])
                if positions and weights:
                    raw = sum(
                        w * (1 - abs(p - a))
                        for w, p, a in zip(weights, positions, agreement)
                    )
                else:
                    raw = 0.0
                discounted = raw * (gamma ** (round_num - 1))
                utility_data.append({
                    "Round": round_num,
                    "Agent": agent,
                    "Utility": discounted,
                    "Proposer": proposer,
                })
        elif allocation:
            # Item allocation format: compute utility for each agent
            # For diplomacy data, preferences = ideal positions; sum assigned values
            for agent, items in allocation.items():
                if agent in preferences:
                    prefs = preferences[agent]
                    raw = sum(prefs[i] for i in items if i < len(prefs))
                    discounted = raw * (gamma ** (round_num - 1))
                    utility_data.append({
                        "Round": round_num,
                        "Agent": agent,
                        "Utility": discounted,
                        "Proposer": proposer,
                    })

    if utility_data:
        df = pd.DataFrame(utility_data)
        fig = px.line(
            df, x="Round", y="Utility", color="Agent", markers=True,
            hover_data=["Proposer"],
            color_discrete_map={"Agent_Alpha": "#3b82f6", "Agent_Beta": "#ef4444"},
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True, key=f"util_chart_{experiment_id}")
    else:
        st.info("No proposal data available for utility chart.")

    # --- Voting heatmap ---
    st.subheader("Voting Patterns")
    vote_data = []

    # First try: extract individual votes from all_interactions
    if interactions:
        for entry in interactions:
            phase = entry.get("phase", "")
            if not phase.startswith("voting_round_"):
                continue
            voter = entry.get("agent_id", "Unknown")
            response = entry.get("response", "")
            round_num = entry.get("round", 0)
            # Extract proposal number from phase: "voting_round_1_proposal_2"
            prop_match = re.search(r"proposal_(\d+)", phase)
            prop_id = prop_match.group(1) if prop_match else "?"
            # Parse vote from response
            try:
                parsed = json.loads(response) if isinstance(response, str) else response
                if isinstance(parsed, dict):
                    decision = (parsed.get("vote_decision")
                                or parsed.get("vote", "unknown"))
                else:
                    decision = "accept" if "accept" in response.lower() else "reject"
            except (json.JSONDecodeError, TypeError):
                decision = "accept" if "accept" in response.lower() else "reject"
            vote_data.append({
                "Voter": voter,
                "Proposal": f"R{round_num}-P{prop_id}",
                "Vote": 1 if decision == "accept" else 0,
            })

    # Fallback: try conversation_logs voting entries
    if not vote_data:
        for entry in logs:
            phase = classify_phase(entry)
            if phase == "voting":
                voter = entry.get("from", "Unknown")
                content = entry.get("content", "")
                round_num = entry.get("round", 0)
                try:
                    parsed = json.loads(content) if isinstance(content, str) else content
                    if isinstance(parsed, dict):
                        decision = (parsed.get("vote_decision")
                                    or parsed.get("vote", "unknown"))
                        votes = parsed.get("votes", {})
                        if votes:
                            for prop_id, vote in votes.items():
                                vote_data.append({
                                    "Voter": voter,
                                    "Proposal": f"R{round_num}-P{prop_id}",
                                    "Vote": 1 if vote == "accept" else 0,
                                })
                        else:
                            vote_data.append({
                                "Voter": voter,
                                "Proposal": f"R{round_num}",
                                "Vote": 1 if decision == "accept" else 0,
                            })
                except (json.JSONDecodeError, TypeError):
                    pass

    # Last fallback: parse vote_tabulation text
    if not vote_data:
        for entry in logs:
            if classify_phase(entry) == "vote_tabulation":
                content = entry.get("content", "")
                round_num = entry.get("round", 0)
                # Parse lines like "Proposal #1: 0 accept, 2 reject"
                for match in re.finditer(
                    r"Proposal #(\d+):\s*(\d+)\s*accept,\s*(\d+)\s*reject", content
                ):
                    prop_id = match.group(1)
                    accepts = int(match.group(2))
                    rejects = int(match.group(3))
                    vote_data.append({
                        "Voter": "(aggregate)",
                        "Proposal": f"R{round_num}-P{prop_id}",
                        "Vote": round(accepts / max(accepts + rejects, 1), 2),
                    })

    if vote_data:
        df = pd.DataFrame(vote_data)
        pivot = df.pivot_table(index="Voter", columns="Proposal", values="Vote", aggfunc="first")
        fig = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            colorscale=[[0, "#ef4444"], [1, "#10b981"]],
            showscale=False,
            hovertemplate="Voter: %{y}<br>Proposal: %{x}<br>Vote: %{z}<extra></extra>",
        ))
        fig.update_layout(height=250, xaxis_title="Proposal", yaxis_title="Voter")
        st.plotly_chart(fig, use_container_width=True, key=f"vote_heatmap_{experiment_id}")
    else:
        st.info("No voting data available.")

    # --- Preference heatmap (ideal positions for diplomacy) ---
    st.subheader("Agent Ideal Positions")
    if preferences:
        agents = list(preferences.keys())
        values = [preferences[a] for a in agents]
        fig = go.Figure(data=go.Heatmap(
            z=values,
            x=item_names,
            y=agents,
            colorscale="Blues",
            showscale=True,
            hovertemplate="Agent: %{y}<br>Issue: %{x}<br>Position: %{z:.3f}<extra></extra>",
        ))
        fig.update_layout(height=250)
        st.plotly_chart(fig, use_container_width=True, key=f"pref_heatmap_{experiment_id}")


def _render_cofunding_analytics(results: Dict, experiment_id: str):
    """Cofunding-specific analytics: funding progress + budget utilization."""
    logs = results.get("conversation_logs", [])
    config = results.get("config", {})
    preferences = results.get("agent_preferences", {})
    item_names = get_item_names(config)
    item_costs = get_item_costs(config)
    agent_budgets = _get_agent_budgets(config, results, "co_funding", item_costs)
    agents = config.get("agents", ["Agent_Alpha", "Agent_Beta"])

    # --- Funding progress per round ---
    st.subheader("Funding Progress by Round")
    pledge_data = []
    for entry in logs:
        if classify_phase(entry) != "pledge_submission":
            continue
        pledge = entry.get("pledge", {})
        contributions = pledge.get("contributions", [])
        proposer = pledge.get("proposed_by", entry.get("from", ""))
        round_num = entry.get("round", 1)
        for i, amount in enumerate(contributions):
            name = item_names[i] if i < len(item_names) else f"Project {i}"
            pledge_data.append({
                "Round": round_num,
                "Project": name,
                "Agent": proposer,
                "Contribution": amount,
            })

    if pledge_data:
        df = pd.DataFrame(pledge_data)
        # Total per project per round
        totals = df.groupby(["Round", "Project"])["Contribution"].sum().reset_index()
        fig = px.bar(
            totals, x="Project", y="Contribution", color="Round",
            barmode="group",
            title="Total Contributions per Project by Round",
        )
        # Add cost threshold lines
        for i, cost in enumerate(item_costs):
            name = item_names[i] if i < len(item_names) else f"Project {i}"
            fig.add_hline(y=cost, line_dash="dash", line_color="#ef4444",
                          annotation_text=f"Cost: ${cost:.0f}",
                          annotation_position="top right")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True, key=f"funding_progress_{experiment_id}")

        # --- Per-agent contribution stacked bar ---
        st.subheader("Per-Agent Contributions")
        # Get last round's pledges
        last_round = df["Round"].max()
        last_df = df[df["Round"] == last_round]
        fig2 = px.bar(
            last_df, x="Project", y="Contribution", color="Agent",
            barmode="stack",
            title=f"Agent Contributions (Round {last_round})",
            color_discrete_map={"Agent_Alpha": "#3b82f6", "Agent_Beta": "#ef4444"},
        )
        fig2.update_layout(height=350)
        st.plotly_chart(fig2, use_container_width=True, key=f"agent_contrib_{experiment_id}")
    else:
        st.info("No pledge data available.")

    # --- Budget utilization ---
    st.subheader("Budget Utilization")
    if agent_budgets and pledge_data:
        df = pd.DataFrame(pledge_data)
        last_round = df["Round"].max()
        last_df = df[df["Round"] == last_round]
        for agent in agents:
            agent_df = last_df[last_df["Agent"] == agent]
            spent = agent_df["Contribution"].sum()
            budget = agent_budgets.get(agent, 0)
            pct = (spent / budget * 100) if budget > 0 else 0
            color = AGENT_COLORS.get(agent, "#6b7280")
            st.markdown(
                f'<div style="margin: 8px 0;">'
                f'<div style="display: flex; justify-content: space-between;">'
                f'<strong style="color: {color};">{agent}</strong>'
                f'<span>${spent:.2f} / ${budget:.2f} ({pct:.0f}%)</span></div>'
                f'<div style="background: #e5e7eb; border-radius: 4px; height: 12px; overflow: hidden;">'
                f'<div style="background: {color}; height: 100%; width: {min(pct, 100):.0f}%;"></div>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

    # --- Preference heatmap ---
    st.subheader("Agent Valuations")
    if preferences:
        agents_list = list(preferences.keys())
        values = [preferences[a] for a in agents_list]
        fig = go.Figure(data=go.Heatmap(
            z=values,
            x=item_names,
            y=agents_list,
            colorscale="Greens",
            showscale=True,
            hovertemplate="Agent: %{y}<br>Project: %{x}<br>Valuation: %{z:.2f}<extra></extra>",
        ))
        fig.update_layout(height=250)
        st.plotly_chart(fig, use_container_width=True, key=f"pref_heatmap_{experiment_id}")


def render_raw_data_tab(results: Dict, interactions: List[Dict], experiment_id: str):
    """Render JSON viewer + download."""
    data_choice = st.selectbox(
        "Select data",
        ["Conversation Logs", "Experiment Results", "All Interactions", "Config", "Preferences"],
        key=f"raw_data_select_{experiment_id}",
    )

    if data_choice == "Conversation Logs":
        logs = results.get("conversation_logs", [])
        st.caption(f"{len(logs)} entries")
        st.json(logs[:20] if len(logs) > 20 else logs)
    elif data_choice == "Experiment Results":
        # Show everything except conversation_logs (too large)
        summary = {k: v for k, v in results.items() if k != "conversation_logs"}
        st.json(summary)
    elif data_choice == "All Interactions":
        st.caption(f"{len(interactions)} entries")
        st.json(interactions[:10] if len(interactions) > 10 else interactions)
    elif data_choice == "Config":
        st.json(results.get("config", {}))
    elif data_choice == "Preferences":
        st.json(results.get("agent_preferences", {}))

    # Download button
    st.download_button(
        "📥 Download Full Results JSON",
        data=json.dumps(results, indent=2, default=str),
        file_name=f"experiment_results_{experiment_id}.json",
        mime="application/json",
        key=f"download_{experiment_id}",
    )


# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------

def render_sidebar(experiments: List[ExperimentInfo]):
    """Render cascading sidebar filters. Returns selected experiment + options."""
    st.sidebar.title("🔬 Experiment Viewer")
    st.sidebar.markdown("Games 2 & 3: Diplomacy & Co-Funding")
    st.sidebar.markdown("---")

    if not experiments:
        st.sidebar.warning("No experiments found in results directory.")
        return None, {}, {}

    # --- Level 1: Game type ---
    available_games = sorted(set(e.game_type for e in experiments))
    game_labels = {g: GAME_TYPES.get(g, g) for g in available_games}
    selected_game_label = st.sidebar.selectbox(
        "Game Type",
        options=[game_labels[g] for g in available_games],
        key="sidebar_game_type",
    )
    selected_game = [g for g, label in game_labels.items() if label == selected_game_label][0]
    filtered = [e for e in experiments if e.game_type == selected_game]

    # --- Level 2: Experiment (top-level dir) ---
    exp_names = sorted(set(e.experiment_name for e in filtered))
    # Build short label → full name mapping for the dropdown
    exp_label_map = {_short_experiment_label(n): n for n in exp_names}
    exp_labels = list(exp_label_map.keys())
    selected_exp_label = st.sidebar.selectbox(
        "Experiment",
        options=exp_labels,
        index=len(exp_labels) - 1 if exp_labels else 0,  # default to latest
        key="sidebar_experiment",
    )
    selected_exp = exp_label_map.get(selected_exp_label, selected_exp_label)
    filtered = [e for e in filtered if e.experiment_name == selected_exp]

    # --- Level 3: Sub-experiment type ---
    sub_exps = sorted(set(e.sub_experiment for e in filtered))
    if len(sub_exps) > 1:
        selected_sub = st.sidebar.selectbox(
            "Sub-experiment",
            options=sub_exps,
            key="sidebar_sub_experiment",
        )
        filtered = [e for e in filtered if e.sub_experiment == selected_sub]

    # --- Level 4: Model pair ---
    pairs = sorted(set(e.model_pair for e in filtered))
    selected_pair = st.sidebar.selectbox(
        "Model Pair",
        options=["All"] + pairs if len(pairs) > 1 else pairs,
        key="sidebar_model_pair",
    )
    if selected_pair != "All":
        filtered = [e for e in filtered if e.model_pair == selected_pair]

    # --- Level 5: Order ---
    orders = sorted(set(e.model_order for e in filtered))
    order_labels = {"weak_first": "W→S (weak first)", "strong_first": "S→W (strong first)", "(flat)": "(flat)"}
    if len(orders) > 1:
        selected_order_label = st.sidebar.selectbox(
            "Model Order",
            options=["All"] + [order_labels.get(o, o) for o in orders],
            key="sidebar_order",
        )
        if selected_order_label != "All":
            selected_order = [o for o, l in order_labels.items() if l == selected_order_label]
            if selected_order:
                filtered = [e for e in filtered if e.model_order == selected_order[0]]

    # --- Level 5b: Budget (ttc_scaling only) ---
    budgets = sorted(set(e.budget_level for e in filtered if e.budget_level))
    if budgets:
        selected_budget = st.sidebar.selectbox(
            "Budget Level",
            options=["All"] + budgets,
            key="sidebar_budget",
        )
        if selected_budget != "All":
            filtered = [e for e in filtered if e.budget_level == selected_budget]

    # --- Level 6: Parameters ---
    params = sorted(set(e.params for e in filtered))
    if len(params) > 1:
        selected_params = st.sidebar.selectbox(
            "Parameters",
            options=params,
            key="sidebar_params",
        )
        filtered = [e for e in filtered if e.params == selected_params]

    # --- View options ---
    st.sidebar.markdown("---")
    st.sidebar.markdown("### View Options")
    show_prompts = st.sidebar.checkbox("Show prompts", value=False, key="sidebar_show_prompts")
    show_private = st.sidebar.checkbox(
        "Show private thinking", value=True, key="sidebar_show_private"
    )

    options = {"show_prompts": show_prompts, "show_private": show_private}

    # --- Quick stats ---
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**{len(filtered)}** experiment(s) matched")

    if len(filtered) == 1:
        return filtered[0], options, {}
    elif len(filtered) > 1:
        # Let user pick one from the remaining
        labels = [e.display_label for e in filtered]
        selected_label = st.sidebar.selectbox(
            "Select experiment",
            options=labels,
            key="sidebar_final_select",
        )
        idx = labels.index(selected_label)
        return filtered[idx], options, {}
    else:
        return None, options, {}


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="Experiment Viewer — Diplomacy & Co-Funding",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Discover all experiments
    experiments = discover_experiments()

    # Sidebar
    selected_exp, options, _ = render_sidebar(experiments)

    if selected_exp is None:
        st.title("🔬 Experiment Trajectory Viewer")
        st.markdown(
            "Select a **Diplomacy** or **Co-Funding** experiment from the sidebar to begin."
        )
        st.info(f"Found **{len(experiments)}** total experiments in `{RESULTS_DIR}`")
        return

    # Load data
    results = load_experiment_results(selected_exp.folder_path)
    if results is None:
        st.error(f"Could not load experiment results from: `{selected_exp.folder_path}`")
        return

    interactions = load_all_interactions(selected_exp.folder_path)

    # Header
    config = results.get("config", {})
    game_label = GAME_TYPES.get(selected_exp.game_type, selected_exp.game_type)
    st.title(f"🔬 {game_label}")
    st.caption(f"`{selected_exp.folder_path}`")

    # Show parsed parameters
    params_dict = parse_params(selected_exp.params)
    perf = results.get("agent_performance", {})
    agent_models = {agent: info.get("model", "?") for agent, info in perf.items()}
    param_str = " | ".join(f"**{k}** = {v}" for k, v in params_dict.items())
    model_str = " vs ".join(f"{agent} ({model})" for agent, model in agent_models.items())
    if param_str:
        st.markdown(f"{param_str} | {model_str}")
    elif model_str:
        st.markdown(model_str)

    # Metrics overview
    render_metrics_overview(results, selected_exp.game_type)
    st.markdown("---")

    # Create experiment ID for unique widget keys (must be stable across reruns)
    experiment_id = (
        f"{selected_exp.experiment_name}_{selected_exp.sub_experiment}_"
        f"{selected_exp.model_pair}_{selected_exp.model_order}_"
        f"{selected_exp.params}_{selected_exp.budget_level}"
    )

    # Main tabs
    tab_timeline, tab_comparison, tab_analytics, tab_raw = st.tabs(
        ["📜 Timeline", "👥 Agent Comparison", "📊 Analytics", "🔍 Raw Data"]
    )

    with tab_timeline:
        render_timeline_tab(
            results, selected_exp.game_type,
            options["show_prompts"], options["show_private"],
            experiment_id, interactions,
        )

    with tab_comparison:
        render_agent_comparison_tab(
            results, interactions, selected_exp.game_type,
            options["show_prompts"], experiment_id,
        )

    with tab_analytics:
        render_analytics_tab(results, selected_exp.game_type, experiment_id, interactions)

    with tab_raw:
        render_raw_data_tab(results, interactions, experiment_id)


if __name__ == "__main__":
    main()
