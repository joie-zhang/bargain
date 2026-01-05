"""
Advanced UI Components for Negotiation Viewer

Provides reusable components for visualizing negotiation data.
"""

import streamlit as st
import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

import pandas as pd


# =============================================================================
# STYLED COMPONENTS
# =============================================================================

def styled_metric_card(
    title: str,
    value: str,
    delta: Optional[str] = None,
    icon: str = "📊",
    color: str = "#3b82f6"
):
    """Render a styled metric card."""
    delta_html = ""
    if delta:
        delta_color = "#10b981" if not delta.startswith("-") else "#ef4444"
        delta_html = f'<span style="color: {delta_color}; font-size: 12px;">({delta})</span>'

    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, {color}22 0%, {color}11 100%);
        border: 1px solid {color}44;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    ">
        <div style="font-size: 24px; margin-bottom: 4px;">{icon}</div>
        <div style="font-size: 28px; font-weight: bold; color: {color};">{value}</div>
        <div style="font-size: 14px; color: #6b7280;">{title} {delta_html}</div>
    </div>
    """, unsafe_allow_html=True)


def render_message_bubble(
    agent_id: str,
    content: str,
    phase: str,
    timestamp: Optional[float] = None,
    is_private: bool = False
):
    """Render a chat-style message bubble."""
    # Agent colors
    colors = {
        "Agent_Alpha": ("#3b82f6", "#dbeafe"),
        "Agent_Beta": ("#ef4444", "#fee2e2"),
        "system": ("#6b7280", "#f3f4f6"),
    }
    text_color, bg_color = colors.get(agent_id, ("#6b7280", "#f3f4f6"))

    # Alignment based on agent
    align = "flex-start" if agent_id == "Agent_Alpha" else "flex-end"
    if agent_id == "system":
        align = "center"

    # Private indicator
    private_badge = ""
    if is_private:
        bg_color = "#f1f5f9"
        private_badge = '<span style="background: #64748b; color: white; padding: 2px 6px; border-radius: 8px; font-size: 10px; margin-left: 8px;">🔒 PRIVATE</span>'

    # Format timestamp
    time_str = ""
    if timestamp:
        time_str = datetime.fromtimestamp(timestamp).strftime("%H:%M:%S")

    st.markdown(f"""
    <div style="display: flex; justify-content: {align}; margin: 8px 0;">
        <div style="
            max-width: 80%;
            background-color: {bg_color};
            border-radius: 16px;
            padding: 12px 16px;
            border-left: 4px solid {text_color};
        ">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <span style="font-weight: bold; color: {text_color};">{agent_id}{private_badge}</span>
                <span style="font-size: 11px; color: #9ca3af;">{time_str}</span>
            </div>
            <div style="color: #374151; line-height: 1.5;">
                {content[:500]}{'...' if len(content) > 500 else ''}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_proposal_card(
    proposal: Dict,
    items: List[str],
    preferences: Dict[str, List[float]],
    round_num: int
):
    """Render a proposal with allocation visualization."""
    allocation = proposal.get("allocation", {})
    proposed_by = proposal.get("proposed_by", "Unknown")
    reasoning = proposal.get("reasoning", "")

    st.markdown(f"""
    <div style="
        border: 2px solid #10b981;
        border-radius: 12px;
        padding: 16px;
        background: linear-gradient(135deg, #10b98111 0%, #10b98122 100%);
        margin: 12px 0;
    ">
        <div style="font-weight: bold; color: #059669; margin-bottom: 12px;">
            📝 Proposal by {proposed_by}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Display allocation
    cols = st.columns(len(allocation))
    for idx, (agent, item_indices) in enumerate(allocation.items()):
        with cols[idx]:
            agent_color = "#3b82f6" if "Alpha" in agent else "#ef4444"
            item_names = [items[i] if i < len(items) else f"Item {i}" for i in item_indices]

            # Calculate utility
            utility = 0
            if agent in preferences:
                utility = sum(preferences[agent][i] for i in item_indices if i < len(preferences[agent]))
                discount = 0.9 ** (round_num - 1) if round_num > 0 else 1.0
                utility *= discount

            st.markdown(f"""
            <div style="
                background: {agent_color}11;
                border: 1px solid {agent_color};
                border-radius: 8px;
                padding: 12px;
            ">
                <div style="font-weight: bold; color: {agent_color};">{agent}</div>
                <div style="font-size: 14px; margin: 8px 0;">
                    {', '.join(item_names) or 'No items'}
                </div>
                <div style="font-size: 12px; color: #6b7280;">
                    Utility: {utility:.1f}
                </div>
            </div>
            """, unsafe_allow_html=True)

    if reasoning:
        with st.expander("Reasoning"):
            st.markdown(reasoning)


def render_vote_result(
    votes: List[Dict],
    proposal_num: int
):
    """Render voting results for a proposal."""
    accepts = []
    rejects = []

    for vote in votes:
        voter = vote.get("voter", "Unknown")
        decision = vote.get("vote_decision") or vote.get("vote", "")
        if decision == "accept":
            accepts.append(voter)
        else:
            rejects.append(voter)

    total = len(accepts) + len(rejects)
    accept_pct = len(accepts) / total * 100 if total > 0 else 0

    # Progress bar color
    bar_color = "#10b981" if accept_pct == 100 else "#f59e0b" if accept_pct >= 50 else "#ef4444"

    st.markdown(f"""
    <div style="margin: 8px 0;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
            <span style="font-weight: bold;">Proposal #{proposal_num}</span>
            <span style="color: {bar_color};">{len(accepts)}/{total} Accept</span>
        </div>
        <div style="
            background: #e5e7eb;
            border-radius: 4px;
            height: 8px;
            overflow: hidden;
        ">
            <div style="
                background: {bar_color};
                height: 100%;
                width: {accept_pct}%;
                transition: width 0.3s ease;
            "></div>
        </div>
        <div style="display: flex; justify-content: space-between; font-size: 12px; color: #6b7280; margin-top: 4px;">
            <span>✅ {', '.join(accepts) if accepts else 'None'}</span>
            <span>❌ {', '.join(rejects) if rejects else 'None'}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# CHART COMPONENTS
# =============================================================================

def create_negotiation_timeline(interactions: List[Dict]) -> Optional[go.Figure]:
    """Create an interactive timeline of the negotiation."""
    if not PLOTLY_AVAILABLE:
        return None

    # Process data
    timeline_data = []
    for entry in interactions:
        phase = entry.get("phase", "unknown")
        agent = entry.get("agent_id", "system")
        round_num = entry.get("round", 0)
        timestamp = entry.get("timestamp", 0)

        timeline_data.append({
            "Round": round_num,
            "Phase": phase,
            "Agent": agent,
            "Timestamp": timestamp,
            "Content": entry.get("response", "")[:100] + "..."
        })

    df = pd.DataFrame(timeline_data)

    # Create figure
    fig = px.scatter(
        df,
        x="Timestamp",
        y="Round",
        color="Agent",
        symbol="Phase",
        hover_data=["Phase", "Content"],
        title="Negotiation Timeline"
    )

    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Round",
        legend_title="Agent",
        hovermode="closest"
    )

    return fig


def create_utility_comparison(
    proposals: List[Dict],
    preferences: Dict[str, List[float]],
    gamma: float = 0.9
) -> Optional[go.Figure]:
    """Create a comparison chart of utilities across proposals."""
    if not PLOTLY_AVAILABLE or not proposals:
        return None

    data = []
    for idx, prop in enumerate(proposals):
        allocation = prop.get("allocation", {})
        round_num = prop.get("round", 1)
        proposed_by = prop.get("proposed_by", "Unknown")

        for agent, items in allocation.items():
            if agent in preferences:
                raw_utility = sum(preferences[agent][i] for i in items if i < len(preferences[agent]))
                discounted = raw_utility * (gamma ** (round_num - 1))
                data.append({
                    "Proposal": f"P{idx + 1} ({proposed_by})",
                    "Agent": agent,
                    "Utility": discounted,
                    "Round": round_num
                })

    df = pd.DataFrame(data)

    fig = px.bar(
        df,
        x="Proposal",
        y="Utility",
        color="Agent",
        barmode="group",
        title="Utility by Proposal",
        color_discrete_map={"Agent_Alpha": "#3b82f6", "Agent_Beta": "#ef4444"}
    )

    fig.update_layout(
        xaxis_title="Proposal",
        yaxis_title="Discounted Utility",
        legend_title="Agent"
    )

    return fig


def create_preference_heatmap(
    preferences: Dict[str, List[float]],
    items: List[str]
) -> Optional[go.Figure]:
    """Create a heatmap showing agent preferences for items."""
    if not PLOTLY_AVAILABLE or not preferences:
        return None

    agents = list(preferences.keys())
    values = [preferences[a] for a in agents]

    fig = go.Figure(data=go.Heatmap(
        z=values,
        x=items if items else [f"Item {i}" for i in range(len(values[0]))],
        y=agents,
        colorscale="Blues",
        showscale=True,
        hovertemplate="Agent: %{y}<br>Item: %{x}<br>Value: %{z}<extra></extra>"
    ))

    fig.update_layout(
        title="Agent Preferences",
        xaxis_title="Items",
        yaxis_title="Agents"
    )

    return fig


def create_round_progress_chart(
    interactions: List[Dict],
    total_rounds: int = 10
) -> Optional[go.Figure]:
    """Create a progress chart showing negotiation advancement."""
    if not PLOTLY_AVAILABLE:
        return None

    rounds = {}
    for entry in interactions:
        round_num = entry.get("round", 0)
        if round_num not in rounds:
            rounds[round_num] = {"discussions": 0, "proposals": 0, "votes": 0}

        phase = entry.get("phase", "").lower()
        if "discussion" in phase:
            rounds[round_num]["discussions"] += 1
        elif "proposal" in phase and "vote" not in phase and "enum" not in phase:
            rounds[round_num]["proposals"] += 1
        elif "voting" in phase:
            rounds[round_num]["votes"] += 1

    df = pd.DataFrame([
        {"Round": r, "Type": t, "Count": v}
        for r, counts in rounds.items()
        for t, v in counts.items()
    ])

    fig = px.bar(
        df,
        x="Round",
        y="Count",
        color="Type",
        title="Activity by Round",
        color_discrete_map={
            "discussions": "#3b82f6",
            "proposals": "#10b981",
            "votes": "#f59e0b"
        }
    )

    return fig


# =============================================================================
# EXPORT FUNCTIONS
# =============================================================================

def export_to_markdown(interactions: List[Dict], experiment_name: str = "negotiation") -> str:
    """Export interactions to a markdown document."""
    lines = [
        f"# Negotiation Transcript: {experiment_name}",
        f"*Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
        "",
        "---",
        ""
    ]

    current_round = -1
    for entry in interactions:
        round_num = entry.get("round", 0)
        if round_num != current_round:
            current_round = round_num
            lines.append(f"\n## Round {round_num}\n")

        agent = entry.get("agent_id", "system")
        phase = entry.get("phase", "unknown")
        response = entry.get("response", "")

        lines.append(f"### {agent} - {phase}\n")
        lines.append(f"{response}\n")

    return "\n".join(lines)


def export_to_csv(interactions: List[Dict]) -> str:
    """Export interactions to CSV format."""
    import csv
    import io

    output = io.StringIO()
    writer = csv.writer(output)

    # Header
    writer.writerow(["timestamp", "round", "agent_id", "phase", "response"])

    for entry in interactions:
        writer.writerow([
            entry.get("timestamp", ""),
            entry.get("round", ""),
            entry.get("agent_id", ""),
            entry.get("phase", ""),
            entry.get("response", "")[:1000]  # Truncate for CSV
        ])

    return output.getvalue()


# =============================================================================
# FILTER COMPONENTS
# =============================================================================

def filter_sidebar(interactions: List[Dict]) -> Tuple[List[str], List[str], List[int]]:
    """Render filter controls in sidebar and return filter values."""

    st.sidebar.markdown("### 🔍 Filters")

    # Get unique values
    agents = list(set(e.get("agent_id", "system") for e in interactions))
    phases = list(set(e.get("phase", "unknown") for e in interactions))
    rounds = sorted(set(e.get("round", 0) for e in interactions))

    # Agent filter
    selected_agents = st.sidebar.multiselect(
        "Agents",
        options=agents,
        default=agents
    )

    # Phase filter
    selected_phases = st.sidebar.multiselect(
        "Phases",
        options=phases,
        default=phases
    )

    # Round filter
    round_range = st.sidebar.slider(
        "Round Range",
        min_value=min(rounds),
        max_value=max(rounds),
        value=(min(rounds), max(rounds))
    )

    selected_rounds = [r for r in rounds if round_range[0] <= r <= round_range[1]]

    return selected_agents, selected_phases, selected_rounds


def apply_filters(
    interactions: List[Dict],
    agents: List[str],
    phases: List[str],
    rounds: List[int]
) -> List[Dict]:
    """Apply filters to interactions list."""
    return [
        e for e in interactions
        if e.get("agent_id", "system") in agents
        and e.get("phase", "unknown") in phases
        and e.get("round", 0) in rounds
    ]
