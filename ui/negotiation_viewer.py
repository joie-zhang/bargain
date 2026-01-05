"""
Multi-Agent Negotiation Experiment Viewer

A Streamlit-based UI for visualizing negotiation experiments between LLM agents.
Supports both live streaming and post-hoc analysis of experiment results.

Usage:
    streamlit run ui/negotiation_viewer.py
"""

import streamlit as st
import json
import os
import glob
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import pandas as pd

# Try to import plotly for charts
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# =============================================================================
# CONFIGURATION
# =============================================================================

# Phase color mapping for visual distinction
PHASE_COLORS = {
    "game_setup": "#6366f1",           # Indigo
    "preference_assignment": "#8b5cf6", # Purple
    "discussion": "#3b82f6",            # Blue
    "private_thinking": "#64748b",      # Slate (muted for private)
    "proposal": "#10b981",              # Emerald
    "proposal_enumeration": "#14b8a6",  # Teal
    "voting": "#f59e0b",                # Amber
    "vote_tabulation": "#ef4444",       # Red
    "reflection": "#ec4899",            # Pink
}

PHASE_ICONS = {
    "game_setup": "🎮",
    "preference_assignment": "🔒",
    "discussion": "💬",
    "private_thinking": "🧠",
    "proposal": "📝",
    "proposal_enumeration": "📋",
    "voting": "🗳️",
    "vote_tabulation": "📊",
    "reflection": "🔄",
}

AGENT_COLORS = {
    "Agent_Alpha": "#3b82f6",  # Blue
    "Agent_Beta": "#ef4444",   # Red
    "system": "#64748b",       # Slate
}

# Default results directory
RESULTS_DIR = Path(__file__).parent.parent / "experiments" / "results"


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_experiment_folders() -> List[str]:
    """Find all experiment result folders."""
    if not RESULTS_DIR.exists():
        return []

    folders = []
    for item in sorted(RESULTS_DIR.iterdir()):
        if item.is_dir() and not item.name.startswith('.'):
            folders.append(item.name)
    return folders


def load_all_interactions(folder_name: str) -> List[Dict]:
    """Load the all_interactions.json file for an experiment folder."""
    # Try multiple possible file patterns
    patterns = [
        RESULTS_DIR / folder_name / "run_1_all_interactions.json",
        RESULTS_DIR / folder_name / "all_interactions.json",
    ]

    for pattern in patterns:
        if pattern.exists():
            with open(pattern, 'r') as f:
                return json.load(f)

    # Try glob for any all_interactions file
    matches = list((RESULTS_DIR / folder_name).glob("*all_interactions*.json"))
    if matches:
        with open(matches[0], 'r') as f:
            return json.load(f)

    return []


def load_experiment_results(folder_name: str, run_num: int = 1) -> Optional[Dict]:
    """Load experiment results for a specific run."""
    result_file = RESULTS_DIR / folder_name / f"run_{run_num}_experiment_results.json"
    if result_file.exists():
        with open(result_file, 'r') as f:
            return json.load(f)
    return None


def load_summary(folder_name: str) -> Optional[Dict]:
    """Load the batch summary for an experiment folder."""
    summary_file = RESULTS_DIR / folder_name / "_summary.json"
    if summary_file.exists():
        with open(summary_file, 'r') as f:
            return json.load(f)
    return None


def get_available_runs(folder_name: str) -> List[int]:
    """Get list of available run numbers for an experiment."""
    folder_path = RESULTS_DIR / folder_name
    runs = []
    for f in folder_path.glob("run_*_experiment_results.json"):
        try:
            run_num = int(f.stem.split('_')[1])
            runs.append(run_num)
        except (ValueError, IndexError):
            pass
    return sorted(runs)


# =============================================================================
# DATA PROCESSING FUNCTIONS
# =============================================================================

def parse_phase_from_entry(entry: Dict) -> str:
    """Extract the phase type from an interaction entry."""
    phase = entry.get("phase", "unknown")
    # Normalize phase names
    if "discussion" in phase.lower():
        return "discussion"
    elif "private_thinking" in phase.lower():
        return "private_thinking"
    elif "voting" in phase.lower():
        return "voting"
    elif "proposal" in phase.lower() and "enum" not in phase.lower():
        return "proposal"
    elif "reflection" in phase.lower():
        return "reflection"
    elif "preference" in phase.lower():
        return "preference_assignment"
    elif "setup" in phase.lower():
        return "game_setup"
    return phase


def extract_round_number(entry: Dict) -> int:
    """Extract round number from entry."""
    return entry.get("round", 0)


def group_by_round(interactions: List[Dict]) -> Dict[int, List[Dict]]:
    """Group interactions by round number."""
    rounds = {}
    for entry in interactions:
        round_num = extract_round_number(entry)
        if round_num not in rounds:
            rounds[round_num] = []
        rounds[round_num].append(entry)
    return dict(sorted(rounds.items()))


def extract_preferences(interactions: List[Dict]) -> Dict[str, List[float]]:
    """Extract agent preferences from preference assignment entries."""
    preferences = {}
    for entry in interactions:
        if "preference" in entry.get("phase", "").lower():
            agent_id = entry.get("agent_id", "")
            prompt = entry.get("prompt", "")
            # Parse preferences from prompt
            if "YOUR PRIVATE ITEM PREFERENCES" in prompt:
                prefs = []
                for line in prompt.split('\n'):
                    if '→' in line and ('HIGH' in line or 'Low' in line):
                        try:
                            value = float(line.split('→')[1].split()[0])
                            prefs.append(value)
                        except (ValueError, IndexError):
                            pass
                if prefs:
                    preferences[agent_id] = prefs
    return preferences


def parse_proposal(entry: Dict) -> Optional[Dict]:
    """Parse proposal from entry response."""
    response = entry.get("response", "")
    try:
        # Try to parse as JSON
        if response.strip().startswith('{'):
            data = json.loads(response)
            if "allocation" in data:
                return data
    except json.JSONDecodeError:
        pass
    return None


def parse_vote(entry: Dict) -> Optional[Dict]:
    """Parse vote from entry response."""
    response = entry.get("response", "")
    try:
        if response.strip().startswith('{'):
            data = json.loads(response)
            if "vote_decision" in data or "vote" in data:
                return data
    except json.JSONDecodeError:
        pass
    return None


def calculate_utilities(allocation: Dict[str, List[int]],
                       preferences: Dict[str, List[float]],
                       round_num: int,
                       gamma: float = 0.9) -> Dict[str, float]:
    """Calculate discounted utilities for an allocation."""
    utilities = {}
    discount = gamma ** (round_num - 1) if round_num > 0 else 1.0

    for agent, items in allocation.items():
        if agent in preferences:
            raw_utility = sum(preferences[agent][i] for i in items if i < len(preferences[agent]))
            utilities[agent] = raw_utility * discount
        else:
            utilities[agent] = 0.0

    return utilities


# =============================================================================
# STREAMLIT COMPONENTS
# =============================================================================

def render_phase_badge(phase: str) -> str:
    """Render a colored badge for a phase."""
    normalized_phase = parse_phase_from_entry({"phase": phase})
    color = PHASE_COLORS.get(normalized_phase, "#6b7280")
    icon = PHASE_ICONS.get(normalized_phase, "📌")
    return f'<span style="background-color: {color}; color: white; padding: 2px 8px; border-radius: 12px; font-size: 12px;">{icon} {phase}</span>'


def render_agent_header(agent_id: str) -> str:
    """Render a colored header for an agent."""
    color = AGENT_COLORS.get(agent_id, "#6b7280")
    return f'<span style="color: {color}; font-weight: bold;">{agent_id}</span>'


def render_interaction_card(entry: Dict, show_prompt: bool = False):
    """Render a single interaction as a styled card."""
    phase = entry.get("phase", "unknown")
    agent_id = entry.get("agent_id", "system")
    response = entry.get("response", "")
    prompt = entry.get("prompt", "")
    round_num = entry.get("round", 0)

    normalized_phase = parse_phase_from_entry(entry)
    color = PHASE_COLORS.get(normalized_phase, "#6b7280")
    agent_color = AGENT_COLORS.get(agent_id, "#6b7280")

    # Create the card
    st.markdown(f"""
    <div style="
        border-left: 4px solid {color};
        background-color: rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.1);
        padding: 12px 16px;
        margin: 8px 0;
        border-radius: 0 8px 8px 0;
    ">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <span style="font-weight: bold; color: {agent_color};">{PHASE_ICONS.get(normalized_phase, '📌')} {agent_id}</span>
            <span style="background-color: {color}; color: white; padding: 2px 8px; border-radius: 12px; font-size: 11px;">{phase}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Show prompt if requested
    if show_prompt and prompt:
        with st.expander("📥 Prompt", expanded=False):
            st.text(prompt[:2000] + "..." if len(prompt) > 2000 else prompt)

    # Show response
    if response:
        # Check if it's a JSON response
        try:
            if response.strip().startswith('{'):
                data = json.loads(response)
                if "allocation" in data:
                    # It's a proposal
                    st.markdown("**📝 Proposed Allocation:**")
                    for agent, items in data.get("allocation", {}).items():
                        item_names = [f"Item {i}" for i in items]
                        st.markdown(f"- **{agent}**: {', '.join(item_names)}")
                    if "reasoning" in data:
                        st.markdown(f"*Reasoning: {data['reasoning'][:300]}...*" if len(data.get('reasoning', '')) > 300 else f"*Reasoning: {data.get('reasoning', '')}*")
                elif "vote_decision" in data or "vote" in data:
                    # It's a vote
                    vote = data.get("vote_decision") or data.get("vote")
                    vote_color = "#10b981" if vote == "accept" else "#ef4444"
                    st.markdown(f"**🗳️ Vote:** <span style='color: {vote_color}; font-weight: bold;'>{vote.upper()}</span>", unsafe_allow_html=True)
                    if "reasoning" in data:
                        st.markdown(f"*{data['reasoning']}*")
                else:
                    # Other JSON
                    st.json(data)
            else:
                # Plain text response
                st.markdown(response)
        except json.JSONDecodeError:
            st.markdown(response)


def render_round_summary(round_num: int, entries: List[Dict], preferences: Dict[str, List[float]]):
    """Render a summary of a single round."""
    # Count phases
    phases = [parse_phase_from_entry(e) for e in entries]
    phase_counts = {}
    for p in phases:
        phase_counts[p] = phase_counts.get(p, 0) + 1

    # Find proposals and votes
    proposals = [e for e in entries if "proposal" in e.get("phase", "").lower() and "enum" not in e.get("phase", "").lower() and "voting" not in e.get("phase", "").lower()]
    votes = [e for e in entries if "voting" in e.get("phase", "").lower()]

    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        st.markdown(f"**Phases:** {', '.join(f'{PHASE_ICONS.get(p, '')} {p}({c})' for p, c in phase_counts.items())}")

    with col2:
        if proposals:
            st.markdown(f"**Proposals:** {len(proposals)}")

    with col3:
        if votes:
            accepts = sum(1 for v in votes if "accept" in str(v.get("response", "")).lower())
            st.markdown(f"**Votes:** {accepts}✓ / {len(votes) - accepts}✗")


def render_utility_chart(interactions: List[Dict], preferences: Dict[str, List[float]]):
    """Render utility tracking chart over rounds."""
    if not PLOTLY_AVAILABLE:
        st.warning("Install plotly for charts: pip install plotly")
        return

    # Extract proposals and calculate utilities per round
    rounds_data = []

    for entry in interactions:
        if "proposal" in entry.get("phase", "").lower() and "enum" not in entry.get("phase", "").lower() and "voting" not in entry.get("phase", "").lower():
            proposal = parse_proposal(entry)
            if proposal and "allocation" in proposal:
                round_num = entry.get("round", 0)
                utilities = calculate_utilities(
                    proposal["allocation"],
                    preferences,
                    round_num
                )
                for agent, utility in utilities.items():
                    rounds_data.append({
                        "Round": round_num,
                        "Agent": agent,
                        "Utility": utility,
                        "Proposed By": proposal.get("proposed_by", "Unknown")
                    })

    if not rounds_data:
        st.info("No proposal data found for utility tracking.")
        return

    df = pd.DataFrame(rounds_data)

    fig = px.line(
        df,
        x="Round",
        y="Utility",
        color="Agent",
        markers=True,
        title="Proposed Utility by Round",
        color_discrete_map=AGENT_COLORS
    )

    fig.update_layout(
        xaxis_title="Round",
        yaxis_title="Discounted Utility",
        legend_title="Agent",
        hovermode="x unified"
    )

    st.plotly_chart(fig, use_container_width=True)


def render_voting_heatmap(interactions: List[Dict]):
    """Render a heatmap of voting patterns."""
    if not PLOTLY_AVAILABLE:
        return

    votes_data = []
    for entry in interactions:
        if "voting" in entry.get("phase", "").lower():
            vote_info = parse_vote(entry)
            if vote_info:
                votes_data.append({
                    "Round": entry.get("round", 0),
                    "Voter": vote_info.get("voter", entry.get("agent_id", "Unknown")),
                    "Proposal": vote_info.get("proposal_number", 1),
                    "Vote": 1 if vote_info.get("vote_decision", vote_info.get("vote", "")) == "accept" else 0
                })

    if not votes_data:
        return

    df = pd.DataFrame(votes_data)

    # Create pivot table for heatmap
    pivot = df.pivot_table(
        index="Voter",
        columns=["Round", "Proposal"],
        values="Vote",
        aggfunc="first"
    )

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=[f"R{r}P{p}" for r, p in pivot.columns],
        y=pivot.index,
        colorscale=[[0, "#ef4444"], [1, "#10b981"]],
        showscale=False,
        hovertemplate="Agent: %{y}<br>Round-Proposal: %{x}<br>Vote: %{z}<extra></extra>"
    ))

    fig.update_layout(
        title="Voting Patterns (Green=Accept, Red=Reject)",
        xaxis_title="Round-Proposal",
        yaxis_title="Agent"
    )

    st.plotly_chart(fig, use_container_width=True)


def render_agent_comparison(interactions: List[Dict], round_num: int):
    """Render side-by-side agent comparison for a specific round."""
    round_entries = [e for e in interactions if e.get("round") == round_num]

    # Separate by agent
    agents = {}
    for entry in round_entries:
        agent = entry.get("agent_id", "system")
        if agent not in agents:
            agents[agent] = []
        agents[agent].append(entry)

    if len(agents) < 2:
        st.info("Not enough agents for comparison view")
        return

    agent_names = sorted(agents.keys())
    cols = st.columns(len(agent_names))

    for idx, agent in enumerate(agent_names):
        with cols[idx]:
            color = AGENT_COLORS.get(agent, "#6b7280")
            st.markdown(f"""
            <div style="
                background-color: rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.1);
                padding: 8px 12px;
                border-radius: 8px;
                border: 2px solid {color};
                margin-bottom: 12px;
            ">
                <h3 style="color: {color}; margin: 0;">{agent}</h3>
            </div>
            """, unsafe_allow_html=True)

            for entry in agents[agent]:
                phase = parse_phase_from_entry(entry)
                icon = PHASE_ICONS.get(phase, "📌")

                # Distinguish private vs public
                if "private" in phase:
                    st.markdown(f"**{icon} Private Thinking**")
                    with st.expander("View private reasoning", expanded=False):
                        response = entry.get("response", "")
                        try:
                            data = json.loads(response)
                            if "reasoning" in data:
                                st.markdown(data["reasoning"])
                            if "strategy" in data:
                                st.markdown(f"**Strategy:** {data['strategy']}")
                            if "target_items" in data:
                                st.markdown(f"**Targets:** {', '.join(map(str, data['target_items']))}")
                        except:
                            st.text(response[:500])
                else:
                    st.markdown(f"**{icon} {entry.get('phase', 'Unknown')}**")
                    response = entry.get("response", "")
                    if len(response) > 300:
                        st.markdown(response[:300] + "...")
                        with st.expander("Read more"):
                            st.markdown(response)
                    else:
                        st.markdown(response)


def render_experiment_overview(summary: Dict, results: Dict):
    """Render experiment overview metrics."""
    cols = st.columns(4)

    with cols[0]:
        consensus = results.get("consensus_reached", False) if results else False
        st.metric(
            "Consensus",
            "✅ Reached" if consensus else "❌ Not Reached",
        )

    with cols[1]:
        final_round = results.get("final_round", "N/A") if results else "N/A"
        st.metric("Final Round", final_round)

    with cols[2]:
        if results and "final_utilities" in results:
            utilities = results["final_utilities"]
            avg_util = sum(utilities.values()) / len(utilities) if utilities else 0
            st.metric("Avg Utility", f"{avg_util:.1f}")

    with cols[3]:
        if summary:
            consensus_rate = summary.get("consensus_rate", 0) * 100
            st.metric("Batch Consensus Rate", f"{consensus_rate:.0f}%")


# =============================================================================
# LIVE STREAMING COMPONENT
# =============================================================================

def setup_live_streaming():
    """Setup live streaming mode for watching experiments in real-time."""
    st.subheader("🔴 Live Streaming Mode")

    # Watch folder selection
    watch_folder = st.text_input(
        "Watch Folder Path",
        value=str(RESULTS_DIR),
        help="Path to folder where experiment results are being written"
    )

    # File pattern
    file_pattern = st.text_input(
        "File Pattern",
        value="*all_interactions*.json",
        help="Glob pattern for interaction files"
    )

    # Refresh rate
    refresh_rate = st.slider("Refresh Rate (seconds)", 1, 10, 2)

    # Start/Stop streaming
    if "streaming" not in st.session_state:
        st.session_state.streaming = False

    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶️ Start Streaming" if not st.session_state.streaming else "⏸️ Pause"):
            st.session_state.streaming = not st.session_state.streaming

    with col2:
        if st.button("🔄 Refresh Now"):
            st.rerun()

    if st.session_state.streaming:
        st.info(f"📡 Watching for changes in: {watch_folder}")

        # Find latest file
        watch_path = Path(watch_folder)
        if watch_path.exists():
            files = list(watch_path.rglob(file_pattern))
            if files:
                latest_file = max(files, key=lambda f: f.stat().st_mtime)
                st.success(f"Latest file: {latest_file.name}")

                # Load and display
                try:
                    with open(latest_file, 'r') as f:
                        interactions = json.load(f)

                    # Show latest entries
                    st.markdown("### Latest Interactions")
                    for entry in interactions[-5:]:  # Show last 5
                        render_interaction_card(entry)

                    # Auto-refresh
                    time.sleep(refresh_rate)
                    st.rerun()
                except Exception as e:
                    st.error(f"Error reading file: {e}")
            else:
                st.warning(f"No files matching pattern: {file_pattern}")
        else:
            st.error(f"Folder not found: {watch_folder}")


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    st.set_page_config(
        page_title="Negotiation Viewer",
        page_icon="🤝",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS
    st.markdown("""
    <style>
        .stMarkdown { max-width: 100%; }
        .block-container { padding-top: 1rem; }
        div[data-testid="stExpander"] details summary p {
            font-size: 14px;
        }
    </style>
    """, unsafe_allow_html=True)

    st.title("🤝 Multi-Agent Negotiation Viewer")

    # Sidebar
    with st.sidebar:
        st.header("📁 Experiment Selection")

        # Mode selection
        mode = st.radio(
            "Mode",
            ["📂 Post-hoc Analysis", "🔴 Live Streaming"],
            index=0
        )

        if mode == "📂 Post-hoc Analysis":
            # Folder selection
            folders = load_experiment_folders()

            if not folders:
                st.warning(f"No experiments found in {RESULTS_DIR}")
                st.info("Run some experiments first!")
                return

            selected_folder = st.selectbox(
                "Experiment",
                folders,
                format_func=lambda x: x.replace("_", " ").replace("runs", "| runs").replace("comp", "| comp")
            )

            # Run selection
            runs = get_available_runs(selected_folder)
            if runs:
                selected_run = st.selectbox("Run", runs)
            else:
                selected_run = 1

            st.divider()

            # View options
            st.header("👁️ View Options")
            show_prompts = st.checkbox("Show Prompts", value=False)
            show_private = st.checkbox("Show Private Thinking", value=True)
            comparison_mode = st.checkbox("Agent Comparison Mode", value=False)

            st.divider()

            # Quick stats
            st.header("📊 Quick Stats")
            summary = load_summary(selected_folder)
            if summary:
                st.metric("Total Runs", summary.get("num_runs", "N/A"))
                st.metric("Consensus Rate", f"{summary.get('consensus_rate', 0) * 100:.0f}%")

    # Main content area
    if mode == "🔴 Live Streaming":
        setup_live_streaming()
    else:
        # Post-hoc analysis mode
        if 'selected_folder' not in dir() or not selected_folder:
            st.info("Select an experiment from the sidebar")
            return

        # Load data
        interactions = load_all_interactions(selected_folder)
        results = load_experiment_results(selected_folder, selected_run)
        summary = load_summary(selected_folder)

        if not interactions:
            st.error(f"No interaction data found for {selected_folder}")
            return

        # Extract preferences
        preferences = extract_preferences(interactions)

        # Overview section
        st.header("📈 Experiment Overview")
        render_experiment_overview(summary, results)

        # Tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs([
            "📜 Timeline View",
            "👥 Agent Comparison",
            "📊 Analytics",
            "🔍 Raw Data"
        ])

        with tab1:
            st.subheader("Round-by-Round Timeline")

            # Group by round
            rounds = group_by_round(interactions)

            # Round selector
            if rounds:
                selected_rounds = st.multiselect(
                    "Filter Rounds",
                    options=list(rounds.keys()),
                    default=list(rounds.keys())
                )

                for round_num in selected_rounds:
                    entries = rounds[round_num]

                    with st.expander(f"🔄 Round {round_num}", expanded=(round_num == max(selected_rounds) if selected_rounds else True)):
                        render_round_summary(round_num, entries, preferences)
                        st.divider()

                        for entry in entries:
                            phase = parse_phase_from_entry(entry)

                            # Skip private thinking if disabled
                            if phase == "private_thinking" and not show_private:
                                continue

                            render_interaction_card(entry, show_prompt=show_prompts)

        with tab2:
            st.subheader("Side-by-Side Agent Comparison")

            rounds = group_by_round(interactions)
            if rounds:
                compare_round = st.selectbox(
                    "Select Round to Compare",
                    options=list(rounds.keys()),
                    index=len(rounds) - 1 if rounds else 0
                )

                render_agent_comparison(interactions, compare_round)

        with tab3:
            st.subheader("Experiment Analytics")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 📈 Utility Tracking")
                render_utility_chart(interactions, preferences)

            with col2:
                st.markdown("### 🗳️ Voting Patterns")
                render_voting_heatmap(interactions)

            # Strategic behavior analysis
            if results and "strategic_behaviors" in results:
                st.markdown("### 🧠 Strategic Behaviors")
                behaviors = results["strategic_behaviors"]

                cols = st.columns(4)
                metrics = [
                    ("Manipulation", behaviors.get("manipulation_attempts", 0)),
                    ("Anger", behaviors.get("anger_expressions", 0)),
                    ("Gaslighting", behaviors.get("gaslighting_attempts", 0)),
                    ("Cooperation", behaviors.get("cooperation_signals", 0)),
                ]

                for col, (name, value) in zip(cols, metrics):
                    with col:
                        st.metric(name, value)

        with tab4:
            st.subheader("Raw Data Explorer")

            data_type = st.selectbox(
                "Select Data Type",
                ["Interactions", "Results", "Summary", "Preferences"]
            )

            if data_type == "Interactions":
                st.json(interactions[:10] if len(interactions) > 10 else interactions)
                st.caption(f"Showing first 10 of {len(interactions)} interactions")
            elif data_type == "Results":
                if results:
                    st.json(results)
                else:
                    st.info("No results data available")
            elif data_type == "Summary":
                if summary:
                    st.json(summary)
                else:
                    st.info("No summary data available")
            elif data_type == "Preferences":
                st.json(preferences)


if __name__ == "__main__":
    main()
