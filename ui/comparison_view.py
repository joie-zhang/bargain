"""
Batch Comparison View for Negotiation Viewer

Provides multi-select comparison of experiment batches with:
- Token usage comparison
- Payoff analysis
- Consensus round tracking
- Nash welfare visualization
- Qualitative reasoning extraction
"""

import streamlit as st
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from analysis import (
    analyze_batch,
    compare_batches,
    extract_concession_reasoning,
    load_all_interactions,
    BatchMetrics,
    ExperimentMetrics,
    ProposalMetrics,
    PhaseTokens,
)

# Default results directory
RESULTS_DIR = Path(__file__).parent.parent / "experiments" / "results"


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_short_experiment_name(folder_name: str, max_length: int = 30) -> str:
    """
    Create a smart short name for experiments that preserves distinguishing info.

    Examples:
        'gpt-5-low-effort_vs_gpt-5-high-effort_runs5_comp1' -> 'G5-Low vs G5-High'
        'Qwen2.5-14B-Instruct_vs_claude-3-7-sonnet_runs5_comp1' -> 'Qwen14B vs Claude3.7'
    """
    import re

    # Remove common suffixes
    name = re.sub(r'_runs\d+_comp\d+.*$', '', folder_name)

    # Split by _vs_
    if '_vs_' not in name:
        # Fallback for non-standard names
        if len(name) > max_length:
            return name[:max_length-3] + "..."
        return name

    parts = name.split('_vs_')
    if len(parts) != 2:
        if len(name) > max_length:
            return name[:max_length-3] + "..."
        return name

    def abbreviate_model(model_name: str) -> str:
        """Abbreviate a model name to be concise but distinguishable."""
        name = model_name.lower()

        # Handle GPT-5 effort levels
        if 'gpt-5' in name or 'gpt5' in name:
            if 'low-effort' in name or 'low_effort' in name:
                return 'G5-Low'
            elif 'medium-effort' in name or 'med-effort' in name or 'medium_effort' in name:
                return 'G5-Med'
            elif 'high-effort' in name or 'high_effort' in name:
                return 'G5-High'
            else:
                return 'GPT5'

        # Handle Claude models
        if 'claude' in name:
            if '3-7' in name or '3.7' in name:
                return 'Cl3.7'
            elif '3-5' in name or '3.5' in name:
                return 'Cl3.5'
            elif 'haiku' in name:
                return 'ClHaiku'
            elif 'sonnet' in name:
                return 'ClSonnet'
            elif 'opus' in name:
                return 'ClOpus'
            else:
                return 'Claude'

        # Handle Qwen models
        if 'qwen' in name:
            match = re.search(r'(\d+)b', name, re.IGNORECASE)
            size = match.group(1) if match else ''
            return f'Qwen{size}B' if size else 'Qwen'

        # Handle GPT-4
        if 'gpt-4' in name or 'gpt4' in name:
            if 'turbo' in name:
                return 'GPT4T'
            return 'GPT4'

        # Handle o3/o1 models
        if name.startswith('o3') or name == 'o3':
            return 'O3'
        if name.startswith('o1') or name == 'o1':
            return 'O1'

        # Fallback: take first part up to reasonable length
        short = model_name.replace('-', '').replace('_', '')[:8]
        return short

    alpha_abbrev = abbreviate_model(parts[0])
    beta_abbrev = abbreviate_model(parts[1])

    result = f"{alpha_abbrev} vs {beta_abbrev}"

    if len(result) > max_length:
        return result[:max_length-3] + "..."

    return result


def get_experiment_folders() -> List[str]:
    """Get all experiment folder names."""
    if not RESULTS_DIR.exists():
        return []
    return sorted([
        f.name for f in RESULTS_DIR.iterdir()
        if f.is_dir() and not f.name.startswith('.')
    ])


def parse_effort_level(folder_name: str) -> str:
    """Parse effort level from folder name."""
    name = folder_name.lower()
    if "high-effort" in name:
        return "high"
    elif "medium-effort" in name:
        return "medium"
    elif "low-effort" in name:
        return "low"
    return "unknown"


def categorize_experiments(folders: List[str]) -> Dict[str, List[str]]:
    """Categorize experiments by type."""
    categories = {
        "GPT-5 Reasoning Effort": [],
        "Qwen Models": [],
        "Claude Models": [],
        "Other": [],
    }

    for folder in folders:
        name = folder.lower()
        if "gpt-5" in name:
            categories["GPT-5 Reasoning Effort"].append(folder)
        elif "qwen" in name:
            categories["Qwen Models"].append(folder)
        elif "claude" in name:
            categories["Claude Models"].append(folder)
        else:
            categories["Other"].append(folder)

    return {k: v for k, v in categories.items() if v}


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def create_token_comparison_chart(batches: List[BatchMetrics]) -> Optional[go.Figure]:
    """Create token usage comparison chart."""
    if not PLOTLY_AVAILABLE or not batches:
        return None

    data = []
    for batch in batches:
        # Shorten the name for display
        short_name = get_short_experiment_name(batch.folder_name)

        data.append({
            "Experiment": short_name,
            "Agent Alpha": batch.avg_alpha_tokens,
            "Agent Beta": batch.avg_beta_tokens,
        })

    df = pd.DataFrame(data)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Agent Alpha",
        x=df["Experiment"],
        y=df["Agent Alpha"],
        marker_color="#3b82f6"
    ))
    fig.add_trace(go.Bar(
        name="Agent Beta",
        x=df["Experiment"],
        y=df["Agent Beta"],
        marker_color="#ef4444"
    ))

    fig.update_layout(
        title="Average Token Usage by Agent",
        xaxis_title="Experiment",
        yaxis_title="Tokens (estimated)",
        barmode="group",
        legend_title="Agent",
        xaxis_tickangle=-45,
        height=400,
    )

    return fig


def create_payoff_comparison_chart(batches: List[BatchMetrics]) -> Optional[go.Figure]:
    """Create payoff comparison chart with error bars."""
    if not PLOTLY_AVAILABLE or not batches:
        return None

    data = []
    for batch in batches:
        short_name = get_short_experiment_name(batch.folder_name)

        data.append({
            "Experiment": short_name,
            "Alpha Avg": batch.avg_alpha_utility,
            "Alpha Std": batch.std_alpha_utility,
            "Beta Avg": batch.avg_beta_utility,
            "Beta Std": batch.std_beta_utility,
        })

    df = pd.DataFrame(data)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="Agent Alpha",
        x=df["Experiment"],
        y=df["Alpha Avg"],
        error_y=dict(type='data', array=df["Alpha Std"]),
        marker_color="#3b82f6"
    ))

    fig.add_trace(go.Bar(
        name="Agent Beta",
        x=df["Experiment"],
        y=df["Beta Avg"],
        error_y=dict(type='data', array=df["Beta Std"]),
        marker_color="#ef4444"
    ))

    fig.update_layout(
        title="Average Payoff by Agent (with Std Dev)",
        xaxis_title="Experiment",
        yaxis_title="Utility",
        barmode="group",
        legend_title="Agent",
        xaxis_tickangle=-45,
        height=400,
    )

    return fig


def create_consensus_round_chart(batches: List[BatchMetrics]) -> Optional[go.Figure]:
    """Create consensus round comparison chart."""
    if not PLOTLY_AVAILABLE or not batches:
        return None

    data = []
    for batch in batches:
        short_name = get_short_experiment_name(batch.folder_name)

        data.append({
            "Experiment": short_name,
            "Avg Round": batch.avg_consensus_round,
            "Consensus Rate": batch.consensus_rate * 100,
        })

    df = pd.DataFrame(data)

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(
            name="Avg Consensus Round",
            x=df["Experiment"],
            y=df["Avg Round"],
            marker_color="#10b981"
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            name="Consensus Rate (%)",
            x=df["Experiment"],
            y=df["Consensus Rate"],
            mode="lines+markers",
            marker_color="#f59e0b",
            line=dict(width=3)
        ),
        secondary_y=True,
    )

    fig.update_layout(
        title="Consensus Round and Rate Comparison",
        xaxis_title="Experiment",
        xaxis_tickangle=-45,
        height=400,
    )
    fig.update_yaxes(title_text="Average Round", secondary_y=False)
    fig.update_yaxes(title_text="Consensus Rate (%)", secondary_y=True, range=[0, 100])

    return fig


def create_nash_welfare_chart(batches: List[BatchMetrics]) -> Optional[go.Figure]:
    """Create Nash welfare comparison chart."""
    if not PLOTLY_AVAILABLE or not batches:
        return None

    data = []
    for batch in batches:
        short_name = get_short_experiment_name(batch.folder_name)

        data.append({
            "Experiment": short_name,
            "Nash Welfare": batch.avg_nash_welfare,
            "Std": batch.std_nash_welfare,
        })

    df = pd.DataFrame(data)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["Experiment"],
        y=df["Nash Welfare"],
        error_y=dict(type='data', array=df["Std"]),
        marker_color="#8b5cf6",
        name="Nash Welfare"
    ))

    fig.update_layout(
        title="Nash Welfare Comparison (Geometric Mean of Utilities)",
        xaxis_title="Experiment",
        yaxis_title="Nash Welfare",
        xaxis_tickangle=-45,
        height=400,
    )

    return fig


def create_reasoning_length_chart(batches: List[BatchMetrics]) -> Optional[go.Figure]:
    """Create reasoning trace length comparison chart."""
    if not PLOTLY_AVAILABLE or not batches:
        return None

    data = []
    for batch in batches:
        short_name = get_short_experiment_name(batch.folder_name)

        data.append({
            "Experiment": short_name,
            "Avg Response": batch.avg_response_length,
            "Avg Reasoning Trace": batch.avg_reasoning_trace_length,
        })

    df = pd.DataFrame(data)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Avg Response Length",
        x=df["Experiment"],
        y=df["Avg Response"],
        marker_color="#64748b"
    ))
    fig.add_trace(go.Bar(
        name="Avg Reasoning Trace",
        x=df["Experiment"],
        y=df["Avg Reasoning Trace"],
        marker_color="#ec4899"
    ))

    fig.update_layout(
        title="Response and Reasoning Trace Lengths",
        xaxis_title="Experiment",
        yaxis_title="Characters",
        barmode="group",
        xaxis_tickangle=-45,
        height=400,
    )

    return fig


# =============================================================================
# ENHANCED VISUALIZATION FUNCTIONS
# =============================================================================

def create_nash_welfare_by_round_chart(batches: List[BatchMetrics]) -> Optional[go.Figure]:
    """
    Create Nash welfare by round chart showing proposals from each agent.
    X-axis: rounds, Y-axis: Nash welfare, grouped by experiment and proposer.
    """
    if not PLOTLY_AVAILABLE or not batches:
        return None

    # Collect all proposals across experiments
    all_data = []
    for batch in batches:
        short_name = get_short_experiment_name(batch.folder_name)

        # Aggregate proposals by round across all runs
        proposals_by_round: Dict[int, Dict[str, List[float]]] = {}

        for exp in batch.experiments:
            for proposal in exp.proposals_by_round:
                round_num = proposal.round_num
                proposer = proposal.proposer

                if round_num not in proposals_by_round:
                    proposals_by_round[round_num] = {"Agent_Alpha": [], "Agent_Beta": []}

                if proposer in proposals_by_round[round_num]:
                    proposals_by_round[round_num][proposer].append(proposal.nash_welfare)

        # Calculate mean Nash welfare per round per proposer
        for round_num in sorted(proposals_by_round.keys()):
            for proposer, values in proposals_by_round[round_num].items():
                if values:
                    all_data.append({
                        "Experiment": short_name,
                        "Round": round_num,
                        "Proposer": proposer,
                        "Nash Welfare": sum(values) / len(values),
                    })

    if not all_data:
        return None

    df = pd.DataFrame(all_data)

    # Create grouped bar chart
    fig = px.bar(
        df,
        x="Round",
        y="Nash Welfare",
        color="Proposer",
        facet_col="Experiment",
        barmode="group",
        color_discrete_map={"Agent_Alpha": "#3b82f6", "Agent_Beta": "#ef4444"},
        title="Nash Welfare of Proposals by Round and Agent",
    )

    fig.update_layout(
        height=400,
        xaxis_title="Round",
        yaxis_title="Nash Welfare",
    )

    return fig


def create_consensus_violin_chart(batches: List[BatchMetrics], use_discounted: bool = True) -> Optional[go.Figure]:
    """
    Create violin plots showing distribution of consensus rounds.
    """
    if not PLOTLY_AVAILABLE or not batches:
        return None

    all_data = []
    for batch in batches:
        short_name = get_short_experiment_name(batch.folder_name)

        for exp in batch.experiments:
            if exp.consensus_reached:
                all_data.append({
                    "Experiment": short_name,
                    "Consensus Round": exp.final_round,
                    "Type": "Discounted" if use_discounted else "Raw",
                })

    if not all_data:
        return None

    df = pd.DataFrame(all_data)

    fig = go.Figure()

    for experiment in df["Experiment"].unique():
        exp_data = df[df["Experiment"] == experiment]["Consensus Round"]
        fig.add_trace(go.Violin(
            x=[experiment] * len(exp_data),
            y=exp_data,
            name=experiment,
            box_visible=True,
            meanline_visible=True,
            points="all",
            jitter=0.3,
        ))

    title_suffix = "(Discounted Utility)" if use_discounted else "(Raw Utility)"
    fig.update_layout(
        title=f"Consensus Round Distribution {title_suffix}",
        xaxis_title="Experiment",
        yaxis_title="Round at Consensus",
        xaxis_tickangle=-45,
        height=450,
        showlegend=False,
    )

    return fig


def create_utility_violin_chart(batches: List[BatchMetrics], use_discounted: bool = True) -> Optional[go.Figure]:
    """
    Create violin plots showing distribution of final utilities.
    """
    if not PLOTLY_AVAILABLE or not batches:
        return None

    all_data = []
    for batch in batches:
        short_name = get_short_experiment_name(batch.folder_name)

        for exp in batch.experiments:
            if use_discounted:
                alpha_util = exp.agent_alpha_utility
                beta_util = exp.agent_beta_utility
            else:
                alpha_util = exp.agent_alpha_raw_utility
                beta_util = exp.agent_beta_raw_utility

            all_data.append({
                "Experiment": short_name,
                "Agent": "Alpha",
                "Utility": alpha_util,
            })
            all_data.append({
                "Experiment": short_name,
                "Agent": "Beta",
                "Utility": beta_util,
            })

    if not all_data:
        return None

    df = pd.DataFrame(all_data)

    fig = go.Figure()

    experiments = df["Experiment"].unique()
    for i, experiment in enumerate(experiments):
        exp_data = df[df["Experiment"] == experiment]

        for agent, color in [("Alpha", "#3b82f6"), ("Beta", "#ef4444")]:
            agent_data = exp_data[exp_data["Agent"] == agent]["Utility"]
            offset = -0.2 if agent == "Alpha" else 0.2

            fig.add_trace(go.Violin(
                x=[i + offset] * len(agent_data),
                y=agent_data,
                name=f"{experiment} - {agent}",
                legendgroup=agent,
                scalegroup=agent,
                side="negative" if agent == "Alpha" else "positive",
                line_color=color,
                fillcolor=color,
                opacity=0.6,
                box_visible=True,
                meanline_visible=True,
                points="all",
                jitter=0.1,
            ))

    title_suffix = "(Discounted)" if use_discounted else "(Raw/Non-discounted)"
    fig.update_layout(
        title=f"Utility Distribution by Agent {title_suffix}",
        xaxis=dict(
            tickmode="array",
            tickvals=list(range(len(experiments))),
            ticktext=list(experiments),
            tickangle=-45,
        ),
        yaxis_title="Utility",
        height=450,
        violinmode="overlay",
        showlegend=False,
    )

    return fig


def create_phase_token_breakdown_chart(batches: List[BatchMetrics]) -> Optional[go.Figure]:
    """
    Create stacked bar chart showing token usage by phase.
    """
    if not PLOTLY_AVAILABLE or not batches:
        return None

    phases = ["game_setup", "preference_assignment", "discussion",
              "private_thinking", "proposal", "voting", "reflection"]
    phase_colors = {
        "game_setup": "#6366f1",       # Indigo
        "preference_assignment": "#8b5cf6",  # Purple
        "discussion": "#3b82f6",        # Blue
        "private_thinking": "#64748b",  # Slate
        "proposal": "#10b981",          # Emerald
        "voting": "#f59e0b",            # Amber
        "reflection": "#ec4899",        # Pink
    }

    data = []
    for batch in batches:
        short_name = get_short_experiment_name(batch.folder_name)

        row = {"Experiment": short_name}
        for phase in phases:
            row[phase] = getattr(batch.avg_phase_tokens, phase, 0)
        data.append(row)

    df = pd.DataFrame(data)

    fig = go.Figure()

    for phase in phases:
        fig.add_trace(go.Bar(
            name=phase.replace("_", " ").title(),
            x=df["Experiment"],
            y=df[phase],
            marker_color=phase_colors[phase],
        ))

    fig.update_layout(
        title="Token Usage by Negotiation Phase",
        xaxis_title="Experiment",
        yaxis_title="Tokens (estimated)",
        barmode="stack",
        xaxis_tickangle=-45,
        height=500,
        legend_title="Phase",
    )

    return fig


def create_phase_token_by_agent_chart(batches: List[BatchMetrics]) -> Optional[go.Figure]:
    """
    Create grouped bar chart showing token usage by phase for each agent.
    """
    if not PLOTLY_AVAILABLE or not batches:
        return None

    phases = ["game_setup", "preference_assignment", "discussion",
              "private_thinking", "proposal", "voting", "reflection"]

    # Use first batch for simplicity (can be extended to multi-batch)
    if len(batches) > 0:
        batch = batches[0]
    else:
        return None

    data_alpha = []
    data_beta = []

    for phase in phases:
        data_alpha.append(getattr(batch.avg_phase_tokens_alpha, phase, 0))
        data_beta.append(getattr(batch.avg_phase_tokens_beta, phase, 0))

    phase_labels = [p.replace("_", " ").title() for p in phases]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="Agent Alpha",
        x=phase_labels,
        y=data_alpha,
        marker_color="#3b82f6",
    ))

    fig.add_trace(go.Bar(
        name="Agent Beta",
        x=phase_labels,
        y=data_beta,
        marker_color="#ef4444",
    ))

    short_name = get_short_experiment_name(batch.folder_name)
    fig.update_layout(
        title=f"Token Usage by Phase and Agent ({short_name})",
        xaxis_title="Phase",
        yaxis_title="Tokens (estimated)",
        barmode="group",
        xaxis_tickangle=-45,
        height=450,
    )

    return fig


def create_reasoning_vs_payoff_chart(batches: List[BatchMetrics]) -> Optional[go.Figure]:
    """
    Create scatter plot: X = reasoning tokens, Y = final payoff.
    Shows relationship between reasoning effort and outcomes.
    """
    if not PLOTLY_AVAILABLE or not batches:
        return None

    all_data = []
    for batch in batches:
        short_name = get_short_experiment_name(batch.folder_name)

        for exp in batch.experiments:
            # Total reasoning tokens (private_thinking phase)
            reasoning_tokens = exp.phase_tokens.private_thinking

            # Calculate average payoff
            avg_payoff = (exp.agent_alpha_utility + exp.agent_beta_utility) / 2

            all_data.append({
                "Experiment": short_name,
                "Reasoning Tokens": reasoning_tokens,
                "Avg Payoff": avg_payoff,
                "Alpha Payoff": exp.agent_alpha_utility,
                "Beta Payoff": exp.agent_beta_utility,
                "Nash Welfare": exp.nash_welfare,
                "Consensus": "Yes" if exp.consensus_reached else "No",
            })

    if not all_data:
        return None

    df = pd.DataFrame(all_data)

    fig = px.scatter(
        df,
        x="Reasoning Tokens",
        y="Avg Payoff",
        color="Experiment",
        symbol="Consensus",
        size="Nash Welfare",
        hover_data=["Alpha Payoff", "Beta Payoff", "Nash Welfare"],
        title="Reasoning Effort vs. Average Payoff",
    )

    fig.update_layout(
        xaxis_title="Reasoning Tokens (Private Thinking Phase)",
        yaxis_title="Average Payoff (Discounted)",
        height=500,
    )

    return fig


def create_phase_reasoning_payoff_chart(batches: List[BatchMetrics]) -> Optional[go.Figure]:
    """
    Create detailed breakdown: reasoning by phase vs. payoff impact.
    Shows which phase provides best "bang for buck" in reasoning.
    """
    if not PLOTLY_AVAILABLE or not batches:
        return None

    phases = ["discussion", "private_thinking", "proposal", "reflection"]

    # Calculate correlation between phase tokens and payoffs
    all_data = []

    for batch in batches:
        short_name = get_short_experiment_name(batch.folder_name)

        for exp in batch.experiments:
            avg_payoff = (exp.agent_alpha_utility + exp.agent_beta_utility) / 2

            for phase in phases:
                tokens = getattr(exp.phase_tokens, phase, 0)
                all_data.append({
                    "Experiment": short_name,
                    "Phase": phase.replace("_", " ").title(),
                    "Tokens": tokens,
                    "Avg Payoff": avg_payoff,
                    "Nash Welfare": exp.nash_welfare,
                })

    if not all_data:
        return None

    df = pd.DataFrame(all_data)

    fig = px.scatter(
        df,
        x="Tokens",
        y="Avg Payoff",
        color="Phase",
        facet_col="Experiment",
        facet_col_wrap=2,
        trendline="ols",
        title="Phase-Specific Reasoning vs. Payoff (with Trendlines)",
        hover_data=["Nash Welfare"],
    )

    fig.update_layout(
        height=600,
        xaxis_title="Tokens in Phase",
        yaxis_title="Average Payoff",
    )

    return fig


def create_phase_efficiency_chart(batches: List[BatchMetrics]) -> Optional[go.Figure]:
    """
    Calculate and visualize reasoning efficiency by phase.
    Efficiency = payoff gained per token spent in each phase.
    """
    if not PLOTLY_AVAILABLE or not batches:
        return None

    phases = ["discussion", "private_thinking", "proposal", "voting", "reflection"]

    # Aggregate data per phase per experiment
    efficiency_data = []

    for batch in batches:
        short_name = get_short_experiment_name(batch.folder_name)

        # Calculate average payoff and tokens per phase
        avg_payoff = (batch.avg_alpha_utility + batch.avg_beta_utility) / 2

        for phase in phases:
            tokens = getattr(batch.avg_phase_tokens, phase, 0)
            # Efficiency ratio: higher is better (more payoff, fewer tokens)
            efficiency = avg_payoff / max(tokens, 1) * 1000  # Scale for readability

            efficiency_data.append({
                "Experiment": short_name,
                "Phase": phase.replace("_", " ").title(),
                "Tokens": tokens,
                "Avg Payoff": avg_payoff,
                "Efficiency": efficiency,
            })

    if not efficiency_data:
        return None

    df = pd.DataFrame(efficiency_data)

    fig = px.bar(
        df,
        x="Phase",
        y="Tokens",
        color="Experiment",
        barmode="group",
        title="Token Investment by Phase Across Experiments",
        text="Tokens",
    )

    fig.update_traces(textposition="outside")
    fig.update_layout(
        height=450,
        xaxis_title="Negotiation Phase",
        yaxis_title="Tokens (estimated)",
        xaxis_tickangle=-45,
    )

    return fig


# =============================================================================
# MAIN COMPARISON VIEW
# =============================================================================

def render_comparison_view():
    """Render the main comparison view."""
    st.header("📊 Batch Comparison View")
    st.markdown("*Compare multiple experiment batches side-by-side*")

    # Get available experiments
    folders = get_experiment_folders()

    if not folders:
        st.warning("No experiment folders found. Run some experiments first!")
        return

    # Categorize experiments
    categories = categorize_experiments(folders)

    # Sidebar: Multi-select with categories
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔍 Select Experiments to Compare")

    selected_folders = []

    # Show categories with expandable sections
    for category, category_folders in categories.items():
        with st.sidebar.expander(f"📁 {category} ({len(category_folders)})", expanded=True):
            selected = st.multiselect(
                f"Select from {category}",
                options=category_folders,
                default=[],
                key=f"select_{category}",
                format_func=lambda x: x.replace("_runs5_comp1", "").replace("_", " ")[:50]
            )
            selected_folders.extend(selected)

    # Quick select buttons
    st.sidebar.markdown("---")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("Select All GPT-5"):
            selected_folders = categories.get("GPT-5 Reasoning Effort", [])
            st.rerun()
    with col2:
        if st.button("Clear All"):
            st.rerun()

    # Main content area
    if not selected_folders:
        st.info("👈 Select experiments from the sidebar to compare")

        # Show available experiments summary
        st.subheader("Available Experiments")
        for category, category_folders in categories.items():
            st.markdown(f"**{category}**: {len(category_folders)} experiments")
            for f in category_folders[:3]:
                st.markdown(f"  - {f.replace('_', ' ')}")
            if len(category_folders) > 3:
                st.markdown(f"  - ... and {len(category_folders) - 3} more")
        return

    # Analyze selected batches
    st.info(f"Analyzing {len(selected_folders)} experiments...")

    batches = []
    for folder in selected_folders:
        folder_path = RESULTS_DIR / folder
        batch = analyze_batch(folder_path)
        if batch:
            batches.append(batch)

    if not batches:
        st.error("Failed to analyze selected experiments")
        return

    # Summary metrics at top
    st.subheader("📈 Summary Metrics")
    cols = st.columns(5)

    with cols[0]:
        total_runs = sum(b.num_runs for b in batches)
        st.metric("Total Runs", total_runs)

    with cols[1]:
        avg_consensus = sum(b.consensus_rate for b in batches) / len(batches)
        st.metric("Avg Consensus Rate", f"{avg_consensus:.0%}")

    with cols[2]:
        avg_nash = sum(b.avg_nash_welfare for b in batches) / len(batches)
        st.metric("Avg Nash Welfare", f"{avg_nash:.1f}")

    with cols[3]:
        avg_alpha = sum(b.avg_alpha_utility for b in batches) / len(batches)
        st.metric("Avg Alpha Utility", f"{avg_alpha:.1f}")

    with cols[4]:
        avg_beta = sum(b.avg_beta_utility for b in batches) / len(batches)
        st.metric("Avg Beta Utility", f"{avg_beta:.1f}")

    # Comparison table
    st.subheader("📋 Comparison Table")
    comparison_df = compare_batches(batches)
    st.dataframe(comparison_df, use_container_width=True)

    # Download button for CSV
    csv = comparison_df.to_csv(index=False)
    st.download_button(
        "📥 Download as CSV",
        csv,
        "experiment_comparison.csv",
        "text/csv"
    )

    # Visualization tabs
    st.subheader("📊 Visualizations")

    viz_tabs = st.tabs([
        "💰 Payoffs",
        "🔢 Tokens by Phase",
        "🤝 Consensus Distribution",
        "⚖️ Nash Welfare by Round",
        "🧠 Reasoning vs Payoff"
    ])

    with viz_tabs[0]:
        st.markdown("### Agent Payoff Comparison")

        # Toggle for discounted vs raw utilities
        utility_type = st.radio(
            "Utility Type",
            ["Discounted", "Non-Discounted (Raw)"],
            horizontal=True,
            key="payoff_utility_type"
        )
        use_discounted = utility_type == "Discounted"

        # Bar chart with error bars
        fig = create_payoff_comparison_chart(batches)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Install plotly for visualizations: `pip install plotly`")

        # Violin plot for distribution
        st.markdown("#### Utility Distribution (Violin Plot)")
        fig_violin = create_utility_violin_chart(batches, use_discounted=use_discounted)
        if fig_violin:
            st.plotly_chart(fig_violin, use_container_width=True)

        # Detailed payoff table
        with st.expander("📊 Detailed Payoff Data"):
            payoff_data = []
            for batch in batches:
                payoff_data.append({
                    "Experiment": batch.folder_name,
                    "Alpha Avg (Disc)": f"{batch.avg_alpha_utility:.2f}",
                    "Alpha Avg (Raw)": f"{batch.avg_alpha_raw_utility:.2f}",
                    "Beta Avg (Disc)": f"{batch.avg_beta_utility:.2f}",
                    "Beta Avg (Raw)": f"{batch.avg_beta_raw_utility:.2f}",
                    "Diff (Alpha-Beta)": f"{batch.avg_alpha_utility - batch.avg_beta_utility:.2f}",
                })
            st.dataframe(pd.DataFrame(payoff_data), use_container_width=True)

    with viz_tabs[1]:
        st.markdown("### Token Usage by Negotiation Phase")
        st.markdown("""
        *Token usage broken down by the 7 phases: Game Setup, Preference Assignment,
        Discussion, Private Thinking, Proposal, Voting, and Reflection.*
        """)

        # Stacked bar chart by phase
        fig = create_phase_token_breakdown_chart(batches)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

        # Per-agent breakdown
        st.markdown("#### Token Usage by Phase and Agent")
        if len(batches) > 0:
            selected_batch_idx = st.selectbox(
                "Select experiment for agent breakdown",
                range(len(batches)),
                format_func=lambda i: batches[i].folder_name.replace("_", " ")[:50],
                key="phase_agent_select"
            )
            fig_agent = create_phase_token_by_agent_chart([batches[selected_batch_idx]])
            if fig_agent:
                st.plotly_chart(fig_agent, use_container_width=True)

        # Detailed phase table
        with st.expander("📊 Detailed Phase Token Data"):
            for batch in batches:
                st.markdown(f"**{batch.folder_name}**")
                phase_data = batch.avg_phase_tokens.to_dict()
                phase_df = pd.DataFrame([
                    {"Phase": k.replace("_", " ").title(), "Tokens": v}
                    for k, v in phase_data.items()
                ])
                st.dataframe(phase_df, use_container_width=True)

    with viz_tabs[2]:
        st.markdown("### Consensus Round Distribution")
        st.markdown("""
        *Violin plots showing the distribution of rounds at which consensus was reached.*
        """)

        # Toggle for discounted vs raw
        consensus_type = st.radio(
            "View Type",
            ["Discounted Utility Context", "Raw Utility Context"],
            horizontal=True,
            key="consensus_type"
        )

        # Violin plot for consensus rounds
        fig = create_consensus_violin_chart(batches, use_discounted=(consensus_type == "Discounted Utility Context"))
        if fig:
            st.plotly_chart(fig, use_container_width=True)

        # Also show the original bar chart
        st.markdown("#### Average Consensus Round with Rate")
        fig_bar = create_consensus_round_chart(batches)
        if fig_bar:
            st.plotly_chart(fig_bar, use_container_width=True)

        # Consensus details
        with st.expander("📊 Consensus Details"):
            for batch in batches:
                st.markdown(f"**{batch.folder_name}**")
                st.markdown(f"- Rate: {batch.consensus_rate:.0%}")
                st.markdown(f"- Avg Round: {batch.avg_consensus_round:.1f}")
                st.markdown(f"- All Rounds: {batch.consensus_rounds}")

    with viz_tabs[3]:
        st.markdown("### Nash Welfare by Round")
        st.markdown("""
        *Track Nash Welfare of proposals made by each agent across negotiation rounds.
        This shows how the quality of proposals evolves over time.*

        **Formula**: `Nash Welfare = (U_alpha × U_beta)^(1/2)`
        """)

        # Nash welfare by round chart
        fig = create_nash_welfare_by_round_chart(batches)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No proposal data available for Nash welfare by round analysis.")

        # Also show the aggregate Nash welfare
        st.markdown("#### Aggregate Nash Welfare Comparison")
        fig_agg = create_nash_welfare_chart(batches)
        if fig_agg:
            st.plotly_chart(fig_agg, use_container_width=True)

        # Show Nash welfare table
        with st.expander("📊 Nash Welfare Details"):
            nash_data = []
            for batch in batches:
                nash_data.append({
                    "Experiment": batch.folder_name,
                    "Nash Welfare (Disc)": f"{batch.avg_nash_welfare:.2f} ± {batch.std_nash_welfare:.2f}",
                    "Nash Welfare (Raw)": f"{batch.avg_nash_welfare_raw:.2f} ± {batch.std_nash_welfare_raw:.2f}",
                })
            st.dataframe(pd.DataFrame(nash_data), use_container_width=True)

    with viz_tabs[4]:
        st.markdown("### Reasoning Effort vs. Payoff Analysis")
        st.markdown("""
        *Analyze the relationship between reasoning tokens (especially private thinking)
        and final payoffs. Identify which phases provide the best "bang for buck".*
        """)

        # Overall reasoning vs payoff scatter
        st.markdown("#### Overall Reasoning Tokens vs. Average Payoff")
        fig = create_reasoning_vs_payoff_chart(batches)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Insufficient data for reasoning vs payoff analysis.")

        # Phase-specific breakdown
        st.markdown("#### Phase-Specific Reasoning vs. Payoff")
        st.markdown("*Faceted scatter plots with trendlines to identify which phase's reasoning correlates most with payoff.*")
        try:
            fig_phase = create_phase_reasoning_payoff_chart(batches)
            if fig_phase:
                st.plotly_chart(fig_phase, use_container_width=True)
        except Exception as e:
            st.info(f"Phase-specific analysis requires statsmodels for trendlines. Error: {e}")

        # Phase efficiency comparison
        st.markdown("#### Token Investment by Phase")
        fig_eff = create_phase_efficiency_chart(batches)
        if fig_eff:
            st.plotly_chart(fig_eff, use_container_width=True)

        # Legacy reasoning length chart
        with st.expander("📊 Response Length Details"):
            fig_legacy = create_reasoning_length_chart(batches)
            if fig_legacy:
                st.plotly_chart(fig_legacy, use_container_width=True)

    # Qualitative Analysis Section
    st.subheader("💬 Qualitative Analysis: Concession Reasoning")

    # Select which experiment to examine
    selected_for_qual = st.selectbox(
        "Select experiment for qualitative analysis",
        options=[b.folder_name for b in batches],
        format_func=lambda x: x.replace("_", " ")
    )

    if selected_for_qual:
        folder_path = RESULTS_DIR / selected_for_qual
        interactions = load_all_interactions(folder_path, 1)

        if interactions:
            reasoning_entries = extract_concession_reasoning(interactions)

            if reasoning_entries:
                # Filter by type
                entry_types = list(set(e["type"] for e in reasoning_entries))
                selected_types = st.multiselect(
                    "Filter by type",
                    options=entry_types,
                    default=entry_types
                )

                filtered = [e for e in reasoning_entries if e["type"] in selected_types]

                # Display entries
                for entry in filtered[:20]:  # Limit to 20
                    agent_color = "#3b82f6" if "Alpha" in entry["agent"] else "#ef4444"

                    col1, col2 = st.columns([1, 4])
                    with col1:
                        st.markdown(f"""
                        <div style="
                            background: {agent_color}22;
                            padding: 8px;
                            border-radius: 8px;
                            text-align: center;
                        ">
                            <div style="font-weight: bold; color: {agent_color};">{entry['agent']}</div>
                            <div style="font-size: 12px;">Round {entry['round']}</div>
                            <div style="font-size: 11px; color: #6b7280;">{entry['type']}</div>
                        </div>
                        """, unsafe_allow_html=True)

                    with col2:
                        if entry["type"] == "vote":
                            vote_color = "#10b981" if entry["decision"] == "accept" else "#ef4444"
                            st.markdown(f"**Vote:** <span style='color: {vote_color};'>{entry['decision'].upper()}</span>", unsafe_allow_html=True)

                        st.markdown(f"*{entry['reasoning'][:300]}{'...' if len(entry['reasoning']) > 300 else ''}*")

                    st.markdown("---")

                if len(reasoning_entries) > 20:
                    st.info(f"Showing 20 of {len(reasoning_entries)} entries")
            else:
                st.info("No concession reasoning found in this experiment")
        else:
            st.warning("Could not load interactions for this experiment")


# =============================================================================
# STANDALONE PAGE
# =============================================================================

def main():
    """Main function for standalone usage."""
    st.set_page_config(
        page_title="Experiment Comparison",
        page_icon="📊",
        layout="wide"
    )

    render_comparison_view()


if __name__ == "__main__":
    main()
