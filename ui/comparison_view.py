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
    aggregate_batches,
    compare_batches,
    extract_concession_reasoning,
    get_aggregated_effort_comparison,
    get_experiment_type,
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


def get_effort_level(model_name: str) -> str:
    """
    Extract the effort level from a model name.
    Returns 'Low', 'Medium', 'High', or 'Unknown'.
    """
    name = model_name.lower()
    if 'low-effort' in name or 'low_effort' in name:
        return 'Low'
    elif 'medium-effort' in name or 'med-effort' in name or 'medium_effort' in name:
        return 'Medium'
    elif 'high-effort' in name or 'high_effort' in name:
        return 'High'
    else:
        return 'Unknown'


def parse_experiment_agents(folder_name: str) -> Tuple[str, str, str, str]:
    """
    Parse folder name to get model names and effort levels for Alpha and Beta.

    Returns:
        Tuple of (alpha_model, beta_model, alpha_effort, beta_effort)
    """
    import re

    # Remove common suffixes
    name = re.sub(r'_runs\d+_comp\d+.*$', '', folder_name)

    if '_vs_' not in name:
        return ('Unknown', 'Unknown', 'Unknown', 'Unknown')

    parts = name.split('_vs_')
    if len(parts) != 2:
        return ('Unknown', 'Unknown', 'Unknown', 'Unknown')

    alpha_model = parts[0]
    beta_model = parts[1]
    alpha_effort = get_effort_level(alpha_model)
    beta_effort = get_effort_level(beta_model)

    return (alpha_model, beta_model, alpha_effort, beta_effort)


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


def create_reasoning_vs_payoff_by_effort_chart(batches: List[BatchMetrics]) -> Optional[go.Figure]:
    """
    Create scatter plot showing per-agent reasoning tokens vs payoff,
    colored by effort level (Low/Medium/High).

    This helps answer: Do higher-effort agents get better payoffs?
    """
    if not PLOTLY_AVAILABLE or not batches:
        return None

    all_data = []
    for batch in batches:
        short_name = get_short_experiment_name(batch.folder_name)
        _, _, alpha_effort, beta_effort = parse_experiment_agents(batch.folder_name)

        for exp in batch.experiments:
            # Get per-agent reasoning tokens
            alpha_tokens = 0
            beta_tokens = 0
            if "Agent_Alpha" in exp.phase_tokens_by_agent:
                alpha_tokens = exp.phase_tokens_by_agent["Agent_Alpha"].private_thinking
            if "Agent_Beta" in exp.phase_tokens_by_agent:
                beta_tokens = exp.phase_tokens_by_agent["Agent_Beta"].private_thinking

            # Add Alpha agent data point
            all_data.append({
                "Experiment": short_name,
                "Agent": "Alpha",
                "Effort Level": alpha_effort,
                "Reasoning Tokens": alpha_tokens,
                "Payoff": exp.agent_alpha_utility,
                "Opponent Payoff": exp.agent_beta_utility,
                "Run": exp.run_number,
                "Consensus": "Yes" if exp.consensus_reached else "No",
            })

            # Add Beta agent data point
            all_data.append({
                "Experiment": short_name,
                "Agent": "Beta",
                "Effort Level": beta_effort,
                "Reasoning Tokens": beta_tokens,
                "Payoff": exp.agent_beta_utility,
                "Opponent Payoff": exp.agent_alpha_utility,
                "Run": exp.run_number,
                "Consensus": "Yes" if exp.consensus_reached else "No",
            })

    if not all_data:
        return None

    df = pd.DataFrame(all_data)

    # Define color mapping for effort levels
    color_map = {
        "Low": "#22c55e",      # Green
        "Medium": "#f59e0b",   # Orange/Amber
        "High": "#ef4444",     # Red
        "Unknown": "#6b7280",  # Gray
    }

    fig = px.scatter(
        df,
        x="Reasoning Tokens",
        y="Payoff",
        color="Effort Level",
        symbol="Agent",
        color_discrete_map=color_map,
        hover_data=["Experiment", "Agent", "Opponent Payoff", "Run", "Consensus"],
        title="Reasoning Tokens vs. Payoff by Effort Level",
    )

    # Make markers bigger
    fig.update_traces(marker=dict(size=14, line=dict(width=1, color='DarkSlateGrey')))

    fig.update_layout(
        xaxis_title="Reasoning Tokens (Private Thinking Phase)",
        yaxis_title="Agent Payoff (Discounted)",
        height=500,
    )

    # Add a horizontal line at y=50 (equal split reference)
    fig.add_hline(y=50, line_dash="dash", line_color="gray",
                  annotation_text="Equal Split", annotation_position="right")

    return fig


def create_reasoning_vs_payoff_by_position_chart(batches: List[BatchMetrics]) -> Optional[go.Figure]:
    """
    Create scatter plot showing per-agent reasoning tokens vs payoff,
    colored by position (Alpha/Beta).

    This helps answer: Is there a first-mover advantage (Alpha vs Beta)?
    """
    if not PLOTLY_AVAILABLE or not batches:
        return None

    all_data = []
    for batch in batches:
        short_name = get_short_experiment_name(batch.folder_name)
        _, _, alpha_effort, beta_effort = parse_experiment_agents(batch.folder_name)

        for exp in batch.experiments:
            # Get per-agent reasoning tokens
            alpha_tokens = 0
            beta_tokens = 0
            if "Agent_Alpha" in exp.phase_tokens_by_agent:
                alpha_tokens = exp.phase_tokens_by_agent["Agent_Alpha"].private_thinking
            if "Agent_Beta" in exp.phase_tokens_by_agent:
                beta_tokens = exp.phase_tokens_by_agent["Agent_Beta"].private_thinking

            # Add Alpha agent data point
            all_data.append({
                "Experiment": short_name,
                "Position": "Alpha (First)",
                "Effort Level": alpha_effort,
                "Reasoning Tokens": alpha_tokens,
                "Payoff": exp.agent_alpha_utility,
                "Opponent Payoff": exp.agent_beta_utility,
                "Run": exp.run_number,
                "Consensus": "Yes" if exp.consensus_reached else "No",
            })

            # Add Beta agent data point
            all_data.append({
                "Experiment": short_name,
                "Position": "Beta (Second)",
                "Effort Level": beta_effort,
                "Reasoning Tokens": beta_tokens,
                "Payoff": exp.agent_beta_utility,
                "Opponent Payoff": exp.agent_alpha_utility,
                "Run": exp.run_number,
                "Consensus": "Yes" if exp.consensus_reached else "No",
            })

    if not all_data:
        return None

    df = pd.DataFrame(all_data)

    # Define color mapping for positions
    color_map = {
        "Alpha (First)": "#3b82f6",   # Blue
        "Beta (Second)": "#ef4444",   # Red
    }

    fig = px.scatter(
        df,
        x="Reasoning Tokens",
        y="Payoff",
        color="Position",
        symbol="Effort Level",
        color_discrete_map=color_map,
        hover_data=["Experiment", "Effort Level", "Opponent Payoff", "Run", "Consensus"],
        title="Reasoning Tokens vs. Payoff by Agent Position",
    )

    # Make markers bigger
    fig.update_traces(marker=dict(size=14, line=dict(width=1, color='DarkSlateGrey')))

    fig.update_layout(
        xaxis_title="Reasoning Tokens (Private Thinking Phase)",
        yaxis_title="Agent Payoff (Discounted)",
        height=500,
    )

    # Add a horizontal line at y=50 (equal split reference)
    fig.add_hline(y=50, line_dash="dash", line_color="gray",
                  annotation_text="Equal Split", annotation_position="right")

    return fig


def create_payoff_difference_chart(batches: List[BatchMetrics]) -> Optional[go.Figure]:
    """
    Create scatter plot showing paired agent comparison:
    X = Reasoning token difference (Beta - Alpha)
    Y = Payoff difference (Beta - Alpha)

    Positive Y means Beta (second agent) won more.
    This helps identify if higher reasoning effort leads to better outcomes.
    """
    if not PLOTLY_AVAILABLE or not batches:
        return None

    all_data = []
    for batch in batches:
        short_name = get_short_experiment_name(batch.folder_name)
        _, _, alpha_effort, beta_effort = parse_experiment_agents(batch.folder_name)

        # Create a matchup label
        matchup = f"{alpha_effort} vs {beta_effort}"

        for exp in batch.experiments:
            # Get per-agent reasoning tokens
            alpha_tokens = 0
            beta_tokens = 0
            if "Agent_Alpha" in exp.phase_tokens_by_agent:
                alpha_tokens = exp.phase_tokens_by_agent["Agent_Alpha"].private_thinking
            if "Agent_Beta" in exp.phase_tokens_by_agent:
                beta_tokens = exp.phase_tokens_by_agent["Agent_Beta"].private_thinking

            # Calculate differences
            token_diff = beta_tokens - alpha_tokens
            payoff_diff = exp.agent_beta_utility - exp.agent_alpha_utility

            all_data.append({
                "Experiment": short_name,
                "Matchup": matchup,
                "Alpha Effort": alpha_effort,
                "Beta Effort": beta_effort,
                "Token Diff (β-α)": token_diff,
                "Payoff Diff (β-α)": payoff_diff,
                "Alpha Tokens": alpha_tokens,
                "Beta Tokens": beta_tokens,
                "Alpha Payoff": exp.agent_alpha_utility,
                "Beta Payoff": exp.agent_beta_utility,
                "Run": exp.run_number,
                "Consensus": "Yes" if exp.consensus_reached else "No",
            })

    if not all_data:
        return None

    df = pd.DataFrame(all_data)

    # Define color mapping for matchups
    color_map = {
        "Low vs Low": "#6b7280",      # Gray
        "Low vs Medium": "#f59e0b",   # Orange
        "Low vs High": "#ef4444",     # Red
        "Medium vs Low": "#84cc16",   # Lime
        "Medium vs Medium": "#6b7280",
        "Medium vs High": "#f97316",  # Orange-red
        "High vs Low": "#22c55e",     # Green
        "High vs Medium": "#14b8a6",  # Teal
        "High vs High": "#6b7280",    # Gray
    }

    fig = px.scatter(
        df,
        x="Token Diff (β-α)",
        y="Payoff Diff (β-α)",
        color="Matchup",
        color_discrete_map=color_map,
        hover_data=["Experiment", "Alpha Effort", "Beta Effort",
                    "Alpha Tokens", "Beta Tokens", "Alpha Payoff", "Beta Payoff", "Run"],
        title="Paired Agent Comparison: Token & Payoff Differences",
    )

    # Make markers bigger
    fig.update_traces(marker=dict(size=14, line=dict(width=1, color='DarkSlateGrey')))

    fig.update_layout(
        xaxis_title="Reasoning Token Difference (Beta - Alpha)",
        yaxis_title="Payoff Difference (Beta - Alpha)",
        height=550,
    )

    # Add reference lines
    fig.add_hline(y=0, line_dash="dash", line_color="gray",
                  annotation_text="Equal Payoff", annotation_position="right")
    fig.add_vline(x=0, line_dash="dash", line_color="gray",
                  annotation_text="Equal Tokens", annotation_position="top")

    # Add quadrant annotations
    fig.add_annotation(x=0.95, y=0.95, xref="paper", yref="paper",
                       text="β reasons more & wins", showarrow=False,
                       font=dict(size=10, color="gray"), opacity=0.7)
    fig.add_annotation(x=0.05, y=0.95, xref="paper", yref="paper",
                       text="α reasons more, β wins", showarrow=False,
                       font=dict(size=10, color="gray"), opacity=0.7)
    fig.add_annotation(x=0.95, y=0.05, xref="paper", yref="paper",
                       text="β reasons more, α wins", showarrow=False,
                       font=dict(size=10, color="gray"), opacity=0.7)
    fig.add_annotation(x=0.05, y=0.05, xref="paper", yref="paper",
                       text="α reasons more & wins", showarrow=False,
                       font=dict(size=10, color="gray"), opacity=0.7)

    return fig


def create_order_effect_chart(batches: List[BatchMetrics]) -> Optional[go.Figure]:
    """
    Compare the same matchups in both directions to isolate the effect of position.
    E.g., compare "Low vs High" (Low=Alpha) with "High vs Low" (High=Alpha).

    This helps answer: Does being Alpha give an advantage independent of effort?
    """
    if not PLOTLY_AVAILABLE or not batches:
        return None

    all_data = []
    for batch in batches:
        _, _, alpha_effort, beta_effort = parse_experiment_agents(batch.folder_name)

        if alpha_effort == "Unknown" or beta_effort == "Unknown":
            continue

        # Create a canonical matchup name (alphabetically sorted)
        efforts = sorted([alpha_effort, beta_effort])
        canonical_matchup = f"{efforts[0]} vs {efforts[1]}"

        # Who has the "stronger" effort in this matchup?
        effort_rank = {"Low": 1, "Medium": 2, "High": 3}
        alpha_rank = effort_rank.get(alpha_effort, 0)
        beta_rank = effort_rank.get(beta_effort, 0)

        for exp in batch.experiments:
            # Record Alpha's performance
            all_data.append({
                "Canonical Matchup": canonical_matchup,
                "Agent Effort": alpha_effort,
                "Position": "Alpha (First)",
                "Payoff": exp.agent_alpha_utility,
                "Is Stronger": "Yes" if alpha_rank > beta_rank else ("Equal" if alpha_rank == beta_rank else "No"),
                "Run": exp.run_number,
                "Experiment": get_short_experiment_name(batch.folder_name),
            })

            # Record Beta's performance
            all_data.append({
                "Canonical Matchup": canonical_matchup,
                "Agent Effort": beta_effort,
                "Position": "Beta (Second)",
                "Payoff": exp.agent_beta_utility,
                "Is Stronger": "Yes" if beta_rank > alpha_rank else ("Equal" if alpha_rank == beta_rank else "No"),
                "Run": exp.run_number,
                "Experiment": get_short_experiment_name(batch.folder_name),
            })

    if not all_data:
        return None

    df = pd.DataFrame(all_data)

    # Create box plot comparing Alpha vs Beta payoffs for each matchup
    fig = px.box(
        df,
        x="Canonical Matchup",
        y="Payoff",
        color="Position",
        color_discrete_map={
            "Alpha (First)": "#3b82f6",
            "Beta (Second)": "#ef4444",
        },
        hover_data=["Agent Effort", "Is Stronger", "Experiment"],
        title="Position Effect: Alpha vs Beta Payoffs by Matchup Type",
    )

    fig.update_layout(
        xaxis_title="Matchup (Canonical - alphabetically sorted)",
        yaxis_title="Agent Payoff (Discounted)",
        height=500,
        boxmode="group",
    )

    # Add equal split reference
    fig.add_hline(y=50, line_dash="dash", line_color="gray",
                  annotation_text="Equal Split", annotation_position="right")

    return fig


def create_effort_by_position_chart(batches: List[BatchMetrics]) -> Optional[go.Figure]:
    """
    Show how each effort level performs when playing as Alpha vs Beta.
    Aggregates across all matchups to see: Does High-effort do better as Alpha or Beta?
    """
    if not PLOTLY_AVAILABLE or not batches:
        return None

    all_data = []
    for batch in batches:
        _, _, alpha_effort, beta_effort = parse_experiment_agents(batch.folder_name)

        if alpha_effort == "Unknown" or beta_effort == "Unknown":
            continue

        for exp in batch.experiments:
            # Alpha's data point
            all_data.append({
                "Effort Level": alpha_effort,
                "Position": "Alpha (First)",
                "Payoff": exp.agent_alpha_utility,
                "Opponent Effort": beta_effort,
                "Experiment": get_short_experiment_name(batch.folder_name),
                "Run": exp.run_number,
            })

            # Beta's data point
            all_data.append({
                "Effort Level": beta_effort,
                "Position": "Beta (Second)",
                "Payoff": exp.agent_beta_utility,
                "Opponent Effort": alpha_effort,
                "Experiment": get_short_experiment_name(batch.folder_name),
                "Run": exp.run_number,
            })

    if not all_data:
        return None

    df = pd.DataFrame(all_data)

    # Order effort levels
    effort_order = ["Low", "Medium", "High"]
    df["Effort Level"] = pd.Categorical(df["Effort Level"], categories=effort_order, ordered=True)

    fig = px.box(
        df,
        x="Effort Level",
        y="Payoff",
        color="Position",
        color_discrete_map={
            "Alpha (First)": "#3b82f6",
            "Beta (Second)": "#ef4444",
        },
        hover_data=["Opponent Effort", "Experiment"],
        title="Effort Level Performance by Position",
        category_orders={"Effort Level": effort_order},
    )

    fig.update_layout(
        xaxis_title="Agent's Effort Level",
        yaxis_title="Agent Payoff (Discounted)",
        height=500,
        boxmode="group",
    )

    fig.add_hline(y=50, line_dash="dash", line_color="gray",
                  annotation_text="Equal Split", annotation_position="right")

    return fig


def create_asymmetric_matchup_chart(batches: List[BatchMetrics]) -> Optional[go.Figure]:
    """
    Compare asymmetric matchups side by side.
    E.g., "Low α vs High β" compared to "High α vs Low β"

    This directly shows if order matters for the same effort pairing.
    """
    if not PLOTLY_AVAILABLE or not batches:
        return None

    all_data = []
    for batch in batches:
        _, _, alpha_effort, beta_effort = parse_experiment_agents(batch.folder_name)

        if alpha_effort == "Unknown" or beta_effort == "Unknown":
            continue

        # Skip symmetric matchups for this chart
        if alpha_effort == beta_effort:
            continue

        matchup_label = f"{alpha_effort} α vs {beta_effort} β"

        for exp in batch.experiments:
            # We want to track: given this matchup, who won?
            alpha_won = exp.agent_alpha_utility > exp.agent_beta_utility
            payoff_diff = exp.agent_alpha_utility - exp.agent_beta_utility

            all_data.append({
                "Matchup": matchup_label,
                "Alpha Effort": alpha_effort,
                "Beta Effort": beta_effort,
                "Alpha Payoff": exp.agent_alpha_utility,
                "Beta Payoff": exp.agent_beta_utility,
                "Payoff Diff (α-β)": payoff_diff,
                "Winner": "Alpha" if alpha_won else ("Tie" if payoff_diff == 0 else "Beta"),
                "Run": exp.run_number,
            })

    if not all_data:
        return None

    df = pd.DataFrame(all_data)

    # Create grouped bar showing Alpha vs Beta payoff for each asymmetric matchup
    fig = go.Figure()

    matchups = df["Matchup"].unique()

    for matchup in sorted(matchups):
        subset = df[df["Matchup"] == matchup]
        avg_alpha = subset["Alpha Payoff"].mean()
        avg_beta = subset["Beta Payoff"].mean()
        std_alpha = subset["Alpha Payoff"].std()
        std_beta = subset["Beta Payoff"].std()

        fig.add_trace(go.Bar(
            name=f"{matchup} - Alpha",
            x=[matchup],
            y=[avg_alpha],
            error_y=dict(type='data', array=[std_alpha]),
            marker_color="#3b82f6",
            legendgroup=matchup,
        ))

        fig.add_trace(go.Bar(
            name=f"{matchup} - Beta",
            x=[matchup],
            y=[avg_beta],
            error_y=dict(type='data', array=[std_beta]),
            marker_color="#ef4444",
            legendgroup=matchup,
        ))

    fig.update_layout(
        title="Asymmetric Matchup Comparison: Who Wins?",
        xaxis_title="Matchup Configuration",
        yaxis_title="Average Payoff (± std)",
        barmode="group",
        height=500,
        showlegend=True,
    )

    fig.add_hline(y=50, line_dash="dash", line_color="gray",
                  annotation_text="Equal Split", annotation_position="right")

    return fig


def create_first_mover_advantage_chart(batches: List[BatchMetrics]) -> Optional[go.Figure]:
    """
    Aggregate analysis: Does Alpha (first mover) have an advantage overall?
    Shows win rate and average payoff advantage for Alpha across all experiments.
    """
    if not PLOTLY_AVAILABLE or not batches:
        return None

    all_data = []
    for batch in batches:
        _, _, alpha_effort, beta_effort = parse_experiment_agents(batch.folder_name)

        for exp in batch.experiments:
            alpha_won = exp.agent_alpha_utility > exp.agent_beta_utility
            payoff_diff = exp.agent_alpha_utility - exp.agent_beta_utility

            matchup_type = "Symmetric" if alpha_effort == beta_effort else "Asymmetric"

            all_data.append({
                "Matchup Type": matchup_type,
                "Alpha Effort": alpha_effort,
                "Beta Effort": beta_effort,
                "Alpha Won": 1 if alpha_won else 0,
                "Payoff Diff (α-β)": payoff_diff,
                "Experiment": get_short_experiment_name(batch.folder_name),
            })

    if not all_data:
        return None

    df = pd.DataFrame(all_data)

    # Create subplot with win rate and payoff difference
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Alpha Win Rate by Matchup Type", "Alpha Payoff Advantage (α-β)"),
        specs=[[{"type": "bar"}, {"type": "box"}]]
    )

    # Win rate by matchup type
    win_rates = df.groupby("Matchup Type")["Alpha Won"].mean().reset_index()
    win_rates.columns = ["Matchup Type", "Win Rate"]

    fig.add_trace(
        go.Bar(
            x=win_rates["Matchup Type"],
            y=win_rates["Win Rate"],
            marker_color=["#6b7280", "#f59e0b"],
            text=[f"{r:.0%}" for r in win_rates["Win Rate"]],
            textposition="outside",
        ),
        row=1, col=1
    )

    # Payoff difference distribution
    for matchup_type, color in [("Symmetric", "#6b7280"), ("Asymmetric", "#f59e0b")]:
        subset = df[df["Matchup Type"] == matchup_type]
        fig.add_trace(
            go.Box(
                y=subset["Payoff Diff (α-β)"],
                name=matchup_type,
                marker_color=color,
            ),
            row=1, col=2
        )

    fig.update_layout(
        height=450,
        showlegend=False,
        title_text="First-Mover (Alpha) Advantage Analysis",
    )

    # Add reference lines
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray", row=1, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=2)

    fig.update_yaxes(title_text="Win Rate", row=1, col=1)
    fig.update_yaxes(title_text="Payoff Difference", row=1, col=2)

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

def render_aggregated_effort_view():
    """
    Render the aggregated effort level comparison view.

    This view automatically aggregates all folders of the same experiment type
    (e.g., all low_vs_low folders) into single buckets for statistical analysis.
    """
    st.header("📊 Aggregated Effort Level Comparison")
    st.markdown("""
    *Automatically aggregates all runs of the same experiment type across multiple batch folders.*

    This view is ideal for statistical analysis when you have multiple SLURM jobs
    producing results for the same experiment configuration.
    """)

    # Get aggregated data
    with st.spinner("Aggregating experiments by effort level..."):
        aggregated = get_aggregated_effort_comparison(RESULTS_DIR)

    if not aggregated:
        st.warning("No GPT-5 effort experiments found to aggregate.")
        return

    # Display aggregation summary
    st.subheader("📦 Aggregation Summary")

    # Count folders per type for display
    folders = get_experiment_folders()
    folder_counts = {}
    for folder in folders:
        exp_type = get_experiment_type(folder)
        if exp_type:
            folder_counts[exp_type] = folder_counts.get(exp_type, 0) + 1

    # Create summary cards
    cols = st.columns(len(aggregated))
    for i, (exp_type, batch) in enumerate(sorted(aggregated.items())):
        with cols[i]:
            # Format the experiment type nicely
            alpha, beta = exp_type.split('_vs_')
            display_name = f"{alpha.title()} vs {beta.title()}"

            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
                padding: 15px;
                border-radius: 10px;
                text-align: center;
                color: white;
            ">
                <h3 style="margin: 0; color: white;">{display_name}</h3>
                <p style="margin: 5px 0; font-size: 24px; font-weight: bold;">{batch.num_runs} runs</p>
                <p style="margin: 0; font-size: 12px; opacity: 0.8;">
                    from {folder_counts.get(exp_type, 1)} batch folder(s)
                </p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # Convert to list for visualization functions
    # IMPORTANT: Do NOT modify batch.folder_name - it's needed for effort level parsing
    # The get_short_experiment_name() function handles display names automatically
    batches = list(aggregated.values())

    # Summary metrics
    st.subheader("📈 Aggregated Metrics")
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

    # Detailed comparison table
    st.subheader("📋 Detailed Comparison")

    # Create a more detailed table for aggregated data
    table_data = []
    for exp_type, batch in sorted(aggregated.items()):
        alpha, beta = exp_type.split('_vs_')
        table_data.append({
            "Matchup": f"{alpha.title()} vs {beta.title()}",
            "Total Runs": batch.num_runs,
            "Folders": folder_counts.get(exp_type, 1),
            "Consensus Rate": f"{batch.consensus_rate:.0%}",
            "Avg Round": f"{batch.avg_consensus_round:.1f}",
            "Alpha Utility": f"{batch.avg_alpha_utility:.1f} ± {batch.std_alpha_utility:.1f}",
            "Beta Utility": f"{batch.avg_beta_utility:.1f} ± {batch.std_beta_utility:.1f}",
            "Utility Diff (α-β)": f"{batch.avg_alpha_utility - batch.avg_beta_utility:+.1f}",
            "Nash Welfare": f"{batch.avg_nash_welfare:.1f} ± {batch.std_nash_welfare:.1f}",
        })

    df = pd.DataFrame(table_data)
    st.dataframe(df, use_container_width=True)

    # Download button
    csv = df.to_csv(index=False)
    st.download_button(
        "📥 Download Aggregated Results CSV",
        csv,
        "aggregated_effort_comparison.csv",
        "text/csv"
    )

    # Statistical significance indicator
    st.subheader("📊 Statistical Power")

    # Show a table indicating statistical power based on sample size
    power_data = []
    for exp_type, batch in sorted(aggregated.items()):
        alpha, beta = exp_type.split('_vs_')
        n = batch.num_runs

        # Rough power estimate (simplified)
        # With n=5, hard to detect small effects; n=20+ for medium effects
        if n >= 30:
            power_level = "🟢 Good (n≥30)"
        elif n >= 15:
            power_level = "🟡 Moderate (15≤n<30)"
        elif n >= 10:
            power_level = "🟠 Limited (10≤n<15)"
        else:
            power_level = "🔴 Low (n<10)"

        power_data.append({
            "Matchup": f"{alpha.title()} vs {beta.title()}",
            "N": n,
            "Power": power_level,
            "Recommendation": "Sufficient" if n >= 15 else f"Need ~{max(15-n, 5)} more runs"
        })

    st.dataframe(pd.DataFrame(power_data), use_container_width=True)

    # Visualizations using the same functions but with aggregated data
    st.subheader("📊 Visualizations")

    viz_tabs = st.tabs([
        "💰 Payoffs",
        "🔢 Tokens by Phase",
        "🤝 Consensus Distribution",
        "⚖️ Nash Welfare",
        "🧠 Reasoning vs Payoff"
    ])

    with viz_tabs[0]:
        st.markdown("### Agent Payoff Comparison (Aggregated)")

        utility_type = st.radio(
            "Utility Type",
            ["Discounted", "Non-Discounted (Raw)"],
            horizontal=True,
            key="agg_payoff_utility_type"
        )
        use_discounted = utility_type == "Discounted"

        fig = create_payoff_comparison_chart(batches)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

        fig_violin = create_utility_violin_chart(batches, use_discounted=use_discounted)
        if fig_violin:
            st.plotly_chart(fig_violin, use_container_width=True)

    with viz_tabs[1]:
        st.markdown("### Token Usage by Phase (Aggregated)")

        fig = create_phase_token_breakdown_chart(batches)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    with viz_tabs[2]:
        st.markdown("### Consensus Round Distribution (Aggregated)")

        fig = create_consensus_violin_chart(batches)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

        fig_bar = create_consensus_round_chart(batches)
        if fig_bar:
            st.plotly_chart(fig_bar, use_container_width=True)

    with viz_tabs[3]:
        st.markdown("### Nash Welfare (Aggregated)")

        fig = create_nash_welfare_chart(batches)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    with viz_tabs[4]:
        st.markdown("### Reasoning vs Payoff (Aggregated)")

        fig_effort = create_reasoning_vs_payoff_by_effort_chart(batches)
        if fig_effort:
            st.plotly_chart(fig_effort, use_container_width=True)

        fig_diff = create_payoff_difference_chart(batches)
        if fig_diff:
            st.plotly_chart(fig_diff, use_container_width=True)


def render_comparison_view():
    """Render the main comparison view."""
    st.header("📊 Batch Comparison View")
    st.markdown("*Compare multiple experiment batches side-by-side*")

    # Add view mode selector at the top
    st.sidebar.markdown("## 🎛️ View Mode")
    view_mode = st.sidebar.radio(
        "Select view mode",
        ["Individual Batches", "Aggregated by Effort Level"],
        help="Aggregated mode combines all folders of the same experiment type (e.g., all low_vs_low folders) into single statistical buckets."
    )

    if view_mode == "Aggregated by Effort Level":
        render_aggregated_effort_view()
        return

    # Original individual batch view continues below...

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
        "🧠 Reasoning vs Payoff",
        "🔄 Order Analysis"
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

        # Per-agent analysis by effort level
        st.markdown("#### Reasoning vs. Payoff by Effort Level")
        st.markdown("*Each dot is an individual agent. Color = effort level (Low/Medium/High). Symbol = position (Alpha/Beta).*")
        fig_effort = create_reasoning_vs_payoff_by_effort_chart(batches)
        if fig_effort:
            st.plotly_chart(fig_effort, use_container_width=True)
        else:
            st.info("Insufficient data for effort-level analysis.")

        # Per-agent analysis by position
        st.markdown("#### Reasoning vs. Payoff by Agent Position")
        st.markdown("*Each dot is an individual agent. Color = position (Alpha/Beta). Symbol = effort level.*")
        fig_position = create_reasoning_vs_payoff_by_position_chart(batches)
        if fig_position:
            st.plotly_chart(fig_position, use_container_width=True)
        else:
            st.info("Insufficient data for position analysis.")

        # Paired agent comparison (difference plot)
        st.markdown("#### Paired Agent Comparison (Differences)")
        st.markdown("""
        *Each dot = one negotiation. Shows the **difference** between paired agents (Beta − Alpha).*
        - **X-axis**: Token difference (positive = Beta reasoned more)
        - **Y-axis**: Payoff difference (positive = Beta won more)
        - **Quadrants**: Top-right = β reasons more & wins; Bottom-left = α reasons more & wins
        """)
        fig_diff = create_payoff_difference_chart(batches)
        if fig_diff:
            st.plotly_chart(fig_diff, use_container_width=True)
        else:
            st.info("Insufficient data for paired comparison.")

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

    with viz_tabs[5]:
        st.markdown("### Order & Position Analysis")
        st.markdown("""
        *Analyze whether agent position (Alpha = first mover, Beta = second mover) affects outcomes,
        and how this interacts with reasoning effort levels.*
        """)

        # First-mover advantage overview
        st.markdown("#### First-Mover Advantage Overview")
        st.markdown("*Does Alpha (first mover) have a systematic advantage? Compare symmetric vs asymmetric matchups.*")
        fig_fma = create_first_mover_advantage_chart(batches)
        if fig_fma:
            st.plotly_chart(fig_fma, use_container_width=True)
        else:
            st.info("Insufficient data for first-mover analysis.")

        # Order effect by matchup type
        st.markdown("#### Position Effect by Matchup Type")
        st.markdown("*Box plots comparing Alpha vs Beta payoffs for each canonical matchup (e.g., 'Low vs High' combines both orderings).*")
        fig_order = create_order_effect_chart(batches)
        if fig_order:
            st.plotly_chart(fig_order, use_container_width=True)
        else:
            st.info("Insufficient data for order effect analysis.")

        # Effort level performance by position
        st.markdown("#### Effort Level Performance by Position")
        st.markdown("*How does each effort level (Low/Medium/High) perform when playing as Alpha vs Beta?*")
        fig_effort_pos = create_effort_by_position_chart(batches)
        if fig_effort_pos:
            st.plotly_chart(fig_effort_pos, use_container_width=True)
        else:
            st.info("Insufficient data for effort-by-position analysis.")

        # Asymmetric matchup comparison
        st.markdown("#### Asymmetric Matchup Comparison")
        st.markdown("""
        *Compare asymmetric matchups directly. For example:*
        - *'Low α vs High β': Low-effort is Alpha (first), High-effort is Beta (second)*
        - *'High α vs Low β': High-effort is Alpha (first), Low-effort is Beta (second)*
        """)
        fig_asym = create_asymmetric_matchup_chart(batches)
        if fig_asym:
            st.plotly_chart(fig_asym, use_container_width=True)
        else:
            st.info("Insufficient asymmetric matchup data. Need experiments with both orderings (e.g., Low vs High AND High vs Low).")

        # Summary statistics
        with st.expander("📊 Order Analysis Summary Statistics"):
            order_stats = []
            for batch in batches:
                _, _, alpha_effort, beta_effort = parse_experiment_agents(batch.folder_name)
                alpha_wins = sum(1 for e in batch.experiments if e.agent_alpha_utility > e.agent_beta_utility)
                beta_wins = sum(1 for e in batch.experiments if e.agent_beta_utility > e.agent_alpha_utility)
                ties = len(batch.experiments) - alpha_wins - beta_wins

                order_stats.append({
                    "Experiment": get_short_experiment_name(batch.folder_name),
                    "Alpha Effort": alpha_effort,
                    "Beta Effort": beta_effort,
                    "Alpha Wins": alpha_wins,
                    "Beta Wins": beta_wins,
                    "Ties": ties,
                    "Alpha Win Rate": f"{alpha_wins / len(batch.experiments):.0%}" if batch.experiments else "N/A",
                    "Avg α Payoff": f"{batch.avg_alpha_utility:.1f}",
                    "Avg β Payoff": f"{batch.avg_beta_utility:.1f}",
                })
            st.dataframe(pd.DataFrame(order_stats), use_container_width=True)

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
