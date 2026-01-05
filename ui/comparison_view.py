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
from typing import Dict, List, Optional
import pandas as pd

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
)

# Default results directory
RESULTS_DIR = Path(__file__).parent.parent / "experiments" / "results"


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

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
        short_name = batch.folder_name.replace("_runs5_comp1", "").replace("_vs_", " vs ")
        if len(short_name) > 40:
            short_name = short_name[:37] + "..."

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
        short_name = batch.folder_name.replace("_runs5_comp1", "").replace("_vs_", " vs ")
        if len(short_name) > 40:
            short_name = short_name[:37] + "..."

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
        short_name = batch.folder_name.replace("_runs5_comp1", "").replace("_vs_", " vs ")
        if len(short_name) > 40:
            short_name = short_name[:37] + "..."

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
        short_name = batch.folder_name.replace("_runs5_comp1", "").replace("_vs_", " vs ")
        if len(short_name) > 40:
            short_name = short_name[:37] + "..."

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
        short_name = batch.folder_name.replace("_runs5_comp1", "").replace("_vs_", " vs ")
        if len(short_name) > 40:
            short_name = short_name[:37] + "..."

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
        "🔢 Tokens",
        "🤝 Consensus",
        "⚖️ Nash Welfare",
        "💭 Reasoning Length"
    ])

    with viz_tabs[0]:
        st.markdown("### Agent Payoff Comparison")
        fig = create_payoff_comparison_chart(batches)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Install plotly for visualizations: `pip install plotly`")

        # Detailed payoff table
        with st.expander("📊 Detailed Payoff Data"):
            payoff_data = []
            for batch in batches:
                payoff_data.append({
                    "Experiment": batch.folder_name,
                    "Alpha Avg": f"{batch.avg_alpha_utility:.2f}",
                    "Alpha Std": f"{batch.std_alpha_utility:.2f}",
                    "Beta Avg": f"{batch.avg_beta_utility:.2f}",
                    "Beta Std": f"{batch.std_beta_utility:.2f}",
                    "Diff (Alpha-Beta)": f"{batch.avg_alpha_utility - batch.avg_beta_utility:.2f}",
                })
            st.dataframe(pd.DataFrame(payoff_data), use_container_width=True)

    with viz_tabs[1]:
        st.markdown("### Token Usage Comparison")
        fig = create_token_comparison_chart(batches)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

        # Token breakdown
        with st.expander("📊 Token Breakdown by Phase"):
            for batch in batches:
                st.markdown(f"**{batch.folder_name}**")
                for exp in batch.experiments[:1]:  # Just show first run
                    if exp.tokens_by_phase:
                        phase_df = pd.DataFrame([
                            {"Phase": k, "Tokens": v}
                            for k, v in sorted(exp.tokens_by_phase.items())
                        ])
                        st.dataframe(phase_df, use_container_width=True)

    with viz_tabs[2]:
        st.markdown("### Consensus Round Comparison")
        fig = create_consensus_round_chart(batches)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

        # Consensus details
        with st.expander("📊 Consensus Details"):
            for batch in batches:
                st.markdown(f"**{batch.folder_name}**")
                st.markdown(f"- Rate: {batch.consensus_rate:.0%}")
                st.markdown(f"- Avg Round: {batch.avg_consensus_round:.1f}")
                st.markdown(f"- Rounds: {batch.consensus_rounds}")

    with viz_tabs[3]:
        st.markdown("### Nash Welfare Comparison")
        st.markdown("""
        *Nash Welfare is the geometric mean of agent utilities, balancing efficiency and fairness.*

        **Formula**: `Nash Welfare = (U_alpha × U_beta)^(1/2)`
        """)

        fig = create_nash_welfare_chart(batches)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    with viz_tabs[4]:
        st.markdown("### Reasoning Trace Length Comparison")
        fig = create_reasoning_length_chart(batches)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

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
