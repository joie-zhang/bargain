#!/usr/bin/env python3
"""
=============================================================================
Visualize Test-Time Compute Scaling Experiment Results
=============================================================================

Creates plots analyzing reasoning tokens vs negotiation payoff for TTC scaling
experiments. By default, aggregates results from ALL ttc_scaling_* directories.

Usage:
    python scripts/visualize_ttc_scaling.py                    # Aggregate all ttc_scaling dirs
    python scripts/visualize_ttc_scaling.py --single DIR       # Analyze single directory only
    python scripts/visualize_ttc_scaling.py --single DIR --model-pairs MODEL1 MODEL2  # Filter specific models

What it creates:
    visualization/figures/
    ├── plot1_avg_reasoning_vs_payoff.png    # Avg reasoning tokens vs payoff
    ├── plot2_per_round_reasoning.png        # Per-round reasoning analysis
    ├── plot3_phase_breakdown.png            # Reasoning by phase
    ├── plot4_instructed_vs_actual.png       # Prompted vs actual tokens
    ├── plot5_instructed_vs_payoff.png       # Prompted budget vs payoff
    └── data_summary.csv                     # Extracted data for analysis

Examples:
    # Aggregate all TTC scaling results (default behavior)
    python scripts/visualize_ttc_scaling.py

    # Analyze only a specific experiment directory
    python scripts/visualize_ttc_scaling.py --single experiments/results/ttc_scaling_20260124_222356

    # Filter for specific model pairs from a single directory
    python scripts/visualize_ttc_scaling.py --single experiments/results/ttc_scaling_20260125_050451 \
        --model-pairs o3-mini-high_vs_gpt-5-nano gpt-5.2-high_vs_gpt-5-nano

Dependencies:
    - matplotlib, seaborn, pandas, numpy

=============================================================================
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style with larger fonts globally
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'DejaVu Sans', 'Liberation Sans', 'sans-serif'],
    'font.size': 14,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 12,
    'legend.title_fontsize': 14,
    'figure.titlesize': 20,
    'mathtext.fontset': 'dejavusans',
})


def find_all_ttc_scaling_dirs(results_base: Path) -> List[Path]:
    """Find all ttc_scaling_* directories under the results base directory."""
    ttc_dirs = []

    for path in results_base.iterdir():
        if path.is_dir() and path.name.startswith("ttc_scaling_") and not path.is_symlink():
            ttc_dirs.append(path)

    # Sort by name (which includes timestamp) for consistent ordering
    return sorted(ttc_dirs)


def find_experiment_dirs(base_dir: Path) -> List[Path]:
    """Find all experiment result directories under base_dir."""
    experiment_dirs = []

    # Look for directories with experiment_results.json
    for path in base_dir.rglob("experiment_results.json"):
        experiment_dirs.append(path.parent)

    return sorted(experiment_dirs)


def load_experiment_data(exp_dir: Path) -> Optional[Dict[str, Any]]:
    """Load experiment results and agent interactions from a directory."""
    results_file = exp_dir / "experiment_results.json"
    if not results_file.exists():
        return None

    try:
        with open(results_file) as f:
            results = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Error loading {results_file}: {e}")
        return None

    # Load all interactions from run_*_all_interactions.json files
    # These contain the reasoning_tokens field in token_usage
    agent_interactions = defaultdict(lambda: {"interactions": []})

    for all_interactions_file in exp_dir.glob("run_*_all_interactions.json"):
        try:
            with open(all_interactions_file) as f:
                interactions_list = json.load(f)
                # Group interactions by agent_id
                for interaction in interactions_list:
                    agent_id = interaction.get("agent_id")
                    if agent_id:
                        agent_interactions[agent_id]["agent_id"] = agent_id
                        agent_interactions[agent_id]["interactions"].append(interaction)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error loading {all_interactions_file}: {e}")
            continue

    return {
        "results": results,
        "agent_interactions": dict(agent_interactions),
        "dir": exp_dir
    }


def extract_reasoning_tokens_by_phase(agent_data: Dict) -> Dict[str, List[int]]:
    """Extract reasoning tokens grouped by phase from agent interactions."""
    phase_tokens = defaultdict(list)

    interactions = agent_data.get("interactions", [])
    for interaction in interactions:
        phase = interaction.get("phase", "unknown")
        # reasoning_tokens is in token_usage.reasoning_tokens
        token_usage = interaction.get("token_usage", {})
        reasoning_tokens = token_usage.get("reasoning_tokens", 0)
        if reasoning_tokens and reasoning_tokens > 0:
            phase_tokens[phase].append(reasoning_tokens)

    return dict(phase_tokens)




def extract_reasoning_tokens_by_round(agent_data: Dict) -> Dict[int, List[int]]:
    """Extract reasoning tokens grouped by round from agent interactions."""
    round_tokens = defaultdict(list)

    interactions = agent_data.get("interactions", [])
    for interaction in interactions:
        round_num = interaction.get("round", 0)
        # reasoning_tokens is in token_usage.reasoning_tokens
        token_usage = interaction.get("token_usage", {})
        reasoning_tokens = token_usage.get("reasoning_tokens", 0)
        if reasoning_tokens and reasoning_tokens > 0:
            round_tokens[round_num].append(reasoning_tokens)

    return dict(round_tokens)


def parse_experiment_path(exp_dir: Path) -> Dict[str, Any]:
    """Parse experiment parameters from directory path."""
    parts = exp_dir.parts

    # Expected structure: .../model_vs_baseline/order/budget_X/comp_Y/
    params = {
        "model_pair": None,
        "model_order": None,
        "token_budget": None,
        "competition_level": None
    }

    for i, part in enumerate(parts):
        if "_vs_" in part:
            params["model_pair"] = part
        elif part in ("weak_first", "strong_first"):
            params["model_order"] = part
        elif part.startswith("budget_"):
            try:
                params["token_budget"] = int(part.replace("budget_", ""))
            except ValueError:
                pass
        elif part.startswith("comp_"):
            try:
                # Handle comp_1_0 format
                comp_str = part.replace("comp_", "").replace("_", ".")
                params["competition_level"] = float(comp_str)
            except ValueError:
                pass

    return params


def collect_all_data(base_dirs: List[Path], model_pairs_filter: Optional[List[str]] = None) -> pd.DataFrame:
    """Collect data from all experiments across multiple base directories into a DataFrame.

    Args:
        base_dirs: List of base directories to search for experiments
        model_pairs_filter: Optional list of model pair names to include (e.g., ["o3-mini-high_vs_gpt-5-nano"])
    """
    all_data = []
    total_experiment_dirs = 0

    for base_dir in base_dirs:
        experiment_dirs = find_experiment_dirs(base_dir)

        # Filter by model pairs if specified
        if model_pairs_filter:
            experiment_dirs = [
                d for d in experiment_dirs
                if any(mp in str(d) for mp in model_pairs_filter)
            ]

        total_experiment_dirs += len(experiment_dirs)
        print(f"  {base_dir.name}: {len(experiment_dirs)} experiments")

        for exp_dir in experiment_dirs:
            exp_data = load_experiment_data(exp_dir)
            if not exp_data:
                continue

            results = exp_data["results"]
            agent_interactions = exp_data["agent_interactions"]
            path_params = parse_experiment_path(exp_dir)

            # Get config info
            config = results.get("config", {})
            reasoning_config = config.get("reasoning_config", {})

            # Get final utilities
            final_utilities = results.get("final_utilities", {})

            # Identify which agent received the reasoning budget prompt
            reasoning_agent_ids = reasoning_config.get("reasoning_agent_ids", [])

            # Process each agent
            for agent_id, utility in final_utilities.items():
                agent_data = agent_interactions.get(agent_id, {})

                # Check if this agent is the one with the prompted reasoning budget
                is_prompted_reasoning = agent_id in reasoning_agent_ids

                # Extract reasoning tokens
                phase_tokens = extract_reasoning_tokens_by_phase(agent_data)
                round_tokens = extract_reasoning_tokens_by_round(agent_data)

                # Calculate totals and averages
                all_reasoning_tokens = []
                for tokens in phase_tokens.values():
                    all_reasoning_tokens.extend(tokens)

                total_reasoning = sum(all_reasoning_tokens) if all_reasoning_tokens else 0
                avg_reasoning = np.mean(all_reasoning_tokens) if all_reasoning_tokens else 0

                # Reasoning tokens in specific phases
                thinking_tokens = sum(phase_tokens.get("private_thinking_round_1", []) +
                                     phase_tokens.get("private_thinking_round_2", []) +
                                     phase_tokens.get("private_thinking_round_3", []) +
                                     phase_tokens.get("private_thinking_round_4", []) +
                                     phase_tokens.get("private_thinking_round_5", []) +
                                     phase_tokens.get("private_thinking_round_6", []) +
                                     phase_tokens.get("private_thinking_round_7", []) +
                                     phase_tokens.get("private_thinking_round_8", []) +
                                     phase_tokens.get("private_thinking_round_9", []) +
                                     phase_tokens.get("private_thinking_round_10", []))

                # Calculate average tokens per interaction for each phase type
                def get_phase_avg(phase_tokens, keyword):
                    all_tokens = []
                    for k, v in phase_tokens.items():
                        if keyword in k.lower():
                            all_tokens.extend(v)
                    return np.mean(all_tokens) if all_tokens else 0

                thinking_total = get_phase_avg(phase_tokens, "thinking")
                reflection_total = get_phase_avg(phase_tokens, "reflection")
                discussion_total = get_phase_avg(phase_tokens, "discussion")
                proposal_total = get_phase_avg(phase_tokens, "proposal")
                voting_total = get_phase_avg(phase_tokens, "voting")

                row = {
                    "experiment_dir": str(exp_dir),
                    "source_batch": base_dir.name,  # Track which ttc_scaling_* dir this came from
                    "agent_id": agent_id,
                    "utility": utility,
                    "model_pair": path_params["model_pair"],
                    "model_order": path_params["model_order"],
                    "token_budget_prompted": path_params["token_budget"] or reasoning_config.get("budget", 0),
                    "competition_level": path_params["competition_level"] or config.get("competition_level", 1.0),
                    "consensus_reached": results.get("consensus_reached", False),
                    "final_round": results.get("final_round", 0),
                    "total_reasoning_tokens": total_reasoning,
                    "avg_reasoning_tokens": avg_reasoning,
                    "is_prompted_reasoning": is_prompted_reasoning,  # True = adversary (budget-prompted), False = baseline
                    "thinking_tokens": thinking_total,
                    "reflection_tokens": reflection_total,
                    "discussion_tokens": discussion_total,
                    "proposal_tokens": proposal_total,
                    "voting_tokens": voting_total,
                    "num_interactions": len(agent_data.get("interactions", [])),
                }

                # Add per-round reasoning tokens
                for round_num in range(1, 11):
                    round_total = sum(round_tokens.get(round_num, []))
                    row[f"round_{round_num}_tokens"] = round_total

                all_data.append(row)

    print(f"Total: {total_experiment_dirs} experiment directories across {len(base_dirs)} batches")
    return pd.DataFrame(all_data)


def plot1_avg_reasoning_vs_payoff(df: pd.DataFrame, output_dir: Path):
    """
    Plot 1: Average payoff vs average reasoning tokens per stage, grouped by budget
    X-axis: Average reasoning tokens per negotiation stage (adversary model only)
    Y-axis: Average payoff (utility)
    Different colors for different model pairs.
    Only includes the adversary model (the one with prompted reasoning budget), not the baseline.
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Filter for the adversary model only (the one with prompted reasoning budget)
    # This excludes gpt-5-nano baseline which also has reasoning tokens
    df_reasoning = df[(df["avg_reasoning_tokens"] > 0) & (df["is_prompted_reasoning"] == True)].copy()

    if df_reasoning.empty:
        print("Warning: No reasoning model data found for Plot 1")
        return

    # Define colors for different model pairs
    model_pair_colors = {
        "o3-mini-high_vs_gpt-5-nano": "steelblue",
        "gpt-5.2-high_vs_gpt-5-nano": "coral",
        "claude-opus-4-5-thinking-32k_vs_gpt-5-nano": "forestgreen",
        "deepseek-r1_vs_gpt-5-nano": "purple",
        "grok-4_vs_gpt-5-nano": "goldenrod",
        "QwQ-32B_vs_gpt-5-nano": "crimson",
    }
    default_color = "gray"

    # Get unique model pairs
    model_pairs = df_reasoning["model_pair"].unique()

    for model_pair in model_pairs:
        df_mp = df_reasoning[df_reasoning["model_pair"] == model_pair]

        # Group by token budget (averaging across model_order)
        budget_groups = df_mp.groupby("token_budget_prompted").agg({
            "avg_reasoning_tokens": "mean",
            "utility": "mean"
        }).reset_index()
        budget_groups.columns = ["budget", "tokens_mean", "utility_mean"]

        color = model_pair_colors.get(model_pair, default_color)
        label = model_pair.replace("_vs_", " vs ").replace("-", "-")

        # Scatter plot (no lines)
        ax.scatter(
            budget_groups["tokens_mean"],
            budget_groups["utility_mean"],
            s=180,
            color=color,
            label=label,
            alpha=0.85,
            edgecolors='white',
            linewidth=1.5
        )

        # Label points with budget
        for _, row in budget_groups.iterrows():
            ax.annotate(
                f'{int(row["budget"]):,}',
                (row["tokens_mean"], row["utility_mean"]),
                textcoords="offset points",
                xytext=(8, 8),
                fontsize=10,
                color=color
            )

    ax.set_xlabel("Avg Reasoning Tokens per Stage (Adversary Model)")
    ax.set_ylabel("Avg Payoff (Utility)")
    ax.set_title("Adversary Model Payoff vs Reasoning Tokens\nper Stage by Prompting Budget")
    ax.legend(loc='best', fontsize=10)

    plt.tight_layout()
    output_path = output_dir / "plot1_avg_reasoning_vs_payoff.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot2_per_round_reasoning(df: pd.DataFrame, output_dir: Path):
    """
    Plot 2: Payoff vs Reasoning Tokens by Negotiation Stage
    Subplots for each phase (thinking, discussion, proposal, voting, reflection)
    - Averages across strong_first and weak_first
    - Consistent legend and colors across all subplots
    - Normalized axes
    - Only includes the adversary model (prompted reasoning), not baseline
    """
    # Filter for the adversary model only (the one with prompted reasoning budget)
    df_reasoning = df[(df["total_reasoning_tokens"] > 0) & (df["is_prompted_reasoning"] == True)].copy()

    if df_reasoning.empty:
        print("Warning: No reasoning model data found for Plot 2")
        return

    # Define phases and their display names
    phases = [
        ("thinking_tokens", "Private Thinking"),
        ("discussion_tokens", "Discussion"),
        ("proposal_tokens", "Proposal"),
        ("voting_tokens", "Voting"),
        ("reflection_tokens", "Reflection"),
    ]

    # Define budget levels and consistent color mapping
    budget_levels = [100, 500, 1000, 3000, 5000, 10000, 20000, 30000]
    colors = plt.cm.viridis(np.linspace(0, 1, len(budget_levels)))
    budget_color_map = {b: colors[i] for i, b in enumerate(budget_levels)}

    # Average across strong_first and weak_first by grouping
    # Group by experiment parameters (excluding model_order) and average
    group_cols = ["token_budget_prompted", "competition_level"]
    phase_cols = [p[0] for p in phases]

    # Aggregate: average utility and phase tokens across model orders
    agg_data = df_reasoning.groupby(group_cols).agg({
        "utility": "mean",
        **{col: "mean" for col in phase_cols}
    }).reset_index()

    # Calculate global axis limits for normalization
    all_x_vals = []
    all_y_vals = []
    for phase_col, _ in phases:
        phase_data = agg_data[agg_data[phase_col] > 0]
        if not phase_data.empty:
            all_x_vals.extend(phase_data[phase_col].tolist())
            all_y_vals.extend(phase_data["utility"].tolist())

    if not all_x_vals:
        print("Warning: No phase token data found for Plot 2")
        return

    # Set axis limits with padding
    x_min, x_max = min(all_x_vals), max(all_x_vals)
    y_min, y_max = min(all_y_vals), max(all_y_vals)
    y_padding = (y_max - y_min) * 0.1
    y_lim = (y_min - y_padding, y_max + y_padding)
    x_padding = (x_max - x_min) * 0.1
    x_min = max(x_min - x_padding, 0)  # Ensure non-negative
    x_max = x_max + x_padding

    # Create subplot grid for 5 phases (1 row, 5 columns)
    fig, axes = plt.subplots(1, 5, figsize=(22, 5.5))

    for idx, (phase_col, phase_name) in enumerate(phases):
        ax = axes[idx]

        # Filter for non-zero tokens in this phase
        phase_data = agg_data[agg_data[phase_col] > 0].copy()

        if phase_data.empty:
            ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title(phase_name, fontweight='bold')
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_lim)
            continue

        # Plot each budget level with consistent colors
        for budget in budget_levels:
            budget_data = phase_data[phase_data["token_budget_prompted"] == budget]
            if not budget_data.empty:
                ax.scatter(
                    budget_data[phase_col],
                    budget_data["utility"],
                    c=[budget_color_map[budget]],
                    s=180,
                    alpha=0.85,
                    edgecolors='white',
                    linewidth=1,
                    label=f"{budget:,}"
                )

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_lim)
        ax.set_xlabel("Reasoning Tokens")
        ax.set_ylabel("Payoff" if idx == 0 else "")
        ax.set_title(phase_name, fontweight='bold')

        # Only show y-axis label on leftmost plot
        if idx > 0:
            ax.set_yticklabels([])

    # Create single shared legend outside the plots
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=budget_color_map[b],
                          markersize=18, label=f"{b:,}") for b in budget_levels if b in agg_data["token_budget_prompted"].values]
    fig.legend(handles=handles, title="Budget", loc='center left', bbox_to_anchor=(1.01, 0.5),
               fontsize=14, title_fontsize=16)

    plt.suptitle("Adversary Model: Payoff vs Reasoning Tokens by Negotiation Stage", fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.subplots_adjust(right=0.95)  # Make room for legend
    output_path = output_dir / "plot2_per_round_reasoning.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot2b_per_round_reasoning_by_model(df: pd.DataFrame, output_dir: Path):
    """
    Plot 2b: Payoff vs Reasoning Tokens by Negotiation Stage, disaggregated by model
    Subplots for each phase (thinking, discussion, proposal, voting, reflection)
    Different colors for different adversary models (o3-mini-high vs gpt-5.2-high)
    - Only includes the adversary model (prompted reasoning), not baseline
    """
    # Filter for the adversary model only (the one with prompted reasoning budget)
    df_reasoning = df[(df["total_reasoning_tokens"] > 0) & (df["is_prompted_reasoning"] == True)].copy()

    if df_reasoning.empty:
        print("Warning: No reasoning model data found for Plot 2b")
        return

    # Define phases and their display names
    phases = [
        ("thinking_tokens", "Private Thinking"),
        ("discussion_tokens", "Discussion"),
        ("proposal_tokens", "Proposal"),
        ("voting_tokens", "Voting"),
        ("reflection_tokens", "Reflection"),
    ]

    # Define colors for different model pairs
    model_pair_colors = {
        "o3-mini-high_vs_gpt-5-nano": "steelblue",
        "gpt-5.2-high_vs_gpt-5-nano": "coral",
        "claude-opus-4-5-thinking-32k_vs_gpt-5-nano": "forestgreen",
        "deepseek-r1_vs_gpt-5-nano": "purple",
        "grok-4_vs_gpt-5-nano": "goldenrod",
        "QwQ-32B_vs_gpt-5-nano": "crimson",
    }
    default_color = "gray"

    # Get unique model pairs
    model_pairs = df_reasoning["model_pair"].unique()
    phase_cols = [p[0] for p in phases]

    # Calculate global axis limits for normalization across all models
    all_x_vals = []
    all_y_vals = []
    for model_pair in model_pairs:
        df_mp = df_reasoning[df_reasoning["model_pair"] == model_pair]
        agg_data = df_mp.groupby("token_budget_prompted").agg({
            "utility": "mean",
            **{col: "mean" for col in phase_cols}
        }).reset_index()

        for phase_col, _ in phases:
            phase_data = agg_data[agg_data[phase_col] > 0]
            if not phase_data.empty:
                all_x_vals.extend(phase_data[phase_col].tolist())
                all_y_vals.extend(phase_data["utility"].tolist())

    if not all_x_vals:
        print("Warning: No phase token data found for Plot 2b")
        return

    # Set axis limits with padding
    x_min, x_max = min(all_x_vals), max(all_x_vals)
    y_min, y_max = min(all_y_vals), max(all_y_vals)
    y_padding = (y_max - y_min) * 0.1
    y_lim = (y_min - y_padding, y_max + y_padding)
    x_padding = (x_max - x_min) * 0.1
    x_min = max(x_min - x_padding, 0)
    x_max = x_max + x_padding

    # Create subplot grid for 5 phases (1 row, 5 columns)
    fig, axes = plt.subplots(1, 5, figsize=(24, 6))

    for idx, (phase_col, phase_name) in enumerate(phases):
        ax = axes[idx]

        for model_pair in model_pairs:
            df_mp = df_reasoning[df_reasoning["model_pair"] == model_pair]

            # Average across model_order for each budget
            agg_data = df_mp.groupby("token_budget_prompted").agg({
                phase_col: "mean",
                "utility": "mean"
            }).reset_index()
            agg_data = agg_data[agg_data[phase_col] > 0]

            if agg_data.empty:
                continue

            color = model_pair_colors.get(model_pair, default_color)
            label = model_pair.split("_vs_")[0]  # Just the adversary model name

            ax.scatter(
                agg_data[phase_col],
                agg_data["utility"],
                c=color,
                s=120,
                alpha=0.85,
                edgecolors='white',
                linewidth=1,
                label=label if idx == 0 else None  # Only label in first subplot
            )

            # Label points with budget
            for _, row in agg_data.iterrows():
                ax.annotate(
                    f'{int(row["token_budget_prompted"]):,}',
                    (row[phase_col], row["utility"]),
                    textcoords="offset points",
                    xytext=(4, 4),
                    fontsize=7,
                    color=color,
                    alpha=0.8
                )

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_lim)
        ax.set_xlabel("Reasoning Tokens")
        ax.set_ylabel("Payoff" if idx == 0 else "")
        ax.set_title(phase_name, fontweight='bold')

        # Only show y-axis label on leftmost plot
        if idx > 0:
            ax.set_yticklabels([])

    # Create legend from first subplot
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Model", loc='center left', bbox_to_anchor=(1.01, 0.5),
               fontsize=12, title_fontsize=14)

    plt.suptitle("Adversary Model: Payoff vs Reasoning Tokens by Phase (by Model)", fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.subplots_adjust(right=0.92)  # Make room for legend
    output_path = output_dir / "plot2b_per_round_reasoning_by_model.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot3_phase_breakdown(df: pd.DataFrame, output_dir: Path):
    """
    Plot 3: Stacked bar chart showing reasoning tokens by phase
    Only includes the adversary model (prompted reasoning), not baseline.
    """
    # Filter for the adversary model only (the one with prompted reasoning budget)
    df_reasoning = df[(df["total_reasoning_tokens"] > 0) & (df["is_prompted_reasoning"] == True)].copy()

    if df_reasoning.empty:
        print("Warning: No reasoning model data found for Plot 3")
        return

    # Aggregate by token budget
    phase_cols = ["thinking_tokens", "reflection_tokens", "discussion_tokens",
                  "proposal_tokens", "voting_tokens"]

    agg_data = df_reasoning.groupby("token_budget_prompted")[phase_cols + ["utility"]].mean().reset_index()
    agg_data = agg_data.sort_values("token_budget_prompted")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Plot 3a: Stacked bar chart of tokens by phase
    ax1 = axes[0]
    x = range(len(agg_data))
    bottom = np.zeros(len(agg_data))

    colors = plt.cm.Set2(np.linspace(0, 1, len(phase_cols)))
    phase_labels = ["Thinking", "Reflection", "Discussion", "Proposal", "Voting"]

    for i, (col, label) in enumerate(zip(phase_cols, phase_labels)):
        values = agg_data[col].values
        ax1.bar(x, values, bottom=bottom, label=label, color=colors[i], alpha=0.85, width=0.7)
        bottom += values

    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{int(b):,}" for b in agg_data["token_budget_prompted"]], rotation=45, ha='right')
    ax1.set_xlabel("Prompted Token Budget")
    ax1.set_ylabel("Avg Reasoning Tokens per Stage")
    ax1.set_title("Adversary Model: Reasoning Tokens by Phase")
    ax1.legend(title="Phase", loc="upper left")

    # Plot 3b: Payoff vs total tokens with phase breakdown
    ax2 = axes[1]
    total_tokens = agg_data[phase_cols].sum(axis=1)

    ax2.scatter(total_tokens, agg_data["utility"], s=220, c=agg_data["token_budget_prompted"],
                cmap="viridis", alpha=0.85, edgecolors='black', linewidth=1.5)

    # Add trend line (only if we have enough data points)
    if len(total_tokens) >= 2 and total_tokens.std() > 0:
        try:
            z = np.polyfit(total_tokens, agg_data["utility"], 1)
            p = np.poly1d(z)
            x_line = np.linspace(total_tokens.min(), total_tokens.max(), 100)
            ax2.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=3, label="Trend")
        except np.linalg.LinAlgError:
            pass  # Skip trend line if fitting fails

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap="viridis",
                               norm=plt.Normalize(vmin=agg_data["token_budget_prompted"].min(),
                                                  vmax=agg_data["token_budget_prompted"].max()))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax2)
    cbar.set_label("Prompted Budget")

    ax2.set_xlabel("Sum of Avg Tokens per Stage (across phases)")
    ax2.set_ylabel("Avg Payoff (Utility)")
    ax2.set_title("Adversary Model: Payoff vs Reasoning")
    ax2.legend()

    plt.tight_layout()
    output_path = output_dir / "plot3_phase_breakdown.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot3b_phase_breakdown_by_model(df: pd.DataFrame, output_dir: Path):
    """
    Plot 3b: Stacked bar chart showing reasoning tokens by phase, disaggregated by model
    2x2 grid: each row is a different adversary model
    Left column: stacked bar chart, Right column: payoff vs tokens scatter
    Only includes the adversary model (prompted reasoning), not baseline.
    """
    # Filter for the adversary model only
    df_reasoning = df[(df["total_reasoning_tokens"] > 0) & (df["is_prompted_reasoning"] == True)].copy()

    if df_reasoning.empty:
        print("Warning: No reasoning model data found for Plot 3b")
        return

    phase_cols = ["thinking_tokens", "reflection_tokens", "discussion_tokens",
                  "proposal_tokens", "voting_tokens"]
    phase_labels = ["Thinking", "Reflection", "Discussion", "Proposal", "Voting"]
    phase_colors = plt.cm.Set2(np.linspace(0, 1, len(phase_cols)))

    # Get unique model pairs (limit to first 2 for 2x2 grid)
    model_pairs = df_reasoning["model_pair"].unique()[:2]

    if len(model_pairs) < 2:
        print("Warning: Need at least 2 model pairs for Plot 3b")
        return

    # Create 2x2 subplot grid
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    for row_idx, model_pair in enumerate(model_pairs):
        df_mp = df_reasoning[df_reasoning["model_pair"] == model_pair]
        model_name = model_pair.split("_vs_")[0]

        # Aggregate by token budget
        agg_data = df_mp.groupby("token_budget_prompted")[phase_cols + ["utility"]].mean().reset_index()
        agg_data = agg_data.sort_values("token_budget_prompted")

        # Left column: Stacked bar chart
        ax1 = axes[row_idx, 0]
        x = range(len(agg_data))
        bottom = np.zeros(len(agg_data))

        for i, (col, label) in enumerate(zip(phase_cols, phase_labels)):
            values = agg_data[col].values
            ax1.bar(x, values, bottom=bottom, label=label, color=phase_colors[i], alpha=0.85, width=0.7)
            bottom += values

        ax1.set_xticks(x)
        ax1.set_xticklabels([f"{int(b):,}" for b in agg_data["token_budget_prompted"]], rotation=45, ha='right')
        ax1.set_xlabel("Prompted Token Budget")
        ax1.set_ylabel("Avg Reasoning Tokens per Stage")
        ax1.set_title(f"{model_name}: Reasoning Tokens by Phase")
        if row_idx == 0:
            ax1.legend(title="Phase", loc="upper left", fontsize=9)

        # Right column: Payoff vs total tokens
        ax2 = axes[row_idx, 1]
        total_tokens = agg_data[phase_cols].sum(axis=1)

        scatter = ax2.scatter(total_tokens, agg_data["utility"], s=180, c=agg_data["token_budget_prompted"],
                    cmap="viridis", alpha=0.85, edgecolors='black', linewidth=1.5)

        # Add trend line
        if len(total_tokens) >= 2 and total_tokens.std() > 0:
            try:
                z = np.polyfit(total_tokens, agg_data["utility"], 1)
                p = np.poly1d(z)
                x_line = np.linspace(total_tokens.min(), total_tokens.max(), 100)
                ax2.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2, label="Trend")
            except np.linalg.LinAlgError:
                pass

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label("Prompted Budget")

        ax2.set_xlabel("Sum of Avg Tokens per Stage")
        ax2.set_ylabel("Avg Payoff (Utility)")
        ax2.set_title(f"{model_name}: Payoff vs Reasoning")
        ax2.legend(loc="best", fontsize=9)

    plt.suptitle("Phase Breakdown by Model (Adversary Only)", fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    output_path = output_dir / "plot3b_phase_breakdown_by_model.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot4_instructed_vs_actual(df: pd.DataFrame, output_dir: Path):
    """
    Plot 4: Instructed (prompted) reasoning tokens vs Actual reasoning tokens used
    X-axis: Token budget that was prompted/instructed (logarithmic)
    Y-axis: Average reasoning tokens per negotiation stage (not sum)
    - Averages across weak_first and strong_first for each budget
    """
    # Filter for reasoning model by checking who actually has reasoning tokens
    df_reasoning = df[df["total_reasoning_tokens"] > 0].copy()

    if df_reasoning.empty:
        print("Warning: No reasoning model data found for Plot 4")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # First, average across model_order (weak_first/strong_first) for each budget
    # Use avg_reasoning_tokens (per-stage average) instead of total
    agg_by_budget = df_reasoning.groupby("token_budget_prompted").agg({
        "avg_reasoning_tokens": "mean"
    }).reset_index()
    agg_by_budget.columns = ["budget", "avg_tokens"]

    # Plot 4a: Scatter plot with averaged data points
    ax1 = axes[0]

    ax1.scatter(
        agg_by_budget["budget"],
        agg_by_budget["avg_tokens"],
        s=220,
        alpha=0.85,
        edgecolors='black',
        linewidth=1.5,
        c='steelblue'
    )

    # Set logarithmic x-axis
    ax1.set_xscale('log')

    ax1.set_xlabel("Instructed Token Budget (log scale)")
    ax1.set_ylabel("Avg Reasoning Tokens per Stage")
    ax1.set_title("Actual Reasoning Tokens vs Instructed Budget")

    # Label points with budget values
    for _, row in agg_by_budget.iterrows():
        # Position 3,000 label below and to the right
        if row["budget"] == 3000:
            xytext = (12, -22)
        else:
            xytext = (10, 10)
        ax1.annotate(
            f'{int(row["budget"]):,}',
            (row["budget"], row["avg_tokens"]),
            textcoords="offset points",
            xytext=xytext,
            fontsize=18
        )

    # Plot 4b: Same data with ratio annotations
    ax2 = axes[1]

    ax2.plot(
        agg_by_budget["budget"],
        agg_by_budget["avg_tokens"],
        'o-',
        markersize=14,
        linewidth=3,
        color='steelblue',
        label='Avg tokens per stage'
    )

    # Set logarithmic x-axis
    ax2.set_xscale('log')

    # Add ratio annotations
    for _, row in agg_by_budget.iterrows():
        ratio = row["avg_tokens"] / row["budget"] if row["budget"] > 0 else 0
        # Position certain labels below and to the right to avoid overlap
        if row["budget"] in [100, 500, 1000, 3000, 5000]:
            xytext = (12, -22)  # below and to the right
        else:
            xytext = (10, 14)   # above and to the right
        ax2.annotate(
            f'{ratio:.1f}x',
            (row["budget"], row["avg_tokens"]),
            textcoords="offset points",
            xytext=xytext,
            fontsize=18,
            color='darkred'
        )

    ax2.set_xlabel("Instructed Token Budget (log scale)")
    ax2.set_ylabel("Avg Reasoning Tokens per Stage")
    ax2.set_title("Compliance with Token Instructions")

    plt.tight_layout()
    output_path = output_dir / "plot4_instructed_vs_actual.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot4b_instructed_vs_actual_by_model(df: pd.DataFrame, output_dir: Path):
    """
    Plot 4b: Instructed (prompted) reasoning tokens vs Actual reasoning tokens used
    Disaggregated by model pair (different colors for each model).
    X-axis: Token budget that was prompted/instructed (logarithmic)
    Y-axis: Average reasoning tokens per negotiation stage
    """
    # Filter for reasoning model by checking who actually has reasoning tokens
    df_reasoning = df[df["total_reasoning_tokens"] > 0].copy()

    if df_reasoning.empty:
        print("Warning: No reasoning model data found for Plot 4b")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Define colors for different model pairs
    model_pair_colors = {
        "o3-mini-high_vs_gpt-5-nano": "steelblue",
        "gpt-5.2-high_vs_gpt-5-nano": "coral",
        "claude-opus-4-5-thinking-32k_vs_gpt-5-nano": "forestgreen",
        "deepseek-r1_vs_gpt-5-nano": "purple",
        "grok-4_vs_gpt-5-nano": "goldenrod",
        "QwQ-32B_vs_gpt-5-nano": "crimson",
    }
    default_color = "gray"

    # Get unique model pairs
    model_pairs = df_reasoning["model_pair"].unique()

    # Plot 4b-a: Scatter plot disaggregated by model
    ax1 = axes[0]

    for model_pair in model_pairs:
        df_mp = df_reasoning[df_reasoning["model_pair"] == model_pair]

        # Average across model_order for each budget
        agg_by_budget = df_mp.groupby("token_budget_prompted").agg({
            "avg_reasoning_tokens": "mean"
        }).reset_index()
        agg_by_budget.columns = ["budget", "avg_tokens"]

        color = model_pair_colors.get(model_pair, default_color)
        label = model_pair.replace("_vs_", " vs ")

        ax1.scatter(
            agg_by_budget["budget"],
            agg_by_budget["avg_tokens"],
            s=150,
            alpha=0.85,
            edgecolors='white',
            linewidth=1,
            c=color,
            label=label
        )

        # Label points with budget values
        for _, row in agg_by_budget.iterrows():
            ax1.annotate(
                f'{int(row["budget"]):,}',
                (row["budget"], row["avg_tokens"]),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=9,
                color=color
            )

    ax1.set_xscale('log')
    ax1.set_xlabel("Instructed Token Budget (log scale)")
    ax1.set_ylabel("Avg Reasoning Tokens per Stage")
    ax1.set_title("Actual Reasoning Tokens vs Instructed Budget\n(by Model)")
    ax1.legend(loc='best', fontsize=9)

    # Plot 4b-b: Compliance ratio by model
    ax2 = axes[1]

    for model_pair in model_pairs:
        df_mp = df_reasoning[df_reasoning["model_pair"] == model_pair]

        agg_by_budget = df_mp.groupby("token_budget_prompted").agg({
            "avg_reasoning_tokens": "mean"
        }).reset_index()
        agg_by_budget.columns = ["budget", "avg_tokens"]

        # Calculate compliance ratio
        agg_by_budget["ratio"] = agg_by_budget["avg_tokens"] / agg_by_budget["budget"]

        color = model_pair_colors.get(model_pair, default_color)
        label = model_pair.replace("_vs_", " vs ")

        ax2.scatter(
            agg_by_budget["budget"],
            agg_by_budget["ratio"],
            s=150,
            alpha=0.85,
            edgecolors='white',
            linewidth=1,
            c=color,
            label=label
        )

    # Add reference line at ratio = 1.0
    ax2.axhline(y=1.0, color='gray', linestyle=':', linewidth=2, alpha=0.7, label='Perfect compliance (1.0x)')

    ax2.set_xscale('log')
    ax2.set_xlabel("Instructed Token Budget (log scale)")
    ax2.set_ylabel("Compliance Ratio (Actual / Instructed)")
    ax2.set_title("Token Compliance Ratio by Model")
    ax2.legend(loc='best', fontsize=9)

    plt.tight_layout()
    output_path = output_dir / "plot4b_instructed_vs_actual_by_model.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot5_instructed_vs_payoff(df: pd.DataFrame, output_dir: Path):
    """
    Plot 5: Instructed (prompted) reasoning tokens vs Model payoff
    X-axis: Token budget that was prompted/instructed (log scale)
    Y-axis: Final payoff (utility)
    - Averages across weak_first and strong_first for each budget
    Only includes the adversary model (prompted reasoning), not baseline.
    """
    # Filter for the adversary model only (the one with prompted reasoning budget)
    df_reasoning = df[(df["total_reasoning_tokens"] > 0) & (df["is_prompted_reasoning"] == True)].copy()

    if df_reasoning.empty:
        print("Warning: No reasoning model data found for Plot 5")
        return

    # Average across model_order (weak_first/strong_first) for each budget
    agg_by_budget = df_reasoning.groupby("token_budget_prompted").agg({
        "utility": "mean"
    }).reset_index()
    agg_by_budget.columns = ["budget", "utility_mean"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Plot 5a: Scatter plot with averaged data
    ax1 = axes[0]

    ax1.scatter(
        agg_by_budget["budget"],
        agg_by_budget["utility_mean"],
        s=220,
        alpha=0.85,
        edgecolors='black',
        linewidth=1.5,
        c='steelblue'
    )

    ax1.set_xscale('log')
    ax1.set_xlabel("Instructed Token Budget (log scale)")
    ax1.set_ylabel("Payoff (Utility)")
    ax1.set_title("Instructed Reasoning Budget vs Payoff")

    # Label points with budget values
    for _, row in agg_by_budget.iterrows():
        ax1.annotate(
            f'{int(row["budget"]):,}',
            (row["budget"], row["utility_mean"]),
            textcoords="offset points",
            xytext=(10, 10),
            fontsize=18
        )

    # Plot 5b: Line plot with averaged data
    ax2 = axes[1]

    ax2.plot(
        agg_by_budget["budget"],
        agg_by_budget["utility_mean"],
        'o-',
        markersize=14,
        linewidth=3,
        color='steelblue'
    )

    ax2.set_xscale('log')
    ax2.set_xlabel("Instructed Token Budget (log scale)")
    ax2.set_ylabel("Avg Payoff (Utility)")
    ax2.set_title("Payoff vs Instructed Budget")

    # Add horizontal line at 50 (fair split)
    ax2.axhline(y=50, color='gray', linestyle=':', alpha=0.7, linewidth=2, label='Fair split (50)')
    ax2.legend(loc='best')

    plt.tight_layout()
    output_path = output_dir / "plot5_instructed_vs_payoff.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot5b_instructed_vs_payoff_by_model(df: pd.DataFrame, output_dir: Path):
    """
    Plot 5b: Instructed (prompted) reasoning tokens vs Model payoff by model
    Similar to plot5, but differentiates between adversary models with different colors.
    X-axis: Token budget that was prompted/instructed (log scale)
    Y-axis: Final payoff (utility)
    Only includes the adversary model (prompted reasoning), not baseline.
    """
    # Filter for the adversary model only (the one with prompted reasoning budget)
    df_reasoning = df[(df["total_reasoning_tokens"] > 0) & (df["is_prompted_reasoning"] == True)].copy()

    if df_reasoning.empty:
        print("Warning: No reasoning model data found for Plot 5b")
        return

    # Define model colors
    model_pair_colors = {
        "o3-mini-high_vs_gpt-5-nano": "steelblue",
        "gpt-5.2-high_vs_gpt-5-nano": "coral",
        "claude-opus-4-5-thinking-32k_vs_gpt-5-nano": "forestgreen",
        "deepseek-r1_vs_gpt-5-nano": "purple",
        "grok-4_vs_gpt-5-nano": "goldenrod",
        "QwQ-32B_vs_gpt-5-nano": "crimson",
    }
    default_color = "gray"

    # Get unique model pairs
    model_pairs = df_reasoning["model_pair"].unique()

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Plot 5b-a: Scatter plot of budget vs payoff by model
    ax1 = axes[0]

    for model_pair in model_pairs:
        df_mp = df_reasoning[df_reasoning["model_pair"] == model_pair]

        agg_by_budget = df_mp.groupby("token_budget_prompted").agg({
            "utility": "mean"
        }).reset_index()
        agg_by_budget.columns = ["budget", "utility_mean"]

        color = model_pair_colors.get(model_pair, default_color)
        label = model_pair.replace("_vs_", " vs ")

        ax1.scatter(
            agg_by_budget["budget"],
            agg_by_budget["utility_mean"],
            s=180,
            alpha=0.85,
            edgecolors='white',
            linewidth=1,
            c=color,
            label=label
        )

        # Label points with budget values
        for _, row in agg_by_budget.iterrows():
            ax1.annotate(
                f'{int(row["budget"]):,}',
                (row["budget"], row["utility_mean"]),
                textcoords="offset points",
                xytext=(8, 8),
                fontsize=9,
                color=color
            )

    ax1.set_xscale('log')
    ax1.set_xlabel("Instructed Token Budget (log scale)")
    ax1.set_ylabel("Payoff (Utility)")
    ax1.set_title("Instructed Reasoning Budget vs Payoff\n(by Model)")
    ax1.axhline(y=50, color='gray', linestyle=':', alpha=0.7, linewidth=2, label='Fair split (50)')
    ax1.legend(loc='best', fontsize=9)

    # Plot 5b-b: Line plot showing trend by model
    ax2 = axes[1]

    for model_pair in model_pairs:
        df_mp = df_reasoning[df_reasoning["model_pair"] == model_pair]

        agg_by_budget = df_mp.groupby("token_budget_prompted").agg({
            "utility": "mean"
        }).reset_index()
        agg_by_budget.columns = ["budget", "utility_mean"]
        agg_by_budget = agg_by_budget.sort_values("budget")

        color = model_pair_colors.get(model_pair, default_color)
        label = model_pair.replace("_vs_", " vs ")

        ax2.plot(
            agg_by_budget["budget"],
            agg_by_budget["utility_mean"],
            'o-',
            markersize=10,
            linewidth=2,
            color=color,
            label=label
        )

    ax2.set_xscale('log')
    ax2.set_xlabel("Instructed Token Budget (log scale)")
    ax2.set_ylabel("Avg Payoff (Utility)")
    ax2.set_title("Payoff vs Instructed Budget\n(by Model)")
    ax2.axhline(y=50, color='gray', linestyle=':', alpha=0.7, linewidth=2, label='Fair split (50)')
    ax2.legend(loc='best', fontsize=9)

    plt.tight_layout()
    output_path = output_dir / "plot5b_instructed_vs_payoff_by_model.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot6_payoff_vs_reasoning_by_phase(df: pd.DataFrame, output_dir: Path):
    """
    Plot 6: Payoff vs Reasoning Tokens by Phase for Multiple Model Pairs

    Creates a 5x2 grid for EACH competition level (0.0, 0.25, 0.5, 0.75, 1.0):
    - 5 rows: one for each phase (Thinking, Discussion, Proposal, Voting, Reflection)
    - 2 columns: o3-mini-high_vs_gpt-5-nano (left), gpt-5.2-high_vs_gpt-5-nano (right)

    Each subplot shows scatter points (no connecting lines) with trendlines:
    - Adversary model (o3-mini-high or gpt-5.2-high) - the one with prompted reasoning budget
    - Baseline model (gpt-5-nano) - the non-prompted reasoning model

    X-axis: Average reasoning tokens per stage for that phase (normalized across subplots)
    Y-axis: Utility (payoff) (normalized across subplots)

    Includes trendline equations, R², and p-values.
    """
    from scipy import stats

    # Define the two model pairs we want to compare
    model_pairs = ["o3-mini-high_vs_gpt-5-nano", "gpt-5.2-high_vs_gpt-5-nano"]
    model_pair_labels = ["o3-mini-high vs gpt-5-nano", "gpt-5.2-high vs gpt-5-nano"]

    # Define phases and their corresponding column names
    phases = [
        ("thinking_tokens", "Private Thinking"),
        ("discussion_tokens", "Discussion"),
        ("proposal_tokens", "Proposal"),
        ("voting_tokens", "Voting"),
        ("reflection_tokens", "Reflection"),
    ]

    # Competition levels to iterate through
    competition_levels = [0.0, 0.25, 0.5, 0.75, 1.0]

    for comp_level in competition_levels:
        # Filter for this competition level
        df_filtered = df[df["competition_level"] == comp_level].copy()

        if df_filtered.empty:
            print(f"Warning: No data found for competition_level={comp_level} in Plot 6")
            continue

        # Check for missing data (budgets without both strong_first and weak_first)
        print(f"\n--- Checking for missing strong_first/weak_first data (comp={comp_level}) ---")
        for model_pair in model_pairs:
            df_mp = df_filtered[df_filtered["model_pair"] == model_pair]
            if df_mp.empty:
                print(f"  {model_pair}: NO DATA FOUND")
                continue

            budgets = df_mp["token_budget_prompted"].unique()
            for budget in sorted(budgets):
                df_budget = df_mp[df_mp["token_budget_prompted"] == budget]
                orders = df_budget["model_order"].unique()
                if "strong_first" not in orders or "weak_first" not in orders:
                    missing = []
                    if "strong_first" not in orders:
                        missing.append("strong_first")
                    if "weak_first" not in orders:
                        missing.append("weak_first")
                    print(f"  {model_pair}, budget={budget}: MISSING {', '.join(missing)}")

        print(f"--- End of missing data check (comp={comp_level}) ---\n")

        # First pass: collect all data to determine global axis limits
        all_x_vals = []
        all_y_vals = []

        for model_pair in model_pairs:
            df_mp = df_filtered[df_filtered["model_pair"] == model_pair]
            if df_mp.empty:
                continue

            df_adversary = df_mp[df_mp["is_prompted_reasoning"] == True]
            df_baseline = df_mp[df_mp["is_prompted_reasoning"] == False]

            for phase_col, _ in phases:
                for df_subset in [df_adversary, df_baseline]:
                    agg = df_subset.groupby("token_budget_prompted").agg({
                        phase_col: "mean",
                        "utility": "mean"
                    }).reset_index()
                    agg = agg[agg[phase_col] > 0]
                    if not agg.empty:
                        all_x_vals.extend(agg[phase_col].tolist())
                        all_y_vals.extend(agg["utility"].tolist())

        if not all_x_vals or not all_y_vals:
            print(f"Warning: No valid data for comp={comp_level}")
            continue

        # Calculate global axis limits with padding
        x_min, x_max = min(all_x_vals), max(all_x_vals)
        y_min, y_max = min(all_y_vals), max(all_y_vals)
        x_padding = (x_max - x_min) * 0.15 if x_max > x_min else 100
        y_padding = (y_max - y_min) * 0.15 if y_max > y_min else 5
        x_lim = (max(0, x_min - x_padding), x_max + x_padding)
        y_lim = (y_min - y_padding, y_max + y_padding)

        # Create 5x2 subplot grid
        fig, axes = plt.subplots(5, 2, figsize=(14, 24))

        for col_idx, (model_pair, model_pair_label) in enumerate(zip(model_pairs, model_pair_labels)):
            df_mp = df_filtered[df_filtered["model_pair"] == model_pair]

            if df_mp.empty:
                # Fill column with "No data" message
                for row_idx in range(5):
                    axes[row_idx, col_idx].text(
                        0.5, 0.5, f"No data for\n{model_pair_label}",
                        ha='center', va='center', transform=axes[row_idx, col_idx].transAxes,
                        fontsize=14
                    )
                    axes[row_idx, col_idx].set_xlim(x_lim)
                    axes[row_idx, col_idx].set_ylim(y_lim)
                continue

            # Separate adversary (prompted) and baseline (non-prompted) models
            df_adversary = df_mp[df_mp["is_prompted_reasoning"] == True]
            df_baseline = df_mp[df_mp["is_prompted_reasoning"] == False]

            # Get model names from the model_pair string
            parts = model_pair.split("_vs_")
            adversary_name = parts[0] if len(parts) == 2 else "Adversary"
            baseline_name = parts[1] if len(parts) == 2 else "Baseline"

            for row_idx, (phase_col, phase_name) in enumerate(phases):
                ax = axes[row_idx, col_idx]

                # Aggregate adversary data by budget (averaging across model orders)
                adv_agg = df_adversary.groupby("token_budget_prompted").agg({
                    phase_col: "mean",
                    "utility": "mean"
                }).reset_index()
                adv_agg.columns = ["budget", "tokens", "utility"]
                adv_agg = adv_agg[adv_agg["tokens"] > 0]  # Filter out zero token entries

                # Aggregate baseline data by budget (averaging across model orders)
                base_agg = df_baseline.groupby("token_budget_prompted").agg({
                    phase_col: "mean",
                    "utility": "mean"
                }).reset_index()
                base_agg.columns = ["budget", "tokens", "utility"]
                base_agg = base_agg[base_agg["tokens"] > 0]  # Filter out zero token entries

                # Plot adversary (blue circles, no connecting lines)
                if not adv_agg.empty:
                    ax.scatter(
                        adv_agg["tokens"],
                        adv_agg["utility"],
                        s=100,
                        marker='o',
                        label=adversary_name,
                        color='steelblue',
                        alpha=0.85,
                        edgecolors='white',
                        linewidth=1
                    )

                    # Add trendline if enough points
                    if len(adv_agg) >= 2:
                        slope, intercept, r_value, p_value, std_err = stats.linregress(
                            adv_agg["tokens"], adv_agg["utility"]
                        )
                        x_trend = np.linspace(adv_agg["tokens"].min(), adv_agg["tokens"].max(), 100)
                        y_trend = slope * x_trend + intercept
                        ax.plot(x_trend, y_trend, '--', color='steelblue', linewidth=1.5, alpha=0.7)

                        # Add equation and stats text
                        eq_text = f"y={slope:.3f}x+{intercept:.1f}\nR²={r_value**2:.3f}, p={p_value:.3f}"
                        ax.text(0.02, 0.98, eq_text, transform=ax.transAxes, fontsize=8,
                                verticalalignment='top', color='steelblue',
                                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

                # Plot baseline (coral squares, no connecting lines)
                if not base_agg.empty:
                    ax.scatter(
                        base_agg["tokens"],
                        base_agg["utility"],
                        s=100,
                        marker='s',
                        label=baseline_name,
                        color='coral',
                        alpha=0.85,
                        edgecolors='white',
                        linewidth=1
                    )

                    # Add trendline if enough points
                    if len(base_agg) >= 2:
                        slope, intercept, r_value, p_value, std_err = stats.linregress(
                            base_agg["tokens"], base_agg["utility"]
                        )
                        x_trend = np.linspace(base_agg["tokens"].min(), base_agg["tokens"].max(), 100)
                        y_trend = slope * x_trend + intercept
                        ax.plot(x_trend, y_trend, '--', color='coral', linewidth=1.5, alpha=0.7)

                        # Add equation and stats text
                        eq_text = f"y={slope:.3f}x+{intercept:.1f}\nR²={r_value**2:.3f}, p={p_value:.3f}"
                        ax.text(0.98, 0.02, eq_text, transform=ax.transAxes, fontsize=8,
                                verticalalignment='bottom', horizontalalignment='right', color='coral',
                                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

                # Add fair split line
                ax.axhline(y=50, color='gray', linestyle=':', alpha=0.6, linewidth=1.5)

                # Set normalized axis limits
                ax.set_xlim(x_lim)
                ax.set_ylim(y_lim)

                # Labels and title
                ax.set_xlabel("Avg Reasoning Tokens per Stage")
                ax.set_ylabel("Payoff (Utility)")

                # Title: phase name for first row, just phase for others
                if row_idx == 0:
                    ax.set_title(f"{phase_name}\n({model_pair_label})", fontweight='bold')
                else:
                    ax.set_title(f"{phase_name}")

                # Legend only on top row
                if row_idx == 0:
                    ax.legend(loc='upper right', fontsize=9)

        # Format competition level for filename
        comp_str = str(comp_level).replace(".", "_")

        plt.suptitle(
            f"Payoff vs Reasoning Tokens by Phase (Competition Level = {comp_level})\n"
            "Averaged across strong_first and weak_first",
            fontsize=16, fontweight='bold', y=1.01
        )

        plt.tight_layout()
        output_path = output_dir / f"plot6_payoff_vs_reasoning_by_phase_comp_{comp_str}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize TTC scaling experiment results")
    parser.add_argument("--single", type=str, default=None,
                       help="Analyze only a single directory instead of aggregating all")
    parser.add_argument("--results-base", type=str, default="experiments/results",
                       help="Base directory containing ttc_scaling_* folders (default: experiments/results)")
    parser.add_argument("--output-dir", "-o", type=str, default=None,
                       help="Output directory for figures (default: visualization/figures)")
    parser.add_argument("--model-pairs", type=str, nargs="+", default=None,
                       help="Filter for specific model pairs (e.g., o3-mini-high_vs_gpt-5-nano)")
    args = parser.parse_args()

    results_base = Path(args.results_base)

    if args.single:
        # Single directory mode (legacy behavior)
        base_dir = Path(args.single)
        if not base_dir.exists():
            print(f"Error: Results directory not found: {base_dir}")
            return 1
        base_dirs = [base_dir]
        output_dir = Path(args.output_dir) if args.output_dir else Path("visualization/figures")
        print(f"Analyzing single directory: {base_dir}")
    else:
        # Aggregate mode (default) - find all ttc_scaling_* directories
        if not results_base.exists():
            print(f"Error: Results base directory not found: {results_base}")
            return 1

        base_dirs = find_all_ttc_scaling_dirs(results_base)

        if not base_dirs:
            print(f"Error: No ttc_scaling_* directories found in {results_base}")
            return 1

        output_dir = Path(args.output_dir) if args.output_dir else Path("visualization/figures")
        print(f"Aggregating results from {len(base_dirs)} ttc_scaling directories:")
        for d in base_dirs:
            print(f"  - {d.name}")

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    print()

    # Collect all data
    print("Collecting experiment data...")
    if args.model_pairs:
        print(f"Filtering for model pairs: {args.model_pairs}")
    df = collect_all_data(base_dirs, model_pairs_filter=args.model_pairs)

    if df.empty:
        print("No data found!")
        return 1

    print(f"\nCollected {len(df)} data points from {df['experiment_dir'].nunique()} experiments")
    print(f"Source batches: {sorted(df['source_batch'].unique())}")
    print(f"Token budgets: {sorted(df['token_budget_prompted'].unique())}")
    print(f"Model orders: {df['model_order'].unique().tolist()}")
    print()

    # Save data summary
    summary_path = output_dir / "data_summary.csv"
    df.to_csv(summary_path, index=False)
    print(f"Saved data summary: {summary_path}")
    print()

    # Generate plots
    print("Generating plots...")
    plot1_avg_reasoning_vs_payoff(df, output_dir)
    plot2_per_round_reasoning(df, output_dir)
    plot2b_per_round_reasoning_by_model(df, output_dir)
    plot3_phase_breakdown(df, output_dir)
    plot3b_phase_breakdown_by_model(df, output_dir)
    plot4_instructed_vs_actual(df, output_dir)
    plot4b_instructed_vs_actual_by_model(df, output_dir)
    plot5_instructed_vs_payoff(df, output_dir)
    plot5b_instructed_vs_payoff_by_model(df, output_dir)
    plot6_payoff_vs_reasoning_by_phase(df, output_dir)

    print("\nDone!")
    return 0


if __name__ == "__main__":
    exit(main())
