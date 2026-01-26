#!/usr/bin/env python3
"""
=============================================================================
Visualize Non-Reasoning Model Token Usage in TTC Scaling Experiments
=============================================================================

Analyzes the effect of the REASONING DEPTH prompt (accidentally inserted into
the non-reasoning model's prompts) on output length. Tracks total_tokens for
Agent_Alpha (gpt-5-nano) across different budget conditions.

This is useful for investigating whether prompting a non-reasoning model with
reasoning depth instructions affects its output verbosity.

Usage:
    python visualization/visualize_nonreasoning_tokens.py                    # Aggregate all ttc_scaling dirs
    python visualization/visualize_nonreasoning_tokens.py --single DIR       # Analyze single directory

What it creates:
    visualization/figures/nonreasoning/
    ├── plot1_budget_vs_total_tokens.png       # Prompted budget vs non-reasoning model output
    ├── plot2_tokens_by_phase.png              # Token usage by negotiation phase
    ├── plot3_tokens_by_round.png              # Token usage by round
    ├── plot4_model_comparison.png             # Side-by-side reasoning vs non-reasoning
    ├── plot5_budget_effect_distribution.png   # Distribution of token counts per budget
    ├── plot6_payoff_vs_tokens.png             # Payoff vs tokens (both models, comp=1.0)
    ├── plot7_payoff_vs_budget.png             # Payoff vs prompted budget (both models, comp=1.0)
    └── nonreasoning_data_summary.csv          # Extracted data for analysis

Examples:
    # Aggregate all TTC scaling results (default behavior)
    python visualization/visualize_nonreasoning_tokens.py

    # Analyze only a specific experiment directory
    python visualization/visualize_nonreasoning_tokens.py --single experiments/results/ttc_scaling_20260124_224426

Dependencies:
    - matplotlib, seaborn, pandas, numpy

=============================================================================
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any, Optional
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

    return sorted(ttc_dirs)


def find_experiment_dirs(base_dir: Path) -> List[Path]:
    """Find all experiment result directories under base_dir."""
    experiment_dirs = []

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
    agent_interactions = defaultdict(lambda: {"interactions": []})

    for all_interactions_file in exp_dir.glob("run_*_all_interactions.json"):
        try:
            with open(all_interactions_file) as f:
                interactions_list = json.load(f)
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


def extract_total_tokens_by_phase(agent_data: Dict) -> Dict[str, List[int]]:
    """Extract total tokens grouped by phase from agent interactions."""
    phase_tokens = defaultdict(list)

    interactions = agent_data.get("interactions", [])
    for interaction in interactions:
        phase = interaction.get("phase", "unknown")
        token_usage = interaction.get("token_usage", {})
        total_tokens = token_usage.get("total_tokens", 0)
        if total_tokens and total_tokens > 0:
            phase_tokens[phase].append(total_tokens)

    return dict(phase_tokens)


def extract_total_tokens_by_round(agent_data: Dict) -> Dict[int, List[int]]:
    """Extract total tokens grouped by round from agent interactions."""
    round_tokens = defaultdict(list)

    interactions = agent_data.get("interactions", [])
    for interaction in interactions:
        round_num = interaction.get("round", 0)
        token_usage = interaction.get("token_usage", {})
        total_tokens = token_usage.get("total_tokens", 0)
        if total_tokens and total_tokens > 0:
            round_tokens[round_num].append(total_tokens)

    return dict(round_tokens)


def get_phase_category(phase_name: str) -> str:
    """Categorize phase into higher-level categories for plotting."""
    phase_lower = phase_name.lower()
    if "thinking" in phase_lower:
        return "Thinking"
    elif "discussion" in phase_lower:
        return "Discussion"
    elif "proposal" in phase_lower:
        return "Proposal"
    elif "voting" in phase_lower:
        return "Voting"
    elif "reflection" in phase_lower:
        return "Reflection"
    elif "setup" in phase_lower or "preference" in phase_lower:
        return "Setup"
    else:
        return "Other"


def parse_experiment_path(exp_dir: Path) -> Dict[str, Any]:
    """Parse experiment parameters from directory path."""
    parts = exp_dir.parts

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
                comp_str = part.replace("comp_", "").replace("_", ".")
                params["competition_level"] = float(comp_str)
            except ValueError:
                pass

    return params


def identify_nonreasoning_agent(agent_interactions: Dict, model_pair: str) -> Optional[str]:
    """
    Identify which agent is the non-reasoning model.

    In strong_first: Agent_Alpha = strong model (reasoning), Agent_Beta = weak (non-reasoning)
    In weak_first: Agent_Alpha = weak model (non-reasoning), Agent_Beta = strong (reasoning)

    But we also check for actual reasoning_tokens to be sure.
    """
    for agent_id, agent_data in agent_interactions.items():
        interactions = agent_data.get("interactions", [])
        has_reasoning_tokens = False
        for interaction in interactions:
            token_usage = interaction.get("token_usage", {})
            reasoning_tokens = token_usage.get("reasoning_tokens")
            # Handle None values explicitly
            if reasoning_tokens is not None and reasoning_tokens > 0:
                has_reasoning_tokens = True
                break

        # The non-reasoning agent is the one WITHOUT reasoning_tokens
        if not has_reasoning_tokens:
            return agent_id

    return None


def collect_all_data(base_dirs: List[Path]) -> pd.DataFrame:
    """Collect data from all experiments into a DataFrame."""
    all_data = []
    total_experiment_dirs = 0

    for base_dir in base_dirs:
        experiment_dirs = find_experiment_dirs(base_dir)
        total_experiment_dirs += len(experiment_dirs)
        print(f"  {base_dir.name}: {len(experiment_dirs)} experiments")

        for exp_dir in experiment_dirs:
            exp_data = load_experiment_data(exp_dir)
            if not exp_data:
                continue

            results = exp_data["results"]
            agent_interactions = exp_data["agent_interactions"]
            path_params = parse_experiment_path(exp_dir)

            config = results.get("config", {})
            reasoning_config = config.get("reasoning_config", {})
            final_utilities = results.get("final_utilities", {})

            # Identify the non-reasoning agent
            nonreasoning_agent = identify_nonreasoning_agent(
                agent_interactions, path_params.get("model_pair", "")
            )

            # Process each agent
            for agent_id, utility in final_utilities.items():
                agent_data = agent_interactions.get(agent_id, {})
                interactions = agent_data.get("interactions", [])

                # Check if this is the reasoning or non-reasoning model
                has_reasoning_tokens = False
                for interaction in interactions:
                    token_usage = interaction.get("token_usage", {})
                    reasoning_tokens = token_usage.get("reasoning_tokens")
                    if reasoning_tokens is not None and reasoning_tokens > 0:
                        has_reasoning_tokens = True
                        break

                is_nonreasoning = not has_reasoning_tokens

                # Extract token data
                phase_tokens = extract_total_tokens_by_phase(agent_data)
                round_tokens = extract_total_tokens_by_round(agent_data)

                # Calculate totals
                all_total_tokens = []
                for tokens in phase_tokens.values():
                    all_total_tokens.extend(tokens)

                total_tokens_sum = sum(all_total_tokens) if all_total_tokens else 0
                avg_tokens_per_interaction = np.mean(all_total_tokens) if all_total_tokens else 0

                # Categorized phase tokens
                category_tokens = defaultdict(list)
                for phase, tokens in phase_tokens.items():
                    category = get_phase_category(phase)
                    category_tokens[category].extend(tokens)

                row = {
                    "experiment_dir": str(exp_dir),
                    "source_batch": base_dir.name,
                    "agent_id": agent_id,
                    "is_nonreasoning": is_nonreasoning,
                    "utility": utility,
                    "model_pair": path_params["model_pair"],
                    "model_order": path_params["model_order"],
                    "token_budget_prompted": path_params["token_budget"] or reasoning_config.get("budget", 0),
                    "competition_level": path_params["competition_level"] or config.get("competition_level", 1.0),
                    "consensus_reached": results.get("consensus_reached", False),
                    "final_round": results.get("final_round", 0),
                    "total_tokens_sum": total_tokens_sum,
                    "avg_tokens_per_interaction": avg_tokens_per_interaction,
                    "num_interactions": len(interactions),
                    # Phase category averages
                    "setup_tokens_avg": np.mean(category_tokens["Setup"]) if category_tokens["Setup"] else 0,
                    "thinking_tokens_avg": np.mean(category_tokens["Thinking"]) if category_tokens["Thinking"] else 0,
                    "discussion_tokens_avg": np.mean(category_tokens["Discussion"]) if category_tokens["Discussion"] else 0,
                    "proposal_tokens_avg": np.mean(category_tokens["Proposal"]) if category_tokens["Proposal"] else 0,
                    "voting_tokens_avg": np.mean(category_tokens["Voting"]) if category_tokens["Voting"] else 0,
                    "reflection_tokens_avg": np.mean(category_tokens["Reflection"]) if category_tokens["Reflection"] else 0,
                }

                # Add per-round token totals
                for round_num in range(0, 11):
                    round_total = sum(round_tokens.get(round_num, []))
                    row[f"round_{round_num}_tokens"] = round_total

                all_data.append(row)

    print(f"Total: {total_experiment_dirs} experiment directories across {len(base_dirs)} batches")
    return pd.DataFrame(all_data)


def plot1_budget_vs_total_tokens(df: pd.DataFrame, output_dir: Path):
    """
    Plot 1: Prompted budget vs non-reasoning model output tokens
    Shows whether the reasoning depth prompt affects verbosity of non-reasoning model.
    """
    df_nonreasoning = df[df["is_nonreasoning"] == True].copy()

    if df_nonreasoning.empty:
        print("Warning: No non-reasoning model data found for Plot 1")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Aggregate by token budget (averaging across model_order)
    budget_groups = df_nonreasoning.groupby("token_budget_prompted").agg({
        "avg_tokens_per_interaction": ["mean", "std"],
        "total_tokens_sum": ["mean", "std"],
        "utility": "mean"
    }).reset_index()
    budget_groups.columns = ["budget", "avg_tokens_mean", "avg_tokens_std",
                             "total_tokens_mean", "total_tokens_std", "utility_mean"]

    # Plot 1a: Avg tokens per interaction vs budget
    ax1 = axes[0]
    ax1.errorbar(
        budget_groups["budget"],
        budget_groups["avg_tokens_mean"],
        yerr=budget_groups["avg_tokens_std"],
        fmt='o-',
        markersize=12,
        linewidth=2,
        capsize=5,
        color='steelblue'
    )
    ax1.set_xscale('log')
    ax1.set_xlabel("Prompted Reasoning Budget (to wrong model)")
    ax1.set_ylabel("Avg Tokens per Interaction (Non-Reasoning Model)")
    ax1.set_title("Effect of Misplaced Reasoning Prompt\non Non-Reasoning Model Output Length")

    # Label points
    for _, row in budget_groups.iterrows():
        ax1.annotate(
            f'{int(row["budget"]):,}',
            (row["budget"], row["avg_tokens_mean"]),
            textcoords="offset points",
            xytext=(10, 10),
            fontsize=14
        )

    # Plot 1b: Total tokens sum vs budget
    ax2 = axes[1]
    ax2.errorbar(
        budget_groups["budget"],
        budget_groups["total_tokens_mean"],
        yerr=budget_groups["total_tokens_std"],
        fmt='s-',
        markersize=12,
        linewidth=2,
        capsize=5,
        color='coral'
    )
    ax2.set_xscale('log')
    ax2.set_xlabel("Prompted Reasoning Budget (to wrong model)")
    ax2.set_ylabel("Total Tokens (Sum across all interactions)")
    ax2.set_title("Total Token Output by\nPrompted Budget Level")

    plt.tight_layout()
    output_path = output_dir / "plot1_budget_vs_total_tokens.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot2_tokens_by_phase(df: pd.DataFrame, output_dir: Path):
    """
    Plot 2: Token usage by negotiation phase for non-reasoning model
    """
    df_nonreasoning = df[df["is_nonreasoning"] == True].copy()

    if df_nonreasoning.empty:
        print("Warning: No non-reasoning model data found for Plot 2")
        return

    phase_cols = ["setup_tokens_avg", "discussion_tokens_avg",
                  "proposal_tokens_avg", "voting_tokens_avg"]
    phase_labels = ["Setup", "Discussion", "Proposal", "Voting"]

    # Aggregate by budget
    agg_data = df_nonreasoning.groupby("token_budget_prompted")[phase_cols].mean().reset_index()
    agg_data = agg_data.sort_values("token_budget_prompted")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Plot 2a: Stacked bar chart of tokens by phase
    ax1 = axes[0]
    x = range(len(agg_data))
    bottom = np.zeros(len(agg_data))

    colors = plt.cm.Set2(np.linspace(0, 1, len(phase_cols)))

    for i, (col, label) in enumerate(zip(phase_cols, phase_labels)):
        values = agg_data[col].values
        ax1.bar(x, values, bottom=bottom, label=label, color=colors[i], alpha=0.85, width=0.7)
        bottom += values

    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{int(b):,}" for b in agg_data["token_budget_prompted"]], rotation=45, ha='right')
    ax1.set_xlabel("Prompted Reasoning Budget")
    ax1.set_ylabel("Avg Tokens per Phase")
    ax1.set_title("Non-Reasoning Model Token Usage by Phase")
    ax1.legend(title="Phase", loc="upper left")

    # Plot 2b: Line plot showing each phase separately
    ax2 = axes[1]
    for i, (col, label) in enumerate(zip(phase_cols, phase_labels)):
        ax2.plot(
            agg_data["token_budget_prompted"],
            agg_data[col],
            'o-',
            markersize=10,
            linewidth=2,
            label=label,
            color=colors[i]
        )

    ax2.set_xscale('log')
    ax2.set_xlabel("Prompted Reasoning Budget (log scale)")
    ax2.set_ylabel("Avg Tokens per Phase")
    ax2.set_title("Phase-wise Token Usage vs Prompted Budget")
    ax2.legend(title="Phase")

    plt.tight_layout()
    output_path = output_dir / "plot2_tokens_by_phase.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot3_tokens_by_round(df: pd.DataFrame, output_dir: Path):
    """
    Plot 3: Token usage by round for non-reasoning model
    """
    df_nonreasoning = df[df["is_nonreasoning"] == True].copy()

    if df_nonreasoning.empty:
        print("Warning: No non-reasoning model data found for Plot 3")
        return

    round_cols = [f"round_{i}_tokens" for i in range(0, 6)]  # Rounds 0-5
    round_labels = [f"R{i}" for i in range(0, 6)]

    # Aggregate by budget
    agg_data = df_nonreasoning.groupby("token_budget_prompted")[round_cols].mean().reset_index()
    agg_data = agg_data.sort_values("token_budget_prompted")

    fig, ax = plt.subplots(figsize=(12, 7))

    # Define budget levels and consistent color mapping
    budgets = sorted(agg_data["token_budget_prompted"].unique())
    colors = plt.cm.viridis(np.linspace(0, 1, len(budgets)))

    x = np.arange(len(round_cols))
    width = 0.8 / len(budgets)

    for i, budget in enumerate(budgets):
        budget_data = agg_data[agg_data["token_budget_prompted"] == budget]
        if not budget_data.empty:
            values = budget_data[round_cols].values.flatten()
            offset = (i - len(budgets)/2 + 0.5) * width
            ax.bar(x + offset, values, width, label=f"{int(budget):,}", color=colors[i], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(round_labels)
    ax.set_xlabel("Round")
    ax.set_ylabel("Avg Total Tokens")
    ax.set_title("Non-Reasoning Model Token Usage by Round\n(Grouped by Prompted Budget)")
    ax.legend(title="Prompted Budget", loc="upper right", ncol=2)

    plt.tight_layout()
    output_path = output_dir / "plot3_tokens_by_round.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot4_model_comparison(df: pd.DataFrame, output_dir: Path):
    """
    Plot 4: Side-by-side comparison of reasoning vs non-reasoning model tokens
    """
    df_reasoning = df[df["is_nonreasoning"] == False].copy()
    df_nonreasoning = df[df["is_nonreasoning"] == True].copy()

    if df_reasoning.empty or df_nonreasoning.empty:
        print("Warning: Missing data for model comparison in Plot 4")
        return

    # Aggregate both by budget
    reasoning_agg = df_reasoning.groupby("token_budget_prompted").agg({
        "avg_tokens_per_interaction": "mean"
    }).reset_index()
    reasoning_agg.columns = ["budget", "avg_tokens_reasoning"]

    nonreasoning_agg = df_nonreasoning.groupby("token_budget_prompted").agg({
        "avg_tokens_per_interaction": "mean"
    }).reset_index()
    nonreasoning_agg.columns = ["budget", "avg_tokens_nonreasoning"]

    # Merge
    merged = pd.merge(reasoning_agg, nonreasoning_agg, on="budget", how="outer")
    merged = merged.sort_values("budget")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Plot 4a: Both models on same axes
    ax1 = axes[0]
    ax1.plot(
        merged["budget"],
        merged["avg_tokens_reasoning"],
        'o-',
        markersize=12,
        linewidth=2,
        label="Reasoning Model (Claude Opus)",
        color='steelblue'
    )
    ax1.plot(
        merged["budget"],
        merged["avg_tokens_nonreasoning"],
        's-',
        markersize=12,
        linewidth=2,
        label="Non-Reasoning Model (GPT-5 Nano)",
        color='coral'
    )

    ax1.set_xscale('log')
    ax1.set_xlabel("Prompted Reasoning Budget")
    ax1.set_ylabel("Avg Tokens per Interaction")
    ax1.set_title("Token Usage Comparison:\nReasoning vs Non-Reasoning Model")
    ax1.legend()

    # Plot 4b: Ratio of non-reasoning to reasoning tokens
    ax2 = axes[1]
    merged["ratio"] = merged["avg_tokens_nonreasoning"] / merged["avg_tokens_reasoning"]

    ax2.plot(
        merged["budget"],
        merged["ratio"],
        'o-',
        markersize=12,
        linewidth=2,
        color='purple'
    )
    ax2.axhline(y=1.0, color='gray', linestyle=':', linewidth=2, label='Equal tokens')

    ax2.set_xscale('log')
    ax2.set_xlabel("Prompted Reasoning Budget")
    ax2.set_ylabel("Ratio (Non-Reasoning / Reasoning)")
    ax2.set_title("Token Ratio: Non-Reasoning / Reasoning")
    ax2.legend()

    # Label points
    for _, row in merged.iterrows():
        if not np.isnan(row["ratio"]):
            ax2.annotate(
                f'{row["ratio"]:.2f}',
                (row["budget"], row["ratio"]),
                textcoords="offset points",
                xytext=(10, 5),
                fontsize=12
            )

    plt.tight_layout()
    output_path = output_dir / "plot4_model_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot5_budget_effect_distribution(df: pd.DataFrame, output_dir: Path):
    """
    Plot 5: Distribution of token counts per budget level for non-reasoning model
    """
    df_nonreasoning = df[df["is_nonreasoning"] == True].copy()

    if df_nonreasoning.empty:
        print("Warning: No non-reasoning model data found for Plot 5")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Plot 5a: Box plot of avg tokens per interaction by budget
    ax1 = axes[0]
    budgets = sorted(df_nonreasoning["token_budget_prompted"].unique())
    box_data = [df_nonreasoning[df_nonreasoning["token_budget_prompted"] == b]["avg_tokens_per_interaction"].values
                for b in budgets]

    bp = ax1.boxplot(box_data, labels=[f"{int(b):,}" for b in budgets], patch_artist=True)

    colors = plt.cm.viridis(np.linspace(0, 1, len(budgets)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax1.set_xlabel("Prompted Reasoning Budget")
    ax1.set_ylabel("Avg Tokens per Interaction")
    ax1.set_title("Distribution of Token Usage\nby Prompted Budget")
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Plot 5b: Violin plot
    ax2 = axes[1]
    df_plot = df_nonreasoning[["token_budget_prompted", "avg_tokens_per_interaction"]].copy()
    df_plot["budget_str"] = df_plot["token_budget_prompted"].apply(lambda x: f"{int(x):,}")

    # Order categories properly
    budget_order = [f"{int(b):,}" for b in budgets]

    sns.violinplot(
        data=df_plot,
        x="budget_str",
        y="avg_tokens_per_interaction",
        order=budget_order,
        ax=ax2,
        palette="viridis"
    )

    ax2.set_xlabel("Prompted Reasoning Budget")
    ax2.set_ylabel("Avg Tokens per Interaction")
    ax2.set_title("Token Usage Distribution (Violin Plot)")
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    output_path = output_dir / "plot5_budget_effect_distribution.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot6_payoff_vs_tokens(df: pd.DataFrame, output_dir: Path):
    """
    Plot 6: Avg payoff utility vs avg tokens per stage
    X-axis: Avg tokens per interaction (total_tokens for non-reasoning model)
    Y-axis: Avg payoff (utility)
    Different colors for reasoning vs non-reasoning model
    Filtered to competition_level = 1.0 only
    """
    # Filter for competition level = 1.0 only
    df_filtered = df[df["competition_level"] == 1.0].copy()

    if df_filtered.empty:
        print("Warning: No data found for competition_level=1.0 in Plot 6")
        return

    df_reasoning = df_filtered[df_filtered["is_nonreasoning"] == False].copy()
    df_nonreasoning = df_filtered[df_filtered["is_nonreasoning"] == True].copy()

    if df_reasoning.empty and df_nonreasoning.empty:
        print("Warning: No data found for Plot 6")
        return

    # Aggregate by budget for both model types
    reasoning_agg = df_reasoning.groupby("token_budget_prompted").agg({
        "avg_tokens_per_interaction": "mean",
        "utility": "mean"
    }).reset_index()
    reasoning_agg.columns = ["budget", "avg_tokens", "utility"]

    nonreasoning_agg = df_nonreasoning.groupby("token_budget_prompted").agg({
        "avg_tokens_per_interaction": "mean",
        "utility": "mean"
    }).reset_index()
    nonreasoning_agg.columns = ["budget", "avg_tokens", "utility"]

    fig, ax = plt.subplots(figsize=(14, 8))

    # Collect all utility values for y-axis scaling
    all_utilities = []

    # Plot reasoning model (blue circles, no connecting lines)
    if not reasoning_agg.empty:
        ax.scatter(
            reasoning_agg["avg_tokens"],
            reasoning_agg["utility"],
            s=180,
            marker='o',
            label="Reasoning Model (Claude Opus)",
            color='steelblue',
            edgecolors='white',
            linewidth=1.5,
            zorder=3
        )
        all_utilities.extend(reasoning_agg["utility"].tolist())

        # Add trendline for reasoning model
        if len(reasoning_agg) >= 2:
            z = np.polyfit(reasoning_agg["avg_tokens"].values, reasoning_agg["utility"].values, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(reasoning_agg["avg_tokens"].min(),
                                   reasoning_agg["avg_tokens"].max(), 100)
            ax.plot(x_trend, p(x_trend), '--', color='steelblue',
                    linewidth=2, alpha=0.7, zorder=2)

        # Label points with budget
        for _, row in reasoning_agg.iterrows():
            ax.annotate(
                f'{int(row["budget"]):,}',
                (row["avg_tokens"], row["utility"]),
                textcoords="offset points",
                xytext=(8, 8),
                fontsize=11,
                color='steelblue'
            )

    # Plot non-reasoning model (coral squares, no connecting lines)
    if not nonreasoning_agg.empty:
        ax.scatter(
            nonreasoning_agg["avg_tokens"],
            nonreasoning_agg["utility"],
            s=180,
            marker='s',
            label="Non-Reasoning Model (GPT-5 Nano)",
            color='coral',
            edgecolors='white',
            linewidth=1.5,
            zorder=3
        )
        all_utilities.extend(nonreasoning_agg["utility"].tolist())

        # Add trendline for non-reasoning model
        if len(nonreasoning_agg) >= 2:
            z = np.polyfit(nonreasoning_agg["avg_tokens"].values, nonreasoning_agg["utility"].values, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(nonreasoning_agg["avg_tokens"].min(),
                                   nonreasoning_agg["avg_tokens"].max(), 100)
            ax.plot(x_trend, p(x_trend), '--', color='coral',
                    linewidth=2, alpha=0.7, zorder=2)

        # Label points with budget
        for _, row in nonreasoning_agg.iterrows():
            ax.annotate(
                f'{int(row["budget"]):,}',
                (row["avg_tokens"], row["utility"]),
                textcoords="offset points",
                xytext=(8, -12),
                fontsize=11,
                color='coral'
            )

    # Add horizontal line at 50 (fair split)
    ax.axhline(y=50, color='gray', linestyle=':', linewidth=2, alpha=0.7, label='Fair split (50)')

    # Adjust y-axis for better legibility
    if all_utilities:
        y_min, y_max = min(all_utilities), max(all_utilities)
        y_range = y_max - y_min
        # Add padding and ensure we include 50 (fair split line)
        y_lower = min(y_min - y_range * 0.15, 48)
        y_upper = max(y_max + y_range * 0.15, 52)
        ax.set_ylim(y_lower, y_upper)

    ax.set_xlabel("Avg Tokens per Interaction")
    ax.set_ylabel("Avg Payoff (Utility)")
    ax.set_title("Payoff vs Token Usage (Competition Level = 1.0):\nReasoning vs Non-Reasoning Model")

    # Legend outside the plot on the right
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=12)

    plt.tight_layout()
    output_path = output_dir / "plot6_payoff_vs_tokens.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot7_payoff_vs_budget(df: pd.DataFrame, output_dir: Path):
    """
    Plot 7: Avg payoff utility vs prompted token budget
    X-axis: Prompted token budget (log scale)
    Y-axis: Avg payoff (utility)
    Different colors for reasoning vs non-reasoning model
    Filtered to competition_level = 1.0 only
    """
    # Filter for competition level = 1.0 only
    df_filtered = df[df["competition_level"] == 1.0].copy()

    if df_filtered.empty:
        print("Warning: No data found for competition_level=1.0 in Plot 7")
        return

    df_reasoning = df_filtered[df_filtered["is_nonreasoning"] == False].copy()
    df_nonreasoning = df_filtered[df_filtered["is_nonreasoning"] == True].copy()

    if df_reasoning.empty and df_nonreasoning.empty:
        print("Warning: No data found for Plot 7")
        return

    # Aggregate by budget for both model types
    reasoning_agg = df_reasoning.groupby("token_budget_prompted").agg({
        "utility": "mean"
    }).reset_index()
    reasoning_agg.columns = ["budget", "utility"]

    nonreasoning_agg = df_nonreasoning.groupby("token_budget_prompted").agg({
        "utility": "mean"
    }).reset_index()
    nonreasoning_agg.columns = ["budget", "utility"]

    fig, ax = plt.subplots(figsize=(14, 8))

    # Collect all utility values for y-axis scaling
    all_utilities = []

    # Plot reasoning model (blue circles, no connecting lines)
    if not reasoning_agg.empty:
        ax.scatter(
            reasoning_agg["budget"],
            reasoning_agg["utility"],
            s=180,
            marker='o',
            label="Reasoning Model (Claude Opus)",
            color='steelblue',
            edgecolors='white',
            linewidth=1.5,
            zorder=3
        )
        all_utilities.extend(reasoning_agg["utility"].tolist())

        # Add trendline for reasoning model (fit on log scale)
        if len(reasoning_agg) >= 2:
            log_budget = np.log10(reasoning_agg["budget"].values)
            z = np.polyfit(log_budget, reasoning_agg["utility"].values, 1)
            p = np.poly1d(z)
            x_trend = np.logspace(np.log10(reasoning_agg["budget"].min()),
                                   np.log10(reasoning_agg["budget"].max()), 100)
            ax.plot(x_trend, p(np.log10(x_trend)), '--', color='steelblue',
                    linewidth=2, alpha=0.7, zorder=2)

        # Label points with budget
        for _, row in reasoning_agg.iterrows():
            ax.annotate(
                f'{int(row["budget"]):,}',
                (row["budget"], row["utility"]),
                textcoords="offset points",
                xytext=(8, 8),
                fontsize=11,
                color='steelblue'
            )

    # Plot non-reasoning model (coral squares, no connecting lines)
    if not nonreasoning_agg.empty:
        ax.scatter(
            nonreasoning_agg["budget"],
            nonreasoning_agg["utility"],
            s=180,
            marker='s',
            label="Non-Reasoning Model (GPT-5 Nano)",
            color='coral',
            edgecolors='white',
            linewidth=1.5,
            zorder=3
        )
        all_utilities.extend(nonreasoning_agg["utility"].tolist())

        # Add trendline for non-reasoning model (fit on log scale)
        if len(nonreasoning_agg) >= 2:
            log_budget = np.log10(nonreasoning_agg["budget"].values)
            z = np.polyfit(log_budget, nonreasoning_agg["utility"].values, 1)
            p = np.poly1d(z)
            x_trend = np.logspace(np.log10(nonreasoning_agg["budget"].min()),
                                   np.log10(nonreasoning_agg["budget"].max()), 100)
            ax.plot(x_trend, p(np.log10(x_trend)), '--', color='coral',
                    linewidth=2, alpha=0.7, zorder=2)

        # Label points with budget
        for _, row in nonreasoning_agg.iterrows():
            ax.annotate(
                f'{int(row["budget"]):,}',
                (row["budget"], row["utility"]),
                textcoords="offset points",
                xytext=(8, -12),
                fontsize=11,
                color='coral'
            )

    # Add horizontal line at 50 (fair split)
    ax.axhline(y=50, color='gray', linestyle=':', linewidth=2, alpha=0.7, label='Fair split (50)')

    # Log scale for x-axis
    ax.set_xscale('log')

    # Adjust y-axis for better legibility
    if all_utilities:
        y_min, y_max = min(all_utilities), max(all_utilities)
        y_range = y_max - y_min
        # Add padding and ensure we include 50 (fair split line)
        y_lower = min(y_min - y_range * 0.15, 48)
        y_upper = max(y_max + y_range * 0.15, 52)
        ax.set_ylim(y_lower, y_upper)

    ax.set_xlabel("Prompted Token Budget")
    ax.set_ylabel("Avg Payoff (Utility)")
    ax.set_title("Payoff vs Prompted Budget (Competition Level = 1.0):\nReasoning vs Non-Reasoning Model")

    # Legend outside the plot on the right
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=12)

    plt.tight_layout()
    output_path = output_dir / "plot7_payoff_vs_budget.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize non-reasoning model token usage in TTC scaling experiments")
    parser.add_argument("--single", type=str, default=None,
                       help="Analyze only a single directory instead of aggregating all")
    parser.add_argument("--results-base", type=str, default="experiments/results",
                       help="Base directory containing ttc_scaling_* folders (default: experiments/results)")
    parser.add_argument("--output-dir", "-o", type=str, default=None,
                       help="Output directory for figures (default: visualization/figures/nonreasoning)")
    args = parser.parse_args()

    results_base = Path(args.results_base)

    if args.single:
        base_dir = Path(args.single)
        if not base_dir.exists():
            print(f"Error: Results directory not found: {base_dir}")
            return 1
        base_dirs = [base_dir]
        output_dir = Path(args.output_dir) if args.output_dir else base_dir / "figures" / "nonreasoning"
        print(f"Analyzing single directory: {base_dir}")
    else:
        if not results_base.exists():
            print(f"Error: Results base directory not found: {results_base}")
            return 1

        base_dirs = find_all_ttc_scaling_dirs(results_base)

        if not base_dirs:
            print(f"Error: No ttc_scaling_* directories found in {results_base}")
            return 1

        output_dir = Path(args.output_dir) if args.output_dir else Path("visualization/figures/nonreasoning")
        print(f"Aggregating results from {len(base_dirs)} ttc_scaling directories:")
        for d in base_dirs:
            print(f"  - {d.name}")

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    print()

    # Collect all data
    print("Collecting experiment data...")
    df = collect_all_data(base_dirs)

    if df.empty:
        print("No data found!")
        return 1

    print(f"\nCollected {len(df)} data points from {df['experiment_dir'].nunique()} experiments")
    print(f"  Non-reasoning model entries: {len(df[df['is_nonreasoning'] == True])}")
    print(f"  Reasoning model entries: {len(df[df['is_nonreasoning'] == False])}")
    print(f"Source batches: {sorted(df['source_batch'].unique())}")
    print(f"Token budgets: {sorted(df['token_budget_prompted'].unique())}")
    print(f"Model orders: {df['model_order'].unique().tolist()}")
    print()

    # Save data summary
    summary_path = output_dir / "nonreasoning_data_summary.csv"
    df.to_csv(summary_path, index=False)
    print(f"Saved data summary: {summary_path}")
    print()

    # Generate plots
    print("Generating plots...")
    plot1_budget_vs_total_tokens(df, output_dir)
    plot2_tokens_by_phase(df, output_dir)
    plot3_tokens_by_round(df, output_dir)
    plot4_model_comparison(df, output_dir)
    plot5_budget_effect_distribution(df, output_dir)
    plot6_payoff_vs_tokens(df, output_dir)
    plot7_payoff_vs_budget(df, output_dir)

    print("\nDone!")
    return 0


if __name__ == "__main__":
    exit(main())
