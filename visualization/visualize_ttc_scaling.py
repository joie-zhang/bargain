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

What it creates:
    experiments/results/ttc_scaling_combined/figures/
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

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


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


def collect_all_data(base_dirs: List[Path]) -> pd.DataFrame:
    """Collect data from all experiments across multiple base directories into a DataFrame."""
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

            # Get config info
            config = results.get("config", {})
            reasoning_config = config.get("reasoning_config", {})

            # Get final utilities
            final_utilities = results.get("final_utilities", {})

            # Process each agent
            for agent_id, utility in final_utilities.items():
                agent_data = agent_interactions.get(agent_id, {})

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

                # Sum all phases containing "thinking"
                thinking_total = sum(sum(v) for k, v in phase_tokens.items() if "thinking" in k.lower())
                reflection_total = sum(sum(v) for k, v in phase_tokens.items() if "reflection" in k.lower())
                discussion_total = sum(sum(v) for k, v in phase_tokens.items() if "discussion" in k.lower())
                proposal_total = sum(sum(v) for k, v in phase_tokens.items() if "proposal" in k.lower())
                voting_total = sum(sum(v) for k, v in phase_tokens.items() if "voting" in k.lower())

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
    Plot 1: Average reasoning tokens vs payoff
    X-axis: Average reasoning tokens used across all phases
    Y-axis: Final payoff (utility)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Filter for reasoning model only (Alpha in weak_first, Beta in strong_first)
    # In weak_first: Alpha=Claude(reasoning), Beta=GPT(baseline)
    # In strong_first: Alpha=GPT(baseline), Beta=Claude(reasoning)
    df_reasoning = df[
        ((df["model_order"] == "weak_first") & (df["agent_id"] == "Agent_Alpha")) |
        ((df["model_order"] == "strong_first") & (df["agent_id"] == "Agent_Beta"))
    ].copy()

    if df_reasoning.empty:
        print("Warning: No reasoning model data found for Plot 1")
        return

    # Plot 1a: Scatter with regression line
    ax1 = axes[0]
    sns.regplot(
        data=df_reasoning,
        x="total_reasoning_tokens",
        y="utility",
        ax=ax1,
        scatter_kws={"alpha": 0.6, "s": 100},
        line_kws={"color": "red", "linewidth": 2}
    )
    ax1.set_xlabel("Total Reasoning Tokens", fontsize=12)
    ax1.set_ylabel("Payoff (Utility)", fontsize=12)
    ax1.set_title("Reasoning Tokens vs Payoff (Reasoning Model)", fontsize=14)

    # Add correlation coefficient
    corr = df_reasoning["total_reasoning_tokens"].corr(df_reasoning["utility"])
    ax1.text(0.05, 0.95, f"r = {corr:.3f}", transform=ax1.transAxes,
             fontsize=12, verticalalignment='top')

    # Plot 1b: Grouped by token budget
    ax2 = axes[1]
    budget_groups = df_reasoning.groupby("token_budget_prompted").agg({
        "total_reasoning_tokens": ["mean", "std"],
        "utility": ["mean", "std", "count"]
    }).reset_index()
    budget_groups.columns = ["budget", "tokens_mean", "tokens_std",
                            "utility_mean", "utility_std", "count"]

    ax2.errorbar(
        budget_groups["tokens_mean"],
        budget_groups["utility_mean"],
        xerr=budget_groups["tokens_std"],
        yerr=budget_groups["utility_std"],
        fmt='o-',
        capsize=5,
        markersize=10,
        linewidth=2
    )

    # Label points with budget
    for _, row in budget_groups.iterrows():
        ax2.annotate(
            f'{int(row["budget"])}',
            (row["tokens_mean"], row["utility_mean"]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=9
        )

    ax2.set_xlabel("Avg Reasoning Tokens (by budget group)", fontsize=12)
    ax2.set_ylabel("Avg Payoff (Utility)", fontsize=12)
    ax2.set_title("Reasoning vs Payoff by Prompted Budget", fontsize=14)

    plt.tight_layout()
    output_path = output_dir / "plot1_avg_reasoning_vs_payoff.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot2_per_round_reasoning(df: pd.DataFrame, output_dir: Path):
    """
    Plot 2: Per-round reasoning tokens analysis
    Subplots for each round showing reasoning tokens vs payoff
    """
    # Filter for reasoning model
    df_reasoning = df[
        ((df["model_order"] == "weak_first") & (df["agent_id"] == "Agent_Alpha")) |
        ((df["model_order"] == "strong_first") & (df["agent_id"] == "Agent_Beta"))
    ].copy()

    if df_reasoning.empty:
        print("Warning: No reasoning model data found for Plot 2")
        return

    # Create subplot grid for rounds 1-10
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()

    for round_num in range(1, 11):
        ax = axes[round_num - 1]
        col_name = f"round_{round_num}_tokens"

        if col_name in df_reasoning.columns:
            # Filter out zeros for better visualization
            round_data = df_reasoning[df_reasoning[col_name] > 0]

            if not round_data.empty:
                sns.scatterplot(
                    data=round_data,
                    x=col_name,
                    y="utility",
                    hue="token_budget_prompted",
                    palette="viridis",
                    ax=ax,
                    alpha=0.7,
                    s=80
                )
                ax.set_xlabel("Reasoning Tokens", fontsize=10)
                ax.set_ylabel("Payoff", fontsize=10)
                ax.set_title(f"Round {round_num}", fontsize=12, fontweight='bold')
                ax.legend(title="Budget", fontsize=8, title_fontsize=9)
            else:
                ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f"Round {round_num}", fontsize=12)
        else:
            ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"Round {round_num}", fontsize=12)

    plt.suptitle("Reasoning Tokens vs Payoff by Round", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    output_path = output_dir / "plot2_per_round_reasoning.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot3_phase_breakdown(df: pd.DataFrame, output_dir: Path):
    """
    Plot 3: Stacked bar chart showing reasoning tokens by phase
    """
    # Filter for reasoning model
    df_reasoning = df[
        ((df["model_order"] == "weak_first") & (df["agent_id"] == "Agent_Alpha")) |
        ((df["model_order"] == "strong_first") & (df["agent_id"] == "Agent_Beta"))
    ].copy()

    if df_reasoning.empty:
        print("Warning: No reasoning model data found for Plot 3")
        return

    # Aggregate by token budget
    phase_cols = ["thinking_tokens", "reflection_tokens", "discussion_tokens",
                  "proposal_tokens", "voting_tokens"]

    agg_data = df_reasoning.groupby("token_budget_prompted")[phase_cols + ["utility"]].mean().reset_index()
    agg_data = agg_data.sort_values("token_budget_prompted")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 3a: Stacked bar chart of tokens by phase
    ax1 = axes[0]
    x = range(len(agg_data))
    bottom = np.zeros(len(agg_data))

    colors = plt.cm.Set2(np.linspace(0, 1, len(phase_cols)))
    phase_labels = ["Thinking", "Reflection", "Discussion", "Proposal", "Voting"]

    for i, (col, label) in enumerate(zip(phase_cols, phase_labels)):
        values = agg_data[col].values
        ax1.bar(x, values, bottom=bottom, label=label, color=colors[i], alpha=0.8)
        bottom += values

    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{int(b)}" for b in agg_data["token_budget_prompted"]], rotation=45)
    ax1.set_xlabel("Prompted Token Budget", fontsize=12)
    ax1.set_ylabel("Avg Reasoning Tokens", fontsize=12)
    ax1.set_title("Reasoning Tokens by Phase", fontsize=14)
    ax1.legend(title="Phase", loc="upper left")

    # Plot 3b: Payoff vs total tokens with phase breakdown
    ax2 = axes[1]
    total_tokens = agg_data[phase_cols].sum(axis=1)

    ax2.scatter(total_tokens, agg_data["utility"], s=150, c=agg_data["token_budget_prompted"],
                cmap="viridis", alpha=0.8, edgecolors='black', linewidth=1)

    # Add trend line (only if we have enough data points)
    if len(total_tokens) >= 2 and total_tokens.std() > 0:
        try:
            z = np.polyfit(total_tokens, agg_data["utility"], 1)
            p = np.poly1d(z)
            x_line = np.linspace(total_tokens.min(), total_tokens.max(), 100)
            ax2.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2, label="Trend")
        except np.linalg.LinAlgError:
            pass  # Skip trend line if fitting fails

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap="viridis",
                               norm=plt.Normalize(vmin=agg_data["token_budget_prompted"].min(),
                                                  vmax=agg_data["token_budget_prompted"].max()))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax2)
    cbar.set_label("Prompted Budget", fontsize=10)

    ax2.set_xlabel("Total Reasoning Tokens", fontsize=12)
    ax2.set_ylabel("Avg Payoff (Utility)", fontsize=12)
    ax2.set_title("Payoff vs Total Reasoning (by Budget)", fontsize=14)
    ax2.legend()

    plt.tight_layout()
    output_path = output_dir / "plot3_phase_breakdown.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot4_instructed_vs_actual(df: pd.DataFrame, output_dir: Path):
    """
    Plot 4: Instructed (prompted) reasoning tokens vs Actual reasoning tokens used
    X-axis: Token budget that was prompted/instructed
    Y-axis: Actual reasoning tokens used
    """
    # Filter for reasoning model
    df_reasoning = df[
        ((df["model_order"] == "weak_first") & (df["agent_id"] == "Agent_Alpha")) |
        ((df["model_order"] == "strong_first") & (df["agent_id"] == "Agent_Beta"))
    ].copy()

    if df_reasoning.empty:
        print("Warning: No reasoning model data found for Plot 4")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 4a: Scatter plot with identity line
    ax1 = axes[0]

    # Get unique budgets for coloring
    budgets = sorted(df_reasoning["token_budget_prompted"].unique())

    sns.scatterplot(
        data=df_reasoning,
        x="token_budget_prompted",
        y="total_reasoning_tokens",
        hue="model_order",
        style="model_order",
        s=100,
        alpha=0.7,
        ax=ax1
    )

    # Add identity line (y = x) for reference
    max_val = max(df_reasoning["token_budget_prompted"].max(),
                  df_reasoning["total_reasoning_tokens"].max())
    ax1.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label="y = x (perfect compliance)")

    ax1.set_xlabel("Instructed Token Budget", fontsize=12)
    ax1.set_ylabel("Actual Reasoning Tokens Used", fontsize=12)
    ax1.set_title("Instructed vs Actual Reasoning Tokens", fontsize=14)
    ax1.legend(title="Model Order")

    # Plot 4b: Aggregated view with error bars
    ax2 = axes[1]

    agg_data = df_reasoning.groupby("token_budget_prompted").agg({
        "total_reasoning_tokens": ["mean", "std", "count"]
    }).reset_index()
    agg_data.columns = ["budget", "actual_mean", "actual_std", "count"]
    agg_data["actual_sem"] = agg_data["actual_std"] / np.sqrt(agg_data["count"])

    ax2.errorbar(
        agg_data["budget"],
        agg_data["actual_mean"],
        yerr=agg_data["actual_sem"] * 1.96,  # 95% CI
        fmt='o-',
        capsize=5,
        markersize=10,
        linewidth=2,
        color='steelblue',
        label='Actual tokens'
    )

    # Add identity line
    max_budget = agg_data["budget"].max()
    ax2.plot([0, max_budget], [0, max_budget], 'k--', alpha=0.5, label="y = x")

    # Add ratio annotations
    for _, row in agg_data.iterrows():
        ratio = row["actual_mean"] / row["budget"] if row["budget"] > 0 else 0
        ax2.annotate(
            f'{ratio:.1f}x',
            (row["budget"], row["actual_mean"]),
            textcoords="offset points",
            xytext=(5, 10),
            fontsize=9,
            color='darkred'
        )

    ax2.set_xlabel("Instructed Token Budget", fontsize=12)
    ax2.set_ylabel("Actual Reasoning Tokens (mean +/- 95% CI)", fontsize=12)
    ax2.set_title("Compliance with Token Instructions", fontsize=14)
    ax2.legend()

    plt.tight_layout()
    output_path = output_dir / "plot4_instructed_vs_actual.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot5_instructed_vs_payoff(df: pd.DataFrame, output_dir: Path):
    """
    Plot 5: Instructed (prompted) reasoning tokens vs Model payoff
    X-axis: Token budget that was prompted/instructed
    Y-axis: Final payoff (utility)
    """
    # Filter for reasoning model
    df_reasoning = df[
        ((df["model_order"] == "weak_first") & (df["agent_id"] == "Agent_Alpha")) |
        ((df["model_order"] == "strong_first") & (df["agent_id"] == "Agent_Beta"))
    ].copy()

    if df_reasoning.empty:
        print("Warning: No reasoning model data found for Plot 5")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 5a: Scatter plot
    ax1 = axes[0]

    sns.scatterplot(
        data=df_reasoning,
        x="token_budget_prompted",
        y="utility",
        hue="model_order",
        style="model_order",
        s=100,
        alpha=0.7,
        ax=ax1
    )

    ax1.set_xlabel("Instructed Token Budget", fontsize=12)
    ax1.set_ylabel("Payoff (Utility)", fontsize=12)
    ax1.set_title("Instructed Reasoning Budget vs Payoff", fontsize=14)
    ax1.legend(title="Model Order")

    # Plot 5b: Aggregated view with error bars, split by order
    ax2 = axes[1]

    for order, color in [("weak_first", "steelblue"), ("strong_first", "darkorange")]:
        order_data = df_reasoning[df_reasoning["model_order"] == order]
        if order_data.empty:
            continue

        agg_data = order_data.groupby("token_budget_prompted").agg({
            "utility": ["mean", "std", "count"]
        }).reset_index()
        agg_data.columns = ["budget", "utility_mean", "utility_std", "count"]
        agg_data["utility_sem"] = agg_data["utility_std"] / np.sqrt(agg_data["count"])

        label = "Reasoning First" if order == "strong_first" else "Baseline First"
        ax2.errorbar(
            agg_data["budget"],
            agg_data["utility_mean"],
            yerr=agg_data["utility_sem"] * 1.96,  # 95% CI
            fmt='o-',
            capsize=5,
            markersize=10,
            linewidth=2,
            color=color,
            label=label
        )

    ax2.set_xlabel("Instructed Token Budget", fontsize=12)
    ax2.set_ylabel("Payoff (mean +/- 95% CI)", fontsize=12)
    ax2.set_title("Payoff vs Instructed Budget by Order", fontsize=14)
    ax2.legend()

    # Add horizontal line at 50 (fair split)
    ax2.axhline(y=50, color='gray', linestyle=':', alpha=0.7, label='Fair split (50)')

    plt.tight_layout()
    output_path = output_dir / "plot5_instructed_vs_payoff.png"
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
                       help="Output directory for figures (default: results_base/ttc_scaling_combined/figures)")
    args = parser.parse_args()

    results_base = Path(args.results_base)

    if args.single:
        # Single directory mode (legacy behavior)
        base_dir = Path(args.single)
        if not base_dir.exists():
            print(f"Error: Results directory not found: {base_dir}")
            return 1
        base_dirs = [base_dir]
        output_dir = Path(args.output_dir) if args.output_dir else base_dir / "figures"
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

        output_dir = Path(args.output_dir) if args.output_dir else results_base / "ttc_scaling_combined" / "figures"
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
    plot3_phase_breakdown(df, output_dir)
    plot4_instructed_vs_actual(df, output_dir)
    plot5_instructed_vs_payoff(df, output_dir)

    print("\nDone!")
    return 0


if __name__ == "__main__":
    exit(main())
