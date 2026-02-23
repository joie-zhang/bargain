#!/usr/bin/env python3
"""
=============================================================================
Co-Funding (Game 3) Experiment Analysis and Visualization
=============================================================================

Loads co-funding experiment results and computes game-theoretic metrics
(Lindahl distance, free-rider index, provision rate, etc.), then produces
publication-quality figures.

Usage:
    python visualization/visualize_cofunding.py
    python visualization/visualize_cofunding.py --results-dir experiments/results/cofunding_20260221_123456
    python visualization/visualize_cofunding.py --results-dir experiments/results/cofunding_latest

What it creates:
    visualization/figures/cofunding/
    ├── efficiency_heatmap.png            # Utilitarian efficiency vs alpha x sigma
    ├── provision_rate_heatmap.png        # Provision rate vs alpha x sigma
    ├── coordination_failure_heatmap.png  # Coordination failure vs alpha x sigma
    ├── free_rider_by_model.png           # Free-rider index by adversary model
    ├── lindahl_distance_by_model.png     # Distance from Lindahl equilibrium
    ├── exploitation_index.png            # Exploitation index by model
    ├── utility_vs_elo.png               # Agent utility vs adversary Elo
    ├── adaptation_rate_by_sigma.png      # Adaptation rate vs budget scarcity
    ├── num_funded_vs_sigma.png           # Projects funded vs sigma
    └── summary_table.csv                 # Full metrics table

Dependencies:
    - numpy, pandas, matplotlib, seaborn
    - game_environments.cofunding_metrics
"""

import argparse
import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from game_environments.cofunding_metrics import (
    coordination_failure_rate,
    exploitation_index_cofunding,
    free_rider_index,
    lindahl_distance,
    lindahl_equilibrium,
    optimal_funded_set,
    provision_rate,
    social_welfare_cofunding,
    utilitarian_efficiency,
)

# Style
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 11

# Model info (Elo ratings from Chatbot Arena leaderboard, Jan 2026)
MODEL_INFO = {
    "gemini-3-pro": {"tier": "Strong", "elo": 1490},
    "gemini-3-flash": {"tier": "Strong", "elo": 1472},
    "claude-opus-4-5-thinking-32k": {"tier": "Strong", "elo": 1470},
    "claude-opus-4-5": {"tier": "Strong", "elo": 1467},
    "claude-sonnet-4-5": {"tier": "Strong", "elo": 1450},
    "glm-4.7": {"tier": "Strong", "elo": 1441},
    "gpt-5.2-high": {"tier": "Strong", "elo": 1436},
    "qwen3-max": {"tier": "Strong", "elo": 1434},
    "deepseek-r1-0528": {"tier": "Strong", "elo": 1418},
    "grok-4": {"tier": "Strong", "elo": 1409},
    "claude-haiku-4-5": {"tier": "Medium", "elo": 1403},
    "deepseek-r1": {"tier": "Medium", "elo": 1397},
    "claude-sonnet-4": {"tier": "Medium", "elo": 1390},
    "claude-3.5-sonnet": {"tier": "Medium", "elo": 1373},
    "o3-mini-high": {"tier": "Medium", "elo": 1364},
    "deepseek-v3": {"tier": "Medium", "elo": 1358},
    "gpt-4o": {"tier": "Medium", "elo": 1346},
    "gpt-5-nano": {"tier": "Weak", "elo": 1338},
}

TIER_COLORS = {"Strong": "#e74c3c", "Medium": "#f39c12", "Weak": "#27ae60"}
BASELINE_MODEL = "gpt-5-nano"


# =============================================================================
# Data Loading
# =============================================================================


def load_results(results_dir: Path) -> List[Dict]:
    """Load all experiment results from a co-funding experiment directory."""
    all_results = []
    seen_experiment_ids = set()

    # Walk the directory tree looking for experiment_results.json files
    # Prefer run_N_experiment_results.json to avoid duplicates with experiment_results.json
    for result_file in sorted(results_dir.rglob("*experiment_results.json")):
        try:
            with open(result_file) as f:
                data = json.load(f)
            if not _is_valid_result(data):
                continue
            # Deduplicate: same experiment_id in same directory
            dedup_key = (str(result_file.parent), data.get("experiment_id", ""))
            if dedup_key in seen_experiment_ids:
                continue
            seen_experiment_ids.add(dedup_key)
            data["_file_path"] = str(result_file)
            data["_path_config"] = _parse_config_from_path(result_file)
            all_results.append(data)
        except (json.JSONDecodeError, FileNotFoundError):
            continue

    return all_results


def _is_valid_result(result: Dict) -> bool:
    """Check if an experiment result has all required fields."""
    required = ["final_utilities", "final_round", "config", "agent_preferences"]
    for field in required:
        if field not in result:
            return False
    utilities = result.get("final_utilities", {})
    if not utilities or len(utilities) < 2:
        return False
    for val in utilities.values():
        if val is None or not isinstance(val, (int, float)):
            return False
    return True


def _extract_final_pledges(result: Dict) -> Optional[Dict[str, List[float]]]:
    """Extract the last-round pledge contributions from conversation logs."""
    logs = result.get("conversation_logs", [])
    if not logs:
        return None

    # Find all pledge submissions, grouped by round
    pledges_by_round = defaultdict(dict)
    for log in logs:
        if log.get("phase") == "pledge_submission" and "pledge" in log:
            pledge = log["pledge"]
            if pledge and "contributions" in pledge:
                agent = log.get("from", pledge.get("proposed_by", ""))
                round_num = log.get("round", 0)
                pledges_by_round[round_num][agent] = pledge["contributions"]

    if not pledges_by_round:
        return None

    # Return pledges from the last round
    last_round = max(pledges_by_round.keys())
    final_pledges = pledges_by_round[last_round]
    return final_pledges if final_pledges else None


def _parse_config_from_path(path: Path) -> Dict:
    """Extract alpha, sigma, model_order from directory structure."""
    parts = path.parts
    config = {"model_order": None, "alpha": None, "sigma": None}
    for part in parts:
        if part in ("weak_first", "strong_first"):
            config["model_order"] = part
        elif part.startswith("alpha_"):
            # alpha_0_5_sigma_0_5 or alpha_0_5
            alpha_str = part.replace("alpha_", "").split("_sigma_")[0]
            config["alpha"] = float(alpha_str.replace("_", "."))
            if "_sigma_" in part:
                sigma_str = part.split("_sigma_")[1]
                config["sigma"] = float(sigma_str.replace("_", "."))
    # Also parse model pair
    for part in parts:
        if "_vs_" in part:
            models = part.split("_vs_")
            if len(models) == 2:
                config["model1"] = models[0]
                config["model2"] = models[1]
    return config


# =============================================================================
# Metric Computation
# =============================================================================


def compute_metrics_for_experiment(result: Dict) -> Optional[Dict]:
    """Compute all co-funding metrics for a single experiment result.

    Returns a flat dict of metric values, or None if data is insufficient.
    """
    config = result.get("config", {})
    utilities = result.get("final_utilities", {})
    preferences = result.get("agent_preferences", {})
    allocation = result.get("final_allocation", [])

    # Determine agent IDs (order matters for model mapping)
    agents = sorted(utilities.keys())
    if len(agents) < 2:
        return None

    # Extract game state info from config
    items = config.get("items", [])
    costs = [item.get("cost", 0.0) for item in items] if items else []
    m_projects = len(costs)
    if m_projects == 0:
        return None

    # Funded set
    if isinstance(allocation, list):
        funded_set = allocation
    else:
        funded_set = []

    # Determine alpha, sigma from config or path
    path_config = result.get("_path_config", {})
    alpha = config.get("alpha", path_config.get("alpha"))
    sigma = config.get("sigma", path_config.get("sigma"))

    # Get model info
    model_order = config.get("model_order", path_config.get("model_order", "weak_first"))
    agent_models = config.get("agents", agents)

    # Determine which agent is baseline vs adversary
    if model_order == "weak_first":
        baseline_agent = agents[0]
        adversary_agent = agents[1] if len(agents) > 1 else agents[0]
    else:
        baseline_agent = agents[1] if len(agents) > 1 else agents[0]
        adversary_agent = agents[0]

    # Get total budget from config
    total_budget = config.get("total_budget", 0.0)
    agent_budgets = config.get("agent_budgets", {})
    if not agent_budgets:
        # Estimate from sigma and costs
        total_cost = sum(costs)
        s = sigma if sigma else 0.5
        total_budget = s * total_cost
        per_agent = total_budget / len(agents)
        agent_budgets = {a: per_agent for a in agents}

    # Extract final-round contributions from conversation logs
    contributions = _extract_final_pledges(result)
    if contributions and not all(
        len(v) == m_projects for v in contributions.values()
    ):
        contributions = None

    metrics = {
        "alpha": alpha,
        "sigma": sigma,
        "m_projects": m_projects,
        "model_order": model_order,
        "num_funded": len(funded_set),
        "funded_fraction": len(funded_set) / m_projects if m_projects > 0 else 0,
        "final_round": result.get("final_round", 0),
        "consensus_reached": result.get("consensus_reached", False),
        "exploitation_detected": result.get("exploitation_detected", False),
        "baseline_utility": utilities.get(baseline_agent, 0),
        "adversary_utility": utilities.get(adversary_agent, 0),
        "total_utility": sum(utilities.values()),
        "utility_gap": utilities.get(adversary_agent, 0) - utilities.get(baseline_agent, 0),
    }

    # Optimal funded set
    if preferences and costs and total_budget > 0:
        optimal_set = optimal_funded_set(preferences, costs, total_budget)
        metrics["optimal_set_size"] = len(optimal_set)
        metrics["provision_rate"] = provision_rate(funded_set, optimal_set)

        # Compute optimal social welfare for efficiency
        # Optimal SW = sum_{j in S*} (sum_i v_ij - c_j)
        optimal_sw = 0.0
        for j in optimal_set:
            total_val = sum(preferences[a][j] for a in agents)
            optimal_sw += total_val - costs[j]
        metrics["optimal_sw"] = optimal_sw

        # Allocation-based actual SW: sum_{j in S} (sum_i v_ij - c_j)
        # This measures whether the RIGHT projects were funded, independent of overpayment
        actual_sw_allocation = 0.0
        for j in funded_set:
            total_val = sum(preferences[a][j] for a in agents)
            actual_sw_allocation += total_val - costs[j]
        metrics["actual_sw"] = actual_sw_allocation

        # Utility-based actual SW (includes overpayment penalty)
        utility_sw = sum(utilities.values())
        metrics["utility_sw"] = utility_sw

        # Overpayment = allocation_sw - utility_sw (excess contributions beyond project costs)
        metrics["overpayment"] = actual_sw_allocation - utility_sw if funded_set else 0.0

        # Utilitarian efficiency: allocation-based, clamped to [0, 1]
        metrics["utilitarian_efficiency"] = utilitarian_efficiency(actual_sw_allocation, optimal_sw)

        # Surplus ratio: unclamped, can be negative (shows how bad the allocation was)
        if optimal_sw > 1e-12:
            metrics["surplus_ratio"] = actual_sw_allocation / optimal_sw
        else:
            metrics["surplus_ratio"] = 1.0 if actual_sw_allocation >= 0 else 0.0

    # Lindahl equilibrium (always computable)
    if preferences and costs and funded_set:
        lindahl_contribs = lindahl_equilibrium(preferences, costs, funded_set)

        # Compute Lindahl utilities for exploitation index
        lindahl_utils = {}
        for a in agents:
            u = 0.0
            for j in funded_set:
                u += preferences[a][j] - lindahl_contribs[a][j]
            lindahl_utils[a] = u

        exploitation = exploitation_index_cofunding(utilities, lindahl_utils)
        metrics["exploitation_baseline"] = exploitation.get(baseline_agent, 0)
        metrics["exploitation_adversary"] = exploitation.get(adversary_agent, 0)

        # If we have actual contributions, compute more metrics
        if contributions:
            metrics["lindahl_distance"] = lindahl_distance(contributions, lindahl_contribs)

            # Coordination failure
            metrics["coordination_failure"] = coordination_failure_rate(
                preferences, costs, contributions
            )

            # Free-rider index (average across funded projects)
            # Cap inf at 10.0 (pure free-rider) rather than filtering out
            fri = free_rider_index(preferences, contributions, funded_set)
            for a in agents:
                vals = [min(v, 10.0) for v in fri[a].values()]
                agent_label = "baseline" if a == baseline_agent else "adversary"
                metrics[f"free_rider_avg_{agent_label}"] = np.mean(vals) if vals else float("nan")
                # Track pure free-riding (any inf values)
                pure_fr_count = sum(1 for v in fri[a].values() if v == float("inf"))
                metrics[f"pure_free_rider_{agent_label}"] = pure_fr_count > 0

    # Model info
    path_cfg = result.get("_path_config", {})
    m1 = path_cfg.get("model1", "")
    m2 = path_cfg.get("model2", "")
    if model_order == "weak_first":
        metrics["baseline_model"] = m1
        metrics["adversary_model"] = m2
    else:
        metrics["baseline_model"] = m2
        metrics["adversary_model"] = m1

    adversary = metrics.get("adversary_model", "")
    if adversary in MODEL_INFO:
        metrics["adversary_elo"] = MODEL_INFO[adversary]["elo"]
        metrics["adversary_tier"] = MODEL_INFO[adversary]["tier"]

    return metrics


# =============================================================================
# Plotting
# =============================================================================


def plot_heatmap(
    df: pd.DataFrame,
    metric: str,
    title: str,
    filename: str,
    figures_dir: Path,
    vmin: float = 0,
    vmax: float = 1,
    cmap: str = "YlOrRd",
    fmt: str = ".2f",
):
    """Plot a heatmap of a metric vs alpha (x) and sigma (y)."""
    if "alpha" not in df.columns or "sigma" not in df.columns:
        print(f"  Skipping {filename}: missing alpha/sigma columns")
        return

    pivot = df.groupby(["sigma", "alpha"])[metric].mean().unstack(fill_value=np.nan)

    if pivot.empty:
        print(f"  Skipping {filename}: no data")
        return

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        ax=ax,
        linewidths=0.5,
        cbar_kws={"label": metric.replace("_", " ").title()},
    )
    ax.set_xlabel("Alpha (Preference Alignment)", fontsize=12)
    ax.set_ylabel("Sigma (Budget Scarcity)", fontsize=12)
    ax.set_title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(figures_dir / filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {filename}")


def plot_metric_by_model(
    df: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    filename: str,
    figures_dir: Path,
):
    """Bar chart of a metric grouped by adversary model, colored by tier."""
    if "adversary_model" not in df.columns or metric not in df.columns:
        print(f"  Skipping {filename}: missing columns")
        return

    grouped = df.groupby("adversary_model")[metric].agg(["mean", "std", "count"]).reset_index()
    grouped = grouped.sort_values("mean", ascending=False)
    grouped = grouped[grouped["count"] >= 1]

    if grouped.empty:
        print(f"  Skipping {filename}: no data")
        return

    # Add Elo and tier
    grouped["elo"] = grouped["adversary_model"].map(
        lambda m: MODEL_INFO.get(m, {}).get("elo", 0)
    )
    grouped["tier"] = grouped["adversary_model"].map(
        lambda m: MODEL_INFO.get(m, {}).get("tier", "Unknown")
    )
    grouped = grouped.sort_values("elo", ascending=True)

    colors = [TIER_COLORS.get(t, "#95a5a6") for t in grouped["tier"]]

    fig, ax = plt.subplots(figsize=(14, 7))
    bars = ax.bar(
        range(len(grouped)),
        grouped["mean"],
        yerr=grouped["std"],
        color=colors,
        capsize=3,
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_xticks(range(len(grouped)))
    ax.set_xticklabels(grouped["adversary_model"], rotation=45, ha="right", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)

    # Legend for tiers
    from matplotlib.patches import Patch

    legend_elements = [Patch(facecolor=c, label=t) for t, c in TIER_COLORS.items()]
    ax.legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()
    plt.savefig(figures_dir / filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {filename}")


def plot_utility_vs_elo(df: pd.DataFrame, figures_dir: Path):
    """Scatter plot of agent utilities vs adversary Elo."""
    if "adversary_elo" not in df.columns:
        print("  Skipping utility_vs_elo.png: no Elo data")
        return

    valid = df.dropna(subset=["adversary_elo"])
    if valid.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 7))

    ax.scatter(
        valid["adversary_elo"],
        valid["baseline_utility"],
        alpha=0.5,
        label=f"Baseline ({BASELINE_MODEL})",
        marker="o",
        s=40,
    )
    ax.scatter(
        valid["adversary_elo"],
        valid["adversary_utility"],
        alpha=0.5,
        label="Adversary",
        marker="^",
        s=40,
    )

    # Trend lines
    for col, label in [("baseline_utility", "Baseline"), ("adversary_utility", "Adversary")]:
        mask = ~valid[col].isna()
        if mask.sum() >= 3:
            z = np.polyfit(valid.loc[mask, "adversary_elo"], valid.loc[mask, col], 1)
            p = np.poly1d(z)
            x_range = np.linspace(valid["adversary_elo"].min(), valid["adversary_elo"].max(), 100)
            ax.plot(x_range, p(x_range), "--", alpha=0.7)

    ax.set_xlabel("Adversary Model Elo Rating", fontsize=12)
    ax.set_ylabel("Final Utility", fontsize=12)
    ax.set_title("Agent Utility vs Adversary Elo (Co-Funding)", fontsize=14)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(figures_dir / "utility_vs_elo.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved utility_vs_elo.png")


def plot_num_funded_vs_sigma(df: pd.DataFrame, figures_dir: Path):
    """Line plot of average projects funded vs sigma (budget scarcity)."""
    if "sigma" not in df.columns or "num_funded" not in df.columns:
        print("  Skipping num_funded_vs_sigma.png: missing columns")
        return

    grouped = df.groupby("sigma")["num_funded"].agg(["mean", "std"]).reset_index()
    if grouped.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(
        grouped["sigma"],
        grouped["mean"],
        yerr=grouped["std"],
        marker="o",
        capsize=4,
        linewidth=2,
        markersize=8,
    )
    ax.set_xlabel("Sigma (Budget/Cost Ratio)", fontsize=12)
    ax.set_ylabel("Average Projects Funded", fontsize=12)
    ax.set_title("Projects Funded vs Budget Scarcity", fontsize=14)

    # Add reference line for m_projects
    m_projects = df["m_projects"].mode().iloc[0] if "m_projects" in df.columns else 5
    ax.axhline(y=m_projects, color="red", linestyle="--", alpha=0.5, label=f"Total projects ({m_projects})")
    ax.legend()

    plt.tight_layout()
    plt.savefig(figures_dir / "num_funded_vs_sigma.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved num_funded_vs_sigma.png")


def plot_exploitation_index(df: pd.DataFrame, figures_dir: Path):
    """Bar chart comparing exploitation index of baseline vs adversary by model."""
    if "adversary_model" not in df.columns:
        print("  Skipping exploitation_index.png: missing columns")
        return

    cols = ["exploitation_baseline", "exploitation_adversary"]
    available = [c for c in cols if c in df.columns]
    if not available:
        print("  Skipping exploitation_index.png: no exploitation data")
        return

    # Clip extreme exploitation values for visualization (near-zero Lindahl utility → huge ratios)
    df_clipped = df.copy()
    for col in available:
        df_clipped[col] = df_clipped[col].clip(-5, 5)
    grouped = df_clipped.groupby("adversary_model")[available].mean().reset_index()
    grouped["elo"] = grouped["adversary_model"].map(
        lambda m: MODEL_INFO.get(m, {}).get("elo", 0)
    )
    grouped = grouped.sort_values("elo", ascending=True)

    if grouped.empty:
        return

    x = np.arange(len(grouped))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 7))

    if "exploitation_baseline" in available:
        ax.bar(
            x - width / 2,
            grouped["exploitation_baseline"],
            width,
            label=f"Baseline ({BASELINE_MODEL})",
            color="#3498db",
            edgecolor="black",
            linewidth=0.5,
        )
    if "exploitation_adversary" in available:
        ax.bar(
            x + width / 2,
            grouped["exploitation_adversary"],
            width,
            label="Adversary",
            color="#e74c3c",
            edgecolor="black",
            linewidth=0.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(grouped["adversary_model"], rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Exploitation Index (vs Lindahl)", fontsize=12)
    ax.set_title("Exploitation Index by Model (Co-Funding)", fontsize=14)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "exploitation_index.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved exploitation_index.png")


def plot_adaptation_by_sigma(df: pd.DataFrame, figures_dir: Path):
    """Adaptation rate vs sigma."""
    # Check for adaptation columns
    adapt_cols = [c for c in df.columns if c.startswith("adaptation_")]
    if not adapt_cols or "sigma" not in df.columns:
        print("  Skipping adaptation_rate_by_sigma.png: no adaptation data")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    for col in adapt_cols:
        label = col.replace("adaptation_", "").replace("_", " ").title()
        grouped = df.groupby("sigma")[col].agg(["mean", "std"]).reset_index()
        ax.errorbar(grouped["sigma"], grouped["mean"], yerr=grouped["std"],
                     marker="o", capsize=4, label=label)

    ax.set_xlabel("Sigma (Budget/Cost Ratio)", fontsize=12)
    ax.set_ylabel("Adaptation Rate", fontsize=12)
    ax.set_title("Strategy Adaptation Rate vs Budget Scarcity", fontsize=14)
    ax.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "adaptation_rate_by_sigma.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved adaptation_rate_by_sigma.png")


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Analyze co-funding experiment results")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="experiments/results/cofunding_latest",
        help="Path to co-funding experiment results directory",
    )
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    results_dir = project_root / args.results_dir
    if not results_dir.exists():
        # Try as absolute path
        results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"ERROR: Results directory not found: {args.results_dir}")
        sys.exit(1)

    figures_dir = Path(__file__).parent / "figures" / "cofunding"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from: {results_dir}")
    experiments = load_results(results_dir)
    print(f"Found {len(experiments)} valid experiment results")

    if not experiments:
        print("No results to analyze. Run experiments first.")
        sys.exit(0)

    # Compute metrics for all experiments
    print("\nComputing metrics...")
    all_metrics = []
    for exp in experiments:
        metrics = compute_metrics_for_experiment(exp)
        if metrics is not None:
            all_metrics.append(metrics)

    print(f"Computed metrics for {len(all_metrics)} experiments")

    if not all_metrics:
        print("No computable metrics. Check result format.")
        sys.exit(0)

    df = pd.DataFrame(all_metrics)

    # Save full metrics table
    csv_path = figures_dir / "summary_table.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved metrics table: {csv_path}")

    # Print summary stats
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    print(f"Total experiments:    {len(df)}")
    if "num_funded" in df.columns:
        print(f"Avg projects funded:  {df['num_funded'].mean():.1f} / {df['m_projects'].mean():.0f}")
    if "utilitarian_efficiency" in df.columns:
        print(f"Avg efficiency:       {df['utilitarian_efficiency'].mean():.3f}")
    if "provision_rate" in df.columns:
        print(f"Avg provision rate:   {df['provision_rate'].mean():.3f}")
    if "coordination_failure" in df.columns:
        print(f"Avg coord. failure:   {df['coordination_failure'].mean():.3f}")
    print(f"Avg baseline utility: {df['baseline_utility'].mean():.1f}")
    print(f"Avg adversary utility:{df['adversary_utility'].mean():.1f}")
    print(f"Avg utility gap:      {df['utility_gap'].mean():.1f}")
    if "adversary_model" in df.columns:
        print(f"\nModels tested: {df['adversary_model'].nunique()}")
        print(f"Alpha values:  {sorted(df['alpha'].dropna().unique())}")
        print(f"Sigma values:  {sorted(df['sigma'].dropna().unique())}")

    # Generate plots
    print("\n" + "=" * 60)
    print("GENERATING FIGURES")
    print("=" * 60)

    # Heatmaps (alpha x sigma)
    if "utilitarian_efficiency" in df.columns:
        plot_heatmap(
            df, "utilitarian_efficiency",
            "Utilitarian Efficiency (Co-Funding)\n(Allocation-based: did agents fund the right projects?)",
            "efficiency_heatmap.png", figures_dir,
            vmin=0, vmax=1, cmap="YlGn",
        )

    if "surplus_ratio" in df.columns:
        sr_min = df["surplus_ratio"].min()
        sr_max = df["surplus_ratio"].max()
        plot_heatmap(
            df, "surplus_ratio",
            "Surplus Ratio (Actual / Optimal SW)\n(Negative = funded value-destroying projects)",
            "surplus_ratio_heatmap.png", figures_dir,
            vmin=min(sr_min, -0.5), vmax=max(sr_max, 1.0), cmap="RdYlGn", fmt=".2f",
        )

    if "provision_rate" in df.columns:
        plot_heatmap(
            df, "provision_rate",
            "Provision Rate (Fraction of Optimal Projects Funded)",
            "provision_rate_heatmap.png", figures_dir,
            vmin=0, vmax=1, cmap="YlGn",
        )

    if "coordination_failure" in df.columns:
        plot_heatmap(
            df, "coordination_failure",
            "Coordination Failure Rate",
            "coordination_failure_heatmap.png", figures_dir,
            vmin=0, vmax=1, cmap="YlOrRd",
        )

    plot_heatmap(
        df, "num_funded",
        "Average Number of Projects Funded",
        "num_funded_heatmap.png", figures_dir,
        vmin=0, vmax=df["m_projects"].max() if "m_projects" in df.columns else 5,
        cmap="YlGn", fmt=".1f",
    )

    # Model-level bar charts
    if "adversary_model" in df.columns and df["adversary_model"].nunique() > 1:
        plot_metric_by_model(
            df, "utility_gap",
            "Utility Gap (Adversary - Baseline) by Model",
            "Utility Gap", "utility_gap_by_model.png", figures_dir,
        )

        if "free_rider_avg_adversary" in df.columns:
            plot_metric_by_model(
                df, "free_rider_avg_adversary",
                "Free-Rider Index (Adversary) by Model",
                "Free-Rider Index (>1 = free-riding)",
                "free_rider_by_model.png", figures_dir,
            )

        if "lindahl_distance" in df.columns:
            plot_metric_by_model(
                df, "lindahl_distance",
                "Lindahl Distance by Adversary Model",
                "Frobenius Distance from Lindahl Equilibrium",
                "lindahl_distance_by_model.png", figures_dir,
            )

    # Scatter: utility vs Elo
    plot_utility_vs_elo(df, figures_dir)

    # Line: projects funded vs sigma
    plot_num_funded_vs_sigma(df, figures_dir)

    # Exploitation index
    plot_exploitation_index(df, figures_dir)

    # Adaptation rate
    plot_adaptation_by_sigma(df, figures_dir)

    print("\nDone! Figures saved to:", figures_dir)


if __name__ == "__main__":
    main()
