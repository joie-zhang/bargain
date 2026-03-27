#!/usr/bin/env python3
"""
=============================================================================
N-Agent Scaling Experiment Analysis and Visualization
=============================================================================

Produces publication-quality figures for N > 2 agent negotiation experiments:

  1. Scaling slope plot: Focal advantage vs N, one line per model
  2. Coalition power curve: Strong agent utility share vs K weak opponents
  3. Reward distribution: Utility share vs Elo rank with fitted curves
  4. Rating comparison: Arena Elo vs negotiation skill scatter
  5. Gini heatmap: Utility concentration vs (N, competition)

Usage:
    python visualization/visualize_nagent.py
    python visualization/visualize_nagent.py --results-dir experiments/results/nagent_20260306_123456
    python visualization/visualize_nagent.py --game-type co_funding

What it creates:
    visualization/figures/nagent/
    ├── fig1_scaling_slope.png          # Focal advantage vs N
    ├── fig2_coalition_power.png        # Strong utility share vs K
    ├── fig3_reward_distribution.png    # Utility share vs Elo rank
    ├── fig4_rating_comparison.png      # Arena Elo vs negotiation skill
    ├── fig5_gini_heatmap.png           # Gini vs (N, competition)
    └── summary_table.csv              # Curated metrics table

Dependencies:
    numpy, pandas, matplotlib, seaborn, scipy
    game_environments.multiagent_metrics
    analysis.multiagent_rating
=============================================================================
"""

import argparse
import json
import re
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from game_environments.multiagent_metrics import (
    gini_coefficient,
    rank_stability,
    utility_advantage,
    utility_share,
)

# =============================================================================
# Style & Constants
# =============================================================================

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.dpi": 150,
})

# Model info (Elo ratings from Chatbot Arena leaderboard, Jan 2026)
MODEL_INFO = {
    "gemini-3-pro": {"tier": "Strong", "elo": 1490},
    "claude-opus-4-6": {"tier": "Strong", "elo": 1475},
    "claude-sonnet-4-5": {"tier": "Strong", "elo": 1450},
    "gpt-5.2-high": {"tier": "Strong", "elo": 1436},
    "gpt-5.2-chat-latest-20260210": {"tier": "Strong", "elo": 1436},
    "grok-4": {"tier": "Strong", "elo": 1409},
    "claude-haiku-4-5": {"tier": "Medium", "elo": 1403},
    "deepseek-r1": {"tier": "Medium", "elo": 1397},
    "o3-mini-high": {"tier": "Medium", "elo": 1364},
    "qwen3-32b": {"tier": "Medium", "elo": 1360},
    "gpt-4o": {"tier": "Medium", "elo": 1346},
    "gpt-5-nano": {"tier": "Weak", "elo": 1338},
    "gpt-3.5-turbo": {"tier": "Weak", "elo": 1225},
    "llama-3.1-8b-instruct": {"tier": "Weak", "elo": 1180},
}

TIER_COLORS = {"Strong": "#e74c3c", "Medium": "#f39c12", "Weak": "#27ae60"}

# Per-model colors for line plots (colorblind-friendly)
MODEL_COLORS = {
    "claude-haiku-4-5": "#0072B2",
    "qwen3-32b": "#D55E00",
    "llama-3.1-8b-instruct": "#009E73",
    "gpt-5-nano": "#CC79A7",
}

GAME_LABELS = {
    "item_allocation": "Item Allocation",
    "diplomacy": "Diplomacy",
    "co_funding": "Co-Funding",
}


def _short_name(model: str) -> str:
    """Shorten a model name for display."""
    name = re.sub(r"-\d{8,}$", "", model)
    name = name.replace("-chat-latest", "")
    name = name.replace("-instruct", "")
    return name


def _get_elo(model: str) -> float:
    """Get Elo rating for a model, defaulting to 1300 if unknown."""
    info = MODEL_INFO.get(model)
    if info:
        return info["elo"]
    return 1300.0


# =============================================================================
# Data Loading
# =============================================================================


def load_results(results_dir: Path) -> List[Dict]:
    """Load all experiment results from an N-agent experiment directory."""
    all_results = []
    seen = set()

    for result_file in sorted(results_dir.rglob("*experiment_results.json")):
        try:
            with open(result_file) as f:
                data = json.load(f)
            # Deduplicate
            key = (str(result_file.parent), data.get("experiment_id", ""))
            if key in seen:
                continue
            seen.add(key)
            data["_source_path"] = str(result_file)
            all_results.append(data)
        except (json.JSONDecodeError, KeyError):
            continue

    return all_results


def load_from_csv(results_dir: Path) -> pd.DataFrame:
    """Load results from aggregated CSV if available."""
    csv_file = results_dir / "results.csv"
    if csv_file.exists():
        return pd.read_csv(csv_file)
    return pd.DataFrame()


def results_to_dataframe(results: List[Dict], configs_dir: Optional[Path] = None) -> pd.DataFrame:
    """Convert raw results + config data into a flat DataFrame."""
    rows = []

    # Load configs if available
    configs = {}
    if configs_dir and configs_dir.exists():
        for cf in configs_dir.glob("config_*.json"):
            with open(cf) as f:
                cfg = json.load(f)
            configs[cfg["experiment_id"]] = cfg

    for result in results:
        exp_id = result.get("experiment_id", -1)
        cfg = configs.get(exp_id, {})

        utilities = result.get("final_utilities", {})
        models = cfg.get("models", list(utilities.keys()))
        n_agents = len(models)

        row = {
            "experiment_id": exp_id,
            "experiment_type": cfg.get("experiment_type", ""),
            "experiment_design": cfg.get("experiment_design", ""),
            "game_type": cfg.get("game_type", ""),
            "n_agents": n_agents,
            "models": "+".join(models),
            "consensus_reached": result.get("consensus_reached", False),
            "final_round": result.get("final_round", -1),
        }

        # Game-specific params
        row["competition_level"] = cfg.get("competition_level", np.nan)
        row["rho"] = cfg.get("rho", np.nan)
        row["theta"] = cfg.get("theta", np.nan)
        row["alpha"] = cfg.get("alpha", np.nan)
        row["sigma"] = cfg.get("sigma", np.nan)

        # Per-agent utilities
        for agent_id, util in utilities.items():
            row[f"utility_{agent_id}"] = util

        # Compute multi-agent metrics
        if utilities:
            shares = utility_share(utilities)
            row["gini"] = gini_coefficient(utilities)

            elo_map = {a: _get_elo(m) for a, m in zip(utilities.keys(), models)}
            tau, pval = rank_stability(utilities, elo_map)
            row["rank_tau"] = tau
            row["rank_pval"] = pval

            # Focal agent = first model (typically the adversary)
            focal_agent = list(utilities.keys())[0]
            row["focal_model"] = models[0] if models else ""
            row["focal_elo"] = _get_elo(models[0]) if models else np.nan
            row["focal_utility"] = utilities.get(focal_agent, np.nan)
            row["focal_share"] = shares.get(focal_agent, np.nan)
            row["focal_advantage"] = utility_advantage(utilities, focal_agent)

            # Per-agent shares and Elos
            for i, (agent_id, share) in enumerate(shares.items()):
                row[f"share_{agent_id}"] = share
                if i < len(models):
                    row[f"elo_agent_{i}"] = _get_elo(models[i])

        rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
# Figure 1: Scaling Slope Plot
# =============================================================================


def fig1_scaling_slope(df: pd.DataFrame, output_dir: Path, game_filter: Optional[str] = None):
    """Focal advantage vs N, one line per focal model."""
    data = df[df["experiment_type"] == "exp_a_scaling"].copy()
    if game_filter:
        data = data[data["game_type"] == game_filter]

    if data.empty:
        print("  Skipping fig1: no Experiment A data")
        return

    game_types = sorted(data["game_type"].unique())
    n_games = len(game_types)
    fig, axes = plt.subplots(1, n_games, figsize=(5 * n_games, 4.5), squeeze=False)

    for col, game_type in enumerate(game_types):
        ax = axes[0, col]
        gdata = data[data["game_type"] == game_type]

        for focal_model in sorted(gdata["focal_model"].unique()):
            mdata = gdata[gdata["focal_model"] == focal_model]
            grouped = mdata.groupby("n_agents")["focal_advantage"].agg(["mean", "std"]).reset_index()

            color = MODEL_COLORS.get(focal_model, "#333333")
            ax.plot(
                grouped["n_agents"], grouped["mean"],
                marker="o", color=color, linewidth=2,
                label=_short_name(focal_model),
            )
            if "std" in grouped.columns and not grouped["std"].isna().all():
                ax.fill_between(
                    grouped["n_agents"],
                    grouped["mean"] - grouped["std"],
                    grouped["mean"] + grouped["std"],
                    alpha=0.15, color=color,
                )

        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Number of Agents (N)")
        ax.set_ylabel("Focal Advantage")
        ax.set_title(GAME_LABELS.get(game_type, game_type))
        ax.set_xticks(sorted(data["n_agents"].unique()))
        ax.legend(fontsize=9)

    fig.suptitle("Scaling Robustness: Focal Agent Advantage vs N", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "fig1_scaling_slope.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig1_scaling_slope.png")


# =============================================================================
# Figure 2: Coalition Power Curve
# =============================================================================


def fig2_coalition_power(df: pd.DataFrame, output_dir: Path, game_filter: Optional[str] = None):
    """Strong agent utility share vs K weak opponents."""
    data = df[df["experiment_type"] == "exp_b_coalition"].copy()
    if game_filter:
        data = data[data["game_type"] == game_filter]

    if data.empty:
        print("  Skipping fig2: no Experiment B data")
        return

    game_types = sorted(data["game_type"].unique())
    n_games = len(game_types)
    fig, axes = plt.subplots(1, n_games, figsize=(5 * n_games, 4.5), squeeze=False)

    for col, game_type in enumerate(game_types):
        ax = axes[0, col]
        gdata = data[data["game_type"] == game_type]

        # K = n_agents - 1 (number of weak opponents)
        gdata = gdata.copy()
        gdata["K"] = gdata["n_agents"] - 1

        grouped = gdata.groupby("K")["focal_share"].agg(["mean", "std"]).reset_index()

        ax.plot(
            grouped["K"], grouped["mean"],
            marker="s", color="#0072B2", linewidth=2.5,
            label="Strong agent share",
        )
        if not grouped["std"].isna().all():
            ax.fill_between(
                grouped["K"],
                grouped["mean"] - grouped["std"],
                grouped["mean"] + grouped["std"],
                alpha=0.15, color="#0072B2",
            )

        # Equal share reference line: 1/N = 1/(K+1)
        k_vals = np.array(grouped["K"])
        equal_share = 1.0 / (k_vals + 1)
        ax.plot(k_vals, equal_share, "--", color="gray", alpha=0.6, label="Equal share (1/N)")

        ax.set_xlabel("Number of Weak Opponents (K)")
        ax.set_ylabel("Strong Agent Utility Share")
        ax.set_title(GAME_LABELS.get(game_type, game_type))
        ax.set_ylim(0, 1)
        ax.legend(fontsize=9)

    fig.suptitle("Coalition Power: Strong Agent Share vs Weak Coalition Size", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "fig2_coalition_power.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig2_coalition_power.png")


# =============================================================================
# Figure 3: Reward Distribution (Utility Share vs Elo)
# =============================================================================


def fig3_reward_distribution(df: pd.DataFrame, output_dir: Path, game_filter: Optional[str] = None):
    """Utility share vs Elo rank within mixed-capability groups."""
    data = df[df["experiment_type"] == "exp_c_gradient"].copy()
    if game_filter:
        data = data[data["game_type"] == game_filter]

    if data.empty:
        print("  Skipping fig3: no Experiment C data")
        return

    # Collect (elo, share) pairs for each agent in each game
    elo_share_pairs = []
    for _, row in data.iterrows():
        models_str = row.get("models", "")
        if not models_str:
            continue
        models = models_str.split("+")
        for i, model in enumerate(models):
            agent_id = f"Agent_{chr(65 + i)}"  # Agent_A, Agent_B, ...
            share_col = f"share_{agent_id}"
            if share_col in row and not pd.isna(row[share_col]):
                elo_share_pairs.append({
                    "model": model,
                    "elo": _get_elo(model),
                    "share": row[share_col],
                    "game_type": row["game_type"],
                    "n_agents": row["n_agents"],
                })

    if not elo_share_pairs:
        print("  Skipping fig3: no share data found")
        return

    pair_df = pd.DataFrame(elo_share_pairs)

    game_types = sorted(pair_df["game_type"].unique())
    n_games = len(game_types)
    fig, axes = plt.subplots(1, n_games, figsize=(5 * n_games, 4.5), squeeze=False)

    for col, game_type in enumerate(game_types):
        ax = axes[0, col]
        gdata = pair_df[pair_df["game_type"] == game_type]

        # Scatter with color by model
        for model in sorted(gdata["model"].unique()):
            mdata = gdata[gdata["model"] == model]
            color = MODEL_COLORS.get(model, "#333333")
            ax.scatter(
                mdata["elo"], mdata["share"],
                color=color, s=60, alpha=0.7,
                label=_short_name(model), edgecolors="white", linewidth=0.5,
            )

        # Linear fit
        if len(gdata) >= 3:
            from numpy.polynomial import polynomial as P
            coeffs = P.polyfit(gdata["elo"], gdata["share"], 1)
            elo_range = np.linspace(gdata["elo"].min() - 20, gdata["elo"].max() + 20, 100)
            ax.plot(elo_range, P.polyval(elo_range, coeffs), "--", color="gray", alpha=0.5, label="Linear fit")

        # Equal share reference
        n_vals = gdata["n_agents"].unique()
        for n in n_vals:
            ax.axhline(y=1.0 / n, color="gray", linestyle=":", alpha=0.3)

        ax.set_xlabel("Arena Elo")
        ax.set_ylabel("Utility Share")
        ax.set_title(GAME_LABELS.get(game_type, game_type))
        ax.legend(fontsize=8)

    fig.suptitle("Reward Distribution: Utility Share vs Model Capability", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "fig3_reward_distribution.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig3_reward_distribution.png")


# =============================================================================
# Figure 4: Rating Comparison (Arena Elo vs Negotiation Skill)
# =============================================================================


def fig4_rating_comparison(df: pd.DataFrame, output_dir: Path):
    """Arena Elo vs negotiation skill scatter (from Plackett-Luce fit)."""
    # Collect utility rankings across all games
    games = []
    for _, row in df.iterrows():
        utils = {}
        models_str = row.get("models", "")
        if not models_str:
            continue
        models = models_str.split("+")
        for i, model in enumerate(models):
            agent_id = f"Agent_{chr(65 + i)}"
            util_col = f"utility_{agent_id}"
            if util_col in row and not pd.isna(row[util_col]):
                utils[model] = row[util_col]
        if len(utils) >= 2:
            games.append({"utilities": utils})

    if len(games) < 3:
        print("  Skipping fig4: insufficient data for Plackett-Luce fit")
        return

    try:
        from analysis.multiagent_rating import fit_plackett_luce, plackett_luce_to_ratings
        theta = fit_plackett_luce(games)
        neg_ratings = plackett_luce_to_ratings(theta)
    except Exception as e:
        print(f"  Skipping fig4: Plackett-Luce fitting error: {e}")
        return

    fig, ax = plt.subplots(figsize=(6, 5))

    models = sorted(neg_ratings.keys())
    arena_elos = [_get_elo(m) for m in models]
    neg_skills = [neg_ratings[m] for m in models]

    for i, model in enumerate(models):
        color = MODEL_COLORS.get(model, "#333333")
        tier = MODEL_INFO.get(model, {}).get("tier", "Unknown")
        ax.scatter(
            arena_elos[i], neg_skills[i],
            color=TIER_COLORS.get(tier, "#333"),
            s=100, zorder=5, edgecolors="black", linewidth=0.8,
        )
        ax.annotate(
            _short_name(model), (arena_elos[i], neg_skills[i]),
            textcoords="offset points", xytext=(8, 4), fontsize=9,
        )

    # Diagonal reference (perfect correlation)
    all_vals = arena_elos + neg_skills
    lo, hi = min(all_vals) - 30, max(all_vals) + 30
    ax.plot([lo, hi], [lo, hi], "--", color="gray", alpha=0.4, label="y = x")

    ax.set_xlabel("Arena Elo")
    ax.set_ylabel("Negotiation Skill Rating")
    ax.set_title("Arena Elo vs Negotiation Skill")
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_dir / "fig4_rating_comparison.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig4_rating_comparison.png")


# =============================================================================
# Figure 5: Gini Heatmap
# =============================================================================


def fig5_gini_heatmap(df: pd.DataFrame, output_dir: Path, game_filter: Optional[str] = None):
    """Gini coefficient heatmap across N and competition level."""
    data = df.dropna(subset=["gini"]).copy()
    if game_filter:
        data = data[data["game_type"] == game_filter]

    if data.empty:
        print("  Skipping fig5: no Gini data")
        return

    game_types = sorted(data["game_type"].unique())
    n_games = len(game_types)
    fig, axes = plt.subplots(1, n_games, figsize=(5 * n_games, 4), squeeze=False)

    for col, game_type in enumerate(game_types):
        ax = axes[0, col]
        gdata = data[data["game_type"] == game_type]

        # Choose competition dimension based on game type
        if game_type == "item_allocation":
            comp_col = "competition_level"
            comp_label = "Competition Level"
        elif game_type == "diplomacy":
            comp_col = "rho"
            comp_label = "Rho"
        else:  # co_funding
            comp_col = "alpha"
            comp_label = "Alpha"

        if comp_col not in gdata.columns or gdata[comp_col].isna().all():
            ax.text(0.5, 0.5, "No competition data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(GAME_LABELS.get(game_type, game_type))
            continue

        # Pivot for heatmap
        pivot = gdata.groupby(["n_agents", comp_col])["gini"].mean().reset_index()
        pivot_table = pivot.pivot(index=comp_col, columns="n_agents", values="gini")

        sns.heatmap(
            pivot_table, annot=True, fmt=".2f", cmap="YlOrRd",
            ax=ax, vmin=0, vmax=1,
            cbar_kws={"label": "Gini"},
        )
        ax.set_xlabel("N (Agents)")
        ax.set_ylabel(comp_label)
        ax.set_title(GAME_LABELS.get(game_type, game_type))

    fig.suptitle("Utility Concentration (Gini) by N and Competition", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "fig5_gini_heatmap.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig5_gini_heatmap.png")


# =============================================================================
# Summary Table
# =============================================================================


def create_summary_table(df: pd.DataFrame, output_dir: Path):
    """Create a summary CSV with key metrics."""
    if df.empty:
        print("  Skipping summary table: no data")
        return

    summary_rows = []
    for (exp_type, game_type, n_agents), group in df.groupby(
        ["experiment_type", "game_type", "n_agents"]
    ):
        row = {
            "experiment_type": exp_type,
            "game_type": game_type,
            "n_agents": n_agents,
            "n_experiments": len(group),
            "consensus_rate": group["consensus_reached"].mean() if "consensus_reached" in group else np.nan,
            "mean_gini": group["gini"].mean() if "gini" in group else np.nan,
            "mean_rank_tau": group["rank_tau"].mean() if "rank_tau" in group else np.nan,
            "mean_focal_advantage": group["focal_advantage"].mean() if "focal_advantage" in group else np.nan,
            "mean_focal_share": group["focal_share"].mean() if "focal_share" in group else np.nan,
        }
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "summary_table.csv", index=False)
    print("  Saved summary_table.csv")

    # Print summary
    print("\n  Summary:")
    print(summary_df.to_string(index=False))


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Visualize N-agent scaling experiments")
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Path to experiment results directory (default: nagent_latest)",
    )
    parser.add_argument(
        "--game-type",
        type=str,
        default=None,
        choices=["item_allocation", "diplomacy", "co_funding"],
        help="Filter to a single game type",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for figures (default: visualization/figures/nagent/)",
    )
    args = parser.parse_args()

    # Resolve paths
    project_root = Path(__file__).parent.parent

    if args.results_dir:
        results_dir = Path(args.results_dir)
        if not results_dir.is_absolute():
            results_dir = project_root / results_dir
    else:
        results_dir = project_root / "experiments" / "results" / "nagent_latest"

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = project_root / "visualization" / "figures" / "nagent"

    output_dir.mkdir(parents=True, exist_ok=True)

    if not results_dir.exists():
        print(f"ERROR: Results directory not found: {results_dir}")
        sys.exit(1)

    print(f"Loading results from: {results_dir}")

    # Try CSV first, fall back to raw JSON
    df = load_from_csv(results_dir)
    if df.empty:
        results = load_results(results_dir)
        if not results:
            print("ERROR: No results found")
            sys.exit(1)
        configs_dir = results_dir / "configs"
        df = results_to_dataframe(results, configs_dir)

    print(f"Loaded {len(df)} experiments")
    if "game_type" in df.columns:
        print(f"  Game types: {sorted(df['game_type'].unique())}")
    if "n_agents" in df.columns:
        print(f"  Agent counts: {sorted(df['n_agents'].unique())}")
    if "experiment_type" in df.columns:
        print(f"  Experiment types: {sorted(df['experiment_type'].unique())}")
    print()

    game_filter = args.game_type

    # Generate figures
    print("Generating figures...")
    fig1_scaling_slope(df, output_dir, game_filter)
    fig2_coalition_power(df, output_dir, game_filter)
    fig3_reward_distribution(df, output_dir, game_filter)
    fig4_rating_comparison(df, output_dir)
    fig5_gini_heatmap(df, output_dir, game_filter)
    create_summary_table(df, output_dir)

    print(f"\nAll figures saved to: {output_dir}")


if __name__ == "__main__":
    main()
