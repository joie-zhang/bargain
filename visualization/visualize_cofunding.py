#!/usr/bin/env python3
"""
=============================================================================
Co-Funding (Game 3) Experiment Analysis and Visualization
=============================================================================

Produces a focused set of publication-quality figures organized by narrative:

  1. Group efficiency: how well did agents collectively perform?
  2. Individual asymmetry: did stronger models exploit weaker ones?
  3. Cost-sharing fairness: free-riding and Lindahl deviation.
  4. Strategic dynamics: adaptation and signaling (conditional).
  5. Model scaling: metrics vs Elo (conditional on >=5 focal models).

Usage:
    python visualization/visualize_cofunding.py
    python visualization/visualize_cofunding.py --results-dir experiments/results/cofunding_20260304_031023

What it creates:
    visualization/figures/cofunding/
    ├── fig1_efficiency_landscape.png     # Group metrics vs (α, σ)
    ├── fig2_utility_asymmetry.png        # Focal advantage & exploitation
    ├── fig3_cost_sharing_fairness.png    # Free-rider index & Lindahl distance
    ├── fig4_strategic_dynamics.png       # Adaptation rate (if data exists)
    ├── fig5_model_scaling.png            # vs Elo (if ≥5 focal models)
    ├── fig6_competition_index.png        # Key metrics vs CI₃ = (1−α)·(1−σ)
    └── summary_table.csv                 # Curated metrics table

Dependencies:
    - numpy, pandas, matplotlib, seaborn
    - game_environments.cofunding_metrics
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
from matplotlib.patches import Patch

warnings.filterwarnings("ignore")

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from game_environments.cofunding_metrics import (
    adaptation_rate,
    coordination_failure_weighted,
    coordination_funding_gap_ratio,
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

# Colorblind-friendly palette (Okabe-Ito) for alpha values
ALPHA_COLORS = {
    0.0: "#E69F00",  # amber  — misaligned
    0.5: "#56B4E9",  # sky blue — partial
    1.0: "#009E73",  # teal   — fully aligned
}

# Focal vs reference role colors
FOCAL_COLOR = "#0072B2"
REFERENCE_COLOR = "#D55E00"

# Model info (Elo ratings from Chatbot Arena leaderboard, Jan 2026)
MODEL_INFO = {
    "gemini-3-pro": {"tier": "Strong", "elo": 1490},
    "gemini-3-flash": {"tier": "Strong", "elo": 1472},
    "claude-opus-4-6": {"tier": "Strong", "elo": 1475},
    "claude-opus-4-5-thinking-32k": {"tier": "Strong", "elo": 1470},
    "claude-opus-4-5": {"tier": "Strong", "elo": 1467},
    "claude-sonnet-4-5": {"tier": "Strong", "elo": 1450},
    "glm-4.7": {"tier": "Strong", "elo": 1441},
    "gpt-5.2-high": {"tier": "Strong", "elo": 1436},
    "gpt-5.2-chat-latest-20260210": {"tier": "Strong", "elo": 1436},
    "qwen3-max": {"tier": "Strong", "elo": 1434},
    "deepseek-r1-0528": {"tier": "Strong", "elo": 1418},
    "grok-4": {"tier": "Strong", "elo": 1409},
    "claude-haiku-4-5": {"tier": "Medium", "elo": 1403},
    "deepseek-r1": {"tier": "Medium", "elo": 1397},
    "claude-sonnet-4": {"tier": "Medium", "elo": 1390},
    "claude-3.5-sonnet": {"tier": "Medium", "elo": 1373},
    "o3-mini-high": {"tier": "Medium", "elo": 1364},
    "qwen3-32b": {"tier": "Medium", "elo": 1360},
    "deepseek-v3": {"tier": "Medium", "elo": 1358},
    "gpt-4o": {"tier": "Medium", "elo": 1346},
    "gpt-5-nano": {"tier": "Weak", "elo": 1338},
    "llama-3.1-8b-instruct": {"tier": "Weak", "elo": 1180},
}

TIER_COLORS = {"Strong": "#e74c3c", "Medium": "#f39c12", "Weak": "#27ae60"}

# The constant opponent model; focal models are measured against this.
REFERENCE_PATTERNS = ["gpt-5-nano", "gpt-5.2"]


def _is_reference_model(name: str) -> bool:
    s = str(name).lower()
    return any(p in s for p in REFERENCE_PATTERNS)


def _short_name(model: str) -> str:
    """Shorten a model name for display in legends and labels."""
    name = re.sub(r"-\d{8,}$", "", model)          # strip date suffix
    name = name.replace("-chat-latest", "")
    name = name.replace("-instruct", "")
    return name


# =============================================================================
# Data Loading
# =============================================================================


def load_results(results_dir: Path) -> List[Dict]:
    """Load all experiment results from a co-funding experiment directory."""
    all_results = []
    seen_experiment_ids = set()

    for result_file in sorted(results_dir.rglob("*experiment_results.json")):
        try:
            with open(result_file) as f:
                data = json.load(f)
            if not _is_valid_result(data):
                continue
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

    last_round = max(pledges_by_round.keys())
    final_pledges = pledges_by_round[last_round]
    return final_pledges if final_pledges else None


def _extract_pledge_history(result: Dict) -> List[Dict[str, List[float]]]:
    """Extract round-by-round pledge history from conversation logs."""
    logs = result.get("conversation_logs", [])
    if not logs:
        return []

    pledges_by_round: Dict[int, Dict[str, List[float]]] = defaultdict(dict)
    for log in logs:
        if log.get("phase") != "pledge_submission":
            continue
        pledge = log.get("pledge", {})
        contributions = pledge.get("contributions")
        if not isinstance(contributions, list):
            continue
        round_num = int(log.get("round", 0))
        if round_num <= 0:
            continue
        agent = log.get("from", pledge.get("proposed_by", ""))
        if not agent:
            continue
        pledges_by_round[round_num][agent] = [float(x) for x in contributions]

    if not pledges_by_round:
        return []
    return [pledges_by_round[r] for r in sorted(pledges_by_round.keys())]


def _parse_config_from_path(path: Path) -> Dict:
    """Extract alpha, sigma, model_order from directory structure."""
    parts = path.parts
    config = {"model_order": None, "alpha": None, "sigma": None}
    for part in parts:
        if part in ("weak_first", "strong_first"):
            config["model_order"] = part
        elif part.startswith("alpha_"):
            alpha_str = part.replace("alpha_", "").split("_sigma_")[0]
            config["alpha"] = float(alpha_str.replace("_", "."))
            if "_sigma_" in part:
                sigma_str = part.split("_sigma_")[1]
                config["sigma"] = float(sigma_str.replace("_", "."))
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

    agents = sorted(utilities.keys())
    if len(agents) < 2:
        return None

    items = config.get("items", [])
    costs = [item.get("cost", 0.0) for item in items] if items else []
    m_projects = len(costs)
    if m_projects == 0:
        return None

    if isinstance(allocation, list):
        funded_set = allocation
    else:
        funded_set = []

    path_config = result.get("_path_config", {})
    alpha = config.get("alpha", path_config.get("alpha"))
    sigma = config.get("sigma", path_config.get("sigma"))

    model_order = config.get("model_order", path_config.get("model_order", "weak_first"))
    agent_models = config.get("agents", agents)

    if model_order == "weak_first":
        baseline_agent = agents[0]
        adversary_agent = agents[1] if len(agents) > 1 else agents[0]
    else:
        baseline_agent = agents[1] if len(agents) > 1 else agents[0]
        adversary_agent = agents[0]

    total_budget = config.get("total_budget", 0.0)
    agent_budgets = config.get("agent_budgets", {})
    if not agent_budgets:
        total_cost = sum(costs)
        s = sigma if sigma else 0.5
        total_budget = (0.5 + 0.5 * s) * total_cost
        per_agent = total_budget / len(agents)
        agent_budgets = {a: per_agent for a in agents}

    contributions = _extract_final_pledges(result)
    if contributions and not all(
        len(v) == m_projects for v in contributions.values()
    ):
        contributions = None
    pledge_history = _extract_pledge_history(result)

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
    metrics["competition_index"] = (
        (1.0 - float(alpha)) * (1.0 - float(sigma))
        if alpha is not None and sigma is not None
        else float("nan")
    )

    if pledge_history and agent_budgets:
        adaptation = adaptation_rate(pledge_history, agent_budgets)
        metrics["adaptation_baseline"] = adaptation.get(baseline_agent, float("nan"))
        metrics["adaptation_adversary"] = adaptation.get(adversary_agent, float("nan"))

    qual = result.get("qualitative_metrics_v1", {})
    if isinstance(qual, dict) and qual:
        promise = qual.get("promise_keeping", {})
        persuasion = qual.get("persuasion_effectiveness", {})
        coalition = qual.get("coalition_formation", {})
        metrics["promise_keep_rate"] = promise.get("overall_keep_rate", float("nan"))
        metrics["promise_mean_abs_error"] = promise.get("mean_abs_error", float("nan"))
        metrics["persuasion_other_agent_delta"] = persuasion.get("overall_other_agent_delta", float("nan"))
        metrics["coalition_persistent_fraction"] = coalition.get("persistent_project_fraction", float("nan"))
        metrics["coalition_active_round_fraction"] = coalition.get("coalition_active_round_fraction", float("nan"))

    if preferences and costs and total_budget > 0:
        optimal_set = optimal_funded_set(preferences, costs, total_budget)
        metrics["optimal_set_size"] = len(optimal_set)
        metrics["provision_rate"] = provision_rate(funded_set, optimal_set)

        optimal_sw = 0.0
        for j in optimal_set:
            total_val = sum(preferences[a][j] for a in agents)
            optimal_sw += total_val - costs[j]
        metrics["optimal_sw"] = optimal_sw

        actual_sw_allocation = 0.0
        for j in funded_set:
            total_val = sum(preferences[a][j] for a in agents)
            actual_sw_allocation += total_val - costs[j]
        metrics["actual_sw"] = actual_sw_allocation

        utility_sw = sum(utilities.values())
        metrics["utility_sw"] = utility_sw
        metrics["overpayment"] = actual_sw_allocation - utility_sw if funded_set else 0.0
        metrics["utilitarian_efficiency"] = utilitarian_efficiency(actual_sw_allocation, optimal_sw)

        if optimal_sw > 1e-12:
            metrics["surplus_ratio"] = actual_sw_allocation / optimal_sw
        else:
            metrics["surplus_ratio"] = 1.0 if actual_sw_allocation >= 0 else 0.0

    if contributions and preferences and costs:
        metrics["coordination_failure_count"] = coordination_failure_rate(
            preferences, costs, contributions
        )
        metrics["coordination_failure_weighted"] = coordination_failure_weighted(
            preferences, costs, contributions
        )
        metrics["coordination_gap_ratio"] = coordination_funding_gap_ratio(
            preferences, costs, contributions
        )
        metrics["coordination_failure"] = metrics["coordination_failure_weighted"]

    if preferences and costs and funded_set:
        lindahl_contribs = lindahl_equilibrium(preferences, costs, funded_set)

        lindahl_utils = {}
        for a in agents:
            u = 0.0
            for j in funded_set:
                u += preferences[a][j] - lindahl_contribs[a][j]
            lindahl_utils[a] = u

        exploitation = exploitation_index_cofunding(utilities, lindahl_utils)
        metrics["exploitation_baseline"] = exploitation.get(baseline_agent, 0)
        metrics["exploitation_adversary"] = exploitation.get(adversary_agent, 0)

        if contributions:
            metrics["lindahl_distance"] = lindahl_distance(contributions, lindahl_contribs)

            fri = free_rider_index(preferences, contributions, funded_set)
            for a in agents:
                vals = [min(v, 10.0) for v in fri[a].values()]
                agent_label = "baseline" if a == baseline_agent else "adversary"
                metrics[f"free_rider_avg_{agent_label}"] = np.mean(vals) if vals else float("nan")
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
# Post-Processing: Focal vs Reference
# =============================================================================


def _add_focal_reference_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Canonicalize into focal (varying) vs reference (constant opponent) model.

    The 'focal' model is the one being studied (varies across experiments).
    The 'reference' model is the constant opponent (e.g. gpt-5.2).
    """
    focal_side_map = []  # "baseline" or "adversary"

    for _, row in df.iterrows():
        if _is_reference_model(row.get("baseline_model", "")):
            focal_side_map.append("adversary")
        else:
            focal_side_map.append("baseline")

    df["_focal_side"] = focal_side_map

    # Map per-role columns to focal/reference
    role_columns = {
        "utility": "{side}_utility",
        "exploitation": "exploitation_{side}",
        "free_rider": "free_rider_avg_{side}",
        "adaptation": "adaptation_{side}",
        "model": "{side}_model",
    }

    for target, pattern in role_columns.items():
        focal_vals = []
        ref_vals = []
        for _, row in df.iterrows():
            focal_side = row["_focal_side"]
            ref_side = "baseline" if focal_side == "adversary" else "adversary"
            focal_col = pattern.format(side=focal_side)
            ref_col = pattern.format(side=ref_side)
            focal_vals.append(row.get(focal_col, np.nan))
            ref_vals.append(row.get(ref_col, np.nan))
        df[f"focal_{target}"] = focal_vals
        df[f"reference_{target}"] = ref_vals

    # Derived columns
    df["focal_advantage"] = pd.to_numeric(df["focal_utility"], errors="coerce") - pd.to_numeric(
        df["reference_utility"], errors="coerce"
    )

    # Clip exploitation index: near-zero Lindahl utility creates huge ratios
    for col in ["focal_exploitation", "reference_exploitation"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").clip(-5, 5)

    # Focal Elo and tier
    df["focal_elo"] = df["focal_model"].map(lambda m: MODEL_INFO.get(m, {}).get("elo", np.nan))
    df["focal_tier"] = df["focal_model"].map(lambda m: MODEL_INFO.get(m, {}).get("tier", "Unknown"))
    df["focal_short"] = df["focal_model"].map(_short_name)

    df.drop(columns=["_focal_side"], inplace=True)
    return df


# =============================================================================
# Plotting Helpers
# =============================================================================


def _metric_vs_sigma(ax, df, metric, ylabel, ylim=None, show_legend=True):
    """Standard line plot: metric vs σ, colored lines per α, error bars = SEM."""
    alpha_vals = sorted(df["alpha"].dropna().unique())
    for alpha_val in alpha_vals:
        sub = df[df["alpha"] == alpha_val].dropna(subset=[metric])
        if sub.empty:
            continue
        g = sub.groupby("sigma")[metric].agg(["mean", "std", "count"]).reset_index()
        g["sem"] = g["std"] / np.sqrt(g["count"].clip(lower=1))
        color = ALPHA_COLORS.get(alpha_val, "#999999")
        ax.errorbar(
            g["sigma"], g["mean"], yerr=g["sem"],
            marker="o", capsize=4, linewidth=2, markersize=7,
            label=rf"$\alpha = {alpha_val:.1f}$", color=color,
        )
    ax.set_xlabel(r"$\sigma$ (Budget Abundance $\rightarrow$)")
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    if show_legend:
        ax.legend()


def _strip_by_sigma(ax, df, metric, ylabel, show_legend=True):
    """Strip/dot plot: individual experiments as dots, colored by α,
    with mean trend lines overlaid. x = σ, y = metric."""
    focal_models = sorted(df["focal_short"].dropna().unique())
    markers = ["o", "^", "s", "D", "v", "P"]
    model_markers = {m: markers[i % len(markers)] for i, m in enumerate(focal_models)}

    valid = df.dropna(subset=[metric])

    # Individual dots
    for _, row in valid.iterrows():
        alpha = row["alpha"]
        color = ALPHA_COLORS.get(alpha, "#999999")
        marker = model_markers.get(row.get("focal_short", ""), "o")
        jitter = np.random.uniform(-0.015, 0.015)
        ax.scatter(
            row["sigma"] + jitter, row[metric],
            c=color, marker=marker, s=55, alpha=0.65,
            edgecolors="white", linewidth=0.5, zorder=3,
        )

    # Mean trend lines per α
    for alpha_val in sorted(df["alpha"].dropna().unique()):
        sub = valid[valid["alpha"] == alpha_val]
        if len(sub) < 2:
            continue
        g = sub.groupby("sigma")[metric].mean().reset_index()
        color = ALPHA_COLORS.get(alpha_val, "#999999")
        ax.plot(g["sigma"], g[metric], "--", color=color, alpha=0.6, linewidth=1.5)

    ax.set_xlabel(r"$\sigma$ (Budget Abundance $\rightarrow$)")
    ax.set_ylabel(ylabel)

    if show_legend:
        legend_els = []
        for av, c in sorted(ALPHA_COLORS.items()):
            legend_els.append(Patch(facecolor=c, alpha=0.7, label=rf"$\alpha={av:.1f}$"))
        for model, marker in model_markers.items():
            legend_els.append(
                Line2D([0], [0], marker=marker, color="gray", linestyle="",
                       markersize=7, label=model)
            )
        ax.legend(handles=legend_els, fontsize=9, loc="best")


def _savefig(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path.name}")


# =============================================================================
# Figure 1: Efficiency Landscape
# =============================================================================


def fig1_efficiency_landscape(df: pd.DataFrame, figures_dir: Path):
    """Three-panel overview: efficiency, provision rate, coordination failure."""
    panels = [
        ("utilitarian_efficiency", "Utilitarian Efficiency ($\\eta$)", (0, 1.05)),
        ("provision_rate", "Provision Rate", (0, 1.05)),
        ("coordination_failure_weighted", "Coordination Failure", (0, 1.05)),
    ]
    available = [(m, l, yl) for m, l, yl in panels if m in df.columns and df[m].notna().any()]
    if not available:
        print("  Skipping fig1: no efficiency metrics")
        return

    ncols = len(available)
    fig, axes = plt.subplots(1, ncols, figsize=(5.5 * ncols, 5))
    if ncols == 1:
        axes = [axes]

    for i, (metric, label, ylim) in enumerate(available):
        _metric_vs_sigma(axes[i], df, metric, label, ylim=ylim, show_legend=(i == 0))
        axes[i].set_title(label, fontsize=12)

    fig.suptitle(
        "Group Efficiency Across Game Conditions",
        fontsize=14, fontweight="bold", y=1.03,
    )
    fig.tight_layout()
    _savefig(fig, figures_dir / "fig1_efficiency_landscape.png")


# =============================================================================
# Figure 2: Utility Asymmetry
# =============================================================================


def fig2_utility_asymmetry(df: pd.DataFrame, figures_dir: Path):
    """Strip plots showing whether stronger (focal) models extract more surplus."""
    metrics = [
        ("focal_advantage", "Focal Advantage\n(Focal $-$ Reference Utility)"),
        ("focal_exploitation", "Exploitation Index\n(vs Lindahl Equilibrium)"),
    ]
    available = [(m, l) for m, l in metrics if m in df.columns and df[m].notna().any()]
    if not available:
        print("  Skipping fig2: no asymmetry metrics")
        return

    ncols = len(available)
    fig, axes = plt.subplots(1, ncols, figsize=(6.5 * ncols, 5.5))
    if ncols == 1:
        axes = [axes]

    np.random.seed(42)  # reproducible jitter

    for i, (metric, ylabel) in enumerate(available):
        _strip_by_sigma(axes[i], df, metric, ylabel, show_legend=(i == ncols - 1))
        axes[i].axhline(0, color="black", linewidth=0.8, linestyle="-", alpha=0.3, zorder=1)
        axes[i].set_title(ylabel.split("\n")[0], fontsize=12)

    fig.suptitle(
        "Capability-Driven Utility Asymmetry",
        fontsize=14, fontweight="bold", y=1.03,
    )
    fig.tight_layout()
    _savefig(fig, figures_dir / "fig2_utility_asymmetry.png")


# =============================================================================
# Figure 3: Cost-Sharing Fairness
# =============================================================================


def fig3_cost_sharing_fairness(df: pd.DataFrame, figures_dir: Path):
    """Free-riding and Lindahl distance: how exploitation manifests mechanically."""
    panels = [
        ("focal_free_rider", "Free-Rider Index (Focal)\n($>$1 = free-riding)", None),
        ("lindahl_distance", "Lindahl Distance\n(Frobenius Norm)", None),
    ]
    available = [(m, l, yl) for m, l, yl in panels if m in df.columns and df[m].notna().any()]
    if not available:
        print("  Skipping fig3: no cost-sharing metrics")
        return

    ncols = len(available)
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5))
    if ncols == 1:
        axes = [axes]

    for i, (metric, ylabel, ylim) in enumerate(available):
        _metric_vs_sigma(axes[i], df, metric, ylabel, ylim=ylim, show_legend=(i == 0))
        axes[i].set_title(ylabel.split("\n")[0], fontsize=12)
        if "free_rider" in metric:
            axes[i].axhline(
                1, color="red", linewidth=0.8, linestyle="--", alpha=0.5,
            )
            # Rebuild legend with reference line included
            handles, labels = axes[i].get_legend_handles_labels()
            handles.append(Line2D([0], [0], color="red", linestyle="--", alpha=0.5))
            labels.append("Fair share ($F$=1)")
            axes[i].legend(handles, labels, fontsize=9)

    fig.suptitle(
        "Cost-Sharing Fairness",
        fontsize=14, fontweight="bold", y=1.03,
    )
    fig.tight_layout()
    _savefig(fig, figures_dir / "fig3_cost_sharing_fairness.png")


# =============================================================================
# Figure 4: Strategic Dynamics (conditional)
# =============================================================================


def fig4_strategic_dynamics(df: pd.DataFrame, figures_dir: Path):
    """Adaptation rate (focal vs reference) and game length.

    Only generated if adaptation data has sufficient coverage.
    """
    has_adapt = (
        "focal_adaptation" in df.columns
        and df["focal_adaptation"].notna().sum() >= 5
    )
    has_rounds = "final_round" in df.columns and df["final_round"].notna().sum() >= 5

    if not has_adapt and not has_rounds:
        print("  Skipping fig4: insufficient strategic dynamics data")
        return

    panels = []
    if has_adapt:
        panels.append("adaptation")
    if has_rounds:
        panels.append("rounds")

    ncols = len(panels)
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5))
    if ncols == 1:
        axes = [axes]

    panel_idx = 0

    if has_adapt:
        ax = axes[panel_idx]
        for metric, label, color in [
            ("focal_adaptation", "Focal Model", FOCAL_COLOR),
            ("reference_adaptation", "Reference Model", REFERENCE_COLOR),
        ]:
            if metric not in df.columns:
                continue
            g = df.groupby("sigma")[metric].agg(["mean", "std", "count"]).reset_index()
            g["sem"] = g["std"] / np.sqrt(g["count"].clip(lower=1))
            ax.errorbar(
                g["sigma"], g["mean"], yerr=g["sem"],
                marker="o", capsize=4, linewidth=2, markersize=7,
                label=label, color=color,
            )
        ax.set_xlabel(r"$\sigma$ (Budget Abundance $\rightarrow$)")
        ax.set_ylabel("Adaptation Rate")
        ax.set_title("Strategy Adaptation", fontsize=12)
        ax.legend()
        panel_idx += 1

    if has_rounds:
        ax = axes[panel_idx]
        _metric_vs_sigma(ax, df, "final_round", "Rounds Played", show_legend=True)
        ax.set_title("Negotiation Length", fontsize=12)

    fig.suptitle(
        "Strategic Dynamics",
        fontsize=14, fontweight="bold", y=1.03,
    )
    fig.tight_layout()
    _savefig(fig, figures_dir / "fig4_strategic_dynamics.png")


# =============================================================================
# Figure 5: Model Scaling (conditional on >=5 focal models)
# =============================================================================


def fig5_model_scaling(df: pd.DataFrame, figures_dir: Path):
    """Bar charts of key metrics grouped by focal model (sorted by Elo).

    With >=5 focal models this reveals scaling patterns; with fewer models
    it still provides a useful per-model comparison.
    """
    if "focal_model" not in df.columns:
        print("  Skipping fig5: no focal model data")
        return

    focal_models = df.dropna(subset=["focal_elo"])["focal_model"].unique()
    if len(focal_models) < 2:
        print(f"  Skipping fig5: only {len(focal_models)} focal model(s)")
        return

    metrics = [
        ("focal_advantage", "Focal Advantage"),
        ("utilitarian_efficiency", "Efficiency ($\\eta$)"),
        ("focal_free_rider", "Free-Rider Index"),
    ]
    available = [(m, l) for m, l in metrics if m in df.columns and df[m].notna().any()]
    if not available:
        print("  Skipping fig5: no metrics for scaling plot")
        return

    ncols = len(available)
    fig, axes = plt.subplots(1, ncols, figsize=(5.5 * ncols, 5.5))
    if ncols == 1:
        axes = [axes]

    for ax, (metric, label) in zip(axes, available):
        # Compute per-model means and sort by Elo
        grouped = (
            df.dropna(subset=[metric, "focal_elo"])
            .groupby("focal_model")
            .agg(
                mean=(metric, "mean"),
                std=(metric, "std"),
                count=(metric, "count"),
                elo=("focal_elo", "first"),
                tier=("focal_tier", "first"),
            )
            .reset_index()
            .sort_values("elo")
        )

        if grouped.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue

        grouped["sem"] = grouped["std"] / np.sqrt(grouped["count"].clip(lower=1))
        colors = [TIER_COLORS.get(t, "#95a5a6") for t in grouped["tier"]]
        x = np.arange(len(grouped))

        bars = ax.bar(
            x, grouped["mean"], yerr=grouped["sem"],
            color=colors, capsize=3, edgecolor="black", linewidth=0.5,
        )
        # Annotate sample sizes inside/near bars
        for xi, (_, row) in zip(x, grouped.iterrows()):
            n = int(row["count"])
            ax.text(
                xi, row["mean"] * 0.5, f"n={n}",
                ha="center", va="center", fontsize=8, color="white",
                fontweight="bold",
            )

        ax.set_xticks(x)
        short_names = [_short_name(m) for m in grouped["focal_model"]]
        ax.set_xticklabels(short_names, rotation=35, ha="right", fontsize=9)
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(f"{label} by Focal Model", fontsize=12)

        if "advantage" in metric or "exploitation" in metric:
            ax.axhline(0, color="black", linewidth=0.5)

    # Shared tier legend on last axis
    legend_elements = [Patch(facecolor=c, label=t) for t, c in TIER_COLORS.items()]
    axes[-1].legend(handles=legend_elements, loc="upper left", fontsize=9)

    fig.suptitle(
        "Per-Model Performance (sorted by Elo $\\rightarrow$)",
        fontsize=14, fontweight="bold", y=1.03,
    )
    fig.tight_layout()
    _savefig(fig, figures_dir / "fig5_model_scaling.png")


# =============================================================================
# Summary
# =============================================================================


def save_summary_table(df: pd.DataFrame, figures_dir: Path):
    """Save a curated metrics table with the most important columns."""
    cols = [
        "focal_model", "focal_short", "focal_elo", "focal_tier",
        "alpha", "sigma", "competition_index", "model_order",
        "utilitarian_efficiency", "provision_rate",
        "coordination_failure_weighted",
        "focal_advantage", "focal_exploitation",
        "focal_free_rider", "lindahl_distance",
        "focal_adaptation", "reference_adaptation",
        "promise_keep_rate", "persuasion_other_agent_delta",
        "coalition_persistent_fraction",
        "num_funded", "m_projects", "final_round",
    ]
    available = [c for c in cols if c in df.columns]
    out = df[available].copy()
    path = figures_dir / "summary_table.csv"
    out.to_csv(path, index=False, float_format="%.4f")
    print(f"  Saved summary_table.csv ({len(out)} rows, {len(available)} columns)")


def print_summary_statistics(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    print(f"Total experiments:     {len(df)}")
    if "focal_model" in df.columns:
        print(f"Focal models:          {sorted(df['focal_short'].unique())}")
    print(f"Alpha values:          {sorted(df['alpha'].dropna().unique())}")
    print(f"Sigma values:          {sorted(df['sigma'].dropna().unique())}")
    print()

    summary_metrics = [
        ("utilitarian_efficiency", "Efficiency (η)"),
        ("provision_rate", "Provision rate"),
        ("coordination_failure_weighted", "Coordination failure"),
        ("focal_advantage", "Focal advantage"),
        ("focal_exploitation", "Focal exploitation"),
        ("focal_free_rider", "Focal free-rider"),
        ("lindahl_distance", "Lindahl distance"),
        ("promise_keep_rate", "Promise-keeping"),
        ("persuasion_other_agent_delta", "Persuasion delta"),
    ]
    for metric, label in summary_metrics:
        if metric in df.columns:
            vals = df[metric].dropna()
            if len(vals) > 0:
                print(f"  {label:25s}: {vals.mean():+8.3f} ± {vals.std():.3f}  (n={len(vals)})")


# =============================================================================
# Figure 6: Competition Index
# =============================================================================


def fig6_competition_index(df: pd.DataFrame, figures_dir: Path):
    """Three-panel figure showing key metrics vs Competition Index CI₃ = (1−α)·(1−σ).

    Panel 1: focal_advantage vs CI₃ bucket
    Panel 2: utilitarian_efficiency vs CI₃ bucket
    Panel 3: focal_utility and reference_utility vs CI₃ bucket (two lines)
    """
    if "competition_index" not in df.columns or df["competition_index"].isna().all():
        print("  Skipping fig6: competition_index column missing or all NaN")
        return

    df = df.copy()
    df["ci_bucket"] = df["competition_index"].round(2)

    panels_bar = [
        ("focal_advantage", "Focal Advantage\n(Focal − Reference Utility)"),
        ("utilitarian_efficiency", "Utilitarian Efficiency ($\\eta$)"),
    ]
    available_bar = [(m, l) for m, l in panels_bar if m in df.columns and df[m].notna().any()]

    has_utilities = (
        "focal_utility" in df.columns and df["focal_utility"].notna().any()
        and "reference_utility" in df.columns and df["reference_utility"].notna().any()
    )

    if not available_bar and not has_utilities:
        print("  Skipping fig6: no metrics available for competition index plot")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ci_label = "Competition Index: CI$_3$ = $(1-\\alpha)(1-\\sigma)$"

    # Panel 1: focal_advantage vs ci_bucket
    ax = axes[0]
    if "focal_advantage" in df.columns and df["focal_advantage"].notna().any():
        g = (
            df.dropna(subset=["focal_advantage", "ci_bucket"])
            .groupby("ci_bucket")["focal_advantage"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        g["sem"] = g["std"] / np.sqrt(g["count"].clip(lower=1))
        ax.bar(
            g["ci_bucket"].astype(str), g["mean"], yerr=g["sem"],
            color=FOCAL_COLOR, capsize=4, edgecolor="black", linewidth=0.5, alpha=0.8,
        )
        ax.axhline(0, color="black", linewidth=0.8, linestyle="-", alpha=0.4)
        ax.set_xlabel(ci_label)
        ax.set_ylabel("Focal Advantage\n(Focal − Reference Utility)")
        ax.set_title("Focal Advantage vs CI$_3$", fontsize=12)
        ax.tick_params(axis="x", rotation=45)
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Focal Advantage vs CI$_3$", fontsize=12)

    # Panel 2: utilitarian_efficiency vs ci_bucket
    ax = axes[1]
    if "utilitarian_efficiency" in df.columns and df["utilitarian_efficiency"].notna().any():
        g = (
            df.dropna(subset=["utilitarian_efficiency", "ci_bucket"])
            .groupby("ci_bucket")["utilitarian_efficiency"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        g["sem"] = g["std"] / np.sqrt(g["count"].clip(lower=1))
        ax.bar(
            g["ci_bucket"].astype(str), g["mean"], yerr=g["sem"],
            color="#2ecc71", capsize=4, edgecolor="black", linewidth=0.5, alpha=0.8,
        )
        ax.set_xlabel(ci_label)
        ax.set_ylabel("Utilitarian Efficiency ($\\eta$)")
        ax.set_title("Efficiency vs CI$_3$", fontsize=12)
        ax.tick_params(axis="x", rotation=45)
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Efficiency vs CI$_3$", fontsize=12)

    # Panel 3: focal_utility and reference_utility vs ci_bucket (two lines)
    ax = axes[2]
    if has_utilities:
        ci_buckets_sorted = sorted(df["ci_bucket"].dropna().unique())
        x_labels = [str(b) for b in ci_buckets_sorted]
        x_pos = np.arange(len(ci_buckets_sorted))

        for col, label, color in [
            ("focal_utility", "Focal Utility", FOCAL_COLOR),
            ("reference_utility", "Reference Utility", REFERENCE_COLOR),
        ]:
            g = (
                df.dropna(subset=[col, "ci_bucket"])
                .groupby("ci_bucket")[col]
                .agg(["mean", "std", "count"])
                .reset_index()
            )
            g["sem"] = g["std"] / np.sqrt(g["count"].clip(lower=1))
            # Align to sorted ci_buckets
            g = g.set_index("ci_bucket").reindex(ci_buckets_sorted).reset_index()
            ax.errorbar(
                x_pos, g["mean"], yerr=g["sem"],
                marker="o", capsize=4, linewidth=2, markersize=7,
                label=label, color=color,
            )

        ax.axhline(50, color="gray", linewidth=0.8, linestyle="--", alpha=0.5, label="y = 50")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, rotation=45)
        ax.set_xlabel(ci_label)
        ax.set_ylabel("Utility")
        ax.set_title("Focal vs Reference Utility vs CI$_3$", fontsize=12)
        ax.legend(fontsize=9)
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Focal vs Reference Utility vs CI$_3$", fontsize=12)

    fig.suptitle(
        r"Competition Index: CI$_3$ = $(1-\alpha)(1-\sigma)$",
        fontsize=14, fontweight="bold", y=1.03,
    )
    fig.tight_layout()
    _savefig(fig, figures_dir / "fig6_competition_index.png")


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
    df = _add_focal_reference_columns(df)

    # Summary stats
    print_summary_statistics(df)

    # Save full metrics table
    save_summary_table(df, figures_dir)

    # Generate figures
    print("\n" + "=" * 60)
    print("GENERATING FIGURES")
    print("=" * 60)

    fig1_efficiency_landscape(df, figures_dir)
    fig2_utility_asymmetry(df, figures_dir)
    fig3_cost_sharing_fairness(df, figures_dir)
    fig4_strategic_dynamics(df, figures_dir)
    fig5_model_scaling(df, figures_dir)
    fig6_competition_index(df, figures_dir)

    print(f"\nDone! Figures saved to: {figures_dir}")


if __name__ == "__main__":
    main()
