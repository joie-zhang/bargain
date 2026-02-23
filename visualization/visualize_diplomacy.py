#!/usr/bin/env python3
"""
=============================================================================
Visualize Game 2 (Diplomatic Treaty) Experiment Results
=============================================================================

Creates plots analyzing model-scale and rho/theta effects on negotiation
outcomes in the Diplomatic Treaty game.

Usage:
    python visualization/visualize_diplomacy.py                          # Latest diplomacy results
    python visualization/visualize_diplomacy.py --dir experiments/results/diplomacy_20260222_004806

What it creates:
    visualization/figures/diplomacy/
    ├── plot1_model_pair_utilities.png       # Utility comparison by model pair
    ├── plot2_rho_effect.png                 # Effect of rho on social welfare
    ├── plot3_theta_effect.png               # Effect of theta on social welfare
    ├── plot4_rho_theta_heatmap.png          # Heatmap of SW across rho x theta
    ├── plot4b_rho_theta_per_agent.png      # Heatmap of per-agent utility across rho x theta
    ├── plot5_exploitation_by_condition.png  # Exploitation rates
    ├── plot6_ttc_scaling.png               # TTC scaling (if available)
    ├── plot7_competition_index.png         # 1D collapsed competition index
    ├── plot8_by_theta.png                  # SW vs rho, faceted by theta
    ├── plot9_by_rho.png                    # SW vs theta, faceted by rho
    └── diplomacy_results.csv               # Full results table

Dependencies:
    - matplotlib, seaborn, pandas, numpy

=============================================================================
"""

import json
import os
import sys
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Style
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Helvetica",
            "DejaVu Sans",
            "Liberation Sans",
            "sans-serif",
        ],
        "font.size": 14,
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 12,
        "figure.figsize": (12, 8),
        "figure.dpi": 150,
        "savefig.dpi": 300,
    }
)

# Short model names for display
MODEL_SHORT_NAMES = {
    "claude-opus-4-5-thinking-32k": "Claude Opus",
    "gpt-5-nano": "GPT-5-nano",
    "o3-mini-high": "O3-mini",
    "gpt-5.2-high": "GPT-5.2",
    "grok-4": "Grok-4",
    "deepseek-r1": "DeepSeek-R1",
    "QwQ-32B": "QwQ-32B",
    "gpt-3.5-turbo-0125": "GPT-3.5",
    "gpt-4o": "GPT-4o",
    "amazon-nova-micro": "Nova Micro",
    "claude-haiku-4-5": "Haiku 4.5",
    "claude-sonnet-4-5": "Sonnet 4.5",
    "gemini-3-pro": "Gemini 3 Pro",
}

# Elo ratings (Chatbot Arena, approx. Jan 2026)
MODEL_ELO = {
    "gemini-3-pro": 1490,
    "claude-opus-4-5-thinking-32k": 1470,
    "claude-sonnet-4-5": 1450,
    "gpt-5.2-high": 1436,
    "grok-4": 1409,
    "claude-haiku-4-5": 1403,
    "deepseek-r1": 1397,
    "o3-mini-high": 1364,
    "gpt-4o": 1346,
    "gpt-5-nano": 1338,
    "QwQ-32B": 1316,
    "amazon-nova-micro": 1220,
    "gpt-3.5-turbo-0125": 1105,
}


MARKER_POOL = ["o", "s", "D", "^", "v", "P", "X", "*", "h", "<", ">"]


def short_name(model_id: str) -> str:
    return MODEL_SHORT_NAMES.get(model_id, model_id)


def _markers_for(hue_levels):
    """Return a list of markers matching the number of unique hue levels."""
    n = len(hue_levels)
    return MARKER_POOL[:n] if n <= len(MARKER_POOL) else ["o"] * n


def collect_results(experiment_dir: str) -> pd.DataFrame:
    """Collect results from all experiment configs into a DataFrame."""
    config_dir = os.path.join(experiment_dir, "configs")
    index_file = os.path.join(config_dir, "experiment_index.csv")

    if not os.path.exists(index_file):
        raise FileNotFoundError(f"No experiment index at {index_file}")

    index_df = pd.read_csv(index_file)
    print(f"Found {len(index_df)} experiment configs")

    rows = []
    missing = 0
    for _, idx_row in index_df.iterrows():
        config_path = os.path.join(config_dir, idx_row["config_file"])
        with open(config_path) as f:
            config = json.load(f)

        output_dir = config["output_dir"]
        result_file = os.path.join(output_dir, "run_1_experiment_results.json")

        if not os.path.exists(result_file):
            missing += 1
            continue

        with open(result_file) as f:
            result = json.load(f)

        final_utils = result.get("final_utilities", {})
        alpha_util = final_utils.get("Agent_Alpha", 0)
        beta_util = final_utils.get("Agent_Beta", 0)

        # Determine which model is which agent
        models_list = config["models"]
        model_alpha = models_list[0]
        model_beta = models_list[1] if len(models_list) > 1 else models_list[0]

        row = {
            "experiment_id": config["experiment_id"],
            "experiment_type": config.get("experiment_type", "unknown"),
            "model_alpha": model_alpha,
            "model_beta": model_beta,
            "model_alpha_short": short_name(model_alpha),
            "model_beta_short": short_name(model_beta),
            "model_order": config["model_order"],
            "rho": config["rho"],
            "theta": config["theta"],
            "n_issues": config["n_issues"],
            "consensus": result.get("consensus_reached", False),
            "final_round": result.get("final_round", -1),
            "exploitation": result.get("exploitation_detected", False),
            "alpha_util": alpha_util,
            "beta_util": beta_util,
            "social_welfare": alpha_util + beta_util,
            "util_ratio": alpha_util / beta_util if beta_util > 0 else float("inf"),
        }

        # TTC-specific fields
        if config.get("experiment_type") == "ttc_scaling":
            row["token_budget"] = config.get("reasoning_token_budget", None)
            row["reasoning_model"] = config.get("reasoning_model", "")
            row["baseline_model"] = config.get("baseline_model", "")
        else:
            row["token_budget"] = None
            m1 = config.get("model1", model_alpha)
            m2 = config.get("model2", model_beta)
            row["model_pair"] = f"{short_name(m1)} vs {short_name(m2)}"

        # Adversary model info (model2 is always the adversary)
        adversary_raw = config.get("model2", model_beta)
        row["adversary_model"] = adversary_raw
        row["adversary_short"] = short_name(adversary_raw)
        row["adversary_elo"] = MODEL_ELO.get(adversary_raw, 0)

        rows.append(row)

    if missing > 0:
        print(f"  WARNING: {missing} results files missing")

    df = pd.DataFrame(rows)
    print(f"Collected {len(df)} results")
    return df


def _elo_sorted_pairs(df: pd.DataFrame) -> list:
    """Return model_pair values sorted by adversary Elo (ascending)."""
    pair_elo = (
        df.groupby("model_pair")["adversary_elo"]
        .first()
        .sort_values()
    )
    return list(pair_elo.index)


def _pair_tick_label(pair: str, df: pd.DataFrame) -> str:
    """Build an x-tick label like 'GPT-5-nano vs O3-mini\n(Elo 1364)'."""
    row = df[df["model_pair"] == pair].iloc[0]
    elo = int(row["adversary_elo"]) if row["adversary_elo"] else "?"
    return f"{pair}\n(Elo {elo})"


def plot_model_pair_utilities(df: pd.DataFrame, out_dir: str):
    """Plot 1: Utility comparison by model pair, ordered by adversary Elo."""
    ms = df[df["experiment_type"] == "model_scale"].copy()
    if ms.empty:
        print("  Skipping plot 1: no model-scale data")
        return

    # Remap to weak/strong utilities (position-independent)
    ms["weak_util"] = ms.apply(
        lambda r: r["alpha_util"] if r["model_order"] == "weak_first" else r["beta_util"],
        axis=1,
    )
    ms["strong_util"] = ms.apply(
        lambda r: r["beta_util"] if r["model_order"] == "weak_first" else r["alpha_util"],
        axis=1,
    )

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Order pairs by adversary Elo
    pairs_ordered = _elo_sorted_pairs(ms)

    # Bar chart: mean weak vs strong utility
    pair_stats = []
    for pair in pairs_ordered:
        pdata = ms[ms["model_pair"] == pair]
        pair_stats.append(
            {
                "pair": pair,
                "tick": _pair_tick_label(pair, ms),
                "Baseline (GPT-5-nano)": pdata["weak_util"].mean(),
                "Adversary": pdata["strong_util"].mean(),
                "weak_std": pdata["weak_util"].std(),
                "strong_std": pdata["strong_util"].std(),
            }
        )
    stats_df = pd.DataFrame(pair_stats)

    x = np.arange(len(stats_df))
    width = 0.35
    axes[0].bar(
        x - width / 2,
        stats_df["Baseline (GPT-5-nano)"],
        width,
        yerr=stats_df["weak_std"],
        label="Baseline (GPT-5-nano)",
        alpha=0.8,
        capsize=4,
        color="#3498db",
    )
    axes[0].bar(
        x + width / 2,
        stats_df["Adversary"],
        width,
        yerr=stats_df["strong_std"],
        label="Adversary",
        alpha=0.8,
        capsize=4,
        color="#e74c3c",
    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(stats_df["tick"], rotation=30, ha="right", fontsize=10)
    axes[0].set_ylabel("Mean Utility")
    axes[0].set_title("Mean Utility by Model Pair (ordered by Adversary Elo)")
    axes[0].set_xlabel(r"Adversary Model  (Elo $\rightarrow$)")
    axes[0].legend()

    # Box plot of social welfare, ordered by adversary Elo
    sns.boxplot(
        data=ms, x="model_pair", y="social_welfare",
        order=pairs_ordered, ax=axes[1],
    )
    tick_labels = [_pair_tick_label(p, ms) for p in pairs_ordered]
    axes[1].set_xticklabels(tick_labels, rotation=30, ha="right", fontsize=10)
    axes[1].set_xlabel(r"Adversary Model  (Elo $\rightarrow$)")
    axes[1].set_ylabel("Social Welfare (sum of utilities)")
    axes[1].set_title("Social Welfare Distribution (ordered by Adversary Elo)")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "plot1_model_pair_utilities.png"), bbox_inches="tight")
    plt.close()
    print("  Saved plot1_model_pair_utilities.png")


def plot_rho_effect(df: pd.DataFrame, out_dir: str):
    """Plot 2: Effect of rho on social welfare, rounds, exploitation."""
    ms = df[df["experiment_type"] == "model_scale"].copy()
    if ms.empty:
        print("  Skipping plot 2: no model-scale data")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    pairs_ordered = _elo_sorted_pairs(ms)
    pair_markers = _markers_for(pairs_ordered)

    # Social welfare vs rho
    sns.pointplot(
        data=ms,
        x="rho",
        y="social_welfare",
        hue="model_pair",
        hue_order=pairs_ordered,
        ax=axes[0],
        dodge=0.2,
        markers=pair_markers,
        capsize=0.1,
    )
    axes[0].set_xlabel(r"$\rho$  ($\leftarrow$ Competitive | Cooperative $\rightarrow$)")
    axes[0].set_ylabel("Social Welfare")
    axes[0].set_title(r"Social Welfare vs $\rho$")
    axes[0].legend(fontsize=9)

    # Rounds vs rho
    sns.pointplot(
        data=ms,
        x="rho",
        y="final_round",
        hue="model_pair",
        hue_order=pairs_ordered,
        ax=axes[1],
        dodge=0.2,
        markers=pair_markers,
        capsize=0.1,
    )
    axes[1].set_xlabel(r"$\rho$  ($\leftarrow$ Competitive | Cooperative $\rightarrow$)")
    axes[1].set_ylabel("Rounds to Consensus")
    axes[1].set_title(r"Negotiation Length vs $\rho$")
    axes[1].legend(fontsize=9)

    # Exploitation rate vs rho
    exploit_by_rho = ms.groupby("rho")["exploitation"].mean()
    axes[2].bar(exploit_by_rho.index.astype(str), exploit_by_rho.values, alpha=0.8)
    axes[2].set_xlabel(r"$\rho$  ($\leftarrow$ Competitive | Cooperative $\rightarrow$)")
    axes[2].set_ylabel("Exploitation Rate")
    axes[2].set_title(r"Exploitation Rate vs $\rho$")
    axes[2].set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "plot2_rho_effect.png"))
    plt.close()
    print("  Saved plot2_rho_effect.png")


def plot_theta_effect(df: pd.DataFrame, out_dir: str):
    """Plot 3: Effect of theta on social welfare, rounds, exploitation."""
    ms = df[df["experiment_type"] == "model_scale"].copy()
    if ms.empty:
        print("  Skipping plot 3: no model-scale data")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    pairs_ordered = _elo_sorted_pairs(ms)
    pair_markers = _markers_for(pairs_ordered)

    # Social welfare vs theta
    sns.pointplot(
        data=ms,
        x="theta",
        y="social_welfare",
        hue="model_pair",
        hue_order=pairs_ordered,
        ax=axes[0],
        dodge=0.2,
        markers=pair_markers,
        capsize=0.1,
    )
    axes[0].set_xlabel(r"$\theta$  (amplifies conflict $\rightarrow$)")
    axes[0].set_ylabel("Social Welfare")
    axes[0].set_title(r"Social Welfare vs $\theta$")
    axes[0].legend(fontsize=9)

    # Rounds vs theta
    sns.pointplot(
        data=ms,
        x="theta",
        y="final_round",
        hue="model_pair",
        hue_order=pairs_ordered,
        ax=axes[1],
        dodge=0.2,
        markers=pair_markers,
        capsize=0.1,
    )
    axes[1].set_xlabel(r"$\theta$  (amplifies conflict $\rightarrow$)")
    axes[1].set_ylabel("Rounds to Consensus")
    axes[1].set_title(r"Negotiation Length vs $\theta$")
    axes[1].legend(fontsize=9)

    # Exploitation rate vs theta
    exploit_by_theta = ms.groupby("theta")["exploitation"].mean()
    axes[2].bar(
        exploit_by_theta.index.astype(str), exploit_by_theta.values, alpha=0.8
    )
    axes[2].set_xlabel(r"$\theta$  (amplifies conflict $\rightarrow$)")
    axes[2].set_ylabel("Exploitation Rate")
    axes[2].set_title(r"Exploitation Rate vs $\theta$")
    axes[2].set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "plot3_theta_effect.png"))
    plt.close()
    print("  Saved plot3_theta_effect.png")


def _annotate_diplomacy_heatmap(ax):
    """Add competition/cooperation corner annotations to a rho x theta heatmap.

    Heatmap layout (rho sorted ascending, so -1 at top, 1 at bottom):
      - Top-right = high theta + low rho = most competitive
      - Bottom-left = low theta + high rho = most cooperative
    """
    ax.text(
        0.97, 0.97, "Comp.",
        transform=ax.transAxes, fontsize=9, fontstyle="italic",
        ha="right", va="top", color="white",
        bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.5),
    )
    ax.text(
        0.03, 0.03, "Coop.",
        transform=ax.transAxes, fontsize=9, fontstyle="italic",
        ha="left", va="bottom", color="black",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.5),
    )


def plot_rho_theta_heatmap(df: pd.DataFrame, out_dir: str):
    """Plot 4: Heatmap of social welfare across rho x theta grid."""
    ms = df[df["experiment_type"] == "model_scale"].copy()
    if ms.empty:
        print("  Skipping plot 4: no model-scale data")
        return

    pairs_ordered = _elo_sorted_pairs(ms)
    n_pairs = len(pairs_ordered)
    fig, axes = plt.subplots(1, n_pairs + 1, figsize=(6 * (n_pairs + 1), 5))
    if n_pairs + 1 == 1:
        axes = [axes]

    # Overall heatmap
    pivot = ms.pivot_table(
        values="social_welfare", index="rho", columns="theta", aggfunc="mean"
    )
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        ax=axes[0],
        cbar_kws={"label": "Social Welfare"},
    )
    axes[0].set_title("All Model Pairs")
    axes[0].set_xlabel(r"$\theta$ (Interest Overlap — amplifies conflict $\rightarrow$)")
    axes[0].set_ylabel(r"$\rho$ (Pref. Correlation)  $\leftarrow$ Cooperative | Competitive $\rightarrow$")
    _annotate_diplomacy_heatmap(axes[0])

    # Per-pair heatmaps (ordered by adversary Elo)
    for idx, pair in enumerate(pairs_ordered):
        pdata = ms[ms["model_pair"] == pair]
        pivot = pdata.pivot_table(
            values="social_welfare", index="rho", columns="theta", aggfunc="mean"
        )
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".2f",
            cmap="YlOrRd",
            ax=axes[idx + 1],
            cbar_kws={"label": "SW"},
        )
        axes[idx + 1].set_title(pair)
        axes[idx + 1].set_xlabel(r"$\theta$ (amplifies conflict $\rightarrow$)")
        axes[idx + 1].set_ylabel(r"$\rho$ ($\leftarrow$ Coop. | Comp. $\rightarrow$)")
        _annotate_diplomacy_heatmap(axes[idx + 1])

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "plot4_rho_theta_heatmap.png"))
    plt.close()
    print("  Saved plot4_rho_theta_heatmap.png")


def plot_rho_theta_heatmap_per_agent(df: pd.DataFrame, out_dir: str):
    """Plot 4b: Heatmap of per-agent utility across rho x theta grid.

    Remaps positional alpha/beta roles to weak/strong model identity,
    so utilities are attributed to the correct model regardless of
    speaking order.
    """
    ms = df[df["experiment_type"] == "model_scale"].copy()
    if ms.empty:
        print("  Skipping plot 4b: no model-scale data")
        return

    # Remap positional alpha/beta to weak/strong model utility
    # weak_first: alpha = weak model, beta = strong model
    # strong_first: alpha = strong model, beta = weak model
    ms["weak_util"] = ms.apply(
        lambda r: r["alpha_util"]
        if r["model_order"] == "weak_first"
        else r["beta_util"],
        axis=1,
    )
    ms["strong_util"] = ms.apply(
        lambda r: r["beta_util"]
        if r["model_order"] == "weak_first"
        else r["alpha_util"],
        axis=1,
    )

    pairs_ordered = _elo_sorted_pairs(ms)
    n_pairs = len(pairs_ordered)
    n_cols = n_pairs + 1  # "All" + each pair

    fig, axes = plt.subplots(2, n_cols, figsize=(6 * n_cols, 10))
    if n_cols == 1:
        axes = axes.reshape(2, 1)

    metrics = [
        ("weak_util", "Weak Model (GPT-5-nano)"),
        ("strong_util", "Strong Model"),
    ]

    for row_idx, (metric, row_label) in enumerate(metrics):
        # Overall heatmap
        pivot = ms.pivot_table(
            values=metric, index="rho", columns="theta", aggfunc="mean"
        )
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".2f",
            cmap="YlOrRd",
            ax=axes[row_idx, 0],
            cbar_kws={"label": "Utility"},
        )
        axes[row_idx, 0].set_title(f"All Pairs — {row_label}")
        axes[row_idx, 0].set_xlabel(r"$\theta$ (amplifies conflict $\rightarrow$)")
        axes[row_idx, 0].set_ylabel(r"$\rho$ ($\leftarrow$ Coop. | Comp. $\rightarrow$)")
        _annotate_diplomacy_heatmap(axes[row_idx, 0])

        # Per-pair heatmaps (ordered by adversary Elo)
        for col_idx, pair in enumerate(pairs_ordered):
            pdata = ms[ms["model_pair"] == pair]

            # For strong model label, extract the strong model name from the pair
            if row_idx == 1:
                # pair format: "GPT-5-nano vs X" — strong model is X
                strong_name = pair.split(" vs ")[-1]
                title = f"{pair}\n{strong_name}"
            else:
                title = f"{pair}\nGPT-5-nano"

            pivot = pdata.pivot_table(
                values=metric, index="rho", columns="theta", aggfunc="mean"
            )
            sns.heatmap(
                pivot,
                annot=True,
                fmt=".2f",
                cmap="YlOrRd",
                ax=axes[row_idx, col_idx + 1],
                cbar_kws={"label": "Utility"},
            )
            axes[row_idx, col_idx + 1].set_title(title)
            axes[row_idx, col_idx + 1].set_xlabel(r"$\theta$ (amplifies conflict $\rightarrow$)")
            axes[row_idx, col_idx + 1].set_ylabel(r"$\rho$ ($\leftarrow$ Coop. | Comp. $\rightarrow$)")
            _annotate_diplomacy_heatmap(axes[row_idx, col_idx + 1])

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "plot4b_rho_theta_per_agent.png"))
    plt.close()
    print("  Saved plot4b_rho_theta_per_agent.png")


def plot_exploitation_by_condition(df: pd.DataFrame, out_dir: str):
    """Plot 5: Exploitation rates broken down by model pair, rho, theta."""
    ms = df[df["experiment_type"] == "model_scale"].copy()
    if ms.empty:
        print("  Skipping plot 5: no model-scale data")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Utility advantage heatmap (who benefits more: model1 or model2?)
    ms["util_advantage"] = ms["beta_util"] - ms["alpha_util"]
    pivot = ms.pivot_table(
        values="util_advantage", index="rho", columns="theta", aggfunc="mean"
    )
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        ax=axes[0],
        cbar_kws={"label": "Model2 - Model1 Utility"},
    )
    axes[0].set_title("Utility Advantage (Beta - Alpha)")
    axes[0].set_xlabel(r"$\theta$ (amplifies conflict $\rightarrow$)")
    axes[0].set_ylabel(r"$\rho$ ($\leftarrow$ Coop. | Comp. $\rightarrow$)")
    _annotate_diplomacy_heatmap(axes[0])

    # Model order effect
    order_stats = ms.groupby(["model_pair", "model_order"]).agg(
        mean_sw=("social_welfare", "mean"),
        mean_alpha=("alpha_util", "mean"),
        mean_beta=("beta_util", "mean"),
    ).reset_index()

    order_pivot = order_stats.pivot(
        index="model_pair", columns="model_order", values="mean_sw"
    )
    order_pivot.plot(kind="bar", ax=axes[1], alpha=0.8)
    axes[1].set_ylabel("Mean Social Welfare")
    axes[1].set_title("Effect of Speaking Order")
    axes[1].tick_params(axis="x", rotation=15)
    axes[1].legend(title="Speaking Order")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "plot5_exploitation_by_condition.png"))
    plt.close()
    print("  Saved plot5_exploitation_by_condition.png")


def plot_ttc_scaling(df: pd.DataFrame, out_dir: str):
    """Plot 6: TTC scaling results."""
    ttc = df[df["experiment_type"] == "ttc_scaling"].copy()
    if ttc.empty:
        print("  Skipping plot 6: no TTC scaling data")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # SW vs token budget
    sns.pointplot(
        data=ttc,
        x="token_budget",
        y="social_welfare",
        ax=axes[0],
        capsize=0.1,
        color="steelblue",
    )
    axes[0].set_xlabel("Reasoning Token Budget")
    axes[0].set_ylabel("Social Welfare")
    axes[0].set_title("Social Welfare vs Token Budget")

    # Remap to reasoning/baseline utility based on actual model assignment
    # When model_order=strong_first, alpha IS the reasoning model, not baseline
    ttc["reasoning_util"] = ttc.apply(
        lambda r: r["alpha_util"]
        if r["model_alpha"] == r["reasoning_model"]
        else r["beta_util"],
        axis=1,
    )
    ttc["baseline_util"] = ttc.apply(
        lambda r: r["alpha_util"]
        if r["model_alpha"] == r["baseline_model"]
        else r["beta_util"],
        axis=1,
    )

    ttc_melted = ttc.melt(
        id_vars=["token_budget", "model_order"],
        value_vars=["baseline_util", "reasoning_util"],
        var_name="agent",
        value_name="utility",
    )
    ttc_melted["agent"] = ttc_melted["agent"].map(
        {"baseline_util": "Baseline (GPT-5-nano)", "reasoning_util": "Reasoning (Claude Opus)"}
    )
    sns.pointplot(
        data=ttc_melted,
        x="token_budget",
        y="utility",
        hue="agent",
        ax=axes[1],
        capsize=0.1,
        dodge=0.2,
    )
    axes[1].set_xlabel("Reasoning Token Budget")
    axes[1].set_ylabel("Utility")
    axes[1].set_title("Per-Agent Utility vs Token Budget")
    axes[1].legend()

    # Rounds vs budget
    sns.pointplot(
        data=ttc,
        x="token_budget",
        y="final_round",
        ax=axes[2],
        capsize=0.1,
        color="coral",
    )
    axes[2].set_xlabel("Reasoning Token Budget")
    axes[2].set_ylabel("Rounds to Consensus")
    axes[2].set_title("Negotiation Length vs Token Budget")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "plot6_ttc_scaling.png"))
    plt.close()
    print("  Saved plot6_ttc_scaling.png")


def plot_competition_index(df: pd.DataFrame, out_dir: str):
    """Plot 7: 1D collapsed competition index view.

    CI = theta * (1 - rho) / 2, in [0, 1].
    """
    ms = df[df["experiment_type"] == "model_scale"].copy()
    if ms.empty:
        print("  Skipping plot 7: no model-scale data")
        return

    ms["competition_index"] = ms["theta"] * (1 - ms["rho"]) / 2

    # Remap to weak/strong utilities
    ms["weak_util"] = ms.apply(
        lambda r: r["alpha_util"] if r["model_order"] == "weak_first" else r["beta_util"],
        axis=1,
    )
    ms["strong_util"] = ms.apply(
        lambda r: r["beta_util"] if r["model_order"] == "weak_first" else r["alpha_util"],
        axis=1,
    )

    # Bin CI for cleaner plotting
    ms["ci_bin"] = ms["competition_index"].round(2)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Panel 1: Social welfare vs CI
    pairs_ordered = _elo_sorted_pairs(ms)
    if "model_pair" in ms.columns:
        sns.pointplot(
            data=ms, x="ci_bin", y="social_welfare", hue="model_pair",
            hue_order=pairs_ordered,
            ax=axes[0], dodge=0.15, capsize=0.1,
            markers=_markers_for(pairs_ordered),
        )
        axes[0].legend(fontsize=8, title="Model Pair")
    else:
        sns.pointplot(data=ms, x="ci_bin", y="social_welfare", ax=axes[0], capsize=0.1)
    axes[0].set_xlabel(r"Competition Index  ($\leftarrow$ Coop. | Comp. $\rightarrow$)")
    axes[0].set_ylabel("Social Welfare")
    axes[0].set_title("Social Welfare vs Competition Index")
    axes[0].tick_params(axis="x", rotation=45)

    # Panel 2: Exploitation rate vs CI
    exploit_by_ci = ms.groupby("ci_bin")["exploitation"].mean()
    axes[1].bar(
        [str(x) for x in exploit_by_ci.index],
        exploit_by_ci.values,
        alpha=0.8, color="coral",
    )
    axes[1].set_xlabel(r"Competition Index  ($\leftarrow$ Coop. | Comp. $\rightarrow$)")
    axes[1].set_ylabel("Exploitation Rate")
    axes[1].set_title("Exploitation Rate vs Competition Index")
    axes[1].set_ylim(0, 1.05)
    axes[1].tick_params(axis="x", rotation=45)

    # Panel 3: Per-agent utility vs CI
    melted = ms.melt(
        id_vars=["ci_bin"],
        value_vars=["weak_util", "strong_util"],
        var_name="agent",
        value_name="utility",
    )
    melted["agent"] = melted["agent"].map(
        {"weak_util": "Weak Model", "strong_util": "Strong Model"}
    )
    sns.pointplot(
        data=melted, x="ci_bin", y="utility", hue="agent",
        ax=axes[2], dodge=0.15, capsize=0.1,
    )
    axes[2].set_xlabel(r"Competition Index  ($\leftarrow$ Coop. | Comp. $\rightarrow$)")
    axes[2].set_ylabel("Utility")
    axes[2].set_title("Per-Agent Utility vs Competition Index")
    axes[2].legend(fontsize=10)
    axes[2].tick_params(axis="x", rotation=45)

    plt.suptitle(
        r"Competition Index: CI = $\theta \cdot (1 - \rho) / 2$",
        fontsize=16, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "plot7_competition_index.png"), bbox_inches="tight")
    plt.close()
    print("  Saved plot7_competition_index.png")


def plot_disaggregated(df: pd.DataFrame, out_dir: str):
    """Plots 8 & 9: Disaggregated views — social welfare faceted by one parameter."""
    ms = df[df["experiment_type"] == "model_scale"].copy()
    if ms.empty:
        print("  Skipping plots 8-9: no model-scale data")
        return

    pairs_ordered = _elo_sorted_pairs(ms)

    # --- Plot 8: SW vs rho, faceted by theta ---
    theta_vals = sorted(ms["theta"].unique())
    n_theta = len(theta_vals)
    if n_theta > 0:
        fig, axes = plt.subplots(1, n_theta, figsize=(6 * n_theta, 5), sharey=True)
        if n_theta == 1:
            axes = [axes]
        for idx, theta_val in enumerate(theta_vals):
            sub = ms[ms["theta"] == theta_val]
            sub_pairs = [p for p in pairs_ordered if p in sub["model_pair"].values]
            if "model_pair" in sub.columns and sub["model_pair"].nunique() > 1:
                sns.pointplot(
                    data=sub, x="rho", y="social_welfare", hue="model_pair",
                    hue_order=sub_pairs,
                    ax=axes[idx], dodge=0.15, capsize=0.1,
                    markers=_markers_for(sub_pairs),
                )
                if idx == n_theta - 1:
                    axes[idx].legend(fontsize=8)
                else:
                    axes[idx].get_legend().remove()
            else:
                sns.pointplot(
                    data=sub, x="rho", y="social_welfare",
                    ax=axes[idx], capsize=0.1,
                )
            axes[idx].set_title(rf"$\theta = {theta_val}$")
            axes[idx].set_xlabel(r"$\rho$  ($\leftarrow$ Competitive | Cooperative $\rightarrow$)")
            if idx == 0:
                axes[idx].set_ylabel("Social Welfare")
            else:
                axes[idx].set_ylabel("")

        plt.suptitle(
            r"Social Welfare vs $\rho$, disaggregated by $\theta$",
            fontsize=16, fontweight="bold", y=1.02,
        )
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "plot8_by_theta.png"), bbox_inches="tight")
        plt.close()
        print("  Saved plot8_by_theta.png")

    # --- Plot 9: SW vs theta, faceted by rho ---
    rho_vals = sorted(ms["rho"].unique())
    n_rho = len(rho_vals)
    if n_rho > 0:
        fig, axes = plt.subplots(1, n_rho, figsize=(6 * n_rho, 5), sharey=True)
        if n_rho == 1:
            axes = [axes]
        for idx, rho_val in enumerate(rho_vals):
            sub = ms[ms["rho"] == rho_val]
            sub_pairs = [p for p in pairs_ordered if p in sub["model_pair"].values]
            if "model_pair" in sub.columns and sub["model_pair"].nunique() > 1:
                sns.pointplot(
                    data=sub, x="theta", y="social_welfare", hue="model_pair",
                    hue_order=sub_pairs,
                    ax=axes[idx], dodge=0.15, capsize=0.1,
                    markers=_markers_for(sub_pairs),
                )
                if idx == n_rho - 1:
                    axes[idx].legend(fontsize=8)
                else:
                    axes[idx].get_legend().remove()
            else:
                sns.pointplot(
                    data=sub, x="theta", y="social_welfare",
                    ax=axes[idx], capsize=0.1,
                )
            axes[idx].set_title(rf"$\rho = {rho_val}$")
            axes[idx].set_xlabel(r"$\theta$  (amplifies conflict $\rightarrow$)")
            if idx == 0:
                axes[idx].set_ylabel("Social Welfare")
            else:
                axes[idx].set_ylabel("")

        plt.suptitle(
            r"Social Welfare vs $\theta$, disaggregated by $\rho$",
            fontsize=16, fontweight="bold", y=1.02,
        )
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "plot9_by_rho.png"), bbox_inches="tight")
        plt.close()
        print("  Saved plot9_by_rho.png")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize Game 2 (Diplomacy) experiment results"
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=None,
        help="Experiment results directory (default: latest diplomacy_* dir)",
    )
    args = parser.parse_args()

    # Find experiment directory
    if args.dir:
        experiment_dir = args.dir
    else:
        # Look for latest diplomacy results
        results_base = "experiments/results"
        diplomacy_dirs = sorted(
            [
                d
                for d in os.listdir(results_base)
                if d.startswith("diplomacy_") and d != "diplomacy_latest"
            ]
        )
        if not diplomacy_dirs:
            print("ERROR: No diplomacy experiment directories found")
            sys.exit(1)
        experiment_dir = os.path.join(results_base, diplomacy_dirs[-1])

    print(f"Analyzing: {experiment_dir}")

    # Collect results
    df = collect_results(experiment_dir)

    # Create output directory
    out_dir = "visualization/figures/diplomacy"
    os.makedirs(out_dir, exist_ok=True)

    # Save CSV
    csv_path = os.path.join(out_dir, "diplomacy_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved results CSV: {csv_path}")

    if df.empty:
        print("\nNo results to plot. Check that experiments have completed.")
        return

    # Generate plots
    print("\nGenerating plots...")
    plot_model_pair_utilities(df, out_dir)
    plot_rho_effect(df, out_dir)
    plot_theta_effect(df, out_dir)
    plot_rho_theta_heatmap(df, out_dir)
    plot_rho_theta_heatmap_per_agent(df, out_dir)
    plot_exploitation_by_condition(df, out_dir)
    plot_ttc_scaling(df, out_dir)
    plot_competition_index(df, out_dir)
    plot_disaggregated(df, out_dir)

    # Print summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    ms = df[df["experiment_type"] == "model_scale"]
    if not ms.empty:
        print(f"\nModel-scale experiments: {len(ms)}")
        print(f"  Consensus rate: {ms['consensus'].mean():.0%}")
        print(f"  Mean social welfare: {ms['social_welfare'].mean():.3f}")
        print(f"  Mean rounds: {ms['final_round'].mean():.1f}")
        print(f"  Exploitation rate: {ms['exploitation'].mean():.0%}")

    ttc = df[df["experiment_type"] == "ttc_scaling"]
    if not ttc.empty:
        print(f"\nTTC-scaling experiments: {len(ttc)}")
        print(f"  Consensus rate: {ttc['consensus'].mean():.0%}")
        print(f"  Mean social welfare: {ttc['social_welfare'].mean():.3f}")

    print(f"\nAll plots saved to: {out_dir}/")


if __name__ == "__main__":
    main()
