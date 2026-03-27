#!/usr/bin/env python3
"""
=============================================================================
Co-Funding Detailed Analysis Plots
=============================================================================

Reads raw JSON experiment results and produces four figure sets focused on
resource allocation efficiency and free-riding behavior:

  Figure 1 — Efficiency overview (3-panel):
      (A) Projects funded / total projects
      (B) $ raised (funded project costs) / total project costs
      (C) Budget spent / total budget available

  Figure 2 — Per-agent contribution breakdown:
      Grouped bars: each agent's actual spend vs fair-share spend,
      split by condition (alpha × sigma) and model pair.

  Figure 3 — Free-rider scatter:
      x = contribution / own budget  (0→spent nothing, 1→spent all)
      y = utility received
      Each point = one agent in one run. Reference lines at equal split.

  Figure 4 — Free-rider index heatmap:
      free_rider_index = (utility / equal_split_utility) − (spend / fair_share_spend)
      Positive = benefited more than contributed; negative = overpaid relative to gains.
      One cell per (model, condition).

Usage:
    python visualization/visualize_cofunding_detail.py
    python visualization/visualize_cofunding_detail.py \\
        --experiment-dir experiments/results/cofunding_20260316_071259 \\
                         experiments/results/cofunding_20260316_071418
    python visualization/visualize_cofunding_detail.py \\
        --experiment-dir experiments/results/cofunding_latest

What it creates:
    visualization/figures/cofunding_detail/
    ├── DETAIL_1_EFFICIENCY_OVERVIEW.png
    ├── DETAIL_2_PER_AGENT_CONTRIBUTIONS.png
    ├── DETAIL_3_FREERIDER_SCATTER.png
    ├── DETAIL_4_FREERIDER_INDEX_HEATMAP.png
    └── cofunding_detail_data.csv

Configuration:
    - CONDITION_ORDER: display order for (alpha, sigma) conditions
    - MODEL_SHORT: short display names for model keys

Dependencies:
    - matplotlib, numpy, pandas, seaborn

=============================================================================
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd


plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["font.size"] = 11


# ---------------------------------------------------------------------------
# Model display names
# ---------------------------------------------------------------------------

MODEL_SHORT: Dict[str, str] = {
    "gpt-5-nano": "GPT-5 Nano",
    "gpt-5.2-chat": "GPT-5.2",
    "claude-opus-4-6": "Opus 4.6",
    "claude-opus-4-5-thinking-32k": "Opus 4.5T",
    "amazon-nova-micro-v1.0": "Nova Micro",
    "gemini-3-pro": "Gemini 3 Pro",
    "gpt-4.1": "GPT-4.1",
    "gpt-4.1-mini": "GPT-4.1-mini",
    "gpt-4o": "GPT-4o",
    "gpt-4o-mini": "GPT-4o-mini",
    "o3-mini-high": "o3-mini-H",
    "o3": "o3",
    "o4-mini": "o4-mini",
}

CONDITION_COLORS: Dict[Tuple[float, float], str] = {
    (0.0, 0.3): "#e74c3c",   # competitive + scarce → red
    (0.0, 1.0): "#e67e22",   # competitive + abundant → orange
    (1.0, 0.3): "#3498db",   # cooperative + scarce → blue
    (1.0, 1.0): "#2ecc71",   # cooperative + abundant → green
}


def _short(model: str) -> str:
    return MODEL_SHORT.get(model, model)


def _cond_label(alpha: float, sigma: float) -> str:
    return f"α={alpha:.1f}, σ={sigma:.1f}"


def _cond_color(alpha: float, sigma: float) -> str:
    key = (round(alpha, 1), round(sigma, 1))
    return CONDITION_COLORS.get(key, "#888888")


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------

def _parse_alpha_sigma(dirname: str) -> Tuple[Optional[float], Optional[float]]:
    try:
        alpha_str = dirname.replace("alpha_", "").split("_sigma_")[0]
        sigma_str = dirname.split("_sigma_")[1]

        def _val(s: str) -> float:
            parts = s.split("_")
            return float(parts[0]) + (
                float(parts[1]) / (10 ** len(parts[1])) if len(parts) > 1 and parts[1] else 0.0
            )

        return _val(alpha_str), _val(sigma_str)
    except Exception:
        return None, None


def _get_final_pledges(conversation_logs: list, final_round: int) -> Dict[str, List[float]]:
    """Extract last-round pledge contributions per agent."""
    pledges = {}
    for entry in conversation_logs:
        if (
            entry.get("phase") == "pledge_submission"
            and entry.get("round") == final_round
            and "pledge" in entry
        ):
            agent = entry["from"]
            contribs = entry["pledge"].get("contributions", [])
            pledges[agent] = contribs
    return pledges


def extract_data(experiment_dirs: List[str]) -> pd.DataFrame:
    """
    Walk experiment directories and extract per-agent rows.
    Each row = one agent in one run.
    """
    rows = []

    for experiment_dir in experiment_dirs:
        model_scale_dir = os.path.join(experiment_dir, "model_scale")
        if not os.path.isdir(model_scale_dir):
            print(f"  WARNING: No model_scale dir at {model_scale_dir}, skipping.")
            continue

        for pair_name in sorted(os.listdir(model_scale_dir)):
            pair_path = os.path.join(model_scale_dir, pair_name)
            if not os.path.isdir(pair_path) or "_vs_" not in pair_name:
                continue
            model1, model2 = pair_name.split("_vs_", 1)

            for order_name in sorted(os.listdir(pair_path)):
                order_path = os.path.join(pair_path, order_name)
                if not os.path.isdir(order_path) or order_name not in ("weak_first", "strong_first"):
                    continue

                for cond_name in sorted(os.listdir(order_path)):
                    cond_path = os.path.join(order_path, cond_name)
                    if not os.path.isdir(cond_path):
                        continue
                    alpha, sigma = _parse_alpha_sigma(cond_name)
                    if alpha is None:
                        continue

                    # Collect result files (flat structure)
                    result_files = []
                    for fname in sorted(os.listdir(cond_path)):
                        if fname.startswith("run_") and fname.endswith("_experiment_results.json"):
                            result_files.append((os.path.join(cond_path, fname), fname.replace("_experiment_results.json", "")))

                    for result_file, run_id in result_files:
                        with open(result_file) as f:
                            result = json.load(f)

                        cfg = result["config"]
                        items = cfg["items"]
                        m_projects = len(items)
                        total_cost = sum(i["cost"] for i in items)
                        total_budget = cfg.get("total_budget", 0.0)
                        n_agents = cfg.get("n_agents", 2)
                        agent_budgets = cfg.get("agent_budgets", {})

                        funded_indices = result.get("final_allocation", [])
                        n_funded = len(funded_indices)
                        funded_cost = sum(items[i]["cost"] for i in funded_indices if i < m_projects)

                        final_round = result.get("final_round", 5)
                        logs = result.get("conversation_logs", [])
                        final_pledges = _get_final_pledges(logs, final_round)

                        total_spent = sum(sum(v) for v in final_pledges.values())
                        final_utilities = result.get("final_utilities", {})
                        agent_perf = result.get("agent_performance", {})

                        # Game-level metrics
                        projects_funded_rate = n_funded / m_projects
                        dollar_coverage = funded_cost / total_cost if total_cost > 0 else 0.0
                        budget_utilization = total_spent / total_budget if total_budget > 0 else 0.0
                        social_welfare = sum(final_utilities.values())
                        equal_split_utility = social_welfare / n_agents if n_agents > 0 else 0.0
                        fair_share_budget = total_budget / n_agents if n_agents > 0 else 0.0

                        for agent_id, perf_info in agent_perf.items():
                            model = perf_info.get("model", "unknown")
                            utility = final_utilities.get(agent_id, 0.0)
                            own_budget = agent_budgets.get(agent_id, fair_share_budget)
                            contributions = final_pledges.get(agent_id, [])
                            spend = sum(contributions) if contributions else 0.0

                            # Contribution relative to own budget
                            contrib_frac = spend / own_budget if own_budget > 0 else 0.0
                            # Contribution relative to fair-share cost burden
                            fair_share_cost = total_cost / n_agents
                            contrib_vs_fair_cost = spend / fair_share_cost if fair_share_cost > 0 else 0.0
                            # Free-rider index: utility advantage relative to contribution shortfall
                            # +ve = got more utility than contributed proportionally
                            utility_frac = utility / equal_split_utility if equal_split_utility > 0 else 0.0
                            spend_frac = spend / fair_share_budget if fair_share_budget > 0 else 0.0
                            free_rider_index = utility_frac - spend_frac

                            rows.append({
                                "experiment_dir": experiment_dir,
                                "pair": pair_name,
                                "model1": model1,
                                "model2": model2,
                                "order": order_name,
                                "alpha": round(alpha, 4),
                                "sigma": round(sigma, 4),
                                "cond_label": _cond_label(alpha, sigma),
                                "run_id": run_id,
                                "agent_id": agent_id,
                                "model": model,
                                "model_short": _short(model),
                                "pair_short": f"{_short(model1)} vs {_short(model2)}",
                                # Game-level
                                "n_funded": n_funded,
                                "m_projects": m_projects,
                                "projects_funded_rate": projects_funded_rate,
                                "funded_cost": funded_cost,
                                "total_cost": total_cost,
                                "dollar_coverage": dollar_coverage,
                                "total_spent": total_spent,
                                "total_budget": total_budget,
                                "budget_utilization": budget_utilization,
                                "social_welfare": social_welfare,
                                # Agent-level
                                "utility": utility,
                                "own_budget": own_budget,
                                "spend": spend,
                                "contrib_frac": contrib_frac,          # spend / own_budget
                                "contrib_vs_fair_cost": contrib_vs_fair_cost,  # spend / fair_cost_share
                                "fair_share_budget": fair_share_budget,
                                "fair_share_cost": fair_share_cost,
                                "free_rider_index": free_rider_index,
                            })

    df = pd.DataFrame(rows)
    if not df.empty:
        print(f"Extracted {len(df)} agent-run rows from {df['pair'].nunique()} pairs, "
              f"{df['cond_label'].nunique()} conditions")
    return df


# ---------------------------------------------------------------------------
# Figure 1: Efficiency overview (3-panel bar chart)
# ---------------------------------------------------------------------------

def plot_efficiency_overview(df: pd.DataFrame, output_path: Path) -> None:
    """
    3-panel bar chart: efficiency metrics by condition, grouped by model pair.
    Uses one row per run (game-level metrics), deduplicating by (pair, order, alpha, sigma, run_id).
    """
    # One row per game (not per agent)
    game_df = df.drop_duplicates(subset=["pair", "order", "alpha", "sigma", "run_id"])

    metrics = [
        ("projects_funded_rate", "Projects Funded\n(fraction of total)", "A"),
        ("dollar_coverage",      "Cost Coverage\n(funded $ / total $)", "B"),
        ("budget_utilization",   "Budget Utilization\n(spent / available)", "C"),
    ]

    cond_labels = sorted(game_df["cond_label"].unique(),
                         key=lambda s: (float(s.split(",")[0].split("=")[1]),
                                        float(s.split(",")[1].split("=")[1])))
    pair_labels = sorted(game_df["pair_short"].unique())
    n_conds = len(cond_labels)
    x = np.arange(n_conds)
    width = 0.8 / max(len(pair_labels), 1)

    pair_colors = plt.cm.Set2(np.linspace(0, 1, max(len(pair_labels), 1)))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, (col, ylabel, panel) in zip(axes, metrics):
        for i, pair in enumerate(pair_labels):
            sub = game_df[game_df["pair_short"] == pair]
            vals = []
            for cond in cond_labels:
                cell = sub[sub["cond_label"] == cond][col]
                vals.append(cell.mean() if len(cell) > 0 else float("nan"))
            offset = (i - len(pair_labels) / 2 + 0.5) * width
            ax.bar(x + offset, vals, width, label=pair, color=pair_colors[i], alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(cond_labels, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_ylim(0, 1.15)
        ax.axhline(1.0, color="black", linewidth=0.7, linestyle="--", alpha=0.5)
        ax.set_title(f"({panel})", fontsize=11, loc="left")

    axes[0].legend(title="Model pair", bbox_to_anchor=(0.0, 1.02), loc="lower left", fontsize=8)
    fig.suptitle("Co-Funding Efficiency: Funded Projects, Cost Coverage, and Budget Use",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Figure 2: Per-agent contribution breakdown
# ---------------------------------------------------------------------------

def plot_agent_contributions(df: pd.DataFrame, output_path: Path) -> None:
    """
    Grouped bar chart: each agent's actual spend vs their fair-share budget,
    grouped by condition and model pair. Shows who contributes how much.
    """
    cond_labels = sorted(df["cond_label"].unique(),
                         key=lambda s: (float(s.split(",")[0].split("=")[1]),
                                        float(s.split(",")[1].split("=")[1])))
    pair_labels = sorted(df["pair_short"].unique())

    n_pairs = len(pair_labels)
    fig, axes = plt.subplots(1, max(n_pairs, 1), figsize=(7 * n_pairs, 6), squeeze=False)

    for col_idx, pair in enumerate(pair_labels):
        ax = axes[0][col_idx]
        pair_df = df[df["pair_short"] == pair]

        model_order_list = sorted(pair_df["model_short"].unique())
        n_models = len(model_order_list)
        cond_colors = [_cond_color(
            float(c.split(",")[0].split("=")[1]),
            float(c.split(",")[1].split("=")[1])
        ) for c in cond_labels]

        x = np.arange(n_models)
        width = 0.8 / max(len(cond_labels), 1)

        for ci, cond in enumerate(cond_labels):
            cond_df = pair_df[pair_df["cond_label"] == cond]
            spend_vals = []
            fair_vals = []
            for model_s in model_order_list:
                cell = cond_df[cond_df["model_short"] == model_s]
                spend_vals.append(cell["spend"].mean() if len(cell) > 0 else float("nan"))
                fair_vals.append(cell["fair_share_budget"].mean() if len(cell) > 0 else float("nan"))

            offset = (ci - len(cond_labels) / 2 + 0.5) * width
            ax.bar(x + offset, spend_vals, width,
                   label=cond, color=cond_colors[ci], alpha=0.85)
            # Overlay fair-share as a horizontal line segment per group
            for xi, (fv, sv) in enumerate(zip(fair_vals, spend_vals)):
                if not np.isnan(fv):
                    ax.plot([xi + offset - width/2, xi + offset + width/2], [fv, fv],
                            color="black", linewidth=1.5, linestyle="--")

        ax.set_xticks(x)
        ax.set_xticklabels(model_order_list, rotation=15, ha="right", fontsize=10)
        ax.set_ylabel("Budget Spent ($)", fontsize=10)
        ax.set_title(pair, fontsize=11)
        ax.legend(title="Condition", fontsize=8)

    # Shared legend note
    fair_patch = mpatches.Patch(facecolor="none", edgecolor="black", linestyle="--", linewidth=1.5,
                                label="-- fair-share budget")
    axes[0][-1].legend(handles=axes[0][-1].get_legend_handles_labels()[0] + [fair_patch],
                       labels=axes[0][-1].get_legend_handles_labels()[1] + ["-- fair-share"],
                       title="Condition", fontsize=8)

    fig.suptitle("Per-Agent Budget Spent vs Fair Share (dashed line = own budget / n_agents)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Figure 3: Free-rider scatter
# ---------------------------------------------------------------------------

def plot_freerider_scatter(df: pd.DataFrame, output_path: Path) -> None:
    """
    Scatter: x = fraction of own budget contributed, y = utility received.
    Each point = one agent in one run. Reference lines at equal-split values.
    """
    model_list = sorted(df["model_short"].unique())
    cond_list = sorted(df["cond_label"].unique(),
                       key=lambda s: (float(s.split(",")[0].split("=")[1]),
                                      float(s.split(",")[1].split("=")[1])))

    markers = ["o", "s", "^", "D", "v", "P", "*", "X"]
    model_marker = {m: markers[i % len(markers)] for i, m in enumerate(model_list)}
    cond_color = {c: _cond_color(
        float(c.split(",")[0].split("=")[1]),
        float(c.split(",")[1].split("=")[1])
    ) for c in cond_list}

    fig, ax = plt.subplots(figsize=(9, 7))

    for cond in cond_list:
        for model_s in model_list:
            sub = df[(df["cond_label"] == cond) & (df["model_short"] == model_s)]
            if sub.empty:
                continue
            ax.scatter(
                sub["contrib_frac"],
                sub["utility"],
                c=cond_color[cond],
                marker=model_marker[model_s],
                s=100,
                alpha=0.85,
                edgecolors="white",
                linewidths=0.5,
                label=f"{model_s} / {cond}",
                zorder=3,
            )

    # Reference lines
    mean_fair_contrib = df["fair_share_budget"].mean() / df["own_budget"].mean() if df["own_budget"].mean() > 0 else 0.5
    ax.axvline(mean_fair_contrib, color="gray", linewidth=1.0, linestyle="--",
               label=f"Equal split (x={mean_fair_contrib:.2f})")
    # Equal-split utility line: social_welfare / n_agents
    mean_equal_util = df.groupby(["pair", "order", "alpha", "sigma", "run_id"])["utility"].mean().mean()
    ax.axhline(mean_equal_util, color="gray", linewidth=1.0, linestyle=":",
               label=f"Equal utility (y={mean_equal_util:.1f})")

    # Shade free-rider quadrant: low contribution, high utility
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.fill_betweenx(
        [mean_equal_util, max(ylim[1], mean_equal_util * 1.5)],
        0, mean_fair_contrib,
        alpha=0.06, color="#e74c3c", label="Free-rider zone"
    )

    ax.set_xlabel("Contribution / Own Budget  (0 = spent nothing, 1 = spent all)", fontsize=11)
    ax.set_ylabel("Utility Received", fontsize=11)
    ax.set_title("Free-Rider Scatter: Contribution vs Utility\n"
                 "Upper-left = free-rider zone (low contribution, high utility)",
                 fontsize=12)

    # Build compact legend
    cond_handles = [mpatches.Patch(facecolor=cond_color[c], label=c) for c in cond_list]
    model_handles = [
        plt.Line2D([0], [0], marker=model_marker[m], color="gray", linestyle="none",
                   markersize=8, label=m)
        for m in model_list
    ]
    ref_handles = [
        plt.Line2D([0], [0], color="gray", linestyle="--", label="Equal split (contribution)"),
        plt.Line2D([0], [0], color="gray", linestyle=":", label="Equal utility"),
        mpatches.Patch(facecolor="#e74c3c", alpha=0.12, label="Free-rider zone"),
    ]
    ax.legend(
        handles=cond_handles + model_handles + ref_handles,
        fontsize=8,
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Figure 4: Free-rider index heatmap
# ---------------------------------------------------------------------------

def plot_freerider_heatmap(df: pd.DataFrame, output_path: Path) -> None:
    """
    Heatmap: rows = model, columns = condition, values = mean free_rider_index.
    free_rider_index = (utility / equal_split_utility) - (spend / fair_share_budget)
    Positive = received more utility than their contribution share warrants.
    """
    cond_labels = sorted(df["cond_label"].unique(),
                         key=lambda s: (float(s.split(",")[0].split("=")[1]),
                                        float(s.split(",")[1].split("=")[1])))
    model_list = sorted(df["model_short"].unique())

    pivot = df.pivot_table(
        index="model_short",
        columns="cond_label",
        values="free_rider_index",
        aggfunc="mean",
    ).reindex(index=model_list, columns=cond_labels)

    n_rows, n_cols = pivot.shape
    fig, ax = plt.subplots(figsize=(max(6, n_cols * 2), max(4, n_rows * 1.2)))

    vmax = max(abs(pivot.values[~np.isnan(pivot.values)]).max(), 0.5) if not pivot.empty else 1.0
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn_r", vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(cond_labels, rotation=20, ha="right", fontsize=10)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(model_list, fontsize=10)

    # Annotate cells
    for i in range(n_rows):
        for j in range(n_cols):
            val = pivot.values[i, j]
            if not np.isnan(val):
                text_color = "white" if abs(val) > vmax * 0.5 else "black"
                ax.text(j, i, f"{val:+.2f}", ha="center", va="center",
                        fontsize=10, color=text_color, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("Free-Rider Index\n(+) = free-rider  (−) = over-contributor", fontsize=9)

    ax.set_title(
        "Free-Rider Index by Model and Condition\n"
        r"= $\frac{utility}{equal\text{-}split\;utility}$ − $\frac{spend}{fair\text{-}share\;budget}$",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _find_experiment_dirs(script_dir: Path) -> List[str]:
    results_base = script_dir.parent / "experiments" / "results"
    dirs = sorted(d for d in os.listdir(results_base)
                  if d.startswith("cofunding_") and d not in ("cofunding_latest", "cofunding_smoke_test"))
    if not dirs:
        latest = results_base / "cofunding_latest"
        if latest.exists():
            return [str(latest)]
        raise FileNotFoundError("No cofunding_* experiment directories found.")
    # Return the two most recent
    return [str(results_base / d) for d in dirs[-2:]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detailed co-funding efficiency and free-rider analysis plots."
    )
    parser.add_argument(
        "--experiment-dir",
        nargs="+",
        default=None,
        metavar="DIR",
        help="Experiment result directories (default: two most recent cofunding_* dirs).",
    )
    parser.add_argument(
        "--output-dir",
        default="figures/cofunding_detail",
        help="Output directory relative to this script's location.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    output_dir = (
        (script_dir / args.output_dir)
        if not Path(args.output_dir).is_absolute()
        else Path(args.output_dir)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    experiment_dirs = args.experiment_dir or _find_experiment_dirs(script_dir)
    print(f"Experiment dir(s): {experiment_dirs}")

    df = extract_data(experiment_dirs)
    if df.empty:
        raise ValueError("No data extracted. Check experiment directory paths.")

    # Save raw data
    csv_path = output_dir / "cofunding_detail_data.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Saved CSV: {csv_path}")

    print("\nGenerating Figure 1: Efficiency overview...")
    plot_efficiency_overview(df, output_dir / "DETAIL_1_EFFICIENCY_OVERVIEW.png")

    print("Generating Figure 2: Per-agent contributions...")
    plot_agent_contributions(df, output_dir / "DETAIL_2_PER_AGENT_CONTRIBUTIONS.png")

    print("Generating Figure 3: Free-rider scatter...")
    plot_freerider_scatter(df, output_dir / "DETAIL_3_FREERIDER_SCATTER.png")

    print("Generating Figure 4: Free-rider index heatmap...")
    plot_freerider_heatmap(df, output_dir / "DETAIL_4_FREERIDER_INDEX_HEATMAP.png")

    print(f"\nDone. 4 plots written to {output_dir}")


if __name__ == "__main__":
    main()
