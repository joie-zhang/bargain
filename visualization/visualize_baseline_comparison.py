#!/usr/bin/env python3
"""
=============================================================================
Baseline Comparison Visualizer — Game 2 (Diplomatic Treaty) & Game 3 (Co-Funding)
=============================================================================

Reads pre-computed CSV exports from lewis_slides_diplomacy.py and
lewis_slides_cofunding.py (or re-runs extraction live from raw JSON dirs)
and produces three families of comparison plots:

  1. Payoff gap by adversary × condition
     - Bar chart: (adversary_utility - baseline_utility) per condition bucket
     - One panel per game, grouped by adversary model on x-axis

  2. Order effects (weak_first vs strong_first)
     - Bar chart: payoff gap split by model_order, per adversary
     - Reveals whether "going first" advantages the baseline or adversary

  3. Cross-game comparison panel
     - Side-by-side: Game 2 (diplomatic treaty) vs Game 3 (cofunding)
     - X-axis: adversary/focal model Elo
     - Y-axis: payoff advantage over baseline/reference
     - Color: competition index bucket (CI₂ or CI₃)

Usage:
    # Use pre-computed CSVs (fastest — run lewis_slides_*.py first):
    python visualization/visualize_baseline_comparison.py \\
        --diplomacy-csv visualization/figures/diplomacy_lewis/diplomacy_lewis_data.csv \\
        --cofunding-csv visualization/figures/cofunding_lewis/cofunding_lewis_data.csv

    # Or point directly at experiment dirs (re-runs extraction):
    python visualization/visualize_baseline_comparison.py \\
        --diplomacy-dirs experiments/results/diplomacy_nano_vs_opus_20260315_120000 \\
                         experiments/results/diplomacy_nano_vs_weak_20260315_130000 \\
        --cofunding-dirs experiments/results/cofunding_nano_vs_opus_20260315_120000 \\
                         experiments/results/cofunding_nano_vs_weak_20260315_130000

    # Mixed: one from CSV, one from dir:
    python visualization/visualize_baseline_comparison.py \\
        --diplomacy-csv path/to/diplomacy_lewis_data.csv \\
        --cofunding-dirs experiments/results/cofunding_latest

What it creates:
    visualization/figures/baseline_comparison/
    ├── COMP_1_PAYOFF_GAP_GAME2.png          # payoff gap bars, Game 2
    ├── COMP_2_PAYOFF_GAP_GAME3.png          # payoff gap bars, Game 3
    ├── COMP_3_ORDER_EFFECT_GAME2.png        # order effect, Game 2
    ├── COMP_4_ORDER_EFFECT_GAME3.png        # order effect, Game 3
    └── COMP_5_CROSS_GAME_PANEL.png          # side-by-side cross-game

Configuration:
    - MIN_RUNS: Minimum runs per (adversary, condition) cell to include (default 1)
    - CI_BINS: Bin edges for competition index (default [0, 0.25, 0.5, 0.75, 1.01])

Dependencies:
    - matplotlib, numpy, pandas
    - Optional: lewis_slides_diplomacy.py and lewis_slides_cofunding.py (for live extraction)

=============================================================================
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["font.size"] = 11


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CI_BINS = [0.0, 0.25, 0.5, 0.75, 1.01]
CI_LABELS = ["CI=[0,0.25)", "CI=[0.25,0.5)", "CI=[0.5,0.75)", "CI=[0.75,1]"]
CI_COLORS = ["#2ecc71", "#f39c12", "#e67e22", "#e74c3c"]  # green → red

# Minimum runs per cell to include in plots
MIN_RUNS = 1

GAME2_LABEL = "Game 2: Diplomatic Treaty"
GAME3_LABEL = "Game 3: Co-Funding"


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")
    return df


def _extract_via_script(script_name: str, dirs: List[str], script_dir: Path) -> pd.DataFrame:
    """
    Run a lewis_slides_*.py script with --experiment-dir dirs and capture
    its CSV output from the default figures location.
    """
    script = script_dir / script_name
    if not script.exists():
        raise FileNotFoundError(f"Lewis slides script not found: {script}")

    cmd = [sys.executable, str(script)] + ["--experiment-dir"] + dirs
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"{script_name} failed:\n{result.stderr}")

    # Infer CSV path from script output
    for line in result.stdout.splitlines():
        if "data.csv" in line.lower() or "lewis_data" in line.lower():
            csv_path = line.strip().split()[-1]
            if os.path.exists(csv_path):
                return _load_csv(csv_path)

    # Fallback: look for CSV in figures dirs
    if "diplomacy" in script_name:
        csv_glob = list((script_dir / "figures" / "diplomacy_lewis").glob("*data*.csv"))
    else:
        csv_glob = list((script_dir / "figures" / "cofunding_lewis").glob("*data*.csv"))

    if csv_glob:
        return _load_csv(str(sorted(csv_glob)[-1]))
    raise FileNotFoundError(f"Could not find CSV output from {script_name}")


def _ci_bucket(ci: float) -> str:
    for i, (lo, hi) in enumerate(zip(CI_BINS[:-1], CI_BINS[1:])):
        if lo <= ci < hi:
            return CI_LABELS[i]
    return CI_LABELS[-1]


def _ci_bucket_color(label: str) -> str:
    if label in CI_LABELS:
        return CI_COLORS[CI_LABELS.index(label)]
    return "#888888"


# ---------------------------------------------------------------------------
# Normalise diplomacy DataFrame to a unified schema
# ---------------------------------------------------------------------------
# Unified columns used downstream:
#   game, adversary_model, adversary_elo, adversary_short,
#   baseline_model, baseline_elo, model_order, run_id,
#   competition_index, ci_bucket,
#   adversary_utility, baseline_utility, payoff_gap

def _normalise_diplomacy(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Rename to unified schema
    if "adversary_model" not in out.columns and "focal_model" in out.columns:
        # If someone ran a diplomacy script with focal/reference naming
        out = out.rename(columns={
            "focal_model": "adversary_model",
            "focal_elo": "adversary_elo",
            "focal_short": "adversary_short",
            "reference_model": "baseline_model",
            "reference_elo": "baseline_elo",
            "focal_utility": "adversary_utility",
            "reference_utility": "baseline_utility",
        })
    if "payoff_diff" in out.columns:
        out["payoff_gap"] = out["payoff_diff"]
    else:
        out["payoff_gap"] = out["adversary_utility"] - out["baseline_utility"]
    out["game"] = "Game 2"
    out["ci_bucket"] = out["competition_index"].apply(_ci_bucket)
    return out


def _normalise_cofunding(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Cofunding uses focal/reference naming — rename to adversary/baseline
    out = out.rename(columns={
        "focal_model": "adversary_model",
        "focal_elo": "adversary_elo",
        "focal_short": "adversary_short",
        "reference_model": "baseline_model",
        "reference_elo": "baseline_elo",
        "focal_utility": "adversary_utility",
        "reference_utility": "baseline_utility",
        "focal_advantage": "payoff_gap",
    })
    out["game"] = "Game 3"
    out["ci_bucket"] = out["competition_index"].apply(_ci_bucket)
    return out


# ---------------------------------------------------------------------------
# Plot 1 & 2: Payoff gap by adversary × condition
# ---------------------------------------------------------------------------

def plot_payoff_gap(
    df: pd.DataFrame,
    game_label: str,
    output_path: Path,
    min_runs: int = MIN_RUNS,
) -> None:
    """
    Grouped bar chart: payoff gap (adversary - baseline) per CI bucket,
    one cluster per adversary model, sorted by Elo.
    """
    # Aggregate: mean payoff gap per (adversary_model, ci_bucket)
    grp = (
        df.groupby(["adversary_model", "adversary_elo", "ci_bucket"])
        .agg(
            payoff_gap=("payoff_gap", "mean"),
            n_runs=("payoff_gap", "count"),
        )
        .reset_index()
    )
    grp = grp[grp["n_runs"] >= min_runs]

    if grp.empty:
        print(f"  Skipping payoff gap plot for {game_label}: no data after filtering")
        return

    # Sort adversaries by Elo
    model_order = (
        grp[["adversary_model", "adversary_elo"]]
        .drop_duplicates()
        .sort_values("adversary_elo")["adversary_model"]
        .tolist()
    )
    ci_levels = sorted(grp["ci_bucket"].unique(), key=lambda s: CI_LABELS.index(s) if s in CI_LABELS else 99)

    x = np.arange(len(model_order))
    width = 0.8 / max(len(ci_levels), 1)

    fig, ax = plt.subplots(figsize=(max(10, len(model_order) * 1.5), 6))
    for i, bucket in enumerate(ci_levels):
        sub = grp[grp["ci_bucket"] == bucket].set_index("adversary_model")
        vals = [sub.loc[m, "payoff_gap"] if m in sub.index else float("nan") for m in model_order]
        offset = (i - len(ci_levels) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=bucket, color=_ci_bucket_color(bucket), alpha=0.85)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(model_order, rotation=30, ha="right", fontsize=9)
    ax.set_xlabel("Adversary Model (sorted by Elo)", fontsize=11)
    ax.set_ylabel("Payoff Gap (adversary − baseline)", fontsize=11)
    ax.set_title(f"Payoff Gap by Adversary & Competition Index\n{game_label}", fontsize=12)
    ax.legend(title="CI bucket", bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Plot 3 & 4: Order effects
# ---------------------------------------------------------------------------

def plot_order_effect(
    df: pd.DataFrame,
    game_label: str,
    output_path: Path,
    min_runs: int = MIN_RUNS,
) -> None:
    """
    Grouped bars: payoff gap for weak_first vs strong_first,
    per adversary model, sorted by Elo. Reveals whether order matters.
    """
    if "model_order" not in df.columns:
        print(f"  Skipping order effect for {game_label}: no model_order column")
        return

    grp = (
        df.groupby(["adversary_model", "adversary_elo", "model_order"])
        .agg(
            payoff_gap=("payoff_gap", "mean"),
            n_runs=("payoff_gap", "count"),
        )
        .reset_index()
    )
    grp = grp[grp["n_runs"] >= min_runs]

    if grp.empty:
        print(f"  Skipping order effect for {game_label}: no data after filtering")
        return

    model_order_list = (
        grp[["adversary_model", "adversary_elo"]]
        .drop_duplicates()
        .sort_values("adversary_elo")["adversary_model"]
        .tolist()
    )
    orders = sorted(grp["model_order"].unique())
    order_colors = {"weak_first": "#3498db", "strong_first": "#e74c3c"}

    x = np.arange(len(model_order_list))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(10, len(model_order_list) * 1.5), 6))
    for i, order in enumerate(orders):
        sub = grp[grp["model_order"] == order].set_index("adversary_model")
        vals = [sub.loc[m, "payoff_gap"] if m in sub.index else float("nan") for m in model_order_list]
        offset = (i - len(orders) / 2 + 0.5) * width
        color = order_colors.get(order, f"C{i}")
        ax.bar(x + offset, vals, width, label=order, color=color, alpha=0.85)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(model_order_list, rotation=30, ha="right", fontsize=9)
    ax.set_xlabel("Adversary Model (sorted by Elo)", fontsize=11)
    ax.set_ylabel("Payoff Gap (adversary − baseline)", fontsize=11)
    ax.set_title(f"Order Effect: weak_first vs strong_first\n{game_label}", fontsize=12)
    ax.legend(title="Model order")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Plot 5: Cross-game comparison panel
# ---------------------------------------------------------------------------

def plot_cross_game_panel(
    df2: pd.DataFrame,
    df3: pd.DataFrame,
    output_path: Path,
    min_runs: int = MIN_RUNS,
) -> None:
    """
    Side-by-side scatter/line: payoff advantage vs adversary Elo,
    colored by CI bucket. Left = Game 2, Right = Game 3.
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=False)

    for ax, df, game_label in zip(axes, [df2, df3], [GAME2_LABEL, GAME3_LABEL]):
        grp = (
            df.groupby(["adversary_model", "adversary_elo", "ci_bucket"])
            .agg(
                payoff_gap=("payoff_gap", "mean"),
                n_runs=("payoff_gap", "count"),
            )
            .reset_index()
        )
        grp = grp[grp["n_runs"] >= min_runs]

        if grp.empty:
            ax.set_title(f"{game_label}\n(no data)")
            continue

        ci_levels = sorted(grp["ci_bucket"].unique(), key=lambda s: CI_LABELS.index(s) if s in CI_LABELS else 99)
        for bucket in ci_levels:
            sub = grp[grp["ci_bucket"] == bucket].sort_values("adversary_elo")
            ax.plot(
                sub["adversary_elo"],
                sub["payoff_gap"],
                marker="o",
                markersize=6,
                linewidth=1.5,
                label=bucket,
                color=_ci_bucket_color(bucket),
                alpha=0.85,
            )

        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Adversary Model Elo", fontsize=11)
        ax.set_ylabel("Payoff Gap (adversary − baseline)", fontsize=11)
        ax.set_title(game_label, fontsize=12)
        ax.legend(title="CI bucket", fontsize=9)

    fig.suptitle("Cross-Game Payoff Advantage vs Model Strength", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Baseline comparison plots for Game 2 and Game 3 experiments."
    )
    parser.add_argument(
        "--diplomacy-csv",
        default=None,
        metavar="CSV",
        help="Pre-computed CSV from lewis_slides_diplomacy.py.",
    )
    parser.add_argument(
        "--cofunding-csv",
        default=None,
        metavar="CSV",
        help="Pre-computed CSV from lewis_slides_cofunding.py.",
    )
    parser.add_argument(
        "--diplomacy-dirs",
        nargs="+",
        default=None,
        metavar="DIR",
        help="Raw experiment result dirs for Game 2 (triggers live extraction).",
    )
    parser.add_argument(
        "--cofunding-dirs",
        nargs="+",
        default=None,
        metavar="DIR",
        help="Raw experiment result dirs for Game 3 (triggers live extraction).",
    )
    parser.add_argument(
        "--output-dir",
        default="figures/baseline_comparison",
        help="Output directory relative to this script's location.",
    )
    parser.add_argument(
        "--min-runs",
        type=int,
        default=MIN_RUNS,
        help="Minimum runs per (adversary, condition) cell to include.",
    )
    parser.add_argument(
        "--skip-cross-game",
        action="store_true",
        help="Skip the cross-game comparison panel (requires both games).",
    )
    return parser.parse_args()


def _resolve_df(
    csv_path: Optional[str],
    dirs: Optional[List[str]],
    script_name: str,
    script_dir: Path,
    normalise_fn,
) -> Optional[pd.DataFrame]:
    if csv_path:
        return normalise_fn(_load_csv(csv_path))
    if dirs:
        raw = _extract_via_script(script_name, dirs, script_dir)
        return normalise_fn(raw)
    return None


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    output_dir = (
        (script_dir / args.output_dir)
        if not Path(args.output_dir).is_absolute()
        else Path(args.output_dir)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    df2 = _resolve_df(
        args.diplomacy_csv,
        args.diplomacy_dirs,
        "lewis_slides_diplomacy.py",
        script_dir,
        _normalise_diplomacy,
    )
    df3 = _resolve_df(
        args.cofunding_csv,
        args.cofunding_dirs,
        "lewis_slides_cofunding.py",
        script_dir,
        _normalise_cofunding,
    )

    if df2 is None and df3 is None:
        raise ValueError(
            "No data provided. Supply at least one of: "
            "--diplomacy-csv, --diplomacy-dirs, --cofunding-csv, --cofunding-dirs"
        )

    # ---- Plot 1: Payoff gap — Game 2 ----
    if df2 is not None:
        plot_payoff_gap(
            df2, GAME2_LABEL,
            output_dir / "COMP_1_PAYOFF_GAP_GAME2.png",
            min_runs=args.min_runs,
        )
        plot_order_effect(
            df2, GAME2_LABEL,
            output_dir / "COMP_3_ORDER_EFFECT_GAME2.png",
            min_runs=args.min_runs,
        )

    # ---- Plot 2: Payoff gap — Game 3 ----
    if df3 is not None:
        plot_payoff_gap(
            df3, GAME3_LABEL,
            output_dir / "COMP_2_PAYOFF_GAP_GAME3.png",
            min_runs=args.min_runs,
        )
        plot_order_effect(
            df3, GAME3_LABEL,
            output_dir / "COMP_4_ORDER_EFFECT_GAME3.png",
            min_runs=args.min_runs,
        )

    # ---- Plot 5: Cross-game panel ----
    if not args.skip_cross_game and df2 is not None and df3 is not None:
        plot_cross_game_panel(
            df2, df3,
            output_dir / "COMP_5_CROSS_GAME_PANEL.png",
            min_runs=args.min_runs,
        )

    print(f"\nDone. Plots written to {output_dir}")


if __name__ == "__main__":
    main()
