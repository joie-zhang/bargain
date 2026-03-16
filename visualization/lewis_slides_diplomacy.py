#!/usr/bin/env python3
"""
=============================================================================
Lewis Slides Script — Game 2 (Diplomatic Treaty)
=============================================================================

Mirrors the structure of lewis_slides_script_mar_1.py (Game 1) but adapted
for Game 2's two-parameter competition space: rho (preference correlation)
and theta (interest overlap).

Reads raw JSON results directly from the experiment directory tree and
produces two families of "payoff vs adversary Elo" plots:

  2-VARIABLE version  — one line per rho level (theta averaged out)
  1-VARIABLE version  — one line per CI bucket, where CI = theta*(1-rho)/2

Usage:
    python visualization/lewis_slides_diplomacy.py
    python visualization/lewis_slides_diplomacy.py --experiment-dir experiments/results/diplomacy_20260223_032204
    python visualization/lewis_slides_diplomacy.py --param rho        # only rho-bucketed plots
    python visualization/lewis_slides_diplomacy.py --param ci         # only CI-bucketed plots
    python visualization/lewis_slides_diplomacy.py --make-all-variants

What it creates:
    visualization/figures/diplomacy_lewis/
    ├── MAIN_PLOT_1_BASELINE_PAYOFF_RHO.png      # baseline (GPT-5-nano) utility, lines = rho
    ├── MAIN_PLOT_2_ADVERSARY_PAYOFF_RHO.png     # adversary utility, lines = rho
    ├── MAIN_PLOT_3_BASELINE_PAYOFF_CI.png       # baseline utility, lines = CI bucket
    ├── MAIN_PLOT_4_ADVERSARY_PAYOFF_CI.png      # adversary utility, lines = CI bucket
    ├── diplomacy_lewis_data.csv                 # flat per-run data
    └── lewis_diplomacy_report.md                # coverage report

Competition index motivation:
    rho in [-1, 1]: preference correlation.
        rho = 1  → identical preferences (pure coordination)
        rho = -1 → perfectly opposed preferences (pure conflict)
    theta in [0, 1]: interest overlap / issue contestedness.
        theta = 0 → each issue is privately valued by one agent (no structural overlap)
        theta = 1 → both agents care equally about every issue
    CI = theta * (1 - rho) / 2, in [0, 1].
        CI = 0 → no competition (either identical prefs or no overlap)
        CI = 1 → maximal competition (fully opposed AND fully overlapping interests)

Dependencies:
    - matplotlib, numpy, pandas

=============================================================================
"""

from __future__ import annotations

import argparse
import os
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 11


# ---------------------------------------------------------------------------
# Model metadata
# ---------------------------------------------------------------------------

MODEL_ELO: Dict[str, int] = {
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

MODEL_SHORT: Dict[str, str] = {
    "gemini-3-pro": "Gemini 3 Pro",
    "claude-opus-4-5-thinking-32k": "Claude Opus",
    "claude-sonnet-4-5": "Sonnet 4.5",
    "gpt-5.2-high": "GPT-5.2",
    "grok-4": "Grok-4",
    "claude-haiku-4-5": "Haiku 4.5",
    "deepseek-r1": "DeepSeek-R1",
    "o3-mini-high": "O3-mini",
    "gpt-4o": "GPT-4o",
    "gpt-5-nano": "GPT-5-nano",
    "QwQ-32B": "QwQ-32B",
    "amazon-nova-micro": "Nova Micro",
    "gpt-3.5-turbo-0125": "GPT-3.5",
}

BASELINE_MODEL = "gpt-5-nano"

# CI bucket labels (nearest 0.25)
CI_BUCKET_LABELS: Dict[float, str] = {
    0.00: "CI=0.00 (cooperative)",
    0.25: "CI=0.25",
    0.50: "CI=0.50",
    0.75: "CI=0.75",
    1.00: "CI=1.00 (competitive)",
}

RHO_LABELS: Dict[float, str] = {
    -1.0: "ρ=−1 (max conflict)",
    -0.5: "ρ=−0.5",
     0.0: "ρ=0 (neutral)",
     0.5: "ρ=0.5",
     1.0: "ρ=1 (cooperative)",
}

MEAN_COLS = {
    "baseline_utility",
    "adversary_utility",
    "social_welfare",
    "util_ratio",
    "payoff_diff",
    "consensus_reached",
    "final_round",
    "competition_index",
}


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------

def _parse_rho_theta_from_dirname(dirname: str) -> Tuple[Optional[float], Optional[float]]:
    """Parse rho and theta from a directory name like 'rho_n1_0_theta_0_25'."""
    try:
        rho_part = dirname.split("_theta_")[0].replace("rho_", "")
        theta_part = dirname.split("_theta_")[1]

        def _parse_val(s: str) -> float:
            negative = s.startswith("n")
            s = s.lstrip("n")
            parts = s.split("_")
            val = float(parts[0]) + (float(parts[1]) / (10 ** len(parts[1])) if len(parts) > 1 and parts[1] else 0.0)
            return -val if negative else val

        return _parse_val(rho_part), _parse_val(theta_part)
    except Exception:
        return None, None


def extract_results(experiment_dir: str) -> pd.DataFrame:
    """
    Walk the experiment directory tree and collect one row per run.

    Directory structure expected:
        experiment_dir/model_scale/
            [model1]_vs_[model2]/
                {weak_first,strong_first}/
                    rho_*_theta_*/
                        run_1/, run_2/, run_3/
                            run_N_experiment_results.json
    """
    model_scale_dir = os.path.join(experiment_dir, "model_scale")
    if not os.path.isdir(model_scale_dir):
        raise FileNotFoundError(f"No model_scale directory at {model_scale_dir}")

    rows = []
    missing = 0

    for pair_name in sorted(os.listdir(model_scale_dir)):
        pair_path = os.path.join(model_scale_dir, pair_name)
        if not os.path.isdir(pair_path):
            continue

        # Extract model names from directory name
        if "_vs_" not in pair_name:
            continue
        model1, model2 = pair_name.split("_vs_", 1)

        # Skip self-play
        if model1 == model2:
            continue

        adversary_model = model2
        adversary_elo = MODEL_ELO.get(adversary_model, 0)

        for order_name in sorted(os.listdir(pair_path)):
            order_path = os.path.join(pair_path, order_name)
            if not os.path.isdir(order_path):
                continue
            if order_name not in ("weak_first", "strong_first"):
                continue

            for cond_name in sorted(os.listdir(order_path)):
                cond_path = os.path.join(order_path, cond_name)
                if not os.path.isdir(cond_path):
                    continue

                rho, theta = _parse_rho_theta_from_dirname(cond_name)
                if rho is None:
                    continue

                # Iterate over run subdirectories
                for entry in sorted(os.listdir(cond_path)):
                    run_path = os.path.join(cond_path, entry)
                    if not os.path.isdir(run_path):
                        continue
                    run_id = entry  # e.g. "run_1", "run_2", "run_3"

                    # All run dirs use the fixed filename "run_1_experiment_results.json"
                    result_file = os.path.join(run_path, "run_1_experiment_results.json")
                    if not os.path.exists(result_file):
                        missing += 1
                        continue

                    with open(result_file) as f:
                        result = json.load(f)

                    final_utils = result.get("final_utilities", {})
                    alpha_util = final_utils.get("Agent_Alpha", 0.0)
                    beta_util = final_utils.get("Agent_Beta", 0.0)

                    # Determine which agent is the baseline (GPT-5-nano)
                    # Use agent_performance if available; fall back to model_order
                    agent_perf = result.get("agent_performance", {})
                    alpha_model = agent_perf.get("Agent_Alpha", {}).get("model", "")

                    if alpha_model == BASELINE_MODEL:
                        baseline_util = alpha_util
                        adv_util = beta_util
                    elif order_name == "weak_first":
                        # weak_first → Agent_Alpha = model1 = gpt-5-nano
                        baseline_util = alpha_util
                        adv_util = beta_util
                    else:
                        # strong_first → Agent_Alpha = model2 = adversary
                        baseline_util = beta_util
                        adv_util = alpha_util

                    competition_index = theta * (1 - rho) / 2

                    rows.append({
                        "adversary_model": adversary_model,
                        "adversary_elo": adversary_elo,
                        "adversary_short": MODEL_SHORT.get(adversary_model, adversary_model),
                        "model_order": order_name,
                        "run_id": run_id,
                        "rho": round(rho, 4),
                        "theta": round(theta, 4),
                        "competition_index": round(competition_index, 4),
                        "baseline_utility": baseline_util,
                        "adversary_utility": adv_util,
                        "social_welfare": baseline_util + adv_util,
                        "payoff_diff": adv_util - baseline_util,
                        "util_ratio": adv_util / baseline_util if baseline_util > 0 else float("inf"),
                        "consensus_reached": float(result.get("consensus_reached", False)),
                        "final_round": result.get("final_round", -1),
                    })

    if missing > 0:
        print(f"  WARNING: {missing} result files missing")

    df = pd.DataFrame(rows)
    print(f"Extracted {len(df)} runs from {df['adversary_model'].nunique()} adversary models")
    return df


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def _bucket_rho(rho: float) -> float:
    """Round rho to nearest 0.5 for bucketing."""
    return round(round(rho * 2) / 2, 1)


def _bucket_ci(ci: float) -> float:
    """Round CI to nearest 0.25 for bucketing."""
    return round(round(ci * 4) / 4, 2)


def build_avg_df(df: pd.DataFrame, param_col: str) -> pd.DataFrame:
    """Average over model_order, run_id, and (if param_col='rho') also theta."""
    group_cols = ["adversary_model", param_col]
    agg_map: Dict[str, str] = {}
    for col in df.columns:
        if col in group_cols:
            continue
        if col in MEAN_COLS:
            agg_map[col] = "mean"
        elif col == "model_order":
            agg_map[col] = "count"
        elif col == "run_id":
            agg_map[col] = "count"
        else:
            agg_map[col] = "first"

    df_avg = df.groupby(group_cols, as_index=False).agg(agg_map)
    return df_avg.sort_values([param_col, "adversary_elo"])


# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------

def smooth_series(values: np.ndarray, method: str, alpha: float, window: int) -> np.ndarray:
    series = pd.Series(values, dtype="float64")
    if method == "moving_average":
        return series.rolling(window=window, min_periods=1).mean().to_numpy()
    if method == "ewm":
        return series.ewm(alpha=alpha, adjust=False).mean().to_numpy()
    raise ValueError(f"Unsupported smoothing method: {method}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_payoff_lines(
    df_avg: pd.DataFrame,
    param_col: str,           # "rho" or "competition_index"
    value_col: str,           # "baseline_utility" or "adversary_utility"
    ylabel: str,
    title: str,
    output_path: Path,
    apply_smoothing: bool,
    smoothing_method: str,
    smoothing_alpha: float,
    smoothing_window: int,
    param_labels: Dict[float, str],
) -> List[str]:
    """
    Plot payoff vs adversary Elo, with one line per param_col level.
    Returns list of adversary models actually plotted.
    """
    df_line = df_avg.copy()
    if df_line.empty:
        raise ValueError("No rows available for plotting.")

    param_levels = sorted(df_line[param_col].dropna().unique())
    colors = plt.cm.viridis(np.linspace(0, 1, len(param_levels)))

    fig, ax = plt.subplots(figsize=(12, 8))

    plotted_models = set()
    for color, level in zip(colors, param_levels):
        level_df = df_line[df_line[param_col] == level].sort_values("adversary_elo")
        if level_df.empty:
            continue

        x_vals = level_df["adversary_elo"].to_numpy()
        y_raw = level_df[value_col].to_numpy()
        plotted_models.update(level_df["adversary_model"].tolist())

        if apply_smoothing:
            y_vals = smooth_series(y_raw, smoothing_method, smoothing_alpha, smoothing_window)
            ax.plot(x_vals, y_raw, color=color, linewidth=1.0, alpha=0.22)
        else:
            y_vals = y_raw

        label = param_labels.get(round(level, 2), f"{param_col}={level:.2f}")
        ax.plot(
            x_vals,
            y_vals,
            color=color,
            linewidth=1.8,
            marker="o",
            markersize=4,
            alpha=0.95,
            label=label,
        )

    # Add model name annotations on the x-axis
    tick_models = (
        df_line.groupby("adversary_elo")["adversary_short"].first().sort_index()
    )
    ax.set_xticks(tick_models.index)
    ax.set_xticklabels(tick_models.values, rotation=30, ha="right", fontsize=9)

    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, linewidth=1.1, label="Equal split (0.5)")
    ax.set_xlabel("Adversary Model  (ordered by Elo →)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8.5, title_fontsize=9.5, ncol=1, loc="best", frameon=True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return sorted(plotted_models)


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def get_model_filter(df: pd.DataFrame, min_runs: int) -> Tuple[List[str], pd.Series]:
    run_counts = df["adversary_model"].value_counts().sort_index()
    if min_runs <= 0:
        return sorted(run_counts.index.tolist()), run_counts
    eligible = run_counts[run_counts >= min_runs].index.tolist()
    return sorted(eligible), run_counts


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_report(
    report_path: Path,
    experiment_dir: str,
    min_runs: int,
    generated_plots: List[str],
    models_used_by_plot: Dict[str, List[str]],
    run_counts: pd.Series,
    df: pd.DataFrame,
) -> None:
    rho_vals = sorted(df["rho"].dropna().unique())
    theta_vals = sorted(df["theta"].dropna().unique())

    with report_path.open("w", encoding="utf-8") as f:
        f.write("# Lewis Slides Diplomacy Report\n\n")
        f.write(f"- Experiment dir: `{experiment_dir}`\n")
        f.write(f"- Model inclusion rule: `runs >= {min_runs}`\n")
        f.write(f"- rho values: {rho_vals}\n")
        f.write(f"- theta values: {theta_vals}\n")
        f.write(f"- Competition index: CI = theta × (1 − rho) / 2\n\n")

        f.write("## Generated Plots\n\n")
        for name in generated_plots:
            f.write(f"- `{name}`\n")

        f.write("\n## Models Used Per Plot\n\n")
        for plot_name, models in models_used_by_plot.items():
            f.write(f"### `{plot_name}`\n")
            f.write(f"- Count: {len(models)}\n")
            f.write(f"- Models: {', '.join(models)}\n\n")

        f.write("## Run Counts Per Adversary Model\n\n")
        f.write("| model | elo | total_runs |\n|---|---:|---:|\n")
        elo_map = df.groupby("adversary_model")["adversary_elo"].first()
        for model in sorted(run_counts.index):
            elo = elo_map.get(model, 0)
            f.write(f"| {model} | {elo} | {run_counts[model]} |\n")

        f.write("\n## Grid Coverage (runs per rho × theta cell)\n\n")
        pivot = (
            df.groupby(["rho", "theta"])["baseline_utility"]
            .count()
            .reset_index()
            .rename(columns={"baseline_utility": "n_runs"})
            .pivot(index="rho", columns="theta", values="n_runs")
            .fillna(0)
            .astype(int)
        )
        f.write(pivot.to_markdown())
        f.write("\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Lewis-style slide plots for Game 2 (Diplomatic Treaty).")
    parser.add_argument(
        "--experiment-dir",
        default=None,
        help="Path to experiment results directory (default: most recent diplomacy_* dir).",
    )
    parser.add_argument(
        "--output-dir",
        default="figures/diplomacy_lewis",
        help="Output directory relative to this script's location.",
    )
    parser.add_argument(
        "--param",
        choices=["rho", "ci", "both"],
        default="both",
        help="Which parameter to use as competition lines: rho, ci, or both (default).",
    )
    parser.add_argument(
        "--min-runs-per-model",
        type=int,
        default=5,
        help="Minimum total runs per adversary model to include in plots.",
    )
    parser.add_argument(
        "--smoothing-method",
        choices=["ewm", "moving_average"],
        default="ewm",
    )
    parser.add_argument("--smoothing-alpha", type=float, default=0.35)
    parser.add_argument("--smoothing-window", type=int, default=3)
    parser.add_argument(
        "--make-all-variants",
        action="store_true",
        help="Also generate raw (no smoothing) variants.",
    )
    parser.add_argument(
        "--report-name",
        default="lewis_diplomacy_report.md",
    )
    return parser.parse_args()


def _find_experiment_dir(script_dir: Path) -> str:
    results_base = script_dir.parent / "experiments" / "results"
    dirs = sorted(
        d for d in os.listdir(results_base)
        if d.startswith("diplomacy_") and d != "diplomacy_latest"
    )
    if not dirs:
        raise FileNotFoundError("No diplomacy_* experiment directories found.")
    return str(results_base / dirs[-1])


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    output_dir = (script_dir / args.output_dir) if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    experiment_dir = args.experiment_dir or _find_experiment_dir(script_dir)
    print(f"Experiment dir: {experiment_dir}")

    # Extract data
    df = extract_results(experiment_dir)

    # Filter models
    included_models, run_counts = get_model_filter(df, args.min_runs_per_model)
    df = df[df["adversary_model"].isin(included_models)].copy()
    print(f"After filtering: {df['adversary_model'].nunique()} models, {len(df)} rows")

    if df.empty:
        raise ValueError("No data after filtering. Lower --min-runs-per-model.")

    # Save raw CSV
    csv_path = output_dir / "diplomacy_lewis_data.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")

    generated_plots: List[str] = []
    models_used_by_plot: Dict[str, List[str]] = {}

    smoothing_kwargs = dict(
        apply_smoothing=True,
        smoothing_method=args.smoothing_method,
        smoothing_alpha=args.smoothing_alpha,
        smoothing_window=args.smoothing_window,
    )

    # ------------------------------------------------------------------
    # RHO version (2-variable)
    # ------------------------------------------------------------------
    if args.param in ("rho", "both"):
        # Bucket rho to canonical values and average over theta
        df_rho = df.copy()
        df_rho["rho"] = df_rho["rho"].apply(_bucket_rho)
        avg_rho = build_avg_df(df_rho, "rho")

        for value_col, label, plot_name in [
            ("baseline_utility", f"Baseline ({BASELINE_MODEL}) Utility", "MAIN_PLOT_1_BASELINE_PAYOFF_RHO.png"),
            ("adversary_utility", "Adversary Utility", "MAIN_PLOT_2_ADVERSARY_PAYOFF_RHO.png"),
        ]:
            title = (
                f"{label} vs Adversary Elo  [lines = ρ level, θ averaged]\n"
                f"({args.smoothing_method.upper()} smoothing)"
            )
            used = plot_payoff_lines(
                df_avg=avg_rho,
                param_col="rho",
                value_col=value_col,
                ylabel=label,
                title=title,
                output_path=output_dir / plot_name,
                param_labels=RHO_LABELS,
                **smoothing_kwargs,
            )
            generated_plots.append(plot_name)
            models_used_by_plot[plot_name] = used
            print(f"Saved {plot_name}")

        if args.make_all_variants:
            for value_col, label, suffix in [
                ("baseline_utility", f"Baseline ({BASELINE_MODEL}) Utility", "baseline_rho_raw.png"),
                ("adversary_utility", "Adversary Utility", "adversary_rho_raw.png"),
            ]:
                used = plot_payoff_lines(
                    df_avg=avg_rho,
                    param_col="rho",
                    value_col=value_col,
                    ylabel=label,
                    title=f"{label} vs Adversary Elo [ρ lines, raw]",
                    output_path=output_dir / suffix,
                    apply_smoothing=False,
                    smoothing_method=args.smoothing_method,
                    smoothing_alpha=args.smoothing_alpha,
                    smoothing_window=args.smoothing_window,
                    param_labels=RHO_LABELS,
                )
                generated_plots.append(suffix)
                models_used_by_plot[suffix] = used
                print(f"Saved {suffix}")

    # ------------------------------------------------------------------
    # CI version (1-variable)
    # ------------------------------------------------------------------
    if args.param in ("ci", "both"):
        df_ci = df.copy()
        df_ci["ci_bucket"] = df_ci["competition_index"].apply(_bucket_ci)
        avg_ci = build_avg_df(df_ci, "ci_bucket")

        for value_col, label, plot_name in [
            ("baseline_utility", f"Baseline ({BASELINE_MODEL}) Utility", "MAIN_PLOT_3_BASELINE_PAYOFF_CI.png"),
            ("adversary_utility", "Adversary Utility", "MAIN_PLOT_4_ADVERSARY_PAYOFF_CI.png"),
        ]:
            title = (
                f"{label} vs Adversary Elo  [lines = CI = θ·(1−ρ)/2]\n"
                f"({args.smoothing_method.upper()} smoothing)"
            )
            used = plot_payoff_lines(
                df_avg=avg_ci,
                param_col="ci_bucket",
                value_col=value_col,
                ylabel=label,
                title=title,
                output_path=output_dir / plot_name,
                param_labels=CI_BUCKET_LABELS,
                **smoothing_kwargs,
            )
            generated_plots.append(plot_name)
            models_used_by_plot[plot_name] = used
            print(f"Saved {plot_name}")

        if args.make_all_variants:
            for value_col, label, suffix in [
                ("baseline_utility", f"Baseline ({BASELINE_MODEL}) Utility", "baseline_ci_raw.png"),
                ("adversary_utility", "Adversary Utility", "adversary_ci_raw.png"),
            ]:
                used = plot_payoff_lines(
                    df_avg=avg_ci,
                    param_col="ci_bucket",
                    value_col=value_col,
                    ylabel=label,
                    title=f"{label} vs Adversary Elo [CI lines, raw]",
                    output_path=output_dir / suffix,
                    apply_smoothing=False,
                    smoothing_method=args.smoothing_method,
                    smoothing_alpha=args.smoothing_alpha,
                    smoothing_window=args.smoothing_window,
                    param_labels=CI_BUCKET_LABELS,
                )
                generated_plots.append(suffix)
                models_used_by_plot[suffix] = used
                print(f"Saved {suffix}")

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    report_path = output_dir / args.report_name
    write_report(
        report_path=report_path,
        experiment_dir=experiment_dir,
        min_runs=args.min_runs_per_model,
        generated_plots=generated_plots,
        models_used_by_plot=models_used_by_plot,
        run_counts=run_counts,
        df=df,
    )
    print(f"Saved report: {report_path}")
    print(f"\nDone. {len(generated_plots)} plot(s) written to {output_dir}")


if __name__ == "__main__":
    main()
