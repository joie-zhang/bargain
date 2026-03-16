#!/usr/bin/env python3
"""
=============================================================================
Lewis Slides Script — Game 3 (Co-Funding / Participatory Budgeting)
=============================================================================

Mirrors lewis_slides_script_mar_1.py (Game 1) and lewis_slides_diplomacy.py
(Game 2), adapted for Game 3's two-parameter competition space:
  alpha (preference alignment) and sigma (budget abundance).

Reads raw JSON results directly from the experiment directory tree and
produces two families of "payoff vs focal model Elo" plots:

  2-VARIABLE version  — one line per alpha level (sigma averaged out)
  1-VARIABLE version  — one line per CI₃ bucket, where CI₃ = (1-alpha)*(1-sigma)

Usage:
    python visualization/lewis_slides_cofunding.py
    python visualization/lewis_slides_cofunding.py --experiment-dir experiments/results/cofunding_latest
    python visualization/lewis_slides_cofunding.py --param alpha       # only alpha-bucketed plots
    python visualization/lewis_slides_cofunding.py --param ci          # only CI₃-bucketed plots
    python visualization/lewis_slides_cofunding.py --make-all-variants

What it creates:
    visualization/figures/cofunding_lewis/
    ├── MAIN_PLOT_1_FOCAL_PAYOFF_ALPHA.png       # focal model utility, lines = alpha
    ├── MAIN_PLOT_2_REFERENCE_PAYOFF_ALPHA.png   # reference (gpt-5.2) utility, lines = alpha
    ├── MAIN_PLOT_3_FOCAL_PAYOFF_CI.png          # focal utility, lines = CI₃ bucket
    ├── MAIN_PLOT_4_REFERENCE_PAYOFF_CI.png      # reference utility, lines = CI₃ bucket
    ├── cofunding_lewis_data.csv                 # flat per-run data
    └── lewis_cofunding_report.md                # coverage report

Competition index motivation:
    alpha in [0, 1]: preference alignment.
        alpha = 1 → agents want to fund the same projects (pure coordination)
        alpha = 0 → agents want to fund completely different projects (pure conflict)
    sigma in (0, 1]: budget abundance (budget = sigma * total_project_cost).
        sigma = 1 → budget covers everything, no scarcity
        sigma → 0 → severe budget scarcity, agents must compete for funding
    CI₃ = (1 - alpha) * (1 - sigma), in [0, 1].
        CI₃ = 0 → no competition (either identical prefs OR unlimited budget)
        CI₃ = 1 → maximal competition (fully misaligned prefs AND no budget)

Note on roles:
    The 'reference' model is the constant opponent (gpt-5.2-chat).
    The 'focal' model is the one being studied (varies across experiments).
    X-axis = focal model Elo (ascending).

Dependencies:
    - matplotlib, numpy, pandas

=============================================================================
"""

from __future__ import annotations

import argparse
import json
import os
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
    "claude-opus-4-6": 1475,
    "claude-opus-4-5-thinking-32k": 1470,
    "claude-sonnet-4-5": 1450,
    "gpt-5.2-high": 1436,
    "gpt-5.2-chat-latest-20260210": 1436,
    "gpt-5.2-chat": 1436,
    "qwen3-max": 1434,
    "grok-4": 1409,
    "claude-haiku-4-5": 1403,
    "deepseek-r1": 1397,
    "o3-mini-high": 1364,
    "qwen3-32b": 1360,
    "gpt-4o": 1346,
    "gpt-5-nano": 1338,
    "llama-3.1-8b-instruct": 1180,
}

MODEL_SHORT: Dict[str, str] = {
    "gemini-3-pro": "Gemini 3 Pro",
    "claude-opus-4-6": "Opus 4.6",
    "claude-opus-4-5-thinking-32k": "Claude Opus",
    "claude-sonnet-4-5": "Sonnet 4.5",
    "gpt-5.2-high": "GPT-5.2",
    "gpt-5.2-chat-latest-20260210": "GPT-5.2-chat",
    "gpt-5.2-chat": "GPT-5.2-chat",
    "qwen3-max": "Qwen3-Max",
    "grok-4": "Grok-4",
    "claude-haiku-4-5": "Haiku 4.5",
    "deepseek-r1": "DeepSeek-R1",
    "o3-mini-high": "O3-mini",
    "qwen3-32b": "Qwen3-32B",
    "gpt-4o": "GPT-4o",
    "gpt-5-nano": "GPT-5-nano",
    "llama-3.1-8b-instruct": "Llama 3.1 8B",
}

# Patterns that identify the reference (constant opponent) model
REFERENCE_PATTERNS = ["gpt-5.2", "gpt-5-nano"]

# Equal-split reference line for raw utility (~0-100 scale per experiment)
# We'll compute this dynamically from mean social welfare in the data

ALPHA_LABELS: Dict[float, str] = {
    0.0: "α=0 (misaligned)",
    0.5: "α=0.5",
    1.0: "α=1 (aligned)",
}

CI_BUCKET_LABELS: Dict[float, str] = {
    0.00: r"$CI_3$=0.00 (cooperative)",
    0.20: r"$CI_3$=0.20",
    0.40: r"$CI_3$=0.40",
    0.60: r"$CI_3$=0.60",
    0.80: r"$CI_3$=0.80 (competitive)",
}

MEAN_COLS = {
    "focal_utility",
    "reference_utility",
    "social_welfare",
    "focal_advantage",
    "competition_index",
    "consensus_reached",
    "final_round",
}


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------

def _is_reference(model: str) -> bool:
    s = model.lower()
    return any(p in s for p in REFERENCE_PATTERNS)


def _short(model: str) -> str:
    return MODEL_SHORT.get(model, model)


def _parse_alpha_sigma(dirname: str) -> Tuple[Optional[float], Optional[float]]:
    """Parse alpha and sigma from directory name like 'alpha_0_5_sigma_0_2'."""
    try:
        alpha_str = dirname.replace("alpha_", "").split("_sigma_")[0]
        sigma_str = dirname.split("_sigma_")[1]

        def _val(s: str) -> float:
            parts = s.split("_")
            return float(parts[0]) + (float(parts[1]) / (10 ** len(parts[1])) if len(parts) > 1 and parts[1] else 0.0)

        return _val(alpha_str), _val(sigma_str)
    except Exception:
        return None, None


def extract_results(experiment_dir: str) -> pd.DataFrame:
    """
    Walk the cofunding experiment directory tree and collect one row per run.

    Directory structure expected:
        experiment_dir/model_scale/
            [model1]_vs_[model2]/
                {weak_first,strong_first}/
                    alpha_*_sigma_*/
                        experiment_results.json  (or run_1_experiment_results.json)
    """
    model_scale_dir = os.path.join(experiment_dir, "model_scale")
    if not os.path.isdir(model_scale_dir):
        raise FileNotFoundError(f"No model_scale directory at {model_scale_dir}")

    rows = []
    missing = 0

    for pair_name in sorted(os.listdir(model_scale_dir)):
        pair_path = os.path.join(model_scale_dir, pair_name)
        if not os.path.isdir(pair_path) or "_vs_" not in pair_name:
            continue

        model1, model2 = pair_name.split("_vs_", 1)

        # Skip self-play
        if model1 == model2:
            continue

        # Determine which is focal (non-reference) and which is reference
        if _is_reference(model1) and not _is_reference(model2):
            reference_model, focal_model = model1, model2
        elif _is_reference(model2) and not _is_reference(model1):
            reference_model, focal_model = model2, model1
        else:
            # Both or neither match reference pattern — use model1 as reference
            reference_model, focal_model = model1, model2

        focal_elo = MODEL_ELO.get(focal_model, 0)
        reference_elo = MODEL_ELO.get(reference_model, 0)

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

                alpha, sigma = _parse_alpha_sigma(cond_name)
                if alpha is None:
                    continue

                # Cofunding results may live directly in the condition dir
                # (no run subdirectory), or inside run_N subdirs
                result_files = []
                for fname in ["experiment_results.json", "run_1_experiment_results.json"]:
                    fp = os.path.join(cond_path, fname)
                    if os.path.exists(fp):
                        result_files.append((fp, "run_1"))

                # Also check run_N subdirs
                for entry in sorted(os.listdir(cond_path)):
                    entry_path = os.path.join(cond_path, entry)
                    if os.path.isdir(entry_path) and entry.startswith("run_"):
                        for fname in ["run_1_experiment_results.json", "experiment_results.json"]:
                            fp = os.path.join(entry_path, fname)
                            if os.path.exists(fp):
                                result_files.append((fp, entry))
                                break

                # Deduplicate by file path
                seen_files = set()
                unique_results = []
                for fp, run_id in result_files:
                    if fp not in seen_files:
                        seen_files.add(fp)
                        unique_results.append((fp, run_id))

                if not unique_results:
                    missing += 1
                    continue

                for result_file, run_id in unique_results:
                    with open(result_file) as f:
                        result = json.load(f)

                    final_utils = result.get("final_utilities", {})
                    if not final_utils or len(final_utils) < 2:
                        continue

                    # Assign utilities using agent_performance
                    agent_perf = result.get("agent_performance", {})
                    focal_util = None
                    ref_util = None

                    for agent_id, info in agent_perf.items():
                        agent_model = info.get("model", "")
                        u = final_utils.get(agent_id, 0.0)
                        # Match by model name suffix
                        if any(part in agent_model for part in focal_model.split("-")[:2]):
                            focal_util = u
                        elif any(part in agent_model for part in reference_model.split("-")[:2]):
                            ref_util = u

                    # Fallback: use model_order from path
                    if focal_util is None or ref_util is None:
                        agents = sorted(final_utils.keys())
                        if order_name == "weak_first":
                            # model1 is Agent_Alpha (first), model2 is Agent_Beta
                            alpha_agent, beta_agent = agents[0], agents[1]
                        else:
                            alpha_agent, beta_agent = agents[0], agents[1]

                        if _is_reference(model1):
                            # model1=reference=Agent_Alpha in weak_first, or check order
                            if order_name == "strong_first":
                                ref_util = final_utils.get(alpha_agent, 0.0)
                                focal_util = final_utils.get(beta_agent, 0.0)
                            else:
                                ref_util = final_utils.get(alpha_agent, 0.0)
                                focal_util = final_utils.get(beta_agent, 0.0)
                        else:
                            if order_name == "weak_first":
                                focal_util = final_utils.get(alpha_agent, 0.0)
                                ref_util = final_utils.get(beta_agent, 0.0)
                            else:
                                ref_util = final_utils.get(alpha_agent, 0.0)
                                focal_util = final_utils.get(beta_agent, 0.0)

                    competition_index = (1.0 - float(alpha)) * (1.0 - float(sigma))

                    rows.append({
                        "focal_model": focal_model,
                        "focal_elo": focal_elo,
                        "focal_short": _short(focal_model),
                        "reference_model": reference_model,
                        "reference_elo": reference_elo,
                        "model_order": order_name,
                        "run_id": run_id,
                        "alpha": round(alpha, 4),
                        "sigma": round(sigma, 4),
                        "competition_index": round(competition_index, 4),
                        "focal_utility": focal_util if focal_util is not None else float("nan"),
                        "reference_utility": ref_util if ref_util is not None else float("nan"),
                        "social_welfare": (focal_util or 0) + (ref_util or 0),
                        "focal_advantage": (focal_util or 0) - (ref_util or 0),
                        "consensus_reached": float(result.get("consensus_reached", False)),
                        "final_round": result.get("final_round", -1),
                    })

    if missing > 0:
        print(f"  WARNING: {missing} condition directories had no results")

    df = pd.DataFrame(rows)
    if not df.empty:
        print(f"Extracted {len(df)} runs from {df['focal_model'].nunique()} focal models")
        print(f"  Reference model(s): {df['reference_model'].unique().tolist()}")
        print(f"  Alpha values: {sorted(df['alpha'].unique())}")
        print(f"  Sigma values: {sorted(df['sigma'].unique())}")
    return df


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def build_avg_df(df: pd.DataFrame, param_col: str) -> pd.DataFrame:
    """Average over model_order and run_id (and sigma if param_col is alpha)."""
    group_cols = ["focal_model", param_col]
    agg_map: Dict[str, str] = {}
    for col in df.columns:
        if col in group_cols:
            continue
        if col in MEAN_COLS:
            agg_map[col] = "mean"
        elif col in ("model_order", "run_id"):
            agg_map[col] = "count"
        else:
            agg_map[col] = "first"

    df_avg = df.groupby(group_cols, as_index=False).agg(agg_map)
    return df_avg.sort_values([param_col, "focal_elo"])


def _bucket_ci(ci: float) -> float:
    """Round CI₃ to nearest 0.2 for bucketing."""
    return round(round(ci * 5) / 5, 2)


def _bucket_alpha(alpha: float) -> float:
    """Round alpha to nearest 0.5."""
    return round(round(alpha * 2) / 2, 1)


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
    param_col: str,
    value_col: str,
    ylabel: str,
    title: str,
    output_path: Path,
    apply_smoothing: bool,
    smoothing_method: str,
    smoothing_alpha: float,
    smoothing_window: int,
    param_labels: Dict[float, str],
    equal_split_value: Optional[float] = None,
) -> List[str]:
    """
    Plot payoff vs focal model Elo, with one line per param_col level.
    Returns list of focal models actually plotted.
    """
    df_line = df_avg.copy()
    if df_line.empty:
        raise ValueError("No rows available for plotting.")

    param_levels = sorted(df_line[param_col].dropna().unique())
    colors = plt.cm.viridis(np.linspace(0, 1, len(param_levels)))

    fig, ax = plt.subplots(figsize=(12, 8))
    plotted_models = set()

    for color, level in zip(colors, param_levels):
        level_df = df_line[df_line[param_col] == level].sort_values("focal_elo")
        if level_df.empty:
            continue

        x_vals = level_df["focal_elo"].to_numpy()
        y_raw = level_df[value_col].to_numpy()
        plotted_models.update(level_df["focal_model"].tolist())

        if apply_smoothing and len(y_raw) > 2:
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
            markersize=5,
            alpha=0.95,
            label=label,
        )

    # x-axis tick labels = short model names ordered by Elo
    tick_models = df_line.groupby("focal_elo")["focal_short"].first().sort_index()
    ax.set_xticks(tick_models.index)
    ax.set_xticklabels(tick_models.values, rotation=30, ha="right", fontsize=9)

    if equal_split_value is not None:
        ax.axhline(
            y=equal_split_value,
            color="gray",
            linestyle="--",
            alpha=0.5,
            linewidth=1.1,
            label=f"Equal split ({equal_split_value:.0f})",
        )

    ax.set_xlabel("Focal Model  (ordered by Elo →)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=8.5, title_fontsize=9.5, ncol=1, loc="best", frameon=True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return sorted(plotted_models)


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def get_model_filter(df: pd.DataFrame, min_runs: int) -> Tuple[List[str], pd.Series]:
    run_counts = df["focal_model"].value_counts().sort_index()
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
    alpha_vals = sorted(df["alpha"].dropna().unique())
    sigma_vals = sorted(df["sigma"].dropna().unique())

    with report_path.open("w", encoding="utf-8") as f:
        f.write("# Lewis Slides Co-Funding (Game 3) Report\n\n")
        f.write(f"- Experiment dir: `{experiment_dir}`\n")
        f.write(f"- Model inclusion rule: `runs >= {min_runs}`\n")
        f.write(f"- alpha values: {alpha_vals}\n")
        f.write(f"- sigma values: {sigma_vals}\n")
        f.write(f"- Competition index: CI₃ = (1−α)·(1−σ)\n\n")

        f.write("## Generated Plots\n\n")
        for name in generated_plots:
            f.write(f"- `{name}`\n")

        f.write("\n## Models Used Per Plot\n\n")
        for plot_name, models in models_used_by_plot.items():
            f.write(f"### `{plot_name}`\n")
            f.write(f"- Count: {len(models)}\n")
            f.write(f"- Models: {', '.join(models)}\n\n")

        f.write("## Run Counts Per Focal Model\n\n")
        f.write("| model | elo | total_runs |\n|---|---:|---:|\n")
        elo_map = df.groupby("focal_model")["focal_elo"].first()
        for model in sorted(run_counts.index):
            elo = elo_map.get(model, 0)
            f.write(f"| {model} | {elo} | {run_counts[model]} |\n")

        f.write("\n## Grid Coverage (runs per alpha × sigma cell)\n\n")
        pivot = (
            df.groupby(["alpha", "sigma"])["focal_utility"]
            .count()
            .reset_index()
            .rename(columns={"focal_utility": "n_runs"})
            .pivot(index="alpha", columns="sigma", values="n_runs")
            .fillna(0)
            .astype(int)
        )
        f.write(pivot.to_markdown())
        f.write("\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Lewis-style slide plots for Game 3 (Co-Funding)."
    )
    parser.add_argument(
        "--experiment-dir",
        default=None,
        help="Path to experiment results directory (default: most recent cofunding_* dir).",
    )
    parser.add_argument(
        "--output-dir",
        default="figures/cofunding_lewis",
        help="Output directory relative to this script's location.",
    )
    parser.add_argument(
        "--param",
        choices=["alpha", "ci", "both"],
        default="both",
        help="Parameter for competition lines: alpha, ci (CI₃), or both (default).",
    )
    parser.add_argument(
        "--min-runs-per-model",
        type=int,
        default=3,
        help="Minimum total runs per focal model to include in plots.",
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
    parser.add_argument("--report-name", default="lewis_cofunding_report.md")
    return parser.parse_args()


def _find_experiment_dir(script_dir: Path) -> str:
    results_base = script_dir.parent / "experiments" / "results"
    dirs = sorted(
        d for d in os.listdir(results_base)
        if d.startswith("cofunding_") and d != "cofunding_latest"
    )
    if not dirs:
        # Fall back to cofunding_latest symlink
        latest = results_base / "cofunding_latest"
        if latest.exists():
            return str(latest)
        raise FileNotFoundError("No cofunding_* experiment directories found.")
    return str(results_base / dirs[-1])


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    output_dir = (
        (script_dir / args.output_dir)
        if not Path(args.output_dir).is_absolute()
        else Path(args.output_dir)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    experiment_dir = args.experiment_dir or _find_experiment_dir(script_dir)
    print(f"Experiment dir: {experiment_dir}")

    # Extract data
    df = extract_results(experiment_dir)

    # Filter models
    included_models, run_counts = get_model_filter(df, args.min_runs_per_model)
    df = df[df["focal_model"].isin(included_models)].copy()
    print(f"After filtering: {df['focal_model'].nunique()} focal models, {len(df)} rows")

    if df.empty:
        raise ValueError("No data after filtering. Lower --min-runs-per-model.")

    # Equal-split reference: mean social welfare / 2 across all experiments
    mean_sw = df["social_welfare"].mean()
    equal_split = mean_sw / 2.0
    print(f"Mean social welfare: {mean_sw:.1f}  →  equal split reference: {equal_split:.1f}")

    # Save raw CSV
    csv_path = output_dir / "cofunding_lewis_data.csv"
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
    # ALPHA version (2-variable)
    # ------------------------------------------------------------------
    if args.param in ("alpha", "both"):
        df_alpha = df.copy()
        df_alpha["alpha"] = df_alpha["alpha"].apply(_bucket_alpha)
        avg_alpha = build_avg_df(df_alpha, "alpha")

        for value_col, ylabel, plot_name in [
            ("focal_utility", "Focal Model Utility", "MAIN_PLOT_1_FOCAL_PAYOFF_ALPHA.png"),
            ("reference_utility", "Reference Model Utility", "MAIN_PLOT_2_REFERENCE_PAYOFF_ALPHA.png"),
        ]:
            ref_models = df["reference_model"].unique()
            ref_label = _short(ref_models[0]) if len(ref_models) == 1 else "Reference"
            title = (
                f"{ylabel} vs Focal Model Elo  [lines = α level, σ averaged]\n"
                f"Reference: {ref_label} | ({args.smoothing_method.upper()} smoothing)"
            )
            used = plot_payoff_lines(
                df_avg=avg_alpha,
                param_col="alpha",
                value_col=value_col,
                ylabel=ylabel,
                title=title,
                output_path=output_dir / plot_name,
                param_labels=ALPHA_LABELS,
                equal_split_value=equal_split,
                **smoothing_kwargs,
            )
            generated_plots.append(plot_name)
            models_used_by_plot[plot_name] = used
            print(f"Saved {plot_name}")

        if args.make_all_variants:
            for value_col, ylabel, suffix in [
                ("focal_utility", "Focal Model Utility", "focal_alpha_raw.png"),
                ("reference_utility", "Reference Model Utility", "reference_alpha_raw.png"),
            ]:
                used = plot_payoff_lines(
                    df_avg=avg_alpha,
                    param_col="alpha",
                    value_col=value_col,
                    ylabel=ylabel,
                    title=f"{ylabel} vs Focal Elo [α lines, raw]",
                    output_path=output_dir / suffix,
                    apply_smoothing=False,
                    smoothing_method=args.smoothing_method,
                    smoothing_alpha=args.smoothing_alpha,
                    smoothing_window=args.smoothing_window,
                    param_labels=ALPHA_LABELS,
                    equal_split_value=equal_split,
                )
                generated_plots.append(suffix)
                models_used_by_plot[suffix] = used
                print(f"Saved {suffix}")

    # ------------------------------------------------------------------
    # CI₃ version (1-variable)
    # ------------------------------------------------------------------
    if args.param in ("ci", "both"):
        df_ci = df.copy()
        df_ci["ci_bucket"] = df_ci["competition_index"].apply(_bucket_ci)
        avg_ci = build_avg_df(df_ci, "ci_bucket")

        for value_col, ylabel, plot_name in [
            ("focal_utility", "Focal Model Utility", "MAIN_PLOT_3_FOCAL_PAYOFF_CI.png"),
            ("reference_utility", "Reference Model Utility", "MAIN_PLOT_4_REFERENCE_PAYOFF_CI.png"),
        ]:
            ref_models = df["reference_model"].unique()
            ref_label = _short(ref_models[0]) if len(ref_models) == 1 else "Reference"
            sm = args.smoothing_method.upper()
            title = (
                f"{ylabel} vs Focal Model Elo"
                r"  [lines = $CI_3 = (1-\alpha)(1-\sigma)$]"
                f"\nReference: {ref_label} | ({sm} smoothing)"
            )
            used = plot_payoff_lines(
                df_avg=avg_ci,
                param_col="ci_bucket",
                value_col=value_col,
                ylabel=ylabel,
                title=title,
                output_path=output_dir / plot_name,
                param_labels=CI_BUCKET_LABELS,
                equal_split_value=equal_split,
                **smoothing_kwargs,
            )
            generated_plots.append(plot_name)
            models_used_by_plot[plot_name] = used
            print(f"Saved {plot_name}")

        if args.make_all_variants:
            for value_col, ylabel, suffix in [
                ("focal_utility", "Focal Model Utility", "focal_ci_raw.png"),
                ("reference_utility", "Reference Model Utility", "reference_ci_raw.png"),
            ]:
                used = plot_payoff_lines(
                    df_avg=avg_ci,
                    param_col="ci_bucket",
                    value_col=value_col,
                    ylabel=ylabel,
                    title=f"{ylabel} vs Focal Elo [CI₃ lines, raw]",
                    output_path=output_dir / suffix,
                    apply_smoothing=False,
                    smoothing_method=args.smoothing_method,
                    smoothing_alpha=args.smoothing_alpha,
                    smoothing_window=args.smoothing_window,
                    param_labels=CI_BUCKET_LABELS,
                    equal_split_value=equal_split,
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
