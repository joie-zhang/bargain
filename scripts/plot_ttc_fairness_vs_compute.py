#!/usr/bin/env python3
"""Plot TTC fairness and inequality metrics against observed target tokens."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_n2_baseline_comparison import compute_solution_metrics  # noqa: E402


RESULTS_CSV = (
    PROJECT_ROOT
    / "experiments/results/ttc_native_scaling_20260502_212943/monitoring/partial_results_latest.csv"
)
TABLE_DIR = PROJECT_ROOT / "analysis/neurips_revision_20260504"
RUN_METRICS_CSV = TABLE_DIR / "ttc_fairness_inequality_by_run.csv"
ORDER_AVG_CSV = TABLE_DIR / "ttc_fairness_inequality_order_averaged.csv"
SUMMARY_CSV = TABLE_DIR / "ttc_fairness_inequality_game_averaged_by_effort.csv"
GRAPHICS_DIR = PROJECT_ROOT / "overleaf/neurips/graphics"

FAMILY_ORDER = ["gpt-5", "claude-sonnet-4-6", "gemini-3-flash"]
FAMILY_LABELS = {
    "gpt-5": "GPT-5",
    "claude-sonnet-4-6": "Claude Sonnet 4.6",
    "gemini-3-flash": "Gemini 3 Flash",
}
FAMILY_MARKERS = {
    "gpt-5": "o",
    "claude-sonnet-4-6": "s",
    "gemini-3-flash": "^",
}
EFFORT_ORDER = ["minimal", "low", "medium", "high", "max"]
EFFORT_LABELS = {
    "minimal": "Minimal",
    "low": "Low",
    "medium": "Medium",
    "high": "High",
    "max": "Max",
}
EFFORT_COLORS = {
    "minimal": "#64748b",
    "low": "#2563eb",
    "medium": "#0f766e",
    "high": "#f97316",
    "max": "#7c3aed",
}

PLOTS = {
    "fairness_distance": {
        "ylabel": "Mean NBS/Lindahl distance",
        "filename": "ttc_game_averaged_fairness_distance_vs_compute.png",
        "ylim_floor": 0.0,
    },
    "payoff_gini_corrected": {
        "ylabel": "Mean corrected payoff Gini",
        "filename": "ttc_game_averaged_corrected_gini_vs_compute.png",
        "ylim_floor": 0.0,
    },
    "utility_gap": {
        "ylabel": "Mean target - baseline payoff",
        "filename": "ttc_game_averaged_payoff_difference_vs_compute.png",
        "zero_line": True,
    },
    "absolute_payoff_gap": {
        "ylabel": "Mean absolute payoff difference",
        "filename": "ttc_game_averaged_absolute_payoff_difference_vs_compute.png",
        "ylim_floor": 0.0,
    },
    "payoff_variance": {
        "ylabel": "Mean within-run payoff variance",
        "filename": "ttc_game_averaged_payoff_variance_vs_compute.png",
        "ylim_floor": 0.0,
    },
}


def sem(series: pd.Series) -> float:
    clean = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) <= 1:
        return 0.0
    return float(clean.std(ddof=1) / math.sqrt(len(clean)))


def gini_shifted_corrected(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return math.nan
    if float(arr.min()) < 0.0:
        arr = arr - float(arr.min())
    if arr.size < 2 or np.allclose(arr, arr[0]) or np.allclose(arr, 0.0):
        return 0.0
    mean_value = float(arr.mean())
    if math.isclose(mean_value, 0.0):
        return 0.0
    raw_gini = float(np.mean(np.abs(arr[:, None] - arr[None, :])) / (2.0 * mean_value))
    return min(raw_gini * float(arr.size / (arr.size - 1)), 1.0)


def payoff_variance(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return math.nan
    return float(np.var(arr, ddof=0))


def read_payload(path_value: str) -> dict[str, Any]:
    path = Path(path_value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return json.loads(path.read_text(encoding="utf-8"))


def build_run_metrics() -> pd.DataFrame:
    df = pd.read_csv(RESULTS_CSV)
    rows: list[dict[str, Any]] = []

    for _, row in df.iterrows():
        payload = read_payload(str(row["path"]))
        config_payload = payload.get("config") or {}
        metrics = compute_solution_metrics(
            game_id=str(row["game"]),
            payload=payload,
            config_payload=config_payload,
            baseline_agent=str(row["baseline_agent"]),
            adversary_agent=str(row["target_agent"]),
        )
        target_utility = float(row["target_utility"])
        baseline_utility = float(row["baseline_utility"])
        payoffs = [target_utility, baseline_utility]
        rows.append(
            {
                **row.to_dict(),
                "fairness_distance": metrics.get("fairness_distance", np.nan),
                "adversary_fairness_excess": metrics.get("adversary_fairness_excess", np.nan),
                "baseline_fairness_excess": metrics.get("baseline_fairness_excess", np.nan),
                "payoff_gini_corrected": gini_shifted_corrected(payoffs),
                "payoff_variance": payoff_variance(payoffs),
                "absolute_payoff_gap": abs(target_utility - baseline_utility),
            }
        )

    run_metrics = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)
    RUN_METRICS_CSV.parent.mkdir(parents=True, exist_ok=True)
    run_metrics.to_csv(RUN_METRICS_CSV, index=False)
    return run_metrics


def build_order_average(run_metrics: pd.DataFrame) -> pd.DataFrame:
    metric_cols = list(PLOTS) + ["absolute_payoff_gap"]
    agg_spec: dict[str, Any] = {
        "order_count": ("order", "nunique"),
        "run_count": ("config_id", "size"),
        "target_compute_tokens_per_call": ("target_compute_tokens_per_call", "mean"),
        "target_output_tokens_per_call": ("target_output_tokens_per_call", "mean"),
        "target_reasoning_tokens_raw_per_call": ("target_reasoning_tokens_raw_per_call", "mean"),
        "target_utility": ("target_utility", "mean"),
        "baseline_utility": ("baseline_utility", "mean"),
        "consensus_rate": ("consensus", "mean"),
        "mean_round": ("round", "mean"),
    }
    for col in metric_cols:
        agg_spec[col] = (col, "mean")
    order_avg = (
        run_metrics.groupby(["family", "provider", "level", "level_index", "game", "game_cell"], dropna=False)
        .agg(**agg_spec)
        .reset_index()
        .sort_values(["family", "game_cell", "level_index"])
    )
    ORDER_AVG_CSV.parent.mkdir(parents=True, exist_ok=True)
    order_avg.to_csv(ORDER_AVG_CSV, index=False)
    return order_avg


def build_summary(order_avg: pd.DataFrame) -> pd.DataFrame:
    agg_spec: dict[str, Any] = {
        "game_cell_count": ("game_cell", "nunique"),
        "target_tokens_mean": ("target_compute_tokens_per_call", "mean"),
        "target_tokens_sem": ("target_compute_tokens_per_call", sem),
        "target_utility_mean": ("target_utility", "mean"),
        "target_utility_sem": ("target_utility", sem),
        "consensus_rate_mean": ("consensus_rate", "mean"),
        "mean_round": ("mean_round", "mean"),
    }
    for metric in list(PLOTS) + ["absolute_payoff_gap"]:
        agg_spec[f"{metric}_mean"] = (metric, "mean")
        agg_spec[f"{metric}_sem"] = (metric, sem)
    summary = (
        order_avg.groupby(["family", "provider", "level", "level_index"], dropna=False)
        .agg(**agg_spec)
        .reset_index()
        .sort_values(["family", "level_index"])
    )
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_CSV, index=False)
    return summary


def y_limits(summary: pd.DataFrame, metric: str, config: dict[str, Any]) -> tuple[float, float]:
    mean_col = f"{metric}_mean"
    sem_col = f"{metric}_sem"
    vals = pd.to_numeric(summary[mean_col], errors="coerce")
    errs = pd.to_numeric(summary[sem_col], errors="coerce").fillna(0.0)
    lower = float((vals - errs).min())
    upper = float((vals + errs).max())
    if not math.isfinite(lower) or not math.isfinite(upper) or math.isclose(lower, upper):
        lower, upper = 0.0, 1.0
    span = upper - lower
    pad = 0.12 * span if span > 0 else 1.0
    lower -= pad
    upper += pad
    if "ylim_floor" in config:
        lower = float(config["ylim_floor"])
        upper = max(upper, lower + 1e-6)
    return lower, upper


def plot_metric(summary: pd.DataFrame, metric: str, config: dict[str, Any]) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(14.6, 4.9), sharex=True, sharey=True)
    fig.patch.set_facecolor("white")

    x_max = max(2200, float(summary["target_tokens_mean"].max()) * 1.15)
    ylim = y_limits(summary, metric, config)
    y_mean = f"{metric}_mean"
    y_sem = f"{metric}_sem"

    for ax, family in zip(axes, FAMILY_ORDER):
        family_df = summary[summary["family"].eq(family)].sort_values("level_index")
        if family_df.empty:
            continue

        ax.plot(
            family_df["target_tokens_mean"],
            family_df[y_mean],
            color="#475569",
            linewidth=2.55,
            alpha=0.60,
            zorder=2,
        )
        if config.get("zero_line"):
            ax.axhline(0.0, color="#111827", linewidth=0.95, alpha=0.42, zorder=1)

        for _, row in family_df.iterrows():
            effort = str(row["level"])
            ax.errorbar(
                row["target_tokens_mean"],
                row[y_mean],
                yerr=max(float(row[y_sem]), 0.0) if pd.notna(row[y_sem]) else 0.0,
                fmt=FAMILY_MARKERS[family],
                markersize=10.8,
                color=EFFORT_COLORS.get(effort, "#475569"),
                markeredgecolor="white",
                markeredgewidth=0.95,
                elinewidth=1.55,
                capsize=4.2,
                alpha=0.95,
                zorder=3,
            )

        ax.set_title(FAMILY_LABELS[family], fontsize=24, pad=10)
        ax.tick_params(axis="both", labelsize=15)
        ax.grid(True, color="#d1d5db", alpha=0.42, linewidth=0.85)
        ax.set_xlim(-70, x_max)
        ax.set_ylim(*ylim)

    axes[0].set_ylabel(str(config["ylabel"]), fontsize=20, labelpad=10)
    fig.supxlabel("Mean observed target tokens/call", fontsize=19, y=0.07)

    present_efforts = set(summary["level"].astype(str))
    handles = [
        Line2D(
            [0],
            [0],
            color=EFFORT_COLORS[effort],
            marker="o",
            linestyle="",
            markersize=10.5,
            markeredgecolor="white",
            markeredgewidth=0.9,
            label=EFFORT_LABELS[effort],
        )
        for effort in EFFORT_ORDER
        if effort in present_efforts
    ]
    legend = fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.07),
        ncol=len(handles),
        title="Reasoning effort",
        title_fontsize=13.5,
        fontsize=13.2,
        frameon=True,
        facecolor="white",
        framealpha=0.94,
        columnspacing=1.0,
        handletextpad=0.45,
        borderpad=0.45,
    )
    legend.get_frame().set_edgecolor("#d1d5db")

    fig.subplots_adjust(left=0.08, right=0.995, top=0.84, bottom=0.25, wspace=0.11)
    output_path = GRAPHICS_DIR / str(config["filename"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=260, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    return output_path


def main() -> None:
    run_metrics = build_run_metrics()
    order_avg = build_order_average(run_metrics)
    summary = build_summary(order_avg)

    for metric, config in PLOTS.items():
        print(plot_metric(summary, metric, config))
    print(RUN_METRICS_CSV)
    print(ORDER_AVG_CSV)
    print(SUMMARY_CSV)


if __name__ == "__main__":
    main()
