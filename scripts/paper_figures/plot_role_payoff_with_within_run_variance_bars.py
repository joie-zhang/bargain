#!/usr/bin/env python3
"""Role payoff curves with bars showing within-run payoff dispersion.

The plotted means match the role-payoff curves. The vertical bars are not SEMs:
they are +/- sqrt(mean within-run payoff variance) for the bucket, so they are
on the same payoff scale as the y-axis.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "overleaf/neurips/graphics/n_gt_2_report"

ROLE_METRICS = OUT_DIR / "role_payoff_curves_by_strength_run_metrics.csv"
RUN_METRICS = OUT_DIR / "homogeneous_heterogeneous_bucketed_variance_mean_payoff_breakdown_run_metrics.csv"

SCENARIOS = {
    "homogeneous_adversary": {
        "title": "Homogeneous adversary",
        "file_prefix": "homogeneous_adversary_role_payoff_vs_adversary_elo_within_run_variance_bars",
        "x_label": "Adversary Elo bucket",
        "high_role": "Adversary",
        "low_role": "Baseline agents",
        "high_color": "#D95F02",
        "low_color": "#4E79A7",
    },
    "heterogeneous_max": {
        "title": "Heterogeneous max-Elo agent",
        "file_prefix": "heterogeneous_max_role_payoff_vs_max_elo_within_run_variance_bars",
        "x_label": "Max roster Elo bucket",
        "high_role": "Max-Elo agent(s)",
        "low_role": "Non-max agents",
        "high_color": "#2CA02C",
        "low_color": "#4E79A7",
    },
}


def summarize(frame: pd.DataFrame, scenario: str) -> pd.DataFrame:
    sub = frame[frame["scenario"].eq(scenario)].copy()
    rows: list[dict[str, object]] = []
    for bucket_code, bucket in sub.groupby("bucket_code", sort=True):
        mean_variance = float(bucket["payoff_variance"].mean())
        rows.append(
            {
                "scenario": scenario,
                "bucket_code": int(bucket_code),
                "bucket_x": float(bucket["bucket_x"].iloc[0]),
                "bucket_label": str(bucket["bucket_label"].iloc[0]),
                "n_runs": int(bucket["run_key"].nunique()),
                "high_role_payoff_mean": float(bucket["high_role_payoff"].mean()),
                "low_role_payoff_mean": float(bucket["low_role_payoff"].mean()),
                "mean_within_run_payoff_variance": mean_variance,
                "spread_for_errorbar": math.sqrt(mean_variance),
            }
        )
    return pd.DataFrame(rows)


def plot_summary(
    summary: pd.DataFrame,
    scenario: str,
    filename_suffix: str = "",
    figsize: tuple[float, float] = (6.2, 4.3),
    fixed_y_min: float | None = None,
    fixed_y_max: float | None = None,
    show_errorbars: bool = True,
) -> Path:
    cfg = SCENARIOS[scenario]
    summary = summary.sort_values("bucket_code").reset_index(drop=True)
    x = np.arange(len(summary))
    spread = summary["spread_for_errorbar"].to_numpy(dtype=float)
    high = summary["high_role_payoff_mean"].to_numpy(dtype=float)
    low = summary["low_role_payoff_mean"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=figsize)
    if show_errorbars:
        ax.errorbar(
            x,
            high,
            yerr=spread,
            color=cfg["high_color"],
            marker="o",
            markersize=5.2,
            linewidth=2.1,
            capsize=3.5,
            label=cfg["high_role"],
        )
        ax.errorbar(
            x,
            low,
            yerr=spread,
            color=cfg["low_color"],
            marker="o",
            markersize=5.2,
            linewidth=2.1,
            capsize=3.5,
            label=cfg["low_role"],
        )
    else:
        ax.plot(
            x,
            high,
            color=cfg["high_color"],
            marker="o",
            markersize=5.2,
            linewidth=2.3,
            label=cfg["high_role"],
        )
        ax.plot(
            x,
            low,
            color=cfg["low_color"],
            marker="o",
            markersize=5.2,
            linewidth=2.3,
            label=cfg["low_role"],
        )

    y_min = min(float(np.min(high - spread)), float(np.min(low - spread)), 0.0)
    if fixed_y_min is not None:
        y_min = fixed_y_min
    y_max = max(float(np.max(high + spread)), float(np.max(low + spread)))
    if fixed_y_max is not None:
        y_max = fixed_y_max
    pad = 0.08 * max(y_max - y_min, 1.0)
    lower = y_min if fixed_y_min is not None else y_min - pad
    upper = y_max if fixed_y_max is not None else y_max + pad
    ax.set_ylim(lower, upper)
    ax.set_xlim(-0.35, len(summary) - 0.65)
    ax.set_xticks(x)
    ax.set_xticklabels(summary["bucket_label"].tolist(), fontsize=9)
    ax.set_xlabel(cfg["x_label"], fontsize=10)
    ax.set_ylabel("Payoff", fontsize=10)
    title_suffix = "with within-run spread bars" if show_errorbars else "role payoff means"
    ax.set_title(f"{cfg['title']}: {title_suffix}", fontsize=12, pad=10)
    ax.grid(True, axis="y", alpha=0.24, linewidth=0.6)
    ax.tick_params(axis="y", labelsize=9)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    handles = [
        Line2D([0], [0], color=cfg["high_color"], marker="o", lw=2.1, label=cfg["high_role"]),
        Line2D([0], [0], color=cfg["low_color"], marker="o", lw=2.1, label=cfg["low_role"]),
    ]
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=9)
    if show_errorbars:
        ax.text(
            0.0,
            -0.23,
            "Bars show +/- sqrt(mean within-run payoff variance), not SEM.",
            transform=ax.transAxes,
            fontsize=8.5,
            color="#444444",
            ha="left",
        )
    fig.tight_layout()
    out_path = OUT_DIR / f"{cfg['file_prefix']}{filename_suffix}.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    role = pd.read_csv(ROLE_METRICS)
    run_metrics = pd.read_csv(RUN_METRICS)[["run_key", "payoff_variance"]]
    frame = role.merge(run_metrics, on="run_key", how="left", validate="many_to_one")
    if frame["payoff_variance"].isna().any():
        missing = int(frame["payoff_variance"].isna().sum())
        raise ValueError(f"Missing payoff_variance for {missing} role rows")

    summaries = []
    paths = []
    for scenario in SCENARIOS:
        summary = summarize(frame, scenario)
        summaries.append(summary)
        paths.append(plot_summary(summary, scenario))
        paths.append(
            plot_summary(
                summary,
                scenario,
                filename_suffix="_tall_y25",
                figsize=(4.4, 6.2),
                fixed_y_min=25.0,
            )
        )
        paths.append(
            plot_summary(
                summary,
                scenario,
                filename_suffix="_tall_y40_60_no_errorbars",
                figsize=(3.8, 6.6),
                fixed_y_min=40.0,
                fixed_y_max=60.0,
                show_errorbars=False,
            )
        )

    summary_path = OUT_DIR / "role_payoff_with_within_run_variance_bars_summary.csv"
    pd.concat(summaries, ignore_index=True).to_csv(summary_path, index=False)
    print(f"Wrote {summary_path}")
    for path in paths:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
