#!/usr/bin/env python3
"""Outlier sensitivity version of the endpoint fair-share role plot.

This recreates the three-game endpoint plot with four curves per panel:

    baseline / adversary x max-cooperative / max-competitive

The plotted metric is the signed symmetric percent gap from each role's own
NBS fair-share benchmark:

    200 * (actual utility - NBS utility) / (abs(actual utility) + abs(NBS utility))

This script writes a sensitivity version that drops the lowest-Elo Game 2
max-competitive model-level point.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from strong_models_experiment.analysis.active_model_roster import (  # noqa: E402
    canonical_model_name,
)


AGENT_METRICS_CSV = PROJECT_ROOT / "analysis" / "nash_lindahl_fairness_20260505" / "agent_metrics.csv"
ELO_CSV = PROJECT_ROOT / "Figures" / "game_1" / "average_utility_vs_elo.csv"
OUT_DIR = PROJECT_ROOT / "overleaf" / "neurips" / "graphics" / "n2_gpt5_nano" / "fairness_explanation"

OUT_PNG = OUT_DIR / "baseline_adversary_fair_share_symmetric_percent_endpoints_ewm_drop_game2_lowest_elo.png"
OUT_CSV = OUT_DIR / "baseline_adversary_fair_share_symmetric_percent_endpoints_drop_game2_lowest_elo_model_means.csv"
OUT_SLOPES = OUT_DIR / "baseline_adversary_fair_share_symmetric_percent_endpoints_drop_game2_lowest_elo_slopes.csv"

SOURCE_GROUP = "n2_main_gpt5_baseline"
EWM_ALPHA = 0.10
DROP_GAME2_MODEL = "llama-3.2-1b-instruct"

GAME_ORDER = ("game1", "game2", "game3")
GAME_LABELS = {
    "game1": "Game 1: Item Allocation",
    "game2": "Game 2: Diplomacy",
    "game3": "Game 3: Co-funding",
}

STYLE = {
    ("baseline", "max_cooperative"): {"color": "#2563eb", "marker": "o", "linestyle": "-", "label": "Baseline, max cooperative"},
    ("adversary", "max_cooperative"): {"color": "#0f766e", "marker": "s", "linestyle": "-", "label": "Adversary, max cooperative"},
    ("baseline", "max_competitive"): {"color": "#dc2626", "marker": "o", "linestyle": "--", "label": "Baseline, max competitive"},
    ("adversary", "max_competitive"): {"color": "#f97316", "marker": "s", "linestyle": "--", "label": "Adversary, max competitive"},
}


def sem(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) <= 1:
        return 0.0
    return float(clean.std(ddof=1) / math.sqrt(len(clean)))


def symmetric_percent(actual: pd.Series, fair: pd.Series) -> pd.Series:
    actual_num = pd.to_numeric(actual, errors="coerce")
    fair_num = pd.to_numeric(fair, errors="coerce")
    denom = actual_num.abs() + fair_num.abs()
    out = 200.0 * (actual_num - fair_num) / denom
    out = out.mask(denom <= 1e-12, 0.0)
    return out.replace([np.inf, -np.inf], np.nan)


def fit_slope(x: pd.Series, y: pd.Series) -> float:
    x_arr = pd.to_numeric(x, errors="coerce").to_numpy(float)
    y_arr = pd.to_numeric(y, errors="coerce").to_numpy(float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    if mask.sum() < 3:
        return np.nan
    slope, _ = np.polyfit(x_arr[mask], y_arr[mask], 1)
    return float(slope * 100.0)


def load_endpoint_means() -> pd.DataFrame:
    metrics = pd.read_csv(AGENT_METRICS_CSV)
    metrics = metrics[metrics["source_group"].eq(SOURCE_GROUP)].copy()

    elo = pd.read_csv(ELO_CSV)[["model", "elo"]].copy()
    elo["canonical"] = elo["model"].apply(canonical_model_name)
    elo_lookup = elo.drop_duplicates("canonical").set_index("canonical")["elo"]

    adv = metrics[metrics["role"].eq("adversary")][["result_path", "model"]].copy()
    adv["adversary_canonical"] = adv["model"].apply(canonical_model_name)
    adv["adv_elo"] = adv["adversary_canonical"].map(elo_lookup)

    metrics = metrics.merge(
        adv[["result_path", "adversary_canonical", "adv_elo"]],
        on="result_path",
        how="left",
    )
    metrics = metrics.dropna(subset=["adv_elo"])
    metrics["signed_symmetric_percent"] = symmetric_percent(metrics["actual_raw_utility"], metrics["nbs_utility"])

    endpoint_rows = []
    for game_id, sub in metrics.groupby("game_id", sort=False):
        min_comp = float(sub["competition_value"].min())
        max_comp = float(sub["competition_value"].max())
        for endpoint, comp in (("max_cooperative", min_comp), ("max_competitive", max_comp)):
            endpoint_sub = sub[np.isclose(sub["competition_value"], comp)].copy()
            endpoint_sub["endpoint"] = endpoint
            endpoint_rows.append(endpoint_sub)

    endpoint_df = pd.concat(endpoint_rows, ignore_index=True)
    endpoint_df = endpoint_df[
        ~(
            endpoint_df["game_id"].eq("game2")
            & endpoint_df["endpoint"].eq("max_competitive")
            & endpoint_df["adversary_canonical"].eq(DROP_GAME2_MODEL)
        )
    ].copy()

    means = (
        endpoint_df.groupby(
            ["game_id", "endpoint", "competition_value", "role", "adversary_canonical", "adv_elo"],
            as_index=False,
        )
        .agg(
            fair_share_gap_pct=("signed_symmetric_percent", "mean"),
            fair_share_gap_pct_sem=("signed_symmetric_percent", sem),
            n_runs=("signed_symmetric_percent", "size"),
        )
        .sort_values(["game_id", "endpoint", "role", "adv_elo"])
    )
    return means


def plot(means: pd.DataFrame) -> pd.DataFrame:
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.75), sharex=True, sharey=True)
    slope_rows: list[dict[str, float | str | int]] = []

    for ax, game_id in zip(axes, GAME_ORDER):
        sub_game = means[means["game_id"].eq(game_id)]
        for role in ("baseline", "adversary"):
            for endpoint in ("max_cooperative", "max_competitive"):
                sub = sub_game[
                    sub_game["role"].eq(role) & sub_game["endpoint"].eq(endpoint)
                ].sort_values("adv_elo")
                if sub.empty:
                    continue
                style = STYLE[(role, endpoint)]
                yerr = pd.to_numeric(sub["fair_share_gap_pct_sem"], errors="coerce").fillna(0.0).to_numpy(float)
                ax.errorbar(
                    sub["adv_elo"],
                    sub["fair_share_gap_pct"],
                    yerr=np.maximum(yerr, 0.0),
                    fmt=style["marker"],
                    markersize=4.6,
                    linewidth=0,
                    elinewidth=0.65,
                    capsize=1.5,
                    color=style["color"],
                    ecolor=style["color"],
                    alpha=0.30,
                )
                ewm = sub["fair_share_gap_pct"].ewm(alpha=EWM_ALPHA, adjust=False).mean()
                ax.plot(
                    sub["adv_elo"],
                    ewm,
                    color=style["color"],
                    linestyle=style["linestyle"],
                    linewidth=2.5,
                    alpha=0.96,
                    label=style["label"],
                )
                slope_rows.append(
                    {
                        "game_id": game_id,
                        "endpoint": endpoint,
                        "competition_value": float(sub["competition_value"].iloc[0]),
                        "role": role,
                        "n_models": int(len(sub)),
                        "slope_per_100_elo": fit_slope(sub["adv_elo"], sub["fair_share_gap_pct"]),
                        "ewm_alpha": EWM_ALPHA,
                        "dropped": f"game2 max_competitive {DROP_GAME2_MODEL}",
                    }
                )

        ax.axhline(0, color="#111827", linewidth=1.0, alpha=0.72)
        ax.grid(alpha=0.22, linewidth=0.75)
        ax.set_title(GAME_LABELS[game_id], fontsize=13.4, pad=9)
        ax.tick_params(axis="both", labelsize=10.5)
        ax.set_xlabel("Adversary Elo", fontsize=12.2, labelpad=7)

    axes[0].set_ylabel("Signed fair-share gap (%)", fontsize=12.4, labelpad=8)
    handles, labels = axes[0].get_legend_handles_labels()
    legend = fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.035),
        ncol=2,
        frameon=True,
        fontsize=10.6,
        handlelength=3.0,
    )
    legend.get_frame().set_alpha(0.94)
    legend.get_frame().set_edgecolor("#d1d5db")
    fig.suptitle("Fair-Share Role Gaps at Endpoints (drop Game 2 lowest-Elo outlier)", fontsize=15.2, y=0.98)
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.255, top=0.82, wspace=0.08)
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(slope_rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    means = load_endpoint_means()
    slopes = plot(means)
    means.to_csv(OUT_CSV, index=False)
    slopes.to_csv(OUT_SLOPES, index=False)
    print(f"Wrote {OUT_PNG}")
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_SLOPES}")


if __name__ == "__main__":
    main()
