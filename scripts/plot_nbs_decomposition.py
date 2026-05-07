#!/usr/bin/env python3
"""
=============================================================================
NBS Decomposition Figure: Utility = NBS Fair Share + NBS Residual
=============================================================================

Decomposes adversary utility into NBS fair share and NBS residual for each
game, producing a 1×3 panel scatter plot showing that the entire utility-Elo
slope lives in the residual (fair share is flat).

Usage:
    python scripts/plot_nbs_decomposition.py

What it creates:
    overleaf/neurips/graphics/n2_gpt5_nano/14_nbs_decomposition_overall.png
    overleaf/neurips/graphics/n2_gpt5_nano/14_nbs_decomposition_slopes.csv

Dependencies:
    - pandas, numpy, matplotlib, scipy
    - analysis/nash_lindahl_fairness_20260505/agent_metrics.csv
    - Figures/game_1/average_utility_vs_elo.csv (for Elo join)
    - strong_models_experiment.analysis.active_model_roster (for model name mapping)

=============================================================================
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
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from strong_models_experiment.analysis.active_model_roster import (
    canonical_model_name,
    short_model_name,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
AGENT_METRICS_CSV = PROJECT_ROOT / "analysis" / "nash_lindahl_fairness_20260505" / "agent_metrics.csv"
ELO_CSV = PROJECT_ROOT / "Figures" / "game_1" / "average_utility_vs_elo.csv"
OUT_DIR = PROJECT_ROOT / "overleaf" / "neurips" / "graphics" / "n2_gpt5_nano"

# ---------------------------------------------------------------------------
# Style constants (matching analyze_n2_baseline_comparison.py)
# ---------------------------------------------------------------------------
COLOR_UTILITY = "#b45309"   # amber – adversary utility
COLOR_NBS_FAIR = "#9ca3af"  # gray – NBS fair share
COLOR_RESIDUAL = "#dc2626"  # red – NBS residual
COLOR_FIT = "#111827"       # near-black – fit lines

GAME_LABELS = {
    "game1": "Game 1: Item allocation",
    "game2": "Game 2: Diplomacy",
    "game3": "Game 3: Co-funding",
}


def sem(s: pd.Series) -> float:
    clean = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) <= 1:
        return 0.0
    return float(clean.std(ddof=1) / math.sqrt(len(clean)))


def finite_yerr(values: pd.Series) -> np.ndarray:
    arr = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
    return np.maximum(arr, 0.0)


def style_axis(ax: plt.Axes, ylabel: str) -> None:
    ax.set_xlabel("Adversary Chatbot Arena Elo", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(alpha=0.22)
    ax.tick_params(labelsize=8)


def fit_line_stats(
    ax: plt.Axes | None,
    x: np.ndarray,
    y: np.ndarray,
    color: str,
    linestyle: str = "--",
    draw: bool = True,
) -> dict:
    """Fit OLS line, optionally draw, return slope/100 Elo, r, p."""
    mask = np.isfinite(x) & np.isfinite(y)
    x_clean, y_clean = x[mask], y[mask]
    if len(x_clean) < 3:
        return {"slope_per_100": np.nan, "r": np.nan, "p": np.nan, "intercept": np.nan}
    slope, intercept = np.polyfit(x_clean, y_clean, 1)
    r_val, p_val = stats.pearsonr(x_clean, y_clean)
    if draw and ax is not None:
        xs = np.linspace(float(x_clean.min()), float(x_clean.max()), 100)
        ax.plot(xs, slope * xs + intercept, color=color, linestyle=linestyle, linewidth=1.3, alpha=0.9)
    return {
        "slope_per_100": float(slope * 100.0),
        "r": float(r_val),
        "p": float(p_val),
        "intercept": float(intercept),
        "slope_raw": float(slope),
    }


def zero_crossing_elo(slope_raw: float, intercept: float) -> float:
    """Elo where the regression line crosses y=0."""
    if abs(slope_raw) < 1e-12:
        return np.nan
    return -intercept / slope_raw


def annotate_models(ax: plt.Axes, df: pd.DataFrame, x_col: str, y_col: str, fontsize: float = 5.4, alpha: float = 0.8) -> None:
    for _, row in df.iterrows():
        if pd.isna(row[y_col]):
            continue
        ax.annotate(
            str(row["short_name"]),
            (row[x_col], row[y_col]),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=fontsize,
            alpha=alpha,
        )


def significance_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return " (ns)"


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    metrics = pd.read_csv(AGENT_METRICS_CSV)
    elo_df = pd.read_csv(ELO_CSV)[["model", "elo"]].rename(columns={"elo": "arena_elo"})

    # Filter to N=2 GPT-5-nano baseline, adversary role
    adv = metrics[
        (metrics["source_group"] == "n2_main_gpt5_baseline")
        & (metrics["role"] == "adversary")
    ].copy()

    # ------------------------------------------------------------------
    # 2. Join Elo via canonical model name
    # ------------------------------------------------------------------
    adv["canonical"] = adv["model"].apply(canonical_model_name)
    elo_df["canonical"] = elo_df["model"].apply(canonical_model_name)
    elo_lookup = elo_df.drop_duplicates("canonical").set_index("canonical")["arena_elo"]
    adv["elo"] = adv["canonical"].map(elo_lookup)
    adv = adv.dropna(subset=["elo"])
    adv["short_name"] = adv["model"].apply(short_model_name)

    print(f"After Elo join: {len(adv)} adversary rows, {adv['model'].nunique()} models")

    # ------------------------------------------------------------------
    # 3. Aggregate to model-level means per game
    # ------------------------------------------------------------------
    group_cols = ["game_id", "model", "canonical", "short_name", "elo"]
    agg = (
        adv.groupby(group_cols, as_index=False)
        .agg(
            utility_mean=("actual_raw_utility", "mean"),
            utility_sem=("actual_raw_utility", sem),
            nbs_fair_mean=("nbs_utility", "mean"),
            nbs_fair_sem=("nbs_utility", sem),
            residual_mean=("nbs_residual", "mean"),
            residual_sem=("nbs_residual", sem),
            n_runs=("actual_raw_utility", "size"),
        )
    )

    # ------------------------------------------------------------------
    # 4. Plot: 1×3 panels
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), sharey=False)
    slope_rows: list[dict] = []

    for ax, game_id in zip(axes, ["game1", "game2", "game3"], strict=True):
        gdf = agg[agg["game_id"] == game_id].sort_values("elo").copy()
        x = gdf["elo"].to_numpy(dtype=float)

        # --- Series 1: Adversary utility (filled circles, amber) ---
        ax.errorbar(
            gdf["elo"], gdf["utility_mean"],
            yerr=finite_yerr(gdf["utility_sem"]),
            fmt="o", markersize=4.4, color=COLOR_UTILITY, ecolor=COLOR_UTILITY,
            capsize=2.0, capthick=0.7, elinewidth=0.75, alpha=0.9,
            label="Adversary utility",
        )
        util_stats = fit_line_stats(ax, x, gdf["utility_mean"].to_numpy(), COLOR_UTILITY)

        # --- Series 2: NBS fair share (open circles, gray) ---
        ax.errorbar(
            gdf["elo"], gdf["nbs_fair_mean"],
            yerr=finite_yerr(gdf["nbs_fair_sem"]),
            fmt="o", markersize=4.0, color=COLOR_NBS_FAIR, ecolor=COLOR_NBS_FAIR,
            capsize=2.0, capthick=0.7, elinewidth=0.75, alpha=0.85,
            markerfacecolor="none", markeredgewidth=1.0,
            label="NBS fair share",
        )
        nbs_stats = fit_line_stats(ax, x, gdf["nbs_fair_mean"].to_numpy(), COLOR_NBS_FAIR)

        # --- Series 3: NBS residual (filled diamonds, red) ---
        ax.errorbar(
            gdf["elo"], gdf["residual_mean"],
            yerr=finite_yerr(gdf["residual_sem"]),
            fmt="D", markersize=4.4, color=COLOR_RESIDUAL, ecolor=COLOR_RESIDUAL,
            capsize=2.0, capthick=0.7, elinewidth=0.75, alpha=0.9,
            label="NBS residual",
        )
        resid_stats = fit_line_stats(ax, x, gdf["residual_mean"].to_numpy(), COLOR_RESIDUAL)

        # --- Zero reference line ---
        ax.axhline(0, color="black", linewidth=0.6, alpha=0.4)

        # --- Zero-crossing vertical line + shading ---
        zc = zero_crossing_elo(resid_stats.get("slope_raw", 0), resid_stats.get("intercept", 0))
        xlims = (float(x.min()) - 15, float(x.max()) + 15)
        if np.isfinite(zc) and xlims[0] < zc < xlims[1]:
            ax.axvline(zc, color="#6b7280", linewidth=0.8, linestyle=":", alpha=0.5)
            ax.axvspan(xlims[0], zc, color="#3b82f6", alpha=0.06)
            ax.axvspan(zc, xlims[1], color="#ef4444", alpha=0.06)

        # --- Model name annotations (on utility series only) ---
        annotate_models(ax, gdf, "elo", "utility_mean")

        # --- Title with residual slope and r ---
        resid_slope_str = f"{resid_stats['slope_per_100']:+.2f}"
        resid_r_str = f"{resid_stats['r']:.2f}"
        ax.set_title(
            f"{GAME_LABELS[game_id]}\nresidual slope = {resid_slope_str} / 100 Elo (r = {resid_r_str})",
            fontsize=10,
        )
        style_axis(ax, "Utility points")

        # --- Legend ---
        util_label = f"Utility: {util_stats['slope_per_100']:+.2f}/100 Elo{significance_stars(util_stats['p'])}"
        nbs_label = f"NBS fair: {nbs_stats['slope_per_100']:+.2f}/100 Elo{significance_stars(nbs_stats['p'])}"
        resid_label = f"Residual: {resid_stats['slope_per_100']:+.2f}/100 Elo{significance_stars(resid_stats['p'])}"
        ax.legend(
            [
                plt.Line2D([], [], color=COLOR_UTILITY, marker="o", linestyle="--", markersize=4),
                plt.Line2D([], [], color=COLOR_NBS_FAIR, marker="o", linestyle="--", markersize=4, markerfacecolor="none"),
                plt.Line2D([], [], color=COLOR_RESIDUAL, marker="D", linestyle="--", markersize=4),
            ],
            [util_label, nbs_label, resid_label],
            fontsize=6.5,
            frameon=True,
            loc="upper left",
        )

        # --- Collect slope table rows ---
        for metric_name, st in [("utility", util_stats), ("nbs_fair_share", nbs_stats), ("nbs_residual", resid_stats)]:
            zc_val = zero_crossing_elo(st.get("slope_raw", 0), st.get("intercept", 0)) if metric_name == "nbs_residual" else np.nan
            slope_rows.append({
                "game": game_id,
                "metric": metric_name,
                "slope_per_100_elo": round(st["slope_per_100"], 2),
                "pearson_r": round(st["r"], 2),
                "p_value": st["p"],
                "zero_crossing_elo": round(zc_val) if np.isfinite(zc_val) else "",
            })

    plt.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig_path = OUT_DIR / "14_nbs_decomposition_overall.png"
    fig.savefig(fig_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {fig_path}")

    # ------------------------------------------------------------------
    # 5. Save slopes CSV
    # ------------------------------------------------------------------
    slopes_df = pd.DataFrame(slope_rows)
    csv_path = OUT_DIR / "14_nbs_decomposition_slopes.csv"
    slopes_df.to_csv(csv_path, index=False)
    print(f"Saved slopes CSV: {csv_path}")

    # ------------------------------------------------------------------
    # 6. Print summary table
    # ------------------------------------------------------------------
    print("\n--- NBS Decomposition Slopes ---")
    for _, row in slopes_df.iterrows():
        p_str = f"p={row['p_value']:.4f}" if isinstance(row["p_value"], float) else ""
        zc_str = f"  zero-crossing Elo {row['zero_crossing_elo']}" if row["zero_crossing_elo"] != "" else ""
        print(f"  {row['game']:6s} {row['metric']:16s}  {row['slope_per_100_elo']:+6.2f}/100 Elo  r={row['pearson_r']:.2f}  {p_str}{zc_str}")


if __name__ == "__main__":
    main()
