#!/usr/bin/env python3
"""Signed absolute fair-share gap plots for N=2 games.

This creates two corrected plots:

1. All competition levels.
2. Endpoint-only max cooperative / max competitive levels.

The plotted metric is the signed absolute utility gap:

    actual_raw_utility - nbs_utility

Positive values mean the role is above its fair-share benchmark. Negative values
mean it is below fair share. This is the utility-point version of the signed
relative percentage plots.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from strong_models_experiment.analysis.active_model_roster import (  # noqa: E402
    canonical_model_name,
)


AGENT_METRICS_CSV = PROJECT_ROOT / "analysis" / "nash_lindahl_fairness_20260505" / "agent_metrics.csv"
ELO_CSV = PROJECT_ROOT / "Figures" / "game_1" / "average_utility_vs_elo.csv"
OUT_DIR = PROJECT_ROOT / "overleaf" / "neurips" / "graphics" / "n2_gpt5_nano" / "fairness_explanation"

OUT_ALL_PNG = OUT_DIR / "baseline_adversary_fair_share_absolute_gap_all_competition_levels_ewm_drop_game2_lowest_elo_style_matched_baseline_elo.png"
OUT_ENDPOINT_PNG = OUT_DIR / "baseline_adversary_fair_share_absolute_gap_endpoints_tall_ewm_drop_game2_lowest_elo_style_matched_baseline_elo.png"
OUT_ALL_CSV = OUT_DIR / "baseline_adversary_fair_share_absolute_gap_all_competition_levels_ewm_drop_game2_lowest_elo_model_means.csv"
OUT_ENDPOINT_CSV = OUT_DIR / "baseline_adversary_fair_share_absolute_gap_endpoints_tall_ewm_drop_game2_lowest_elo_model_means.csv"
OUT_ALL_CELLS = OUT_DIR / "baseline_adversary_fair_share_absolute_gap_all_competition_levels_ewm_drop_game2_lowest_elo_cells.csv"
OUT_ENDPOINT_CELLS = OUT_DIR / "baseline_adversary_fair_share_absolute_gap_endpoints_tall_ewm_drop_game2_lowest_elo_cells.csv"

SOURCE_GROUP = "n2_main_gpt5_baseline"
DROP_GAME2_MODEL = "llama-3.2-1b-instruct"
BASELINE_CANONICAL = "gpt-5-nano-high"
EWM_ALPHA = 0.10

GAME_ORDER = ("game1", "game2", "game3")
GAME_LABELS = {
    "game1": "Game 1",
    "game2": "Game 2",
    "game3": "Game 3",
}
ENDPOINT_COLORS = {
    "max_cooperative": "#2563eb",
    "max_competitive": "#dc2626",
}
ENDPOINT_LABELS = {
    "max_cooperative": "Max Cooperative",
    "max_competitive": "Max Competitive",
}
COMP_CMAP = LinearSegmentedColormap.from_list(
    "cooperative_to_competitive",
    ["#2563eb", "#14b8a6", "#f59e0b", "#dc2626"],
)


def baseline_elo() -> float:
    elo = pd.read_csv(ELO_CSV)[["model", "elo"]].copy()
    elo["canonical"] = elo["model"].apply(canonical_model_name)
    row = elo[elo["canonical"].eq(BASELINE_CANONICAL)]
    if row.empty:
        raise RuntimeError(f"Could not find baseline Elo for {BASELINE_CANONICAL}")
    return float(row["elo"].iloc[0])


def load_metrics() -> pd.DataFrame:
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
    metrics = metrics.dropna(subset=["adv_elo", "nbs_residual"])

    game2_max_comp = float(metrics[metrics["game_id"].eq("game2")]["competition_value"].max())
    metrics = metrics[
        ~(
            metrics["game_id"].eq("game2")
            & np.isclose(metrics["competition_value"], game2_max_comp)
            & metrics["adversary_canonical"].eq(DROP_GAME2_MODEL)
        )
    ].copy()
    return metrics


def aggregate(means_source: pd.DataFrame) -> pd.DataFrame:
    means = (
        means_source.groupby(
            ["game_id", "competition_value", "role", "adversary_canonical", "adv_elo"],
            as_index=False,
        )
        .agg(
            signed_absolute_gap=("nbs_residual", "mean"),
            n_runs=("nbs_residual", "size"),
        )
        .sort_values(["game_id", "competition_value", "role", "adv_elo"])
    )
    comp_minmax = (
        means.groupby("game_id")["competition_value"]
        .agg(["min", "max"])
        .rename(columns={"min": "comp_min", "max": "comp_max"})
        .reset_index()
    )
    means = means.merge(comp_minmax, on="game_id", how="left")
    denom = (means["comp_max"] - means["comp_min"]).replace(0, np.nan)
    means["competition_norm"] = ((means["competition_value"] - means["comp_min"]) / denom).fillna(0.0)
    return means


def endpoint_subset(metrics: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for game_id, sub in metrics.groupby("game_id", sort=False):
        min_comp = float(sub["competition_value"].min())
        max_comp = float(sub["competition_value"].max())
        for endpoint, comp in (("max_cooperative", min_comp), ("max_competitive", max_comp)):
            tmp = sub[np.isclose(sub["competition_value"], comp)].copy()
            tmp["endpoint"] = endpoint
            frames.append(tmp)
    endpoints = pd.concat(frames, ignore_index=True)
    means = (
        endpoints.groupby(
            ["game_id", "endpoint", "competition_value", "role", "adversary_canonical", "adv_elo"],
            as_index=False,
        )
        .agg(
            signed_absolute_gap=("nbs_residual", "mean"),
            n_runs=("nbs_residual", "size"),
        )
        .sort_values(["game_id", "endpoint", "role", "adv_elo"])
    )
    return means


def padded_ylim(values: pd.Series) -> tuple[float, float]:
    clean = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return -1.0, 1.0
    lo = min(float(clean.min()), 0.0)
    hi = max(float(clean.max()), 0.0)
    pad = max((hi - lo) * 0.10, 4.0)
    return lo - pad, hi + pad


def add_baseline_line(ax: plt.Axes, base_elo: float) -> None:
    ax.axvline(base_elo, color="#64748b", linestyle=(0, (4, 4)), linewidth=1.1, alpha=0.32)
    ax.text(
        base_elo + 4,
        0.965,
        f"baseline Elo {base_elo:.0f}",
        transform=ax.get_xaxis_transform(),
        ha="left",
        va="top",
        fontsize=8.8,
        color="#64748b",
        alpha=0.74,
        rotation=90,
    )


def role_handle(role: str) -> mlines.Line2D:
    if role == "baseline":
        return mlines.Line2D(
            [],
            [],
            color="#111827",
            marker="o",
            markerfacecolor="#111827",
            markeredgecolor="#111827",
            linestyle="-",
            linewidth=2.7,
            markersize=5.5,
            label="Baseline",
        )
    return mlines.Line2D(
        [],
        [],
        color="#111827",
        marker="o",
        markerfacecolor="white",
        markeredgecolor="#111827",
        linestyle="--",
        linewidth=2.7,
        markersize=5.5,
        label="Adversary",
    )


def endpoint_handle(endpoint: str) -> mlines.Line2D:
    return mlines.Line2D(
        [],
        [],
        color=ENDPOINT_COLORS[endpoint],
        marker="o",
        markerfacecolor=ENDPOINT_COLORS[endpoint],
        markeredgecolor=ENDPOINT_COLORS[endpoint],
        linestyle="-",
        linewidth=3.0,
        markersize=5.5,
        label=ENDPOINT_LABELS[endpoint],
    )


def plot_all_competition(means: pd.DataFrame) -> pd.DataFrame:
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 8.2), sharex=False, sharey=False)
    base_elo = baseline_elo()
    cell_rows: list[dict[str, object]] = []

    for ax, game_id in zip(axes, GAME_ORDER):
        game_sub = means[means["game_id"].eq(game_id)].copy()
        plotted_values: list[float] = []
        comp_levels = sorted(float(x) for x in game_sub["competition_value"].dropna().unique())

        for comp in comp_levels:
            comp_sub = game_sub[np.isclose(game_sub["competition_value"], comp)]
            comp_norm = float(comp_sub["competition_norm"].iloc[0])
            color = COMP_CMAP(comp_norm)
            for role in ("baseline", "adversary"):
                sub = comp_sub[comp_sub["role"].eq(role)].sort_values("adv_elo")
                if sub.empty:
                    continue
                y = sub["signed_absolute_gap"].ewm(alpha=EWM_ALPHA, adjust=False).mean()
                plotted_values.extend(y.dropna().tolist())
                is_baseline = role == "baseline"
                ax.plot(
                    sub["adv_elo"],
                    y,
                    color=color,
                    linestyle="-" if is_baseline else "--",
                    linewidth=2.15 if is_baseline else 2.0,
                    marker="o",
                    markersize=3.7,
                    markerfacecolor=color if is_baseline else "white",
                    markeredgecolor=color,
                    markeredgewidth=1.0,
                    alpha=0.88 if comp in (min(comp_levels), max(comp_levels)) else 0.66,
                )
                cell_rows.append(
                    {
                        "plot": OUT_ALL_PNG.name,
                        "game_id": game_id,
                        "competition_value": comp,
                        "competition_norm": comp_norm,
                        "role": role,
                        "n_points": int(len(sub)),
                        "ewm_alpha": EWM_ALPHA,
                        "dropped": f"game2 max_competitive {DROP_GAME2_MODEL}",
                    }
                )

        ax.axhline(0, color="#6b7280", linewidth=1.05, alpha=0.78)
        add_baseline_line(ax, base_elo)
        ax.grid(alpha=0.22, linewidth=0.8)
        ax.set_title(GAME_LABELS[game_id], fontsize=22, pad=11)
        ax.set_xlabel("Adversary Elo", fontsize=15, labelpad=8)
        ax.set_ylabel("Fair-share gap (utility points)", fontsize=15, labelpad=10)
        ax.tick_params(axis="both", labelsize=12)
        ax.set_xlim(1088, 1515)
        ax.set_ylim(*padded_ylim(pd.Series(plotted_values)))
        levels_text = ", ".join(f"{x:g}" for x in comp_levels)
        ax.text(
            0.06,
            0.045,
            f"EWM alpha={EWM_ALPHA:.2f}\nlevels: {levels_text}",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=9.1,
            color="#475569",
        )

    fig.suptitle(
        "Baseline and adversary fair-share gap: all competition levels",
        fontsize=24.5,
        y=0.985,
    )
    role_legend = fig.legend(
        handles=[role_handle("baseline"), role_handle("adversary")],
        title="Role",
        loc="upper center",
        bbox_to_anchor=(0.39, 0.895),
        ncol=2,
        fontsize=12.4,
        title_fontsize=12.4,
        frameon=True,
        handlelength=2.1,
        columnspacing=1.5,
    )
    role_legend.get_frame().set_edgecolor("#d1d5db")
    role_legend.get_frame().set_alpha(0.96)

    sm = ScalarMappable(norm=Normalize(vmin=0.0, vmax=1.0), cmap=COMP_CMAP)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation="vertical", fraction=0.026, pad=0.018)
    cbar.set_label("Competition level,\nnormalized within game", fontsize=12.0)
    cbar.set_ticks([0.0, 1.0])
    cbar.set_ticklabels(["Max cooperative", "Max competitive"])
    cbar.ax.tick_params(labelsize=11.5)

    fig.subplots_adjust(left=0.075, right=0.895, bottom=0.085, top=0.74, wspace=0.30)
    fig.savefig(OUT_ALL_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(cell_rows)


def plot_endpoints(means: pd.DataFrame) -> pd.DataFrame:
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(1, 3, figsize=(10.66, 9.22), sharex=False, sharey=False)
    base_elo = baseline_elo()
    cell_rows: list[dict[str, object]] = []

    for ax, game_id in zip(axes, GAME_ORDER):
        game_sub = means[means["game_id"].eq(game_id)].copy()
        plotted_values: list[float] = []
        for endpoint in ("max_cooperative", "max_competitive"):
            endpoint_sub = game_sub[game_sub["endpoint"].eq(endpoint)]
            color = ENDPOINT_COLORS[endpoint]
            for role in ("baseline", "adversary"):
                sub = endpoint_sub[endpoint_sub["role"].eq(role)].sort_values("adv_elo")
                if sub.empty:
                    continue
                y = sub["signed_absolute_gap"].ewm(alpha=EWM_ALPHA, adjust=False).mean()
                plotted_values.extend(y.dropna().tolist())
                is_baseline = role == "baseline"
                ax.plot(
                    sub["adv_elo"],
                    y,
                    color=color,
                    linestyle="-" if is_baseline else "--",
                    linewidth=3.0,
                    marker="o",
                    markersize=5.2,
                    markerfacecolor=color if is_baseline else "white",
                    markeredgecolor=color,
                    markeredgewidth=1.2,
                    alpha=0.96,
                )
                cell_rows.append(
                    {
                        "plot": OUT_ENDPOINT_PNG.name,
                        "game_id": game_id,
                        "endpoint": endpoint,
                        "competition_value": float(sub["competition_value"].iloc[0]),
                        "role": role,
                        "n_points": int(len(sub)),
                        "ewm_alpha": EWM_ALPHA,
                        "dropped": f"game2 max_competitive {DROP_GAME2_MODEL}",
                    }
                )

        ax.axhline(0, color="#6b7280", linewidth=1.1, alpha=0.82)
        add_baseline_line(ax, base_elo)
        ax.grid(alpha=0.23, linewidth=0.8)
        ax.set_title(GAME_LABELS[game_id], fontsize=22, pad=11)
        ax.set_xlabel("Adversary Elo", fontsize=15, labelpad=8)
        ax.set_ylabel("Fair-share gap (utility points)", fontsize=15, labelpad=10)
        ax.tick_params(axis="both", labelsize=12)
        ax.set_xlim(1088, 1515)
        ax.set_ylim(*padded_ylim(pd.Series(plotted_values)))
        ax.text(
            0.06,
            0.045,
            f"EWM alpha={EWM_ALPHA:.2f}",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=10.5,
            color="#475569",
        )

    fig.suptitle(
        "Baseline and adversary fair-share gap: endpoint EWM trends",
        fontsize=25.5,
        y=0.985,
    )
    endpoint_legend = fig.legend(
        handles=[endpoint_handle("max_cooperative"), endpoint_handle("max_competitive")],
        title="Endpoint Competition",
        loc="upper center",
        bbox_to_anchor=(0.37, 0.875),
        ncol=2,
        fontsize=12.5,
        title_fontsize=12.5,
        frameon=True,
        handlelength=2.1,
        columnspacing=1.5,
    )
    endpoint_legend.get_frame().set_edgecolor("#d1d5db")
    endpoint_legend.get_frame().set_alpha(0.96)

    role_legend = fig.legend(
        handles=[role_handle("baseline"), role_handle("adversary")],
        title="Role",
        loc="upper center",
        bbox_to_anchor=(0.72, 0.875),
        ncol=2,
        fontsize=12.5,
        title_fontsize=12.5,
        frameon=True,
        handlelength=2.1,
        columnspacing=1.5,
    )
    role_legend.get_frame().set_edgecolor("#d1d5db")
    role_legend.get_frame().set_alpha(0.96)

    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.075, top=0.70, wspace=0.35)
    fig.savefig(OUT_ENDPOINT_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(cell_rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics = load_metrics()
    all_means = aggregate(metrics)
    endpoint_means = endpoint_subset(metrics)
    all_cells = plot_all_competition(all_means)
    endpoint_cells = plot_endpoints(endpoint_means)
    all_means.to_csv(OUT_ALL_CSV, index=False)
    endpoint_means.to_csv(OUT_ENDPOINT_CSV, index=False)
    all_cells.to_csv(OUT_ALL_CELLS, index=False)
    endpoint_cells.to_csv(OUT_ENDPOINT_CELLS, index=False)
    print(f"Wrote {OUT_ALL_PNG}")
    print(f"Wrote {OUT_ENDPOINT_PNG}")
    print(f"Wrote {OUT_ALL_CSV}")
    print(f"Wrote {OUT_ENDPOINT_CSV}")
    print(f"Wrote {OUT_ALL_CELLS}")
    print(f"Wrote {OUT_ENDPOINT_CELLS}")


if __name__ == "__main__":
    main()
