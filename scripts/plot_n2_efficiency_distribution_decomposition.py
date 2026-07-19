#!/usr/bin/env python3
"""Plot efficiency-vs-distribution decomposition for max-competition N=2 games.

For each paired N=2 result:

    baseline residual r_b = actual_baseline - NBS_baseline
    adversary residual r_a = actual_adversary - NBS_adversary

Then:

    efficiency gap E = r_b + r_a
                     = actual total utility - NBS total utility

    distributional tilt D = r_a - r_b

E tracks whether the agents created the benchmark total surplus. D tracks which
role captured more relative to its own benchmark share.
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
from matplotlib.colors import Normalize


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from strong_models_experiment.analysis.active_model_roster import (  # noqa: E402
    canonical_model_name,
)


AGENT_METRICS_CSV = PROJECT_ROOT / "analysis" / "nash_lindahl_fairness_20260505" / "agent_metrics.csv"
ELO_CSV = PROJECT_ROOT / "Figures" / "game_1" / "average_utility_vs_elo.csv"
OUT_DIR = PROJECT_ROOT / "overleaf" / "neurips" / "graphics" / "n2_gpt5_nano" / "fairness_explanation"

OUT_DECOMP_PNG = OUT_DIR / "efficiency_distribution_decomposition_2x3_max_competition.png"
OUT_PHASE_PNG = OUT_DIR / "efficiency_vs_distribution_phase_max_competition.png"
OUT_DECOMP_DROP_LOWEST_GAME2_PNG = (
    OUT_DIR / "efficiency_distribution_decomposition_2x3_max_competition_drop_game2_lowest_elo.png"
)
OUT_PHASE_DROP_LOWEST_GAME2_PNG = OUT_DIR / "efficiency_vs_distribution_phase_max_competition_drop_game2_lowest_elo.png"
OUT_MODEL_CSV = OUT_DIR / "efficiency_distribution_decomposition_max_competition_model_means.csv"
OUT_RUN_CSV = OUT_DIR / "efficiency_distribution_decomposition_max_competition_runs.csv"
OUT_SLOPES_CSV = OUT_DIR / "efficiency_distribution_decomposition_max_competition_slopes.csv"
OUT_MODEL_DROP_LOWEST_GAME2_CSV = (
    OUT_DIR / "efficiency_distribution_decomposition_max_competition_model_means_drop_game2_lowest_elo.csv"
)
OUT_SLOPES_DROP_LOWEST_GAME2_CSV = (
    OUT_DIR / "efficiency_distribution_decomposition_max_competition_slopes_drop_game2_lowest_elo.csv"
)

SOURCE_GROUP = "n2_main_gpt5_baseline"
EWM_ALPHA = 0.10

GAME_ORDER = ("game1", "game2", "game3")
GAME_LABELS = {
    "game1": "Game 1: Item Allocation",
    "game2": "Game 2: Diplomacy",
    "game3": "Game 3: Co-funding",
}
GAME_COLORS = {
    "game1": "#7c3aed",
    "game2": "#0f766e",
    "game3": "#f97316",
}
METRIC_COLORS = {
    "efficiency_gap": "#2563eb",
    "distributional_tilt": "#dc2626",
}


def sem(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) <= 1:
        return 0.0
    return float(clean.std(ddof=1) / math.sqrt(len(clean)))


def finite(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)


def fit_slope(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    x_arr = finite(x).to_numpy(float)
    y_arr = finite(y).to_numpy(float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    if mask.sum() < 3:
        return np.nan, np.nan
    slope, intercept = np.polyfit(x_arr[mask], y_arr[mask], 1)
    return float(slope * 100.0), float(intercept)


def ewm_curve(sub: pd.DataFrame, metric: str) -> pd.DataFrame:
    ordered = sub.sort_values("adv_elo").copy()
    ordered[f"{metric}_ewm"] = ordered[metric].ewm(alpha=EWM_ALPHA, adjust=False).mean()
    return ordered


def load_paired_runs() -> pd.DataFrame:
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

    max_comp = metrics.groupby("game_id")["competition_value"].max().to_dict()
    metrics = metrics[
        metrics.apply(
            lambda row: bool(np.isclose(row["competition_value"], max_comp.get(row["game_id"], np.nan))),
            axis=1,
        )
    ].copy()

    pivot = metrics.pivot_table(
        index=[
            "game_id",
            "game_label",
            "competition_value",
            "competition_label",
            "result_path",
            "adversary_canonical",
            "adv_elo",
        ],
        columns="role",
        values=["actual_raw_utility", "nbs_utility", "nbs_residual"],
        aggfunc="first",
    )
    pivot.columns = [f"{value}_{role}" for value, role in pivot.columns]
    paired = pivot.reset_index()

    required = [
        "actual_raw_utility_baseline",
        "actual_raw_utility_adversary",
        "nbs_utility_baseline",
        "nbs_utility_adversary",
        "nbs_residual_baseline",
        "nbs_residual_adversary",
    ]
    paired = paired.dropna(subset=required)

    paired["actual_total"] = paired["actual_raw_utility_baseline"] + paired["actual_raw_utility_adversary"]
    paired["nbs_total"] = paired["nbs_utility_baseline"] + paired["nbs_utility_adversary"]
    paired["efficiency_gap"] = paired["nbs_residual_baseline"] + paired["nbs_residual_adversary"]
    paired["distributional_tilt"] = paired["nbs_residual_adversary"] - paired["nbs_residual_baseline"]
    paired["adversary_advantage_actual"] = (
        paired["actual_raw_utility_adversary"] - paired["actual_raw_utility_baseline"]
    )
    paired["adversary_advantage_nbs"] = paired["nbs_utility_adversary"] - paired["nbs_utility_baseline"]

    return paired.sort_values(["game_id", "adv_elo", "result_path"]).reset_index(drop=True)


def aggregate_model_means(runs: pd.DataFrame) -> pd.DataFrame:
    grouped = runs.groupby(
        ["game_id", "game_label", "competition_value", "competition_label", "adversary_canonical", "adv_elo"],
        as_index=False,
    )
    return grouped.agg(
        n_runs=("result_path", "size"),
        actual_total=("actual_total", "mean"),
        nbs_total=("nbs_total", "mean"),
        efficiency_gap=("efficiency_gap", "mean"),
        efficiency_gap_sem=("efficiency_gap", sem),
        distributional_tilt=("distributional_tilt", "mean"),
        distributional_tilt_sem=("distributional_tilt", sem),
        baseline_residual=("nbs_residual_baseline", "mean"),
        adversary_residual=("nbs_residual_adversary", "mean"),
        baseline_actual=("actual_raw_utility_baseline", "mean"),
        adversary_actual=("actual_raw_utility_adversary", "mean"),
        baseline_nbs=("nbs_utility_baseline", "mean"),
        adversary_nbs=("nbs_utility_adversary", "mean"),
    )


def padded_limits(values: pd.Series, center_zero: bool = False) -> tuple[float, float]:
    clean = finite(values).dropna()
    if clean.empty:
        return -1.0, 1.0
    lo = float(clean.min())
    hi = float(clean.max())
    if center_zero:
        lo = min(lo, 0.0)
        hi = max(hi, 0.0)
    pad = max((hi - lo) * 0.10, 4.0)
    return lo - pad, hi + pad


def plot_decomposition(
    model_means: pd.DataFrame,
    out_png: Path,
    out_slopes_csv: Path,
    title_suffix: str = "",
) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(12.0, 6.8),
        sharex=True,
        gridspec_kw={"hspace": 0.16, "wspace": 0.14},
    )

    eff_ylim = padded_limits(model_means["efficiency_gap"], center_zero=True)
    dist_ylim = padded_limits(model_means["distributional_tilt"], center_zero=True)

    slope_rows: list[dict[str, float | str | int]] = []

    for col, game_id in enumerate(GAME_ORDER):
        sub = model_means[model_means["game_id"].eq(game_id)].sort_values("adv_elo")
        for row, metric in enumerate(("efficiency_gap", "distributional_tilt")):
            ax = axes[row, col]
            color = METRIC_COLORS[metric]
            ewm = ewm_curve(sub, metric)
            slope, intercept = fit_slope(sub["adv_elo"], sub[metric])
            slope_rows.append(
                {
                    "game_id": game_id,
                    "game_label": GAME_LABELS[game_id],
                    "metric": metric,
                    "competition_value": float(sub["competition_value"].iloc[0]) if len(sub) else np.nan,
                    "n_models": int(len(sub)),
                    "slope_per_100_elo": slope,
                    "intercept": intercept,
                    "ewm_alpha": EWM_ALPHA,
                }
            )

            yerr = finite(sub[f"{metric}_sem"]).fillna(0.0).to_numpy(float)
            ax.errorbar(
                sub["adv_elo"],
                sub[metric],
                yerr=np.maximum(yerr, 0.0),
                fmt="o",
                markersize=4.6,
                linewidth=0,
                elinewidth=0.7,
                capsize=1.8,
                color=color,
                ecolor=color,
                alpha=0.34,
            )
            ax.plot(
                ewm["adv_elo"],
                ewm[f"{metric}_ewm"],
                color=color,
                linewidth=2.8,
                alpha=0.96,
            )
            ax.axhline(0, color="#111827", linewidth=1.0, alpha=0.68)
            ax.grid(alpha=0.22, linewidth=0.75)
            ax.tick_params(axis="both", labelsize=10.5)
            if row == 0:
                ax.set_title(GAME_LABELS[game_id], fontsize=13.5, pad=8)
                ax.set_ylim(eff_ylim)
            else:
                ax.set_ylim(dist_ylim)
                ax.set_xlabel("Adversary Elo", fontsize=12.0, labelpad=7)
            if col == 0 and row == 0:
                ax.set_ylabel("Efficiency gap E\nactual total - NBS total", fontsize=12.0, labelpad=8)
            if col == 0 and row == 1:
                ax.set_ylabel("Distributional tilt D\nadv. residual - base residual", fontsize=12.0, labelpad=8)

    title = "Efficiency and Distribution at Max Competition"
    if title_suffix:
        title = f"{title} {title_suffix}"
    fig.suptitle(title, fontsize=15.5, y=0.985)
    fig.text(
        0.5,
        0.012,
        "Points are adversary-model means over model order/runs; solid curves are EWM trends.",
        ha="center",
        va="bottom",
        fontsize=10.5,
        color="#374151",
    )
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.125, top=0.88, hspace=0.16, wspace=0.14)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame(slope_rows).to_csv(out_slopes_csv, index=False)


def plot_phase(model_means: pd.DataFrame, out_png: Path, title_suffix: str = "") -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(12.0, 4.35),
        sharex=True,
        sharey=True,
        gridspec_kw={"wspace": 0.14},
    )

    xlim = padded_limits(model_means["efficiency_gap"], center_zero=True)
    ylim = padded_limits(model_means["distributional_tilt"], center_zero=True)
    norm = Normalize(vmin=float(model_means["adv_elo"].min()), vmax=float(model_means["adv_elo"].max()))
    cmap = plt.get_cmap("viridis")

    for ax, game_id in zip(axes, GAME_ORDER):
        sub = model_means[model_means["game_id"].eq(game_id)].sort_values("adv_elo")
        ax.plot(
            sub["efficiency_gap"],
            sub["distributional_tilt"],
            color="#6b7280",
            linewidth=1.15,
            alpha=0.38,
            zorder=1,
        )
        sc = ax.scatter(
            sub["efficiency_gap"],
            sub["distributional_tilt"],
            c=sub["adv_elo"],
            cmap=cmap,
            norm=norm,
            s=44,
            alpha=0.92,
            edgecolor="white",
            linewidth=0.45,
            zorder=2,
        )
        ax.axvline(0, color="#111827", linewidth=1.0, alpha=0.62)
        ax.axhline(0, color="#111827", linewidth=1.0, alpha=0.62)
        ax.set_title(GAME_LABELS[game_id], fontsize=13.5, pad=8)
        ax.grid(alpha=0.22, linewidth=0.75)
        ax.tick_params(axis="both", labelsize=10.5)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel("Efficiency gap E", fontsize=12.0, labelpad=7)
        if ax is axes[0]:
            ax.set_ylabel("Distributional tilt D", fontsize=12.0, labelpad=8)

    cbar = fig.colorbar(sc, ax=axes, orientation="vertical", fraction=0.025, pad=0.018)
    cbar.set_label("Adversary Elo", fontsize=11.5)
    cbar.ax.tick_params(labelsize=10)

    title = "Max-Competition Trajectories: Efficiency Gap vs Distributional Tilt"
    if title_suffix:
        title = f"{title} {title_suffix}"
    fig.suptitle(title, fontsize=15.5, y=0.99)
    fig.text(
        0.455,
        0.045,
        "Left-to-right movement means total outcome approaches/exceeds NBS total; upward movement means adversary-favored division.",
        ha="center",
        va="bottom",
        fontsize=10.0,
        color="#374151",
    )
    fig.subplots_adjust(left=0.07, right=0.88, bottom=0.22, top=0.78, wspace=0.14)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    runs = load_paired_runs()
    model_means = aggregate_model_means(runs)

    runs.to_csv(OUT_RUN_CSV, index=False)
    model_means.to_csv(OUT_MODEL_CSV, index=False)

    plot_decomposition(model_means, OUT_DECOMP_PNG, OUT_SLOPES_CSV)
    plot_phase(model_means, OUT_PHASE_PNG)

    drop_game2_lowest = model_means[
        ~(
            model_means["game_id"].eq("game2")
            & model_means["adversary_canonical"].eq("llama-3.2-1b-instruct")
        )
    ].copy()
    drop_game2_lowest.to_csv(OUT_MODEL_DROP_LOWEST_GAME2_CSV, index=False)
    plot_decomposition(
        drop_game2_lowest,
        OUT_DECOMP_DROP_LOWEST_GAME2_PNG,
        OUT_SLOPES_DROP_LOWEST_GAME2_CSV,
        "(drop Game 2 lowest-Elo outlier)",
    )
    plot_phase(
        drop_game2_lowest,
        OUT_PHASE_DROP_LOWEST_GAME2_PNG,
        "(drop Game 2 lowest-Elo outlier)",
    )

    print(f"Wrote {OUT_DECOMP_PNG}")
    print(f"Wrote {OUT_PHASE_PNG}")
    print(f"Wrote {OUT_MODEL_CSV}")
    print(f"Wrote {OUT_RUN_CSV}")
    print(f"Wrote {OUT_SLOPES_CSV}")
    print(f"Wrote {OUT_DECOMP_DROP_LOWEST_GAME2_PNG}")
    print(f"Wrote {OUT_PHASE_DROP_LOWEST_GAME2_PNG}")
    print(f"Wrote {OUT_MODEL_DROP_LOWEST_GAME2_CSV}")
    print(f"Wrote {OUT_SLOPES_DROP_LOWEST_GAME2_CSV}")


if __name__ == "__main__":
    main()
