#!/usr/bin/env python3
"""All-competition-level signed fair-share plot for N=2 games.

This is the all-level analogue of the endpoint plot. It uses:

- color: competition level, normalized within each game from max cooperative to
  max competitive
- role style: baseline is solid/filled; adversary is dashed/hollow
- y value: signed symmetric percent gap from each role's NBS fair share
- smoothing: Elo-ordered EWM

The Game 2 max-competitive lowest-Elo outlier is dropped to match the corrected
endpoint figure.
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
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from strong_models_experiment.analysis.active_model_roster import (  # noqa: E402
    canonical_model_name,
)


AGENT_METRICS_CSV = PROJECT_ROOT / "analysis" / "nash_lindahl_fairness_20260505" / "agent_metrics.csv"
ELO_CSV = PROJECT_ROOT / "Figures" / "game_1" / "average_utility_vs_elo.csv"
OUT_DIR = PROJECT_ROOT / "overleaf" / "neurips" / "graphics" / "n2_gpt5_nano" / "fairness_explanation"

OUT_PNG = OUT_DIR / "baseline_adversary_fair_share_symmetric_percent_all_competition_levels_ewm_drop_game2_lowest_elo_style_matched_baseline_elo.png"
OUT_CSV = OUT_DIR / "baseline_adversary_fair_share_symmetric_percent_all_competition_levels_ewm_drop_game2_lowest_elo_model_means.csv"
OUT_CELLS = OUT_DIR / "baseline_adversary_fair_share_symmetric_percent_all_competition_levels_ewm_drop_game2_lowest_elo_cells.csv"

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

COMP_CMAP = LinearSegmentedColormap.from_list(
    "cooperative_to_competitive",
    ["#2563eb", "#14b8a6", "#f59e0b", "#dc2626"],
)


def symmetric_percent(actual: pd.Series, fair: pd.Series) -> pd.Series:
    actual_num = pd.to_numeric(actual, errors="coerce")
    fair_num = pd.to_numeric(fair, errors="coerce")
    denom = actual_num.abs() + fair_num.abs()
    out = 200.0 * (actual_num - fair_num) / denom
    out = out.mask(denom <= 1e-12, 0.0)
    return out.replace([np.inf, -np.inf], np.nan)


def baseline_elo() -> float:
    elo = pd.read_csv(ELO_CSV)[["model", "elo"]].copy()
    elo["canonical"] = elo["model"].apply(canonical_model_name)
    row = elo[elo["canonical"].eq(BASELINE_CANONICAL)]
    if row.empty:
        raise RuntimeError(f"Could not find baseline Elo for {BASELINE_CANONICAL}")
    return float(row["elo"].iloc[0])


def load_model_means() -> pd.DataFrame:
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
    metrics["signed_relative_gap_pct"] = symmetric_percent(metrics["actual_raw_utility"], metrics["nbs_utility"])

    metrics = metrics[
        ~(
            metrics["game_id"].eq("game2")
            & np.isclose(metrics["competition_value"], metrics[metrics["game_id"].eq("game2")]["competition_value"].max())
            & metrics["adversary_canonical"].eq(DROP_GAME2_MODEL)
        )
    ].copy()

    means = (
        metrics.groupby(
            ["game_id", "competition_value", "role", "adversary_canonical", "adv_elo"],
            as_index=False,
        )
        .agg(
            signed_relative_gap_pct=("signed_relative_gap_pct", "mean"),
            n_runs=("signed_relative_gap_pct", "size"),
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


def padded_ylim(values: pd.Series) -> tuple[float, float]:
    clean = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return -1.0, 1.0
    lo = min(float(clean.min()), 0.0)
    hi = max(float(clean.max()), 0.0)
    pad = max((hi - lo) * 0.08, 4.0)
    return lo - pad, hi + pad


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


def plot(means: pd.DataFrame) -> pd.DataFrame:
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
                y = sub["signed_relative_gap_pct"].ewm(alpha=EWM_ALPHA, adjust=False).mean()
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
                        "plot": OUT_PNG.name,
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
        ax.grid(alpha=0.22, linewidth=0.8)
        ax.set_title(GAME_LABELS[game_id], fontsize=22, pad=11)
        ax.set_xlabel("Adversary Elo", fontsize=15, labelpad=8)
        ax.set_ylabel("Signed relative fair-share gap (%)", fontsize=15, labelpad=10)
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
        "Baseline and adversary signed relative fair-share gap: all competition levels",
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
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(cell_rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    means = load_model_means()
    cells = plot(means)
    means.to_csv(OUT_CSV, index=False)
    cells.to_csv(OUT_CELLS, index=False)
    print(f"Wrote {OUT_PNG}")
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_CELLS}")


if __name__ == "__main__":
    main()
