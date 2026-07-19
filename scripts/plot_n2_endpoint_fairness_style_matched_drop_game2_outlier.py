#!/usr/bin/env python3
"""Style-matched endpoint fair-share plot with Game 2 outlier removed.

This recreates the visual format of:

    baseline_adversary_fair_share_symmetric_percent_endpoints_tall_ewm.png

but removes the lowest-Elo Game 2 max-competitive model point
(`llama-3.2-1b-instruct`) from the max-competitive baseline/adversary curves.
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


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from strong_models_experiment.analysis.active_model_roster import (  # noqa: E402
    canonical_model_name,
)


AGENT_METRICS_CSV = PROJECT_ROOT / "analysis" / "nash_lindahl_fairness_20260505" / "agent_metrics.csv"
ELO_CSV = PROJECT_ROOT / "Figures" / "game_1" / "average_utility_vs_elo.csv"
OUT_DIR = PROJECT_ROOT / "overleaf" / "neurips" / "graphics" / "n2_gpt5_nano" / "fairness_explanation"

OUT_PNG = OUT_DIR / "baseline_adversary_fair_share_symmetric_percent_endpoints_tall_ewm_drop_game2_lowest_elo_style_matched.png"
OUT_BASELINE_ELO_PNG = (
    OUT_DIR
    / "baseline_adversary_fair_share_symmetric_percent_endpoints_tall_ewm_drop_game2_lowest_elo_style_matched_baseline_elo.png"
)
OUT_SYMMETRIC_BASELINE_ELO_PNG = (
    OUT_DIR
    / "baseline_adversary_fair_share_symmetric_percent_endpoints_tall_symmetric_ewm_drop_game2_lowest_elo_style_matched_baseline_elo.png"
)
OUT_CSV = OUT_DIR / "baseline_adversary_fair_share_symmetric_percent_endpoints_tall_ewm_drop_game2_lowest_elo_style_matched_model_means.csv"
OUT_CELLS = OUT_DIR / "baseline_adversary_fair_share_symmetric_percent_endpoints_tall_ewm_drop_game2_lowest_elo_style_matched_cells.csv"

SOURCE_GROUP = "n2_main_gpt5_baseline"
DROP_GAME2_MODEL = "llama-3.2-1b-instruct"
EWM_ALPHA = 0.10
BASELINE_CANONICAL = "gpt-5-nano-high"

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


def symmetric_percent(actual: pd.Series, fair: pd.Series) -> pd.Series:
    actual_num = pd.to_numeric(actual, errors="coerce")
    fair_num = pd.to_numeric(fair, errors="coerce")
    denom = actual_num.abs() + fair_num.abs()
    out = 200.0 * (actual_num - fair_num) / denom
    out = out.mask(denom <= 1e-12, 0.0)
    return out.replace([np.inf, -np.inf], np.nan)


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

    endpoint_frames: list[pd.DataFrame] = []
    for game_id, sub in metrics.groupby("game_id", sort=False):
        min_comp = float(sub["competition_value"].min())
        max_comp = float(sub["competition_value"].max())
        for endpoint, comp in (("max_cooperative", min_comp), ("max_competitive", max_comp)):
            endpoint_sub = sub[np.isclose(sub["competition_value"], comp)].copy()
            endpoint_sub["endpoint"] = endpoint
            endpoint_frames.append(endpoint_sub)

    endpoints = pd.concat(endpoint_frames, ignore_index=True)
    endpoints = endpoints[
        ~(
            endpoints["game_id"].eq("game2")
            & endpoints["endpoint"].eq("max_competitive")
            & endpoints["adversary_canonical"].eq(DROP_GAME2_MODEL)
        )
    ].copy()

    means = (
        endpoints.groupby(
            ["game_id", "endpoint", "competition_value", "role", "adversary_canonical", "adv_elo"],
            as_index=False,
        )
        .agg(
            signed_relative_gap_pct=("signed_relative_gap_pct", "mean"),
            n_runs=("signed_relative_gap_pct", "size"),
        )
        .sort_values(["game_id", "endpoint", "role", "adv_elo"])
    )
    return means


def endpoint_handle(endpoint: str) -> mlines.Line2D:
    return mlines.Line2D(
        [],
        [],
        color=ENDPOINT_COLORS[endpoint],
        marker="o",
        linestyle="-",
        linewidth=3.2,
        markersize=6.0,
        label=ENDPOINT_LABELS[endpoint],
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
            linewidth=3.2,
            markersize=6.0,
            label="Baseline",
        )
    return mlines.Line2D(
        [],
        [],
        color="#111827",
        marker="o",
        markerfacecolor="white",
        markeredgecolor="#111827",
        markeredgewidth=1.4,
        linestyle="--",
        linewidth=3.2,
        markersize=6.0,
        label="Adversary",
    )


def padded_ylim(values: pd.Series) -> tuple[float, float]:
    clean = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return -1.0, 1.0
    lo = float(clean.min())
    hi = float(clean.max())
    lo = min(lo, 0.0)
    hi = max(hi, 0.0)
    pad = max((hi - lo) * 0.10, 4.0)
    return lo - pad, hi + pad


def baseline_elo() -> float:
    elo = pd.read_csv(ELO_CSV)[["model", "elo"]].copy()
    elo["canonical"] = elo["model"].apply(canonical_model_name)
    row = elo[elo["canonical"].eq(BASELINE_CANONICAL)]
    if row.empty:
        raise RuntimeError(f"Could not find baseline Elo for {BASELINE_CANONICAL}")
    return float(row["elo"].iloc[0])


def smooth_series(values: pd.Series, symmetric: bool = False) -> pd.Series:
    forward = values.ewm(alpha=EWM_ALPHA, adjust=False).mean()
    if not symmetric:
        return forward
    backward = values.iloc[::-1].ewm(alpha=EWM_ALPHA, adjust=False).mean().iloc[::-1]
    backward.index = values.index
    return (forward + backward) / 2.0


def plot(
    means: pd.DataFrame,
    out_png: Path,
    show_baseline_elo: bool = False,
    symmetric_smoothing: bool = False,
) -> pd.DataFrame:
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(1, 3, figsize=(10.66, 9.22), sharex=False, sharey=False)
    cell_rows: list[dict[str, object]] = []
    base_elo = baseline_elo()

    for ax, game_id in zip(axes, GAME_ORDER):
        game_sub = means[means["game_id"].eq(game_id)].copy()
        plotted_values: list[float] = []

        for endpoint in ("max_cooperative", "max_competitive"):
            for role in ("baseline", "adversary"):
                sub = game_sub[
                    game_sub["endpoint"].eq(endpoint) & game_sub["role"].eq(role)
                ].sort_values("adv_elo")
                if sub.empty:
                    continue

                y = smooth_series(sub["signed_relative_gap_pct"], symmetric=symmetric_smoothing)
                plotted_values.extend(y.dropna().tolist())
                color = ENDPOINT_COLORS[endpoint]
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
                        "plot": OUT_PNG.name,
                        "role": role,
                        "game_id": game_id,
                        "endpoint": endpoint,
                        "competition_value": float(sub["competition_value"].iloc[0]),
                        "n_points": int(len(sub)),
                        "ewm_alpha": EWM_ALPHA,
                        "dropped": f"game2 max_competitive {DROP_GAME2_MODEL}",
                    }
                )

        ax.axhline(0, color="#6b7280", linewidth=1.1, alpha=0.82)
        if show_baseline_elo:
            ax.axvline(
                base_elo,
                color="#64748b",
                linestyle=(0, (4, 4)),
                linewidth=1.15,
                alpha=0.34,
                zorder=0,
            )
            ax.text(
                base_elo + 4,
                0.965,
                f"baseline Elo {base_elo:.0f}",
                transform=ax.get_xaxis_transform(),
                ha="left",
                va="top",
                fontsize=9.4,
                color="#64748b",
                alpha=0.78,
                rotation=90,
            )
        ax.grid(alpha=0.23, linewidth=0.8)
        ax.set_title(GAME_LABELS[game_id], fontsize=22, pad=11)
        ax.set_xlabel("Adversary Elo", fontsize=15, labelpad=8)
        ax.set_ylabel("Signed relative fair-share gap (%)", fontsize=15, labelpad=10)
        ax.tick_params(axis="both", labelsize=12)
        ax.set_xlim(1088, 1515)
        ax.set_ylim(*padded_ylim(pd.Series(plotted_values)))
        ax.text(
            0.06,
            0.045,
            f"{'Symmetric EWM' if symmetric_smoothing else 'EWM'} alpha={EWM_ALPHA:.2f}",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=10.5,
            color="#475569",
        )

    fig.suptitle(
        "Baseline and adversary signed relative fair-share gap: endpoint "
        f"{'symmetric EWM' if symmetric_smoothing else 'EWM'} trends",
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
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(cell_rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    means = load_model_means()
    cells = plot(means, OUT_PNG)
    plot(means, OUT_BASELINE_ELO_PNG, show_baseline_elo=True)
    plot(
        means,
        OUT_SYMMETRIC_BASELINE_ELO_PNG,
        show_baseline_elo=True,
        symmetric_smoothing=True,
    )
    means.to_csv(OUT_CSV, index=False)
    cells.to_csv(OUT_CELLS, index=False)
    print(f"Wrote {OUT_PNG}")
    print(f"Wrote {OUT_BASELINE_ELO_PNG}")
    print(f"Wrote {OUT_SYMMETRIC_BASELINE_ELO_PNG}")
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_CELLS}")


if __name__ == "__main__":
    main()
