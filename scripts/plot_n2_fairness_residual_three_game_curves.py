#!/usr/bin/env python3
"""Create compact N=2 fair-share residual figure.

This is a main-text replacement for the full NBS decomposition panel. It keeps
only the quantity needed for the fairness-extraction claim:

    residual = adversary utility - benchmark fair-share utility

Positive values mean the adversary is above its benchmark share. Negative values
mean it is below that benchmark.
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
OUT_DIR = PROJECT_ROOT / "overleaf" / "neurips" / "graphics" / "n2_gpt5_nano"
OUT_PNG = OUT_DIR / "14_fair_share_residual_three_game_curves.png"
OUT_SLOPES = OUT_DIR / "14_fair_share_residual_three_game_curves_slopes.csv"

GAME_ORDER = ("game1", "game2", "game3")
GAME_LABELS = {
    "game1": "Game 1: Item allocation",
    "game2": "Game 2: Diplomacy",
    "game3": "Game 3: Co-funding",
}
GAME_COLORS = {
    "game1": "#7c3aed",
    "game2": "#0f766e",
    "game3": "#f97316",
}


def sem(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) <= 1:
        return 0.0
    return float(clean.std(ddof=1) / math.sqrt(len(clean)))


def finite_yerr(values: pd.Series) -> np.ndarray:
    arr = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(float)
    return np.maximum(arr, 0.0)


def fit_stats(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    x_clean = x[mask]
    y_clean = y[mask]
    if len(x_clean) < 3:
        return {"slope_per_100": np.nan, "slope_raw": np.nan, "intercept": np.nan, "r": np.nan}
    slope, intercept = np.polyfit(x_clean, y_clean, 1)
    r_val = float(np.corrcoef(x_clean, y_clean)[0, 1])
    return {
        "slope_per_100": float(slope * 100.0),
        "slope_raw": float(slope),
        "intercept": float(intercept),
        "r": r_val,
    }


def zero_crossing(stats: dict[str, float]) -> float:
    slope = stats["slope_raw"]
    if not np.isfinite(slope) or abs(slope) < 1e-12:
        return np.nan
    return float(-stats["intercept"] / slope)


def load_model_means() -> pd.DataFrame:
    metrics = pd.read_csv(AGENT_METRICS_CSV)
    elo_df = pd.read_csv(ELO_CSV)[["model", "elo"]].rename(columns={"elo": "arena_elo"})

    adv = metrics[
        metrics["source_group"].eq("n2_main_gpt5_baseline")
        & metrics["role"].eq("adversary")
    ].copy()
    adv["canonical"] = adv["model"].apply(canonical_model_name)
    elo_df["canonical"] = elo_df["model"].apply(canonical_model_name)
    elo_lookup = elo_df.drop_duplicates("canonical").set_index("canonical")["arena_elo"]
    adv["elo"] = adv["canonical"].map(elo_lookup)
    adv = adv.dropna(subset=["elo"])

    return (
        adv.groupby(["game_id", "model", "canonical", "elo"], as_index=False)
        .agg(
            residual_mean=("nbs_residual", "mean"),
            residual_sem=("nbs_residual", sem),
            n_runs=("nbs_residual", "size"),
        )
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["residual_mean", "elo"])
    )


def main() -> None:
    agg = load_model_means()
    slope_rows: list[dict[str, float | str | int]] = []

    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(figsize=(8.25, 5.05))

    all_x = []
    for game_id in GAME_ORDER:
        sub = agg[agg["game_id"].eq(game_id)].sort_values("elo")
        color = GAME_COLORS[game_id]
        x = sub["elo"].to_numpy(float)
        y = sub["residual_mean"].to_numpy(float)
        all_x.extend(x.tolist())
        stats = fit_stats(x, y)
        zc = zero_crossing(stats)
        slope_rows.append(
            {
                "game_id": game_id,
                "game_label": GAME_LABELS[game_id],
                "n_models": int(len(sub)),
                "slope_per_100_elo": stats["slope_per_100"],
                "r": stats["r"],
                "zero_crossing_elo": zc,
            }
        )

        ax.errorbar(
            x,
            y,
            yerr=finite_yerr(sub["residual_sem"]),
            fmt="o",
            markersize=4.4,
            color=color,
            ecolor=color,
            alpha=0.35,
            elinewidth=0.9,
            capsize=2.0,
            linewidth=0,
        )

        xs = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), 200)
        ys = stats["slope_raw"] * xs + stats["intercept"]
        ax.plot(
            xs,
            ys,
            linestyle="--",
            linewidth=3.2,
            color=color,
            alpha=0.95,
            label=GAME_LABELS[game_id],
        )

        if np.isfinite(zc):
            ax.plot([zc], [0], marker="|", color=color, markersize=11, markeredgewidth=2.0, alpha=0.9)

    ax.axhline(0, color="#111827", linewidth=1.1, alpha=0.75)
    ax.text(
        min(all_x) - 6,
        0.8,
        "fair-share benchmark",
        ha="left",
        va="bottom",
        fontsize=14.0,
        color="#374151",
    )
    ax.text(
        0.02,
        0.92,
        "above fair share",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=14.0,
        color="#374151",
    )
    ax.text(
        0.02,
        0.07,
        "below fair share",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=14.0,
        color="#374151",
    )

    ax.set_xlabel("Adversary Chatbot Arena Elo", fontsize=17, labelpad=10)
    ax.set_ylabel("Adversary utility above fair share", fontsize=17, labelpad=10)
    ax.set_xlim(min(all_x) - 15, max(all_x) + 15)
    ax.set_ylim(-50, 14)
    ax.grid(alpha=0.24, linewidth=0.8)
    ax.tick_params(axis="both", labelsize=12)
    legend = ax.legend(loc="lower right", frameon=True, fontsize=12, handlelength=3.0)
    legend.get_frame().set_alpha(0.92)
    legend.get_frame().set_edgecolor("#d1d5db")

    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame(slope_rows).to_csv(OUT_SLOPES, index=False)
    print(f"Wrote {OUT_PNG}")
    print(f"Wrote {OUT_SLOPES}")


if __name__ == "__main__":
    main()
