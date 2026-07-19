#!/usr/bin/env python3
"""Create the compact main-text N=2 fairness-distance figure.

Reads model-level averages from the N=2 baseline comparison analysis and
plots within-game normalized fairness distance against adversary Elo, with the
three game families overlaid in the same style as the main adversary-payoff
figure.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
IN_CSV = PROJECT_ROOT / "experiments" / "results" / "n2_baseline_comparison_analysis_20260505" / "overall_by_model_game.csv"
OUT_DIR = PROJECT_ROOT / "overleaf" / "neurips" / "graphics" / "n2_gpt5_nano"
OUT_PNG = OUT_DIR / "11_fairness_distance_three_game_curves.png"
OUT_SLOPES = OUT_DIR / "11_fairness_distance_three_game_curves_slopes.csv"

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


def fit_stats(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    x_clean = x[mask]
    y_clean = y[mask]
    if len(x_clean) < 3:
        return {
            "slope_per_100": np.nan,
            "intercept": np.nan,
            "r": np.nan,
            "p": np.nan,
        }
    slope, intercept = np.polyfit(x_clean, y_clean, 1)
    r_val = float(np.corrcoef(x_clean, y_clean)[0, 1])
    return {
        "slope_per_100": float(slope * 100.0),
        "intercept": float(intercept),
        "r": r_val,
        "p": np.nan,
    }


def main() -> None:
    df = pd.read_csv(IN_CSV)
    df = df[df["baseline_key"].eq("gpt5_nano")].copy()

    plot_frames: list[pd.DataFrame] = []
    slope_rows: list[dict[str, float | str | int]] = []
    for game_id in GAME_ORDER:
        sub = df[df["game_id"].eq(game_id)].copy()
        sub["fairness_distance"] = pd.to_numeric(sub["fairness_distance"], errors="coerce")
        sub["fairness_distance_sem"] = pd.to_numeric(sub["fairness_distance_sem"], errors="coerce")
        sub["adversary_elo"] = pd.to_numeric(sub["adversary_elo"], errors="coerce")
        sub = sub.replace([np.inf, -np.inf], np.nan).dropna(subset=["fairness_distance", "adversary_elo"])

        raw_min = float(sub["fairness_distance"].min())
        raw_max = float(sub["fairness_distance"].max())
        raw_range = raw_max - raw_min
        if raw_range <= 0:
            sub["fairness_distance_norm"] = 0.0
            sub["fairness_distance_sem_norm"] = 0.0
        else:
            sub["fairness_distance_norm"] = (sub["fairness_distance"] - raw_min) / raw_range
            sub["fairness_distance_sem_norm"] = sub["fairness_distance_sem"].fillna(0.0) / raw_range

        raw_stats = fit_stats(sub["adversary_elo"].to_numpy(float), sub["fairness_distance"].to_numpy(float))
        norm_stats = fit_stats(sub["adversary_elo"].to_numpy(float), sub["fairness_distance_norm"].to_numpy(float))
        slope_rows.append(
            {
                "game_id": game_id,
                "game_label": GAME_LABELS[game_id],
                "n_models": int(len(sub)),
                "raw_min": raw_min,
                "raw_max": raw_max,
                "raw_slope_per_100_elo": raw_stats["slope_per_100"],
                "raw_r": raw_stats["r"],
                "raw_p": raw_stats["p"],
                "normalized_slope_per_100_elo": norm_stats["slope_per_100"],
                "normalized_r": norm_stats["r"],
                "normalized_p": norm_stats["p"],
            }
        )
        plot_frames.append(sub)

    plot_df = pd.concat(plot_frames, ignore_index=True)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": True,
            "axes.spines.right": True,
        }
    )
    fig, ax = plt.subplots(figsize=(8.25, 5.15))

    for game_id in GAME_ORDER:
        sub = plot_df[plot_df["game_id"].eq(game_id)].sort_values("adversary_elo")
        color = GAME_COLORS[game_id]
        x = sub["adversary_elo"].to_numpy(float)
        y = sub["fairness_distance_norm"].to_numpy(float)
        yerr = sub["fairness_distance_sem_norm"].fillna(0.0).clip(lower=0.0).to_numpy(float)

        ax.errorbar(
            x,
            y,
            yerr=yerr,
            fmt="o",
            markersize=4.2,
            linewidth=0,
            elinewidth=0.9,
            capsize=2.0,
            markeredgewidth=1.0,
            color=color,
            ecolor=color,
            alpha=0.33,
        )

        stats_row = fit_stats(x, y)
        if np.isfinite(stats_row["slope_per_100"]):
            xs = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), 200)
            ys = (stats_row["slope_per_100"] / 100.0) * xs + stats_row["intercept"]
            ax.plot(
                xs,
                ys,
                color=color,
                linestyle="--",
                linewidth=3.0,
                alpha=0.95,
                label=GAME_LABELS[game_id],
            )

    ax.set_xlabel("Adversary Chatbot Arena Elo", fontsize=17, labelpad=10)
    ax.set_ylabel("Normalized benchmark distance", fontsize=17, labelpad=10)
    ax.set_xlim(1090, 1515)
    ax.set_ylim(-0.08, 1.15)
    ax.grid(alpha=0.24, linewidth=0.8)
    ax.tick_params(axis="both", labelsize=12)
    ax.text(
        0.02,
        0.04,
        "Unsigned distance: lower is closer to the benchmark point",
        transform=ax.transAxes,
        fontsize=11,
        color="#374151",
        ha="left",
        va="bottom",
    )
    legend = ax.legend(loc="upper right", frameon=True, fontsize=12, handlelength=3.0)
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
