#!/usr/bin/env python3
"""N>2 fairness residual curves against the relevant strength axis.

Games 1 and 2 use Nash bargaining residuals. Game 3 uses Lindahl-style
residuals, because the public-good question is cost-share fairness.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Callable

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from strong_models_experiment.analysis.active_model_roster import elo_for_model  # noqa: E402


AGENT_METRICS = PROJECT_ROOT / "analysis/nash_lindahl_fairness_20260505/agent_metrics.csv"
OUT_DIR = PROJECT_ROOT / "overleaf/neurips/graphics/n_gt_2_report"

N_ORDER = [4, 6, 8, 10]
GAME_ORDER = ["game1", "game2", "game3"]
GAME_TITLES = {"game1": "Game 1", "game2": "Game 2", "game3": "Game 3"}
COMP_ORDER = ["low", "medium", "high"]
COMP_TITLES = {
    "low": "Low competition",
    "medium": "Medium competition",
    "high": "High competition",
}

SERIES_ORDER = [
    "heterogeneous_agent",
    "homogeneous_adversary_adversary",
    "homogeneous_adversary_baseline",
    "homogeneous_control",
]
SERIES_LABELS = {
    "heterogeneous_agent": "Heterogeneous agents",
    "homogeneous_adversary_adversary": "Hom. adv: inserted model",
    "homogeneous_adversary_baseline": "Hom. adv: GPT-5-nano agents",
    "homogeneous_control": "Hom. control: GPT-5-nano",
}
SERIES_STYLES = {
    "heterogeneous_agent": {
        "color": "#2E7D32",
        "marker": "o",
        "linestyle": "-",
        "mfc": "#2E7D32",
    },
    "homogeneous_adversary_adversary": {
        "color": "#D95F02",
        "marker": "s",
        "linestyle": "-",
        "mfc": "#D95F02",
    },
    "homogeneous_adversary_baseline": {
        "color": "#D95F02",
        "marker": "s",
        "linestyle": "--",
        "mfc": "white",
    },
    "homogeneous_control": {
        "color": "#4E79A7",
        "marker": "D",
        "linestyle": "None",
        "mfc": "#4E79A7",
    },
}
PANEL_BOX_ASPECT = 1.12


def sem(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 2:
        return 0.0
    return float(clean.std(ddof=1) / math.sqrt(len(clean)))


def short_float(value: float) -> str:
    if math.isclose(value, round(value), abs_tol=1e-9):
        return str(int(round(value)))
    return f"{value:.3f}".rstrip("0").rstrip(".")


def safe_elo(model: str) -> float:
    try:
        return float(elo_for_model(str(model)))
    except Exception:
        return math.nan


def add_competition_bands(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["competition_band"] = ""
    for game, sub in df.groupby("game_id"):
        values = sorted(pd.to_numeric(sub["competition_value"], errors="coerce").dropna().unique())
        if not values:
            continue
        if len(values) == 1:
            mapping = {values[0]: "medium"}
        else:
            mapping: dict[float, str] = {}
            for idx, value in enumerate(values):
                position = idx / (len(values) - 1)
                if position < 1 / 3:
                    band = "low"
                elif position > 2 / 3:
                    band = "high"
                else:
                    band = "medium"
                mapping[float(value)] = band
        df.loc[df["game_id"].eq(game), "competition_band"] = (
            pd.to_numeric(df.loc[df["game_id"].eq(game), "competition_value"], errors="coerce")
            .map(mapping)
            .fillna("medium")
        )
    return df


def load_series_rows() -> pd.DataFrame:
    agents = pd.read_csv(AGENT_METRICS)
    agents = agents[
        agents["n_agents"].isin(N_ORDER)
        & agents["experiment_family"].isin(
            ["heterogeneous_random", "homogeneous_adversary", "homogeneous_control"]
        )
    ].copy()
    for col in ["n_agents", "competition_value", "elo", "nbs_residual", "lindahl_residual"]:
        agents[col] = pd.to_numeric(agents[col], errors="coerce")

    agents["fairness_benchmark"] = np.where(agents["game_id"].eq("game3"), "Lindahl", "NBS")
    agents["fairness_residual"] = np.where(
        agents["game_id"].eq("game3"),
        agents["lindahl_residual"],
        agents["nbs_residual"],
    )
    agents = agents.dropna(subset=["fairness_residual"]).copy()
    agents = add_competition_bands(agents)

    filled_elo = agents["elo"].copy()
    missing = ~np.isfinite(pd.to_numeric(filled_elo, errors="coerce"))
    if missing.any():
        filled_elo.loc[missing] = agents.loc[missing, "model"].map(safe_elo)
    agents["own_elo_filled"] = pd.to_numeric(filled_elo, errors="coerce")

    rows: list[pd.DataFrame] = []

    hetero = agents[agents["experiment_family"].eq("heterogeneous_random")].copy()
    hetero = hetero.dropna(subset=["own_elo_filled"])
    hetero["series_key"] = "heterogeneous_agent"
    hetero["reference_elo"] = hetero["own_elo_filled"]
    hetero["unit_kind"] = "agent"
    rows.append(hetero)

    hom_adv = agents[agents["experiment_family"].eq("homogeneous_adversary")].copy()
    adv_elo = (
        hom_adv[hom_adv["role"].eq("adversary")]
        .dropna(subset=["own_elo_filled"])
        .groupby("result_path")["own_elo_filled"]
        .first()
        .rename("adversary_elo")
    )
    hom_adv = hom_adv.merge(adv_elo, on="result_path", how="left")
    hom_adv = hom_adv.dropna(subset=["adversary_elo"])

    for (result_path, role_group), group in hom_adv.groupby(
        ["result_path", hom_adv["role"].eq("adversary")], sort=False
    ):
        series_key = (
            "homogeneous_adversary_adversary"
            if bool(role_group)
            else "homogeneous_adversary_baseline"
        )
        row = group.iloc[0].copy()
        row["series_key"] = series_key
        row["reference_elo"] = float(group["adversary_elo"].iloc[0])
        row["fairness_residual"] = float(group["fairness_residual"].mean())
        row["unit_kind"] = "run_role_mean"
        rows.append(pd.DataFrame([row]))

    control = agents[agents["experiment_family"].eq("homogeneous_control")].copy()
    gpt5_nano_elo = safe_elo("gpt-5-nano")
    for _, group in control.groupby("result_path", sort=False):
        row = group.iloc[0].copy()
        row["series_key"] = "homogeneous_control"
        row["reference_elo"] = gpt5_nano_elo
        row["fairness_residual"] = float(group["fairness_residual"].mean())
        row["unit_kind"] = "run_mean"
        rows.append(pd.DataFrame([row]))

    series = pd.concat(rows, ignore_index=True)
    series["series_label"] = series["series_key"].map(SERIES_LABELS)
    series["n_agents"] = series["n_agents"].astype(int)
    return series[
        [
            "result_path",
            "game_id",
            "game_label",
            "n_agents",
            "competition_value",
            "competition_label",
            "competition_band",
            "experiment_family",
            "series_key",
            "series_label",
            "unit_kind",
            "model",
            "role",
            "reference_elo",
            "fairness_benchmark",
            "fairness_residual",
        ]
    ].copy()


def aggregate_for_panel(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    return (
        frame.groupby(["series_key", "series_label", "reference_elo"], dropna=False)
        .agg(
            residual_mean=("fairness_residual", "mean"),
            residual_sem=("fairness_residual", sem),
            obs_count=("fairness_residual", "size"),
            run_count=("result_path", "nunique"),
        )
        .reset_index()
        .sort_values(["series_key", "reference_elo"])
    )


def set_panel_ylim(ax: plt.Axes, summary: pd.DataFrame) -> None:
    if summary.empty:
        return
    values = pd.concat(
        [
            summary["residual_mean"] - summary["residual_sem"].fillna(0.0),
            summary["residual_mean"] + summary["residual_sem"].fillna(0.0),
            pd.Series([0.0]),
        ],
        ignore_index=True,
    ).replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return
    lo = float(values.min())
    hi = float(values.max())
    if math.isclose(lo, hi):
        lo -= 1.0
        hi += 1.0
    pad = max((hi - lo) * 0.12, 2.0)
    ax.set_ylim(lo - pad, hi + pad)


def plot_panel(ax: plt.Axes, frame: pd.DataFrame, title: str, show_ylabel: bool) -> None:
    summary = aggregate_for_panel(frame)
    if not summary.empty:
        for series_key in SERIES_ORDER:
            sub = summary[summary["series_key"].eq(series_key)].sort_values("reference_elo")
            if sub.empty:
                continue
            style = SERIES_STYLES[series_key]
            linestyle = style["linestyle"]
            if len(sub) < 2:
                linestyle = "None"
            ax.errorbar(
                sub["reference_elo"],
                sub["residual_mean"],
                yerr=sub["residual_sem"],
                color=style["color"],
                marker=style["marker"],
                linestyle=linestyle,
                lw=1.65,
                ms=4.6,
                capsize=2.3,
                alpha=0.92,
                markerfacecolor=style["mfc"],
                markeredgecolor=style["color"],
                markeredgewidth=1.1,
                label=SERIES_LABELS[series_key],
            )
    ax.axhline(0, color="#555555", lw=0.8, alpha=0.75)
    ax.grid(True, alpha=0.22, linewidth=0.6)
    ax.set_title(title, fontsize=11.5, pad=7)
    ax.set_xlabel("Reference Elo", fontsize=9.5)
    if show_ylabel:
        ax.set_ylabel("Actual minus fair utility", fontsize=9.5)
    ax.tick_params(axis="both", labelsize=8.3)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.set_box_aspect(PANEL_BOX_ASPECT)
    set_panel_ylim(ax, summary)


Facet = tuple[str, Callable[[pd.DataFrame], pd.Series]]


def make_grid_plot(
    series: pd.DataFrame,
    row_facets: list[Facet],
    col_facets: list[Facet],
    title: str,
    out_path: Path,
    figsize: tuple[float, float],
) -> Path:
    fig, axes = plt.subplots(
        len(row_facets),
        len(col_facets),
        figsize=figsize,
        squeeze=False,
        sharex=True,
        sharey=False,
    )
    for r, (row_label, row_filter) in enumerate(row_facets):
        for c, (col_label, col_filter) in enumerate(col_facets):
            ax = axes[r][c]
            panel = series[row_filter(series) & col_filter(series)].copy()
            if len(row_facets) == 1:
                panel_title = col_label
            elif len(col_facets) == 1:
                panel_title = row_label
            else:
                panel_title = f"{row_label}\n{col_label}"
            plot_panel(ax, panel, panel_title, show_ylabel=(c == 0))
            if panel.empty:
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="#666666",
                )
    handles = [
        Line2D(
            [0],
            [0],
            color=SERIES_STYLES[key]["color"],
            marker=SERIES_STYLES[key]["marker"],
            linestyle=SERIES_STYLES[key]["linestyle"],
            markerfacecolor=SERIES_STYLES[key]["mfc"],
            markeredgecolor=SERIES_STYLES[key]["color"],
            lw=1.7,
            ms=5,
            label=SERIES_LABELS[key],
        )
        for key in SERIES_ORDER
    ]
    fig.suptitle(title, fontsize=14.5, y=0.995)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.957), ncol=2, frameon=False, fontsize=9.2)
    fig.text(
        0.5,
        0.012,
        "Fair benchmark: NBS for Games 1-2; Lindahl-style benchmark for Game 3. Points are averaged residuals at each reference Elo; bars show SEM.",
        ha="center",
        va="bottom",
        fontsize=8.7,
        color="#444444",
    )
    fig.tight_layout(rect=[0.02, 0.04, 0.995, 0.905])
    fig.savefig(out_path, dpi=230, bbox_inches="tight")
    plt.close(fig)
    return out_path


def true_filter(_: pd.DataFrame) -> pd.Series:
    return pd.Series(True, index=_.index)


def save_summary_tables(series: pd.DataFrame) -> tuple[Path, Path]:
    run_series_path = OUT_DIR / "multiagent_fairness_residual_vs_reference_elo_series_rows.csv"
    series.to_csv(run_series_path, index=False)

    summary = (
        series.groupby(
            [
                "game_id",
                "n_agents",
                "competition_band",
                "experiment_family",
                "series_key",
                "series_label",
                "reference_elo",
            ],
            dropna=False,
        )
        .agg(
            residual_mean=("fairness_residual", "mean"),
            residual_sem=("fairness_residual", sem),
            obs_count=("fairness_residual", "size"),
            run_count=("result_path", "nunique"),
            competition_value_min=("competition_value", "min"),
            competition_value_max=("competition_value", "max"),
        )
        .reset_index()
    )
    summary_path = OUT_DIR / "multiagent_fairness_residual_vs_reference_elo_summary.csv"
    summary.to_csv(summary_path, index=False)
    return run_series_path, summary_path


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    series = load_series_rows()
    run_series_path, summary_path = save_summary_tables(series)

    outputs: list[Path] = []
    game_facets: list[Facet] = [
        (GAME_TITLES[game], lambda df, game=game: df["game_id"].eq(game)) for game in GAME_ORDER
    ]
    n_facets: list[Facet] = [
        (f"N={n}", lambda df, n=n: df["n_agents"].eq(n)) for n in N_ORDER
    ]
    comp_facets: list[Facet] = [
        (COMP_TITLES[band], lambda df, band=band: df["competition_band"].eq(band))
        for band in COMP_ORDER
    ]

    outputs.append(
        make_grid_plot(
            series,
            [("All games, N, and competition", true_filter)],
            [("Overall", true_filter)],
            "N>2 fair-benchmark residual vs reference Elo: overall",
            OUT_DIR / "multiagent_fairness_residual_vs_reference_elo_allseries_overall.png",
            (5.8, 6.4),
        )
    )
    outputs.append(
        make_grid_plot(
            series,
            [("All N and competition", true_filter)],
            game_facets,
            "N>2 fair-benchmark residual vs reference Elo by game",
            OUT_DIR / "multiagent_fairness_residual_vs_reference_elo_allseries_by_game.png",
            (13.6, 6.3),
        )
    )
    outputs.append(
        make_grid_plot(
            series,
            [("All games and N", true_filter)],
            comp_facets,
            "N>2 fair-benchmark residual vs reference Elo by competition band",
            OUT_DIR / "multiagent_fairness_residual_vs_reference_elo_allseries_by_competition.png",
            (13.6, 6.3),
        )
    )
    outputs.append(
        make_grid_plot(
            series,
            [("All games and competition", true_filter)],
            n_facets,
            "N>2 fair-benchmark residual vs reference Elo by N",
            OUT_DIR / "multiagent_fairness_residual_vs_reference_elo_allseries_by_n.png",
            (16.8, 6.3),
        )
    )
    outputs.append(
        make_grid_plot(
            series,
            n_facets,
            game_facets,
            "N>2 fair-benchmark residual vs reference Elo by N and game",
            OUT_DIR / "multiagent_fairness_residual_vs_reference_elo_allseries_by_n_game.png",
            (13.8, 20.6),
        )
    )
    outputs.append(
        make_grid_plot(
            series,
            n_facets,
            comp_facets,
            "N>2 fair-benchmark residual vs reference Elo by N and competition band",
            OUT_DIR / "multiagent_fairness_residual_vs_reference_elo_allseries_by_n_competition.png",
            (13.8, 20.6),
        )
    )
    outputs.append(
        make_grid_plot(
            series,
            comp_facets,
            game_facets,
            "N>2 fair-benchmark residual vs reference Elo by competition band and game",
            OUT_DIR / "multiagent_fairness_residual_vs_reference_elo_allseries_by_competition_game.png",
            (13.8, 15.7),
        )
    )

    file_list_path = OUT_DIR / "multiagent_fairness_residual_vs_reference_elo_allseries_files.txt"
    with file_list_path.open("w", encoding="utf-8") as handle:
        handle.write("CSV tables\n")
        handle.write(f"{run_series_path}\n")
        handle.write(f"{summary_path}\n\n")
        handle.write("Plots\n")
        for path in outputs:
            handle.write(f"{path}\n")

    print("Wrote:")
    print(run_series_path)
    print(summary_path)
    for path in outputs:
        print(path)
    print(file_list_path)


if __name__ == "__main__":
    main()
