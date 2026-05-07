#!/usr/bin/env python3
"""Build the expanded N=2 + N>2 bargaining analysis report.

The script intentionally leaves the original N=2 report untouched. It copies the
existing report content and image assets into a new output directory, parses the
multi-agent raw results fresh, recomputes the competition index used in the
paper, generates the N>2 plots/tables, and appends a second-half report section.
"""

from __future__ import annotations

import csv
import math
import re
import shutil
import sys
import textwrap
import warnings
from itertools import combinations
from pathlib import Path
from typing import Iterable, Optional, Sequence

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from scipy.special import expit


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.plot_full_games123_clean_subset import build_tables, short_model_name
from scripts.analyze_nash_lindahl_fairness import (
    AnalysisRow as FairAnalysisRow,
    analyze_row as analyze_fairness_row,
    load_json as load_fairness_json,
    normalize_game_id as normalize_fairness_game_id,
)


N2_REPORT = (
    PROJECT_ROOT
    / "experiments/results/n2_baseline_comparison_analysis_20260505/"
    / "n2_baseline_comparison_report.md"
)
N2_SUMMARY = (
    PROJECT_ROOT
    / "experiments/results/n2_baseline_comparison_analysis_20260505/"
    / "overall_by_model_game.csv"
)
N2_RUN_METRICS = (
    PROJECT_ROOT
    / "experiments/results/n2_baseline_comparison_analysis_20260505/"
    / "all_runs_with_metrics.csv"
)
HOM_ROOT = (
    PROJECT_ROOT
    / "experiments/results/full_games123_multiagent_production_20260428_085255"
)
HET_ROOT = (
    PROJECT_ROOT
    / "experiments/results/"
    / "full_games123_multiagent_heterogeneous_equal_width_openrouter_repair_20260429_113848"
)
FAIR_DIR = PROJECT_ROOT / "analysis/nash_lindahl_fairness_20260505"
FAIR_RUN_METRICS = FAIR_DIR / "run_metrics.csv"
FAIR_AGENT_METRICS = FAIR_DIR / "agent_metrics.csv"

OUT_DIR = (
    PROJECT_ROOT
    / "experiments/results/n2_plus_multiagent_comparison_analysis_20260505"
)
PLOTS_DIR = OUT_DIR / "plots_multiagent"
TABLES_DIR = OUT_DIR / "tables_multiagent"
REPORT_PATH = OUT_DIR / "n2_plus_multiagent_comparison_report.md"

GAME_ORDER = ["game1", "game2", "game3"]
GAME_TITLES = {
    "game1": "Game 1: Item allocation",
    "game2": "Game 2: Diplomacy",
    "game3": "Game 3: Cofunding",
}
N_ORDER = [2, 4, 6, 8, 10]
N_COLORS = {
    2: "#1f77b4",
    4: "#d62728",
    6: "#2ca02c",
    8: "#9467bd",
    10: "#ff7f0e",
}
BAND_COLORS = {
    "cooperative": "#1b9e77",
    "middle": "#7570b3",
    "competitive": "#d95f02",
    "all": "#333333",
}
FAMILY_LABELS = {
    "homogeneous_control": "Homogeneous control",
    "homogeneous_adversary": "Homogeneous adversary",
    "heterogeneous_random": "Heterogeneous",
}
FAMILY_COLORS = {
    "homogeneous_control": "#1f77b4",
    "homogeneous_adversary": "#d62728",
    "heterogeneous_random": "#2ca02c",
}
MULTIAGENT_FAIR_SOURCE_GROUPS = [
    "n_eq_2_homogeneous",
    "n_eq_2_heterogeneous",
    "n_gt_2_homogeneous",
    "n_gt_2_heterogeneous",
]
EPS = 1e-9


def ensure_output_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)


def save_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def clean_float(value: object) -> float:
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return math.nan
    return value_f if math.isfinite(value_f) else math.nan


def shifted_gini(values: Sequence[float]) -> float:
    arr = np.asarray([clean_float(value) for value in values], dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0 or np.allclose(arr, arr[0]):
        return 0.0
    min_value = float(np.min(arr))
    if min_value < 0:
        arr = arr - min_value
    total = float(np.sum(arr))
    if math.isclose(total, 0.0):
        return 0.0
    diffs = np.abs(arr[:, None] - arr[None, :])
    return float(np.mean(diffs) / (2.0 * float(np.mean(arr))))


def sem_series(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) <= 1:
        return 0.0
    return float(clean.std(ddof=1) / math.sqrt(len(clean)))


def finite_yerr(values: object) -> np.ndarray | float:
    if isinstance(values, pd.Series):
        arr = pd.to_numeric(values, errors="coerce").fillna(0.0).to_numpy(dtype=float)
        return np.maximum(arr, 0.0)
    value = clean_float(values)
    return max(value, 0.0) if math.isfinite(value) else 0.0


def plot_errorbar_series(
    ax: plt.Axes,
    x: Sequence[float],
    y: Sequence[float],
    yerr: Sequence[float] | pd.Series | float,
    color: str,
    label: str,
    marker: str = "o",
    linestyle: str = "-",
    linewidth: float = 1.1,
    markersize: float = 3.8,
    alpha: float = 0.9,
) -> object:
    return ax.errorbar(
        x,
        y,
        yerr=finite_yerr(yerr),
        fmt=marker,
        linestyle=linestyle,
        lw=linewidth,
        ms=markersize,
        capsize=2.0,
        capthick=0.7,
        elinewidth=0.7,
        color=color,
        ecolor=color,
        alpha=alpha,
        label=label,
    )


def add_binned_sem_errorbars(
    ax: plt.Axes,
    frame: pd.DataFrame,
    x_col: str,
    y_col: str,
    color: str,
    label: str,
    bins: int = 4,
) -> pd.DataFrame:
    data = frame[[x_col, y_col]].copy()
    data[x_col] = pd.to_numeric(data[x_col], errors="coerce")
    data[y_col] = pd.to_numeric(data[y_col], errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < 3:
        if len(data):
            ax.errorbar(
                data[x_col],
                data[y_col],
                yerr=np.zeros(len(data)),
                fmt="D",
                ms=4.0,
                capsize=2.0,
                color=color,
                ecolor=color,
                alpha=0.95,
                label=f"{label} mean +/- SEM",
            )
        return data
    q = min(bins, max(1, data[x_col].nunique()), len(data))
    try:
        data["_bin"] = pd.qcut(data[x_col], q=q, duplicates="drop")
    except ValueError:
        data["_bin"] = pd.cut(data[x_col], bins=q, duplicates="drop")
    summary = (
        data.groupby("_bin", observed=True)
        .agg(
            x_mean=(x_col, "mean"),
            y_mean=(y_col, "mean"),
            y_sem=(y_col, sem_series),
            count=(y_col, "count"),
        )
        .reset_index(drop=True)
    )
    if summary.empty:
        return summary
    ax.errorbar(
        summary["x_mean"],
        summary["y_mean"],
        yerr=finite_yerr(summary["y_sem"]),
        fmt="D",
        ms=4.4,
        capsize=2.4,
        capthick=0.8,
        elinewidth=0.85,
        color=color,
        ecolor=color,
        alpha=0.98,
        label=f"{label} mean +/- SEM",
    )
    return summary


def linear_fit(xs: Sequence[float], ys: Sequence[float]) -> tuple[float, float, float]:
    pairs = [
        (clean_float(x), clean_float(y))
        for x, y in zip(xs, ys)
        if math.isfinite(clean_float(x)) and math.isfinite(clean_float(y))
    ]
    if len(pairs) < 2:
        return math.nan, math.nan, math.nan
    x = np.asarray([p[0] for p in pairs], dtype=float)
    y = np.asarray([p[1] for p in pairs], dtype=float)
    if np.allclose(x, x[0]):
        return math.nan, math.nan, math.nan
    slope, intercept = np.polyfit(x, y, 1)
    fitted = slope * x + intercept
    ss_res = float(np.sum((y - fitted) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if not math.isclose(ss_tot, 0.0) else 1.0
    return float(slope), float(intercept), float(r2)


def add_fit_line(
    ax: plt.Axes,
    frame: pd.DataFrame,
    x_col: str,
    y_col: str,
    color: str,
    linestyle: str = "--",
    linewidth: float = 1.0,
    alpha: float = 0.75,
) -> tuple[float, float, float]:
    data = frame[[x_col, y_col]].dropna()
    data = data[
        np.isfinite(pd.to_numeric(data[x_col], errors="coerce"))
        & np.isfinite(pd.to_numeric(data[y_col], errors="coerce"))
    ]
    if len(data) < 2 or data[x_col].nunique() < 2:
        return math.nan, math.nan, math.nan
    slope, intercept, r2 = linear_fit(data[x_col], data[y_col])
    if math.isfinite(slope):
        xs = np.linspace(float(data[x_col].min()), float(data[x_col].max()), 100)
        ax.plot(xs, slope * xs + intercept, color=color, linestyle=linestyle, lw=linewidth, alpha=alpha)
    return slope, intercept, r2


def format_slope_per_100(slope: float, unit: str = "100 Elo") -> str:
    if not math.isfinite(clean_float(slope)):
        return "slope=NA"
    return f"slope={slope * 100.0:+.2f}/{unit}"


def annotate_slope_block(
    ax: plt.Axes,
    entries: Sequence[tuple[str, float]],
    loc: str = "upper left",
    fontsize: float = 6.0,
    unit: str = "100 Elo",
) -> None:
    clean_entries = [(label, slope) for label, slope in entries if math.isfinite(clean_float(slope))]
    if not clean_entries:
        return
    text = f"slope/{unit}\n" + "\n".join(f"{label}: {slope * 100.0:+.2f}" for label, slope in clean_entries)
    anchors = {
        "upper left": (0.02, 0.98, "left", "top"),
        "upper right": (0.98, 0.98, "right", "top"),
        "lower left": (0.02, 0.02, "left", "bottom"),
        "lower right": (0.98, 0.02, "right", "bottom"),
    }
    x, y, ha, va = anchors.get(loc, anchors["upper left"])
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=fontsize,
        bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.78, "pad": 2.0},
    )


def exact_setting_label(
    frame: pd.DataFrame,
    game: str,
    n_agents: Optional[int],
    band: str,
    compact: bool = True,
) -> str:
    sub = frame[(frame["game_label"].eq(game)) & (frame["competition_band"].eq(band))]
    if n_agents is not None:
        sub = sub[sub["n_agents"].eq(n_agents)]
    labels = sorted(str(label) for label in sub["competition_label_ci"].dropna().unique())
    if not labels:
        return band
    label_text = ", ".join(labels)
    if compact and len(label_text) > 34:
        label_text = f"{labels[0]}..{labels[-1]}"
    return f"{band} ({label_text})"


def set_n_ticks_with_ci(ax: plt.Axes, frame: pd.DataFrame, game: str, band: str) -> None:
    tick_labels = []
    for n in N_ORDER:
        label = exact_setting_label(frame, game, n, band, compact=True)
        ci = label[label.find("(") + 1 : -1] if "(" in label and label.endswith(")") else ""
        tick_labels.append(f"{n}\n{ci}")
    ax.set_xticks(N_ORDER)
    ax.set_xticklabels(tick_labels, fontsize=7)


def label_points(
    ax: plt.Axes,
    frame: pd.DataFrame,
    x_col: str,
    y_col: str,
    label_col: str,
    fontsize: float = 4.5,
    alpha: float = 0.65,
    max_labels: int = 140,
) -> None:
    if frame.empty:
        return
    rows = frame.dropna(subset=[x_col, y_col, label_col]).copy()
    if len(rows) > max_labels:
        rows = rows.sort_values(y_col).iloc[
            sorted(set(range(0, min(20, len(rows)))) | set(range(max(0, len(rows) - 20), len(rows))))
        ]
    for _, row in rows.iterrows():
        ax.annotate(
            str(row[label_col]),
            (float(row[x_col]), float(row[y_col])),
            textcoords="offset points",
            xytext=(1.5, 1.5),
            fontsize=fontsize,
            alpha=alpha,
        )


def tidy_axes(ax: plt.Axes, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(True, alpha=0.22, linewidth=0.6)
    ax.tick_params(axis="both", labelsize=8)


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def competition_ci(row: pd.Series) -> float:
    game = row.get("game_label")
    if game == "game1":
        return clean_float(row.get("competition_level"))
    if game == "game2":
        rho = clean_float(row.get("rho"))
        theta = clean_float(row.get("theta"))
        if math.isfinite(rho) and math.isfinite(theta):
            return theta * (1.0 - rho) / 2.0
    if game == "game3":
        sigma = clean_float(row.get("sigma"))
        alpha = clean_float(row.get("alpha"))
        if math.isfinite(sigma) and math.isfinite(alpha):
            return (1.0 - alpha) * (1.0 - sigma)
    return clean_float(row.get("competition_value"))


def competition_label(row: pd.Series) -> str:
    ci = clean_float(row.get("competition_ci"))
    if not math.isfinite(ci):
        return "CI=NA"
    game = row.get("game_label")
    if game == "game1":
        return f"c={ci:.2f}"
    if game == "game2":
        rho = clean_float(row.get("rho"))
        theta = clean_float(row.get("theta"))
        return f"CI={ci:.3f}"
    if game == "game3":
        return f"CI={ci:.3f}"
    return f"CI={ci:.3f}"


def prep_runs(runs: pd.DataFrame, source_group: str) -> pd.DataFrame:
    frame = runs.copy()
    frame["source_group"] = source_group
    for col in [
        "config_id",
        "n_agents",
        "adversary_elo",
        "adversary_utility",
        "baseline_mean_utility",
        "mean_utility",
        "sum_utility",
        "utility_std",
        "elo_std",
        "elo_variance",
        "final_round",
    ]:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame["run_key"] = source_group + ":" + frame["config_id"].astype(int).astype(str).str.zfill(4)
    frame["competition_ci"] = frame.apply(competition_ci, axis=1)
    frame["competition_ci_rounded"] = frame["competition_ci"].round(6)
    frame["competition_label_ci"] = frame.apply(competition_label, axis=1)
    frame["family_label"] = frame["experiment_family"].map(FAMILY_LABELS).fillna(frame["experiment_family"])

    frame["adversary_advantage"] = frame["adversary_utility"] - frame["baseline_mean_utility"]
    denom = frame["utility_std"].where(frame["utility_std"].abs() > EPS)
    frame["adversary_z_advantage"] = (frame["adversary_utility"] - frame["mean_utility"]) / denom
    frame.loc[~np.isfinite(frame["adversary_z_advantage"]), "adversary_z_advantage"] = np.nan

    if "consensus_reached" in frame.columns:
        frame["consensus_numeric"] = frame["consensus_reached"].astype(float)
    else:
        frame["consensus_numeric"] = np.nan
    return frame


def add_bands(runs: pd.DataFrame) -> pd.DataFrame:
    frame = runs.copy()
    frame["competition_band"] = "middle"
    keys = ["source_group", "game_label", "n_agents"]
    for _, idx in frame.groupby(keys, dropna=False).groups.items():
        idx_list = list(idx)
        sub = pd.to_numeric(frame.loc[idx_list, "competition_ci"], errors="coerce")
        finite = sub[np.isfinite(sub)]
        if finite.empty:
            continue
        min_ci = float(finite.min())
        max_ci = float(finite.max())
        if math.isclose(min_ci, max_ci):
            frame.loc[idx_list, "competition_band"] = "all"
        else:
            min_idx = sub.index[sub.sub(min_ci).abs() < 1e-7]
            max_idx = sub.index[sub.sub(max_ci).abs() < 1e-7]
            frame.loc[min_idx, "competition_band"] = "cooperative"
            frame.loc[max_idx, "competition_band"] = "competitive"
    return frame


def prep_agents(agents: pd.DataFrame, runs: pd.DataFrame, source_group: str) -> pd.DataFrame:
    frame = agents.copy()
    frame["source_group"] = source_group
    frame["config_id"] = pd.to_numeric(frame["config_id"], errors="coerce").astype(int)
    frame["run_key"] = source_group + ":" + frame["config_id"].astype(str).str.zfill(4)
    for col in ["n_agents", "elo", "final_utility", "agent_index"]:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    merge_cols = [
        "run_key",
        "competition_ci",
        "competition_ci_rounded",
        "competition_label_ci",
        "competition_band",
        "mean_utility",
        "utility_std",
        "utility_gini_shifted",
        "elo_std",
        "elo_variance",
        "family_label",
    ]
    available = [col for col in merge_cols if col in runs.columns]
    frame = frame.merge(runs[available].drop_duplicates("run_key"), on="run_key", how="left")
    frame["model_short"] = frame["model"].map(lambda value: short_model_name(value) if isinstance(value, str) else value)
    return frame


def add_shifted_gini(runs: pd.DataFrame, agents: pd.DataFrame) -> pd.DataFrame:
    frame = runs.copy()
    gini_by_run = agents.groupby("run_key")["final_utility"].apply(lambda values: shifted_gini(values.tolist()))
    frame["utility_gini_shifted"] = frame["run_key"].map(gini_by_run).astype(float)
    return frame


def parse_config_id_from_path(path_value: object) -> Optional[int]:
    match = re.search(r"config_(\d{4})", str(path_value))
    if not match:
        return None
    return int(match.group(1))


def load_and_prepare_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    hom_runs_raw, hom_agents_raw = build_tables(HOM_ROOT)
    het_runs_raw, het_agents_raw = build_tables(HET_ROOT)

    hom_runs = prep_runs(hom_runs_raw, "homogeneous")
    het_runs = prep_runs(het_runs_raw, "heterogeneous")
    runs = pd.concat([hom_runs, het_runs], ignore_index=True)
    runs = add_bands(runs)
    hom_runs = runs[runs["source_group"].eq("homogeneous")].copy()
    het_runs = runs[runs["source_group"].eq("heterogeneous")].copy()

    hom_agents = prep_agents(hom_agents_raw, hom_runs, "homogeneous")
    het_agents = prep_agents(het_agents_raw, het_runs, "heterogeneous")
    hom_runs = add_shifted_gini(hom_runs, hom_agents)
    het_runs = add_shifted_gini(het_runs, het_agents)
    hom_agents = prep_agents(hom_agents_raw, hom_runs, "homogeneous")
    het_agents = prep_agents(het_agents_raw, het_runs, "heterogeneous")

    return hom_runs, hom_agents, het_runs, het_agents


def aggregate_hom_adversary(runs: pd.DataFrame, by_competition: bool = False, by_band: bool = False) -> pd.DataFrame:
    hom = runs[runs["experiment_family"].eq("homogeneous_adversary")].copy()
    group_cols = ["game_label", "n_agents", "adversary_model", "adversary_elo"]
    if by_competition:
        group_cols += ["competition_ci_rounded", "competition_label_ci"]
    if by_band:
        group_cols += ["competition_band"]
    return (
        hom.groupby(group_cols, dropna=False)
        .agg(
            run_count=("config_id", "count"),
            adversary_utility=("adversary_utility", "mean"),
            adversary_utility_sem=("adversary_utility", sem_series),
            baseline_mean_utility=("baseline_mean_utility", "mean"),
            baseline_mean_utility_sem=("baseline_mean_utility", sem_series),
            adversary_advantage=("adversary_advantage", "mean"),
            adversary_advantage_sem=("adversary_advantage", sem_series),
            adversary_z_advantage=("adversary_z_advantage", "mean"),
            adversary_z_advantage_sem=("adversary_z_advantage", sem_series),
            consensus_rate=("consensus_numeric", "mean"),
            consensus_rate_sem=("consensus_numeric", sem_series),
            mean_final_round=("final_round", "mean"),
            mean_final_round_sem=("final_round", sem_series),
        )
        .reset_index()
    )


def aggregate_het_agents(agents: pd.DataFrame, by_competition: bool = False) -> pd.DataFrame:
    het = agents[agents["experiment_family"].eq("heterogeneous_random")].copy()
    group_cols = ["game_label", "n_agents", "model", "model_short", "elo"]
    if by_competition:
        group_cols += ["competition_ci_rounded", "competition_label_ci", "competition_band"]
    return (
        het.groupby(group_cols, dropna=False)
        .agg(
            obs_count=("final_utility", "count"),
            final_utility=("final_utility", "mean"),
            final_utility_std=("final_utility", "std"),
            final_utility_sem=("final_utility", sem_series),
        )
        .reset_index()
    )


def plot_hom_adversary_payoff_vs_elo(hom_runs: pd.DataFrame) -> pd.DataFrame:
    agg = aggregate_hom_adversary(hom_runs, by_competition=False)
    slopes: list[dict[str, object]] = []
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.6), sharey=False)
    for ax, game in zip(axes, GAME_ORDER):
        game_df = agg[agg["game_label"].eq(game)]
        for n in N_ORDER:
            sub = game_df[game_df["n_agents"].eq(n)].sort_values("adversary_elo")
            if sub.empty:
                continue
            color = N_COLORS[n]
            slope, intercept, r2 = linear_fit(sub["adversary_elo"], sub["adversary_utility"])
            plot_errorbar_series(
                ax,
                sub["adversary_elo"],
                sub["adversary_utility"],
                sub["adversary_utility_sem"],
                color=color,
                label=f"N={n} ({format_slope_per_100(slope)})",
                marker="o",
                linewidth=1.2,
                alpha=0.9,
            )
            add_fit_line(ax, sub, "adversary_elo", "adversary_utility", color)
            slopes.append(
                {
                    "game_label": game,
                    "n_agents": n,
                    "slope_per_elo": slope,
                    "slope_per_100_elo": slope * 100 if math.isfinite(slope) else math.nan,
                    "r_squared": r2,
                    "n_points": len(sub),
                }
            )
            label_points(ax, sub, "adversary_elo", "adversary_utility", "adversary_model", fontsize=3.8, max_labels=40)
        tidy_axes(ax, "Adversary Arena Elo", "Adversary payoff", GAME_TITLES[game])
        ax.legend(fontsize=7, ncol=1, frameon=False)
    save_figure(fig, PLOTS_DIR / "hom_adversary_payoff_vs_elo_by_n.png")
    slopes_df = pd.DataFrame(slopes)
    save_csv(TABLES_DIR / "hom_adversary_payoff_vs_elo_slopes.csv", slopes_df)
    return slopes_df


def plot_hom_adversary_payoff_by_competition(hom_runs: pd.DataFrame) -> pd.DataFrame:
    agg = aggregate_hom_adversary(hom_runs, by_competition=True)
    slopes: list[dict[str, object]] = []
    for game in GAME_ORDER:
        fig, axes = plt.subplots(1, 5, figsize=(18.5, 4.0), sharey=True)
        game_df = agg[agg["game_label"].eq(game)]
        labels = sorted(game_df["competition_label_ci"].dropna().unique())
        color_map = {label: plt.cm.viridis(i / max(1, len(labels) - 1)) for i, label in enumerate(labels)}
        for ax, n in zip(axes, N_ORDER):
            n_df = game_df[game_df["n_agents"].eq(n)]
            slope_entries: list[tuple[str, float]] = []
            for label in labels:
                sub = n_df[n_df["competition_label_ci"].eq(label)].sort_values("adversary_elo")
                if sub.empty:
                    continue
                color = color_map[label]
                plot_errorbar_series(
                    ax,
                    sub["adversary_elo"],
                    sub["adversary_utility"],
                    sub["adversary_utility_sem"],
                    color=color,
                    label=label,
                    marker="o",
                    linewidth=0.95,
                    markersize=3.0,
                    alpha=0.88,
                )
                slope, intercept, r2 = add_fit_line(ax, sub, "adversary_elo", "adversary_utility", color, linewidth=0.8)
                slope_entries.append((label, slope))
                slopes.append(
                    {
                        "game_label": game,
                        "n_agents": n,
                        "competition_label": label,
                        "slope_per_elo": slope,
                        "slope_per_100_elo": slope * 100 if math.isfinite(slope) else math.nan,
                        "r_squared": r2,
                        "n_points": len(sub),
                    }
                )
            annotate_slope_block(ax, slope_entries, loc="upper left", fontsize=5.0)
            tidy_axes(ax, "Adversary Elo", "Adversary payoff", f"N={n}")
        handles, labels_out = axes[-1].get_legend_handles_labels()
        fig.legend(
            handles,
            labels_out,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.05),
            ncol=min(4, max(1, len(labels_out))),
            fontsize=7,
            frameon=False,
        )
        fig.suptitle(f"Homogeneous adversary payoff vs Elo by competition: {GAME_TITLES[game]}", fontsize=12, y=1.02)
        save_figure(fig, PLOTS_DIR / f"hom_adversary_payoff_vs_elo_by_competition_{game}.png")
    slopes_df = pd.DataFrame(slopes)
    save_csv(TABLES_DIR / "hom_adversary_payoff_vs_elo_by_competition_slopes.csv", slopes_df)
    return slopes_df


def plot_heterogeneous_payoff_vs_arena_elo(het_agents: pd.DataFrame) -> pd.DataFrame:
    agg = aggregate_het_agents(het_agents, by_competition=False)
    slopes: list[dict[str, object]] = []
    fig, axes = plt.subplots(1, 3, figsize=(15.8, 4.6), sharey=False)
    for ax, game in zip(axes, GAME_ORDER):
        game_df = agg[agg["game_label"].eq(game)]
        for n in N_ORDER:
            sub = game_df[game_df["n_agents"].eq(n)].sort_values("elo")
            if sub.empty:
                continue
            color = N_COLORS[n]
            slope, intercept, r2 = linear_fit(sub["elo"], sub["final_utility"])
            plot_errorbar_series(
                ax,
                sub["elo"],
                sub["final_utility"],
                sub["final_utility_sem"],
                color=color,
                label=f"N={n} ({format_slope_per_100(slope)})",
                marker="o",
                linestyle="none",
                markersize=3.4,
                alpha=0.82,
            )
            add_fit_line(ax, sub, "elo", "final_utility", color)
            slopes.append(
                {
                    "game_label": game,
                    "n_agents": n,
                    "slope_per_elo": slope,
                    "slope_per_100_elo": slope * 100 if math.isfinite(slope) else math.nan,
                    "r_squared": r2,
                    "n_models": len(sub),
                    "mean_obs_per_model": sub["obs_count"].mean(),
                }
            )
            label_points(ax, sub, "elo", "final_utility", "model_short", fontsize=3.8, alpha=0.55, max_labels=120)
        tidy_axes(ax, "Arena Elo", "Mean model payoff", GAME_TITLES[game])
        ax.legend(fontsize=7, frameon=False)
    save_figure(fig, PLOTS_DIR / "heterogeneous_payoff_vs_arena_elo_by_n.png")
    slopes_df = pd.DataFrame(slopes)
    save_csv(TABLES_DIR / "heterogeneous_payoff_vs_arena_elo_slopes.csv", slopes_df)
    return slopes_df


def plot_heterogeneous_payoff_by_competition(het_agents: pd.DataFrame) -> pd.DataFrame:
    agg = aggregate_het_agents(het_agents, by_competition=True)
    slopes: list[dict[str, object]] = []
    for game in GAME_ORDER:
        fig, axes = plt.subplots(1, 5, figsize=(18.5, 4.0), sharey=True)
        game_df = agg[agg["game_label"].eq(game)]
        labels = sorted(game_df["competition_label_ci"].dropna().unique())
        color_map = {label: plt.cm.plasma(i / max(1, len(labels) - 1)) for i, label in enumerate(labels)}
        for ax, n in zip(axes, N_ORDER):
            n_df = game_df[game_df["n_agents"].eq(n)]
            slope_entries: list[tuple[str, float]] = []
            for label in labels:
                sub = n_df[n_df["competition_label_ci"].eq(label)].sort_values("elo")
                if sub.empty:
                    continue
                color = color_map[label]
                plot_errorbar_series(
                    ax,
                    sub["elo"],
                    sub["final_utility"],
                    sub["final_utility_sem"],
                    color=color,
                    label=label,
                    marker="o",
                    linestyle="none",
                    markersize=2.7,
                    alpha=0.72,
                )
                slope, intercept, r2 = add_fit_line(ax, sub, "elo", "final_utility", color, linewidth=0.8, alpha=0.72)
                slope_entries.append((label, slope))
                slopes.append(
                    {
                        "game_label": game,
                        "n_agents": n,
                        "competition_label": label,
                        "slope_per_elo": slope,
                        "slope_per_100_elo": slope * 100 if math.isfinite(slope) else math.nan,
                        "r_squared": r2,
                        "n_models": len(sub),
                    }
                )
            annotate_slope_block(ax, slope_entries, loc="upper left", fontsize=5.0)
            tidy_axes(ax, "Arena Elo", "Mean model payoff", f"N={n}")
        handles = [
            plt.Line2D([0], [0], color=color_map[label], marker="o", lw=1, ms=4, label=label)
            for label in labels
        ]
        fig.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.05),
            ncol=min(4, max(1, len(handles))),
            fontsize=7,
            frameon=False,
        )
        fig.suptitle(f"Heterogeneous payoff vs Arena Elo by competition: {GAME_TITLES[game]}", fontsize=12, y=1.02)
        save_figure(fig, PLOTS_DIR / f"heterogeneous_payoff_vs_arena_elo_by_competition_{game}.png")
    slopes_df = pd.DataFrame(slopes)
    save_csv(TABLES_DIR / "heterogeneous_payoff_vs_arena_elo_by_competition_slopes.csv", slopes_df)
    return slopes_df


def plot_homogeneous_dilution(hom_runs: pd.DataFrame) -> pd.DataFrame:
    agg = aggregate_hom_adversary(hom_runs, by_band=True)
    pooled = aggregate_hom_adversary(hom_runs, by_band=False)
    summary_rows: list[dict[str, object]] = []

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.5), sharey=False)
    adversaries = sorted(pooled["adversary_model"].dropna().unique())
    color_map = {model: plt.cm.tab10(i % 10) for i, model in enumerate(adversaries)}
    for ax, game in zip(axes, GAME_ORDER):
        game_df = pooled[pooled["game_label"].eq(game)]
        for model in adversaries:
            sub = game_df[game_df["adversary_model"].eq(model)].sort_values("n_agents")
            if sub.empty:
                continue
            plot_errorbar_series(
                ax,
                sub["n_agents"],
                sub["adversary_advantage"],
                sub["adversary_advantage_sem"],
                color=color_map[model],
                label=short_model_name(model),
                marker="o",
                linewidth=1.0,
            )
            n2 = sub[sub["n_agents"].eq(2)]["adversary_advantage"].mean()
            n10 = sub[sub["n_agents"].eq(10)]["adversary_advantage"].mean()
            summary_rows.append(
                {
                    "game_label": game,
                    "adversary_model": model,
                    "n2_advantage": n2,
                    "n10_advantage": n10,
                    "delta_n10_minus_n2": n10 - n2 if math.isfinite(clean_float(n2)) and math.isfinite(clean_float(n10)) else math.nan,
                }
            )
        ax.axhline(0, color="#444444", lw=0.7)
        tidy_axes(ax, "N agents", "Adversary payoff minus GPT-5-nano mean", GAME_TITLES[game])
        ax.set_xticks(N_ORDER)
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, fontsize=7, frameon=False)
    save_figure(fig, PLOTS_DIR / "hom_adversary_dilution_advantage_vs_n.png")

    fig, axes = plt.subplots(3, 2, figsize=(12.5, 11.0), sharex=True, sharey=False)
    for row, game in enumerate(GAME_ORDER):
        for col, band in enumerate(["cooperative", "competitive"]):
            ax = axes[row][col]
            game_df = agg[(agg["game_label"].eq(game)) & (agg["competition_band"].eq(band))]
            for model in adversaries:
                sub = game_df[game_df["adversary_model"].eq(model)].sort_values("n_agents")
                if sub.empty:
                    continue
                plot_errorbar_series(
                    ax,
                    sub["n_agents"],
                    sub["adversary_z_advantage"],
                    sub["adversary_z_advantage_sem"],
                    color=color_map[model],
                    label=short_model_name(model),
                    marker="o",
                    linewidth=0.95,
                    alpha=0.9,
                )
            ax.axhline(0, color="#444444", lw=0.7)
            tidy_axes(ax, "N agents\n(exact CI shown under each N)", "Within-run adversary z-advantage", f"{GAME_TITLES[game]} - {band} setting")
            set_n_ticks_with_ci(ax, hom_runs[hom_runs["experiment_family"].eq("homogeneous_adversary")], game, band)
    save_figure(fig, PLOTS_DIR / "hom_adversary_dilution_z_advantage_by_band.png")

    summary = pd.DataFrame(summary_rows)
    save_csv(TABLES_DIR / "hom_adversary_dilution_summary.csv", summary)
    return summary


def plot_homogeneous_control_order(hom_agents: pd.DataFrame) -> pd.DataFrame:
    control = hom_agents[hom_agents["experiment_family"].eq("homogeneous_control")].copy()
    summary = (
        control.groupby(["game_label", "n_agents", "agent_index", "competition_band"], dropna=False)
        .agg(
            mean_utility=("final_utility", "mean"),
            std_utility=("final_utility", "std"),
            sem_utility=("final_utility", sem_series),
            obs_count=("final_utility", "count"),
        )
        .reset_index()
    )

    fig, axes = plt.subplots(3, 5, figsize=(18.2, 10.0), sharey=False)
    for row, game in enumerate(GAME_ORDER):
        for col, n in enumerate(N_ORDER):
            ax = axes[row][col]
            sub_n = summary[(summary["game_label"].eq(game)) & (summary["n_agents"].eq(n))]
            for band in ["cooperative", "middle", "competitive", "all"]:
                sub = sub_n[sub_n["competition_band"].eq(band)].sort_values("agent_index")
                if sub.empty:
                    continue
                plot_errorbar_series(
                    ax,
                    sub["agent_index"],
                    sub["mean_utility"],
                    sub["sem_utility"],
                    color=BAND_COLORS.get(band, "#555555"),
                    label=exact_setting_label(control, game, int(n), band, compact=True),
                    marker="o",
                    linewidth=1.0,
                    markersize=3,
                )
            tidy_axes(ax, "Order", "Mean payoff", f"{GAME_TITLES[game]}\nN={n}")
            ax.set_xticks(range(1, n + 1))
            ax.legend(fontsize=4.7, frameon=False, loc="best")
    save_figure(fig, PLOTS_DIR / "homogeneous_control_order_dynamics.png")
    save_csv(TABLES_DIR / "homogeneous_control_order_summary.csv", summary)
    return summary


def compute_n2_multiagent_fairness() -> tuple[pd.DataFrame, pd.DataFrame]:
    run_records: list[dict[str, object]] = []
    agent_records: list[dict[str, object]] = []
    skipped_records: list[dict[str, object]] = []
    for root, source_group in [
        (HOM_ROOT, "n_eq_2_homogeneous"),
        (HET_ROOT, "n_eq_2_heterogeneous"),
    ]:
        for path in sorted((root / "runs").glob("*/experiment_results.json")):
            payload = load_fairness_json(path)
            cfg = payload.get("config") or {}
            if int(cfg.get("n_agents") or 0) != 2:
                continue
            row = FairAnalysisRow(
                source_group=source_group,
                dataset="n_eq_2_multiagent",
                game_id=normalize_fairness_game_id(payload),
                result_path=path,
                payload=payload,
                config=cfg,
                agent_model_map=dict(cfg.get("agent_model_map") or {}),
                agent_role_map=dict(cfg.get("agent_role_map") or {}),
                agent_elo_map=dict(cfg.get("agent_elo_map") or {}),
            )
            try:
                run_record, agent_rows = analyze_fairness_row(row)
            except Exception as exc:
                skipped_records.append(
                    {
                        "source_group": source_group,
                        "result_path": str(path.relative_to(PROJECT_ROOT)),
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue
            run_records.append(run_record)
            agent_records.extend(agent_rows)
    runs = pd.DataFrame(run_records)
    agents = pd.DataFrame(agent_records)
    save_csv(TABLES_DIR / "multiagent_n2_fairness_run_metrics.csv", runs)
    save_csv(TABLES_DIR / "multiagent_n2_fairness_agent_metrics.csv", agents)
    save_csv(TABLES_DIR / "multiagent_n2_fairness_skipped_rows.csv", pd.DataFrame(skipped_records))
    return runs, agents


def load_fairness() -> tuple[pd.DataFrame, pd.DataFrame]:
    run_metrics = pd.read_csv(FAIR_RUN_METRICS)
    agent_metrics = pd.read_csv(FAIR_AGENT_METRICS)
    n2_runs, n2_agents = compute_n2_multiagent_fairness()
    if not n2_runs.empty:
        run_metrics = pd.concat([run_metrics, n2_runs], ignore_index=True, sort=False)
    if not n2_agents.empty:
        agent_metrics = pd.concat([agent_metrics, n2_agents], ignore_index=True, sort=False)
    for frame in [run_metrics, agent_metrics]:
        frame["config_id"] = frame["result_path"].map(parse_config_id_from_path)
        frame["family_label"] = frame["experiment_family"].map(FAMILY_LABELS).fillna(frame["experiment_family"])
        if "game_id" in frame.columns:
            frame["game_label_short"] = frame["game_id"]
    return run_metrics, agent_metrics


def fairness_metric_name(game_id: str) -> str:
    return "lindahl_distance_norm" if game_id == "game3" else "nbs_distance_norm"


def add_fairness_metric(run_metrics: pd.DataFrame) -> pd.DataFrame:
    frame = run_metrics.copy()
    frame["fairness_distance"] = np.nan
    for game in GAME_ORDER:
        metric = fairness_metric_name(game)
        mask = frame["game_id"].eq(game)
        if metric in frame.columns:
            frame.loc[mask, "fairness_distance"] = pd.to_numeric(frame.loc[mask, metric], errors="coerce")
    frame["competition_band"] = "middle"
    keys = ["source_group", "game_id", "n_agents"]
    for _, idx in frame.groupby(keys, dropna=False).groups.items():
        idx_list = list(idx)
        values = pd.to_numeric(frame.loc[idx_list, "competition_value"], errors="coerce")
        finite = values[np.isfinite(values)]
        if finite.empty:
            continue
        min_ci = float(finite.min())
        max_ci = float(finite.max())
        if math.isclose(min_ci, max_ci):
            frame.loc[idx_list, "competition_band"] = "all"
        else:
            min_idx = values.index[values.sub(min_ci).abs() < 1e-7]
            max_idx = values.index[values.sub(max_ci).abs() < 1e-7]
            frame.loc[min_idx, "competition_band"] = "cooperative"
            frame.loc[max_idx, "competition_band"] = "competitive"
    return frame


def plot_multiagent_fairness(fair_runs: pd.DataFrame, fair_agents: pd.DataFrame, het_runs: pd.DataFrame) -> pd.DataFrame:
    fair = add_fairness_metric(fair_runs)
    multi = fair[
        fair["source_group"].isin(MULTIAGENT_FAIR_SOURCE_GROUPS)
        & fair["experiment_family"].isin(FAMILY_LABELS.keys())
    ].copy()
    summary = (
        multi.groupby(["game_id", "n_agents", "experiment_family", "family_label"], dropna=False)
        .agg(
            run_count=("result_path", "count"),
            fairness_distance=("fairness_distance", "mean"),
            fairness_distance_sem=("fairness_distance", sem_series),
            utility_gini=("utility_gini", "mean"),
            utility_gini_sem=("utility_gini", sem_series),
            sw_efficiency=("sw_efficiency", "mean"),
            sw_efficiency_sem=("sw_efficiency", sem_series),
            consensus_rate=("consensus_reached", "mean"),
            consensus_rate_sem=("consensus_reached", sem_series),
            final_round=("final_round", "mean"),
            final_round_sem=("final_round", sem_series),
        )
        .reset_index()
    )
    save_csv(TABLES_DIR / "multiagent_fairness_by_n_summary.csv", summary)

    for metric, ylabel, filename in [
        ("fairness_distance", "NBS/Lindahl distance", "multiagent_fairness_distance_vs_n.png"),
        ("utility_gini", "Utility Gini", "multiagent_utility_gini_vs_n.png"),
        ("sw_efficiency", "Social-welfare efficiency", "multiagent_social_welfare_efficiency_vs_n.png"),
    ]:
        fig, axes = plt.subplots(1, 3, figsize=(15.4, 4.4), sharey=False)
        for ax, game in zip(axes, GAME_ORDER):
            game_df = summary[summary["game_id"].eq(game)]
            for family, label in FAMILY_LABELS.items():
                sub = game_df[game_df["experiment_family"].eq(family)].sort_values("n_agents")
                if sub.empty:
                    continue
                plot_errorbar_series(
                    ax,
                    sub["n_agents"],
                    sub[metric],
                    sub.get(f"{metric}_sem", pd.Series(0.0, index=sub.index)),
                    color=FAMILY_COLORS.get(family, "#555555"),
                    label=label,
                    marker="o",
                    linewidth=1.2,
                )
            tidy_axes(ax, "N agents", ylabel, GAME_TITLES[game])
            ax.set_xticks(N_ORDER)
        handles, labels = axes[-1].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=8, frameon=False)
        save_figure(fig, PLOTS_DIR / filename)

    fair_agents_multi = fair_agents[
        fair_agents["source_group"].isin(MULTIAGENT_FAIR_SOURCE_GROUPS)
        & fair_agents["experiment_family"].isin(FAMILY_LABELS.keys())
    ].copy()
    fair_agents_multi["residual_metric"] = np.where(
        fair_agents_multi["game_id"].eq("game3"),
        pd.to_numeric(fair_agents_multi["lindahl_residual"], errors="coerce"),
        pd.to_numeric(fair_agents_multi["nbs_residual"], errors="coerce"),
    )
    agent_summary = (
        fair_agents_multi.groupby(["game_id", "experiment_family", "model", "elo"], dropna=False)
        .agg(
            obs_count=("result_path", "count"),
            residual_metric=("residual_metric", "mean"),
            residual_metric_sem=("residual_metric", sem_series),
            nbs_ratio=("nbs_ratio", "mean"),
            lindahl_ratio=("lindahl_ratio", "mean"),
        )
        .reset_index()
    )
    save_csv(TABLES_DIR / "multiagent_agent_fairness_residual_vs_elo_summary.csv", agent_summary)

    fig, axes = plt.subplots(3, 3, figsize=(14.6, 10.6), sharex=False, sharey=False)
    family_order = ["homogeneous_control", "homogeneous_adversary", "heterogeneous_random"]
    for row, game in enumerate(GAME_ORDER):
        for col, family in enumerate(family_order):
            ax = axes[row][col]
            sub = agent_summary[(agent_summary["game_id"].eq(game)) & (agent_summary["experiment_family"].eq(family))]
            slope = math.nan
            if not sub.empty:
                plot_errorbar_series(
                    ax,
                    sub["elo"],
                    sub["residual_metric"],
                    sub["residual_metric_sem"],
                    color=FAMILY_COLORS.get(family, "#555555"),
                    label=FAMILY_LABELS[family],
                    marker="o",
                    linestyle="none",
                    markersize=3.6,
                    alpha=0.82,
                )
                slope, _, _ = add_fit_line(ax, sub, "elo", "residual_metric", FAMILY_COLORS.get(family, "#555555"))
                label_points(ax, sub, "elo", "residual_metric", "model", fontsize=3.8, max_labels=50)
            ax.axhline(0, color="#444444", lw=0.7)
            tidy_axes(
                ax,
                "Arena Elo",
                "Actual minus fair utility",
                f"{GAME_TITLES[game]}\n{FAMILY_LABELS[family]} ({format_slope_per_100(slope)})",
            )
    save_figure(fig, PLOTS_DIR / "multiagent_agent_fairness_residual_vs_elo.png")

    fair_het = multi[multi["experiment_family"].eq("heterogeneous_random")].copy()
    if not fair_het.empty:
        fair_het = fair_het.merge(
            het_runs[["config_id", "game_label", "n_agents", "elo_std", "elo_variance", "competition_band"]].rename(
                columns={"game_label": "game_id", "competition_band": "competition_band_fresh"}
            ),
            on=["config_id", "game_id", "n_agents"],
            how="left",
        )
        fair_het["competition_band_plot"] = fair_het["competition_band_fresh"].fillna(fair_het["competition_band"])
        fig, axes = plt.subplots(1, 3, figsize=(15.4, 4.4), sharey=False)
        for ax, game in zip(axes, GAME_ORDER):
            sub_game = fair_het[fair_het["game_id"].eq(game)]
            slope_entries: list[tuple[str, float]] = []
            for band in ["cooperative", "middle", "competitive"]:
                sub = sub_game[sub_game["competition_band_plot"].eq(band)]
                if sub.empty:
                    continue
                label = exact_setting_label(
                    het_runs.rename(columns={"game_label": "game_label"}),
                    game,
                    None,
                    band,
                    compact=True,
                )
                ax.scatter(
                    sub["elo_std"],
                    sub["fairness_distance"],
                    s=18,
                    alpha=0.30,
                    color=BAND_COLORS.get(band, "#555555"),
                    label=f"{label} runs",
                )
                add_binned_sem_errorbars(
                    ax,
                    sub,
                    "elo_std",
                    "fairness_distance",
                    BAND_COLORS.get(band, "#555555"),
                    label,
                )
                slope, _, _ = add_fit_line(ax, sub, "elo_std", "fairness_distance", BAND_COLORS.get(band, "#555555"), linewidth=0.9)
                slope_entries.append((label, slope))
            annotate_slope_block(ax, slope_entries, loc="upper left", fontsize=5.2, unit="100 Elo std.")
            tidy_axes(ax, "Roster Elo std. dev.", "NBS/Lindahl distance", GAME_TITLES[game])
        handles, labels = axes[-1].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.06),
            ncol=3,
            fontsize=7,
            frameon=False,
        )
        save_figure(fig, PLOTS_DIR / "heterogeneous_elo_dispersion_vs_fairness_distance.png")
        save_csv(TABLES_DIR / "heterogeneous_elo_dispersion_fairness_rows.csv", fair_het)
    return summary


def plot_heterogeneous_elo_dispersion(het_runs: pd.DataFrame) -> pd.DataFrame:
    het = het_runs[het_runs["experiment_family"].eq("heterogeneous_random")].copy()
    rows: list[dict[str, object]] = []
    fig, axes = plt.subplots(1, 3, figsize=(15.4, 4.4), sharey=False)
    for ax, game in zip(axes, GAME_ORDER):
        game_df = het[het["game_label"].eq(game)]
        slope_entries: list[tuple[str, float]] = []
        for band in ["cooperative", "middle", "competitive"]:
            sub = game_df[game_df["competition_band"].eq(band)]
            if sub.empty:
                continue
            label = exact_setting_label(het, game, None, band, compact=True)
            ax.scatter(
                sub["elo_std"],
                sub["utility_gini_shifted"],
                s=16,
                alpha=0.28,
                color=BAND_COLORS.get(band, "#555555"),
                label=f"{label} runs",
            )
            add_binned_sem_errorbars(
                ax,
                sub,
                "elo_std",
                "utility_gini_shifted",
                BAND_COLORS.get(band, "#555555"),
                label,
            )
            slope, intercept, r2 = add_fit_line(
                ax,
                sub,
                "elo_std",
                "utility_gini_shifted",
                BAND_COLORS.get(band, "#555555"),
                linewidth=0.9,
            )
            slope_entries.append((label, slope))
            rows.append(
                {
                    "game_label": game,
                    "competition_band": band,
                    "slope_gini_per_elo_std": slope,
                    "slope_gini_per_100_elo_std": slope * 100 if math.isfinite(slope) else math.nan,
                    "r_squared": r2,
                    "n_runs": len(sub),
                }
            )
        annotate_slope_block(ax, slope_entries, loc="upper left", fontsize=5.2, unit="100 Elo std.")
        tidy_axes(ax, "Roster Elo std. dev.", "Shifted utility Gini", GAME_TITLES[game])
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.06),
        ncol=3,
        fontsize=7,
        frameon=False,
    )
    save_figure(fig, PLOTS_DIR / "heterogeneous_elo_dispersion_vs_gini.png")

    by_n_rows: list[dict[str, object]] = []
    for n in N_ORDER:
        fig, axes = plt.subplots(1, 3, figsize=(15.4, 4.4), sharey=False)
        for ax, game in zip(axes, GAME_ORDER):
            game_df = het[(het["game_label"].eq(game)) & (het["n_agents"].eq(n))]
            slope_entries = []
            for band in ["cooperative", "middle", "competitive"]:
                sub = game_df[game_df["competition_band"].eq(band)]
                if sub.empty:
                    continue
                label = exact_setting_label(het, game, n, band, compact=True)
                ax.scatter(
                    sub["elo_std"],
                    sub["utility_gini_shifted"],
                    s=20,
                    alpha=0.30,
                    color=BAND_COLORS.get(band, "#555555"),
                    label=f"{label} runs",
                )
                add_binned_sem_errorbars(
                    ax,
                    sub,
                    "elo_std",
                    "utility_gini_shifted",
                    BAND_COLORS.get(band, "#555555"),
                    label,
                )
                slope, intercept, r2 = add_fit_line(
                    ax,
                    sub,
                    "elo_std",
                    "utility_gini_shifted",
                    BAND_COLORS.get(band, "#555555"),
                    linewidth=0.9,
                )
                slope_entries.append((label, slope))
                by_n_rows.append(
                    {
                        "game_label": game,
                        "n_agents": n,
                        "competition_band": band,
                        "competition_setting": label,
                        "slope_gini_per_elo_std": slope,
                        "slope_gini_per_100_elo_std": slope * 100 if math.isfinite(slope) else math.nan,
                        "r_squared": r2,
                        "n_runs": len(sub),
                    }
                )
            annotate_slope_block(ax, slope_entries, loc="upper left", fontsize=5.2, unit="100 Elo std.")
            tidy_axes(ax, "Roster Elo std. dev.", "Shifted utility Gini", f"{GAME_TITLES[game]} - N={n}")
        handles, labels = axes[-1].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.06),
            ncol=min(3, max(1, len(labels))),
            fontsize=7,
            frameon=False,
        )
        save_figure(fig, PLOTS_DIR / f"heterogeneous_elo_dispersion_vs_gini_n{n}.png")
    slopes = pd.DataFrame(rows)
    save_csv(TABLES_DIR / "heterogeneous_elo_dispersion_vs_gini_slopes.csv", slopes)
    save_csv(TABLES_DIR / "heterogeneous_elo_dispersion_vs_gini_by_n_slopes.csv", pd.DataFrame(by_n_rows))
    return slopes


def plot_general_n_dynamics(hom_runs: pd.DataFrame, het_runs: pd.DataFrame) -> pd.DataFrame:
    runs = pd.concat([hom_runs, het_runs], ignore_index=True)
    runs = runs[runs["experiment_family"].isin(FAMILY_LABELS.keys())].copy()
    summary = (
        runs.groupby(["game_label", "n_agents", "experiment_family", "family_label"], dropna=False)
        .agg(
            run_count=("config_id", "count"),
            mean_per_agent_utility=("mean_utility", "mean"),
            mean_per_agent_utility_sem=("mean_utility", sem_series),
            shifted_gini=("utility_gini_shifted", "mean"),
            shifted_gini_sem=("utility_gini_shifted", sem_series),
            consensus_rate=("consensus_numeric", "mean"),
            consensus_rate_sem=("consensus_numeric", sem_series),
            mean_final_round=("final_round", "mean"),
            mean_final_round_sem=("final_round", sem_series),
            mean_elo_std=("elo_std", "mean"),
            mean_elo_std_sem=("elo_std", sem_series),
        )
        .reset_index()
    )
    save_csv(TABLES_DIR / "general_n_dynamics_summary.csv", summary)
    for metric, ylabel, filename in [
        ("mean_per_agent_utility", "Mean per-agent payoff", "general_mean_payoff_vs_n.png"),
        ("shifted_gini", "Shifted utility Gini", "general_shifted_gini_vs_n.png"),
        ("consensus_rate", "Consensus rate", "general_consensus_rate_vs_n.png"),
        ("mean_final_round", "Mean final round", "general_rounds_to_consensus_vs_n.png"),
    ]:
        fig, axes = plt.subplots(1, 3, figsize=(15.4, 4.4), sharey=False)
        for ax, game in zip(axes, GAME_ORDER):
            game_df = summary[summary["game_label"].eq(game)]
            for family, label in FAMILY_LABELS.items():
                sub = game_df[game_df["experiment_family"].eq(family)].sort_values("n_agents")
                if sub.empty:
                    continue
                plot_errorbar_series(
                    ax,
                    sub["n_agents"],
                    sub[metric],
                    sub.get(f"{metric}_sem", pd.Series(0.0, index=sub.index)),
                    color=FAMILY_COLORS.get(family, "#555555"),
                    label=label,
                    marker="o",
                    linewidth=1.2,
                )
            tidy_axes(ax, "N agents", ylabel, GAME_TITLES[game])
            ax.set_xticks(N_ORDER)
        handles, labels = axes[-1].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=8, frameon=False)
        save_figure(fig, PLOTS_DIR / filename)
    return summary


def fit_performance_elo(agent_rows: pd.DataFrame, min_model_obs: int = 5) -> pd.DataFrame:
    rows = agent_rows.dropna(subset=["model", "elo", "final_utility", "run_key"]).copy()
    if rows.empty:
        return pd.DataFrame()
    obs_counts = rows.groupby("model")["run_key"].nunique()
    keep_models = sorted(obs_counts[obs_counts >= min_model_obs].index.tolist())
    if len(keep_models) < 2:
        return pd.DataFrame()
    model_to_idx = {model: i for i, model in enumerate(keep_models)}
    comparisons_i: list[int] = []
    comparisons_j: list[int] = []
    scores: list[float] = []
    for _, run_df in rows[rows["model"].isin(keep_models)].groupby("run_key"):
        run_agents = run_df[["model", "final_utility"]].dropna().reset_index(drop=True)
        if len(run_agents) < 2:
            continue
        for a, b in combinations(range(len(run_agents)), 2):
            model_a = run_agents.loc[a, "model"]
            model_b = run_agents.loc[b, "model"]
            if model_a == model_b:
                continue
            utility_a = float(run_agents.loc[a, "final_utility"])
            utility_b = float(run_agents.loc[b, "final_utility"])
            if math.isclose(utility_a, utility_b, abs_tol=1e-9):
                score = 0.5
            else:
                score = 1.0 if utility_a > utility_b else 0.0
            comparisons_i.append(model_to_idx[model_a])
            comparisons_j.append(model_to_idx[model_b])
            scores.append(score)
    if len(scores) < len(keep_models):
        return pd.DataFrame()

    i_arr = np.asarray(comparisons_i, dtype=int)
    j_arr = np.asarray(comparisons_j, dtype=int)
    score_arr = np.asarray(scores, dtype=float)
    scale = math.log(10.0) / 400.0
    ridge = 2e-5

    def objective(theta: np.ndarray) -> float:
        centered = theta - np.mean(theta)
        logits = (centered[i_arr] - centered[j_arr]) * scale
        prob = np.clip(expit(logits), 1e-8, 1.0 - 1e-8)
        loss = -float(np.sum(score_arr * np.log(prob) + (1.0 - score_arr) * np.log(1.0 - prob)))
        penalty = ridge * float(np.sum(centered**2))
        return loss + penalty

    result = minimize(objective, np.zeros(len(keep_models), dtype=float), method="L-BFGS-B", options={"maxiter": 800})
    theta = result.x - float(np.mean(result.x))
    ratings = 1500.0 + theta
    try:
        h_inv = np.asarray(result.hess_inv.todense(), dtype=float)
        center = np.eye(len(keep_models)) - np.ones((len(keep_models), len(keep_models))) / len(keep_models)
        cov = center @ h_inv @ center
        rating_se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    except Exception:
        rating_se = np.full(len(keep_models), np.nan)
    arena = rows.groupby("model")["elo"].mean()
    short = rows.groupby("model")["model_short"].agg(lambda values: values.dropna().iloc[0] if len(values.dropna()) else "")
    out = pd.DataFrame(
        {
            "model": keep_models,
            "model_short": [short.get(model, short_model_name(model)) for model in keep_models],
            "arena_elo": [arena.get(model, math.nan) for model in keep_models],
            "performance_elo": ratings,
            "performance_elo_se": rating_se,
            "run_obs": [obs_counts.get(model, 0) for model in keep_models],
            "pairwise_comparisons": len(scores),
            "optimizer_success": bool(result.success),
        }
    )
    return out.sort_values("performance_elo", ascending=False)


def build_performance_elos(het_agents: pd.DataFrame) -> pd.DataFrame:
    het = het_agents[het_agents["experiment_family"].eq("heterogeneous_random")].copy()
    records: list[pd.DataFrame] = []
    for game in GAME_ORDER:
        game_df = het[het["game_label"].eq(game)]
        for n in N_ORDER:
            n_df = game_df[game_df["n_agents"].eq(n)]
            fitted = fit_performance_elo(n_df, min_model_obs=3 if n == 2 else 5)
            if not fitted.empty:
                fitted["game_label"] = game
                fitted["n_agents"] = n
                fitted["competition_band"] = "all"
                fitted["scope"] = "by_n"
                records.append(fitted)
            for band in ["cooperative", "competitive"]:
                band_df = n_df[n_df["competition_band"].eq(band)]
                fitted = fit_performance_elo(band_df, min_model_obs=2 if n == 2 else 3)
                if not fitted.empty:
                    fitted["game_label"] = game
                    fitted["n_agents"] = n
                    fitted["competition_band"] = band
                    fitted["scope"] = "by_n"
                    records.append(fitted)
        for band in ["all", "cooperative", "competitive"]:
            band_df = game_df if band == "all" else game_df[game_df["competition_band"].eq(band)]
            fitted = fit_performance_elo(band_df, min_model_obs=5)
            if not fitted.empty:
                fitted["game_label"] = game
                fitted["n_agents"] = -1
                fitted["competition_band"] = band
                fitted["scope"] = "all_n"
                records.append(fitted)
    rankings = pd.concat(records, ignore_index=True) if records else pd.DataFrame()
    save_csv(TABLES_DIR / "heterogeneous_performance_elo_rankings.csv", rankings)
    return rankings


def plot_performance_elos(rankings: pd.DataFrame, het_agents: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    if rankings.empty:
        return pd.DataFrame()
    corr_rows: list[dict[str, object]] = []
    by_n = rankings[(rankings["scope"].eq("by_n")) & (rankings["competition_band"].eq("all"))]
    fig, axes = plt.subplots(3, 5, figsize=(18.2, 10.0), sharex=True, sharey=True)
    for row, game in enumerate(GAME_ORDER):
        for col, n in enumerate(N_ORDER):
            ax = axes[row][col]
            sub = by_n[(by_n["game_label"].eq(game)) & (by_n["n_agents"].eq(n))]
            slope = math.nan
            if not sub.empty:
                plot_errorbar_series(
                    ax,
                    sub["arena_elo"],
                    sub["performance_elo"],
                    sub["performance_elo_se"],
                    color=N_COLORS[n],
                    label=f"N={n}",
                    marker="o",
                    linestyle="none",
                    markersize=3.4,
                    alpha=0.82,
                )
                slope, intercept, r2 = add_fit_line(ax, sub, "arena_elo", "performance_elo", N_COLORS[n])
                corr = sub[["arena_elo", "performance_elo"]].corr().iloc[0, 1] if len(sub) > 1 else math.nan
                corr_rows.append(
                    {
                        "game_label": game,
                        "n_agents": n,
                        "competition_band": "all",
                        "arena_performance_corr": corr,
                        "slope_per_elo": slope,
                        "r_squared": r2,
                        "n_models": len(sub),
                    }
                )
                label_points(ax, sub, "arena_elo", "performance_elo", "model_short", fontsize=3.7, max_labels=55)
            tidy_axes(ax, "Arena Elo", "Payoff performance Elo", f"{GAME_TITLES[game]}\nN={n}; {format_slope_per_100(slope)}")
    save_figure(fig, PLOTS_DIR / "heterogeneous_performance_elo_vs_arena_by_game_n.png")

    all_n = rankings[(rankings["scope"].eq("all_n")) & (rankings["competition_band"].isin(["cooperative", "competitive"]))]
    fig, axes = plt.subplots(3, 2, figsize=(11.8, 10.0), sharex=True, sharey=True)
    for row, game in enumerate(GAME_ORDER):
        for col, band in enumerate(["cooperative", "competitive"]):
            ax = axes[row][col]
            sub = all_n[(all_n["game_label"].eq(game)) & (all_n["competition_band"].eq(band))]
            slope = math.nan
            if not sub.empty:
                color = BAND_COLORS[band]
                plot_errorbar_series(
                    ax,
                    sub["arena_elo"],
                    sub["performance_elo"],
                    sub["performance_elo_se"],
                    color=color,
                    label=band,
                    marker="o",
                    linestyle="none",
                    markersize=3.8,
                    alpha=0.82,
                )
                slope, intercept, r2 = add_fit_line(ax, sub, "arena_elo", "performance_elo", color)
                corr = sub[["arena_elo", "performance_elo"]].corr().iloc[0, 1] if len(sub) > 1 else math.nan
                corr_rows.append(
                    {
                        "game_label": game,
                        "n_agents": "all",
                        "competition_band": band,
                        "arena_performance_corr": corr,
                        "slope_per_elo": slope,
                        "r_squared": r2,
                        "n_models": len(sub),
                    }
                )
                label_points(ax, sub, "arena_elo", "performance_elo", "model_short", fontsize=3.9, max_labels=60)
            band_title = exact_setting_label(het_agents, game, None, band, compact=True) if het_agents is not None else band
            tidy_axes(ax, "Arena Elo", "Payoff performance Elo", f"{GAME_TITLES[game]} - {band_title}; {format_slope_per_100(slope)}")
    save_figure(fig, PLOTS_DIR / "heterogeneous_performance_elo_vs_arena_by_competition_band.png")
    corr_df = pd.DataFrame(corr_rows)
    save_csv(TABLES_DIR / "heterogeneous_performance_elo_correlations.csv", corr_df)
    return corr_df


def plot_n2_baseline_vs_heterogeneous(het_agents: pd.DataFrame) -> pd.DataFrame:
    n2 = pd.read_csv(N2_RUN_METRICS)
    n2 = n2[n2["baseline_key"].eq("gpt5_nano")].copy()
    n2["game_label_short"] = n2["game_id"]
    n2_agg = (
        n2.groupby(["game_label_short", "adversary_model", "adversary_short", "adversary_elo"], dropna=False)
        .agg(
            n2_baseline_adversary_utility=("adversary_utility", "mean"),
            n2_baseline_adversary_utility_sem=("adversary_utility", sem_series),
            n2_baseline_obs=("adversary_utility", "count"),
        )
        .reset_index()
    )
    het = het_agents[
        (het_agents["experiment_family"].eq("heterogeneous_random")) & (het_agents["n_agents"].eq(2))
    ].copy()
    het_agg = (
        het.groupby(["game_label", "model", "model_short", "elo"], dropna=False)
        .agg(
            heterogeneous_n2_utility=("final_utility", "mean"),
            heterogeneous_n2_utility_sem=("final_utility", sem_series),
            heterogeneous_obs=("final_utility", "count"),
        )
        .reset_index()
    )
    slope_rows: list[dict[str, object]] = []
    fig, axes = plt.subplots(1, 3, figsize=(15.8, 4.6), sharey=False)
    for ax, game in zip(axes, GAME_ORDER):
        base_sub = n2_agg[n2_agg["game_label_short"].eq(game)].sort_values("adversary_elo")
        het_sub = het_agg[het_agg["game_label"].eq(game)].sort_values("elo")
        if not base_sub.empty:
            slope, intercept, r2 = linear_fit(base_sub["adversary_elo"], base_sub["n2_baseline_adversary_utility"])
            plot_errorbar_series(
                ax,
                base_sub["adversary_elo"],
                base_sub["n2_baseline_adversary_utility"],
                base_sub["n2_baseline_adversary_utility_sem"],
                color="#d62728",
                label=f"N=2 vs GPT-5-nano baseline ({format_slope_per_100(slope)})",
                marker="o",
                linestyle="none",
                markersize=3.6,
                alpha=0.75,
            )
            add_fit_line(ax, base_sub, "adversary_elo", "n2_baseline_adversary_utility", "#d62728")
            slope_rows.append(
                {
                    "game_label": game,
                    "dataset": "n2_baseline",
                    "slope_per_100_elo": slope * 100 if math.isfinite(slope) else math.nan,
                    "r_squared": r2,
                    "n_models": len(base_sub),
                }
            )
        if not het_sub.empty:
            slope, intercept, r2 = linear_fit(het_sub["elo"], het_sub["heterogeneous_n2_utility"])
            plot_errorbar_series(
                ax,
                het_sub["elo"],
                het_sub["heterogeneous_n2_utility"],
                het_sub["heterogeneous_n2_utility_sem"],
                color="#1f77b4",
                label=f"N=2 heterogeneous pairing ({format_slope_per_100(slope)})",
                marker="o",
                linestyle="none",
                markersize=3.6,
                alpha=0.75,
            )
            add_fit_line(ax, het_sub, "elo", "heterogeneous_n2_utility", "#1f77b4")
            slope_rows.append(
                {
                    "game_label": game,
                    "dataset": "heterogeneous_n2",
                    "slope_per_100_elo": slope * 100 if math.isfinite(slope) else math.nan,
                    "r_squared": r2,
                    "n_models": len(het_sub),
                }
            )
        tidy_axes(ax, "Arena Elo", "Mean payoff", GAME_TITLES[game])
        ax.legend(fontsize=7, frameon=False)
    save_figure(fig, PLOTS_DIR / "n2_baseline_vs_heterogeneous_pairings.png")
    slopes = pd.DataFrame(slope_rows)
    save_csv(TABLES_DIR / "n2_baseline_vs_heterogeneous_pairing_slopes.csv", slopes)
    return slopes


def markdown_table(frame: pd.DataFrame, columns: Sequence[str], max_rows: int = 12, float_digits: int = 3) -> str:
    if frame.empty:
        return "_No rows._"
    view = frame.loc[:, [col for col in columns if col in frame.columns]].head(max_rows).copy()
    for col in view.columns:
        if pd.api.types.is_numeric_dtype(view[col]):
            view[col] = view[col].map(lambda value: "" if pd.isna(value) else f"{float(value):.{float_digits}f}")
    headers = list(view.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in headers) + " |")
    return "\n".join(lines)


def slope_summary_table(slopes: pd.DataFrame, group_name: str) -> pd.DataFrame:
    if slopes.empty:
        return pd.DataFrame()
    return (
        slopes.groupby("game_label", dropna=False)
        .agg(
            mean_slope_per_100_elo=("slope_per_100_elo", "mean"),
            min_slope_per_100_elo=("slope_per_100_elo", "min"),
            max_slope_per_100_elo=("slope_per_100_elo", "max"),
            group_count=("slope_per_100_elo", "count"),
        )
        .reset_index()
        .assign(section=group_name)
    )


def create_report(
    hom_runs: pd.DataFrame,
    hom_agents: pd.DataFrame,
    het_runs: pd.DataFrame,
    het_agents: pd.DataFrame,
    hom_slopes: pd.DataFrame,
    hom_comp_slopes: pd.DataFrame,
    het_slopes: pd.DataFrame,
    het_comp_slopes: pd.DataFrame,
    dilution: pd.DataFrame,
    fairness_summary: pd.DataFrame,
    dispersion_slopes: pd.DataFrame,
    perf_corr: pd.DataFrame,
    n2_compare_slopes: pd.DataFrame,
) -> None:
    old_text = N2_REPORT.read_text(encoding="utf-8")
    for asset_dir in ["gpt5_nano", "llama33"]:
        src = N2_REPORT.parent / asset_dir
        dst = OUT_DIR / asset_dir
        if src.exists():
            shutil.copytree(src, dst, dirs_exist_ok=True)

    total_hom_runs = len(hom_runs)
    total_het_runs = len(het_runs)
    total_hom_agents = len(hom_agents)
    total_het_agents = len(het_agents)
    counts = (
        pd.concat([hom_runs, het_runs], ignore_index=True)
        .groupby(["source_group", "experiment_family", "game_label", "n_agents"], dropna=False)
        .size()
        .reset_index(name="run_count")
    )
    save_csv(TABLES_DIR / "multiagent_run_counts.csv", counts)

    slope_table = pd.concat(
        [
            slope_summary_table(hom_slopes, "Homogeneous adversary"),
            slope_summary_table(het_slopes, "Heterogeneous"),
        ],
        ignore_index=True,
    )

    dilution_summary = (
        dilution.groupby("game_label", dropna=False)
        .agg(
            mean_delta_n10_minus_n2=("delta_n10_minus_n2", "mean"),
            min_delta_n10_minus_n2=("delta_n10_minus_n2", "min"),
            max_delta_n10_minus_n2=("delta_n10_minus_n2", "max"),
        )
        .reset_index()
    )

    fair_simple = fairness_summary.copy()
    if not fair_simple.empty:
        fair_simple = (
            fair_simple.groupby(["game_id", "experiment_family", "family_label"], dropna=False)
            .agg(
                mean_fairness_distance=("fairness_distance", "mean"),
                mean_gini=("utility_gini", "mean"),
                mean_sw_efficiency=("sw_efficiency", "mean"),
            )
            .reset_index()
        )

    perf_top = pd.read_csv(TABLES_DIR / "heterogeneous_performance_elo_rankings.csv")
    perf_comp = perf_top[
        perf_top["scope"].eq("all_n") & perf_top["competition_band"].eq("competitive")
    ].sort_values(["game_label", "performance_elo"], ascending=[True, False])
    top_lines = []
    for game in GAME_ORDER:
        sub = perf_comp[perf_comp["game_label"].eq(game)].head(5)
        if sub.empty:
            continue
        top = ", ".join(f"{row.model_short} ({row.performance_elo:.0f})" for row in sub.itertuples())
        top_lines.append(f"- {GAME_TITLES[game]}: {top}")

    skipped_n2_path = TABLES_DIR / "multiagent_n2_fairness_skipped_rows.csv"
    skipped_n2_count = len(pd.read_csv(skipped_n2_path)) if skipped_n2_path.exists() else 0
    skipped_n2_note = (
        f" Four malformed N=2 Game 1 rows had no `agent_preferences` and were skipped for exact fairness only; they are listed in `tables_multiagent/multiagent_n2_fairness_skipped_rows.csv`."
        if skipped_n2_count
        else ""
    )

    section = f"""

---

# Second Half: N > 2 Multi-Agent Analysis

This section analyzes the multi-agent experiments with `N = 2, 4, 6, 8, 10`. I parsed the raw homogeneous and heterogeneous experiment results rather than reusing the older preliminary plot CSVs, because this report needs the paper's competition index:

- Game 1: `CI = competition_level`.
- Game 2: `CI = theta * (1 - rho) / 2`.
- Game 3: `CI = (1 - alpha) * (1 - sigma)`.

I use per-agent payoff, adversary payoff, adversary minus fleet mean, within-run z-advantage, shifted Gini, NBS/Lindahl distance, and welfare efficiency for cross-`N` plots. I avoid total utility as a primary cross-`N` metric because total utility mechanically changes with the number of agents and with the game scale. Error bars are `mean ± 1 SEM` wherever a plotted point summarizes repeated runs; raw-run dispersion plots retain the raw points and overlay binned `mean ± 1 SEM`; payoff-performance Elo bars use the fitted Bradley-Terry inverse-Hessian standard error.

Parsed rows: `{total_hom_runs}` homogeneous runs / `{total_hom_agents}` homogeneous agent rows, and `{total_het_runs}` heterogeneous runs / `{total_het_agents}` heterogeneous agent rows.

## 1. Capability Scaling With More Agents

### Homogeneous Adversary

In homogeneous adversary runs, one non-nano adversary negotiates with `N-1` GPT-5-nano agents. The main plot shows adversary payoff vs the adversary's Arena Elo, with one curve per `N`.

![Homogeneous adversary payoff vs Elo](plots_multiagent/hom_adversary_payoff_vs_elo_by_n.png)

Slope summary, averaged across the five `N` curves:

{markdown_table(slope_table[slope_table['section'].eq('Homogeneous adversary')], ['game_label', 'mean_slope_per_100_elo', 'min_slope_per_100_elo', 'max_slope_per_100_elo', 'group_count'])}

Competition-stratified versions:

![Homogeneous adversary Game 1 by competition](plots_multiagent/hom_adversary_payoff_vs_elo_by_competition_game1.png)

![Homogeneous adversary Game 2 by competition](plots_multiagent/hom_adversary_payoff_vs_elo_by_competition_game2.png)

![Homogeneous adversary Game 3 by competition](plots_multiagent/hom_adversary_payoff_vs_elo_by_competition_game3.png)

### Heterogeneous Random Societies

In heterogeneous runs, each roster is a random draw of models. The y-axis is the model's mean payoff whenever it appeared in that game and `N`; the x-axis is the canonical Arena Elo.

![Heterogeneous payoff vs Arena Elo](plots_multiagent/heterogeneous_payoff_vs_arena_elo_by_n.png)

Slope summary:

{markdown_table(slope_table[slope_table['section'].eq('Heterogeneous')], ['game_label', 'mean_slope_per_100_elo', 'min_slope_per_100_elo', 'max_slope_per_100_elo', 'group_count'])}

Competition-stratified heterogeneous plots:

![Heterogeneous Game 1 by competition](plots_multiagent/heterogeneous_payoff_vs_arena_elo_by_competition_game1.png)

![Heterogeneous Game 2 by competition](plots_multiagent/heterogeneous_payoff_vs_arena_elo_by_competition_game2.png)

![Heterogeneous Game 3 by competition](plots_multiagent/heterogeneous_payoff_vs_arena_elo_by_competition_game3.png)

## 2. Does a Larger Fleet Dilute Exploitation?

The most comparable homogeneous metric is not total welfare. It is the adversary's payoff advantage over the GPT-5-nano fleet mean, and the within-run z-advantage. These normalize away the mechanical increase in participants.

![Homogeneous adversary advantage vs N](plots_multiagent/hom_adversary_dilution_advantage_vs_n.png)

![Homogeneous adversary z advantage by band](plots_multiagent/hom_adversary_dilution_z_advantage_by_band.png)

Change from `N=2` to `N=10`, averaged across adversaries:

{markdown_table(dilution_summary, ['game_label', 'mean_delta_n10_minus_n2', 'min_delta_n10_minus_n2', 'max_delta_n10_minus_n2'])}

Interpretation: when the average `N=10 - N=2` advantage is negative, the larger GPT-5-nano fleet is diluting the inserted adversary's edge. When it is positive, adding agents is not protective in that game/cell. The z-advantage plot is especially important for competitive cells because it asks whether the adversary is exceptional relative to the realized distribution in that exact run.

## 3. Heterogeneous Performance Elo

I also computed a payoff-based Elo ranking for heterogeneous runs. For each multi-agent result, I converted the realized utilities into all pairwise comparisons among agents in the same run. If model `i` earned higher utility than model `j`, `i` scored 1; if tied, both scored 0.5. I then fit a Bradley-Terry/Elo model by maximum likelihood:

`P(i beats j) = 1 / (1 + 10 ** ((R_j - R_i) / 400))`

Ratings are centered to mean 1500 within each fitted subset. This is not the external Arena Elo; it is an experiment-specific payoff performance Elo inferred from shared-roster outcomes.

![Heterogeneous performance Elo by game and N](plots_multiagent/heterogeneous_performance_elo_vs_arena_by_game_n.png)

![Heterogeneous performance Elo by competition band](plots_multiagent/heterogeneous_performance_elo_vs_arena_by_competition_band.png)

Competitive-setting top payoff-performance Elo models:

{chr(10).join(top_lines) if top_lines else '_No competitive performance Elo rows were fitted._'}

Correlation and slope diagnostics are saved in `tables_multiagent/heterogeneous_performance_elo_correlations.csv`.

## 4. Fairness, Nash/Lindahl Distance, and Gini

For Game 1 and Game 2, the fairness distance is the normalized distance from the Nash bargaining solution. For Game 3, it is the normalized Lindahl-style distance. The N>2 benchmark rows come from `analysis/nash_lindahl_fairness_20260505`; I additionally recomputed the exact `N=2` multiagent rows from the homogeneous and heterogeneous roots so the fairness plots cover the full `N = 2, 4, 6, 8, 10` sweep.{skipped_n2_note} Results are plotted by experiment family so the control, adversary, and heterogeneous regimes remain separate.

![Fairness distance vs N](plots_multiagent/multiagent_fairness_distance_vs_n.png)

![Utility Gini vs N](plots_multiagent/multiagent_utility_gini_vs_n.png)

![Social welfare efficiency vs N](plots_multiagent/multiagent_social_welfare_efficiency_vs_n.png)

Family-level averages:

{markdown_table(fair_simple, ['game_id', 'family_label', 'mean_fairness_distance', 'mean_gini', 'mean_sw_efficiency'], max_rows=12)}

Agent-level residuals show which models end above or below the fair benchmark:

![Agent fairness residual vs Elo](plots_multiagent/multiagent_agent_fairness_residual_vs_elo.png)

## 5. Elo Variance and Inequality

The heterogeneous setting lets us ask whether higher roster Elo dispersion predicts inequality. The first plot uses all `N = 2, 4, 6, 8, 10` heterogeneous runs and shifted Gini, which is well-defined even when raw utilities can be negative. The second plot uses the NBS/Lindahl fairness benchmark rows for the heterogeneous sweep, including the recomputed `N=2` rows.

![Elo dispersion vs Gini](plots_multiagent/heterogeneous_elo_dispersion_vs_gini.png)

N-specific Gini versions:

![Elo dispersion vs Gini N=2](plots_multiagent/heterogeneous_elo_dispersion_vs_gini_n2.png)

![Elo dispersion vs Gini N=4](plots_multiagent/heterogeneous_elo_dispersion_vs_gini_n4.png)

![Elo dispersion vs Gini N=6](plots_multiagent/heterogeneous_elo_dispersion_vs_gini_n6.png)

![Elo dispersion vs Gini N=8](plots_multiagent/heterogeneous_elo_dispersion_vs_gini_n8.png)

![Elo dispersion vs Gini N=10](plots_multiagent/heterogeneous_elo_dispersion_vs_gini_n10.png)

![Elo dispersion vs NBS/Lindahl fairness distance](plots_multiagent/heterogeneous_elo_dispersion_vs_fairness_distance.png)

Gini slope per 100 Elo standard deviation:

{markdown_table(dispersion_slopes, ['game_label', 'competition_band', 'slope_gini_per_100_elo_std', 'r_squared', 'n_runs'], max_rows=12)}

## 6. Homogeneous Control Order Dynamics

The homogeneous control runs hold model capability fixed, so order effects can be read more directly. Each small panel fixes game and `N`; the x-axis is agent order and the y-axis is that order slot's mean payoff.

![Homogeneous control order dynamics](plots_multiagent/homogeneous_control_order_dynamics.png)

## 7. General Dynamics as N Increases

These plots use comparable per-agent or normalized metrics. Mean payoff is per-agent; Gini is shifted for negative-utility games; consensus and final round are direct process metrics.

![Mean payoff vs N](plots_multiagent/general_mean_payoff_vs_n.png)

![Shifted Gini vs N](plots_multiagent/general_shifted_gini_vs_n.png)

![Consensus rate vs N](plots_multiagent/general_consensus_rate_vs_n.png)

![Rounds to consensus vs N](plots_multiagent/general_rounds_to_consensus_vs_n.png)

## 8. N=2 Baseline vs N=2 Random Pairings

This last plot compares the earlier controlled `GPT-5-nano` baseline setting against `N=2` heterogeneous random pairings. The slopes answer whether capability scaling looks different when the other side is always GPT-5-nano versus when the opponent is drawn from the heterogeneous pool.

![N=2 baseline vs heterogeneous pairings](plots_multiagent/n2_baseline_vs_heterogeneous_pairings.png)

{markdown_table(n2_compare_slopes, ['game_label', 'dataset', 'slope_per_100_elo', 'r_squared', 'n_models'])}

## Output Tables

The key generated tables are in `tables_multiagent/`:

- `hom_adversary_payoff_vs_elo_slopes.csv`
- `hom_adversary_payoff_vs_elo_by_competition_slopes.csv`
- `heterogeneous_payoff_vs_arena_elo_slopes.csv`
- `heterogeneous_payoff_vs_arena_elo_by_competition_slopes.csv`
- `hom_adversary_dilution_summary.csv`
- `heterogeneous_performance_elo_rankings.csv`
- `multiagent_fairness_by_n_summary.csv`
- `multiagent_n2_fairness_run_metrics.csv`
- `heterogeneous_elo_dispersion_vs_gini_slopes.csv`
- `heterogeneous_elo_dispersion_vs_gini_by_n_slopes.csv`
- `n2_baseline_vs_heterogeneous_pairing_slopes.csv`
"""

    REPORT_PATH.write_text(old_text.rstrip() + "\n" + textwrap.dedent(section).lstrip(), encoding="utf-8")


def main() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    ensure_output_dirs()

    hom_runs, hom_agents, het_runs, het_agents = load_and_prepare_tables()
    save_csv(TABLES_DIR / "homogeneous_runs_fresh.csv", hom_runs)
    save_csv(TABLES_DIR / "homogeneous_agents_fresh.csv", hom_agents)
    save_csv(TABLES_DIR / "heterogeneous_runs_fresh.csv", het_runs)
    save_csv(TABLES_DIR / "heterogeneous_agents_fresh.csv", het_agents)

    hom_slopes = plot_hom_adversary_payoff_vs_elo(hom_runs)
    hom_comp_slopes = plot_hom_adversary_payoff_by_competition(hom_runs)
    het_slopes = plot_heterogeneous_payoff_vs_arena_elo(het_agents)
    het_comp_slopes = plot_heterogeneous_payoff_by_competition(het_agents)
    dilution = plot_homogeneous_dilution(hom_runs)
    plot_homogeneous_control_order(hom_agents)
    general_summary = plot_general_n_dynamics(hom_runs, het_runs)

    fair_runs, fair_agents = load_fairness()
    fairness_summary = plot_multiagent_fairness(fair_runs, fair_agents, het_runs)
    dispersion_slopes = plot_heterogeneous_elo_dispersion(het_runs)

    rankings = build_performance_elos(het_agents)
    perf_corr = plot_performance_elos(rankings, het_agents)
    n2_compare_slopes = plot_n2_baseline_vs_heterogeneous(het_agents)

    create_report(
        hom_runs=hom_runs,
        hom_agents=hom_agents,
        het_runs=het_runs,
        het_agents=het_agents,
        hom_slopes=hom_slopes,
        hom_comp_slopes=hom_comp_slopes,
        het_slopes=het_slopes,
        het_comp_slopes=het_comp_slopes,
        dilution=dilution,
        fairness_summary=fairness_summary,
        dispersion_slopes=dispersion_slopes,
        perf_corr=perf_corr,
        n2_compare_slopes=n2_compare_slopes,
    )

    print(f"Wrote report: {REPORT_PATH}")
    print(f"Wrote plots: {PLOTS_DIR}")
    print(f"Wrote tables: {TABLES_DIR}")
    print(f"Homogeneous runs: {len(hom_runs)}; heterogeneous runs: {len(het_runs)}")
    print(f"General dynamics rows: {len(general_summary)}")


if __name__ == "__main__":
    main()
