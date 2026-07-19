#!/usr/bin/env python3
"""Analyze heterogeneous vs homogeneous payoff inequality with Elo controls."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import full_games123_multiagent_batch as full  # noqa: E402
import random_monoculture_control_batch as rmc  # noqa: E402
from full_games123_multiagent_batch import elo_for_model  # noqa: E402
from paper_figures.plot_random_monoculture_gini_vs_heterogeneous import shifted_gini  # noqa: E402


DEFAULT_HOM_ROOT = (
    PROJECT_ROOT
    / "experiments/results/full_games123_random_monoculture_control_20260628_014357"
)
DEFAULT_HETERO_RUNS = (
    PROJECT_ROOT
    / "experiments/results/n2_plus_multiagent_comparison_analysis_20260505"
    / "tables_multiagent/heterogeneous_runs_fresh.csv"
)

GROUP_LABELS = {
    "heterogeneous": "Heterogeneous",
    "homogeneous": "Homogeneous",
}
GROUP_COLORS = {
    "heterogeneous": "#D54E6A",
    "homogeneous": "#4E79A7",
}
GAME_ORDER = ["game1", "game2", "game3"]
GAME_LABELS = {"game1": "Game 1", "game2": "Game 2", "game3": "Game 3"}
METRICS = {
    "payoff_gini": "Corrected payoff Gini",
    "payoff_variance": "Payoff variance",
}
ELO_BINS = [1240, 1300, 1350, 1400, 1450, 1505]
ELO_SPREAD_BINS = [-0.001, 25, 50, 75, 100, 140]
ELO_BUCKET_LABELS = ["1240-1300", "1300-1350", "1350-1400", "1400-1450", "1450-1505"]
ELO_SPREAD_BUCKET_LABELS = ["0-25", "25-50", "50-75", "75-100", "100-140"]


def sem(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 2:
        return 0.0
    return float(clean.std(ddof=1) / math.sqrt(len(clean)))


def summarize(
    frame: pd.DataFrame,
    group_cols: list[str],
    metric_cols: tuple[str, ...] = ("payoff_gini", "payoff_variance"),
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, sub in frame.groupby(group_cols, observed=False, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        base = dict(zip(group_cols, key, strict=True))
        base["n_runs"] = int(len(sub))
        for metric in metric_cols:
            values = pd.to_numeric(sub[metric], errors="coerce")
            base[f"{metric}_mean"] = float(values.mean())
            base[f"{metric}_sem"] = sem(values)
        rows.append(base)
    return pd.DataFrame(rows)


def model_list_to_elos(model_list: str) -> list[int]:
    elos = []
    for model in str(model_list).split("+"):
        elo = elo_for_model(model)
        if elo is None:
            raise ValueError(f"Missing Elo for heterogeneous model {model!r}")
        elos.append(int(elo))
    return elos


def competition_id(config: dict[str, Any]) -> str:
    if "competition_id" in config:
        return str(config["competition_id"])
    if config.get("game_label") == "game1":
        return f"comp_{float(config.get('competition_level', 0.0)):.2f}"
    if config.get("game_label") == "game2":
        return f"rho_{float(config.get('rho', 0.0)):.2f}_theta_{float(config.get('theta', 0.0)):.2f}"
    if config.get("game_label") == "game3":
        return f"sigma_{float(config.get('sigma', 0.0)):.2f}_alpha_{float(config.get('alpha', 0.0)):.2f}"
    return "unknown"


def config_id_string(config: dict[str, Any]) -> str:
    return f"config_{rmc.config_number(config['config_id']):04d}"


def load_homogeneous(results_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for config in full.load_configs(results_root):
        result_path = full.result_path_for(config)
        if (
            result_path is None
            or not result_path.exists()
            or full.validate_result_file(rmc.runtime_config(config), result_path) is not None
        ):
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        utilities = np.asarray(
            [float(v) for v in (result.get("final_utilities") or {}).values()],
            dtype=float,
        )
        if utilities.size == 0:
            continue
        _, corrected_gini, shifted = shifted_gini(utilities)
        model_elo = float(config.get("model_elo", np.nan))
        rows.append(
            {
                "run_key": config_id_string(config),
                "group": "homogeneous",
                "game_label": str(config["game_label"]),
                "n_agents": int(config["n_agents"]),
                "competition_id": competition_id(config),
                "model_list": "+".join([str(config["monoculture_model"])] * int(config["n_agents"])),
                "mean_elo": model_elo,
                "elo_std": 0.0,
                "elo_range": 0.0,
                "payoff_gini": corrected_gini,
                "payoff_variance": float(np.var(utilities, ddof=0)),
                "payoff_mean": float(np.mean(utilities)),
                "payoff_std": float(np.std(utilities, ddof=0)),
                "payoff_range": float(np.max(utilities) - np.min(utilities)),
                "gini_shifted_for_negative": bool(shifted),
                "final_round": int(result.get("final_round") or 0),
                "consensus_reached": bool(result.get("consensus_reached")),
            }
        )
    return pd.DataFrame(rows)


def load_heterogeneous(path: Path) -> pd.DataFrame:
    runs = pd.read_csv(path)
    runs = runs[runs["experiment_family"].eq("heterogeneous_random")].copy()
    rows: list[dict[str, Any]] = []
    for row in runs.itertuples(index=False):
        elos = model_list_to_elos(str(row.model_list))
        rows.append(
            {
                "run_key": str(row.run_key),
                "group": "heterogeneous",
                "game_label": str(row.game_label),
                "n_agents": int(row.n_agents),
                "competition_id": str(row.competition_key),
                "model_list": str(row.model_list),
                "mean_elo": float(np.mean(elos)),
                "elo_std": float(np.std(elos, ddof=0)),
                "elo_range": float(max(elos) - min(elos)),
                "payoff_gini": float(row.utility_gini_corrected),
                "payoff_variance": float(row.utility_variance),
                "payoff_mean": float(row.mean_utility),
                "payoff_std": float(row.utility_std),
                "payoff_range": float(row.utility_range),
                "gini_shifted_for_negative": bool(getattr(row, "utility_gini_shifted", False)),
                "final_round": int(row.final_round),
                "consensus_reached": bool(row.consensus_reached),
            }
        )
    return pd.DataFrame(rows)


def add_bins(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["mean_elo_bucket"] = pd.cut(
        out["mean_elo"],
        bins=ELO_BINS,
        labels=ELO_BUCKET_LABELS,
        include_lowest=True,
        right=True,
    ).astype(str)
    out["elo_spread_bucket"] = pd.cut(
        out["elo_std"],
        bins=ELO_SPREAD_BINS,
        labels=ELO_SPREAD_BUCKET_LABELS,
        include_lowest=True,
        right=True,
    ).astype(str)
    return out


def label_counts(ax: plt.Axes, bars: Any, rows: pd.DataFrame, metric: str, y_offset: float) -> None:
    for bar, row in zip(bars, rows.itertuples(), strict=True):
        mean_value = float(getattr(row, f"{metric}_mean"))
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + y_offset,
            f"{mean_value:.3g}\nn={int(row.n_runs)}",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def plot_overall(summary: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.4))
    order = ["heterogeneous", "homogeneous"]
    rows = summary.set_index("group").loc[order].reset_index()
    for ax, metric in zip(axes, METRICS, strict=True):
        means = rows[f"{metric}_mean"].to_numpy(dtype=float)
        sems = rows[f"{metric}_sem"].to_numpy(dtype=float)
        bars = ax.bar(
            np.arange(len(rows)),
            means,
            yerr=sems,
            capsize=4,
            color=[GROUP_COLORS[group] for group in rows["group"]],
            edgecolor="#333333",
            linewidth=0.7,
            alpha=0.9,
        )
        ax.set_xticks(np.arange(len(rows)))
        ax.set_xticklabels([GROUP_LABELS[group] for group in rows["group"]], rotation=20, ha="right")
        ax.set_ylabel(METRICS[metric])
        ax.set_title(METRICS[metric])
        ax.grid(axis="y", alpha=0.25)
        y_offset = max(float(np.nanmax(means + sems)) * 0.025, 0.002)
        label_counts(ax, bars, rows, metric, y_offset)
        ax.set_ylim(0, float(np.nanmax(means + sems)) * 1.2 + y_offset)
    fig.suptitle("Overall payoff inequality: heterogeneous vs homogeneous")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_game_bars(summary: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.7))
    width = 0.36
    x = np.arange(len(GAME_ORDER))
    for ax, metric in zip(axes, METRICS, strict=True):
        for offset, group in [(-width / 2, "heterogeneous"), (width / 2, "homogeneous")]:
            rows = (
                summary[summary["group"].eq(group)]
                .set_index("game_label")
                .loc[GAME_ORDER]
                .reset_index()
            )
            means = rows[f"{metric}_mean"].to_numpy(dtype=float)
            sems = rows[f"{metric}_sem"].to_numpy(dtype=float)
            ax.bar(
                x + offset,
                means,
                width,
                yerr=sems,
                capsize=3,
                label=GROUP_LABELS[group],
                color=GROUP_COLORS[group],
                edgecolor="#333333",
                linewidth=0.6,
                alpha=0.9,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([GAME_LABELS[game] for game in GAME_ORDER])
        ax.set_ylabel(METRICS[metric])
        ax.set_title(f"By game: {METRICS[metric]}")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_bucket_bars(summary: pd.DataFrame, out_path: Path) -> None:
    bucket_order = ELO_BUCKET_LABELS
    fig, axes = plt.subplots(2, 1, figsize=(12.5, 8.0), sharex=True)
    width = 0.36
    x = np.arange(len(bucket_order))
    for ax, metric in zip(axes, METRICS, strict=True):
        for offset, group in [(-width / 2, "heterogeneous"), (width / 2, "homogeneous")]:
            rows = (
                summary[summary["group"].eq(group)]
                .set_index("mean_elo_bucket")
                .reindex(bucket_order)
                .reset_index()
            )
            means = rows[f"{metric}_mean"].to_numpy(dtype=float)
            sems = rows[f"{metric}_sem"].fillna(0.0).to_numpy(dtype=float)
            labels = [f"n={int(n)}" if pd.notna(n) else "" for n in rows["n_runs"]]
            bars = ax.bar(
                x + offset,
                means,
                width,
                yerr=sems,
                capsize=3,
                label=GROUP_LABELS[group],
                color=GROUP_COLORS[group],
                edgecolor="#333333",
                linewidth=0.6,
                alpha=0.9,
            )
            y_offset = max(float(np.nanmax(means + sems)) * 0.015, 0.001)
            for bar, label in zip(bars, labels, strict=True):
                if not label or not np.isfinite(bar.get_height()):
                    continue
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + y_offset,
                    label,
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    rotation=0,
                )
        ax.set_ylabel(METRICS[metric])
        ax.set_title(f"Mean-Elo bucket comparison: {METRICS[metric]}")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(bucket_order, rotation=25, ha="right")
    axes[-1].set_xlabel("Mean model Elo bucket")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_scatter(frame: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    for ax, metric in zip(axes, METRICS, strict=True):
        for group in ["heterogeneous", "homogeneous"]:
            sub = frame[frame["group"].eq(group)]
            ax.scatter(
                sub["mean_elo"],
                sub[metric],
                s=18 if group == "homogeneous" else 12,
                alpha=0.45 if group == "homogeneous" else 0.16,
                color=GROUP_COLORS[group],
                label=GROUP_LABELS[group],
                edgecolors="none",
            )
            if len(sub) >= 2:
                x = sub["mean_elo"].to_numpy(dtype=float)
                y = sub[metric].to_numpy(dtype=float)
                mask = np.isfinite(x) & np.isfinite(y)
                if mask.sum() >= 2:
                    coef = np.polyfit(x[mask], y[mask], deg=1)
                    xs = np.linspace(float(x[mask].min()), float(x[mask].max()), 100)
                    ax.plot(xs, np.polyval(coef, xs), color=GROUP_COLORS[group], linewidth=2.0)
        ax.set_xlabel("Mean model Elo")
        ax.set_ylabel(METRICS[metric])
        ax.set_title(f"{METRICS[metric]} vs mean Elo")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_hetero_spread(summary: pd.DataFrame, out_path: Path) -> None:
    rows = summary.set_index("elo_spread_bucket").reindex(ELO_SPREAD_BUCKET_LABELS).reset_index()
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 7.5), sharex=True)
    x = np.arange(len(rows))
    for ax, metric in zip(axes, METRICS, strict=True):
        means = rows[f"{metric}_mean"].to_numpy(dtype=float)
        sems = rows[f"{metric}_sem"].to_numpy(dtype=float)
        bars = ax.bar(
            x,
            means,
            yerr=sems,
            capsize=3,
            color="#8B6BB1",
            edgecolor="#333333",
            linewidth=0.6,
            alpha=0.9,
        )
        y_offset = max(float(np.nanmax(means + sems)) * 0.015, 0.001)
        for bar, row in zip(bars, rows.itertuples(), strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + y_offset,
                f"n={int(row.n_runs)}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        ax.set_ylabel(METRICS[metric])
        ax.set_title(f"Heterogeneous only: {METRICS[metric]} by within-group Elo spread")
        ax.grid(axis="y", alpha=0.25)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(rows["elo_spread_bucket"], rotation=25, ha="right")
    axes[-1].set_xlabel("Within-run Elo standard deviation bucket")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_bucket_deltas(bucket_summary: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    wide_parts = []
    for metric in METRICS:
        values = bucket_summary.pivot(index="mean_elo_bucket", columns="group", values=f"{metric}_mean")
        counts = bucket_summary.pivot(index="mean_elo_bucket", columns="group", values="n_runs")
        values[f"{metric}_hetero_minus_hom"] = values["heterogeneous"] - values["homogeneous"]
        values[f"{metric}_heterogeneous_n"] = counts["heterogeneous"]
        values[f"{metric}_homogeneous_n"] = counts["homogeneous"]
        wide_parts.append(values[[f"{metric}_hetero_minus_hom", f"{metric}_heterogeneous_n", f"{metric}_homogeneous_n"]])
    deltas = pd.concat(wide_parts, axis=1).reset_index()
    deltas = deltas.set_index("mean_elo_bucket").reindex(ELO_BUCKET_LABELS).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5), sharex=True)
    x = np.arange(len(deltas))
    for ax, metric in zip(axes, METRICS, strict=True):
        vals = deltas[f"{metric}_hetero_minus_hom"].to_numpy(dtype=float)
        colors = ["#D54E6A" if value >= 0 else "#4E79A7" for value in vals]
        ax.bar(x, vals, color=colors, edgecolor="#333333", linewidth=0.6, alpha=0.9)
        ax.axhline(0, color="#333333", linewidth=0.8)
        ax.set_title(f"Bucket delta: hetero minus hom {METRICS[metric]}")
        ax.set_ylabel("Heterogeneous - homogeneous")
        ax.grid(axis="y", alpha=0.25)
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(deltas["mean_elo_bucket"], rotation=25, ha="right")
        ax.set_xlabel("Mean model Elo bucket")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return deltas


def format_table(df: pd.DataFrame, cols: list[str]) -> str:
    table = df[cols].copy()
    for col in table.columns:
        if pd.api.types.is_float_dtype(table[col]):
            table[col] = table[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.4f}")
    return table.to_markdown(index=False)


def write_report(
    out_path: Path,
    plots: dict[str, Path],
    tables: dict[str, pd.DataFrame],
    hetero_path: Path,
    hom_root: Path,
    combined: pd.DataFrame,
) -> None:
    overall = tables["overall"].copy()
    overall["label"] = overall["group"].map(GROUP_LABELS)
    overall_cols = [
        "label",
        "n_runs",
        "payoff_gini_mean",
        "payoff_gini_sem",
        "payoff_variance_mean",
        "payoff_variance_sem",
    ]
    bucket_delta = tables["bucket_deltas"].copy()
    spread = tables["hetero_spread"].copy()

    lines = [
        "# Heterogeneous vs Homogeneous Inequality With Elo Checks",
        "",
        f"- Generated: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- Heterogeneous source: `{hetero_path}`",
        f"- Homogeneous source: `{hom_root}`",
        f"- Runs loaded: heterogeneous `{int((combined['group'] == 'heterogeneous').sum())}`, homogeneous `{int((combined['group'] == 'homogeneous').sum())}`.",
        "- Metrics: shifted, small-N-corrected within-run payoff Gini; population variance of final payoffs within each run.",
        "- Matching view: runs are bucketed by mean model Elo using fixed overlapping bins `1240-1300`, `1300-1350`, `1350-1400`, `1400-1450`, and `1450-1505`.",
        "",
        "## Interpretation Notes",
        "",
        "- The overall bars answer the direct question, but they still mix games, N, competition settings, and different Elo compositions.",
        "- The mean-Elo bucket plots are the cleaner sanity check for whether the conclusion survives among groups with similar average model strength.",
        "- Homogeneous runs have within-run Elo spread zero by construction; the heterogeneous spread plot asks whether more diverse heterogeneous rosters have more or less payoff inequality.",
        "- Payoff variance is scale-sensitive across games, so the game-specific and Elo-bucket views are more informative than the pooled variance alone.",
        "",
        "## Overall Comparison",
        "",
        f"![Overall Gini and variance]({plots['overall'].relative_to(out_path.parent).as_posix()})",
        "",
        format_table(overall, overall_cols),
        "",
        "## By Game",
        "",
        f"![Game-level Gini and variance]({plots['game'].relative_to(out_path.parent).as_posix()})",
        "",
        "## Mean-Elo Bucket Comparison",
        "",
        f"![Mean Elo bucket comparison]({plots['buckets'].relative_to(out_path.parent).as_posix()})",
        "",
        "## Mean-Elo Scatter",
        "",
        f"![Metric scatter by mean Elo]({plots['scatter'].relative_to(out_path.parent).as_posix()})",
        "",
        "## Bucket Deltas",
        "",
        "Positive values mean heterogeneous is more unequal/higher-variance than homogeneous inside the same mean-Elo bucket.",
        "",
        f"![Mean Elo bucket deltas]({plots['deltas'].relative_to(out_path.parent).as_posix()})",
        "",
        format_table(
            bucket_delta,
            [
                "mean_elo_bucket",
                "payoff_gini_hetero_minus_hom",
                "payoff_gini_heterogeneous_n",
                "payoff_gini_homogeneous_n",
                "payoff_variance_hetero_minus_hom",
                "payoff_variance_heterogeneous_n",
                "payoff_variance_homogeneous_n",
            ],
        ),
        "",
        "## Heterogeneous Elo Diversity",
        "",
        f"![Heterogeneous Elo spread]({plots['spread'].relative_to(out_path.parent).as_posix()})",
        "",
        format_table(
            spread,
            [
                "elo_spread_bucket",
                "n_runs",
                "payoff_gini_mean",
                "payoff_gini_sem",
                "payoff_variance_mean",
                "payoff_variance_sem",
            ],
        ),
        "",
        "## Files",
        "",
        "- `combined_run_metrics.csv`: run-level metrics with group, game, mean Elo, Elo spread, Gini, and variance.",
        "- `overall_summary.csv`: overall group means and SEMs.",
        "- `game_summary.csv`: group-by-game means and SEMs.",
        "- `mean_elo_bucket_summary.csv`: group-by-mean-Elo-bucket means and SEMs.",
        "- `mean_elo_bucket_deltas.csv`: heterogeneous-minus-homogeneous deltas within mean-Elo buckets.",
        "- `heterogeneous_elo_spread_summary.csv`: heterogeneous-only inequality by within-roster Elo spread bucket.",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hom-results-root", type=Path, default=DEFAULT_HOM_ROOT)
    parser.add_argument("--heterogeneous-runs", type=Path, default=DEFAULT_HETERO_RUNS)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    hom_root = args.hom_results_root.resolve()
    hetero_path = args.heterogeneous_runs.resolve()
    if args.output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = hom_root / "analysis" / f"inequality_elo_matched_{stamp}"
    else:
        out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    hetero = load_heterogeneous(hetero_path)
    hom = load_homogeneous(hom_root)
    if hetero.empty:
        raise SystemExit(f"No heterogeneous rows loaded from {hetero_path}")
    if hom.empty:
        raise SystemExit(f"No homogeneous rows loaded from {hom_root}")
    combined = add_bins(pd.concat([hetero, hom], ignore_index=True))

    overall = summarize(combined, ["group"]).sort_values("group")
    game_summary = summarize(combined, ["game_label", "group"]).sort_values(["game_label", "group"])
    bucket_summary = summarize(combined, ["mean_elo_bucket", "group"]).sort_values(["mean_elo_bucket", "group"])
    hetero_spread = summarize(
        combined[combined["group"].eq("heterogeneous")],
        ["elo_spread_bucket"],
    )
    hetero_spread = (
        hetero_spread
        .set_index("elo_spread_bucket")
        .reindex(ELO_SPREAD_BUCKET_LABELS)
        .reset_index()
    )

    plots = {
        "overall": out_dir / "01_overall_gini_variance_bars.png",
        "game": out_dir / "02_by_game_gini_variance_bars.png",
        "buckets": out_dir / "03_mean_elo_bucket_gini_variance_bars.png",
        "scatter": out_dir / "04_mean_elo_scatter_trends.png",
        "spread": out_dir / "05_heterogeneous_elo_spread_bars.png",
        "deltas": out_dir / "06_mean_elo_bucket_deltas.png",
    }
    plot_overall(overall, plots["overall"])
    plot_game_bars(game_summary, plots["game"])
    plot_bucket_bars(bucket_summary, plots["buckets"])
    plot_scatter(combined, plots["scatter"])
    plot_hetero_spread(hetero_spread, plots["spread"])
    bucket_deltas = plot_bucket_deltas(bucket_summary, plots["deltas"])

    combined.to_csv(out_dir / "combined_run_metrics.csv", index=False)
    overall.to_csv(out_dir / "overall_summary.csv", index=False)
    game_summary.to_csv(out_dir / "game_summary.csv", index=False)
    bucket_summary.to_csv(out_dir / "mean_elo_bucket_summary.csv", index=False)
    bucket_deltas.to_csv(out_dir / "mean_elo_bucket_deltas.csv", index=False)
    hetero_spread.to_csv(out_dir / "heterogeneous_elo_spread_summary.csv", index=False)

    report_path = out_dir / "heterogeneous_vs_homogeneous_inequality_elo_report.md"
    write_report(
        report_path,
        plots,
        {
            "overall": overall,
            "game": game_summary,
            "bucket": bucket_summary,
            "bucket_deltas": bucket_deltas,
            "hetero_spread": hetero_spread,
        },
        hetero_path,
        hom_root,
        combined,
    )
    print(f"Wrote report: {report_path}")
    for path in plots.values():
        print(f"Wrote plot: {path}")
    print(f"Wrote run metrics: {out_dir / 'combined_run_metrics.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
