"""Shared calculations and plots for the final TTC analyses.

This module preserves the calculations formerly shared through the two-, three-,
and five-seed commands. Use analyze_ttc_ten_seeds.py or
analyze_ttc_complete_seed_panels.py to run an analysis.
"""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent


FAMILY_ORDER = ["gpt-5", "claude-sonnet-4-6", "gemini-3-flash"]


FAMILY_LABELS = {
    "gpt-5": "GPT-5",
    "claude-sonnet-4-6": "Claude Sonnet 4.6",
    "gemini-3-flash": "Gemini 3 Flash",
}


GAME_ORDER = ["game1", "game2", "game3"]


SEED_COLORS = {
    42: "#64748b",
    984: "#2563eb",
    526: "#dc2626",
    423: "#059669",
    1024: "#9333ea",
    128: "#ea580c",
    256: "#0891b2",
    612: "#92400e",
    2048: "#db2777",
    4096: "#65a30d",
}


SEED_MARKERS = {
    42: "s",
    984: "o",
    526: "^",
    423: "D",
    1024: "P",
    128: "v",
    256: "X",
    612: "*",
    2048: "h",
    4096: "<",
}


def sem(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if len(values) <= 1:
        return 0.0
    return float(values.std(ddof=1) / math.sqrt(len(values)))


def ci95(mean: float, standard_error: float, n: int) -> tuple[float, float]:
    if n <= 1:
        return mean, mean
    try:
        from scipy.stats import t

        critical = float(t.ppf(0.975, n - 1))
    except Exception:
        critical = 1.96
    return mean - critical * standard_error, mean + critical * standard_error


def config_paths(run_root: Path) -> List[Path]:
    paths = sorted((run_root / "configs").glob("config_*.json"))
    if not paths:
        raise RuntimeError(f"No configs found under {run_root / 'configs'}")
    return paths


def collect_run_rows(run_root: Path, seed: int) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for config_path in config_paths(run_root):
        config = json.loads(config_path.read_text(encoding="utf-8"))
        output_dir = Path(config["output_dir"])
        if not output_dir.is_absolute():
            output_dir = PROJECT_ROOT / output_dir
        result_path = output_dir / "run_1_experiment_results.json"
        if not result_path.exists():
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        vote_integrity = result.get("vote_integrity") or (result.get("config") or {}).get(
            "vote_integrity"
        ) or {}
        target_position = int(config["target_position"])
        target_agent = f"Agent_{target_position + 1}"
        baseline_agent = "Agent_2" if target_agent == "Agent_1" else "Agent_1"
        utilities = result.get("final_utilities") or {}
        rows.append(
            {
                "seed": seed,
                "config_id": int(config["config_id"]),
                "family": config["target_model_family"],
                "provider": config["target_provider"],
                "level": config["target_reasoning_level_requested"],
                "level_index": int(config["target_reasoning_level_index"]),
                "game": config["game_label"],
                "game_cell": config["game_cell_id"],
                "order": config["order"],
                "target_agent": target_agent,
                "baseline_agent": baseline_agent,
                "target_utility": float(utilities.get(target_agent, 0.0)),
                "baseline_utility": float(utilities.get(baseline_agent, 0.0)),
                "utility_gap": float(utilities.get(target_agent, 0.0))
                - float(utilities.get(baseline_agent, 0.0)),
                "consensus": bool(result.get("consensus_reached")),
                "final_round": result.get("final_round"),
                "hard_failed": bool(vote_integrity.get("hard_failed")),
                "result_path": str(result_path),
            }
        )
    return pd.DataFrame(rows)


def validate_grid(df: pd.DataFrame, expected_seeds: Iterable[int], allow_incomplete: bool) -> None:
    expected_seeds = list(expected_seeds)
    expected_rows = 216 * len(expected_seeds)
    if not allow_incomplete and len(df) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} result rows, found {len(df)}")
    if df.empty:
        raise RuntimeError("No result rows available")
    duplicate = df.duplicated(["seed", "config_id"], keep=False)
    if duplicate.any():
        raise RuntimeError(
            "Duplicate seed/config results: "
            + str(df.loc[duplicate, ["seed", "config_id"]].to_dict("records"))
        )
    if df["hard_failed"].any():
        ids = df.loc[df["hard_failed"], ["seed", "config_id"]].to_dict("records")
        raise RuntimeError(f"Hard-failed result records are not analyzable: {ids}")
    if not allow_incomplete:
        counts = df.groupby(["seed", "family", "level_index"]).size()
        if set(counts.astype(int)) != {18} or len(counts) != 12 * len(expected_seeds):
            raise RuntimeError(f"Unexpected family-effort grid counts: {counts.to_dict()}")


def summarize(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Two order placements share one generated game instance. Collapse them
    # before estimating across-game uncertainty, matching the paper analysis.
    cell = (
        df.groupby(
            ["seed", "family", "provider", "level", "level_index", "game", "game_cell"],
            as_index=False,
        )
        .agg(
            order_count=("config_id", "count"),
            target_utility=("target_utility", "mean"),
            baseline_utility=("baseline_utility", "mean"),
            utility_gap=("utility_gap", "mean"),
            consensus_rate=("consensus", "mean"),
            mean_round=("final_round", "mean"),
        )
    )
    by_seed = (
        cell.groupby(["seed", "family", "provider", "level", "level_index"], as_index=False)
        .agg(
            game_cell_count=("game_cell", "count"),
            target_utility_mean=("target_utility", "mean"),
            target_utility_sem=("target_utility", sem),
            baseline_utility_mean=("baseline_utility", "mean"),
            baseline_utility_sem=("baseline_utility", sem),
            utility_gap_mean=("utility_gap", "mean"),
            utility_gap_sem=("utility_gap", sem),
            consensus_rate=("consensus_rate", "mean"),
            mean_round=("mean_round", "mean"),
        )
    )
    combined = (
        cell.groupby(["family", "provider", "level", "level_index"], as_index=False)
        .agg(
            seed_game_cell_count=("game_cell", "count"),
            target_utility_mean=("target_utility", "mean"),
            target_utility_sem=("target_utility", sem),
            baseline_utility_mean=("baseline_utility", "mean"),
            baseline_utility_sem=("baseline_utility", sem),
            utility_gap_mean=("utility_gap", "mean"),
            utility_gap_sem=("utility_gap", sem),
            consensus_rate=("consensus_rate", "mean"),
            mean_round=("mean_round", "mean"),
        )
    )
    return cell, by_seed, combined


def _set_pooled_ylim(axes: Iterable[Any], values: List[float]) -> None:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return
    lo, hi = min(finite), max(finite)
    pad = max((hi - lo) * 0.12, 2.0)
    for ax in axes:
        ax.set_ylim(max(0.0, lo - pad), hi + pad)


def plot_combined(
    combined: pd.DataFrame,
    output: Path,
    seed_label: str,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 3.8), sharey=True)
    values: List[float] = []
    colors = ["#64748b", "#2563eb", "#0f766e", "#f97316"]
    for ax, family in zip(axes, FAMILY_ORDER):
        subset = combined[combined["family"].eq(family)].sort_values("level_index")
        x = np.arange(len(subset))
        y = subset["target_utility_mean"].to_numpy(dtype=float)
        err = subset["target_utility_sem"].to_numpy(dtype=float)
        values.extend((y - err).tolist())
        values.extend((y + err).tolist())
        ax.bar(
            x,
            y,
            yerr=err,
            color=colors[: len(subset)],
            edgecolor="white",
            capsize=4,
            width=0.72,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(subset["level"].astype(str))
        ax.set_title(FAMILY_LABELS[family])
        ax.set_xlabel("Requested reasoning effort")
        ax.grid(axis="y", alpha=0.3)
        ax.text(
            0.02,
            0.04,
            f"n={int(subset['seed_game_cell_count'].min())} seed×game cells/point",
            transform=ax.transAxes,
            fontsize=8,
            color="#555555",
        )
    axes[0].set_ylabel(f"Mean target payoff ({seed_label})")
    _set_pooled_ylim(axes, values)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_target_and_baseline(
    summary: pd.DataFrame,
    output: Path,
    title_suffix: str,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 3.8), sharey=True)
    values: List[float] = []
    role_styles = {
        "target": {
            "mean": "target_utility_mean",
            "sem": "target_utility_sem",
            "label": "Target utility",
            "color": "#1f5aa6",
            "linestyle": "-",
            "marker": "o",
            "offset": -0.035,
        },
        "baseline": {
            "mean": "baseline_utility_mean",
            "sem": "baseline_utility_sem",
            "label": "Baseline utility",
            "color": "#c55a11",
            "linestyle": (0, (1.2, 2.0)),
            "marker": "s",
            "offset": 0.035,
        },
    }
    for ax, family in zip(axes, FAMILY_ORDER):
        subset = summary[summary["family"].eq(family)].sort_values("level_index")
        x = subset["level_index"].to_numpy(dtype=float)
        for style in role_styles.values():
            y = subset[style["mean"]].to_numpy(dtype=float)
            err = subset[style["sem"]].to_numpy(dtype=float)
            values.extend((y - err).tolist())
            values.extend((y + err).tolist())
            ax.errorbar(
                x + style["offset"],
                y,
                yerr=err,
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=2.0,
                capsize=3,
                color=style["color"],
                label=style["label"],
            )
        ax.set_xticks(x)
        ax.set_xticklabels(subset["level"].astype(str))
        ax.set_title(FAMILY_LABELS[family])
        ax.set_xlabel("Requested reasoning effort")
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Mean discounted utility")
    _set_pooled_ylim(axes, values)
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.suptitle(title_suffix, fontsize=12)
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=[0, 0.09, 1, 0.94])
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def saved_result_caps(run_root: Path) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for config_path in config_paths(run_root):
        config = json.loads(config_path.read_text(encoding="utf-8"))
        result_path = Path(config["output_dir"]) / "run_1_experiment_results.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        cap = str((result.get("config") or {}).get("max_tokens_per_phase"))
        counts[cap] = counts.get(cap, 0) + 1
    return counts


def endpoint_table(cell: pd.DataFrame, seeds: List[int]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for family in FAMILY_ORDER:
        family_data = cell[cell["family"].eq(family)]
        subsets: List[tuple[str, pd.DataFrame]] = [
            (str(seed), family_data[family_data["seed"].eq(seed)]) for seed in seeds
        ]
        subsets.append(("combined", family_data))
        for seed_label, subset in subsets:
            if subset.empty:
                continue
            low_index = int(subset["level_index"].min())
            high_index = int(subset["level_index"].max())
            paired = (
                subset[subset["level_index"].isin([low_index, high_index])]
                .pivot_table(
                    index=["seed", "game_cell"],
                    columns="level_index",
                    values=["target_utility", "utility_gap"],
                    aggfunc="mean",
                )
                .dropna()
            )
            target_delta = (
                paired[("target_utility", high_index)]
                - paired[("target_utility", low_index)]
            )
            gap_delta = paired[("utility_gap", high_index)] - paired[
                ("utility_gap", low_index)
            ]
            target_mean = float(target_delta.mean())
            target_sem = sem(target_delta)
            gap_mean = float(gap_delta.mean())
            gap_sem = sem(gap_delta)
            target_lo, target_hi = ci95(
                target_mean, target_sem, len(target_delta)
            )
            gap_lo, gap_hi = ci95(gap_mean, gap_sem, len(gap_delta))
            rows.append(
                {
                    "family": family,
                    "seed": seed_label,
                    "low_level_index": low_index,
                    "high_level_index": high_index,
                    "paired_seed_game_cells": len(target_delta),
                    "target_utility_endpoint_delta": target_mean,
                    "target_utility_endpoint_delta_sem": target_sem,
                    "target_utility_endpoint_delta_ci95_low": target_lo,
                    "target_utility_endpoint_delta_ci95_high": target_hi,
                    "utility_gap_endpoint_delta": gap_mean,
                    "utility_gap_endpoint_delta_sem": gap_sem,
                    "utility_gap_endpoint_delta_ci95_low": gap_lo,
                    "utility_gap_endpoint_delta_ci95_high": gap_hi,
                }
            )
    return pd.DataFrame(rows)


def correlation(left: np.ndarray, right: np.ndarray, kind: str) -> float:
    if len(left) < 2:
        return float("nan")
    try:
        from scipy.stats import pearsonr, spearmanr

        result = pearsonr(left, right) if kind == "pearson" else spearmanr(left, right)
        return float(result.statistic)
    except Exception:
        return float(pd.Series(left).corr(pd.Series(right), method=kind))


def agreement_summary(
    by_seed: pd.DataFrame,
    endpoints: pd.DataFrame,
    seeds: List[int],
) -> Dict[str, Any]:
    pairwise: Dict[str, Any] = {}
    for left_seed, right_seed in itertools.combinations(seeds, 2):
        left = by_seed[by_seed["seed"].eq(left_seed)].set_index(
            ["family", "level", "level_index"]
        )
        right = by_seed[by_seed["seed"].eq(right_seed)].set_index(
            ["family", "level", "level_index"]
        )
        joined = left[
            ["target_utility_mean", "utility_gap_mean"]
        ].join(
            right[["target_utility_mean", "utility_gap_mean"]],
            how="inner",
            lsuffix="_left",
            rsuffix="_right",
        )
        target_left = joined["target_utility_mean_left"].to_numpy(dtype=float)
        target_right = joined["target_utility_mean_right"].to_numpy(dtype=float)
        gap_left = joined["utility_gap_mean_left"].to_numpy(dtype=float)
        gap_right = joined["utility_gap_mean_right"].to_numpy(dtype=float)
        pairwise[f"{left_seed}_vs_{right_seed}"] = {
            "family_effort_cells": len(joined),
            "target_pearson_r": correlation(target_left, target_right, "pearson"),
            "target_spearman_rho": correlation(
                target_left, target_right, "spearman"
            ),
            "target_mae": float(np.mean(np.abs(target_left - target_right))),
            "utility_gap_pearson_r": correlation(gap_left, gap_right, "pearson"),
            "utility_gap_spearman_rho": correlation(
                gap_left, gap_right, "spearman"
            ),
            "utility_gap_mae": float(np.mean(np.abs(gap_left - gap_right))),
        }

    endpoint_rows = endpoints[endpoints["seed"].isin([str(seed) for seed in seeds])]
    endpoint_direction: Dict[str, Any] = {}
    for family, group in endpoint_rows.groupby("family"):
        deltas = {
            str(row["seed"]): float(row["target_utility_endpoint_delta"])
            for _, row in group.iterrows()
        }
        nonzero_signs = {int(np.sign(value)) for value in deltas.values() if value != 0}
        endpoint_direction[str(family)] = {
            "deltas": deltas,
            "all_same_direction": len(nonzero_signs) <= 1,
        }
    return {
        "pairwise_seed_agreement": pairwise,
        "endpoint_direction_by_family": endpoint_direction,
        "all_family_endpoint_directions_match": all(
            value["all_same_direction"] for value in endpoint_direction.values()
        ),
    }


def seed_level_ci_table(by_seed: pd.DataFrame, seeds: List[int]) -> pd.DataFrame:
    """Compute uncertainty across seed-level family-effort estimates."""

    expected = set(seeds)
    rows: List[Dict[str, Any]] = []
    for (family, provider, level, level_index), group in by_seed.groupby(
        ["family", "provider", "level", "level_index"], sort=False
    ):
        observed = set(group["seed"].astype(int))
        if observed != expected:
            raise RuntimeError(
                f"{family}/{level} has seeds {sorted(observed)}, "
                f"expected {sorted(expected)}"
            )
        row: Dict[str, Any] = {
            "family": family,
            "provider": provider,
            "level": level,
            "level_index": int(level_index),
            "seed_count": len(group),
        }
        metric_sources = {
            "target_utility": "target_utility_mean",
            "baseline_utility": "baseline_utility_mean",
            "utility_gap": "utility_gap_mean",
            "consensus_rate": "consensus_rate",
            "mean_round": "mean_round",
        }
        for metric, source_column in metric_sources.items():
            values = group[source_column].astype(float)
            mean = float(values.mean())
            standard_error = sem(values)
            lo, hi = ci95(mean, standard_error, len(values))
            row[f"{metric}_mean"] = mean
            row[f"{metric}_seed_sem"] = standard_error
            row[f"{metric}_seed_ci95_low"] = lo
            row[f"{metric}_seed_ci95_high"] = hi
        rows.append(row)
    result = pd.DataFrame(rows)
    expected_rows = len(FAMILY_ORDER) * 4
    if len(result) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} family-effort CI rows, found {len(result)}"
        )
    return result.sort_values(["family", "level_index"]).reset_index(drop=True)


def endpoint_across_seed_ci(
    endpoints: pd.DataFrame,
    seeds: List[int],
) -> pd.DataFrame:
    """Compute low-to-high endpoint confidence intervals across seeds."""

    per_seed = endpoints[endpoints["seed"].isin([str(seed) for seed in seeds])].copy()
    rows: List[Dict[str, Any]] = []
    for family in FAMILY_ORDER:
        group = per_seed[per_seed["family"].eq(family)].copy()
        observed = set(group["seed"].astype(int))
        if observed != set(seeds):
            raise RuntimeError(
                f"{family} endpoint rows have seeds {sorted(observed)}, "
                f"expected {sorted(seeds)}"
            )
        row: Dict[str, Any] = {
            "family": family,
            "seed_count": len(group),
            "positive_target_endpoint_seeds": int(
                (group["target_utility_endpoint_delta"] > 0).sum()
            ),
            "negative_target_endpoint_seeds": int(
                (group["target_utility_endpoint_delta"] < 0).sum()
            ),
        }
        for metric in ("target_utility", "utility_gap"):
            values = group[f"{metric}_endpoint_delta"].astype(float)
            mean = float(values.mean())
            standard_error = sem(values)
            lo, hi = ci95(mean, standard_error, len(values))
            row[f"{metric}_endpoint_delta_mean_across_seeds"] = mean
            row[f"{metric}_endpoint_delta_seed_sem"] = standard_error
            row[f"{metric}_endpoint_delta_seed_ci95_low"] = lo
            row[f"{metric}_endpoint_delta_seed_ci95_high"] = hi
        rows.append(row)
    return pd.DataFrame(rows)


def _set_seed_panel_ylim(axes: Iterable[Any], values: List[float]) -> None:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return
    lo, hi = min(finite), max(finite)
    pad = max((hi - lo) * 0.12, 2.0)
    for ax in axes:
        if lo >= 0:
            ax.set_ylim(max(0.0, lo - pad), hi + pad)
        else:
            ax.set_ylim(lo - pad, hi + pad)


def plot_individual_seeds(
    by_seed: pd.DataFrame,
    output: Path,
    metric: str,
    ylabel: str,
    seeds: List[int],
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.2), sharey=True)
    all_values: List[float] = []
    mean_col = f"{metric}_mean"
    sem_col = f"{metric}_sem"
    for ax, family in zip(axes, FAMILY_ORDER):
        family_data = by_seed[by_seed["family"].eq(family)]
        for seed in seeds:
            subset = family_data[family_data["seed"].eq(seed)].sort_values(
                "level_index"
            )
            x = subset["level_index"].to_numpy(dtype=float)
            y = subset[mean_col].to_numpy(dtype=float)
            err = subset[sem_col].to_numpy(dtype=float)
            all_values.extend((y - err).tolist())
            all_values.extend((y + err).tolist())
            ax.errorbar(
                x,
                y,
                yerr=err,
                marker=SEED_MARKERS[seed],
                linewidth=1.65,
                markersize=5,
                capsize=2.5,
                color=SEED_COLORS[seed],
                label=f"Seed {seed}",
            )
        labels = (
            family_data.sort_values("level_index")
            .drop_duplicates("level_index")
            .set_index("level_index")["level"]
        )
        ax.set_xticks(labels.index.astype(float))
        ax.set_xticklabels(labels.astype(str))
        ax.set_title(FAMILY_LABELS[family])
        ax.set_xlabel("Requested reasoning effort")
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel(ylabel)
    _set_seed_panel_ylim(axes, all_values)
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=min(5, len(seeds)),
        frameon=False,
    )
    bottom = 0.14 if len(seeds) > 5 else 0.10
    fig.tight_layout(rect=[0, bottom, 1, 1])
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_across_seed_ci(
    by_seed: pd.DataFrame,
    seed_ci: pd.DataFrame,
    output: Path,
    metric: str,
    ylabel: str,
    seeds: List[int],
) -> None:
    """Show individual seed curves and the across-seed 95% confidence interval."""

    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.2), sharey=True)
    all_values: List[float] = []
    for ax, family in zip(axes, FAMILY_ORDER):
        family_data = by_seed[by_seed["family"].eq(family)]
        for seed in seeds:
            subset = family_data[family_data["seed"].eq(seed)].sort_values(
                "level_index"
            )
            ax.plot(
                subset["level_index"],
                subset[f"{metric}_mean"],
                marker=SEED_MARKERS[seed],
                markersize=3.5,
                linewidth=1.0,
                alpha=0.32,
                color=SEED_COLORS[seed],
                label=f"Seed {seed}",
            )
        summary = seed_ci[seed_ci["family"].eq(family)].sort_values("level_index")
        x = summary["level_index"].to_numpy(dtype=float)
        mean = summary[f"{metric}_mean"].to_numpy(dtype=float)
        lo = summary[f"{metric}_seed_ci95_low"].to_numpy(dtype=float)
        hi = summary[f"{metric}_seed_ci95_high"].to_numpy(dtype=float)
        yerr = np.vstack([mean - lo, hi - mean])
        all_values.extend(lo.tolist())
        all_values.extend(hi.tolist())
        ax.errorbar(
            x,
            mean,
            yerr=yerr,
            marker="o",
            markersize=6.5,
            linewidth=2.6,
            capsize=4,
            color="#111827",
            label="Across-seed mean ± 95% CI",
            zorder=10,
        )
        labels = summary.set_index("level_index")["level"]
        ax.set_xticks(labels.index.astype(float))
        ax.set_xticklabels(labels.astype(str))
        ax.set_title(FAMILY_LABELS[family])
        ax.set_xlabel("Requested reasoning effort")
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel(ylabel)
    _set_seed_panel_ylim(axes, all_values)
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=min(6, len(seeds) + 1),
        frameon=False,
    )
    fig.suptitle(
        f"Across-seed uncertainty: Student-t 95% CI, n={len(seeds)} seeds",
        y=1.01,
    )
    fig.tight_layout(rect=[0, 0.14, 1, 1])
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_endpoint_ci(
    endpoints: pd.DataFrame,
    endpoint_seed_ci: pd.DataFrame,
    output: Path,
    seeds: List[int],
) -> None:
    per_seed = endpoints[endpoints["seed"].isin([str(seed) for seed in seeds])]
    fig_width = max(14.4, 1.5 * (len(seeds) + 1))
    fig, axes = plt.subplots(1, 3, figsize=(fig_width, 4.2), sharey=True)
    for ax, family in zip(axes, FAMILY_ORDER):
        family_data = per_seed[per_seed["family"].eq(family)].set_index("seed")
        values = [
            float(
                family_data.loc[str(seed), "target_utility_endpoint_delta"]
            )
            for seed in seeds
        ]
        for index, (seed, value) in enumerate(zip(seeds, values)):
            ax.scatter(
                index,
                value,
                s=55,
                marker=SEED_MARKERS[seed],
                color=SEED_COLORS[seed],
                zorder=4,
            )
        summary = endpoint_seed_ci[endpoint_seed_ci["family"].eq(family)].iloc[0]
        mean = float(summary["target_utility_endpoint_delta_mean_across_seeds"])
        lo = float(summary["target_utility_endpoint_delta_seed_ci95_low"])
        hi = float(summary["target_utility_endpoint_delta_seed_ci95_high"])
        ax.errorbar(
            len(seeds),
            mean,
            yerr=np.array([[mean - lo], [hi - mean]]),
            marker="D",
            markersize=7,
            color="#111827",
            capsize=5,
            linewidth=2.2,
            zorder=5,
        )
        ax.axhline(0.0, color="#6b7280", linestyle="--", linewidth=1.2)
        ax.set_xticks(range(len(seeds) + 1))
        ax.set_xticklabels([str(seed) for seed in seeds] + ["mean"])
        ax.set_title(FAMILY_LABELS[family])
        ax.set_xlabel("Random seed")
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Highest − lowest effort target payoff")
    fig.suptitle(
        f"Endpoint change across {len(seeds)} seeds "
        "(mean ± Student-t 95% CI)"
    )
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_game_stratified(
    cell: pd.DataFrame,
    output: Path,
    seeds: List[int],
) -> None:
    summary = (
        cell.groupby(
            ["seed", "family", "level", "level_index", "game"], as_index=False
        )
        .agg(
            target_utility_mean=("target_utility", "mean"),
            target_utility_sem=("target_utility", sem),
        )
    )
    fig, axes = plt.subplots(3, 3, figsize=(13.4, 10.4), sharex=False, sharey="row")
    for row_index, game in enumerate(GAME_ORDER):
        for col_index, family in enumerate(FAMILY_ORDER):
            ax = axes[row_index, col_index]
            subset = summary[
                summary["game"].eq(game) & summary["family"].eq(family)
            ]
            for seed in seeds:
                seed_data = subset[subset["seed"].eq(seed)].sort_values(
                    "level_index"
                )
                ax.errorbar(
                    seed_data["level_index"],
                    seed_data["target_utility_mean"],
                    yerr=seed_data["target_utility_sem"],
                    marker=SEED_MARKERS[seed],
                    markersize=3.5,
                    linewidth=1.2,
                    capsize=2,
                    color=SEED_COLORS[seed],
                    label=f"Seed {seed}",
                )
            labels = (
                subset.sort_values("level_index")
                .drop_duplicates("level_index")
                .set_index("level_index")["level"]
            )
            ax.set_xticks(labels.index.astype(float))
            ax.set_xticklabels(labels.astype(str), fontsize=8)
            ax.grid(axis="y", alpha=0.28)
            if row_index == 0:
                ax.set_title(FAMILY_LABELS[family])
            if col_index == 0:
                ax.set_ylabel(f"{game.capitalize()}\nMean target payoff")
            if row_index == 2:
                ax.set_xlabel("Effort")
    handles, labels = axes[0, -1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=min(5, len(seeds)),
        frameon=False,
    )
    bottom = 0.10 if len(seeds) > 5 else 0.06
    fig.tight_layout(rect=[0, bottom, 1, 1])
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_report(
    path: Path,
    rows: pd.DataFrame,
    endpoints: pd.DataFrame,
    endpoint_seed_ci: pd.DataFrame,
    agreement: Dict[str, Any],
    seeds: List[int],
) -> None:
    lines = [
        f"# TTC {len(seeds)}-seed replication comparison",
        "",
        "## Completion",
        "",
        f"- Healthy analyzed runs: {len(rows)} "
        f"({'; '.join(f'{sum(rows.seed.eq(seed))} at seed {seed}' for seed in seeds)}).",
        f"- No-consensus outcomes: {sum(~rows['consensus'])}. These are legitimate "
        "completed outcomes, not infrastructure failures.",
        "",
        "## Confidence-interval method",
        "",
        f"- The headline uncertainty is across the {len(seeds)} independent seed-level "
        f"estimates, using a two-sided Student-t 95% confidence interval (df={len(seeds) - 1}).",
        "- Each seed-level family-effort estimate first averages the nine matched "
        "game cells; each game cell averages the two agent orders.",
        f"- This avoids treating all {len(rows):,} runs as independent for the reviewer’s "
        "seed-sensitivity question.",
        "",
        "## Lowest-to-highest effort endpoint changes",
        "",
        "| Family | Seed | Target payoff Δ | Utility-gap Δ |",
        "|---|---:|---:|---:|",
    ]
    per_seed = endpoints[endpoints["seed"].isin([str(seed) for seed in seeds])].copy()
    order = {str(seed): index for index, seed in enumerate(seeds)}
    per_seed["seed_order"] = per_seed["seed"].map(order)
    for _, row in per_seed.sort_values(["family", "seed_order"]).iterrows():
        lines.append(
            f"| {FAMILY_LABELS.get(row['family'], row['family'])} | "
            f"{row['seed']} | {row['target_utility_endpoint_delta']:.3f} | "
            f"{row['utility_gap_endpoint_delta']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Across-seed endpoint confidence intervals",
            "",
            "| Family | Mean target Δ | 95% CI across seeds | Positive seeds | "
            "Mean utility-gap Δ | 95% CI across seeds |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in endpoint_seed_ci.iterrows():
        lines.append(
            f"| {FAMILY_LABELS.get(row['family'], row['family'])} | "
            f"{row['target_utility_endpoint_delta_mean_across_seeds']:.3f} | "
            f"[{row['target_utility_endpoint_delta_seed_ci95_low']:.3f}, "
            f"{row['target_utility_endpoint_delta_seed_ci95_high']:.3f}] | "
            f"{int(row['positive_target_endpoint_seeds'])}/{len(seeds)} | "
            f"{row['utility_gap_endpoint_delta_mean_across_seeds']:.3f} | "
            f"[{row['utility_gap_endpoint_delta_seed_ci95_low']:.3f}, "
            f"{row['utility_gap_endpoint_delta_seed_ci95_high']:.3f}] |"
        )
    lines.extend(["", "## Pairwise seed agreement", ""])
    for name, metrics in agreement["pairwise_seed_agreement"].items():
        lines.append(
            f"- {name.replace('_', ' ')}: target-payoff Pearson "
            f"r={metrics['target_pearson_r']:.3f}, Spearman "
            f"ρ={metrics['target_spearman_rho']:.3f}, "
            f"MAE={metrics['target_mae']:.3f}; utility-gap Pearson "
            f"r={metrics['utility_gap_pearson_r']:.3f}."
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Use the across-seed confidence intervals as the primary answer to seed "
            "robustness. The seed×game-cell pooled intervals remain available in the "
            "CSV outputs as a secondary description of across-game heterogeneity.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def recovery_ids(run_root: Path) -> List[int]:
    return sorted(
        {
            int(path.name.split("_")[1])
            for path in (run_root / "recovery" / "configs").glob("config_*.json")
        }
    )
