#!/usr/bin/env python3
"""Compare the archived TTC seed-42 batch with a completed seed replication."""

from __future__ import annotations

import argparse
import hashlib
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
DEFAULT_OLD_ROOT = (
    PROJECT_ROOT / "experiments" / "results" / "ttc_native_scaling_20260502_212943"
)
FAMILY_ORDER = ["gpt-5", "claude-sonnet-4-6", "gemini-3-flash"]
FAMILY_LABELS = {
    "gpt-5": "GPT-5",
    "claude-sonnet-4-6": "Claude Sonnet 4.6",
    "gemini-3-flash": "Gemini 3 Flash",
}
SEED_COLORS = {42: "#64748b", 984: "#2563eb"}
GAME_ORDER = ["game1", "game2", "game3"]


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


def endpoint_table(cell: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for family in FAMILY_ORDER:
        family_data = cell[cell["family"].eq(family)]
        for seed_label, subset in [
            ("42", family_data[family_data["seed"].eq(42)]),
            ("984", family_data[family_data["seed"].eq(984)]),
            ("combined", family_data),
        ]:
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
            target_lo, target_hi = ci95(target_mean, target_sem, len(target_delta))
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


def seed_agreement(by_seed: pd.DataFrame, endpoints: pd.DataFrame) -> Dict[str, Any]:
    wide = by_seed.pivot_table(
        index=["family", "level", "level_index"],
        columns="seed",
        values=["target_utility_mean", "utility_gap_mean"],
    ).dropna()
    target_old = wide[("target_utility_mean", 42)].to_numpy(dtype=float)
    target_new = wide[("target_utility_mean", 984)].to_numpy(dtype=float)
    gap_old = wide[("utility_gap_mean", 42)].to_numpy(dtype=float)
    gap_new = wide[("utility_gap_mean", 984)].to_numpy(dtype=float)

    def correlation(left: np.ndarray, right: np.ndarray, kind: str) -> float:
        if len(left) < 2:
            return float("nan")
        try:
            from scipy.stats import pearsonr, spearmanr

            result = pearsonr(left, right) if kind == "pearson" else spearmanr(left, right)
            return float(result.statistic)
        except Exception:
            return float(pd.Series(left).corr(pd.Series(right), method=kind))

    endpoint_seed = endpoints[endpoints["seed"].isin(["42", "984"])].pivot(
        index="family",
        columns="seed",
        values="target_utility_endpoint_delta",
    )
    family_endpoint = {}
    for family, row in endpoint_seed.iterrows():
        old = float(row["42"])
        new = float(row["984"])
        family_endpoint[family] = {
            "seed42_delta": old,
            "seed984_delta": new,
            "same_direction": bool(np.sign(old) == np.sign(new)),
        }
    return {
        "family_effort_cells_compared": len(wide),
        "target_mean_pearson_r": correlation(target_old, target_new, "pearson"),
        "target_mean_spearman_rho": correlation(target_old, target_new, "spearman"),
        "target_mean_mae": float(np.mean(np.abs(target_old - target_new))),
        "utility_gap_pearson_r": correlation(gap_old, gap_new, "pearson"),
        "utility_gap_spearman_rho": correlation(gap_old, gap_new, "spearman"),
        "utility_gap_mae": float(np.mean(np.abs(gap_old - gap_new))),
        "endpoint_direction_by_family": family_endpoint,
        "all_family_endpoint_directions_match": all(
            item["same_direction"] for item in family_endpoint.values()
        ),
    }


def set_shared_ylim(axes: Iterable[Any], values: List[float]) -> None:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return
    lo, hi = min(finite), max(finite)
    pad = max((hi - lo) * 0.12, 2.0)
    for ax in axes:
        ax.set_ylim(max(0.0, lo - pad), hi + pad)


def plot_seed_comparison(
    by_seed: pd.DataFrame,
    output: Path,
    metric: str,
    ylabel: str,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 3.8), sharey=True)
    all_values: List[float] = []
    mean_col = f"{metric}_mean"
    sem_col = f"{metric}_sem"
    for ax, family in zip(axes, FAMILY_ORDER):
        family_data = by_seed[by_seed["family"].eq(family)]
        for seed in [42, 984]:
            subset = family_data[family_data["seed"].eq(seed)].sort_values("level_index")
            if subset.empty:
                continue
            x = subset["level_index"].to_numpy(dtype=float)
            y = subset[mean_col].to_numpy(dtype=float)
            err = subset[sem_col].to_numpy(dtype=float)
            all_values.extend((y - err).tolist())
            all_values.extend((y + err).tolist())
            ax.errorbar(
                x,
                y,
                yerr=err,
                marker="o" if seed == 984 else "s",
                linewidth=2.0,
                capsize=3,
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
    set_shared_ylim(axes, all_values)
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=[0, 0.09, 1, 1])
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_combined(
    combined: pd.DataFrame,
    output: Path,
    seed_label: str = "seeds 42 + 984",
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
    set_shared_ylim(axes, values)
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
    set_shared_ylim(axes, values)
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.suptitle(title_suffix, fontsize=12)
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=[0, 0.09, 1, 0.94])
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_game_stratified(cell: pd.DataFrame, output: Path) -> None:
    summary = (
        cell.groupby(["seed", "family", "level", "level_index", "game"], as_index=False)
        .agg(
            target_utility_mean=("target_utility", "mean"),
            target_utility_sem=("target_utility", sem),
        )
    )
    fig, axes = plt.subplots(3, 3, figsize=(13.0, 10.2), sharex=False, sharey="row")
    for row_index, game in enumerate(GAME_ORDER):
        for col_index, family in enumerate(FAMILY_ORDER):
            ax = axes[row_index, col_index]
            subset = summary[
                summary["game"].eq(game) & summary["family"].eq(family)
            ]
            for seed in [42, 984]:
                seed_data = subset[subset["seed"].eq(seed)].sort_values("level_index")
                if seed_data.empty:
                    continue
                ax.errorbar(
                    seed_data["level_index"],
                    seed_data["target_utility_mean"],
                    yerr=seed_data["target_utility_sem"],
                    marker="o" if seed == 984 else "s",
                    linewidth=1.8,
                    capsize=2.5,
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
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_markdown(
    path: Path,
    rows: pd.DataFrame,
    endpoints: pd.DataFrame,
    agreement: Dict[str, Any],
) -> None:
    lines = [
        "# TTC seed-replication comparison",
        "",
        "## Bottom line",
        "",
        "Seed 984 corroborates the qualitative target-payoff increase from the "
        "lowest to highest effort for GPT-5 and Claude Sonnet 4.6, and the new "
        "endpoint increases are larger than at seed 42. Gemini was essentially flat "
        "at seed 42 (-0.054) but positive at seed 984 (+3.949), so Gemini is better "
        "described as strengthening the directional pattern than as exactly "
        "replicating its original curve.",
        "",
        "This is directional rather than statistically decisive corroboration: every "
        "family's endpoint-change 95% interval still includes zero. The low correlation "
        "between the 12 absolute family×effort means also shows meaningful seed-level "
        "variation in the exact curves. Thus the second seed supports the paper's "
        "qualitative target-payoff trend, but it does not establish a precise, "
        "seed-invariant scaling law from these two seeds alone.",
        "",
        "The target-minus-baseline gap is a different estimand. Its endpoint change is "
        "negative for GPT-5 at both seeds and positive for Claude and Gemini at both "
        "seeds, with wide intervals throughout.",
        "",
        "## Completion and agreement",
        "",
        f"- Healthy analyzed runs: {len(rows)} ({sum(rows['seed'].eq(42))} seed 42; "
        f"{sum(rows['seed'].eq(984))} seed 984).",
        f"- No-consensus outcomes: {sum(~rows['consensus'])}. These are legitimate "
        "completed outcomes, not infrastructure failures.",
        f"- Across the 12 family×effort means, target-payoff Pearson correlation: "
        f"{agreement['target_mean_pearson_r']:.3f}; Spearman correlation: "
        f"{agreement['target_mean_spearman_rho']:.3f}; MAE: "
        f"{agreement['target_mean_mae']:.3f}.",
        f"- Utility-gap Pearson correlation: {agreement['utility_gap_pearson_r']:.3f}; "
        f"Spearman correlation: {agreement['utility_gap_spearman_rho']:.3f}; MAE: "
        f"{agreement['utility_gap_mae']:.3f}.",
        "",
        "## Lowest-to-highest effort endpoint changes",
        "",
        "| Family | Seed | Target payoff Δ | 95% CI | Utility-gap Δ | 95% CI |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in endpoints.iterrows():
        lines.append(
            f"| {FAMILY_LABELS.get(row['family'], row['family'])} | {row['seed']} | "
            f"{row['target_utility_endpoint_delta']:.3f} | "
            f"[{row['target_utility_endpoint_delta_ci95_low']:.3f}, "
            f"{row['target_utility_endpoint_delta_ci95_high']:.3f}] | "
            f"{row['utility_gap_endpoint_delta']:.3f} | "
            f"[{row['utility_gap_endpoint_delta_ci95_low']:.3f}, "
            f"{row['utility_gap_endpoint_delta_ci95_high']:.3f}] |"
        )
    lines.extend(
        [
            "",
            "## Interpretation rule",
            "",
            "Corroboration should be judged from the full family-specific curves, "
            "endpoint direction, and uncertainty—not from one pooled scalar. Provider-side "
            "alias drift is a residual temporal confound for Claude Sonnet 4.6, Gemini 3 "
            "Flash Preview, and the gpt-5-nano baseline; GPT-5 is version-pinned.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def saved_result_caps(run_root: Path) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for config_path in config_paths(run_root):
        config = json.loads(config_path.read_text(encoding="utf-8"))
        result_path = Path(config["output_dir"]) / "run_1_experiment_results.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        cap = str((result.get("config") or {}).get("max_tokens_per_phase"))
        counts[cap] = counts.get(cap, 0) + 1
    return counts


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("new_root", type=Path)
    parser.add_argument("--old-root", type=Path, default=DEFAULT_OLD_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    old_root = args.old_root.resolve()
    new_root = args.new_root.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else new_root / "analysis" / "seed42_vs_seed984"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    old = collect_run_rows(old_root, 42)
    new = collect_run_rows(new_root, 984)
    rows = pd.concat([old, new], ignore_index=True)
    validate_grid(rows, [42, 984], args.allow_incomplete)
    cell, by_seed, combined = summarize(rows)
    endpoints = endpoint_table(cell)
    agreement = seed_agreement(by_seed, endpoints)

    rows.to_csv(output_dir / "run_level_results.csv", index=False)
    cell.to_csv(output_dir / "game_cell_seed_summary.csv", index=False)
    by_seed.to_csv(output_dir / "family_effort_by_seed.csv", index=False)
    combined.to_csv(output_dir / "family_effort_combined.csv", index=False)
    endpoints.to_csv(output_dir / "endpoint_changes.csv", index=False)
    (output_dir / "seed_agreement.json").write_text(
        json.dumps(agreement, indent=2) + "\n", encoding="utf-8"
    )

    plot_seed_comparison(
        by_seed,
        output_dir / "target_payoff_seed42_vs_seed984.png",
        "target_utility",
        "Mean target payoff",
    )
    plot_seed_comparison(
        by_seed,
        output_dir / "utility_gap_seed42_vs_seed984.png",
        "utility_gap",
        "Mean target - baseline payoff",
    )
    plot_combined(combined, output_dir / "target_payoff_combined_seeds.png")
    plot_target_and_baseline(
        by_seed[by_seed["seed"].eq(984)],
        output_dir / "seed984_target_and_baseline_by_effort.png",
        "Seed 984 (216 runs)",
    )
    plot_target_and_baseline(
        combined,
        output_dir / "combined_target_and_baseline_by_effort.png",
        "Seeds 42 + 984 (432 runs)",
    )
    plot_game_stratified(cell, output_dir / "target_payoff_by_game_and_seed.png")
    write_markdown(
        output_dir / "comparison_report.md",
        rows,
        endpoints,
        agreement,
    )
    recovery_ids = sorted(
        {
            int(path.name.split("_")[1])
            for path in (new_root / "recovery" / "configs").glob("config_*.json")
        }
    )
    final_audit = {
        "old_results_root": str(old_root),
        "new_results_root": str(new_root),
        "old_seed": 42,
        "new_seed": 984,
        "old_healthy_results": int(old["config_id"].nunique()),
        "new_healthy_results": int(new["config_id"].nunique()),
        "old_hard_failed_results": int(old["hard_failed"].sum()),
        "new_hard_failed_results": int(new["hard_failed"].sum()),
        "old_no_consensus_results": int((~old["consensus"]).sum()),
        "new_no_consensus_results": int((~new["consensus"]).sum()),
        "new_family_counts": {
            str(key): int(value)
            for key, value in new.groupby("family").size().to_dict().items()
        },
        "new_game_counts": {
            str(key): int(value)
            for key, value in new.groupby("game").size().to_dict().items()
        },
        "new_order_counts": {
            str(key): int(value)
            for key, value in new.groupby("order").size().to_dict().items()
        },
        "new_saved_result_cap_counts": saved_result_caps(new_root),
        "cap_recovery_config_ids": recovery_ids,
        "cap_recovery_count": len(recovery_ids),
        "missing_new_config_ids": sorted(
            set(range(1, 217)) - set(new["config_id"].astype(int))
        ),
    }
    (output_dir / "final_audit.json").write_text(
        json.dumps(final_audit, indent=2) + "\n", encoding="utf-8"
    )
    artifact_hashes = {
        path.name: sha256(path)
        for path in sorted(output_dir.iterdir())
        if path.is_file() and path.name != "analysis_provenance.json"
    }
    (output_dir / "analysis_provenance.json").write_text(
        json.dumps(
            {
                "old_results_root": str(old_root),
                "new_results_root": str(new_root),
                "analysis_script": str(Path(__file__).resolve()),
                "artifacts_sha256": artifact_hashes,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(output_dir), **agreement}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
