#!/usr/bin/env python3
"""Analyze the complete TTC replication across seeds 42, 984, and 526."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts import analyze_ttc_seed_replication as base


DEFAULT_SEED42_ROOT = (
    PROJECT_ROOT / "experiments" / "results" / "ttc_native_scaling_20260502_212943"
)
DEFAULT_SEED984_ROOT = (
    PROJECT_ROOT
    / "experiments"
    / "results"
    / "ttc_native_scaling_seed984_20260725_025700"
)
SEEDS = [42, 984, 526]
SEED_COLORS = {42: "#64748b", 984: "#2563eb", 526: "#dc2626"}
SEED_MARKERS = {42: "s", 984: "o", 526: "^"}


def endpoint_table(cell: pd.DataFrame, seeds: List[int]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for family in base.FAMILY_ORDER:
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
            target_sem = base.sem(target_delta)
            gap_mean = float(gap_delta.mean())
            gap_sem = base.sem(gap_delta)
            target_lo, target_hi = base.ci95(
                target_mean, target_sem, len(target_delta)
            )
            gap_lo, gap_hi = base.ci95(gap_mean, gap_sem, len(gap_delta))
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


def set_shared_ylim(axes: Iterable[Any], values: List[float]) -> None:
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


def plot_seed_comparison(
    by_seed: pd.DataFrame,
    output: Path,
    metric: str,
    ylabel: str,
    seeds: List[int],
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 3.8), sharey=True)
    all_values: List[float] = []
    mean_col = f"{metric}_mean"
    sem_col = f"{metric}_sem"
    for ax, family in zip(axes, base.FAMILY_ORDER):
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
                linewidth=1.9,
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
        ax.set_title(base.FAMILY_LABELS[family])
        ax.set_xlabel("Requested reasoning effort")
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel(ylabel)
    set_shared_ylim(axes, all_values)
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=[0, 0.09, 1, 1])
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
            target_utility_sem=("target_utility", base.sem),
        )
    )
    fig, axes = plt.subplots(3, 3, figsize=(13.0, 10.2), sharex=False, sharey="row")
    for row_index, game in enumerate(base.GAME_ORDER):
        for col_index, family in enumerate(base.FAMILY_ORDER):
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
                    linewidth=1.6,
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
                ax.set_title(base.FAMILY_LABELS[family])
            if col_index == 0:
                ax.set_ylabel(f"{game.capitalize()}\nMean target payoff")
            if row_index == 2:
                ax.set_xlabel("Effort")
    handles, labels = axes[0, -1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_report(
    path: Path,
    rows: pd.DataFrame,
    endpoints: pd.DataFrame,
    agreement: Dict[str, Any],
    seeds: List[int],
) -> None:
    lines = [
        "# TTC three-seed replication comparison",
        "",
        "## Completion",
        "",
        f"- Healthy analyzed runs: {len(rows)} "
        f"({'; '.join(f'{sum(rows.seed.eq(seed))} at seed {seed}' for seed in seeds)}).",
        f"- No-consensus outcomes: {sum(~rows['consensus'])}. These are legitimate "
        "completed outcomes, not infrastructure failures.",
        "",
        "## Lowest-to-highest effort endpoint changes",
        "",
        "| Family | Seed | Target payoff Δ | 95% CI | Utility-gap Δ | 95% CI |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    seed_order = [str(seed) for seed in seeds] + ["combined"]
    ordered = endpoints.assign(
        seed_order=endpoints["seed"].map(
            {seed: index for index, seed in enumerate(seed_order)}
        )
    ).sort_values(["family", "seed_order"])
    for _, row in ordered.iterrows():
        lines.append(
            f"| {base.FAMILY_LABELS.get(row['family'], row['family'])} | "
            f"{row['seed']} | {row['target_utility_endpoint_delta']:.3f} | "
            f"[{row['target_utility_endpoint_delta_ci95_low']:.3f}, "
            f"{row['target_utility_endpoint_delta_ci95_high']:.3f}] | "
            f"{row['utility_gap_endpoint_delta']:.3f} | "
            f"[{row['utility_gap_endpoint_delta_ci95_low']:.3f}, "
            f"{row['utility_gap_endpoint_delta_ci95_high']:.3f}] |"
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
            "Assess corroboration from the family-specific curves, endpoint directions, "
            "and uncertainty. Three random seeds improve the robustness check but do not "
            "remove provider-alias drift or the substantial across-game heterogeneity.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def recovery_ids(run_root: Path) -> List[int]:
    return sorted(
        {
            int(path.name.split("_")[1])
            for path in (run_root / "recovery" / "configs").glob("config_*.json")
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seed526_root", type=Path)
    parser.add_argument("--seed42-root", type=Path, default=DEFAULT_SEED42_ROOT)
    parser.add_argument("--seed984-root", type=Path, default=DEFAULT_SEED984_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    roots = {
        42: args.seed42_root.resolve(),
        984: args.seed984_root.resolve(),
        526: args.seed526_root.resolve(),
    }
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else roots[526] / "analysis" / "seeds42_984_526"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    data = {
        seed: base.collect_run_rows(root, seed) for seed, root in roots.items()
    }
    rows = pd.concat([data[seed] for seed in SEEDS], ignore_index=True)
    base.validate_grid(rows, SEEDS, allow_incomplete=False)
    cell, by_seed, combined = base.summarize(rows)
    endpoints = endpoint_table(cell, SEEDS)
    agreement = agreement_summary(by_seed, endpoints, SEEDS)

    rows.to_csv(output_dir / "run_level_results_all_three.csv", index=False)
    cell.to_csv(output_dir / "game_cell_seed_summary_all_three.csv", index=False)
    by_seed.to_csv(output_dir / "family_effort_by_seed_all_three.csv", index=False)
    combined.to_csv(output_dir / "family_effort_combined_all_three.csv", index=False)
    endpoints.to_csv(output_dir / "endpoint_changes_all_three.csv", index=False)
    (output_dir / "seed_agreement_all_three.json").write_text(
        json.dumps(agreement, indent=2) + "\n", encoding="utf-8"
    )

    plot_seed_comparison(
        by_seed,
        output_dir / "target_payoff_three_seed_comparison.png",
        "target_utility",
        "Mean target payoff",
        SEEDS,
    )
    plot_seed_comparison(
        by_seed,
        output_dir / "utility_gap_three_seed_comparison.png",
        "utility_gap",
        "Mean target - baseline payoff",
        SEEDS,
    )
    base.plot_combined(
        combined,
        output_dir / "target_payoff_all_three_seeds_combined.png",
        "seeds 42 + 984 + 526",
    )
    base.plot_target_and_baseline(
        by_seed[by_seed["seed"].eq(526)],
        output_dir / "seed526_target_and_baseline_by_effort.png",
        "Seed 526 (216 runs)",
    )
    base.plot_target_and_baseline(
        combined,
        output_dir / "combined_target_and_baseline_all_three.png",
        "Seeds 42 + 984 + 526 (648 runs)",
    )
    plot_game_stratified(
        cell,
        output_dir / "target_payoff_by_game_three_seeds.png",
        SEEDS,
    )
    write_report(
        output_dir / "comparison_report_all_three.md",
        rows,
        endpoints,
        agreement,
        SEEDS,
    )

    audits: Dict[str, Any] = {}
    for seed in SEEDS:
        seed_df = data[seed]
        audits[str(seed)] = {
            "results_root": str(roots[seed]),
            "healthy_results": int(seed_df["config_id"].nunique()),
            "hard_failed_results": int(seed_df["hard_failed"].sum()),
            "no_consensus_results": int((~seed_df["consensus"]).sum()),
            "family_counts": {
                str(key): int(value)
                for key, value in seed_df.groupby("family").size().to_dict().items()
            },
            "game_counts": {
                str(key): int(value)
                for key, value in seed_df.groupby("game").size().to_dict().items()
            },
            "order_counts": {
                str(key): int(value)
                for key, value in seed_df.groupby("order").size().to_dict().items()
            },
            "saved_result_cap_counts": base.saved_result_caps(roots[seed]),
            "cap_recovery_config_ids": recovery_ids(roots[seed]),
        }
    final_audit = {
        "expected_seeds": SEEDS,
        "expected_results_per_seed": 216,
        "total_healthy_results": int(rows["config_id"].count()),
        "missing_seed_config_pairs": [],
        "by_seed": audits,
    }
    (output_dir / "final_audit_all_three.json").write_text(
        json.dumps(final_audit, indent=2) + "\n", encoding="utf-8"
    )
    artifact_hashes = {
        path.name: sha256(path)
        for path in sorted(output_dir.iterdir())
        if path.is_file() and path.name != "analysis_provenance_all_three.json"
    }
    (output_dir / "analysis_provenance_all_three.json").write_text(
        json.dumps(
            {
                "seed_roots": {str(seed): str(root) for seed, root in roots.items()},
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
