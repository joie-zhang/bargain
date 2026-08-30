#!/usr/bin/env python3
"""Analyze the complete TTC replication across five random seeds."""

from __future__ import annotations

import argparse
import hashlib
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
from scripts import analyze_ttc_three_seeds as three


DEFAULT_ROOTS = {
    42: (
        PROJECT_ROOT
        / "experiments"
        / "results"
        / "ttc_native_scaling_20260502_212943"
    ),
    984: (
        PROJECT_ROOT
        / "experiments"
        / "results"
        / "ttc_native_scaling_seed984_20260725_025700"
    ),
    526: (
        PROJECT_ROOT
        / "experiments"
        / "results"
        / "ttc_native_scaling_seed526_20260725_181400"
    ),
}
SEEDS = [42, 984, 526, 423, 1024]
SEED_COLORS = {
    42: "#64748b",
    984: "#2563eb",
    526: "#dc2626",
    423: "#059669",
    1024: "#9333ea",
}
SEED_MARKERS = {42: "s", 984: "o", 526: "^", 423: "D", 1024: "P"}


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
            sem = base.sem(values)
            lo, hi = base.ci95(mean, sem, len(values))
            row[f"{metric}_mean"] = mean
            row[f"{metric}_seed_sem"] = sem
            row[f"{metric}_seed_ci95_low"] = lo
            row[f"{metric}_seed_ci95_high"] = hi
        rows.append(row)
    result = pd.DataFrame(rows)
    expected_rows = len(base.FAMILY_ORDER) * 4
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
    for family in base.FAMILY_ORDER:
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
            sem = base.sem(values)
            lo, hi = base.ci95(mean, sem, len(values))
            row[f"{metric}_endpoint_delta_mean_across_seeds"] = mean
            row[f"{metric}_endpoint_delta_seed_sem"] = sem
            row[f"{metric}_endpoint_delta_seed_ci95_low"] = lo
            row[f"{metric}_endpoint_delta_seed_ci95_high"] = hi
        rows.append(row)
    return pd.DataFrame(rows)


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
        ax.set_title(base.FAMILY_LABELS[family])
        ax.set_xlabel("Requested reasoning effort")
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel(ylabel)
    set_shared_ylim(axes, all_values)
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
    for ax, family in zip(axes, base.FAMILY_ORDER):
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
        ax.set_title(base.FAMILY_LABELS[family])
        ax.set_xlabel("Requested reasoning effort")
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel(ylabel)
    set_shared_ylim(axes, all_values)
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
    for ax, family in zip(axes, base.FAMILY_ORDER):
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
        ax.set_title(base.FAMILY_LABELS[family])
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
            target_utility_sem=("target_utility", base.sem),
        )
    )
    fig, axes = plt.subplots(3, 3, figsize=(13.4, 10.4), sharex=False, sharey="row")
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
                ax.set_title(base.FAMILY_LABELS[family])
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
        "estimates, using a two-sided Student-t 95% confidence interval (df=4).",
        "- Each seed-level family-effort estimate first averages the nine matched "
        "game cells; each game cell averages the two agent orders.",
        "- This avoids treating all 1,080 runs as independent for the reviewer’s "
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
            f"| {base.FAMILY_LABELS.get(row['family'], row['family'])} | "
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
            f"| {base.FAMILY_LABELS.get(row['family'], row['family'])} | "
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
    parser.add_argument("seed423_root", type=Path)
    parser.add_argument("seed1024_root", type=Path)
    parser.add_argument("--seed42-root", type=Path, default=DEFAULT_ROOTS[42])
    parser.add_argument("--seed984-root", type=Path, default=DEFAULT_ROOTS[984])
    parser.add_argument("--seed526-root", type=Path, default=DEFAULT_ROOTS[526])
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    roots = {
        42: args.seed42_root.resolve(),
        984: args.seed984_root.resolve(),
        526: args.seed526_root.resolve(),
        423: args.seed423_root.resolve(),
        1024: args.seed1024_root.resolve(),
    }
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else roots[1024] / "analysis" / "seeds42_984_526_423_1024"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    data = {
        seed: base.collect_run_rows(root, seed) for seed, root in roots.items()
    }
    rows = pd.concat([data[seed] for seed in SEEDS], ignore_index=True)
    base.validate_grid(rows, SEEDS, allow_incomplete=False)
    cell, by_seed, combined = base.summarize(rows)
    endpoints = three.endpoint_table(cell, SEEDS)
    agreement = three.agreement_summary(by_seed, endpoints, SEEDS)
    seed_ci = seed_level_ci_table(by_seed, SEEDS)
    endpoint_seed_ci = endpoint_across_seed_ci(endpoints, SEEDS)

    rows.to_csv(output_dir / "run_level_results_all_five.csv", index=False)
    cell.to_csv(output_dir / "game_cell_seed_summary_all_five.csv", index=False)
    by_seed.to_csv(output_dir / "family_effort_by_seed_all_five.csv", index=False)
    combined.to_csv(output_dir / "family_effort_pooled_all_five.csv", index=False)
    seed_ci.to_csv(
        output_dir / "family_effort_across_seed_ci95_all_five.csv", index=False
    )
    endpoints.to_csv(
        output_dir / "endpoint_changes_by_seed_and_pooled_all_five.csv", index=False
    )
    endpoint_seed_ci.to_csv(
        output_dir / "endpoint_changes_across_seed_ci95_all_five.csv", index=False
    )
    (output_dir / "seed_agreement_all_five.json").write_text(
        json.dumps(agreement, indent=2) + "\n", encoding="utf-8"
    )

    plot_individual_seeds(
        by_seed,
        output_dir / "target_payoff_five_seed_comparison.png",
        "target_utility",
        "Mean target payoff",
        SEEDS,
    )
    plot_individual_seeds(
        by_seed,
        output_dir / "utility_gap_five_seed_comparison.png",
        "utility_gap",
        "Mean target − baseline payoff",
        SEEDS,
    )
    plot_across_seed_ci(
        by_seed,
        seed_ci,
        output_dir / "target_payoff_across_seed_mean_ci95.png",
        "target_utility",
        "Mean target payoff",
        SEEDS,
    )
    plot_across_seed_ci(
        by_seed,
        seed_ci,
        output_dir / "utility_gap_across_seed_mean_ci95.png",
        "utility_gap",
        "Mean target − baseline payoff",
        SEEDS,
    )
    plot_endpoint_ci(
        endpoints,
        endpoint_seed_ci,
        output_dir / "target_payoff_endpoint_delta_across_five_seeds.png",
        SEEDS,
    )
    base.plot_combined(
        combined,
        output_dir / "target_payoff_all_five_seeds_pooled.png",
        "seeds 42 + 984 + 526 + 423 + 1024",
    )
    for seed in (423, 1024):
        base.plot_target_and_baseline(
            by_seed[by_seed["seed"].eq(seed)],
            output_dir / f"seed{seed}_target_and_baseline_by_effort.png",
            f"Seed {seed} (216 runs)",
        )
    base.plot_target_and_baseline(
        combined,
        output_dir / "combined_target_and_baseline_all_five.png",
        "Seeds 42 + 984 + 526 + 423 + 1024 (1,080 runs)",
    )
    plot_game_stratified(
        cell,
        output_dir / "target_payoff_by_game_five_seeds.png",
        SEEDS,
    )
    write_report(
        output_dir / "comparison_report_all_five.md",
        rows,
        endpoints,
        endpoint_seed_ci,
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
        "confidence_interval_unit": "seed-level estimate",
        "confidence_interval_seed_count": len(SEEDS),
        "confidence_interval_method": "two-sided Student-t 95% CI, df=4",
        "by_seed": audits,
    }
    (output_dir / "final_audit_all_five.json").write_text(
        json.dumps(final_audit, indent=2) + "\n", encoding="utf-8"
    )
    artifact_hashes = {
        path.name: sha256(path)
        for path in sorted(output_dir.iterdir())
        if path.is_file() and path.name != "analysis_provenance_all_five.json"
    }
    (output_dir / "analysis_provenance_all_five.json").write_text(
        json.dumps(
            {
                "seed_roots": {str(seed): str(root) for seed, root in roots.items()},
                "analysis_script": str(Path(__file__).resolve()),
                "primary_confidence_interval": {
                    "unit": "seed-level estimate",
                    "seed_count": len(SEEDS),
                    "method": "two-sided Student-t 95% CI",
                    "degrees_of_freedom": len(SEEDS) - 1,
                },
                "artifacts_sha256": artifact_hashes,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "endpoint_across_seed_ci": endpoint_seed_ci.to_dict("records"),
                **agreement,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
