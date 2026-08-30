#!/usr/bin/env python3
"""Plot the currently completed portion of the ten-seed TTC replication."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts import analyze_ttc_five_seeds as shared
from scripts import analyze_ttc_seed_replication as base
from scripts import analyze_ttc_ten_seeds as ten


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for seed in ten.NEW_SEEDS:
        parser.add_argument(f"seed{seed}_root", type=Path)
    for seed, root in ten.HISTORICAL_ROOTS.items():
        parser.add_argument(f"--seed{seed}-root", type=Path, default=root)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def available_seed_ci(by_seed: pd.DataFrame) -> pd.DataFrame:
    """Compute Student-t intervals from each available seed-level estimate."""

    rows: List[Dict[str, Any]] = []
    keys = ["family", "provider", "level", "level_index"]
    for key, group in by_seed.groupby(keys, sort=False):
        family, provider, level, level_index = key
        row: Dict[str, Any] = {
            "family": family,
            "provider": provider,
            "level": level,
            "level_index": int(level_index),
            "seed_count": int(group["seed"].nunique()),
            "seeds": ",".join(str(value) for value in sorted(group["seed"].astype(int))),
            "game_cell_count_min": int(group["game_cell_count"].min()),
            "game_cell_count_max": int(group["game_cell_count"].max()),
        }
        for metric, column in {
            "target_utility": "target_utility_mean",
            "baseline_utility": "baseline_utility_mean",
            "utility_gap": "utility_gap_mean",
        }.items():
            values = group[column].astype(float)
            mean = float(values.mean())
            standard_error = base.sem(values)
            low, high = base.ci95(mean, standard_error, len(values))
            row[f"{metric}_mean"] = mean
            row[f"{metric}_seed_sem"] = standard_error
            row[f"{metric}_seed_ci95_low"] = low
            row[f"{metric}_seed_ci95_high"] = high
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["family", "level_index"]).reset_index(
        drop=True
    )


def partial_title(run_count: int) -> str:
    return (
        f"Partial 10-seed snapshot: {run_count:,}/2,160 completed runs; "
        "Claude max incomplete"
    )


def plot_seed_curves(
    by_seed: pd.DataFrame,
    output: Path,
    metric: str,
    ylabel: str,
    run_count: int,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.6), sharey=True)
    all_values: List[float] = []
    mean_col = f"{metric}_mean"
    sem_col = f"{metric}_sem"
    for ax, family in zip(axes, base.FAMILY_ORDER):
        family_data = by_seed[by_seed["family"].eq(family)]
        for seed in ten.SEEDS:
            subset = family_data[family_data["seed"].eq(seed)].sort_values(
                "level_index"
            )
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
                marker=ten.SEED_MARKERS[seed],
                linewidth=1.35,
                markersize=4.5,
                capsize=2,
                color=ten.SEED_COLORS[seed],
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
    shared.set_shared_ylim(axes, all_values)
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False)
    fig.suptitle(partial_title(run_count), fontsize=12)
    fig.tight_layout(rect=[0, 0.14, 1, 0.94])
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_available_seed_ci(
    by_seed: pd.DataFrame,
    ci: pd.DataFrame,
    output: Path,
    metric: str,
    ylabel: str,
    run_count: int,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.6), sharey=True)
    all_values: List[float] = []
    for ax, family in zip(axes, base.FAMILY_ORDER):
        family_data = by_seed[by_seed["family"].eq(family)]
        for seed in ten.SEEDS:
            subset = family_data[family_data["seed"].eq(seed)].sort_values(
                "level_index"
            )
            if subset.empty:
                continue
            all_values.extend(subset[f"{metric}_mean"].astype(float).tolist())
            ax.plot(
                subset["level_index"],
                subset[f"{metric}_mean"],
                marker=ten.SEED_MARKERS[seed],
                markersize=3.2,
                linewidth=0.9,
                alpha=0.28,
                color=ten.SEED_COLORS[seed],
                label=f"Seed {seed}",
            )
        summary = ci[ci["family"].eq(family)].sort_values("level_index")
        x = summary["level_index"].to_numpy(dtype=float)
        mean = summary[f"{metric}_mean"].to_numpy(dtype=float)
        low = summary[f"{metric}_seed_ci95_low"].to_numpy(dtype=float)
        high = summary[f"{metric}_seed_ci95_high"].to_numpy(dtype=float)
        ax.errorbar(
            x,
            mean,
            yerr=np.vstack([mean - low, high - mean]),
            marker="o",
            markersize=6.5,
            linewidth=2.5,
            capsize=4,
            color="#111827",
            label="Available-seed mean ± 95% CI",
            zorder=10,
        )
        for _, point in summary.iterrows():
            n = int(point["seed_count"])
            if n < len(ten.SEEDS):
                ax.annotate(
                    f"n={n}",
                    (
                        float(point["level_index"]),
                        float(point[f"{metric}_seed_ci95_high"]),
                    ),
                    xytext=(0, 6),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                    color="#7c2d12",
                    fontweight="bold",
                )
        all_values.extend(low.tolist())
        all_values.extend(high.tolist())
        labels = summary.set_index("level_index")["level"]
        ax.set_xticks(labels.index.astype(float))
        ax.set_xticklabels(labels.astype(str))
        ax.set_title(base.FAMILY_LABELS[family])
        ax.set_xlabel("Requested reasoning effort")
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel(ylabel)
    shared.set_shared_ylim(axes, all_values)
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, frameon=False)
    fig.suptitle(
        partial_title(run_count)
        + "\nStudent-t 95% CI across available seed estimates (n=8–10)",
        fontsize=11.5,
    )
    fig.tight_layout(rect=[0, 0.14, 1, 0.90])
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_coverage(rows: pd.DataFrame, output: Path) -> None:
    counts = (
        rows.groupby(["seed", "family", "level_index"]).size().rename("count").reset_index()
    )
    columns = [
        (family, level)
        for family in base.FAMILY_ORDER
        for level in range(4)
    ]
    matrix = np.zeros((len(ten.SEEDS), len(columns)), dtype=int)
    for row_index, seed in enumerate(ten.SEEDS):
        seed_counts = counts[counts["seed"].eq(seed)].set_index(
            ["family", "level_index"]
        )["count"]
        for column_index, key in enumerate(columns):
            matrix[row_index, column_index] = int(seed_counts.get(key, 0))

    fig, ax = plt.subplots(figsize=(15.0, 5.2))
    image = ax.imshow(matrix, vmin=0, vmax=18, cmap="YlGn", aspect="auto")
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            ax.text(
                column_index,
                row_index,
                str(value),
                ha="center",
                va="center",
                fontsize=9,
                color="white" if value >= 14 else "#173b2a",
                fontweight="bold",
            )
    short = {"gpt-5": "GPT", "claude-sonnet-4-6": "Claude", "gemini-3-flash": "Gemini"}
    level_names = ["min", "low", "med", "high/max"]
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels(
        [f"{short[family]}\n{level_names[level]}" for family, level in columns],
        fontsize=9,
    )
    ax.set_yticks(range(len(ten.SEEDS)))
    ax.set_yticklabels([str(seed) for seed in ten.SEEDS])
    ax.set_xlabel("Target model family and requested reasoning level")
    ax.set_ylabel("Random seed")
    ax.set_title("Completed runs per seed × family × effort cell (18 expected)")
    fig.colorbar(image, ax=ax, label="Completed runs")
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    roots = {
        seed: getattr(args, f"seed{seed}_root").resolve() for seed in ten.SEEDS
    }
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    shared.SEED_COLORS.clear()
    shared.SEED_COLORS.update(ten.SEED_COLORS)
    shared.SEED_MARKERS.clear()
    shared.SEED_MARKERS.update(ten.SEED_MARKERS)

    data = {
        seed: base.collect_run_rows(root, seed) for seed, root in roots.items()
    }
    rows = pd.concat([data[seed] for seed in ten.SEEDS], ignore_index=True)
    base.validate_grid(rows, ten.SEEDS, allow_incomplete=True)
    cell, by_seed, combined = base.summarize(rows)
    ci = available_seed_ci(by_seed)

    rows.to_csv(output_dir / "run_level_results_partial.csv", index=False)
    by_seed.to_csv(output_dir / "family_effort_by_seed_partial.csv", index=False)
    ci.to_csv(output_dir / "family_effort_available_seed_ci95_partial.csv", index=False)

    plot_seed_curves(
        by_seed,
        output_dir / "target_payoff_ten_seed_partial.png",
        "target_utility",
        "Mean target payoff",
        len(rows),
    )
    plot_seed_curves(
        by_seed,
        output_dir / "utility_gap_ten_seed_partial.png",
        "utility_gap",
        "Mean target − baseline payoff",
        len(rows),
    )
    plot_available_seed_ci(
        by_seed,
        ci,
        output_dir / "target_payoff_available_seed_ci95_partial.png",
        "target_utility",
        "Mean target payoff",
        len(rows),
    )
    plot_available_seed_ci(
        by_seed,
        ci,
        output_dir / "utility_gap_available_seed_ci95_partial.png",
        "utility_gap",
        "Mean target − baseline payoff",
        len(rows),
    )
    plot_coverage(rows, output_dir / "completion_coverage_partial.png")
    base.plot_combined(
        combined,
        output_dir / "target_payoff_pooled_partial.png",
        f"partial 10 seeds, {len(rows):,}/2,160 runs",
    )
    base.plot_target_and_baseline(
        combined,
        output_dir / "combined_target_and_baseline_partial.png",
        partial_title(len(rows)),
    )
    shared.plot_game_stratified(
        cell,
        output_dir / "target_payoff_by_game_partial.png",
        ten.SEEDS,
    )
    for seed in ten.NEW_SEEDS:
        seed_rows = int(rows["seed"].eq(seed).sum())
        base.plot_target_and_baseline(
            by_seed[by_seed["seed"].eq(seed)],
            output_dir / f"seed{seed}_target_and_baseline_partial.png",
            f"Seed {seed}: partial snapshot ({seed_rows}/216 runs)",
        )

    missing = {
        str(seed): sorted(
            set(range(1, 217))
            - set(data[seed]["config_id"].astype(int))
        )
        for seed in ten.SEEDS
    }
    audit = {
        "snapshot_kind": "partial",
        "seeds": ten.SEEDS,
        "completed_runs": int(len(rows)),
        "expected_runs": 2160,
        "by_seed_completed": {
            str(seed): int(len(data[seed])) for seed in ten.SEEDS
        },
        "missing_config_ids": missing,
        "confidence_interval_unit": "available seed-level estimate",
        "confidence_interval_method": "two-sided Student-t 95% CI",
        "confidence_interval_seed_count_range": [
            int(ci["seed_count"].min()),
            int(ci["seed_count"].max()),
        ],
        "important_caveat": (
            "Claude max uses eight available seed estimates; seeds 128, 256, and "
            "612 are themselves based on 11, 12, and 13 of 18 runs, while seeds "
            "2048 and 4096 have no Claude-max runs."
        ),
        "roots": {str(seed): str(root) for seed, root in roots.items()},
    }
    (output_dir / "partial_snapshot_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "completed_runs": len(rows),
                "png_count": len(list(output_dir.glob("*.png"))),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
