#!/usr/bin/env python3
"""Explore Elo-variance stratified sampling for the current 24-model pool.

This is intentionally analysis-only. It imports the live heterogeneous pool used
by the full Games 1-3 generator, enumerates all C(24, N) subsets, and compares
pure random subset draws against equal 5-stratum sampling.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
import sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.full_games123_multiagent_batch import (  # noqa: E402
    N_VALUES,
    filtered_heterogeneous_pool,
)
from strong_models_experiment.analysis.active_model_roster import (  # noqa: E402
    active_model_elo_map,
)


OUTPUT_DIR = Path(__file__).resolve().parent / "stratified_sampling_24pool"
DEFAULT_SEED = 20260428
DEFAULT_K_VALUES = [1000, 10000]
FIVE_LABELS = ["very_low", "low", "mid", "high", "very_high"]
THREE_LABELS = ["low", "mid", "high"]


@dataclass(frozen=True)
class SubsetRow:
    indices: tuple[int, ...]
    models: tuple[str, ...]
    variance: float
    stddev: float


def stable_seed(seed: int, *parts: object) -> int:
    raw = "|".join([str(seed), *(str(part) for part in parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


def quantile_chunks(rows: list[SubsetRow], n_bins: int) -> list[list[SubsetRow]]:
    total = len(rows)
    return [
        rows[(bin_index * total) // n_bins : ((bin_index + 1) * total) // n_bins]
        for bin_index in range(n_bins)
    ]


def band_index(stddev: float, chunks: list[list[SubsetRow]]) -> int:
    for index, chunk in enumerate(chunks[:-1]):
        if stddev <= chunk[-1].stddev + 1e-12:
            return index
    return len(chunks) - 1


def summarize(values: Iterable[float]) -> dict[str, float]:
    arr = np.asarray(list(values), dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "min": float(np.min(arr)),
        "p05": float(np.percentile(arr, 5)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def enumerate_subsets(pool: list[str], elo_by_model: dict[str, int]) -> dict[int, list[SubsetRow]]:
    elo_values = np.asarray([elo_by_model[model] for model in pool], dtype=float)
    rows_by_n: dict[int, list[SubsetRow]] = {}
    for n_agents in N_VALUES:
        rows: list[SubsetRow] = []
        for indices in combinations(range(len(pool)), n_agents):
            selected_elos = elo_values[list(indices)]
            variance = float(np.var(selected_elos, ddof=0))
            rows.append(
                SubsetRow(
                    indices=tuple(indices),
                    models=tuple(pool[index] for index in indices),
                    variance=variance,
                    stddev=float(np.sqrt(variance)),
                )
            )
        rows.sort(key=lambda row: row.stddev)
        rows_by_n[n_agents] = rows
    return rows_by_n


def random_draws(rows: list[SubsetRow], k: int, rng: random.Random) -> list[SubsetRow]:
    if k <= len(rows):
        return rng.sample(rows, k)
    return [rng.choice(rows) for _ in range(k)]


def five_stratified_draws(
    five_chunks: list[list[SubsetRow]],
    k: int,
    rng: random.Random,
) -> list[SubsetRow]:
    per_bin = [k // 5 + (1 if index < k % 5 else 0) for index in range(5)]
    selected: list[SubsetRow] = []
    for chunk, count in zip(five_chunks, per_bin):
        if count <= len(chunk):
            selected.extend(rng.sample(chunk, count))
        else:
            selected.extend(rng.choice(chunk) for _ in range(count))
    rng.shuffle(selected)
    return selected


def write_model_pool(output_dir: Path, pool: list[str], elo_by_model: dict[str, int]) -> None:
    with (output_dir / "model_pool_24.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "arena_elo"])
        writer.writeheader()
        for model in pool:
            writer.writerow({"model": model, "arena_elo": elo_by_model[model]})


def write_exact_summary(
    output_dir: Path,
    rows_by_n: dict[int, list[SubsetRow]],
    five_chunks_by_n: dict[int, list[list[SubsetRow]]],
    three_chunks_by_n: dict[int, list[list[SubsetRow]]],
) -> None:
    fieldnames = [
        "n",
        "subset_count",
        "stddev_mean",
        "stddev_min",
        "stddev_p05",
        "stddev_p50",
        "stddev_p95",
        "stddev_max",
        "variance_mean",
        "variance_min",
        "variance_p05",
        "variance_p50",
        "variance_p95",
        "variance_max",
        "five_strata_stddev_ranges",
        "tertile_stddev_ranges",
    ]
    with (output_dir / "exact_subset_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for n_agents, rows in rows_by_n.items():
            stddev_summary = summarize(row.stddev for row in rows)
            variance_summary = summarize(row.variance for row in rows)
            writer.writerow(
                {
                    "n": n_agents,
                    "subset_count": len(rows),
                    **{f"stddev_{key}": f"{value:.6f}" for key, value in stddev_summary.items()},
                    **{f"variance_{key}": f"{value:.6f}" for key, value in variance_summary.items()},
                    "five_strata_stddev_ranges": "; ".join(
                        f"{label}:{chunk[0].stddev:.6f}-{chunk[-1].stddev:.6f} ({len(chunk)})"
                        for label, chunk in zip(FIVE_LABELS, five_chunks_by_n[n_agents])
                    ),
                    "tertile_stddev_ranges": "; ".join(
                        f"{label}:{chunk[0].stddev:.6f}-{chunk[-1].stddev:.6f} ({len(chunk)})"
                        for label, chunk in zip(THREE_LABELS, three_chunks_by_n[n_agents])
                    ),
                }
            )


def summarize_sample(
    *,
    n_agents: int,
    k: int,
    strategy: str,
    draws: list[SubsetRow],
    five_chunks: list[list[SubsetRow]],
    three_chunks: list[list[SubsetRow]],
) -> dict[str, object]:
    stddev_summary = summarize(row.stddev for row in draws)
    variance_summary = summarize(row.variance for row in draws)
    five_counts = {label: 0 for label in FIVE_LABELS}
    three_counts = {label: 0 for label in THREE_LABELS}
    for row in draws:
        five_counts[FIVE_LABELS[band_index(row.stddev, five_chunks)]] += 1
        three_counts[THREE_LABELS[band_index(row.stddev, three_chunks)]] += 1
    unique_subsets = len({row.indices for row in draws})
    summary: dict[str, object] = {
        "n": n_agents,
        "k": k,
        "strategy": strategy,
        "draw_count": len(draws),
        "unique_unordered_subsets": unique_subsets,
    }
    summary.update({f"stddev_{key}": f"{value:.6f}" for key, value in stddev_summary.items()})
    summary.update({f"variance_{key}": f"{value:.6f}" for key, value in variance_summary.items()})
    summary.update({f"five_bin_{label}": five_counts[label] for label in FIVE_LABELS})
    summary.update({f"tertile_{label}": three_counts[label] for label in THREE_LABELS})
    return summary


def generate_samples(
    rows_by_n: dict[int, list[SubsetRow]],
    five_chunks_by_n: dict[int, list[list[SubsetRow]]],
    seed: int,
    k_values: list[int],
) -> dict[tuple[int, int, str], list[SubsetRow]]:
    samples: dict[tuple[int, int, str], list[SubsetRow]] = {}
    for n_agents, rows in rows_by_n.items():
        five_chunks = five_chunks_by_n[n_agents]
        for k in k_values:
            pure_rng = random.Random(stable_seed(seed, "pure_random", n_agents, k))
            strat_rng = random.Random(stable_seed(seed, "five_stratified", n_agents, k))
            samples[(n_agents, k, "pure_random")] = random_draws(rows, k, pure_rng)
            samples[(n_agents, k, "five_stratified")] = five_stratified_draws(
                five_chunks,
                k,
                strat_rng,
            )
    return samples


def write_sampling_summary(
    output_dir: Path,
    samples: dict[tuple[int, int, str], list[SubsetRow]],
    five_chunks_by_n: dict[int, list[list[SubsetRow]]],
    three_chunks_by_n: dict[int, list[list[SubsetRow]]],
) -> None:
    rows = []
    for (n_agents, k, strategy), draws in sorted(samples.items()):
        rows.append(
            summarize_sample(
                n_agents=n_agents,
                k=k,
                strategy=strategy,
                draws=draws,
                five_chunks=five_chunks_by_n[n_agents],
                three_chunks=three_chunks_by_n[n_agents],
            )
        )
    fieldnames = list(rows[0].keys())
    with (output_dir / "sampling_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_exact_variance_histograms(
    output_dir: Path,
    rows_by_n: dict[int, list[SubsetRow]],
    five_chunks_by_n: dict[int, list[list[SubsetRow]]],
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), squeeze=False)
    axes_flat = axes.ravel()
    for axis, n_agents in zip(axes_flat, N_VALUES):
        values = np.asarray([row.variance for row in rows_by_n[n_agents]], dtype=float)
        axis.hist(values, bins=45, color="#4C78A8", edgecolor="white")
        for chunk in five_chunks_by_n[n_agents][:-1]:
            axis.axvline(chunk[-1].variance, color="#B91C1C", linestyle="--", linewidth=1)
        axis.set_title(f"N={n_agents} ({len(values):,} subsets)")
        axis.set_xlabel("Elo variance")
        axis.set_ylabel("Subset count")
        axis.grid(axis="y", alpha=0.25)
    axes_flat[-1].axis("off")
    fig.suptitle("Exact Elo-Variance Distributions for All C(24, N) Subsets")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"exact_elo_variance_histograms_24pool.{suffix}", dpi=200)
    plt.close(fig)


def plot_bin_counts(
    output_dir: Path,
    samples: dict[tuple[int, int, str], list[SubsetRow]],
    five_chunks_by_n: dict[int, list[list[SubsetRow]]],
    k_values: list[int],
) -> None:
    colors = {"pure_random": "#4C78A8", "five_stratified": "#F58518"}
    for k in k_values:
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), squeeze=False)
        axes_flat = axes.ravel()
        for axis, n_agents in zip(axes_flat, N_VALUES):
            x = np.arange(len(FIVE_LABELS), dtype=float)
            width = 0.36
            for offset, strategy in [(-width / 2, "pure_random"), (width / 2, "five_stratified")]:
                counts = {label: 0 for label in FIVE_LABELS}
                for row in samples[(n_agents, k, strategy)]:
                    label = FIVE_LABELS[band_index(row.stddev, five_chunks_by_n[n_agents])]
                    counts[label] += 1
                axis.bar(
                    x + offset,
                    [counts[label] for label in FIVE_LABELS],
                    width=width,
                    color=colors[strategy],
                    label=strategy.replace("_", " "),
                )
            axis.axhline(k / 5, color="#475569", linestyle="--", linewidth=1)
            axis.set_title(f"N={n_agents}")
            axis.set_xticks(x)
            axis.set_xticklabels(FIVE_LABELS, rotation=30, ha="right")
            axis.set_ylabel("Frequency")
            axis.grid(axis="y", alpha=0.25)
        axes_flat[-1].axis("off")
        handles, labels = axes_flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.98, 0.96))
        fig.suptitle(f"5-Stratum Frequency Counts by Sampling Strategy (k={k:,})")
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        for suffix in ("png", "pdf"):
            fig.savefig(output_dir / f"sampling_5bin_counts_k{k}.{suffix}", dpi=200)
        plt.close(fig)


def plot_sampled_variance_histograms(
    output_dir: Path,
    samples: dict[tuple[int, int, str], list[SubsetRow]],
    five_chunks_by_n: dict[int, list[list[SubsetRow]]],
    k_values: list[int],
) -> None:
    colors = {"pure_random": "#4C78A8", "five_stratified": "#F58518"}
    for k in k_values:
        fig, axes = plt.subplots(len(N_VALUES), 2, figsize=(12, 13), squeeze=False)
        for row_index, n_agents in enumerate(N_VALUES):
            for col_index, strategy in enumerate(["pure_random", "five_stratified"]):
                axis = axes[row_index, col_index]
                values = np.asarray(
                    [row.variance for row in samples[(n_agents, k, strategy)]],
                    dtype=float,
                )
                axis.hist(values, bins=35, color=colors[strategy], edgecolor="white")
                for chunk in five_chunks_by_n[n_agents][:-1]:
                    axis.axvline(chunk[-1].variance, color="#B91C1C", linestyle="--", linewidth=0.8)
                axis.set_title(f"N={n_agents}, {strategy.replace('_', ' ')}")
                axis.set_xlabel("Elo variance")
                axis.set_ylabel("Frequency")
                axis.grid(axis="y", alpha=0.25)
        fig.suptitle(f"Sampled Elo-Variance Histograms (k={k:,})")
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        for suffix in ("png", "pdf"):
            fig.savefig(output_dir / f"sampled_elo_variance_histograms_k{k}.{suffix}", dpi=200)
        plt.close(fig)


def write_readme(output_dir: Path, pool_size: int, k_values: list[int]) -> None:
    lines = [
        "# Stratified Elo-Variance Sampling Diagnostics",
        "",
        f"Model pool size: `{pool_size}`.",
        f"Sample sizes: `{', '.join(str(k) for k in k_values)}`.",
        "",
        "This directory is generated by `plot_stratified_sampling_24pool.py`.",
        "It is exploratory only; it does not modify experiment-generation machinery.",
        "",
        "Files:",
        "- `model_pool_24.csv`: live 24-model heterogeneous pool and Elo values.",
        "- `exact_subset_summary.csv`: exact C(24, N) subset summaries and bin ranges.",
        "- `sampling_summary.csv`: simulated pure-random vs 5-stratified sample summaries.",
        "- `exact_elo_variance_histograms_24pool.png`: exact variance histograms.",
        "- `sampling_5bin_counts_k*.png`: frequency counts by 5-stratum bin.",
        "- `sampled_elo_variance_histograms_k*.png`: sampled variance histograms.",
        "",
        "The 5-stratified strategy samples equally from five equal-count Elo-stddev bins.",
        "For diagnostic `k` values larger than a bin's unique subset count, sampling within that bin uses replacement.",
    ]
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--k-values", type=int, nargs="+", default=DEFAULT_K_VALUES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    pool = list(filtered_heterogeneous_pool())
    elo_by_model = active_model_elo_map()
    rows_by_n = enumerate_subsets(pool, elo_by_model)
    five_chunks_by_n = {
        n_agents: quantile_chunks(rows, 5)
        for n_agents, rows in rows_by_n.items()
    }
    three_chunks_by_n = {
        n_agents: quantile_chunks(rows, 3)
        for n_agents, rows in rows_by_n.items()
    }
    samples = generate_samples(rows_by_n, five_chunks_by_n, args.seed, args.k_values)

    write_model_pool(output_dir, pool, elo_by_model)
    write_exact_summary(output_dir, rows_by_n, five_chunks_by_n, three_chunks_by_n)
    write_sampling_summary(output_dir, samples, five_chunks_by_n, three_chunks_by_n)
    plot_exact_variance_histograms(output_dir, rows_by_n, five_chunks_by_n)
    plot_bin_counts(output_dir, samples, five_chunks_by_n, args.k_values)
    plot_sampled_variance_histograms(output_dir, samples, five_chunks_by_n, args.k_values)
    write_readme(output_dir, len(pool), args.k_values)

    print(f"Wrote diagnostics to {output_dir}")
    for n_agents in N_VALUES:
        rows = rows_by_n[n_agents]
        print(
            f"N={n_agents}: subsets={len(rows):,}, "
            f"stddev_range={rows[0].stddev:.2f}-{rows[-1].stddev:.2f}"
        )


if __name__ == "__main__":
    main()
