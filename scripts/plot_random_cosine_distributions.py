#!/usr/bin/env python3
"""
Run random-vector cosine-similarity trials and plot the resulting distributions.

For each even n in {2, 4, 6, 8, 10}:
- sample `trials` independent batches of n random vectors
- each vector has dimension 2.5 * n
- compute all pairwise cosine similarities within each batch
- record the mean and variance across those pairwise similarities

The script writes:
- a 5x2 subplot figure with one row per n
- a CSV with one row per trial
- a CSV summary aggregated by n

Assumption:
- vectors are sampled i.i.d. from a strictly positive uniform distribution on
  (0, 1], so all cosine similarities lie in (0, 1].
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_N_VALUES = (2, 4, 6, 8, 10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot distributions of trial-level mean and variance of pairwise cosine similarities."
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=10000,
        help="Number of independent trials per n (default: 10000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility (default: 0).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("visualization/figures/random_cosine_similarity"),
        help="Directory for plots and CSV outputs.",
    )
    parser.add_argument(
        "--ns",
        type=int,
        nargs="+",
        default=list(DEFAULT_N_VALUES),
        help="Even n values to evaluate (default: 2 4 6 8 10).",
    )
    return parser.parse_args()


def validate_n_values(n_values: list[int]) -> None:
    for n in n_values:
        if n % 2 != 0:
            raise ValueError(f"n must be even, got {n}")
        if (5 * n) % 2 != 0:
            raise ValueError(f"2.5 * n must be an integer, got n={n}")


def run_trials_for_n(n: int, trials: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    dim = (5 * n) // 2
    vectors = rng.uniform(low=1e-12, high=1.0, size=(trials, n, dim))

    norms = np.linalg.norm(vectors, axis=2, keepdims=True)
    vectors = vectors / norms

    cosine_matrices = np.matmul(vectors, np.swapaxes(vectors, 1, 2))
    upper_i, upper_j = np.triu_indices(n, k=1)
    pairwise_cosines = cosine_matrices[:, upper_i, upper_j]

    trial_means = pairwise_cosines.mean(axis=1)
    trial_variances = pairwise_cosines.var(axis=1)

    return {
        "pairwise_cosines": pairwise_cosines,
        "trial_means": trial_means,
        "trial_variances": trial_variances,
        "dim": np.array([dim]),
    }


def compute_histogram_ymax(all_values: list[np.ndarray], bins: np.ndarray) -> float:
    max_count = 0
    for values in all_values:
        counts, _ = np.histogram(values, bins=bins)
        max_count = max(max_count, int(counts.max()))
    return max_count * 1.05


def plot_distribution(
    ax: plt.Axes,
    values: np.ndarray,
    title: str,
    xlabel: str,
    bins: np.ndarray,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> None:
    ax.hist(values, bins=bins, color="#1f77b4", edgecolor="white", alpha=0.9)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.25)


def save_trial_rows(output_path: Path, trial_rows: list[dict[str, float]]) -> None:
    fieldnames = [
        "n",
        "dimension",
        "trial_index",
        "num_pairs",
        "average_pairwise_cosine_similarity",
        "variance_pairwise_cosine_similarity",
    ]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(trial_rows)


def save_summary(output_path: Path, summary_rows: list[dict[str, float]]) -> None:
    fieldnames = [
        "n",
        "dimension",
        "trials",
        "num_pairs",
        "mean_of_trial_means",
        "std_of_trial_means",
        "min_of_trial_means",
        "max_of_trial_means",
        "mean_of_trial_variances",
        "std_of_trial_variances",
        "min_of_trial_variances",
        "max_of_trial_variances",
    ]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)


def main() -> None:
    args = parse_args()
    validate_n_values(args.ns)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    fig, axes = plt.subplots(len(args.ns), 2, figsize=(14, 4 * len(args.ns)))
    if len(args.ns) == 1:
        axes = np.array([axes])

    all_results: list[dict[str, np.ndarray | int]] = []
    trial_rows: list[dict[str, float]] = []
    summary_rows: list[dict[str, float]] = []

    for n in args.ns:
        results = run_trials_for_n(n=n, trials=args.trials, rng=rng)
        all_results.append({"n": n, **results})

    mean_bins = np.linspace(0.0, 1.0, 41)
    variance_max = max(float(np.max(result["trial_variances"])) for result in all_results)
    variance_bins = np.linspace(0.0, variance_max, 41)

    mean_ymax = compute_histogram_ymax(
        [result["trial_means"] for result in all_results],
        mean_bins,
    )
    variance_ymax = compute_histogram_ymax(
        [result["trial_variances"] for result in all_results],
        variance_bins,
    )

    for row_idx, result in enumerate(all_results):
        n = int(result["n"])
        trial_means = result["trial_means"]
        trial_variances = result["trial_variances"]
        dim = int(result["dim"][0])
        num_pairs = n * (n - 1) // 2

        plot_distribution(
            axes[row_idx, 0],
            trial_means,
            title=f"n = {n}, dim = {dim}: Avg Pairwise Cosine",
            xlabel="Average pairwise cosine similarity",
            bins=mean_bins,
            xlim=(0.0, 1.0),
            ylim=(0.0, mean_ymax),
        )
        plot_distribution(
            axes[row_idx, 1],
            trial_variances,
            title=f"n = {n}, dim = {dim}: Variance Across Pairs",
            xlabel="Variance of pairwise cosine similarities",
            bins=variance_bins,
            xlim=(0.0, float(variance_bins[-1])),
            ylim=(0.0, variance_ymax),
        )

        for trial_index, (mean_value, variance_value) in enumerate(zip(trial_means, trial_variances)):
            trial_rows.append(
                {
                    "n": n,
                    "dimension": dim,
                    "trial_index": trial_index,
                    "num_pairs": num_pairs,
                    "average_pairwise_cosine_similarity": float(mean_value),
                    "variance_pairwise_cosine_similarity": float(variance_value),
                }
            )

        summary_rows.append(
            {
                "n": n,
                "dimension": dim,
                "trials": args.trials,
                "num_pairs": num_pairs,
                "mean_of_trial_means": float(trial_means.mean()),
                "std_of_trial_means": float(trial_means.std()),
                "min_of_trial_means": float(trial_means.min()),
                "max_of_trial_means": float(trial_means.max()),
                "mean_of_trial_variances": float(trial_variances.mean()),
                "std_of_trial_variances": float(trial_variances.std()),
                "min_of_trial_variances": float(trial_variances.min()),
                "max_of_trial_variances": float(trial_variances.max()),
            }
        )

    fig.suptitle(
        f"Random Cosine-Similarity Trial Distributions ({args.trials} trials per n, strictly positive uniform vectors)",
        fontsize=14,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.985))

    figure_path = args.output_dir / "random_cosine_similarity_distributions.png"
    fig.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    save_trial_rows(args.output_dir / "random_cosine_similarity_trials.csv", trial_rows)
    save_summary(args.output_dir / "random_cosine_similarity_summary.csv", summary_rows)

    print(f"Saved figure: {figure_path}")
    print(f"Saved trial CSV: {args.output_dir / 'random_cosine_similarity_trials.csv'}")
    print(f"Saved summary CSV: {args.output_dir / 'random_cosine_similarity_summary.csv'}")
    print()
    for row in summary_rows:
        print(
            f"n={row['n']:>2}, dim={row['dimension']:>2}, pairs={row['num_pairs']:>2}, "
            f"mean(avg cosine)={row['mean_of_trial_means']:.6f} +/- {row['std_of_trial_means']:.6f}, "
            f"mean(var)={row['mean_of_trial_variances']:.6f} +/- {row['std_of_trial_variances']:.6f}"
        )


if __name__ == "__main__":
    main()
