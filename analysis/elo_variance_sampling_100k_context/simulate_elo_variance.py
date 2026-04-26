#!/usr/bin/env python3
"""Sample model pools and plot Elo standard-deviation histograms."""

from __future__ import annotations

import csv
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUTPUT_DIR = Path(__file__).resolve().parent
DEFAULT_RANDOM_SEED = 20260426
DEFAULT_K = 1000
N_VALUES = [2, 4, 6, 8, 10]

MODELS = [
    ("claude-opus-4-6-thinking", 1504, "Anthropic", "1M"),
    ("claude-opus-4-6", 1499, "Anthropic", "1M"),
    ("gemini-3-pro", 1486, "Google", "1M"),
    ("gpt-5.4-high", 1484, "OpenAI", "1.1M"),
    ("gpt-5.2-chat-latest-20260210", 1478, "OpenAI", "128K"),
    ("claude-opus-4-5-20251101-thinking-32k", 1474, "Anthropic", "200K"),
    ("claude-opus-4-5-20251101", 1468, "Anthropic", "200K"),
    ("gemini-2.5-pro", 1448, "Google", "1M"),
    ("qwen3-max-preview", 1435, "OpenRouter", "262,144"),
    ("deepseek-r1-0528", 1422, "OpenRouter", "163,840"),
    ("claude-haiku-4-5-20251001", 1407, "Anthropic", "200K"),
    ("claude-sonnet-4-20250514", 1389, "Anthropic", "1M"),
    ("gemma-3-27b-it", 1365, "OpenRouter", "131,072"),
    ("o3-mini-high", 1363, "OpenAI", "200K"),
    ("deepseek-v3", 1358, "OpenRouter", "163,840"),
    ("gpt-4o-2024-05-13", 1345, "OpenAI", "128K"),
    ("gpt-5-nano-high", 1337, "OpenAI", "128K"),
    ("qwq-32b", 1336, "OpenRouter", "131,072"),
    ("gpt-4.1-nano-2025-04-14", 1322, "OpenAI", "1M"),
    ("llama-3.3-70b-instruct", 1318, "OpenRouter", "131,072"),
    ("gpt-4o-mini-2024-07-18", 1317, "OpenAI", "128K"),
    ("amazon-nova-pro-v1.0", 1290, "OpenRouter", "300,000"),
    ("command-r-plus-08-2024", 1276, "OpenRouter", "128,000"),
    ("claude-3-haiku-20240307", 1260, "Anthropic", "200K"),
    ("amazon-nova-micro-v1.0", 1240, "OpenRouter", "128,000"),
]


def write_model_pool() -> None:
    path = OUTPUT_DIR / "filtered_100k_context_model_pool.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model", "arena_elo", "route", "context_used_for_filter"],
        )
        writer.writeheader()
        for model, elo, route, context in MODELS:
            writer.writerow(
                {
                    "model": model,
                    "arena_elo": elo,
                    "route": route,
                    "context_used_for_filter": context,
                }
            )


def output_path(stem: str, k: int, suffix: str) -> Path:
    if k == DEFAULT_K:
        return OUTPUT_DIR / f"{stem}.{suffix}"
    return OUTPUT_DIR / f"{stem}_k{k}.{suffix}"


def simulate(k: int, seed: int) -> dict[int, np.ndarray]:
    rng = np.random.default_rng(seed)
    names = np.array([model[0] for model in MODELS], dtype=object)
    elos = np.array([model[1] for model in MODELS], dtype=float)
    results: dict[int, list[float]] = {n: [] for n in N_VALUES}

    samples_path = output_path("elo_variance_random_samples", k, "csv")
    with samples_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "n",
                "run",
                "elo_variance",
                "elo_stddev",
                "selected_models",
                "selected_elos",
            ],
        )
        writer.writeheader()

        for n in N_VALUES:
            for run_index in range(k):
                selected = rng.choice(len(MODELS), size=n, replace=False)
                selected_elos = elos[selected]
                elo_variance = float(np.var(selected_elos, ddof=0))
                elo_stddev = float(np.sqrt(elo_variance))
                results[n].append(elo_variance)
                writer.writerow(
                    {
                        "n": n,
                        "run": run_index,
                        "elo_variance": f"{elo_variance:.6f}",
                        "elo_stddev": f"{elo_stddev:.6f}",
                        "selected_models": ";".join(names[selected]),
                        "selected_elos": ";".join(str(int(value)) for value in selected_elos),
                    }
                )

    return {n: np.array(values, dtype=float) for n, values in results.items()}


def write_summary(results: dict[int, np.ndarray], k: int) -> None:
    path = output_path("elo_variance_summary", k, "csv")
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "n",
                "k",
                "mean_elo_variance",
                "std_of_elo_variance",
                "min_elo_variance",
                "p05_elo_variance",
                "median_elo_variance",
                "p95_elo_variance",
                "max_elo_variance",
                "mean_elo_stddev",
                "std_of_elo_stddev",
                "min_elo_stddev",
                "p05_elo_stddev",
                "median_elo_stddev",
                "p95_elo_stddev",
                "max_elo_stddev",
            ],
        )
        writer.writeheader()
        for n in N_VALUES:
            values = results[n]
            stddevs = np.sqrt(values)
            writer.writerow(
                {
                    "n": n,
                    "k": k,
                    "mean_elo_variance": f"{float(np.mean(values)):.6f}",
                    "std_of_elo_variance": f"{float(np.std(values, ddof=0)):.6f}",
                    "min_elo_variance": f"{float(np.min(values)):.6f}",
                    "p05_elo_variance": f"{float(np.percentile(values, 5)):.6f}",
                    "median_elo_variance": f"{float(np.median(values)):.6f}",
                    "p95_elo_variance": f"{float(np.percentile(values, 95)):.6f}",
                    "max_elo_variance": f"{float(np.max(values)):.6f}",
                    "mean_elo_stddev": f"{float(np.mean(stddevs)):.6f}",
                    "std_of_elo_stddev": f"{float(np.std(stddevs, ddof=0)):.6f}",
                    "min_elo_stddev": f"{float(np.min(stddevs)):.6f}",
                    "p05_elo_stddev": f"{float(np.percentile(stddevs, 5)):.6f}",
                    "median_elo_stddev": f"{float(np.median(stddevs)):.6f}",
                    "p95_elo_stddev": f"{float(np.percentile(stddevs, 95)):.6f}",
                    "max_elo_stddev": f"{float(np.max(stddevs)):.6f}",
                }
            )


def plot_histograms(results: dict[int, np.ndarray], k: int) -> None:
    stddev_results = {n: np.sqrt(results[n]) for n in N_VALUES}
    all_values = np.concatenate([stddev_results[n] for n in N_VALUES])
    bins = np.linspace(0.0, float(np.max(all_values)), 31)

    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5), sharex=True)
    axes_flat = axes.ravel()

    for axis, n in zip(axes_flat, N_VALUES):
        values = stddev_results[n]
        axis.hist(values, bins=bins, color="#4C78A8", edgecolor="white")
        axis.set_title(f"n = {n}")
        axis.set_xlabel("Elo standard deviation")
        axis.set_ylabel("Frequency")
        axis.grid(axis="y", alpha=0.25)

    axes_flat[-1].axis("off")
    fig.suptitle(
        "Random model-pool Elo standard deviation by number of agents "
        f"(k = {k}, sampled without replacement)"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUTPUT_DIR / f"elo_stddev_histograms_k{k}.png", dpi=200)
    fig.savefig(OUTPUT_DIR / f"elo_stddev_histograms_k{k}.pdf")
    if k == DEFAULT_K:
        # Also overwrite the original plot path so prior links show the updated x-axis.
        fig.savefig(OUTPUT_DIR / "elo_variance_histograms_k1000.png", dpi=200)
        fig.savefig(OUTPUT_DIR / "elo_variance_histograms_k1000.pdf")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample model subsets and plot Elo standard-deviation histograms."
    )
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="random draws per n")
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_model_pool()
    results = simulate(args.k, args.seed)
    write_summary(results, args.k)
    plot_histograms(results, args.k)
    for n in N_VALUES:
        values = np.sqrt(results[n])
        print(
            f"n={n}: mean={np.mean(values):.2f}, "
            f"median={np.median(values):.2f}, "
            f"p05={np.percentile(values, 5):.2f}, "
            f"p95={np.percentile(values, 95):.2f}"
        )


if __name__ == "__main__":
    main()
