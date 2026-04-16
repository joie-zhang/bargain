#!/usr/bin/env python3
"""Replot Game 3 (n=2) figures from CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("Agg")


def plot_overall(csv_dir: Path, out_dir: Path) -> None:
    df = pd.read_csv(csv_dir / "utility_vs_elo_all_models.csv").sort_values("elo")
    fig, ax = plt.subplots(figsize=(11, 7))
    for _, row in df.iterrows():
        x = float(row["elo"])
        y = float(row["avg_utility"])
        ax.scatter(x, y, s=110, color="#2563eb", alpha=0.9)
        ax.annotate(str(row["model_short"]), (x, y), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=10)
    ax.set_title("Game 3: Mean Adversary Utility vs Chatbot Arena Elo\nAll models with any finished runs; averages span completed alpha/sigma settings and both model orders.")
    ax.set_xlabel("Chatbot Arena Elo (adversary model)")
    ax.set_ylabel("Mean Adversary Utility")
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_dir / "utility_vs_elo_all_models.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_by_ci(csv_dir: Path, out_dir: Path) -> None:
    df = pd.read_csv(csv_dir / "utility_vs_elo_by_competition_index.csv").sort_values(["competition_index", "elo"])
    ci_vals = sorted(df["competition_index"].unique().tolist())
    colors = plt.cm.viridis([i / max(len(ci_vals) - 1, 1) for i in range(len(ci_vals))])

    fig, ax = plt.subplots(figsize=(12, 7.2))
    for color, ci in zip(colors, ci_vals):
        subset = df[df["competition_index"] == ci].sort_values("elo")
        ax.plot(subset["elo"], subset["avg_utility"], marker="o", markersize=6.5, linewidth=2.2, color=color, alpha=0.9, label=f"CI3={ci:.2f}")
    ax.set_title("Game 3: Mean Adversary Utility vs Chatbot Arena Elo\nStratified by derived CI3 = (1 - alpha) * (1 - sigma); averages include both model orders.")
    ax.set_xlabel("Chatbot Arena Elo (adversary model)")
    ax.set_ylabel("Mean Adversary Utility")
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(title="Derived CI3", frameon=True)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_dir / "utility_vs_elo_by_competition_index.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_by_alpha_sigma(csv_dir: Path, out_dir: Path) -> None:
    df = pd.read_csv(csv_dir / "utility_vs_elo_by_alpha_sigma.csv").sort_values(["alpha", "sigma", "elo"])
    alpha_vals = sorted(df["alpha"].unique().tolist())
    sigma_vals = sorted(df["sigma"].unique().tolist())
    fig, axes = plt.subplots(1, len(alpha_vals), figsize=(6 * len(alpha_vals), 5.8), sharey=True)
    if len(alpha_vals) == 1:
        axes = [axes]

    colors = plt.cm.viridis([i / max(len(sigma_vals) - 1, 1) for i in range(len(sigma_vals))])
    handles = []
    labels = []

    for ax, alpha in zip(axes, alpha_vals):
        adf = df[df["alpha"] == alpha]
        for color, sigma in zip(colors, sigma_vals):
            subset = adf[adf["sigma"] == sigma].sort_values("elo")
            if subset.empty:
                continue
            (line,) = ax.plot(subset["elo"], subset["avg_utility"], marker="o", markersize=6, linewidth=2.1, color=color, alpha=0.95)
            if len(handles) < len(sigma_vals):
                handles.append(line)
                labels.append(rf"$\sigma={float(sigma):.1f}$")
        ax.set_title(rf"$\alpha={float(alpha):.1f}$")
        ax.set_xlabel("Chatbot Arena Elo")
        ax.grid(alpha=0.25)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Mean Adversary Utility")
    fig.suptitle("Game 3: Mean Adversary Utility vs Elo by alpha and sigma\nPanels fix alpha; curves fix sigma; averages include both model orders.", y=1.03)
    fig.legend(handles, labels, title=r"$\sigma$", loc="upper center", ncol=max(1, len(sigma_vals)))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_dir / "utility_vs_elo_by_alpha_sigma.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_overall(args.input_dir, args.output_dir)
    plot_by_ci(args.input_dir, args.output_dir)
    plot_by_alpha_sigma(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
