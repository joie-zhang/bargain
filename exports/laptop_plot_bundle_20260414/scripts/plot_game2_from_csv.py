#!/usr/bin/env python3
"""Replot Game 2 (n=2) figures from CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("Agg")


def plot_overall(csv_dir: Path, out_dir: Path) -> None:
    df = pd.read_csv(csv_dir / "utility_vs_elo_overall.csv").sort_values("elo")
    fig, ax = plt.subplots(figsize=(11, 7))
    for _, row in df.iterrows():
        x = float(row["elo"])
        y = float(row["avg_utility"])
        ax.scatter(x, y, s=110, color="#2563eb", alpha=0.9)
        ax.annotate(str(row["model_short"]), (x, y), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=10)
    ax.set_title("Game 2: Mean Adversary Utility vs Chatbot Arena Elo\nAverages include all completed (rho, theta, model_order) settings.")
    ax.set_xlabel("Chatbot Arena Elo (adversary model)")
    ax.set_ylabel("Mean Adversary Utility")
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_dir / "utility_vs_elo_overall.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_by_ci(csv_dir: Path, out_dir: Path) -> None:
    df = pd.read_csv(csv_dir / "utility_vs_elo_by_competition_index.csv").sort_values(["competition_index", "elo"])
    ci_vals = sorted(df["competition_index"].unique().tolist())
    colors = plt.cm.viridis([i / max(len(ci_vals) - 1, 1) for i in range(len(ci_vals))])

    fig, ax = plt.subplots(figsize=(11, 7))
    for color, ci in zip(colors, ci_vals):
        subset = df[df["competition_index"] == ci].sort_values("elo")
        ax.plot(subset["elo"], subset["avg_utility"], marker="o", linewidth=2.0, color=color, label=f"CI={ci:.2f}")
    ax.set_title("Game 2: Mean Adversary Utility vs Chatbot Arena Elo\nStratified by derived CI2 = theta * (1 - rho) / 2; averages include both model orders.")
    ax.set_xlabel("Chatbot Arena Elo (adversary model)")
    ax.set_ylabel("Mean Adversary Utility")
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(title="Derived CI2", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_dir / "utility_vs_elo_by_competition_index.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_by_rho_theta(csv_dir: Path, out_dir: Path) -> None:
    df = pd.read_csv(csv_dir / "utility_vs_elo_by_rho_theta.csv")
    df = df.sort_values(["theta", "rho", "elo"])
    theta_vals = sorted(df["theta"].unique().tolist())
    rho_vals = sorted(df["rho"].unique().tolist())
    fig, axes = plt.subplots(1, len(theta_vals), figsize=(6 * len(theta_vals), 5.8), sharey=True)
    if len(theta_vals) == 1:
        axes = [axes]

    colors = plt.cm.viridis([i / max(len(rho_vals) - 1, 1) for i in range(len(rho_vals))])
    handles = []
    labels = []

    for ax, theta in zip(axes, theta_vals):
        tdf = df[df["theta"] == theta]
        for color, rho in zip(colors, rho_vals):
            subset = tdf[tdf["rho"] == rho].sort_values("elo")
            if subset.empty:
                continue
            (line,) = ax.plot(subset["elo"], subset["avg_utility"], marker="o", linewidth=2.0, color=color)
            if len(handles) < len(rho_vals):
                handles.append(line)
                labels.append(rf"$\rho={float(rho):.1f}$")
        ax.set_title(rf"$\theta={float(theta):.1f}$")
        ax.set_xlabel("Chatbot Arena Elo")
        ax.grid(alpha=0.25)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Mean Adversary Utility")
    fig.suptitle("Game 2: Mean Adversary Utility vs Elo by rho and theta\nPanels fix theta; curves fix rho; averages include both model orders.", y=1.03)
    fig.legend(handles, labels, title=r"$\rho$", loc="upper center", ncol=max(1, len(rho_vals)))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_dir / "utility_vs_elo_by_rho_theta.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_overall(args.input_dir, args.output_dir)
    plot_by_ci(args.input_dir, args.output_dir)
    plot_by_rho_theta(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
